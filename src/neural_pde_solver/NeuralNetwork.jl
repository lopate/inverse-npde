"""
    NeuralNetwork

Модуль для создания и настройки нейронных сетей для решения обратной задачи ЭЭГ.
Включает различные архитектуры, функции активации и методы инициализации.

Основные компоненты:
- Создание нейронных сетей с различными архитектурами
- Функции активации (sigmoid, tanh, relu)
- Инициализация параметров сети
- Настройка выходного слоя для разделения на физические переменные
"""

module NeuralNetwork

using Lux, Random, ComponentArrays, CUDA
using ..PDEDefinitions: PhysicalConstants

# Экспортируем основные функции
export NeuralNetworkConfig, create_neural_network, create_output_splitter
export get_device_functions, initialize_parameters

# Структура конфигурации нейронной сети
struct NeuralNetworkConfig
    input_dim::Int              # Размерность входа
    hidden_layers::Vector{Int}  # Размеры скрытых слоев
    output_dim::Int             # Размерность выхода
    activation::Symbol          # Функция активации
    use_gpu::Bool               # Использовать ли GPU
    
    function NeuralNetworkConfig(;input_dim=4, hidden_layers=[32, 32], output_dim=8, activation=:σ, use_gpu=true)
        return new(input_dim, hidden_layers, output_dim, Symbol(activation), use_gpu)
    end
end

"""
    get_activation_function(symbol)

Возвращает функцию активации по символу.
"""
function get_activation_function(symbol::Symbol)
    activations = Dict(
        :σ => σ,                    # sigmoid
        :tanh => tanh,              # hyperbolic tangent  
        :relu => relu,              # rectified linear unit
        :elu => elu,                # exponential linear unit
        :gelu => gelu               # gaussian error linear unit
    )
    
    if !haskey(activations, symbol)
        throw(ArgumentError("Неизвестная функция активации: $symbol. Доступны: $(keys(activations))"))
    end
    
    return activations[symbol]
end

"""
    create_neural_network(config::NeuralNetworkConfig)

Создает нейронную сеть согласно конфигурации.
"""
function create_neural_network(config::NeuralNetworkConfig)
    activation_fn = get_activation_function(config.activation)
    
    # Создаем слои сети
    layers = []
    
    # Входной слой
    push!(layers, Dense(config.input_dim, config.hidden_layers[1], activation_fn))
    
    # Скрытые слои
    for i in 2:length(config.hidden_layers)
        push!(layers, Dense(config.hidden_layers[i-1], config.hidden_layers[i], activation_fn))
    end
    
    # Выходной слой
    push!(layers, Dense(config.hidden_layers[end], config.output_dim))
    
    # Создаем цепь
    chain = Chain(layers...)
    
    # Настраиваем устройство
    if config.use_gpu && CUDA.functional()
        chain = chain |> gpu_device()
    end
    
    return chain
end

"""
    initialize_parameters(chain, rng, use_gpu)

Инициализирует параметры нейронной сети.
"""
function initialize_parameters(chain, rng::Random.AbstractRNG=Random.default_rng(), use_gpu::Bool=true)
    ps = Lux.setup(rng, chain)[1]
    
    # Преобразуем в ComponentArray и перемещаем на нужное устройство
    ps = ps |> ComponentArray
    
    if use_gpu && CUDA.functional()
        ps = ps |> gpu_device() .|> Float32
    else
        ps = ps .|> Float32
    end
    
    return ps
end

"""
    create_output_splitter()

Создает функцию для разделения выхода нейронной сети на физические переменные:
- φ_pred (скалярный потенциал)
- A_pred (векторный потенциал)  
- ρ_pred (плотность заряда)
- j_pred (плотность тока)
"""
function create_output_splitter()
    function split_outputs(output)
        # Выходы: [φ, Ax, Ay, Az, ρ, jx, jy, jz]
        φ_pred = output[1]
        A_pred = output[2:4]  # [Ax, Ay, Az]
        ρ_pred = output[5]
        j_pred = output[6:8]  # [jx, jy, jz]
        
        return φ_pred, A_pred, ρ_pred, j_pred
    end
    
    return split_outputs
end

"""
    get_device_functions()

Возвращает функции для работы с устройствами (CPU/GPU).
"""
function get_device_functions()
    gpud = gpu_device()
    cpud = cpu_device()
    
    return gpud, cpud
end

"""
    evaluate_network(chain, params, inputs)

Вычисляет выход нейронной сети для заданных входов.
"""
function evaluate_network(chain, params, inputs)
    # Перемещаем входы на то же устройство, что и параметры
    if CUDA.functional() && typeof(params) <: ComponentArray && CUDA.isgpu(params)
        inputs = inputs |> gpu_device()
    else
        inputs = inputs |> cpu_device()
    end
    
    # Вычисляем выход сети
    output, _ = chain(inputs, params)
    
    return output
end

"""
    batch_evaluate_network(chain, params, batch_inputs)

Вычисляет выход нейронной сети для батча входов.
"""
function batch_evaluate_network(chain, params, batch_inputs)
    batch_size = size(batch_inputs, 2)
    outputs = []
    
    for i in 1:batch_size
        single_input = batch_inputs[:, i]
        output = evaluate_network(chain, params, single_input)
        push!(outputs, output)
    end
    
    # Собираем в матрицу
    if length(outputs[1]) == 1  # скалярный выход
        return vcat([out[1] for out in outputs]...)
    else  # векторный выход
        return hcat([collect(out) for out in outputs]...)
    end
end

"""
    create_custom_network(; layers::Vector{Int}, activation=:σ, use_gpu=true)

Создает кастомную нейронную сеть с произвольными слоями.
"""
function create_custom_network(; layers::Vector{Int}, activation::Symbol=:σ, use_gpu::Bool=true)
    if length(layers) < 2
        throw(ArgumentError("Должно быть минимум 2 слоя: входной и выходной"))
    end
    
    activation_fn = get_activation_function(activation)
    
    # Создаем слои
    layers_list = []
    
    for i in 1:(length(layers)-2)
        push!(layers_list, Dense(layers[i], layers[i+1], activation_fn))
    end
    
    # Последний слой без активации
    push!(layers_list, Dense(layers[end-1], layers[end]))
    
    chain = Chain(layers_list...)
    
    # Настраиваем устройство
    if use_gpu && CUDA.functional()
        chain = chain |> gpu_device()
    end
    
    return chain
end

"""
    validate_config(config::NeuralNetworkConfig)

Проверяет корректность конфигурации нейронной сети.
"""
function validate_config(config::NeuralNetworkConfig)
    if config.input_dim <= 0
        throw(ArgumentError("Размерность входа должна быть положительной"))
    end
    
    if config.output_dim <= 0
        throw(ArgumentError("Размерность выхода должна быть положительной"))
    end
    
    if any(layer -> layer <= 0, config.hidden_layers)
        throw(ArgumentError("Размеры скрытых слоев должны быть положительными"))
    end
    
    valid_activations = [:σ, :tanh, :relu, :elu, :gelu]
    if !(config.activation in valid_activations)
        throw(ArgumentError("Неизвестная функция активации: $(config.activation)"))
    end
    
    return true
end

end # module