"""
    NeuralNetwork

Модуль для создания и настройки нейронных сетей для решения обратной задачи ЭЭГ.
Включает различные архитектуры, функции активации и методы инициализации.

Основные компоненты:
- Базовая архитектура MLP: [x,y,z,t] → Dense layers → outputs
- Temporal-Aware архитектура: отдельные ветви для пространства и времени с Fourier features
- Функции активации (sigmoid, tanh, relu)
- Инициализация параметров сети
- Настройка выходного слоя для разделения на физические переменные

Примечание: Новый подход PML через затухание (γ) и экранирование (α) не требует
дополнительных выходов нейросети - эти коэффициенты вычисляются аналитически
как функции координат.
"""

module NeuralNetwork

using Lux, Random, ComponentArrays, CUDA
using ..PDEDefinitions: PhysicalConstants

# Экспортируем основные функции
export NeuralNetworkConfig, TemporalAwareNetworkConfig
export create_neural_network, create_temporal_aware_network
export create_output_splitter, get_device_functions
export initialize_parameters, initialize_temporal_aware_parameters
export build_dense_chain, create_fourier_features

# Структура конфигурации нейронной сети
struct NeuralNetworkConfig
    input_dim::Int              # Размерность входа
    hidden_layers::Vector{Int}  # Размеры скрытых слоев
    output_dim::Int             # Размерность выхода (8: φ, Ax, Ay, Az, ρ, jx, jy, jz)
    activation::Symbol          # Функция активации
    use_gpu::Bool               # Использовать ли GPU
    
    function NeuralNetworkConfig(;input_dim=4, hidden_layers=[32, 32], output_dim=8, activation=:σ, use_gpu=true)
        return new(input_dim, hidden_layers, output_dim, Symbol(activation), use_gpu)
    end
end
"""
    TemporalAwareNetworkConfig

Конфигурация для архитектуры с разделением пространственных и временных признаков.

Структура:
- Spatial branch: обрабатывает [x, y, z] через Dense слои
- Temporal branch: кодирует t через Fourier features и Dense слои
- Fusion layer: объединяет пространственные и временные признаки
- Output: финальные физические переменные

Параметры:
- spatial_hidden_layers: размеры Dense слоев для пространства (e.g., [32, 32])
- spatial_output_dim: размерность выхода spatial branch (e.g., 32)
- num_fourier_frequencies: количество Fourier частот (e.g., 8 → 16 признаков)
- temporal_hidden_layers: размеры Dense слоев для времени (e.g., [16])
- temporal_output_dim: размерность выхода temporal branch (e.g., 16)
- fusion_hidden_layers: размеры Dense слоев после объединения (e.g., [32])
- output_dim: финальная размерность выхода (e.g., 8)
- activation: функция активации (Symbol)
- use_gpu: использовать GPU
"""
struct TemporalAwareNetworkConfig
    spatial_hidden_layers::Vector{Int}
    spatial_output_dim::Int
    
    num_fourier_frequencies::Int
    temporal_hidden_layers::Vector{Int}
    temporal_output_dim::Int
    
    fusion_hidden_layers::Vector{Int}
    
    output_dim::Int
    activation::Symbol
    use_gpu::Bool
    
    function TemporalAwareNetworkConfig(;
        spatial_hidden_layers::Vector{Int} = [32, 32],
        spatial_output_dim::Int = 32,
        
        num_fourier_frequencies::Int = 8,
        temporal_hidden_layers::Vector{Int} = [16],
        temporal_output_dim::Int = 16,
        
        fusion_hidden_layers::Vector{Int} = [32],
        
        output_dim::Int = 8,
        activation::Symbol = :tanh,
        use_gpu::Bool = true
    )
        return new(
            spatial_hidden_layers, spatial_output_dim,
            num_fourier_frequencies, temporal_hidden_layers, temporal_output_dim,
            fusion_hidden_layers,
            output_dim, Symbol(activation), use_gpu
        )
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
    build_dense_chain(input_dim::Int, hidden_layers::Vector{Int}, output_dim::Int, 
                      activation::Symbol; use_last_activation::Bool=false)

Строит цепь Dense слоёв с указанной архитектурой.

Параметры:
- input_dim: размерность входа
- hidden_layers: вектор размеров скрытых слоёв
- output_dim: размерность выхода
- activation: функция активации для скрытых слоёв
- use_last_activation: применять ли активацию к выходному слою (обычно false)

Пример:
    build_dense_chain(3, [32, 32], 16, :tanh)
    # Dense(3→32) → tanh → Dense(32→32) → tanh → Dense(32→16)
"""
function build_dense_chain(
    input_dim::Int,
    hidden_layers::Vector{Int},
    output_dim::Int,
    activation::Symbol;
    use_last_activation::Bool = false
)
    if input_dim <= 0 || output_dim <= 0
        throw(ArgumentError("input_dim и output_dim должны быть положительными"))
    end
    
    if any(size -> size <= 0, hidden_layers)
        throw(ArgumentError("Размеры скрытых слоёв должны быть положительными"))
    end
    
    activation_fn = get_activation_function(activation)
    layers = []
    
    # Входной слой: input_dim → первый скрытый слой
    if length(hidden_layers) > 0
        push!(layers, Dense(input_dim, hidden_layers[1], activation_fn))
        
        # Промежуточные скрытые слои
        for i in 2:length(hidden_layers)
            push!(layers, Dense(hidden_layers[i-1], hidden_layers[i], activation_fn))
        end
        
        # Выходной слой: последний скрытый → output_dim
        if use_last_activation
            push!(layers, Dense(hidden_layers[end], output_dim, activation_fn))
        else
            push!(layers, Dense(hidden_layers[end], output_dim))
        end
    else
        # Если нет скрытых слоёв, прямой слой: input_dim → output_dim
        if use_last_activation
            push!(layers, Dense(input_dim, output_dim, activation_fn))
        else
            push!(layers, Dense(input_dim, output_dim))
        end
    end
    
    return Chain(layers...)
end

"""
    create_fourier_features(t::Union{AbstractVector, AbstractMatrix}, num_frequencies::Int)

Генерирует Fourier features для временной переменной t.

Формула:
    features = [sin(π·1·t), cos(π·1·t), sin(π·2·t), cos(π·2·t), ..., sin(π·k·t), cos(π·k·t)]

Результат имеет размер 2*num_frequencies по размерности Fourier.

Параметры:
- t: временная переменная (скаляр, вектор или матрица; форма [...])
- num_frequencies: количество Fourier частот (k = 1..num_frequencies)

Возвращает:
- features: Fourier кодирование времени размера 2*num_frequencies
            (если t - скаляр: вектор [2*k]; если вектор/матрица: матрица [2*k, ...])
"""
function create_fourier_features(t::Union{Real, AbstractVecOrMat}, num_frequencies::Int)
    if num_frequencies <= 0
        throw(ArgumentError("num_frequencies должно быть положительным"))
    end
    
    # Создаем Fourier features БЕЗ мутирования массивов (для совместимости с Zygote)
    # Используем понимание списков вместо push!() чтобы избежать мутирования
    if t isa Real
        # Если t скаляр, результат - вектор [2*num_frequencies]
        features = [sin(π * k * t) for k in 1:num_frequencies]
        features = vcat(
            features,
            [cos(π * k * t) for k in 1:num_frequencies]
        )
        return features
    else
        # Если t - вектор/матрица, результат - матрица [2*num_frequencies, ...]
        # Создаем все sin features и cos features без мутирования
        sin_features = [sin.(π .* k .* t) for k in 1:num_frequencies]
        cos_features = [cos.(π .* k .* t) for k in 1:num_frequencies]
        
        if ndims(t) == 1
            # t - вектор: reshape каждый в строку и собрать матрицу
            sin_rows = [reshape(f, 1, :) for f in sin_features]
            cos_rows = [reshape(f, 1, :) for f in cos_features]
            return vcat(sin_rows..., cos_rows...)
        else
            # t - матрица или более высокая размерность
            return vcat(sin_features..., cos_features...)
        end
    end
end

"""
    create_neural_network(config::NeuralNetworkConfig)

Создает стандартную нейронную сеть согласно конфигурации.
Простая MLP архитектура: [x,y,z,t] → Dense layers → output
"""
function create_neural_network(config::NeuralNetworkConfig)
    activation_fn = get_activation_function(config.activation)
    
    # Используем build_dense_chain для создания сети
    chain = build_dense_chain(
        config.input_dim,
        config.hidden_layers,
        config.output_dim,
        config.activation;
        use_last_activation = false
    )
    
    # Настраиваем устройство
    if config.use_gpu && CUDA.functional()
        chain = chain |> gpu_device()
    end
    
    return chain
end



"""
    create_temporal_aware_network(config::TemporalAwareNetworkConfig; rng=Random.default_rng())

Создает нейронную сеть с Temporal-Aware архитектурой как правильный Lux слой.

Архитектура:
1. Spatial branch: [x, y, z] → Dense layers → spatial_features
2. Temporal branch: [t] → Fourier features (2*num_fourier_frequencies) → Dense layers → temporal_features
3. Fusion: concat[spatial_features, temporal_features] → Dense layers → [output_dim]

Входные данные: [x, y, z, t] размер (4,) или (4, N) для батча
Выходные данные: [φ, Ax, Ay, Az, ρ, jx, jy, jz] размер (8,) или (8, N)

Это разделение гарантирует явную зависимость от времени через Fourier features.
"""
function create_temporal_aware_network(config::TemporalAwareNetworkConfig; rng=Random.default_rng())
    # === BUILD COMPONENT LAYERS ===
    
    # Spatial branch: [x, y, z] → spatial_output_dim
    spatial_branch = build_dense_chain(
        3,
        config.spatial_hidden_layers,
        config.spatial_output_dim,
        config.activation;
        use_last_activation = true
    )
    
    # Temporal branch: 2*num_fourier_frequencies → temporal_output_dim
    fourier_input_dim = 2 * config.num_fourier_frequencies
    temporal_branch = build_dense_chain(
        fourier_input_dim,
        config.temporal_hidden_layers,
        config.temporal_output_dim,
        config.activation;
        use_last_activation = true
    )
    
    # Fusion layers: (spatial_output_dim + temporal_output_dim) → output_dim
    fusion_input_dim = config.spatial_output_dim + config.temporal_output_dim
    fusion_layers = build_dense_chain(
        fusion_input_dim,
        config.fusion_hidden_layers,
        config.output_dim,
        config.activation;
        use_last_activation = false
    )
    
    # === CREATE LUX @COMPACT LAYER ===
    # Используем @compact макрос для создания правильного Lux слоя
    num_fourier_frequencies = config.num_fourier_frequencies
    
    return @compact(
        ;
        spatial_branch=spatial_branch,
        temporal_branch=temporal_branch,
        fusion_layers=fusion_layers,
        num_fourier_frequencies=num_fourier_frequencies
    ) do x
        # Запоминаем была ли входная точка 1D
        is_single_point = ndims(x) == 1
        
        # Разделяем входные данные на пространственную и временную части
        if is_single_point
            # Одна точка: (4,)
            spatial_input = x[1:3]      # [x, y, z]
            t_val = x[4:4]              # [t]
        else
            # Батч: (4, N)
            spatial_input = x[1:3, :]   # [x, y, z] (3, N)
            t_val = vec(x[4, :])        # [t] (N,)
        end
        
        # === SPATIAL BRANCH ===
        spatial_feat = spatial_branch(spatial_input)
        
        # === FOURIER FEATURES ===
        fourier_features = create_fourier_features(t_val, num_fourier_frequencies)
        
        # === TEMPORAL BRANCH ===
        temporal_feat = temporal_branch(fourier_features)
        
        # === FUSION ===
        fusion_input = vcat(spatial_feat, temporal_feat)
        output = fusion_layers(fusion_input)
        
        # Возвращаем в исходной размерности
        @return if is_single_point
            vec(output)  # Преобразуем (8, 1) в (8,)
        else
            output  # Оставляем (8, N) как есть
        end
    end
end

"""
    initialize_parameters(chain, rng, use_gpu)

Инициализирует параметры нейронной сети (обычной MLP).
"""
function initialize_parameters(chain, rng::Random.AbstractRNG=Random.default_rng(); use_gpu::Bool=true)
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
    initialize_temporal_aware_parameters(network, rng::Random.AbstractRNG=Random.default_rng(), use_gpu::Bool=true)

Инициализирует параметры Temporal-Aware сети (@compact слой из create_temporal_aware_network).

Параметры:
- network: Lux слой из create_temporal_aware_network()
- rng: Random Number Generator
- use_gpu: использовать GPU

Возвращает:
- ps: ComponentArray с параметрами всех ветвей (spatial_branch, temporal_branch, fusion_layers)
"""
function initialize_temporal_aware_parameters(network, rng::Random.AbstractRNG=Random.default_rng(); use_gpu::Bool=true)
    # Инициализируем параметры @compact слоя
    ps, st = Lux.setup(rng, network)
    
    # Преобразуем в ComponentArray
    ps = ps |> ComponentArray
    
    # Перемещаем на нужное устройство
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

Выходы имеют следующий порядок:
[φ, Ax, Ay, Az, ρ, jx, jy, jz] - всего 8 значений

Примечание: PML через затухание и экранирование не требует дополнительных
выходов нейросети - коэффициенты γ и α вычисляются аналитически.
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

"""
    validate_config(config::TemporalAwareNetworkConfig)

Проверяет корректность конфигурации Temporal-Aware сети.
"""
function validate_config(config::TemporalAwareNetworkConfig)
    if config.spatial_output_dim <= 0
        throw(ArgumentError("spatial_output_dim должно быть положительным"))
    end
    
    if config.temporal_output_dim <= 0
        throw(ArgumentError("temporal_output_dim должно быть положительным"))
    end
    
    if config.output_dim <= 0
        throw(ArgumentError("output_dim должно быть положительным"))
    end
    
    if config.num_fourier_frequencies <= 0
        throw(ArgumentError("num_fourier_frequencies должно быть положительным"))
    end
    
    if any(layer -> layer <= 0, config.spatial_hidden_layers)
        throw(ArgumentError("Размеры spatial скрытых слоёв должны быть положительными"))
    end
    
    if any(layer -> layer <= 0, config.temporal_hidden_layers)
        throw(ArgumentError("Размеры temporal скрытых слоёв должны быть положительными"))
    end
    
    if any(layer -> layer <= 0, config.fusion_hidden_layers)
        throw(ArgumentError("Размеры fusion скрытых слоёв должны быть положительными"))
    end
    
    valid_activations = [:σ, :tanh, :relu, :elu, :gelu]
    if !(config.activation in valid_activations)
        throw(ArgumentError("Неизвестная функция активации: $(config.activation)"))
    end
    
    return true
end

end # module
