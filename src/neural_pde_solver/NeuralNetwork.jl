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
using ..HeadConstraints: HeadConfig, head_indicator

# Экспортируем основные функции
export NeuralNetworkConfig, TemporalAwareNetworkConfig
export create_neural_network, create_temporal_aware_network
export create_output_splitter, get_device_functions
export initialize_parameters, initialize_temporal_aware_parameters
export build_dense_chain, create_fourier_features
export ConstrainedNeuralNetwork, create_constrained_neural_network, apply_head_constraint

# Структура конфигурации нейронной сети
struct NeuralNetworkConfig
    input_dim::Int              # Размерность входа
    hidden_layers::Vector{Int}  # Размеры скрытых слоев
    output_dim::Int             # Размерность выхода (8: φ, Ax, Ay, Az, ρ, jx, jy, jz) или 24 с производными
    activation::Symbol          # Функция активации
    use_gpu::Bool               # Использовать ли GPU
    use_derivatives::Bool      # Предсказывать ли производные (увеличивает output_dim до 24)
    
    function NeuralNetworkConfig(;input_dim=4, hidden_layers=[32, 32], output_dim=8, activation=:σ, use_gpu=true, use_derivatives::Bool=true)
        # Автоматически устанавливаем output_dim=24 если use_derivatives=true
        actual_output_dim = use_derivatives ? 24 : output_dim
        return new(input_dim, hidden_layers, actual_output_dim, Symbol(activation), use_gpu, use_derivatives)
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
- output_dim: финальная размерность выхода (e.g., 8 или 24)
- activation: функция активации (Symbol)
- use_gpu: использовать GPU
- use_derivatives: предсказывать ли производные (увеличивает output_dim до 24)
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
    use_derivatives::Bool
    
    function TemporalAwareNetworkConfig(;
        spatial_hidden_layers::Vector{Int} = [32, 32],
        spatial_output_dim::Int = 32,
        
        num_fourier_frequencies::Int = 8,
        temporal_hidden_layers::Vector{Int} = [16],
        temporal_output_dim::Int = 16,
        
        fusion_hidden_layers::Vector{Int} = [32],
        
        output_dim::Int = 8,
        activation::Symbol = :tanh,
        use_gpu::Bool = true,
        use_derivatives::Bool = true
    )
        # Автоматически устанавливаем output_dim=24 если use_derivatives=true
        actual_output_dim = use_derivatives ? 24 : output_dim
        return new(
            spatial_hidden_layers, spatial_output_dim,
            num_fourier_frequencies, temporal_hidden_layers, temporal_output_dim,
            fusion_hidden_layers,
            actual_output_dim, Symbol(activation), use_gpu, use_derivatives
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
        :gelu => gelu,               # gaussian error linear unit
        :softplus => softplus,         # softplus (гладкая версия ReLU)
        :swish => swish,               # swish (x * sigmoid(x) - гладкая версия ReLU
        :leakyrelu => leakyrelu         # leaky ReLU (α=0.01)
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
        features = [sin(Float32(π) * k * t) for k in 1:num_frequencies]
        features = vcat(
            features,
            [cos(Float32(π) * k * t) for k in 1:num_frequencies]
        )
        return features
    else
        # Если t - вектор/матрица, результат - матрица [2*num_frequencies, ...]
        # Создаем все sin features и cos features без мутирования
        sin_features = [sin.(Float32(π) .* k .* t) for k in 1:num_frequencies]
        cos_features = [cos.(Float32(π) .* k .* t) for k in 1:num_frequencies]
        
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
    create_output_splitter(use_derivatives::Bool=true)

Создает функцию для разделения выхода нейронной сети на физические переменные.

Базовые переменные (8):
- φ_pred (скалярный потенциал)
- A_pred (векторный потенциал)  
- ρ_pred (плотность заряда)
- j_pred (плотность тока)

Производные (16, если use_derivatives=true):
- DφDt, DφDx, DφDy, DφDz (4 производные для φ)
- DAxDt, DAxDx, DAxDy, DAxDz (4 производные для Ax)
- DAyDt, DAyDx, DAyDy, DAyDz (4 производные для Ay)
- DAzDt, DAzDx, DAzDy, DAzDz (4 производные для Az)

Порядок выходов:
- indices 1-8: [φ, Ax, Ay, Az, ρ, jx, jy, jz]
- indices 9-12: [DφDt, DφDx, DφDy, DφDz]
- indices 13-24: [DAxDt, DAxDx, DAxDy, DAxDz, DAyDt, DAyDx, DAyDy, DAyDz, DAzDt, DAzDx, DAzDy, DAzDz]

Примечание: PML через затухание и экранирование не требует дополнительных
выходов нейросети - коэффициенты γ и α вычисляются аналитически.
"""
function create_output_splitter(use_derivatives::Bool=true)
    function split_outputs(output)
        if use_derivatives
            # 24 выхода: 8 базовых + 16 производных
            φ_pred = output[1]
            A_pred = output[2:4]  # [Ax, Ay, Az]
            ρ_pred = output[5]
            j_pred = output[6:8]  # [jx, jy, jz]
            
            # Производные φ
            DφDt_pred = output[9]
            DφDx_pred = output[10]
            DφDy_pred = output[11]
            DφDz_pred = output[12]
            
            # Производные A
            DA_dt_pred = output[13:15]   # [DAxDt, DAyDt, DAzDt]
            DA_dx_pred = output[16:18]   # [DAxDx, DAyDx, DAzDx]
            DA_dy_pred = output[19:21]   # [DAxDy, DAyDy, DAzDy]
            DA_dz_pred = output[22:24]   # [DAxDz, DAyDz, DAzDz]
            
            return (φ_pred, A_pred, ρ_pred, j_pred, 
                    DφDt_pred, DφDx_pred, DφDy_pred, DφDz_pred,
                    DA_dt_pred, DA_dx_pred, DA_dy_pred, DA_dz_pred)
        else
            # 8 выходов: только базовые переменные
            φ_pred = output[1]
            A_pred = output[2:4]  # [Ax, Ay, Az]
            ρ_pred = output[5]
            j_pred = output[6:8]  # [jx, jy, jz]
            
            return φ_pred, A_pred, ρ_pred, j_pred
        end
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
    # Диагностика типов данных
    @debug "evaluate_network: params type = $(typeof(params)), eltype = $(eltype(params))"
    @debug "evaluate_network: inputs type = $(typeof(inputs)), eltype = $(eltype(inputs))"
    
    # Перемещаем входы на то же устройство, что и параметры, и конвертируем в Float32
    if CUDA.functional() && typeof(params) <: ComponentArray && CUDA.isgpu(params)
        inputs = inputs |> gpu_device() |> Float32
        @debug "evaluate_network: moved inputs to GPU as Float32"
    else
        inputs = inputs |> cpu_device() |> Float32
        @debug "evaluate_network: moved inputs to CPU as Float32"
    end
    
    @debug "evaluate_network: processed inputs type = $(typeof(inputs)), eltype = $(eltype(inputs))"
    
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

"""
    create_constrained_neural_network(inner_network, head_config::HeadConfig; use_derivatives::Bool=true)

Создаёт Lux слой-обёртку для нейронной сети с ограничением на плотность заряда ρ.

Это Lux слой, который NeuralPDE корректно распознаёт без использования FromFluxAdaptor.

Плотность заряда ρ ограничивается индикаторной функцией головы:
    ρ_constrained = ρ * head_indicator(x, y, z, head_config)

Это гарантирует, что ρ = 0 вне эллипсоида головы.

Градиенты протекают напрямую через индикаторную функцию.

Индексы выхода:
- 1: φ (скалярный потенциал)
- 2-4: A (векторный потенциал)
- 5: ρ (плотность заряда) - ОГРАНИЧИВАЕТСЯ
- 6-8: j (плотность тока) - НЕ ограничивается

# Параметры:
- `inner_network`: Внутренняя Lux нейросеть (например, @compact слой)
- `head_config::HeadConfig`: Конфигурация головы для ограничения
- `use_derivatives::Bool`: Использовать ли производные (влияет на размер выхода)

# Возвращает:
- Lux слой (через @compact макрос)
"""
function create_constrained_neural_network(inner_network, head_config::HeadConfig; use_derivatives::Bool=true)
    # Сохраняем конфигурацию как константу для использования внутри замыкания
    _head_config = head_config
    
    return @compact(
        ;
        inner_network=inner_network,
        head_config=_head_config,
        use_derivatives=use_derivatives
    ) do x, p
        # Получаем выход внутренней сети
        output, st = inner_network(x, p)
        
        # Если ограничение отключено, возвращаем как есть
        if !_head_config.enabled
            return output, st
        end
        
        # Определяем формат входа и извлекаем координаты через слайсирование
        # NeuralPDE передаёт данные как (4, N) для батча из N точек
        # Используем x[1:1], x[2:2], x[3:3] чтобы избежать скалярного индексирования
        if ndims(x) == 1
            # Одна точка: x имеет форму (4,)
            # Используем [1:1] чтобы получить 1-элементный массив вместо скаляра
            x_coord = x[1:1]
            y_coord = x[2:2]
            z_coord = x[3:3]
        else
            # Батч: x имеет форму (4, N)
            x_coord = x[1, :]
            y_coord = x[2, :]
            z_coord = x[3, :]
        end
        
        # Вычисляем индикатор - GPU-совместимая функция
        indicator = head_indicator(x_coord, y_coord, z_coord, _head_config)
        
        # Применяем ограничение к ρ (индекс 5)
        # Используем слайсирование вместо getindex для GPU совместимости
        if ndims(output) == 1
            # Одна точка: output имеет форму (8,)
            # Используем [5:5] чтобы получить 1-элементный массив
            rho = output[5:5]
            rho_constrained = rho .* indicator
            
            # Собираем результат через vcat
            # Используем [1:1] вместо [1] для избежания скалярного индексирования на GPU
            output_constrained = vcat(
                output[1:4],
                rho_constrained[1:1],
                output[6:8]
            )
            return output_constrained, st
        else
            # Батч: output имеет форму (8, N)
            # Умножаем строку 5 (ρ) на индикатор - обе формы (N,)
            rho_row = output[5, :]
            rho_constrained = rho_row .* indicator
            
            # Собираем результат через vcat
            output_constrained = vcat(
                output[1:4, :],
                reshape(rho_constrained, 1, :),
                output[6:8, :]
            )
            return output_constrained, st
        end
    end
end

# Сохраняем обратную совместимость - type alias
const ConstrainedNeuralNetwork = typeof(create_constrained_neural_network(
    create_temporal_aware_network(TemporalAwareNetworkConfig()),
    HeadConfig()
))

"""
    apply_head_constraint(output, coords, head_config::HeadConfig)

Применяет ограничение головы к выходу нейросети.

Ограничение применяется только к ρ (индекс 5):
    ρ_constrained = ρ * head_indicator(x, y, z, head_config)

Градиенты протекают напрямую через индикаторную функцию.

# Параметры:
- `output`: Выход нейросети (вектор размера 8 или 24)
- `coords`: Координаты точки (x, y, z) или матрица (3, N)
- `head_config::HeadConfig`: Конфигурация головы

# Возвращает:
- Выход с ограничением на ρ
"""
function apply_head_constraint(output, coords, head_config::HeadConfig)
    if !head_config.enabled
        return output
    end
    
    # Вычисляем индикатор
    indicator = head_indicator(coords, head_config)
    
    # Применяем ограничение только к ρ (индекс 5)
    # output[5] = ρ → ρ_constrained = ρ * indicator
    output_constrained = similar(output)
    output_constrained .= output
    
    if ndims(output) == 1
        # Один выход: (8,) или (24,)
        output_constrained[5] = output[5] * indicator
    else
        # Батч: (8, N) или (24, N)
        output_constrained[5, :] = output[5, :] .* indicator
    end
    
    return output_constrained
end

end # module
