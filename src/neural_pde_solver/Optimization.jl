"""
    Optimization

Модуль для создания функций потерь, настройки оптимизации и callback функций
для решения обратной задачи ЭЭГ с использованием NeuralPDE.

Основные компоненты:
- Создание дополнительных функций потерь для измеренных данных
- Callback функции для логирования и мониторинга
- Настройка оптимизаторов
- Создание discretization объекта для NeuralPDE
"""

module Optimization

using NeuralPDE, Optimization, OptimizationOptimJL, OptimizationOptimisers, OptimizationOptimisers, LineSearches, TensorBoardLogger
using ProgressBars, Printf, CUDA, Lux, LuxCUDA, ComponentArrays
using Zygote
using ..PDEDefinitions: PhysicalConstants
using ..NeuralNetwork: NeuralNetworkConfig

# Экспортируем основные функции
export OptimizationConfig, LossFunctionConfig
export create_additional_loss, create_optimization_callback
export create_data_loss_raw_func
export setup_optimization, create_discretization
export log_training_progress, validate_optimization_config

# Структура конфигурации оптимизации
struct OptimizationConfig
    optimizer::Symbol     # Тип оптимизатора (:adam, :lbfgs, :adamw)
    learning_rate::Float64 # Скорость обучения
    max_iterations::Int   # Максимальное количество итераций
    log_frequency::Int    # Частота логирования
    use_tensorboard::Bool # Использовать ли TensorBoard
    log_directory::String # Директория для логов
    
    function OptimizationConfig(; optimizer=:adam, learning_rate=0.001, max_iterations=3000, 
                               log_frequency=50, use_tensorboard=true, log_directory="logs/inverse_npde_exp")
        return new(Symbol(optimizer), learning_rate, max_iterations, log_frequency, use_tensorboard, log_directory)
    end
end

# Структура конфигурации функции потерь
struct LossFunctionConfig
    lambda_pde::Float64                           # Вес потерь PDE
    lambda_bc::Float64                           # Вес потерь граничных условий  
    lambda_data_init::Float64                    # Начальное значение веса данных
    lambda_min::Union{Float64, Nothing}          # Минимальное значение lambda_data
    lambda_max::Union{Float64, Nothing}          # Максимальное значение lambda_data
    measured_points::Vector                      # Исходные измеренные точки (для совместимости)
    measured_points_coords::Matrix{Float32}      # Батчированные координаты [4, N]
    measured_points_values::Vector{Float32}      # Батчированные значения [N]
    n_measured_points::Int                       # Количество измеренных точек
    # Scheduling options for lambda_data
    lambda_schedule_type::Symbol                  # :improvement (default)
    lambda_schedule::Dict{String,Any}             # Dict with scheduling params
    
    """
        LossFunctionConfig(; kwargs...)
    
    Создаёт конфигурацию функции потерь с адаптивным балансом lambda_data.
    Данные автоматически конвертируются в батчированную форму для GPU операций.
    
    # Аргументы
    - `lambda_pde=1.0`: Вес потерь PDE
    - `lambda_bc=1.0`: Вес потерь граничных условий
    - `lambda_data_init=1.0`: Начальное значение веса данных
    - `lambda_data_init=1.0`: Начальное значение веса данных (используется адаптивный планировщик)
    - `lambda_min=nothing`: Минимум lambda_data (nothing = без ограничений)
    - `lambda_max=nothing`: Максимум lambda_data (nothing = без ограничений)
    - `measured_points`: Вектор измеренных точек [(x, y, z, t, phi), ...]
    
    # Пример
    ```julia
    # Начальное значение веса данных (планировщик будет его адаптировать)
    config = LossFunctionConfig(lambda_data_init=1.0)

    # Использовать improvement-планировщик с пользовательскими параметрами
    config = LossFunctionConfig(lambda_data_init=1.0, lambda_schedule_type=:improvement, lambda_schedule=Dict("window_size" => 20, "patience" => 5))
    ```
    """
    function LossFunctionConfig(; 
        lambda_pde::Float64=1.0, 
        lambda_bc::Float64=1.0, 
        lambda_data_init::Float64=1.0,
        lambda_min::Union{Float64, Nothing}=nothing, 
        lambda_max::Union{Float64, Nothing}=nothing,
        measured_points::Vector=Vector{Any}[],
        lambda_schedule_type::Symbol = :improvement,
        lambda_schedule::Dict{String,Any} = Dict()
    )
        if lambda_min !== nothing && lambda_max !== nothing && lambda_min >= lambda_max
            throw(ArgumentError("lambda_min должно быть меньше lambda_max"))
        end
        
        # Конвертируем измеренные точки в батчированную форму для GPU операций
        n_points = length(measured_points)
        
        if n_points > 0
            # Используем hcat для векторизованного стекирования - работает с GPU и CPU массивами
            # Избегаем скалярного индексирования, применяя матричные операции
            all_data = hcat(measured_points...)  # Создает матрицу [5, N] 
            
            # Извлекаем координаты [4, N] и значения [N]
            coords = Float32.(all_data[1:4, :])
            values = vec(Float32.(all_data[5, :]))
        else
            # Пустой случай
            coords = Matrix{Float32}(undef, 4, 0)
            values = Vector{Float32}(undef, 0)
        end
        
        return new(lambda_pde, lambda_bc, lambda_data_init, lambda_min, lambda_max, measured_points, coords, values, n_points,
                   lambda_schedule_type, lambda_schedule)
    end
end

"""
    create_data_loss_raw_func(loss_config::LossFunctionConfig, phi_pred_fun)

Создает функцию для вычисления data loss БЕЗ веса lambda_data на GPU.
Использует батчированные операции вместо поэлементного цикла.

# Аргументы
- `loss_config`: Конфигурация функции потерь с батчированными координатами и значениями
- `phi_pred_fun`: Функция предсказания потенциала (сеть Lux, должна быть векторизована)

# Возвращает
- Функцию (params) -> L_data_raw, работающую чисто на GPU
"""
function create_data_loss_raw_func(loss_config::LossFunctionConfig, phi_pred_fun)
    n_points = loss_config.n_measured_points
    coords_batch = loss_config.measured_points_coords
    values_batch = loss_config.measured_points_values
    
    function data_loss_raw(params)
        # Получаем GPU устройство из параметров
        dev = Lux.gpu_device()
        
        # Переносим батчированные данные на GPU
        coords_gpu = coords_batch |> dev
        values_gpu = values_batch |> dev
        params_gpu = params |> dev
        
        # Векторизованный вызов: применяем сеть ко всем точкам одновременно
        # Сеть выводит [φ, Ax, Ay, Az, ρ, jx, jy, jz], нам нужен только φ (первая строка)
        pred_all = phi_pred_fun(coords_gpu, params_gpu)
        phi_pred = pred_all[1, :]  # Извлекаем только потенциал φ
        
        # Батчированное вычисление среднеквадратичной ошибки
        # Полностью на GPU, без скалярных операций
        mse = sum(abs2.(phi_pred .- values_gpu)) / n_points
        
        return mse
    end
    
    return data_loss_raw
end

"""
    create_additional_loss(loss_config::LossFunctionConfig, lambda_data_ref::Ref{Float64})

Создает дополнительную функцию потерь для измеренных данных через батчированные операции на GPU.

# Аргументы
- `loss_config`: Конфигурация функции потерь с батчированными данными
- `lambda_data_ref`: Ссылка на текущее значение веса данных (обновляется адаптивно)

# Возвращает
- Функцию потерь для PhysicsInformedNN, работающую чисто на GPU

# Примечание
Использует батчированные операции вместо поэлементного цикла.
lambda_data отсекается от графа градиентов через Zygote.dropgrad().
"""
function create_additional_loss(loss_config::LossFunctionConfig, lambda_data_ref::Ref{Float64})
    n_points = loss_config.n_measured_points
    coords_batch = loss_config.measured_points_coords
    values_batch = loss_config.measured_points_values
    
    function additional_loss(phi_pred_fun, θ, p_)
        # Получаем GPU устройство
        dev = Lux.gpu_device()
        
        # Переносим батчированные данные на GPU
        coords_gpu = coords_batch |> dev
        values_gpu = values_batch |> dev
        θ_gpu = θ |> dev
        
        # Векторизованный вызов сети для всех точек
        # Сеть выводит [φ, Ax, Ay, Az, ρ, jx, jy, jz], нам нужен только φ (первая строка)
        pred_all = phi_pred_fun(coords_gpu, θ_gpu)
        phi_pred = pred_all[1, :]  # Извлекаем только потенциал φ
        
        # Батчированное вычисление среднеквадратичной ошибки
        # Полностью на GPU без скалярных операций
        result = sum(abs2.(phi_pred .- values_gpu)) / n_points
        
        # Применяем адаптивный вес lambda_data
        # ВАЖНО: dropgrad отсекает lambda от графа градиентов,
        # гарантируя, что её динамическое обновление не влияет на градиенты сети
        lambda_detached = Zygote.dropgrad(lambda_data_ref[])
        result = result * lambda_detached
        
        return result
    end

    return additional_loss
end

"""
    create_discretization(chain, ps, loss_config::LossFunctionConfig, opt_config::OptimizationConfig, 
                         domain_config, lambda_data_ref::Ref{Float64})

Создает discretization объект для NeuralPDE с адаптивным весом данных.

# Возвращает
- Кортеж (discretization, lambda_data_ref) где lambda_data_ref - ссылка на текущий вес данных
"""
function create_discretization(chain, ps, loss_config::LossFunctionConfig, 
                              opt_config::OptimizationConfig, domain_config,
                              lambda_data_ref::Ref{Float64}=Ref{Float64}(loss_config.lambda_data_init))
    
    # Создаем стратегию обучения
    strategy = QuasiRandomTraining(domain_config["num_points"])
    
    # Дополнительная функция потерь с адаптивным весом
    additional_loss_func = create_additional_loss(loss_config, lambda_data_ref)
    
    # GPU-first construction: expect chain and params to be on the device desired by the user
    @info "Constructing PhysicsInformedNN (GPU-first). Ensure chain and params are on desired device"
    
    # Обработка настроек логирования - NeuralPDE не поддерживает nothing в log_options
    if opt_config.use_tensorboard
        log_options = LogOptions(; log_frequency = opt_config.log_frequency)
        discretization = PhysicsInformedNN(chain, strategy; 
                                          init_params = ps, 
                                          additional_loss = additional_loss_func,
                                          log_options = log_options)
    else
        # При use_tensorboard=false создаем без параметра log_options
        # Это избегает ошибки MethodError при передаче nothing
        discretization = PhysicsInformedNN(chain, strategy; 
                                          init_params = ps, 
                                          additional_loss = additional_loss_func)
    end

    return discretization, lambda_data_ref
end

"""
    create_optimization_callback(opt_config::OptimizationConfig, discretization, pde_system, bcs, domains,
                                 loss_config::LossFunctionConfig, lambda_data_ref::Ref{Float64},
                                 data_loss_raw_func::Function)

Создает callback функцию для мониторинга обучения с адаптивным балансом lambda_data.

# Аргументы
- `opt_config`: Конфигурация оптимизации
- `discretization`: Объект дискретизации NeuralPDE
- `pde_system`: Система PDE
- `bcs`: Граничные условия
- `domains`: Области
- `loss_config`: Конфигурация функции потерь
- `lambda_data_ref`: Ссылка на текущее значение веса данных
- `data_loss_raw_func`: Функция для вычисления "сырого" data loss без веса

# Адаптивный баланс
# Использует improvement-based планировщик: увеличивает λ при стагнации data loss.
"""
function create_optimization_callback(opt_config::OptimizationConfig, discretization, pde_system, bcs, domains,
                                      loss_config::LossFunctionConfig, lambda_data_ref::Ref{Float64},
                                      data_loss_raw_func::Function)
    maxiters = opt_config.max_iterations
    
    # Создаем логгер для TensorBoard
    logger = opt_config.use_tensorboard ? 
        TBLogger(opt_config.log_directory) : nothing
    
    # Получаем функции потерь
    sym_prob = symbolic_discretize(pde_system, discretization)
    pde_inner_loss_functions = sym_prob.loss_functions.pde_loss_functions
    bcs_inner_loss_functions = sym_prob.loss_functions.bc_loss_functions
    
    iter = 0
    pbar = ProgressBar(1:maxiters, printing_delay=1.)

    # State for improvement-based lambda scheduling
    # Initialized here so the closure can mutate them across iterations
    window_size = get(loss_config.lambda_schedule, "window_size", 10)
    increase_factor = get(loss_config.lambda_schedule, "increase_factor", 1.2)
    decrease_factor = get(loss_config.lambda_schedule, "decrease_factor", 0.98)
    patience = get(loss_config.lambda_schedule, "patience", 3)
    min_improvement = get(loss_config.lambda_schedule, "min_improvement", 1e-8)

    data_buffer = fill(Inf, window_size)
    buffer_pos = Ref(1)
    no_improve = Ref(0)
    lambda_action = Ref(0) # -1 decrease, 0 hold, 1 increase

    return function (p, l)
        iter += 1
        
        # Логируем общую потерю
        if logger !== nothing
            log_value(logger, "Loss/Total", l; step=iter)
        end
        
        # Вычисляем потери по PDE
        pde_losses = map(l_ -> l_(p.u), pde_inner_loss_functions)
        L_pde = sum(pde_losses)
        
        if logger !== nothing
            for (i, pde_loss) in enumerate(pde_losses)
                log_value(logger, "Loss/PDE_$i", pde_loss; step=iter)
            end
            # Логируем суммарный PDE loss
            log_value(logger, "Loss/L_pde", L_pde; step=iter)
        end
        
        # Логируем потери по граничным условиям
        bcs_losses = map(l_ -> l_(p.u), bcs_inner_loss_functions)
        if logger !== nothing
            for (i, bc_loss) in enumerate(bcs_losses)
                log_value(logger, "Loss/BC_$i", bc_loss; step=iter)
            end
        end
        
        # Вычисляем "сырой" data loss (без веса lambda_data)
        L_data_raw = data_loss_raw_func(p.u)
        
        # Improvement-based scheduling: increase λ when data loss stagnates
        if length(data_buffer) > 0
            # best observed in window BEFORE adding current value
            best_in_window = minimum(data_buffer)
        
            # if current is significantly better than best -> improvement
            if L_data_raw < best_in_window - min_improvement
                no_improve[] = 0
                # optionally decrease lambda slightly to allow flexibility
                lambda_data_new = lambda_data_ref[] * decrease_factor
                lambda_action[] = -1
            else
                no_improve[] += 1
                lambda_action[] = 0
            end
        
            # If stagnated for enough consecutive iterations -> increase lambda
            if no_improve[] >= patience
                lambda_data_new = lambda_data_ref[] * increase_factor
                no_improve[] = 0
                lambda_action[] = 1
            end
        
            # Add current value to buffer AFTER checking for improvement
            data_buffer[buffer_pos[]] = L_data_raw
            buffer_pos[] = (buffer_pos[] % length(data_buffer)) + 1

            # Apply bounds (consider both loss_config and schedule-specified bounds)
            schedule_min = get(loss_config.lambda_schedule, "lambda_min", nothing)
            schedule_max = get(loss_config.lambda_schedule, "lambda_max", nothing)
            lower = loss_config.lambda_min !== nothing ? loss_config.lambda_min : (schedule_min !== nothing ? schedule_min : 1e-8)
            upper = loss_config.lambda_max !== nothing ? loss_config.lambda_max : (schedule_max !== nothing ? schedule_max : 1e8)

            if @isdefined(lambda_data_new)
                lambda_data_new = clamp(lambda_data_new, lower, upper)
                lambda_data_ref[] = lambda_data_new
            end
        end
        
        # Логируем data loss и lambda_data
        if logger !== nothing
            log_value(logger, "Loss/L_data_raw", L_data_raw; step=iter)
            log_value(logger, "Loss/L_data_weighted", L_data_raw * lambda_data_ref[]; step=iter)
            log_value(logger, "Params/lambda_data", lambda_data_ref[]; step=iter)
            # iterations without improvement
            iter_no_improve = no_improve[]
            log_value(logger, "Params/iter_no_improve", iter_no_improve; step=iter)
            # action taken this iteration: -1 decrease, 0 hold, 1 increase
            log_value(logger, "Params/lambda_action", lambda_action[]; step=iter)
        end
        
        # Обновляем прогресс бар
        ProgressBars.update(pbar)
        set_postfix(pbar, 
                   Loss = @sprintf("%.3f", l|>Float32), 
                   PDE = @sprintf("%.3f", L_pde|>Float32), 
                   Data = @sprintf("%.3f", L_data_raw|>Float32),
                   λ = @sprintf("%.1f", lambda_data_ref[]|>Float32))
        
        return false  # Продолжаем оптимизацию
    end
end

"""
    setup_optimization(opt_config::OptimizationConfig)

Настраивает оптимизатор согласно конфигурации.
"""
function setup_optimization(opt_config::OptimizationConfig)
    optimizers = Dict(
        :adam => OptimizationOptimisers.Adam(opt_config.learning_rate),
        :adamw => OptimizationOptimisers.AdamW(opt_config.learning_rate),
        :lbfgs => OptimizationOptimJL.LBFGS()
    )
    
    if !haskey(optimizers, opt_config.optimizer)
        throw(ArgumentError("Неизвестный оптимизатор: $(opt_config.optimizer). Доступны: $(keys(optimizers))"))
    end
    
    return optimizers[opt_config.optimizer]
end

"""
    log_training_progress(logger, iteration, losses, metrics)

Логирует прогресс обучения с дополнительными метриками.
"""
function log_training_progress(logger, iteration, losses, metrics::Dict{String, Any}=Dict{String, Any}())
    # Основные потери
    log_value(logger, "Loss/Total", losses.total; step=iteration)
    
    # Потери по компонентам
    if haskey(losses, :pde)
        log_value(logger, "Loss/PDE", losses.pde; step=iteration)
    end
    
    if haskey(losses, :bc)
        log_value(logger, "Loss/BC", losses.bc; step=iteration)
    end
    
    if haskey(losses, :data)
        log_value(logger, "Loss/Data", losses.data; step=iteration)
    end
    
    # Дополнительные метрики
    for (key, value) in metrics
        log_value(logger, "Metrics/$key", value; step=iteration)
    end
end

"""
    validate_optimization_config(opt_config::OptimizationConfig)

Проверяет корректность конфигурации оптимизации.
"""
function validate_optimization_config(opt_config::OptimizationConfig)
    if opt_config.learning_rate <= 0
        throw(ArgumentError("Скорость обучения должна быть положительной"))
    end
    
    if opt_config.max_iterations <= 0
        throw(ArgumentError("Максимальное количество итераций должно быть положительным"))
    end
    
    if opt_config.log_frequency <= 0
        throw(ArgumentError("Частота логирования должна быть положительной"))
    end
    
    valid_optimizers = [:adam, :adamw, :lbfgs]
    if !(opt_config.optimizer in valid_optimizers)
        throw(ArgumentError("Неизвестный оптимизатор: $(opt_config.optimizer)"))
    end
    
    return true
end

"""
    create_loss_callback(opt_config::OptimizationConfig, logger, loss_history)

Создает callback для сохранения истории потерь.
"""
function create_loss_callback(opt_config::OptimizationConfig, logger, loss_history::Vector{Float64}=Float64[])
    return function (p, l)
        push!(loss_history, l)
        
        # Логируем каждые log_frequency итераций
        if length(loss_history) % opt_config.log_frequency == 0
            log_value(logger, "Loss/History", l; step=length(loss_history))
        end
        
        return false
    end
end

"""
    create_early_stopping_patience(patience::Int, min_delta::Float64)

Создает callback для ранней остановки.
"""
function create_early_stopping_patience(patience::Int, min_delta::Float64=1e-6)
    best_loss = Inf
    wait = 0
    
    # Используем ссылочные типы для сохранения состояния
    state = Ref{Float64}(best_loss)
    wait_ref = Ref{Int}(wait)
    
    return function (p, l)
        if l < state[] - min_delta
            state[] = l
            wait_ref[] = 0
        else
            wait_ref[] += 1
        end
        
        if wait_ref[] >= patience
            println("Ранняя остановка после $patience итераций без улучшения")
            return true  # Останавливаем оптимизацию
        end
        
        return false  # Продолжаем оптимизацию
    end
end

"""
    combine_callbacks(callbacks::Vector{Function})

Комбинирует несколько callback функций в одну.
"""
function combine_callbacks(callbacks::Vector{Function})
    return function (p, l)
        should_stop = false
        for callback in callbacks
            stop = callback(p, l)
            should_stop = should_stop || stop
        end
        return should_stop
    end
end

end # module