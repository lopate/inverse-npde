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
using ..PDEDefinitions: PhysicalConstants
using ..NeuralNetwork: NeuralNetworkConfig

# Экспортируем основные функции
export OptimizationConfig, LossFunctionConfig
export create_additional_loss, create_optimization_callback
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
    lambda_pde::Float64       # Вес потерь PDE
    lambda_bc::Float64        # Вес потерь граничных условий  
    lambda_data::Float64      # Вес потерь данных
    measured_points::Vector   # Измеренные точки
    
    function LossFunctionConfig(; lambda_pde=1.0, lambda_bc=1.0, lambda_data=10.0, 
                               measured_points=Vector{Any}[])
        return new(lambda_pde, lambda_bc, lambda_data, measured_points)
    end
end

"""
    create_additional_loss(loss_config::LossFunctionConfig, cpud)

Создает дополнительную функцию потерь для измеренных данных.
"""
function create_additional_loss(loss_config::LossFunctionConfig)
    function additional_loss(phi_pred_fun, θ, p_)
        # Всегда переносим на CPU для вычисления потерь с измеренными данными
        dev = cpu_device()

        CUDA.allowscalar() do
            result = sum(abs2(phi_pred_fun([x, y, z, t]|>dev, θ|>dev)[1] - phi|>dev)
                        for (x, y, z, t, phi) in loss_config.measured_points) /
                        length(loss_config.measured_points) |> dev

            # Применяем вес
            result = result * loss_config.lambda_data

            return result
        end
    end

    return additional_loss
end

"""
    create_discretization(chain, ps, loss_config::LossFunctionConfig, opt_config::OptimizationConfig, 
                         pde_system, bcs, domains)

Создает discretization объект для NeuralPDE.
"""
function create_discretization(chain, ps, loss_config::LossFunctionConfig, 
                              opt_config::OptimizationConfig)
    
    # Создаем стратегию обучения
    strategy = QuasiRandomTraining(4096)
    
    # Дополнительная функция потерь (device-aware) - выбор устройства внутри функции
    additional_loss_func = create_additional_loss(loss_config)
    
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

    return discretization
end

"""
    create_optimization_callback(opt_config::OptimizationConfig, discretization, pde_system, bcs, domains)

Создает callback функцию для мониторинга обучения.
"""
function create_optimization_callback(opt_config::OptimizationConfig, discretization, pde_system, bcs, domains)
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
    
    return function (p, l)
        iter += 1
        
        # Логируем общую потерю
        if logger !== nothing
            log_value(logger, "Loss/Total", l; step=iter)
        end
        
        # Логируем потери по PDE
        pde_losses = map(l_ -> l_(p.u), pde_inner_loss_functions)
        if logger !== nothing
            for (i, pde_loss) in enumerate(pde_losses)
                log_value(logger, "Loss/PDE_$i", pde_loss; step=iter)
            end
        end
        
        # Логируем потери по граничным условиям
        bcs_losses = map(l_ -> l_(p.u), bcs_inner_loss_functions)
        if logger !== nothing
            for (i, bc_loss) in enumerate(bcs_losses)
                log_value(logger, "Loss/BC_$i", bc_loss; step=iter)
            end
        end
        
        # Обновляем прогресс бар
        ProgressBars.update(pbar)
        set_postfix(pbar, 
                   Loss = @sprintf("%.3f", l|>Float32), 
                   PDE_losses = @sprintf("%.3f", sum(pde_losses)|>Float32), 
                   BC_losses = @sprintf("%.3f", sum(bcs_losses)|>Float32))
        
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