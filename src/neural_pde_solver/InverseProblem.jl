"""
    InverseProblem

Главный модуль для решения обратной задачи ЭЭГ с использованием NeuralPDE.
Объединяет все компоненты: PDE определения, нейронные сети и оптимизацию.

Пример использования:
```julia
using .InverseProblem

# Создаем конфигурации
nn_config = NeuralNetworkConfig(; input_dim=4, hidden_layers=[32, 32], output_dim=8)
opt_config = OptimizationConfig(; learning_rate=0.001, max_iterations=3000)
loss_config = LossFunctionConfig(; lambda_data_init=10.0, lambda_schedule_type=:improvement)
pml_config = PMLConfig()

# Запускаем эксперимент
results = run_eeg_inverse_problem(nn_config, opt_config, loss_config, pml_config=pml_config)
```

Примечание: Новый подход PML через затухание (γ) и экранирование (α) не требует
дополнительных выходов нейросети - эти коэффициенты вычисляются аналитически
как функции координат. Размерность выхода всегда равна 8:
[φ, Ax, Ay, Az, ρ, jx, jy, jz]
"""

module InverseProblem

using NeuralPDE, Lux, LuxCUDA, Random, ComponentArrays, CUDA
using OptimizationOptimisers, OptimizationOptimJL, SciMLBase
using ModelingToolkit: @named
using Plots
using Statistics: mean
using JLD2: jldopen
using TensorBoardLogger

using ..PDEDefinitions
using ..NeuralNetwork
using ..Optimization
using ..PML

using ..PDEDefinitions: PhysicalConstants, create_variables, create_domains, create_pde_system, create_boundary_conditions, generate_measured_points, analytic_sol_func 
using ..NeuralNetwork: NeuralNetworkConfig, create_neural_network, initialize_parameters, validate_config
using ..Optimization: OptimizationConfig, LossFunctionConfig, set_inner_domain!, validate_optimization_config, create_discretization, create_optimization_callback, data_loss, derivative_loss, setup_optimization, solve, cosine_annealing_lr, warmup_cosine_lr
using ..PDEDefinitions: create_variables, create_domains, create_pde_system, create_boundary_conditions, generate_measured_points, analytic_sol_func, PhysicalConstants

"""
    normalize_measured_points(measured_points)

Нормирует значения на датчиках по среднему модулю всех измерений.

Args:
    measured_points: Массив измеренных точек [x, y, z, t, phi_measured]

Returns:
    (normalized_points, norm_factor): Нормированные точки и фактор нормировки
"""
function normalize_measured_points(measured_points)
    # Извлекаем все измеренные значения φ
    phi_values = [point[5] for point in measured_points]
    
    # Вычисляем норму - средний модуль всех значений
    norm_factor = mean(abs.(phi_values))
    
    # Нормируем все значения
    normalized_points = map(measured_points) do point
        [point[1], point[2], point[3], point[4], point[5] / norm_factor]
    end
    
    return normalized_points, norm_factor
end

"""
    preprocess_measured_data(measured_points::Vector{Vector{Float32}}, time_steps::Int)
    Interpolate measured data to create continuous functions
    Compute temporal derivatives
"""
function preprocess_measured_data(measured_points::Vector{Vector{Float32}}, time_steps::Int)
    # Interpolate measured data to create continuous functions
    interpolated_data = [CubicSplineInterpolation(1:time_steps, [point[5] for point in measured_points if point[4] == t]) for t in 1:time_steps]

    # Compute temporal derivatives
    derivatives = [diff(interpolated_data[t]) for t in 1:time_steps]

    return interpolated_data, derivatives
end

# Структура конфигурации домена
struct DomainConfig
    x_range::Vector{Float32}
    y_range::Vector{Float32}
    z_range::Vector{Float32}
    t_range::Vector{Float32}
    num_points::Int
    
    function DomainConfig(; x_range=[-10.0f0, 10.0f0], y_range=[-10.0f0, 10.0f0], 
                          z_range=[-10.0f0, 10.0f0], t_range=[0.0f0, 1.0f0], num_points=100)
        return new(x_range, y_range, z_range, t_range, num_points)
    end
end

# Экспортируем основные функции
export run_eeg_inverse_problem, create_complete_setup, run_three_stage_optimization
export analyze_results, save_results, load_results, DomainConfig, PMLConfig

"""
    create_complete_setup(; measured_points, nn_config, opt_config, loss_config, domain_config, pml_config)

Создает полную настройку для эксперимента обратной задачи ЭЭГ.
Поддерживает как стандартную MLP архитектуру, так и Temporal-Aware архитектуру.

Размерность выхода нейросети всегда равна 8 (φ, Ax, Ay, Az, ρ, jx, jy, jz).
PML через затухание и экранирование не требует дополнительных выходов.
"""
function create_complete_setup(; measured_points, nn_config::Union{NeuralNetworkConfig, TemporalAwareNetworkConfig},
                               opt_config::OptimizationConfig,
                               loss_config::LossFunctionConfig,
                               domain_config::Dict{String, Any}=Dict(
                                   "x_range" => [-10.0f0, 10.0f0],
                                   "y_range" => [-10.0f0, 10.0f0], 
                                   "z_range" => [-10.0f0, 10.0f0],
                                   "t_range" => [0.0f0, 1.0f0],
                                   "num_points" => 100
                               ),
                               pml_config::PMLConfig=PMLConfig())
    
    # Делегируем на специализированные приватные функции в зависимости от типа конфигурации
    if nn_config isa NeuralNetworkConfig
        return _create_complete_setup_standard(; measured_points, nn_config, opt_config, loss_config, domain_config, pml_config)
    else
        return _create_complete_setup_temporal(; measured_points, nn_config, opt_config, loss_config, domain_config, pml_config)
    end
end

"""
    _create_complete_setup_standard(...)

Приватная функция для создания настройки со стандартной MLP архитектурой.
"""
function _create_complete_setup_standard(; measured_points, nn_config::NeuralNetworkConfig, 
                               opt_config::OptimizationConfig,
                               loss_config::LossFunctionConfig,
                               domain_config::Dict{String, Any},
                               pml_config::PMLConfig)
    
    # Нормируем измеренные точки
    normalized_points, norm_factor = normalize_measured_points(measured_points)
    println("✓ Данные нормированы, фактор: $(round(norm_factor, digits=6))")
    
    # Определяем output_dim на основе use_derivatives из конфигурации
    output_dim = nn_config.use_derivatives ? 24 : 8
    println("✓ Размерность выхода нейросети: $output_dim (PML: $(pml_config.enabled ? "включён" : "отключён"), производные: $(nn_config.use_derivatives))")
    
    # Обновляем конфигурацию нейросети с правильной размерностью выхода
    nn_config_updated = NeuralNetworkConfig(;
        input_dim=nn_config.input_dim,
        hidden_layers=nn_config.hidden_layers,
        output_dim=output_dim,  # Всегда 8
        activation=nn_config.activation,
        use_gpu=nn_config.use_gpu
    )
    
    # Валидация конфигураций
    validate_config(nn_config_updated)
    validate_optimization_config(opt_config)
    
    # Создаем физические константы
    constants = PhysicalConstants()
    
    # Создаем переменные и области с учетом use_derivatives из конфига
    variables = create_variables(use_derivatives=nn_config.use_derivatives)
    domains = create_domains(variables, domain_config["x_range"], domain_config["y_range"], 
                            domain_config["z_range"], domain_config["t_range"])
    
    # Создаем граничные условия
    bcs = create_boundary_conditions(constants, variables, domains; pml_config=pml_config)

    # Создаем PDE систему с учетом PML и use_derivatives
    pde_system = create_pde_system(constants, variables, bcs, domains; pml_config=pml_config, include_derivatives=nn_config.use_derivatives)

    normalized_points = normalized_points |> gpu_device()
    println("✓ Используем нормированные измеренные точки")
    
    # Обновляем loss_config с измеренными точками и параметрами домена
    set_inner_domain!(loss_config, domain_config["x_range"], domain_config["y_range"], domain_config["z_range"], pml_config.pml_thickness_ratio)
    
    # Создаем нейронную сеть
    chain = create_neural_network(nn_config_updated)
    ps = initialize_parameters(chain, Random.default_rng(), nn_config_updated.use_gpu)
    
    return (chain=chain, ps=ps, constants=constants, variables=variables,
            domains=domains, pde_system=pde_system, bcs=bcs, 
            measured_points=normalized_points, configs=(nn_config=nn_config_updated, opt_config=opt_config, loss_config=loss_config, domain_config=domain_config, pml_config=pml_config),
            norm_factor=norm_factor)  # Добавляем норму в результат
end

"""
    _create_complete_setup_temporal(...)

Приватная функция для создания настройки с Temporal-Aware архитектурой.
Эта архитектура явно разделяет пространственные и временные признаки для лучшей поддержки
нестационарных решений.
"""
function _create_complete_setup_temporal(; measured_points, nn_config::TemporalAwareNetworkConfig, 
                               opt_config::OptimizationConfig,
                               loss_config::LossFunctionConfig,
                               domain_config::Dict{String, Any},
                               pml_config::PMLConfig)
    
    # Нормируем измеренные точки
    normalized_points, norm_factor = normalize_measured_points(measured_points)
    println("✓ Данные нормированы, фактор: $(round(norm_factor, digits=6))")
    
    # Валидация конфигурации (TemporalAwareNetworkConfig структурирована корректно по определению)
    validate_optimization_config(opt_config)
    
    println("✓ Temporal-Aware архитектура:")
    println("   Spatial: [x,y,z] → $(nn_config.spatial_hidden_layers) → $(nn_config.spatial_output_dim)D")
    println("   Temporal: Fourier($(nn_config.num_fourier_frequencies)) → $(nn_config.temporal_hidden_layers) → $(nn_config.temporal_output_dim)D")
    println("   Fusion: $(nn_config.spatial_output_dim + nn_config.temporal_output_dim)D → $(nn_config.fusion_hidden_layers) → $(nn_config.output_dim)D")
    println("   use_derivatives: $(nn_config.use_derivatives)")
    
    # Создаем физические константы
    constants = PhysicalConstants()
    
    # Создаем переменные и области с учетом use_derivatives из конфига
    variables = create_variables(use_derivatives=nn_config.use_derivatives)
    domains = create_domains(variables, domain_config["x_range"], domain_config["y_range"], 
                            domain_config["z_range"], domain_config["t_range"])
    
    # Создаем граничные условия
    bcs = create_boundary_conditions(constants, variables, domains; pml_config=pml_config)

    # Создаем PDE систему с учетом PML и use_derivatives
    pde_system = create_pde_system(constants, variables, bcs, domains; pml_config=pml_config, include_derivatives=nn_config.use_derivatives)

    normalized_points = normalized_points |> gpu_device()
    println("✓ Используем нормированные измеренные точки")
    
    # Обновляем loss_config с измеренными точками и параметрами домена
    set_inner_domain!(loss_config, domain_config["x_range"], domain_config["y_range"], domain_config["z_range"], pml_config.pml_thickness_ratio)
    
    # Создаем Temporal-Aware нейронную сеть
    chain = create_temporal_aware_network(nn_config)
    ps = initialize_temporal_aware_parameters(chain, Random.default_rng(); use_gpu=nn_config.use_gpu)
    
    return (chain=chain, ps=ps, constants=constants, variables=variables,
            domains=domains, pde_system=pde_system, bcs=bcs, 
            measured_points=normalized_points, configs=(nn_config=nn_config, opt_config=opt_config, loss_config=loss_config, domain_config=domain_config, pml_config=pml_config),
            norm_factor=norm_factor)

end

"""
    run_eeg_inverse_problem(nn_config, opt_config, loss_config, domain_config, pml_config)

Запускает полный эксперимент решения обратной задачи ЭЭГ.
"""
function run_eeg_inverse_problem(;measured_points, nn_config::Union{NeuralNetworkConfig, TemporalAwareNetworkConfig},
                                opt_config::OptimizationConfig,
                                loss_config::LossFunctionConfig,
                                domain_config::Dict{Any, Any}=Dict(
                                    "x_range" => [-10.f0, 10.0f0],
                                    "y_range" => [-10.0f0, 10.0f0], 
                                    "z_range" => [-10.0f0, 10.0f0],
                                    "t_range" => [0.0f0, 1.0f0]
                                ),
                                pml_config::PMLConfig=PMLConfig(),
                                )
    domain_config= Dict{String, Any}(domain_config)
    println("🚀 Запуск эксперимента обратной задачи ЭЭГ...")
    
    if pml_config.enabled
        gamma_str = pml_config.gamma_max === nothing ? "auto" : string(round(pml_config.gamma_max, digits=4))
        alpha_str = pml_config.alpha_max === nothing ? "auto" : string(round(pml_config.alpha_max, digits=4))
        println("✅ PML включён (толщина: $(round(pml_config.pml_thickness_ratio*100))%, R: $(pml_config.reflection_coefficient))")
        println("   γ_max: $gamma_str, α_max: $alpha_str")
    else
        println("⚠️ PML отключён")
    end
    
    # Создаем полную настройку
    setup = create_complete_setup(; measured_points, nn_config, opt_config, loss_config, domain_config, pml_config)
    
    println("✓ Настройка создана")
    
    # Создаем discretization с адаптивным весом lambda_data
    discretization, lambda_data_ref = create_discretization(setup.chain, setup.ps, setup.configs.loss_config, 
                                          setup.configs.opt_config, setup.configs.domain_config)

    println("✓ Discretization создан (адаптивный планировщик: $(setup.configs.loss_config.lambda_schedule_type))")
    
    # Дискретизация PDE системы
    prob = discretize(setup.pde_system, discretization)

    
    println("✓ PDE система дискретизирована")
    
    # Создаем функцию для вычисления "сырого" data loss (передаём phi из discretization)
    # data_loss возвращает (mse=mse, deriv=deriv_loss), derivative_loss возвращает отдельный loss
    loss_config = setup.configs.loss_config
    phi_pred_fun = discretization.phi
    
    # Создаём обёртку которая вычисляет (total=mse+deriv, mse=mse, deriv=deriv_loss)
    data_loss_raw_func = let cfg = loss_config, phi = phi_pred_fun
        function(p_vec)
            dev = Lux.gpu_device()
            
            # Получаем данные из конфига
            coords_gpu = cfg.measured_points_coords |> dev
            values_gpu = cfg.measured_points_values |> dev
            n_points = cfg.n_measured_points
            
            # Переносим параметры на GPU
            p_gpu = p_vec |> dev
            
            # Вычисляем MSE
            mse_val = data_loss(phi, p_gpu, coords_gpu, values_gpu, n_points)
            
            # Вычисляем derivative loss если доступны производные
            deriv_val = Float32(0.0)
            if length(cfg.derivatives) > 0 && size(cfg.deriv_coords, 2) > 0
                deriv_val = derivative_loss(
                    phi, p_gpu, 
                    coords_gpu, values_gpu, n_points,
                    cfg.derivatives, cfg.deriv_coords,
                    cfg.num_sensors, cfg.num_time_steps, cfg.measured_time
                )
            end
            
            return (total=mse_val + deriv_val, mse=mse_val, deriv=deriv_val)
        end
    end
    
    # Создаем callback функцию с адаптивным балансом
    callback = create_optimization_callback(setup.configs.opt_config, discretization, 
                                          setup.pde_system, setup.bcs, setup.domains,
                                          setup.configs.loss_config, lambda_data_ref, data_loss_raw_func)
    
    # Настраиваем оптимизатор
    opt = setup_optimization(setup.configs.opt_config)
    
    println("🔄 Начинаем оптимизацию...")
    
    # Проверяем, нужно ли использовать трёхэтапную оптимизацию
    opt_config = setup.configs.opt_config
    loss_config = setup.configs.loss_config
    
    if opt_config.three_stage_optimization
        println("🎯 Используем трёхэтапную оптимизацию")
        res = run_three_stage_optimization(prob, opt_config, loss_config,
                                           discretization, setup.pde_system, setup.bcs, setup.domains, data_loss_raw_func, lambda_data_ref)
    else
        # Запускаем стандартную оптимизацию
        res = solve(prob, opt; maxiters = setup.configs.opt_config.max_iterations, callback)
    end
    
    println("✓ Оптимизация завершена")
    
    # Извлекаем результаты
    phi = discretization.phi
    final_params = res.u
    
    # Анализируем результаты
    results = analyze_results(phi, final_params, setup, domain_config)
    
    return (solution=res, discretization=discretization, phi=phi, 
            params=final_params, results=results, setup=setup,
            final_lambda_data=lambda_data_ref[])
end

"""
    analyze_results(phi, params, setup, domain_config)

Анализирует результаты оптимизации на датчиках и создает метрики.
Сравнивает предсказанные значения потенциала φ на датчиках с измеренными.
"""
function analyze_results(phi, params, setup, domain_config)
    cpud = cpu_device()
    
    # Получаем измеренные точки с датчиков
    measured_points = setup.measured_points
    norm_factor = setup.norm_factor
    cpud = cpu_device()
    
    # Батчированная обработка всех точек одновременно (аналогично loss функциям)
    # ВАЖНО: Используем hcat для векторизованного стекирования вместо цикла
    n_points = length(measured_points)
    
    if n_points > 0
        # Объединяем все точки в одну матрицу [5, N]
        all_data = hcat(measured_points...)
        
        # Переносим на CPU и извлекаем компоненты
        all_data_cpu = all_data isa CuArray ? (all_data |> cpud) : all_data
        
        # [4, N] - координаты
        coords_batch = Float32.(all_data_cpu[1:4, :])
        # [N] - измеренные значения
        measured_phi_norm = vec(Float32.(all_data_cpu[5, :]))
        
        # Вызываем сеть один раз для всех точек
        pred_all = phi(coords_batch, params |> cpud)
        # Извлекаем только φ (первая строка) и переносим на CPU
        phi_pred_norm = Float32.(vec(pred_all[1, :]) |> cpud)
        
        # Деанормализуем все значения сразу
        measured_phi = measured_phi_norm .* norm_factor
        predicted_phi = phi_pred_norm .* norm_factor
        
        # Извлекаем координаты
        x_coords = vec(coords_batch[1, :])
        y_coords = vec(coords_batch[2, :])
        z_coords = vec(coords_batch[3, :])
        t_coords = vec(coords_batch[4, :])
    else
        # Пустой случай - инициализируем все переменные явно
        x_coords = Float32[]
        y_coords = Float32[]
        z_coords = Float32[]
        t_coords = Float32[]
        measured_phi = Float32[]
        predicted_phi = Float32[]
    end
    
    # Группируем по временным шагам для анализа динамики
    # Явно приводим к Float32 для соответствия типу словаря
    time_steps = Float32.(collect(unique(t_coords)))
    sort!(time_steps)
    
    # Словарь для хранения метрик по временным шагам
    time_step_metrics = Dict{Float32, Dict{String, Float32}}()
    
    for t_step in time_steps
        # Находим точки для текущего временного шага
        mask = t_coords .== t_step
        
        if sum(mask) > 0
            measured_t = measured_phi[mask]
            predicted_t = predicted_phi[mask]
            
            # Вычисляем метрики для данного временного шага
            mse_t = sum((measured_t .- predicted_t).^2) / length(measured_t)
            mae_t = sum(abs.(measured_t .- predicted_t)) / length(measured_t)
            max_error_t = maximum(abs.(measured_t .- predicted_t))
            
            time_step_metrics[t_step] = Dict(
                "mse" => mse_t,
                "mae" => mae_t,
                "max_error" => max_error_t,
                "num_sensors" => sum(mask)
            )
        end
    end
    
    # Общие метрики по всем датчикам (с защитой от деления на ноль)
    n_measured = length(measured_phi)
    if n_measured > 0
        overall_mse = sum((measured_phi .- predicted_phi).^2) / n_measured
        overall_mae = sum(abs.(measured_phi .- predicted_phi)) / n_measured
        overall_max_error = maximum(abs.(measured_phi .- predicted_phi))
    else
        overall_mse = Float32(0.0)
        overall_mae = Float32(0.0)
        overall_max_error = Float32(0.0)
    end
    
    # Средние метрики по временным шагам (с защитой от пустого словаря)
    if !isempty(time_step_metrics)
        avg_time_mse = mean([metrics["mse"] for metrics in values(time_step_metrics)])
        avg_time_mae = mean([metrics["mae"] for metrics in values(time_step_metrics)])
        avg_time_max_error = mean([metrics["max_error"] for metrics in values(time_step_metrics)])
    else
        avg_time_mse = Float32(0.0)
        avg_time_mae = Float32(0.0)
        avg_time_max_error = Float32(0.0)
    end
    
    # Добавляем расчет значений φ по равномерной сетке для каждого временного шага
    println("📊 Расчет значений φ по равномерной сетке для визуализации...")
    
    # Создаем равномерную сетку для визуализации
    grid_resolution = 30  # Разрешение сетки
    x_grid = range(domain_config["x_range"][1], domain_config["x_range"][2], length=grid_resolution)
    y_grid = range(domain_config["y_range"][1], domain_config["y_range"][2], length=grid_resolution)
    z_grid = range(domain_config["z_range"][1], domain_config["z_range"][2], length=grid_resolution)
    
    # Словарь для хранения полей потенциала по времени
    uniform_grid_data = Dict{Float32, Dict{String, Any}}()
    
    # Выбираем 5 равномерно распределенных временных шагов
    n_time_steps = length(time_steps)
    selected_indices = collect(1:round(Int, n_time_steps/4):n_time_steps)
    selected_time_steps = time_steps[selected_indices]
    
    for t_step in selected_time_steps
        println("  ⏰ Обработка временного шага t = $(round(t_step, digits=3))")
        
        # Создаем 3D сетку для данного временного шага
        phi_field = Array{Float32, 3}(undef, length(x_grid), length(y_grid), length(z_grid))
        
        # Заполняем сетку значениями потенциала
        for (i, x) in enumerate(x_grid)
            for (j, y) in enumerate(y_grid)
                for (k, z) in enumerate(z_grid)
                    try
                        phi_val = (phi([x, y, z, t_step], params|>cpud))[1]
                        phi_field[i, j, k] = phi_val * norm_factor  # Деанормируем
                    catch e
                        # Если произошла ошибка, заполняем нулем
                        phi_field[i, j, k] = 0.0f0
                    end
                end
            end
        end
        
        # Сохраняем данные для данного временного шага
        uniform_grid_data[t_step] = Dict(
            "x_grid" => collect(x_grid),
            "y_grid" => collect(y_grid), 
            "z_grid" => collect(z_grid),
            "phi_field" => phi_field,
            "time_step" => t_step
        )
    end
    
    println("✓ Расчет равномерной сетки завершен")
    
    results = Dict(
        # Общие метрики
        "overall_mse" => overall_mse,
        "overall_mae" => overall_mae, 
        "overall_max_error" => overall_max_error,
        
        # Средние метрики по временным шагам
        "avg_time_mse" => avg_time_mse,
        "avg_time_mae" => avg_time_mae,
        "avg_time_max_error" => avg_time_max_error,
        
        # Метрики по каждому временному шагу
        "time_step_metrics" => time_step_metrics,
        
        # Данные для визуализации
        "sensor_data" => Dict(
            "x_coords" => x_coords,
            "y_coords" => y_coords,
            "z_coords" => z_coords,
            "t_coords" => t_coords,
            "measured_phi" => measured_phi,
            "predicted_phi" => predicted_phi
        ),
        
        # Временные шаги
        "time_steps" => time_steps,
        
        # Данные равномерной сетки для визуализации полей
        "uniform_grid_data" => uniform_grid_data,
        
        # Информация о сетке
        "grid_info" => Dict(
            "x_range" => domain_config["x_range"],
            "y_range" => domain_config["y_range"],
            "z_range" => domain_config["z_range"],
            "grid_resolution" => grid_resolution
        ),
        
        # Статистика
        "num_sensors" => length(measured_points),
        "num_time_steps" => length(time_steps),
        "final_loss" => sum(abs2.(measured_phi .- predicted_phi)),
        
        # Нормирование
        "norm_factor" => norm_factor
    )
    
    return results
end

"""
    save_results(results, filename)

Сохраняет результаты эксперимента в файл.
"""
function save_results(results, filename::String)
    # Сохраняем основные результаты
    jldopen(filename, "w") do file
        file["solution"] = results.solution
        file["discretization"] = results.discretization
        file["phi"] = results.phi
        file["params"] = results.params
        file["results"] = results.results
        file["setup"] = results.setup
    end
    
    println("✓ Результаты сохранены в $filename")
end

"""
    load_results(filename)

Загружает результаты эксперимента из файла.
"""
function load_results(filename::String)
    if !isfile(filename)
        throw(ArgumentError("Файл $filename не найден"))
    end
    
    jldopen(filename, "r") do file
        result = (
            solution = haskey(file, "solution") ? file["solution"] : nothing,
            discretization = haskey(file, "discretization") ? file["discretization"] : nothing,
            phi = haskey(file, "phi") ? file["phi"] : nothing,
            params = haskey(file, "params") ? file["params"] : nothing,
            results = haskey(file, "results") ? file["results"] : nothing,
            setup = haskey(file, "setup") ? file["setup"] : nothing,
            final_lambda_data = haskey(file, "final_lambda_data") ? file["final_lambda_data"] : nothing,
        )
        
        println("✓ Результаты загружены из $filename")
        return result
    end
end

"""
    create_visualization_plots(results, save_path="figures/")

Создает визуализацию результатов эксперимента на датчиках.
"""
function create_visualization_plots(results, save_path::String="figures/")
    
    # Создаем директорию если не существует
    mkpath(save_path)
    
    # Получаем данные с датчиков
    sensor_data = results["sensor_data"]
    x_coords = sensor_data["x_coords"]
    y_coords = sensor_data["y_coords"]
    t_coords = sensor_data["t_coords"]
    measured_phi = sensor_data["measured_phi"]
    predicted_phi = sensor_data["predicted_phi"]
    
    # Получаем данные равномерной сетки
    uniform_grid_data = results["uniform_grid_data"]
    grid_info = results["grid_info"]
    
    # Создаем графики для разных временных шагов
    unique_times = unique(t_coords)
    sort!(unique_times)
    
    # Выбираем несколько временных шагов для визуализации (по данным датчиков)
    sensor_time_indices = unique_times[1:min(3, length(unique_times))]
    
    # Также показываем графики по равномерной сетке
    grid_time_indices = sort(collect(keys(uniform_grid_data)))
    grid_time_indices = grid_time_indices[1:min(3, length(grid_time_indices))]
    
    plots = []
    
    # Графики 1: Данные датчиков с интерполяцией
    for (i, t_step) in enumerate(sensor_time_indices)
        # Фильтруем данные для текущего временного шага
        mask = t_coords .== t_step
        x_t = x_coords[mask]
        y_t = y_coords[mask]
        measured_t = measured_phi[mask]
        predicted_t = predicted_phi[mask]
        
        # Создаем сетку для интерполяции для лучшей визуализации
        x_range = range(minimum(x_t), maximum(x_t), length=20)
        y_range = range(minimum(y_t), maximum(y_t), length=20)
        
        # Интерполируем данные на сетку для контурных графиков
        measured_grid = Matrix{Float32}(undef, length(x_range), length(y_range))
        predicted_grid = Matrix{Float32}(undef, length(x_range), length(y_range))
        error_grid = Matrix{Float32}(undef, length(x_range), length(y_range))
        
        for j in axes(x_range, 1)
            for k in axes(y_range, 1)
                # Находим ближайшие точки датчиков
                distances = sqrt.((x_t .- x_range[j]).^2 .+ (y_t .- y_range[k]).^2)
                if minimum(distances) < 2.0  # Если есть достаточно близкие датчики
                    # Используем обратно-взвешенную интерполяцию
                    weights = 1.0 ./ (distances .^ 2 .+ 1e-10)
                    total_weight = sum(weights)
                    if total_weight > 0
                        measured_grid[j, k] = sum(weights .* measured_t) / total_weight
                        predicted_grid[j, k] = sum(weights .* predicted_t) / total_weight
                        error_grid[j, k] = abs(measured_grid[j, k] - predicted_grid[j, k])
                    else
                        measured_grid[j, k] = NaN
                        predicted_grid[j, k] = NaN
                        error_grid[j, k] = NaN
                    end
                else
                    measured_grid[j, k] = NaN
                    predicted_grid[j, k] = NaN
                    error_grid[j, k] = NaN
                end
            end
        end
        
        # Создаем графики для данного временного шага
        p1 = heatmap(x_range, y_range, measured_grid', 
                     title="Измеренный потенциал φ (t=$(round(t_step, digits=2)))", 
                     color=:viridis, aspect_ratio=:equal)
        
        p2 = heatmap(x_range, y_range, predicted_grid', 
                     title="Предсказанный потенциал φ (t=$(round(t_step, digits=2)))", 
                     color=:viridis, aspect_ratio=:equal)
        
        p3 = heatmap(x_range, y_range, error_grid', 
                     title="Ошибка |φ_измеренный - φ_предсказанный| (t=$(round(t_step, digits=2)))", 
                     color=:reds, aspect_ratio=:equal)
        
        # Добавляем точки датчиков на графики
        scatter!(p1, x_t, y_t, markersize=4, color=:white, markerstrokecolor=:black, label="Датчики")
        scatter!(p2, x_t, y_t, markersize=4, color=:white, markerstrokecolor=:black, label="Датчики")
        scatter!(p3, x_t, y_t, markersize=4, color=:white, markerstrokecolor=:black, label="Датчики")
        
        push!(plots, plot(p1, p2, p3, layout=(1,3), size=(1200, 400)))
    end
    
    # Графики 2: Поля потенциала на равномерной сетке (показываем полное поле)
    for (i, t_step) in enumerate(grid_time_indices)
        grid_data = uniform_grid_data[t_step]
        x_grid = grid_data["x_grid"]
        y_grid = grid_data["y_grid"]
        z_grid = grid_data["z_grid"]
        phi_field = grid_data["phi_field"]
        
        # Показываем срез в плоскости z=0 (среднее значение по z)
        z_center_idx = length(z_grid) ÷ 2
        phi_slice = phi_field[:, :, z_center_idx]
        
        p4 = heatmap(x_grid, y_grid, phi_slice', 
                     title="Поле потенциала φ на равномерной сетке (t=$(round(t_step, digits=2)))", 
                     color=:viridis, aspect_ratio=:equal)
        
        # Добавляем точки датчиков
        mask = t_coords .== t_step
        if sum(mask) > 0
            x_t = x_coords[mask]
            y_t = y_coords[mask]
            scatter!(p4, x_t, y_t, markersize=3, color=:red, markerstrokecolor=:white, 
                    label="Датчики", alpha=0.8)
        end
        
        push!(plots, p4)
    end
    
    # Объединяем все графики
    if length(plots) > 1
        combined_plot = plot(plots..., layout=(length(plots), 1), size=(1200, 400*length(plots)))
    else
        combined_plot = plots[1]
    end
    
    # Сохраняем
    savefig(combined_plot, joinpath(save_path, "eeg_inverse_problem_sensor_results.png"))
    
    # Создаем дополнительный график с метриками по времени
    time_steps = results["time_steps"]
    time_metrics = results["time_step_metrics"]
    
    # Защита от пустых данных
    if isempty(time_steps) || isempty(time_metrics)
        println("⚠️ Нет данных для построения графика метрик по времени")
        return (combined_plot, nothing)
    end
    
    mse_values = [time_metrics[t]["mse"] for t in time_steps]
    mae_values = [time_metrics[t]["mae"] for t in time_steps]
    max_error_values = [time_metrics[t]["max_error"] for t in time_steps]
    
    metrics_plot = plot(time_steps, mse_values, label="MSE", xlabel="Временной шаг", ylabel="Ошибка")
    plot!(time_steps, mae_values, label="MAE")
    plot!(time_steps, max_error_values, label="Макс. ошибка", yscale=:log10)
    title!("Метрики ошибки по временным шагам на датчиках")
    
    savefig(metrics_plot, joinpath(save_path, "eeg_inverse_problem_metrics_timeline.png"))
    
    println("✓ Визуализация сохранена в $save_path")
    
    return (combined_plot, metrics_plot)
end

"""
    benchmark_experiment(nn_configs, opt_configs, domain_config)

Запускает несколько экспериментов с разными конфигурациями для сравнения.
"""
function benchmark_experiment(nn_configs::Vector{NeuralNetworkConfig},
                             opt_configs::Vector{OptimizationConfig},
                             domain_config::Dict{String, Any}=Dict(
                                 "x_range" => [-10.0, 10.0],
                                 "y_range" => [-10.0, 10.0], 
                                 "z_range" => [-10.0, 10.0],
                                 "t_range" => [0.0, 1.0],
                                 "num_points" => 100
                             ),
                             pml_config::PMLConfig=PMLConfig())
    
    results_comparison = []
    
    for (i, (nn_config, opt_config)) in enumerate(zip(nn_configs, opt_configs))
        println("🔄 Запуск эксперимента $i/$(length(nn_configs))...")
        
        try
            # Запускаем эксперимент
            exp_results = run_eeg_inverse_problem(nn_config, opt_config, 
                                                 LossFunctionConfig(), domain_config, pml_config=pml_config)
            
            # Сохраняем результаты
            push!(results_comparison, (config_i=i, results=exp_results))
            
            println("✓ Эксперимент $i завершен успешно")
            
        catch e
            println("❌ Эксперимент $i завершился с ошибкой: $e")
            continue
        end
    end
    
    return results_comparison
end

"""
    get_available_devices()

Возвращает информацию о доступных вычислительных устройствах.
"""
function get_available_devices()
    devices = Dict{String, Any}()
    
    # CPU
    devices["cpu"] = true
    
    # GPU (CUDA) - проверяем доступность CUDA без перехвата ошибок
    if CUDA.functional()
        devices["cuda"] = true
        # Используем безопасный способ получения количества устройств
        device_count = length(CUDA.devices())
        devices["cuda_device_count"] = device_count
        if device_count > 0
            devices["cuda_device_name"] = [CUDA.name(dev) for dev in CUDA.devices()]
        else
            devices["cuda_device_name"] = String[]
        end
    else
        devices["cuda"] = false
    end
    
    return devices
end

"""
    run_three_stage_optimization(prob, opt_config, loss_config,
                                  discretization, pde_system, bcs, domains, data_loss_raw_func, lambda_data_ref)

Запускает трёхэтапную оптимизацию:
1. Adam с высоким LR (20% итераций)
2. LBFGS для уточнения (10% итераций) 
3. Adam с Cosine Annealing scheduler (70% итераций)

# Аргументы
- `prob`: OptimizationProblem
- `opt_config`: OptimizationConfig с параметрами оптимизатора (max_iterations)
- `loss_config`: LossFunctionConfig с параметрами loss
- `discretization`: Объект дискретизации для создания callback'ов
- `pde_system`: Система PDE
- `bcs`: Граничные условия
- `domains`: Области
- `data_loss_raw_func`: Функция для вычисления сырого data loss
- `lambda_data_ref`: Ссылка на текущее значение веса данных

# Возвращает
- Результат оптимизации
"""
function run_three_stage_optimization(prob, opt_config::OptimizationConfig, loss_config::LossFunctionConfig,
                                       discretization, pde_system, bcs, domains, data_loss_raw_func, lambda_data_ref::Base.RefValue{Float32})
    # Используем max_iterations из opt_config
    total_iters = opt_config.max_iterations > 0 ? opt_config.max_iterations : 3000
    
    # Вычисляем количество итераций для каждого этапа из opt_config
    adam1_iters = round(Int, total_iters * opt_config.adam1_ratio)
    lbfgs_iters = round(Int, total_iters * opt_config.lbfgs_ratio)
    adam2_iters = round(Int, total_iters * opt_config.adam2_ratio)
    

    println("🔄 Трёхэтапная оптимизация:")
    println("   Этап 1: Adam (LR=$(opt_config.adam1_lr), итераций=$adam1_iters)")
    println("   Этап 2: LBFGS (итераций=$lbfgs_iters)")
    println("   Этап 3: Adam + Scheduler (LR=$(opt_config.adam2_lr), итераций=$adam2_iters)")
    
    # Создаём копию prob с начальными параметрами
    current_params = prob.u0
    
    dev = Lux.gpu_device()
    
    # === СОЗДАЁМ ОБЩИЙ ЛОГГЕР ДЛЯ ВСЕХ ЭТАПОВ ===
    logger = opt_config.use_tensorboard ? TBLogger(opt_config.log_directory) : nothing
    if logger !== nothing
        println("   📝 TensorBoard логгер создан: $(opt_config.log_directory)")
    end
    
    # === ЭТАП 1: Adam с высоким LR ===
    println("\n📈 Этап 1: Adam с высоким LR")
    opt1 = OptimizationOptimisers.Adam(opt_config.adam1_lr)
    
    # Создаём новый prob для каждого этапа
    prob1 = SciMLBase.remake(prob; u0=current_params)
    
    # Callback для этапа 1 начинает с initial_iter=0 (по умолчанию)
    # Передаём общий logger
    callback_stage1 = create_optimization_callback(
        opt_config, discretization, pde_system, bcs, domains,
        loss_config, lambda_data_ref, data_loss_raw_func;
        initial_iter=0,
        logger=logger
    )
    
    res1 = solve(prob1, opt1; maxiters=adam1_iters, callback=callback_stage1)
    current_params = res1.u
    
    println("   Loss после этапа 1: $(res1.objective)")
    
    # === ЭТАП 2: LBFGS ===
    println("\n📉 Этап 2: LBFGS")
    opt2 = OptimizationOptimJL.LBFGS()
    
    prob2 = SciMLBase.remake(prob; u0=current_params)
    
    # Callback для этапа 2 начинает с initial_iter=adam1_iters
    # Передаём общий logger
    callback_stage2 = create_optimization_callback(
        opt_config, discretization, pde_system, bcs, domains,
        loss_config, lambda_data_ref, data_loss_raw_func;
        initial_iter=adam1_iters,
        logger=logger
    )
    
    res2 = solve(prob2, opt2; maxiters=lbfgs_iters, callback=callback_stage2)
    current_params = res2.u
    
    println("   Loss после этапа 2: $(res2.objective)")
    
    # === ЭТАП 3: Adam с Cosine Annealing ===
    println("\n📊 Этап 3: Adam с Cosine Annealing Scheduler")
    opt3 = OptimizationOptimisers.Adam(opt_config.adam2_lr)
    
    prob3 = SciMLBase.remake(prob; u0=current_params)
    
    # Создаём scheduler функцию
    # Счетчик итераций для scheduler общий через все этапы (начинается с adam1_iters + lbfgs_iters)
    total_iters_before_stage3 = adam1_iters + lbfgs_iters
    scheduler_fn = opt_config.scheduler_type == :warmup_cosine ?
        (t -> warmup_cosine_lr(Float32(opt_config.adam2_lr), t, adam2_iters, opt_config.scheduler_warmup)) :
        (t -> cosine_annealing_lr(Float32(opt_config.adam2_lr), t, adam2_iters))
    
    # Создаём callback с поддержкой scheduler через общий create_optimization_callback
    # initial_iter - для логирования (общий счетчик)
    # initial_scheduler_iter - для scheduler (начинается с 0 на 3-м этапе, т.к. scheduler_fn работает относительно своего этапа)
    # Передаём общий logger
    callback_stage3 = create_optimization_callback(
        opt_config, discretization, pde_system, bcs, domains,
        loss_config, lambda_data_ref, data_loss_raw_func;
        scheduler_fn=scheduler_fn,
        initial_iter=total_iters_before_stage3,  # Для логирования - общий счетчик
        logger=logger
    )
    
    res3 = solve(prob3, opt3; maxiters=adam2_iters, callback=callback_stage3)
    current_params = res3.u
    
    println("   Loss после этапа 3: $(res3.objective)")
    
    # Объединяем результаты
    total_loss = res3.objective
    
    println("\n✅ Трёхэтапная оптимизация завершена. Финальный loss: $(total_loss)")
    
    # Возвращаем результат в том же формате, что и solve
    return res3
end

# Инициализация при загрузке модуля
function __init__()
    println("📦 InverseProblem модуль загружен")
    
    # Проверяем доступные устройства
    devices = get_available_devices()
    println("🖥️  Доступные устройства: $devices")
end

end # module
