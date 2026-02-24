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

using ..PDEDefinitions
using ..NeuralNetwork
using ..Optimization
using ..PML

using NeuralPDE, Lux, LuxCUDA, Random, ComponentArrays, CUDA
using ModelingToolkit: @named
using ..PDEDefinitions: PhysicalConstants, create_variables, create_domains, create_pde_system, create_boundary_conditions, generate_measured_points, analytic_sol_func 
using ..NeuralNetwork: NeuralNetworkConfig, create_neural_network, initialize_parameters, validate_config
using ..Optimization: OptimizationConfig, LossFunctionConfig, validate_optimization_config, create_discretization, create_optimization_callback, create_data_loss_raw_func, setup_optimization, solve
using Plots
using Statistics: mean
using JLD2: jldopen
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


# Структура конфигурации домена
struct DomainConfig
    x_range::Vector{Float64}
    y_range::Vector{Float64}
    z_range::Vector{Float64}
    t_range::Vector{Float64}
    num_points::Int
    
    function DomainConfig(; x_range=[-10.0, 10.0], y_range=[-10.0, 10.0], 
                          z_range=[-10.0, 10.0], t_range=[0.0, 1.0], num_points=100)
        return new(x_range, y_range, z_range, t_range, num_points)
    end
end

# Экспортируем основные функции
export run_eeg_inverse_problem, create_complete_setup
export analyze_results, save_results, load_results, DomainConfig, PMLConfig

"""
    create_complete_setup(; nn_config, opt_config, loss_config, domain_config, pml_config)

Создает полную настройку для эксперимента обратной задачи ЭЭГ.

Примечание: Размерность выхода нейросети всегда равна 8 (φ, Ax, Ay, Az, ρ, jx, jy, jz).
PML через затухание и экранирование не требует дополнительных выходов.
"""
function create_complete_setup(; measured_points, nn_config::NeuralNetworkConfig, 
                               opt_config::OptimizationConfig,
                               loss_config::LossFunctionConfig,
                               domain_config::Dict{String, Any}=Dict(
                                   "x_range" => [-10.0, 10.0],
                                   "y_range" => [-10.0, 10.0], 
                                   "z_range" => [-10.0, 10.0],
                                   "t_range" => [0.0, 1.0],
                                   "num_points" => 100
                               ),
                               pml_config::PMLConfig=PMLConfig())
    
    # Нормируем измеренные точки
    normalized_points, norm_factor = normalize_measured_points(measured_points)
    println("✓ Данные нормированы, фактор: $(round(norm_factor, digits=6))")
    
    # Размерность выхода всегда равна 8 (новый подход PML не требует дополнительных выходов)
    output_dim = 8
    println("✓ Размерность выхода нейросети: $output_dim (PML: $(pml_config.enabled ? "включён" : "отключён"))")
    
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
    
    # Создаем переменные и области
    variables = create_variables()
    domains = create_domains(variables, domain_config["x_range"], domain_config["y_range"], 
                            domain_config["z_range"], domain_config["t_range"])
    
    # Создаем граничные условия
    bcs = create_boundary_conditions(constants, variables, domains; pml_config=pml_config)

    # Создаем PDE систему с учетом PML
    pde_system = create_pde_system(constants, variables, bcs, domains; pml_config=pml_config)

    normalized_points = normalized_points |> gpu_device()
    println("✓ Используем нормированные измеренные точки")
    
    # Обновляем loss_config с измеренными точками
    loss_config = LossFunctionConfig(; 
                lambda_pde = loss_config.lambda_pde,
                lambda_bc = loss_config.lambda_bc,
                lambda_data_init = loss_config.lambda_data_init,
                lambda_min = loss_config.lambda_min,
                lambda_max = loss_config.lambda_max,
                lambda_schedule_type = loss_config.lambda_schedule_type,
                lambda_schedule = loss_config.lambda_schedule,
                measured_points=normalized_points)
    
    # Создаем нейронную сеть
    chain = create_neural_network(nn_config_updated)
    ps = initialize_parameters(chain, Random.default_rng(), nn_config_updated.use_gpu)
    
    return (chain=chain, ps=ps, constants=constants, variables=variables,
            domains=domains, pde_system=pde_system, bcs=bcs, 
            measured_points=normalized_points, configs=(nn_config=nn_config_updated, opt_config=opt_config, loss_config=loss_config, domain_config=domain_config, pml_config=pml_config),
            norm_factor=norm_factor)  # Добавляем норму в результат
end

"""
    run_eeg_inverse_problem(nn_config, opt_config, loss_config, domain_config, pml_config)

Запускает полный эксперимент решения обратной задачи ЭЭГ.
"""
function run_eeg_inverse_problem(;measured_points, nn_config::NeuralNetworkConfig,
                                opt_config::OptimizationConfig,
                                loss_config::LossFunctionConfig,
                                domain_config::Dict{Any, Any}=Dict(
                                    "x_range" => [-10.0, 10.0],
                                    "y_range" => [-10.0, 10.0], 
                                    "z_range" => [-10.0, 10.0],
                                    "t_range" => [0.0, 1.0]
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
    data_loss_raw_func = create_data_loss_raw_func(setup.configs.loss_config, discretization.phi)
    
    # Создаем callback функцию с адаптивным балансом
    callback = create_optimization_callback(setup.configs.opt_config, discretization, 
                                          setup.pde_system, setup.bcs, setup.domains,
                                          setup.configs.loss_config, lambda_data_ref, data_loss_raw_func)
    
    # Настраиваем оптимизатор
    opt = setup_optimization(setup.configs.opt_config)
    
    println("🔄 Начинаем оптимизацию...")
    
    # Запускаем оптимизацию
    res = solve(prob, opt; maxiters = setup.configs.opt_config.max_iterations, callback)
    
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
        coords_batch = Float64.(all_data_cpu[1:4, :])
        # [N] - измеренные значения
        measured_phi_norm = vec(Float64.(all_data_cpu[5, :]))
        
        # Вызываем сеть один раз для всех точек
        pred_all = phi(coords_batch, params |> cpud)
        # Извлекаем только φ (первая строка) и переносим на CPU
        phi_pred_norm = Float64.(vec(pred_all[1, :]) |> cpud)
        
        # Деанормализуем все значения сразу
        measured_phi = measured_phi_norm .* norm_factor
        predicted_phi = phi_pred_norm .* norm_factor
        
        # Извлекаем координаты
        x_coords = vec(coords_batch[1, :])
        y_coords = vec(coords_batch[2, :])
        z_coords = vec(coords_batch[3, :])
        t_coords = vec(coords_batch[4, :])
    else
        # Пустой случай
        x_coords = Float64[]
        y_coords = Float64[]
        z_coords = Float64[]
        t_coords = Float64[]
        measured_phi = Float64[]
        predicted_phi = Float64[]
    end
    
    # Группируем по временным шагам для анализа динамики
    time_steps = collect(unique(t_coords))
    sort!(time_steps)
    
    # Словарь для хранения метрик по временным шагам
    time_step_metrics = Dict{Float64, Dict{String, Float64}}()
    
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
    
    # Общие метрики по всем датчикам
    overall_mse = sum((measured_phi .- predicted_phi).^2) / length(measured_phi)
    overall_mae = sum(abs.(measured_phi .- predicted_phi)) / length(measured_phi)
    overall_max_error = maximum(abs.(measured_phi .- predicted_phi))
    
    # Средние метрики по временным шагам
    avg_time_mse = mean([metrics["mse"] for metrics in values(time_step_metrics)])
    avg_time_mae = mean([metrics["mae"] for metrics in values(time_step_metrics)])
    avg_time_max_error = mean([metrics["max_error"] for metrics in values(time_step_metrics)])
    
    # Добавляем расчет значений φ по равномерной сетке для каждого временного шага
    println("📊 Расчет значений φ по равномерной сетке для визуализации...")
    
    # Создаем равномерную сетку для визуализации
    grid_resolution = 30  # Разрешение сетки
    x_grid = range(domain_config["x_range"][1], domain_config["x_range"][2], length=grid_resolution)
    y_grid = range(domain_config["y_range"][1], domain_config["y_range"][2], length=grid_resolution)
    z_grid = range(domain_config["z_range"][1], domain_config["z_range"][2], length=grid_resolution)
    
    # Словарь для хранения полей потенциала по времени
    uniform_grid_data = Dict{Float64, Dict{String, Any}}()
    
    # Выбираем 5 равномерно распределенных временных шагов
    n_time_steps = length(time_steps)
    selected_indices = collect(1:round(Int, n_time_steps/4):n_time_steps)
    selected_time_steps = time_steps[selected_indices]
    
    for t_step in selected_time_steps
        println("  ⏰ Обработка временного шага t = $(round(t_step, digits=3))")
        
        # Создаем 3D сетку для данного временного шага
        phi_field = Array{Float64, 3}(undef, length(x_grid), length(y_grid), length(z_grid))
        
        # Заполняем сетку значениями потенциала
        for (i, x) in enumerate(x_grid)
            for (j, y) in enumerate(y_grid)
                for (k, z) in enumerate(z_grid)
                    try
                        phi_val = (phi([x, y, z, t_step], params|>cpud))[1]
                        phi_field[i, j, k] = phi_val * norm_factor  # Деанормируем
                    catch e
                        # Если произошла ошибка, заполняем нулем
                        phi_field[i, j, k] = 0.0
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
    JLD2.jldopen(filename, "w") do file
        file["results"] = results
        
        if haskey(results, "solution")
            file["solution"] = results.solution
        end
        
        if haskey(results, "params")
            file["params"] = results.params
        end
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
    
    results = JLD2.jldopen(filename, "r") do file
        return file["results"]
    end
    
    println("✓ Результаты загружены из $filename")
    return results
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
        measured_grid = Matrix{Float64}(undef, length(x_range), length(y_range))
        predicted_grid = Matrix{Float64}(undef, length(x_range), length(y_range))
        error_grid = Matrix{Float64}(undef, length(x_range), length(y_range))
        
        for j in 1:length(x_range)
            for k in 1:length(y_range)
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

# Инициализация при загрузке модуля
function __init__()
    println("📦 InverseProblem модуль загружен")
    
    # Проверяем доступные устройства
    devices = get_available_devices()
    println("🖥️  Доступные устройства: $devices")
end

end # module
