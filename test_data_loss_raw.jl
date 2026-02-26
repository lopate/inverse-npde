# Тест для проверки data_loss и derivative_loss функций из Optimization
# Запуск: julia --project test_data_loss_raw.jl

using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using Lux, Zygote, Statistics, Random, CUDA, NeuralPDE

# Подключаем модуль Optimization
push!(LOAD_PATH, joinpath(@__DIR__, "src"))
using InverseNpde.Optimization
using InverseNpde.NeuralNetwork

# Проверяем доступность GPU
if CUDA.functional()
    @info "GPU доступен, тестируем на GPU"
    using LuxCUDA
else
    @warn "GPU недоступен, тест будет пропущен"
    exit(0)
end

# ============================================================
# Создаём тестовые данные и конфигурацию
# ============================================================

# Параметры домена
x_range = [-10.0, 10.0]
y_range = [-10.0, 10.0]
z_range = [-10.0, 10.0]
t_range = [0.0, 1.0]

# Создаём простые синтетические измеренные точки
n_sensors = 5
n_time_steps = 10
n_points_total = n_sensors * n_time_steps

# Сетка датчиков на плоскости z=0
sensor_x = range(-5, 5, length=n_sensors)
sensor_y = range(-5, 5, length=n_sensors)
sensor_z = zeros(Float64, n_sensors)
time_vals = range(t_range[1], t_range[2], length=n_time_steps)

# Создаём тестовые данные - простая волна
measured_points = Vector{Float64}[]
for ti in 1:n_time_steps
    t = time_vals[ti]
    for si in 1:n_sensors
        x = sensor_x[si]
        y = sensor_y[si]
        z = sensor_z[si]
        # Простая пространственно-временная функция
        phi_measured = sin(x * 0.5) * cos(t) + cos(y * 0.5) * sin(t)
        push!(measured_points, [x, y, z, t, phi_measured])
    end
end

@info "Создано $n_points_total измеренных точек"

# ============================================================
# Создаём LossFunctionConfig
# ============================================================

loss_config = LossFunctionConfig(;
    lambda_pde=1.0,
    lambda_bc=1.0,
    lambda_data_init=1.0,
    lambda_min=0.01,
    lambda_max=100.0,
    lambda_time=1.0,
    lambda_schedule_type=:improvement,
    lambda_schedule=Dict{String, Any}("window_size" => 5, "increase_factor" => 1.2, "decrease_factor" => 0.98, "patience" => 3),
    # Параметры регуляризации энергии поля
    lambda_field=0.0,  # Отключаем для этого теста
    field_energy_scale=3.0,
    num_field_time_samples=3,
    # Параметры домена
    x_range=x_range,
    y_range=y_range,
    z_range=z_range,
    t_range=t_range,
    pml_thickness_ratio=0.1,
    measured_points=measured_points
)

@info "LossFunctionConfig создан"
@info "  Число измеренных точек: $(loss_config.n_measured_points)"
@info "  Число датчиков: $(loss_config.num_sensors)"
@info "  Число временных шагов: $(loss_config.num_time_steps)"

# ============================================================
# Создаём нейросеть
# ============================================================

chain = Lux.Chain(
    Dense(4, 32, Lux.σ),
    Dense(32, 32, Lux.σ),
    Dense(32, 8)  # 8 выходов: phi, Ax, Ay, Az, rho, jx, jy, jz
)

ps, st = Lux.setup(Random.default_rng(), chain)
dev = Lux.gpu_device()
ps_gpu = ps |> dev

@info "Нейросеть создана и перенесена на GPU"

# ============================================================
# Функция предсказания сети
# ============================================================

function phi_pred_fun(x, params)
    pred, _ = chain(x, params, st)
    return pred
end

# ============================================================
# Тест 1: Создаём phi функцию через discretization
# ============================================================

@info "=== Тест 1: Создание discretization и phi функции ==="

try
    # Создаём DomainConfig
    domain_config = Dict(
        "x_range" => x_range,
        "y_range" => y_range,
        "z_range" => z_range,
        "t_range" => t_range,
        "num_points" => 100
    )
    
    # Стратегия обучения
    strategy = QuasiRandomTraining(100)
    
    # Создаём PhysicsInformedNN без additional_loss (для простоты)
    discretization = PhysicsInformedNN(chain, strategy; init_params=ps_gpu)
    
    @info "  Discretization создан"
    @info "  Тип discretization: $(typeof(discretization))"
    
    # Получаем phi функцию
    phi_func = discretization.phi
    @info "  phi функция получена"
    
catch e
    @error "Ошибка при создании discretization: $e"
    @error "Стек: $(stacktrace(catch_backtrace()))"
end

# ============================================================
# Тест 2: Используем функцию data_loss напрямую
# ============================================================

@info "=== Тест 2: Использование data_loss ==="

try
    # Получаем данные из loss_config
    coords = loss_config.measured_points_coords
    values = loss_config.measured_points_values
    n_points = loss_config.n_measured_points
    
    # Переносим на GPU
    coords_gpu = coords |> dev
    values_gpu = values |> dev
    
    # Вычисляем data_loss напрямую
    mse = data_loss(phi_pred_fun, ps_gpu, coords_gpu, values_gpu, n_points)
    
    @info "  data_loss выполнен успешно"
    @info "  MSE: $mse"
    
catch e
    @error "Ошибка при использовании data_loss: $e"
    @error "Стек: $(stacktrace(catch_backtrace()))"
end

# ============================================================
# Тест 3: Используем функцию derivative_loss напрямую
# ============================================================

@info "=== Тест 3: Использование derivative_loss ==="

try
    # Получаем данные из loss_config
    coords = loss_config.measured_points_coords
    values = loss_config.measured_points_values
    n_points = loss_config.n_measured_points
    deriv_meas = loss_config.derivatives
    deriv_coords = loss_config.deriv_coords
    num_sensors = loss_config.num_sensors
    num_time_steps = loss_config.num_time_steps
    measured_time = loss_config.measured_time
    
    @info "  Число производных: $(length(deriv_meas))"
    @info "  Число координат производных: $(size(deriv_coords, 2))"
    
    if length(deriv_meas) > 0 && size(deriv_coords, 2) > 0
        # Переносим на GPU
        coords_gpu = coords |> dev
        values_gpu = values |> dev
        
        # Вычисляем derivative_loss
        deriv_loss_val = derivative_loss(
            phi_pred_fun, ps_gpu,
            coords_gpu, values_gpu, n_points,
            deriv_meas, deriv_coords,
            num_sensors, num_time_steps, measured_time
        )
        
        @info "  derivative_loss выполнен успешно"
        @info "  Derivative loss: $deriv_loss_val"
    else
        @warn "  Нет производных для тестирования - пропускаем"
    end
    
catch e
    @error "Ошибка при использовании derivative_loss: $e"
    @error "Стек: $(stacktrace(catch_backtrace()))"
end

# ============================================================
# Тест 4: Вычисление градиента data_loss через Zygote
# ============================================================

@info "=== Тест 4: Градиент data_loss через Zygote ==="

try
    # Получаем данные из loss_config
    coords = loss_config.measured_points_coords
    values = loss_config.measured_points_values
    n_points = loss_config.n_measured_points
    
    # Переносим на GPU
    coords_gpu = coords |> dev
    values_gpu = values |> dev
    
    # Вычисляем градиент
    grad_result = Zygote.gradient(ps_gpu) do params
        loss = data_loss(phi_pred_fun, params, coords_gpu, values_gpu, n_points)
        return loss
    end
    
    if grad_result[1] !== nothing
        @info "Gradient computation успешен!"
        @info "Тип градиента: $(typeof(grad_result[1]))"
    else
        @warn "Градиент равен nothing"
    end
    
catch e
    @error "Ошибка при вычислении градиента: $e"
    @error "Стек: $(stacktrace(catch_backtrace()))"
end

# ============================================================
# Тест 5: Сравнение с ручным вычислением MSE
# ============================================================

@info "=== Тест 5: Сравнение с ручным MSE ==="

try
    # Получаем данные из loss_config
    coords = loss_config.measured_points_coords
    values = loss_config.measured_points_values
    n_points = loss_config.n_measured_points
    
    @info "  Число координат: $(size(coords))"
    @info "  Число значений: $(length(values))"
    
    # Переносим на GPU
    coords_gpu = coords |> dev
    values_gpu = values |> dev
    
    # Предсказание сети
    pred = phi_pred_fun(coords_gpu, ps_gpu)
    phi_pred = pred[1, :]  # Только phi
    
    # Ручное вычисление MSE
    mse_manual = sum(abs2.(phi_pred .- values_gpu)) / max(n_points, 1)
    
    @info "  Ручное MSE: $mse_manual"
    
    # Сравниваем с результатом из data_loss
    mse_from_func = data_loss(phi_pred_fun, ps_gpu, coords_gpu, values_gpu, n_points)
    
    @info "  MSE из функции: $mse_from_func"
    @info "  Разница: $(abs(mse_manual - mse_from_func))"
    
    # Проверяем, что разница мала
    if abs(mse_manual - mse_from_func) < 1e-6
        @info "  ✓ Тест пройден!"
    else
        @warn "  ✗ Тест не пройден - разница слишком большая"
    end
    
catch e
    @error "Ошибка при сравнении: $e"
    @error "Стек: $(stacktrace(catch_backtrace()))"
end

# ============================================================
# Тест 6: Создаём обёртку аналогичную InverseProblem.jl
# ============================================================

@info "=== Тест 6: Обёртка как в InverseProblem.jl ==="

try
    # Эта обёртка используется в InverseProblem.jl для создания data_loss_raw_func
    loss_cfg = loss_config
    phi = phi_pred_fun
    
    data_loss_raw_func = let cfg = loss_cfg, phi_pred = phi
        function(p_vec)
            dev = Lux.gpu_device()
            
            # Получаем данные из конфига
            coords_gpu = cfg.measured_points_coords |> dev
            values_gpu = cfg.measured_points_values |> dev
            n_points = cfg.n_measured_points
            
            # Переносим параметры на GPU
            p_gpu = p_vec |> dev
            
            # Вычисляем MSE
            mse_val = data_loss(phi_pred, p_gpu, coords_gpu, values_gpu, n_points)
            
            # Вычисляем derivative loss если доступны производные
            deriv_val = Float32(0.0)
            if length(cfg.derivatives) > 0 && size(cfg.deriv_coords, 2) > 0
                deriv_val = derivative_loss(
                    phi_pred, p_gpu, 
                    coords_gpu, values_gpu, n_points,
                    cfg.derivatives, cfg.deriv_coords,
                    cfg.num_sensors, cfg.num_time_steps, cfg.measured_time
                )
            end
            
            return (total=mse_val + deriv_val, mse=mse_val, deriv=deriv_val)
        end
    end
    
    # Вызываем обёртку
    loss_result = data_loss_raw_func(ps_gpu)
    
    @info "  Результат: $(typeof(loss_result))"
    @info "  Поля результата: $(keys(loss_result))"
    @info "  total: $(loss_result.total)"
    @info "  mse: $(loss_result.mse)"
    @info "  deriv: $(loss_result.deriv)"
    
catch e
    @error "Ошибка при тестировании обёртки: $e"
    @error "Стек: $(stacktrace(catch_backtrace()))"
end

@info "=== Все тесты завершены! ==="
