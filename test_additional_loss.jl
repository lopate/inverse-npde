# Тест для проверки create_additional_loss с GPU и Zygote градиентами
# Запуск: julia --project test_additional_loss.jl

using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using Lux, Zygote, Statistics, Random, CUDA

# Подключаем модуль Optimization
push!(LOAD_PATH, joinpath(@__DIR__, "src"))
using InverseNpde.Optimization

# Проверяем доступность GPU
if CUDA.functional()
    @info "GPU доступен, тестируем на GPU"
    using LuxCUDA
else
    @warn "GPU недоступен, тест будет пропущен"
    exit(0)
end

# ============================================================
# Создаём тестовые данные и конфигурацию (упрощённая версия)
# ============================================================

# Параметры домена
x_range = [-10.0, 10.0]
y_range = [-10.0, 10.0]
z_range = [-10.0, 10.0]
t_range = [0.0, 1.0]
pml_thickness_ratio = 0.1

# Вычисляем внутреннюю область (домен минус PML)
x_min, x_max = x_range[1], x_range[2]
y_min, y_max = y_range[1], y_range[2]
z_min, z_max = z_range[1], z_range[2]

Dx = (x_max - x_min) * pml_thickness_ratio
Dy = (y_max - y_min) * pml_thickness_ratio
Dz = (z_max - z_min) * pml_thickness_ratio

inner_lb = [x_min + Dx, y_min + Dy, z_min + Dz]
inner_ub = [x_max - Dx, y_max - Dy, z_max - Dz]

@info "Внутренняя область: [$inner_lb] x [$inner_ub]"

# ============================================================
# mc_integral_gpu функция (из Optimization.jl)
# ============================================================

function mc_integral_gpu(f_batch, lb::Vector, ub::Vector, N::Int)
    d = length(lb)
    dev = Lux.gpu_device()
    
    # Генерируем случайные точки в [0,1]^d на GPU
    X = Zygote.ignore_derivatives() do
        CUDA.rand(Float32, d, N)
    end
    
    # Масштабируем в домен [lb, ub]
    scale_vec = (ub .- lb) |> dev
    offset_vec = lb |> dev
    X_scaled = offset_vec .+ X .* scale_vec
    
    # Применяем функцию-батч ко всем точкам сразу
    vals = f_batch(X_scaled)
    
    # Вычисляем среднее и умножаем на объём домена
    volume = prod(ub .- lb)
    return mean(vals) * volume
end

# ============================================================
# Создаём нейросеть
# ============================================================

# MLP: 4 входа (x, y, z, t) -> 8 выходов (phi, Ax, Ay, Az, rho, jx, jy, jz)
chain = Lux.Chain(
    Dense(4, 32, Lux.σ),
    Dense(32, 32, Lux.σ),
    Dense(32, 8)  # 8 выходов: phi, Ax, Ay, Az, rho, jx, jy, jz
)

# Инициализируем параметры
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
# Тест 1: Проверка mc_integral_gpu с простыми функциями
# ============================================================

@info "=== Тест 1: Проверка mc_integral_gpu ==="

try
    # Тест 1a: Интеграл константы 1
    test_f_batch(x) = ones(Float32, size(x, 2))
    volume = prod(inner_ub .- inner_lb)
    @info "  Ожидаемый объём: $volume"
    
    integral_1 = mc_integral_gpu(test_f_batch, inner_lb, inner_ub, 10000)
    @info "  Интеграл константы 1: $integral_1 (ошибка: $(abs(integral_1 - volume) / volume))"
    
    # Тест 1b: Интеграл x^2 + y^2 + z^2 (примерно)
    test_f_batch_sq(x) = vec(sum(x.^2, dims=1))
    integral_sq = mc_integral_gpu(test_f_batch_sq, inner_lb, inner_ub, 10000)
    @info "  Интеграл x^2+y^2+z^2: $integral_sq"
    
catch e
    @error "Ошибка в тесте mc_integral_gpu: $e"
    @error "Стек: $(stacktrace(catch_backtrace()))"
end

# ============================================================
# Тест 2: Проверка compute_field_energy_loss с нормировкой на объём
# ============================================================

@info "=== Тест 2: Проверка compute_field_energy_loss ==="

try
    # НОВАЯ СИГНАТУРА: measured_time + num_field_time_samples
    measured_time = Float32[0.0f0, 0.5f0, 1.0f0]  # Вектор измеренных временных точек
    num_field_time_samples = 3  # Количество точек для семплирования
    field_energy_scale = 1.0f0
    N_mc = 1000
    
    # Используем новую функцию compute_field_energy_loss
    # Функция сама создаёт time_points на основе measured_time
    field_result = compute_field_energy_loss(
        phi_pred_fun,
        ps_gpu,
        measured_time,    # Вместо time_points - вектор измеренных времён
        inner_lb,
        inner_ub,
        field_energy_scale,
        num_field_time_samples,  # НОВЫЙ ПАРАМЕТР
        N_mc
    )
    
    volume = prod(inner_ub .- inner_lb)
    
    @info "  Объём внутренней области: $volume"
    @info "  E_field (интеграл): $(field_result.E_field)"
    @info "  E_field_normalized (плотность энергии): $(field_result.E_field_normalized)"
    @info "  L_field (экспоненциальный лосс): $(field_result.L_field)"
    
    # Проверяем, что E_field_normalized = E_field / volume
    expected_normalized = field_result.E_field / volume
    @info "  Ожидаемая нормировка: $expected_normalized"
    @info "  Проверка нормировки: $(abs(field_result.E_field_normalized - expected_normalized) < 1e-6)"
    
catch e
    @error "Ошибка при расчёте compute_field_energy_loss: $e"
    @error "Стек: $(stacktrace(catch_backtrace()))"
end

# ============================================================
# Тест 3: Вычисление градиента field_energy с Zygote через общую функцию
# ============================================================

@info "=== Тест 3: Градиент field_energy с Zygote ==="

try
    # НОВАЯ СИГНАТУРА
    measured_time = Float32[0.0f0, 0.5f0, 1.0f0]  # Float32 вектор
    num_field_time_samples = 3
    field_energy_scale = 3.0f0
    volume = prod(inner_ub .- inner_lb)
    
    # Используем общую функцию compute_field_energy_loss
    grad_result = Zygote.gradient(ps_gpu) do params
        field_res = compute_field_energy_loss(
            phi_pred_fun,
            params,
            measured_time,  # Вместо time_points
            inner_lb,
            inner_ub,
            field_energy_scale,
            num_field_time_samples,  # НОВЫЙ ПАРАМЕТР
            1000  # N_mc
        )
        # Используем L_field с нормированной энергией
        return field_res.L_field
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
# Тест 4: Проверка простого forward pass сети
# ============================================================

@info "=== Тест 4: Forward pass сети ==="

try
    # Создаём тестовые данные с 4 входами (x, y, z, t)
    n_test_points = 100
    test_x = Float32.(rand(4, n_test_points) .* (vcat(inner_ub, [1.0]) .- vcat(inner_lb, [0.0])) .+ vcat(inner_lb, [0.0])) |> dev
    
    # Оцениваем сеть
    energies = phi_pred_fun(test_x, ps_gpu)
    
    @info "  Размер предсказания: $(size(energies))"
    @info "  Первые 5 значений phi: $(energies[1, 1:5])"
    @info "  Первые 5 значений rho: $(energies[5, 1:5])"
    
catch e
    @error "Ошибка при проверке forward pass: $e"
    @error "Стек: $(stacktrace(catch_backtrace()))"
end

# ============================================================
# Тест 5: Градиент MSE loss
# ============================================================

@info "=== Тест 5: Градиент MSE loss ==="

try
    # Создаём простые тестовые данные с 4 входами (x, y, z, t)
    n_points = 10
    coords = Float32.(rand(4, n_points))  # [x, y, z, t]
    values = Float32.(rand(n_points))     # Целевые значения phi
    
    coords_gpu = coords |> dev
    values_gpu = values |> dev
    
    # MSE loss функция
    function mse_loss(params)
        pred = phi_pred_fun(coords_gpu, params)
        phi_pred = pred[1, :]
        return sum(abs2.(phi_pred .- values_gpu)) / n_points
    end
    
    # Forward pass
    loss = mse_loss(ps_gpu)
    @info "  MSE loss: $loss"
    
    # Gradient
    grad = Zygote.gradient(ps_gpu) do params
        mse_loss(params)
    end
    
    if grad[1] !== nothing
        @info "  Gradient computation успешен!"
    else
        @warn "  Градиент равен nothing"
    end
    
catch e
    @error "Ошибка при проверке MSE gradient: $e"
    @error "Стек: $(stacktrace(catch_backtrace()))"
end

@info "=== Все тесты завершены! ==="
