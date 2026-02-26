# Тест для проверки mc_integral с обычными массивами и Zygote градиентами
# Запуск: julia --project test_mc_integral_gpu.jl

using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using Lux, Zygote, Statistics, Random

# Создаём простую MLP сеть для тестирования
chain = Lux.Chain(
    Dense(4, 32, Lux.σ),
    Dense(32, 32, Lux.σ),
    Dense(32, 8)  # 8 выходов: phi, Ax, Ay, Az, rho, jx, jy, jz
)

# Инициализируем параметры
ps, st = Lux.setup(Random.default_rng(), chain)

# Функция для предсказания сети (удобная обёртка)
function predict(x::AbstractMatrix{Float32}, params)
    pred, _ = chain(x, params, st)
    return pred
end

# Тестируемая функция mc_integral (CPU версия)
function mc_integral(f_batch, lb::Vector, ub::Vector, N::Int)
    d = length(lb)
    
    # Генерируем случайные точки в [0,1]^d на CPU
    # Используем ignore_derivatives() для исключения из AD графа Zygote
    X = Zygote.ignore_derivatives() do
        rand(Float32, d, N)
    end
    
    # Масштабируем в домен [lb, ub]
    scale_vec = ub .- lb
    offset_vec = lb
    X_scaled = offset_vec .+ X .* scale_vec
    
    # Применяем функцию-батч ко всем точкам сразу
    vals = f_batch(X_scaled)
    
    # Вычисляем среднее и умножаем на объём домена
    volume = prod(ub .- lb)
    return mean(vals) * volume
end

# Создаём тестовую функцию field_energy_mc_batch
# Эта функция должна работать с массивами Float32
time_points = range(0.0f0, 1.0f0, length=5)

function field_energy_mc_batch(x_spatial::AbstractArray{Float32}, θ)
    batch_size = size(x_spatial, 2)
    n_times = length(time_points)
    
    # Создаем батч: для каждой пространственной точки - все временные срезы
    x_full = zeros(Float32, 4, batch_size * n_times)
    
    for (ti, t_val) in enumerate(time_points)
        start_idx = (ti - 1) * batch_size + 1
        end_idx = ti * batch_size
        x_full[1:3, start_idx:end_idx] = x_spatial
        x_full[4, start_idx:end_idx] .= t_val
    end
    
    # Получаем предсказание сети для всего батча
    pred = predict(x_full, θ)
    
    # Решейпим для усреднения по времени
    # pred[1,:] = phi, pred[2:4] = A, pred[5] = rho, pred[6:8] = j
    phi = reshape(pred[1, :], n_times, batch_size)
    Ax = reshape(pred[2, :], n_times, batch_size)
    Ay = reshape(pred[3, :], n_times, batch_size)
    Az = reshape(pred[4, :], n_times, batch_size)
    rho = reshape(pred[5, :], n_times, batch_size)
    jx = reshape(pred[6, :], n_times, batch_size)
    jy = reshape(pred[7, :], n_times, batch_size)
    jz = reshape(pred[8, :], n_times, batch_size)
    
    # Энергия: rho * phi + A . j
    energy_per_time = rho .* phi .+ Ax.*jx .+ Ay.*jy .+ Az.*jz
    
    # Усредняем по времени для каждой пространственной точки
    energy_mean = mean(energy_per_time, dims=1)
    
    # Возвращаем вектор усредненных энергий [batch_size]
    return vec(energy_mean)
end

# Обертка для включения θ в замыкание
field_energy_mc = let f = field_energy_mc_batch
    (x, θ) -> f(x, θ)
end

# Параметры домена
inner_lb = Float64[-5.0, -5.0, -5.0]
inner_ub = Float64[5.0, 5.0, 5.0]
N_mc = 1000

@info "Тестирование mc_integral..."

# Тест 1: Вычисление интеграла (forward pass)
try
    E_field = mc_integral(field_energy_mc, inner_lb, inner_ub, N_mc)
    @info "Forward pass успешен! E_field = $E_field"
catch e
    @error "Ошибка в forward pass: $e"
    @error "Стек: $(stacktrace(catch_backtrace()))"
end

# Тест 2: Вычисление градиента с помощью Zygote
@info "Тестирование градиента с Zygote..."
try
    grad_result = Zygote.gradient(ps) do params
        E_field = mc_integral(field_energy_mc, inner_lb, inner_ub, N_mc)
        return E_field
    end
    @info "Gradient computation успешен!"
    @info "Тип градиента: $(typeof(grad_result))"
    
    # Check that gradient is not nothing
    if grad_result[1] !== nothing
        @info "Градиент содержит данные"
    end
catch e
    @error "Ошибка при вычислении градиента: $e"
    @error "Стек: $(stacktrace(catch_backtrace()))"
end

@info "Тесты завершены!"
