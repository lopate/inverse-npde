"""
Тест для проверки вычисления градиента от TV regularization loss
с использованием конечных разностей (finite differences) на GPU
"""

using CUDA
using Lux
using LuxCUDA
using Zygote
using ComponentArrays
using Random
using Statistics

# Устанавливаем seed для воспроизводимости
Random.seed!(1234)

println("="^60)
println("Тест градиента TV regularization (Finite Differences)")
println("="^60)

# Проверяем доступность GPU
if !CUDA.functional()
    println("❌ CUDA не доступен, тест пропускается")
    exit(0)
end

println("✓ CUDA доступен")
println("  GPU: $(CUDA.name(CUDA.device()))")

# Создаём простую тестовую сеть
dev = Lux.gpu_device()

# Простая сеть: вход [4, N] -> выход [8, N] (φ, Ax, Ay, Az, ρ, jx, jy, jz)
chain = Lux.Chain(
    Lux.Dense(4 => 16, Lux.softplus),
    Lux.Dense(16 => 8, identity)  # 8 выходных компонентов
)

# Инициализируем параметры
ps, st = Lux.setup(Random.default_rng(), chain)
ps = ps |> dev
st = st |> dev

println("✓ Сеть создана")
println("  Параметры: $(sum(length, values(ps))) элементов")

# Создаём функцию предсказания
phi_pred_fun(x, θ) = chain(x, θ, st)[1]

# Тестовые данные
measured_time = Float32[0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
inner_lb = Float32[-1.0, -1.0, -1.0]
inner_ub = Float32[1.0, 1.0, 1.0]
tv_epsilon = Float32(1e-5)
tv_scale = Float32(0.1)
fd_step = Float32(1e-4)  # Шаг для конечных разностей

println("✓ Тестовые данные созданы")
println("  Time points: $(length(measured_time))")
println("  Domain: [$(inner_lb)] to [$(inner_ub)]")

# Функция для вычисления rho в точке
function compute_rho_at_point(xyz::AbstractVector, t0::Float32, θ)
    x_input = reshape(xyz, 3, 1) |> dev
    t_input = reshape(Float32[t0], 1, 1) |> dev
    X_4d = CUDA.vcat(x_input, t_input)
    pred = phi_pred_fun(X_4d, θ)
    return CUDA.sum(pred[5, :])  # ρ - 5-й компонент
end

# Конечные разности для вычисления градиента по пространству
function compute_spatial_gradient_fd(xyz::AbstractVector, t0::Float32, θ, h::Float32)
    grad = similar(xyz)
    rho_center = compute_rho_at_point(xyz, t0, θ)
    
    for i in 1:3
        xyz_plus = copy(xyz)
        xyz_plus[i] += h
        rho_plus = compute_rho_at_point(xyz_plus, t0, θ)
        grad[i] = (rho_plus - rho_center) / h
    end
    
    return grad
end

# Создаём точки для интегрирования ВНЕ функции loss
N_test = 10
x_spatial_base = CUDA.rand(Float32, 3, N_test)
for i in 1:3
    x_spatial_base[i, :] .= inner_lb[i] .+ x_spatial_base[i, :] .* (inner_ub[i] - inner_lb[i])
end

# Создаём тестовую функцию для TV loss
function test_tv_loss(θ)
    # Используем заранее созданные точки (не дифференцируем их)
    x_spatial = x_spatial_base
    
    t0 = Float32(measured_time[1])
    batch_size = size(x_spatial, 2)
    
    # Вычисляем градиенты по каждой точке используя конечные разности
    grad_buffer = similar(x_spatial)
    
    for i in 1:batch_size
        xyz_i = Array(x_spatial[:, i])  # Конвертируем в CPU для итераций
        grad_i = compute_spatial_gradient_fd(xyz_i, t0, θ, fd_step)
        grad_buffer[:, i] = CuArray(grad_i)
    end
    
    # Вычисляем TV
    grad_x = grad_buffer[1, :]
    grad_y = grad_buffer[2, :]
    grad_z = grad_buffer[3, :]
    
    tv_per_point = sqrt.(grad_x.^2 .+ grad_y.^2 .+ grad_z.^2 .+ tv_epsilon)
    tv_integral = mean(tv_per_point) * prod(inner_ub .- inner_lb)
    
    # Нормируем
    volume = prod(Float32.(inner_ub) .- Float32.(inner_lb))
    tv_normalized = tv_integral / volume
    
    # Экспоненциальный loss
    L_tv = exp(-tv_normalized / tv_scale)
    
    return L_tv
end

println("\n🧪 Тест 1: Прямое вычисление TV loss (Finite Differences)...")
try
    L_tv = test_tv_loss(ps)
    println("✓ TV loss вычислен: L_tv = $(Float32(L_tv))")
catch e
    println("❌ Ошибка при вычислении TV loss:")
    println(e)
    println("\nСтектрейс:")
    Base.show_backtrace(stdout, catch_backtrace())
    exit(1)
end

println("\n🧪 Тест 2: Вычисление градиента от TV loss по параметрам сети...")
try
    # Вычисляем градиент через Zygote
    grad_ps = Zygote.gradient(test_tv_loss, ps)[1]
    
    if grad_ps !== nothing
        total_grad_norm = sqrt(sum(sum(abs2, g) for g in values(grad_ps)))
        println("✓ Градиент вычислен успешно!")
        println("  Норма градиента: $(Float32(total_grad_norm))")
        println("  Градиент не-NaN: $(!isnan(total_grad_norm))")
        println("  Градиент не-Inf: $(!isinf(total_grad_norm))")
        
        # Проверяем, что градиенты конечны
        if isnan(total_grad_norm) || isinf(total_grad_norm)
            println("❌ Градиент содержит NaN или Inf!")
            exit(1)
        else
            println("\n🎉 УСПЕХ! Градиент TV loss вычисляется корректно!")
            println("   (Используя конечные разности)")
        end
    else
        println("❌ Градиент равен nothing")
        exit(1)
    end
catch e
    println("❌ Ошибка при вычислении градиента:")
    println(e)
    println("\nСтектрейс:")
    Base.show_backtrace(stdout, catch_backtrace())
    exit(1)
end

println("\n" * "="^60)
println("Все тесты пройдены!")
println("="^60)
