"""
Тест для проверки вычисления градиента от TV regularization loss
с использованием Enzyme на GPU
"""

using CUDA
using Lux
using LuxCUDA  # Для поддержки GPU в MLDataDevices
using Zygote
using Enzyme
using ComponentArrays
using Random

# Устанавливаем seed для воспроизводимости
Random.seed!(1234)

println("="^60)
println("Тест градиента TV regularization loss")
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

println("✓ Тестовые данные созданы")
println("  Time points: $(length(measured_time))")
println("  Domain: [$(inner_lb)] to [$(inner_ub)]")

# Создаём тестовую функцию для TV loss
function test_tv_loss(θ)
    # Создаём точки для интегрирования (упрощённая версия)
    N_test = 10
    x_spatial = CUDA.rand(Float32, 3, N_test)
    
    # Масштабируем в inner_domain
    for i in 1:3
        x_spatial[i, :] .= inner_lb[i] .+ x_spatial[i, :] .* (inner_ub[i] - inner_lb[i])
    end
    
    t0 = Float32(measured_time[1])
    batch_size = size(x_spatial, 2)
    
    # Функция для вычисления rho
    function compute_rho_batch(X_in::AbstractArray)
        t_vec = fill(t0, batch_size) |> dev
        X_4d = CUDA.vcat(X_in, reshape(t_vec, 1, :))
        pred = phi_pred_fun(X_4d, θ)
        return pred[5, :]  # ρ - 5-й компонент
    end
    
    # Вычисляем градиенты по каждой точке отдельно используя Enzyme
    grad_buffer = similar(x_spatial)
    
    for i in 1:batch_size
        x_i = x_spatial[:, i:i]
        
        function rho_single_point(x_local::AbstractArray)
            t_vec = fill(t0, 1) |> dev
            X_4d = CUDA.vcat(x_local, reshape(t_vec, 1, :))
            pred = phi_pred_fun(X_4d, θ)
            return sum(pred[5, :])
        end
        
        # Используем Enzyme.autodiff
        grad_i = Enzyme.autodiff(
            Enzyme.Reverse, 
            rho_single_point, 
            Enzyme.Active, 
            Enzyme.Duplicated(x_i, similar(x_i))
        )[1]
        
        grad_buffer[:, i] = grad_i[:, 1]
    end
    
    # Вычисляем TV
    grad_x = grad_buffer[1, :]
    grad_y = grad_buffer[2, :]
    grad_z = grad_buffer[3, :]
    eps = Float32(tv_epsilon)
    
    tv_per_point = sqrt.(grad_x.^2 .+ grad_y.^2 .+ grad_z.^2 .+ eps)
    tv_integral = mean(tv_per_point) * prod(inner_ub .- inner_lb)
    
    # Нормируем
    volume = prod(Float32.(inner_ub) .- Float32.(inner_lb))
    tv_normalized = tv_integral / volume
    
    # Экспоненциальный loss
    scale_tv = Float32(tv_scale)
    L_tv = exp(-tv_normalized / scale_tv)
    
    return L_tv
end

println("\n🧪 Тест 1: Прямое вычисление TV loss...")
try
    L_tv = test_tv_loss(ps)
    println("✓ TV loss вычислен: L_tv = $(Float32(L_tv))")
catch e
    println("❌ Ошибка при вычислении TV loss:")
    println(e)
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
