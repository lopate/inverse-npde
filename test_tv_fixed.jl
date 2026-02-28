#!/usr/bin/env julia
# Тест для проверки исправленного TV loss

using Pkg
Pkg.activate(".")

using Lux, LuxCUDA, Zygote, CUDA, Random, ComponentArrays, LinearAlgebra

# Добавляем src в путь
push!(LOAD_PATH, joinpath(@__DIR__, "src"))

using InverseNpde

println("=" ^ 60)
println("Тестирование исправленного TV loss")
println("=" ^ 60)

# Создаём простую сеть
chain = Lux.Chain(
    Lux.Dense(4 => 16, softplus),
    Lux.Dense(16 => 8, identity)
)

rng = Random.MersenneTwister(42)
ps, st = Lux.setup(rng, chain)
ps = ps |> ComponentArray |> Lux.gpu_device()
st = st |> Lux.gpu_device()

# Функция предсказания (совместимая с Zygote)
function phi_pred(x, θ)
    return Lux.apply(chain, x, θ, st)[1]
end

# Тестовые параметры
measured_time = Float32.(0.0:0.01:0.1)
inner_lb = Float32.([-1.0, -1.0, -1.0])
inner_ub = Float32.([1.0, 1.0, 1.0])
tv_epsilon = 1.0f-5
tv_scale = 0.1f0

println("\n1. Проверка прямого вычисления TV loss...")
try
    tv_result = InverseNpde.Optimization.compute_tv_regularization(
        phi_pred, ps, measured_time, inner_lb, inner_ub,
        tv_epsilon, tv_scale, 5, 100  # num_samples=5, N_mc=100 для скорости
    )
    println("   ✓ TV loss вычислен успешно")
    println("   TV = $(tv_result.TV)")
    println("   TV_normalized = $(tv_result.TV_normalized)")
    println("   L_tv = $(tv_result.L_tv)")
catch e
    println("   ✗ Ошибка: $e")
    println(sprint(showerror, e, catch_backtrace()))
end

println("\n2. Проверка градиентов через Zygote...")
try
    # Функция потерь для градиента
    function loss_fn(θ)
        tv_result = InverseNpde.Optimization.compute_tv_regularization(
            phi_pred, θ, measured_time, inner_lb, inner_ub,
            tv_epsilon, tv_scale, 5, 100
        )
        return tv_result.L_tv
    end
    
    # Вычисляем градиент
    grad = Zygote.gradient(loss_fn, ps)[1]
    
    println("   ✓ Градиент вычислен успешно")
    println("   Размер градиента: $(length(grad))")
    println("   Норма градиента: $(norm(grad))")
    
    # Проверяем, что градиент не NaN и не Inf
    if any(isnan.(grad)) || any(isinf.(grad))
        println("   ⚠ Предупреждение: градиент содержит NaN или Inf")
    else
        println("   ✓ Градиент корректен (без NaN/Inf)")
    end
catch e
    println("   ✗ Ошибка: $e")
    println(sprint(showerror, e, catch_backtrace()))
end

println("\n3. Проверка в составе полной функции потерь...")
try
    # Создаём конфигурацию loss
    measured_points = [
        Float32[0.0, 0.0, 0.0, 0.0, 1.0],
        Float32[0.0, 0.0, 0.0, 0.05, 1.1],
        Float32[0.0, 0.0, 0.0, 0.1, 1.2],
    ]
    
    loss_config = InverseNpde.Optimization.LossFunctionConfig(
        measured_points=measured_points,
        lambda_tv=0.1f0,
        tv_epsilon=1.0f-5,
        tv_scale=0.1f0,
        num_tv_time_samples=3,
        x_range=Float32.([-2.0f0, 2.0f0]),
        y_range=Float32.([-2.0f0, 2.0f0]),
        z_range=Float32.([-2.0f0, 2.0f0]),
        pml_thickness_ratio=0.1f0,
        lambda_schedule=Dict{String,Any}()
    )
    
    lambda_ref = Ref{Float32}(1.0f0)
    additional_loss = InverseNpde.Optimization.create_additional_loss(loss_config, lambda_ref)
    
    # Вычисляем полную функцию потерь
    total_loss = additional_loss(phi_pred, ps, nothing)
    println("   ✓ Полная функция потерь вычислена успешно")
    println("   Total loss = $total_loss")
    
    # Вычисляем градиент полной функции потерь
    grad_total = Zygote.gradient(θ -> additional_loss(phi_pred, θ, nothing), ps)[1]
    println("   ✓ Градиент полной функции потерь вычислен успешно")
    println("   Норма градиента: $(norm(grad_total))")
    
catch e
    println("   ✗ Ошибка: $e")
    println(sprint(showerror, e, catch_backtrace()))
end

println("\n" * "=" ^ 60)
println("Тестирование завершено!")
println("=" ^ 60)
