"""
Простой тест для проверки, что Zygote может вычислять градиенты
через сеть Lux на GPU (аналогично тому, что нужно для TV loss)
"""

using CUDA
using Lux
using LuxCUDA
using Zygote
using ComponentArrays
using Random
using Statistics

Random.seed!(1234)

println("="^60)
println("Простой тест градиента через Zygote + Lux на GPU")
println("="^60)

if !CUDA.functional()
    println("❌ CUDA не доступен")
    exit(0)
end

println("✓ CUDA доступен: $(CUDA.name(CUDA.device()))")

dev = Lux.gpu_device()

# Простая сеть
chain = Lux.Chain(
    Lux.Dense(4 => 16, Lux.softplus),
    Lux.Dense(16 => 8, identity)
)

ps, st = Lux.setup(Random.default_rng(), chain)
ps = ps |> dev
st = st |> dev

println("✓ Сеть создана")

# Функция предсказания
predict(x, θ) = chain(x, θ, st)[1]

# Простая функция потерь: сумма квадратов выходов
function simple_loss(θ)
    # Фиксированные входные данные (не случайные!)
    x_input = CUDA.ones(Float32, 4, 5)  # 4 features, 5 samples
    pred = predict(x_input, θ)
    return sum(pred .^ 2)  # Простая сумма квадратов
end

println("\n🧪 Тест 1: Прямое вычисление loss...")
L = simple_loss(ps)
println("✓ Loss = $L")

println("\n🧪 Тест 2: Вычисление градиента...")
try
    grad_ps = Zygote.gradient(simple_loss, ps)[1]
    
    if grad_ps !== nothing
        total_norm = sqrt(sum(sum(abs2, g) for g in values(grad_ps)))
        println("✓ Градиент вычислен!")
        println("  Норма градиента: $total_norm")
        println("  Не-NaN: $(!isnan(total_norm))")
        println("  Не-Inf: $(!isinf(total_norm))")
        
        if !isnan(total_norm) && !isinf(total_norm) && total_norm > 0
            println("\n🎉 УСПЕХ! Градиент работает корректно!")
        else
            println("\n❌ Градиент некорректен")
            exit(1)
        end
    else
        println("❌ Градиент = nothing")
        exit(1)
    end
catch e
    println("❌ Ошибка: $e")
    println("\nСтектрейс:")
    Base.show_backtrace(stdout, catch_backtrace())
    exit(1)
end

println("\n" * "="^60)
println("Тест пройден!")
println("="^60)
