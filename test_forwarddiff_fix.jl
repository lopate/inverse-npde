#!/usr/bin/env julia

"""
Тестовый скрипт для проверки исправления ошибки `llvmcall requires the compiler`

Этот скрипт проверяет, что ForwardDiff.gradient корректно работает
для вычисления градиентов в TV регуляризации на GPU.
"""

using Pkg
Pkg.activate(".")

using CUDA
using Lux
using ComponentArrays
using ForwardDiff

println("=" ^ 60)
println("Тест исправления ошибки llvmcall для TV регуляризации")
println("=" ^ 60)

# Проверяем доступность GPU
if !CUDA.functional()
    println("⚠️  CUDA не доступна, тест будет пропущен")
    exit(0)
end

println("✅ CUDA доступна")
println("   Устройство: $(CUDA.device())")

# Создаём простую нейросеть для теста
dev = Lux.gpu_device()
cpu_dev = Lux.cpu_device()

# Простая сеть: вход [3] -> выход [5]
chain = Lux.Chain(
    Lux.Dense(3 => 8, tanh),
    Lux.Dense(8 => 5)
)

# Инициализируем параметры
ps, st = Lux.setup(Xoshiro(0), chain)
ps_gpu = ps |> dev
st_gpu = st |> dev

println("✅ Сеть создана и перенесена на GPU")

# Тестовая функция для ForwardDiff
function test_forwarddiff_gradient()
    # Создаём тестовые входные данные на CPU
    x_cpu = randn(Float32, 3, 10)  # [3, batch_size]
    
    # Функция для вычисления суммы выходов сети
    function compute_sum(x_in)
        # Переносим на GPU для вычисления
        x_gpu = x_in |> dev
        pred = chain(x_gpu, ps_gpu, st_gpu)[1]  # [5, batch_size]
        # Возвращаем сумму 5-го компонента (аналог rho)
        return sum(pred[5, :])
    end
    
    println("\n1. Тест ForwardDiff.gradient на CPU данных...")
    try
        # Вычисляем градиент через ForwardDiff
        grad = ForwardDiff.gradient(compute_sum, x_cpu)
        println("   ✅ Градиент вычислен успешно!")
        println("   Размер градиента: $(size(grad))")
        println("   Тип градиента: $(typeof(grad))")
        return true
    catch e
        println("   ❌ Ошибка при вычислении градиента:")
        println("   $e")
        return false
    end
end

# Тест с переносом на GPU (как в реальном коде)
function test_gpu_workflow()
    println("\n2. Тест workflow CPU -> GPU -> CPU (как в Optimization.jl)...")
    
    # Данные на GPU
    x_gpu = CUDA.randn(Float32, 3, 10)
    
    # Переносим на CPU для ForwardDiff
    x_cpu = x_gpu |> cpu_dev
    
    function compute_rho_sum(x_in)
        x_gpu_in = x_in |> dev
        pred = chain(x_gpu_in, ps_gpu, st_gpu)[1]
        return sum(pred[5, :])
    end
    
    try
        # Вычисляем градиент на CPU
        grad_cpu = ForwardDiff.gradient(compute_rho_sum, x_cpu)
        # Переносим результат на GPU
        grad_gpu = grad_cpu |> dev
        
        println("   ✅ Workflow выполнен успешно!")
        println("   Размер grad_cpu: $(size(grad_cpu))")
        println("   Тип grad_gpu: $(typeof(grad_gpu))")
        return true
    catch e
        println("   ❌ Ошибка:")
        println("   $e")
        return false
    end
end

# Основной тест
println("\n" * "=" ^ 60)
println("Запуск тестов...")
println("=" ^ 60)

test1_passed = test_forwarddiff_gradient()
test2_passed = test_gpu_workflow()

println("\n" * "=" ^ 60)
println("Результаты тестов:")
println("=" ^ 60)
println("Тест 1 (ForwardDiff.gradient): $(test1_passed ? "✅ ПРОЙДЕН" : "❌ ПРОВАЛЕН")")
println("Тест 2 (CPU->GPU workflow):   $(test2_passed ? "✅ ПРОЙДЕН" : "❌ ПРОВАЛЕН")")

if test1_passed && test2_passed
    println("\n🎉 Все тесты пройдены! Исправление работает корректно.")
    println("   Теперь TV регуляризация должна работать без ошибки llvmcall.")
    exit(0)
else
    println("\n⚠️  Некоторые тесты не пройдены. Требуется дополнительная диагностика.")
    exit(1)
end
