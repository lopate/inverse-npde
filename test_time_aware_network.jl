#!/usr/bin/env julia

"""
Тестовый скрипт для проверки работоспособности TimeAwareNetwork.

Цель: Проверить, что сеть сходится к функции g = x + y + z + sin(πt) на кубе [0, 1]^4
с использованием MSE потерь, L1 регуляризации и оптимизатора Adam.
"""

# Активируем проект и устанавливаем зависимости
using Pkg
Pkg.activate(".")

using Random, Statistics, LinearAlgebra
using Lux, ComponentArrays
using Optimization, OptimizationOptimisers
using Zygote
using Plots

# Импортируем необходимые модули из проекта
include("src/InverseNpde.jl")
using .InverseNpde.NeuralNetwork: TemporalAwareNetworkConfig, create_temporal_aware_network, initialize_temporal_aware_parameters


# Конфигурация сети
config = TemporalAwareNetworkConfig(
    spatial_hidden_layers=[32, 32],
    spatial_output_dim=16,
    num_fourier_frequencies=8,
    temporal_hidden_layers=[16],
    temporal_output_dim=8,
    fusion_hidden_layers=[16],
    output_dim=1,  # Предсказываем только одно значение (φ)
    activation=:tanh,
    use_gpu=false  # Для простоты тестирования используем CPU
)

# Создаем сеть
println("Создание TemporalAwareNetwork...")
network = create_temporal_aware_network(config)
params = initialize_temporal_aware_parameters(network, Random.default_rng(); use_gpu=false)

# Целевая функция: g = x + y + z + sin(πt)
target_function(x, y, z, t) = x + y + z + sin(π * t)

# Генерация тестовых данных на кубе [0, 1]^4
n_samples = 1000
Random.seed!(42)
X = rand(Float32, 4, n_samples)  # [x, y, z, t]
Y = [target_function(X[1,i], X[2,i], X[3,i], X[4,i]) for i in 1:n_samples]
Y = Float32.(Y)

# Функция потерь MSE + L1 регуляризация
function loss_function(params, X, Y, st)
    # Предсказание сети
    pred, _ = network(X, params, st)
    pred = pred[1, :]  # Извлекаем φ (первый выход)
    
    # MSE loss
    mse_loss = mean((pred .- Y).^2)
    
    # L1 регуляризация (на веса сети)
    l1_reg = 0.001 * sum(abs, params)
    
    # Общий loss
    total_loss = mse_loss + l1_reg
    
    return total_loss, (mse=mse_loss, l1=l1_reg)
end

# Тест прямого прохода
println("Тестирование прямого прохода...")
# Инициализируем состояние слоя
st = Lux.setup(Random.default_rng(), network)[2]
test_pred, _ = network(X, params, st)
test_pred = test_pred[1, :]  # Извлекаем φ (первый выход)
@show size(test_pred)
@show mean((test_pred .- Y).^2)

# Обучение сети
println("\nОбучение сети...")

# Инициализируем состояние слоя
st = Lux.setup(Random.default_rng(), network)[2]

# Создаём OptimizationFunction для использования с Optimization.solve
loss_func = OptimizationFunction((p, _) -> loss_function(p, X, Y, st)[1], Optimization.AutoZygote())

# Создаём проблему оптимизации
prob = OptimizationProblem(loss_func, params)

# Создаём оптимизатор Adam
opt = OptimizationOptimisers.Adam(0.001)

println("Начинаем обучение...")
# Трекинг потерь
loss_history = Float64[]
trained_params = params  # Значение по умолчанию - исходные параметры
try
    # Callback для трекинга потерь
    callback = function (p, l)
        push!(loss_history, l)
        if length(loss_history) % 100 == 0
            println("Iteration $(length(loss_history)): Loss = $l")
        end
        return false
    end
    
    result = Optimization.solve(prob, opt, maxiters=1000, callback=callback)
    
    global trained_params = result.u
    println("Обучение завершено!")
    final_loss = length(loss_history) > 0 ? loss_history[end] : 0.0
    println("Final Loss: $final_loss")
catch e
    println("Ошибка при обучении: ")
    showerror(stdout, e, backtrace())
    error("Обучение не удалось")
end


# Визуализация потерь
epochs = 1:length(loss_history)
plot(epochs, loss_history, label="Total Loss", xlabel="Iteration", ylabel="Loss", title="Training Progress")
savefig("training_loss.png")

# Проверка сходимости - создаём новое состояние сети
st_final = Lux.setup(Random.default_rng(), network)[2]
final_pred, _ = network(X, trained_params, st_final)
final_pred = final_pred[1, :]
final_mse = mean((final_pred .- Y).^2)
println("\nFinal MSE: $final_mse")

# Сравнение предсказаний с целевой функцией
scatter(Y, final_pred, xlabel="True Values", ylabel="Predicted Values", 
        label="Predictions", title="True vs Predicted Values")
plot!(Y, Y, label="Perfect Prediction", linestyle=:dash)
savefig("predictions_comparison.png")

println("\nТестирование завершено. Результаты сохранены в training_loss.png и predictions_comparison.png")
println("Final MSE: $final_mse")

if final_mse < 0.01
    println("✓ Сеть успешно сошлась к целевой функции!")
else
    println("✗ Сходимость не достигнута. Попробуйте увеличить количество эпох или изменить гиперпараметры.")
end