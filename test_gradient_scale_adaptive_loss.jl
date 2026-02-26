# Тест GradientScaleAdaptiveLoss сигнатуры
using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using NeuralPDE: GradientScaleAdaptiveLoss, NonAdaptiveLoss
using Logging

@info "Testing GradientScaleAdaptiveLoss signature..."

# Тест 1: Проверка создания GradientScaleAdaptiveLoss с правильной сигнатурой
try
    adaptive_loss = GradientScaleAdaptiveLoss(
        10;  # reweight_every как позиционный аргумент
        weight_change_inertia = 0.9,
        pde_loss_weights = 1.0,
        bc_loss_weights = 1.0,
        additional_loss_weights = 1.0
    )
    @info "Test 1 PASSED: GradientScaleAdaptiveLoss created successfully"
    @info "adaptive_loss type: $(typeof(adaptive_loss))"
catch e
    @error "Test 1 FAILED: $e"
end

# Тест 2: NonAdaptiveLoss для сравнения
try
    non_adaptive = NonAdaptiveLoss(1.0, 1.0, 1.0)
    @info "Test 2 PASSED: NonAdaptiveLoss created successfully"
    @info "non_adaptive type: $(typeof(non_adaptive))"
catch e
    @error "Test 2 FAILED: $e"
end

@info "All tests completed!"
