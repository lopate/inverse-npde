"""
    InverseNpde

Главный модуль для решения обратной задачи ЭЭГ с использованием NeuralPDE.
Объединяет все компоненты: PDE определения, нейронные сети и оптимизацию.

Пример использования:
```julia
using .InverseNpde

# Создаем конфигурации
nn_config = NeuralNetworkConfig(; input_dim=4, hidden_layers=[32, 32], output_dim=8)
opt_config = OptimizationConfig(; learning_rate=0.001, max_iterations=3000)
loss_config = LossFunctionConfig(; lambda_data=10.0)

# Запускаем эксперимент
results = run_eeg_inverse_problem(nn_config, opt_config, loss_config)
```
"""

module InverseNpde

# Подключаем PML модуль как подмодуль
include("neural_pde_solver/PML.jl")
include("neural_pde_solver/PDEDefinitions.jl")
include("neural_pde_solver/NeuralNetwork.jl") 
include("neural_pde_solver/Optimization.jl")
include("neural_pde_solver/InverseProblem.jl")



# Реэкспортируем все основные структуры и функции
using .PDEDefinitions: PhysicalConstants
using .NeuralNetwork: NeuralNetworkConfig, create_neural_network
using .Optimization: OptimizationConfig, LossFunctionConfig, create_additional_loss, 
                   create_optimization_callback, setup_optimization, create_discretization,
                   validate_optimization_config
using .InverseProblem: DomainConfig, run_eeg_inverse_problem, get_available_devices
using .PML: PMLConfig, default_pml_config

# Экспортируем для использования вне модуля
export PhysicalConstants
export NeuralNetworkConfig, create_neural_network
export OptimizationConfig, LossFunctionConfig, create_additional_loss, 
       create_optimization_callback, setup_optimization, create_discretization,
       validate_optimization_config
export DomainConfig, run_eeg_inverse_problem, get_available_devices
export PMLConfig, default_pml_config

end # module