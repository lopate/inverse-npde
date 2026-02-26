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
using .NeuralNetwork: TemporalAwareNetworkConfig, NeuralNetworkConfig, create_neural_network, 
                      create_temporal_aware_network, initialize_parameters, 
                      initialize_temporal_aware_parameters
using .Optimization: OptimizationConfig, LossFunctionConfig, create_additional_loss, 
                   create_optimization_callback, setup_optimization, create_discretization,
                   validate_optimization_config, compute_field_energy_loss
using .InverseProblem: DomainConfig, run_eeg_inverse_problem, get_available_devices, save_results, load_results
using .PML: PMLConfig, default_pml_config

# Экспортируем для использования вне модуля
export PhysicalConstants
export TemporalAwareNetworkConfig, NeuralNetworkConfig, create_neural_network,
       create_temporal_aware_network, initialize_parameters, initialize_temporal_aware_parameters
export OptimizationConfig, LossFunctionConfig, create_additional_loss, 
       create_optimization_callback, setup_optimization, create_discretization,
       validate_optimization_config, compute_field_energy_loss
export DomainConfig, run_eeg_inverse_problem, get_available_devices
export PMLConfig, default_pml_config
export save_results, load_results

end # module