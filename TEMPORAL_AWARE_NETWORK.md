# Temporal-Aware Neural Network Implementation

## Overview

The `TemporalAwareWrapper` is a neural network architecture designed specifically for solving the EEG inverse problem in the inverse NPDE framework. It addresses the stationary solution problem by explicitly incorporating temporal variation through Fourier feature encoding.

## Architecture Design

### Problem Addressed
Three critical issues with standard MLP approaches:
1. **PDE Loss Dominance**: SpaceDerivativeWeights were 50-100x larger than data loss weights
2. **Derivative Evaluation Mismatch**: Spline derivatives at midpoints vs finite differences at boundaries
3. **No Temporal Structure**: Standard MLPs lack architectural bias toward temporal variation

### Solution: Temporal-Aware Network
```
Input: [x, y, z, t] (4D coordinates)
  ↓
  ├─→ Spatial Branch:      [x,y,z] → Dense(32) → Dense(32) → Dense(32) → 32 features
  │
  ├─→ Temporal Branch:     Fourier(t) → Dense(16) → Dense(16) → 16 features
  │   └─ Fourier encoding: [sin(πt), cos(πt), sin(2πt), cos(2πt), ..., sin(8πt), cos(8πt)]
  │
  └─→ Fusion Layers:       cat[48 features] → Dense(32) → Dense(8) → 8 outputs
       Output: [φ, Ax, Ay, Az, ρ, jx, jy, jz]
```

## Usage

### 1. Create Configuration
```julia
config = TemporalAwareNetworkConfig(
    spatial_hidden_layers = [32, 32],      # Spatial branch hidden dimensions
    spatial_output_dim = 32,                # Spatial feature dimension
    num_fourier_frequencies = 8,            # Number of Fourier frequency pairs (→ 16 features)
    temporal_hidden_layers = [16],          # Temporal branch hidden dimensions  
    temporal_output_dim = 16,               # Temporal feature dimension
    fusion_hidden_layers = [32],            # Fusion/output layers
    output_dim = 8,                         # Final output dimension (for 8 PDE variables)
    activation = :tanh,                     # Activation function (:tanh, :relu, :elu, etc.)
    use_gpu = true                          # Use GPU if available
)
```

### 2. Create Network
```julia
network = create_temporal_aware_network(config)
# Returns: TemporalAwareWrapper with configured layers
```

### 3. Initialize Parameters
```julia
using Random
ps = initialize_temporal_aware_parameters(
    network,
    Random.default_rng(),
    use_gpu = true  # Keyword argument!
)
# Returns: ComponentArray with 4616 parameters (default config)
```

### 4. Forward Pass
```julia
# Single point
x_single = Float32[x, y, z, t]
output = network(x_single, ps)  # Output size: (8,)

# Batch of points
x_batch = randn(Float32, 4, N)  # N points
output_batch = network(x_batch, ps)  # Output size: (8, N)
```

## Key Features

### Fourier Time Encoding
- **Purpose**: Explicitly encode periodic and oscillatory behavior in time
- **Implementation**: `create_fourier_features(t, num_frequencies)` produces `sin(kπt)` and `cos(kπt)` pairs
- **Effect**: Forces network to learn temporal patterns rather than static mappings

### Separated Spatial/Temporal Processing
- **Spatial branch** processes position independently
- **Temporal branch** processes time independently through Fourier features
- **Fusion layer** combines spatial and temporal information
- **Benefit**: Architectural bias toward temporal variation in solutions

### NeuralPDE.jl Compatibility
- Accepts standard input shape: `(4,)` or `(4, N)`
- Produces standard output shape: `(8,)` or `(8, N)`
- Returns `(output, state)` tuple as expected by Lux
- Compatible with automatic differentiation (Zygote)

## Integration with Solver

### Using with NeuralPDE.jl
```julia
using NeuralPDE, Lux

# Create network and parameters
config = TemporalAwareNetworkConfig()
network = create_temporal_aware_network(config)
ps = initialize_temporal_aware_parameters(network)

# Create discretization
domains = [x ∈ DomainData(...), t ∈ DomainData(...)]
prob = PIDEProblem(pde_system, network)

# Use in solver
sym_prob = symbolic_discretize(prob, discretization)
pinn_loss_function = build_loss_function(sym_prob, network, ps)
```

## Configuration Examples

### Lightweight (Fast Training)
```julia
config = TemporalAwareNetworkConfig(
    spatial_hidden_layers = [16, 16],
    num_fourier_frequencies = 4,           # 8 features
    temporal_hidden_layers = [8],
    fusion_hidden_layers = [16]
)
# Total parameters: ~800
```

### Default (Balanced)
```julia
config = TemporalAwareNetworkConfig()
# spatial: [32, 32], temporal: [16], fusion: [32]
# Total parameters: ~4600
```

### Large (Better Expressiveness)
```julia
config = TemporalAwareNetworkConfig(
    spatial_hidden_layers = [64, 64, 32],
    num_fourier_frequencies = 16,          # 32 features
    temporal_hidden_layers = [32, 16],
    fusion_hidden_layers = [64, 32]
)
# Total parameters: ~30000+
```

## Performance Notes

### Mixed-Precision Warning
You may see warnings about mixed-precision matmul (Float32 × Float64). This is typically benign but can be resolved by ensuring all parameters are Float32:
```julia
ps = initialize_temporal_aware_parameters(network) .|> Float32
```

### GPU Acceleration
- Enable with `use_gpu=true` in config
- CUDA will be used automatically if available
- Parameters are transferred via `ps |> gpu_device()`

## Testing

All components have been validated:
```julia
# Configuration validation
config = TemporalAwareNetworkConfig()

# Network creation (TemporalAwareWrapper)
network = create_temporal_aware_network(config)

# Parameter initialization (4616 weights/biases)
ps = initialize_temporal_aware_parameters(network; use_gpu=false)

# Forward pass with single point
out = network(Float32[1, 2, 3, 0.5], ps)  # → (8,)

# Forward pass with batch
out = network(randn(Float32, 4, 5), ps)   # → (8, 5)

# Fourier feature encoding
fourier = create_fourier_features(0.5, 8)  # → 16 features
```

## Related Files

- **Implementation**: [src/neural_pde_solver/NeuralNetwork.jl](../src/neural_pde_solver/NeuralNetwork.jl)
  - `TemporalAwareNetworkConfig` (struct)
  - `TemporalAwareWrapper` (callable struct)
  - `create_temporal_aware_network()` (factory function)
  - `create_fourier_features()` (time encoding)
  - `initialize_temporal_aware_parameters()` (initialization)

- **Examples**: [examples/eeg_realistic_demo_julia.ipynb]
  - Diagnostic cells for architecture validation
  - Data temporal variance analysis
  - Integration examples

## Future Improvements

1. **Adaptive Fourier Frequencies**: Learn frequency weights during training
2. **Attention Mechanisms**: Learned spatial-temporal interaction weights
3. **Residual Connections**: Skip connections between branches
4. **Layer Normalization**: Improve training stability for deep networks
5. **Spectral Methods**: Replace Fourier features with learnable spectral basis

## References

See [NPDE solving with Temporal-Aware architectures](../plans/pml_verification_report.md)
