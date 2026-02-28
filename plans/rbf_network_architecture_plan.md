# План: Архитектура нейросети с RBF базисом для точек датчиков

## Концепция

Использовать гибридный подход, где решение представляется как:

```
φ(x,y,z,t) = Σᵢ wᵢ · RBF(||(x,y,z,t) - cᵢ||) + NN(x,y,z,t)
```

Где:
- `cᵢ` - центры RBF (координаты датчиков)
- `wᵢ` - обучаемые веса RBF
- `NN` - нейросеть для остаточной компоненты (residual network)

## Преимущества подхода

1. **Точное воспроизведение** - RBF часть гарантирует точное прохождение через точки датчиков
2. **Гибкость** - нейросеть учит остаточную компоненту для удовлетворения PDE
3. **Экстраполяция** - нейросеть обеспечивает предсказание вне точек датчиков

## Типы RBF функций

| Тип | Формула | Параметры |
|-----|---------|-----------|
| Gaussian | exp(-γ·r²) | γ - ширина |
| Multiquadric | √(1 + (ε·r)²) | ε - форма |
| Inverse Quadratic | 1/(1 + ε·r²) | ε - форма |
| Polyharmonic | r^k (k=1,3,5...) | k - степень |
| Thin Plate Spline | r²·log(r) | - |

## Реализация

### 1. Структура RBFNetworkConfig

```julia
struct RBFNetworkConfig
    # Параметры RBF
    rbf_type::Symbol              # :gaussian, :multiquadric, etc.
    rbf_epsilon::Float32         # Параметр формы RBF
    
    # Параметры нейросети
    nn_input_dim::Int             # Обычно 4: [x,y,z,t]
    nn_hidden_layers::Vector{Int} # [32, 32]
    nn_output_dim::Int            # 8 или 24
    activation::Symbol            # :tanh, :σ, etc.
    
    # Параметры датчиков
    sensor_coords::Matrix{Float32} # [4, N] - координаты датчиков [x,y,z,t]
    num_sensors::Int               # N - число датчиков
    
    # Параметры гибридизации
    use_hybrid::Bool              # Использовать гибрид или чистый RBF
    use_gpu::Bool
end
```

### 2. Функции RBF

```julia
# RBF kernel functions
gaussian_rbf(r, ε) = exp(-ε * r^2)
multiquadric_rbf(r, ε) = sqrt(1 + (ε * r)^2)
inverse_quadratic_rbf(r, ε) = 1 / (1 + ε * r^2)
thin_plate_spline_rbf(r) = r^2 * log(r + eps())

# Вычисление RBF матрицы
function compute_rbf_matrix(coords::Matrix{Float32}, centers::Matrix{Float32}, ε::Float32)
    # Ψ[i,j] = RBF(||coords[:,i] - centers[:,j]||)
end
```

### 3. Прямой проход гибридной сети

```julia
function forward_rbf_hybrid(x, params, config)
    # x: [4, batch_size] - входные координаты
    
    # 1. RBF часть
    # Ψ = compute_rbf_matrix(x, config.sensor_coords, config.rbf_epsilon)
    # rbf_output = params.rbf_weights * Ψ  [1, batch_size]
    
    # 2. NN часть  
    # nn_output = nn_forward(x, params.nn_params)
    
    # 3. Комбинация
    # return config.rbf_weight * rbf_output + (1 - config.rbf_weight) * nn_output
end
```

### 4. Обучение

- **Веса RBF** - оптимизируются напрямую (линейные параметры)
- **Параметры NN** - оптимизируются через градиентный спуск

## Архитектурная диаграмма

```
                    Вход: [x, y, z, t]
                           │
           ┌───────────────┴───────────────┐
           │                               │
           ▼                               ▼
    ┌─────────────┐                ┌─────────────┐
    │  RBF Layer  │                │  NN Branch  │
    │             │                │             │
    │ Centers:    │                │ Dense(4→32) │
    │  [x₁,y₁,z₁,t₁]               │    ↓        │
    │  [x₂,y₂,z₂,t₂]    ──→        │ Dense(32→32)│
    │  ...                         │    ↓        │
    │  [xN,yN,zN,tN]               │ Dense(32→8) │
    │             │                │             │
    │ Weights: w  │                │             │
    └──────┬──────┘                └──────┬──────┘
           │                               │
           └───────────────┬───────────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │  Weighted Sum   │
                  │  α·RBF + (1-α)·NN│
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ Выход: φ, A, ρ, j│
                  └─────────────────┘
```

## Модификация существующего кода

### Файл: NeuralNetwork.jl

1. Добавить `RBFNetworkConfig` структуру
2. Добавить функции `compute_rbf_kernel`, `compute_rbf_matrix`
3. Добавить `create_rbf_hybrid_network` функцию
4. Добавить инициализацию параметров для RBF слоя

### Файл: Optimization.jl

1. Адаптировать loss функции для гибридной архитектуры
2. Добавить separate optimization для линейных (RBF) и нелинейных (NN) параметров

## TODO список

- [ ] Создать структуру RBFNetworkConfig в NeuralNetwork.jl
- [ ] Реализовать функции ядра RBF (gaussian, multiquadric, etc.)
- [ ] Реализовать compute_rbf_matrix для эффективного вычисления
- [ ] Создать функцию create_rbf_hybrid_network
- [ ] Реализовать forward pass гибридной сети
- [ ] Добавить инициализацию RBF весов
- [ ] Интегрировать с PDE loss функциями
- [ ] Добавить пример использования
