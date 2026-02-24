# Реализация вещественного PML через затухание и экранирование

## Обзор

Замена сложного PML с ψ-полями на простой подход через коэффициенты затухания γ(r) и экранирования α(r).

## Математическая формулировка

### Модифицированные волновые уравнения

Для всех потенциалов (φ, A_x, A_y, A_z):

```
∇²u - (1/c²)∂²u/∂t² - γ(r)∂u/∂t - α(r)u = S
```

Где:
- **γ(r)** — динамическое затухание (вязкое трение), поглощает энергию волн
- **α(r)** — статический экранирующий член, обеспечивает экспоненциальное убывание потенциала

### Физический смысл

#### Динамика (волновой режим)
Член `γ(r)∂u/∂t` действует как вязкое трение: поглощает энергию распространяющихся волн, предотвращая отражения от внешней границы.

#### Статика
При `∂u/∂t = 0` уравнение переходит в экранированное уравнение Пуассона:
```
∇²u - α(r)u = S
```

В области где `S = 0` и `α > 0`, решение убывает экспоненциально:
```
u(x) ≈ u₀ exp(-√α · x)
```

## Профили поглощения

### Квадратичный профиль

Для слоя толщины d, начинающегося при x > x_pml:

```
γ(x) = γ_max · ((x - x_pml) / d)²
α(x) = α_max · ((x - x_pml) / d)²
```

### Трёхмерное поглощение

Итоговое поглощение аддитивно:

```
γ(r) = γ_x(x) + γ_y(y) + γ_z(z)
α(r) = α_x(x) + α_y(y) + α_z(z)
```

## Формулы для параметров (автоматический расчёт)

### Динамическое затухание γ_max

```
γ_max = 6c · ln(1/R) / d
```

Где:
- c — скорость света
- R — требуемый коэффициент отражения (например, 1e-6)
- d — толщина PML слоя

**Примечание:** Коэффициент 6 включает запас 2x для учёта отражений от границы PML.

### Статическое затухание α_max

```
α_max = (2 · ln(1/R) / L)²
```

Где:
- L — толщина PML слоя
- R — требуемый коэффициент отражения

## Структура PMLConfig

```julia
"""
Конфигурация PML через затухание и экранирование.

Параметры:
- pml_thickness_ratio: Доля толщины PML от размера домена (0.1 = 10%)
- reflection_coefficient: Требуемый коэффициент отражения (1e-4 по умолчанию)
- gamma_max: Максимум динамического затухания [1/с]. Если nothing - вычисляется автоматически.
- alpha_max: Максимум статического затухания [1/м²]. Если nothing - вычисляется автоматически.
- enabled: Включён ли PML
"""
struct PMLConfig
    pml_thickness_ratio::Float64
    reflection_coefficient::Float64
    gamma_max::Union{Float64, Nothing}
    alpha_max::Union{Float64, Nothing}
    enabled::Bool
end

# Конструктор по умолчанию
PMLConfig(;
    pml_thickness_ratio::Float64 = 0.1,
    reflection_coefficient::Float64 = 1e-4,  # 0.01% отражения - достаточно для ЭЭГ
    gamma_max::Union{Float64, Nothing} = nothing,
    alpha_max::Union{Float64, Nothing} = nothing,
    enabled::Bool = true
) = PMLConfig(pml_thickness_ratio, reflection_coefficient, gamma_max, alpha_max, enabled)
```

## GPU-дружественная реализация

### Индикаторная функция

```julia
"""
    step_indicator(x, x0) -> Float64

GPU-дружественная индикаторная функция.
Возвращает 0 при x < x0, 1 при x > x0.
Использует sign() вместо ifelse для совместимости с Zygote и GPU.
"""
step_indicator(x, x0) = (sign(x - x0) + 1) * 0.5
```

### Вычисление γ(x) для одной координаты

```julia
"""
    compute_gamma_1d(coord, pml_start, pml_thickness, gamma_max) -> Float64

Вычисляет коэффициент затухания γ для одной координаты.
Работает для PML на правой границе (coord > pml_start).
"""
function compute_gamma_1d(coord, pml_start, pml_thickness, gamma_max)
    # Индикатор: 1 внутри PML, 0 снаружи
    in_pml = step_indicator(coord, pml_start)
    # Нормализованное расстояние от начала PML (0 на границе, 1 на внешней стороне)
    distance = (coord - pml_start) / pml_thickness
    # Квадратичный профиль с затуханием
    return gamma_max * distance^2 * in_pml
end
```

### Полное затухание γ(r)

```julia
"""
    compute_gamma(x, y, z, domain, gamma_max) -> Float64

Вычисляет суммарный коэффициент затухания γ(r).
Учитывает PML со всех 6 сторон домена.
"""
function compute_gamma(x, y, z, domain, gamma_max)
    pml_thickness = domain.pml_thickness
    
    # Правые границы (x, y, z → +∞)
    gamma_x_right = compute_gamma_1d(x, domain.x_max - pml_thickness, pml_thickness, gamma_max)
    gamma_y_right = compute_gamma_1d(y, domain.y_max - pml_thickness, pml_thickness, gamma_max)
    gamma_z_right = compute_gamma_1d(z, domain.z_max - pml_thickness, pml_thickness, gamma_max)
    
    # Левые границы (x, y, z → -∞)
    gamma_x_left = compute_gamma_1d(-x, -domain.x_min - pml_thickness, pml_thickness, gamma_max)
    gamma_y_left = compute_gamma_1d(-y, -domain.y_min - pml_thickness, pml_thickness, gamma_max)
    gamma_z_left = compute_gamma_1d(-z, -domain.z_min - pml_thickness, pml_thickness, gamma_max)
    
    return gamma_x_right + gamma_x_left + 
           gamma_y_right + gamma_y_left + 
           gamma_z_right + gamma_z_left
end
```

### Аналогично для α(r)

```julia
"""
    compute_alpha_1d(coord, pml_start, pml_thickness, alpha_max) -> Float64

Вычисляет коэффициент экранирования α для одной координаты.
"""
function compute_alpha_1d(coord, pml_start, pml_thickness, alpha_max)
    in_pml = step_indicator(coord, pml_start)
    distance = (coord - pml_start) / pml_thickness
    return alpha_max * distance^2 * in_pml
end

"""
    compute_alpha(x, y, z, domain, alpha_max) -> Float64

Вычисляет суммарный коэффициент экранирования α(r).
"""
function compute_alpha(x, y, z, domain, alpha_max)
    pml_thickness = domain.pml_thickness
    
    alpha_x_right = compute_alpha_1d(x, domain.x_max - pml_thickness, pml_thickness, alpha_max)
    alpha_y_right = compute_alpha_1d(y, domain.y_max - pml_thickness, pml_thickness, alpha_max)
    alpha_z_right = compute_alpha_1d(z, domain.z_max - pml_thickness, pml_thickness, alpha_max)
    
    alpha_x_left = compute_alpha_1d(-x, -domain.x_min - pml_thickness, pml_thickness, alpha_max)
    alpha_y_left = compute_alpha_1d(-y, -domain.y_min - pml_thickness, pml_thickness, alpha_max)
    alpha_z_left = compute_alpha_1d(-z, -domain.z_min - pml_thickness, pml_thickness, alpha_max)
    
    return alpha_x_right + alpha_x_left + 
           alpha_y_right + alpha_y_left + 
           alpha_z_right + alpha_z_left
end
```

### Автоматический расчёт параметров

```julia
"""
    compute_gamma_max(c::Float64, d::Float64, R::Float64) -> Float64

Вычисляет γ_max по формуле: γ_max = 6c·ln(1/R)/d

Аргументы:
- c: Скорость света [м/с]
- d: Толщина PML слоя [м]
- R: Требуемый коэффициент отражения (например, 1e-6)
"""
compute_gamma_max(c::Float64, d::Float64, R::Float64) = 6.0 * c * log(1.0 / R) / d

"""
    compute_alpha_max(L::Float64, R::Float64) -> Float64

Вычисляет α_max по формуле: α_max = (2·ln(1/R)/L)²

Аргументы:
- L: Толщина PML слоя [м]
- R: Требуемый коэффициент отражения
"""
compute_alpha_max(L::Float64, R::Float64) = (2.0 * log(1.0 / R) / L)^2

"""
    resolve_pml_config(config::PMLConfig, domain, c::Float64) -> NamedTuple

Разрешает конфигурацию PML, вычисляя gamma_max и alpha_max если они не заданы.

Возвращает NamedTuple с полями:
- gamma_max::Float64
- alpha_max::Float64
- pml_thickness::Float64
"""
function resolve_pml_config(config::PMLConfig, domain, c::Float64)
    if !config.enabled
        return (gamma_max=0.0, alpha_max=0.0, pml_thickness=0.0)
    end
    
    pml_thickness = config.pml_thickness_ratio * min(
        domain.x_max - domain.x_min,
        domain.y_max - domain.y_min,
        domain.z_max - domain.z_min
    )
    
    gamma_max = something(config.gamma_max, compute_gamma_max(c, pml_thickness, config.reflection_coefficient))
    alpha_max = something(config.alpha_max, compute_alpha_max(pml_thickness, config.reflection_coefficient))
    
    return (gamma_max=gamma_max, alpha_max=alpha_max, pml_thickness=pml_thickness)
end
```

## Изменения в файлах

### 1. PML.jl — полная переработка

**Удалить:**
- `create_pml_variables()` — создание ψ-переменных
- `apply_pml_to_dalembert()` — модификация оператора Даламбера
- `create_pml_auxiliary_equations()` — уравнения для ψ-полей

**Добавить:**
- `step_indicator()` — GPU-дружественная индикаторная функция
- `compute_gamma_1d()` — затухание для одной координаты
- `compute_gamma()` — суммарное затухание
- `compute_alpha_1d()` — экранирование для одной координаты
- `compute_alpha()` — суммарное экранирование
- `compute_gamma_max()` — расчёт γ_max по формуле
- `compute_alpha_max()` — расчёт α_max по формуле
- `resolve_pml_config()` — разрешение конфигурации с автоматическим расчётом

### 2. PDEDefinitions.jl — модификация уравнений

**Было (для φ):**
```julia
dalembert(phi, c) ~ -4 * π * rho / epsilon_0
```

**Станет:**
```julia
# PML члены
gamma_r = compute_gamma(x, y, z, domain, pml_config.gamma_max)
alpha_r = compute_alpha(x, y, z, domain, pml_config.alpha_max)

# Модифицированное уравнение
Dxx(phi) + Dyy(phi) + Dzz(phi) - (1/c^2) * Dtt(phi) - gamma_r * Dt(phi) - alpha_r * phi ~ -4 * π * rho / epsilon_0
```

Аналогично для A_x, A_y, A_z.

### 3. NeuralNetwork.jl — уменьшение выходов

**Было:** `output_dim = 20` (φ, A_x, A_y, A_z + 12 ψ-полей + 4 дополнительных)

**Станет:** `output_dim = 8` (φ, A_x, A_y, A_z + 4 дополнительных для обратной задачи)

### 4. InverseProblem.jl — упрощение

**Удалить:**
- Добавление ψ-уравнений в PDE систему
- Граничные условия для ψ-полей

### 5. julia_interface.py — новые параметры

**Было:**
```python
PMLConfig(
    enabled=True,
    sigma_max=1e6,
    kappa_max=1.0,
    alpha_max=0.0,
    pml_order=2,
    pml_thickness_ratio=0.1,
    c_reference=1.0
)
```

**Станет:**
```python
@dataclass
class PMLConfig:
    enabled: bool = True
    pml_thickness_ratio: float = 0.1
    reflection_coefficient: float = 1e-6
    gamma_max: Optional[float] = None  # auto if None
    alpha_max: Optional[float] = None  # auto if None
```

## Сравнение подходов

| Параметр | До (с ψ-полями) | После (γ + α) |
|----------|-----------------|---------------|
| Выходы нейросети | 20 | 8 |
| PDE уравнения | 16 | 4 |
| Дополнительные переменные | 12 ψ-полей | 0 |
| Параметры конфигурации | 7 | 5 |
| Сложность реализации | Высокая | Низкая |
| GPU-совместимость | Ограниченная | Полная |
| Настраиваемость | Ограниченная | Полная (γ_max, α_max можно задать вручную) |

## Пример расчёта параметров

Для ЭЭГ с доменом 0.2 м и скоростью c = 3e8 м/с:

```
Толщина PML: d = 0.2 * 0.1 = 0.02 м
R = 1e-4 (по умолчанию)

γ_max = 6 * 3e8 * ln(1e4) / 0.02
      = 6 * 3e8 * 9.21 / 0.02
      = 8.3e10 1/с

α_max = (2 * ln(1e4) / 0.02)²
      = (2 * 9.21 / 0.02)²
      = 8.5e5 1/м²
```

**Сравнение с R = 1e-6:**
- γ_max: 8.3e10 vs 1.24e11 (в 1.5 раза меньше)
- α_max: 8.5e5 vs 1.9e6 (в 2.2 раза меньше)

Меньшие значения улучшают численную устойчивость при сохранении достаточного затухания.

**С ручной настройкой:**
```python
PMLConfig(
    enabled=True,
    pml_thickness_ratio=0.1,
    reflection_coefficient=1e-6,
    gamma_max=1e11,  # Явное задание
    alpha_max=2e6    # Явное задание
)
```

## Валидация

После реализации проверить:

1. **PDE loss** должен быть порядка 1-10 (не 10^20)
2. **Калибровка Лоренца** должна сохраняться в основной области
3. **Потенциалы** должны затухать в PML-слое
4. **Отражения** от границ должны быть минимальны

## Тесты

Обновить `tests/test_pml_pde_system.jl`:

1. Тест `step_indicator()` на GPU
2. Тест `compute_gamma()` в основной области (должен быть 0)
3. Тест `compute_gamma()` в PML-слое (должен быть > 0)
4. Тест `compute_alpha()` аналогично
5. Тест автоматического расчёта γ_max и α_max
6. Тест ручной настройки γ_max и α_max
7. Интеграционный тест с полным PDE
