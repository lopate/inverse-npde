# PML (Perfectly Matched Layer) в inverse-npde

Документация по использованию граничных условий PML для численного решения волновых уравнений в проекте inverse-npde.

## Содержание

1. [Введение](#введение)
2. [Физические основы](#физические-основы)
3. [Конфигурация PML](#конфигурация-pml)
4. [Использование в Julia](#использование-в-julia)
5. [Использование в Python](#использование-в-python)
6. [Параметры профилей](#параметры-профилей)
7. [Рекомендации по настройке](#рекомендации-по-настройке)

---

## Введение

**PML (Perfectly Matched Layer)** — это метод поглощающих граничных условий, который используется для подавления нежелательных отражений волн от границ вычислительной области. В контексте решения обратных задач ЭЭГ это критически важно для:

- Предотвращения отражений волн от границ расчётной области
- Обеспечения численной устойчивости решения
- Корректного моделирования распространения электромагнитных волн

---

## Физические основы

### Волновое уравнение

В проекте используется формулировка Максвелла через потенциалы в калибровке Лоренца:

- **Скалярный потенциал** $\varphi$: $\Box \varphi = -\frac{\rho}{\varepsilon_0}$
- **Векторный потенциал** $\mathbf{A}$: $\Box \mathbf{A} = -\mu_0 \mathbf{J}$

Где оператор Даламбера: $\Box = \Delta - \frac{1}{c^2}\frac{\partial^2}{\partial t^2}$

### CPML (Convolutional PML)

Реализация основана на CPML (Convolutional PML) с комплексным частотным сдвигом:

**Профили поглощения:**
- $\sigma_x(x), \sigma_y(y), \sigma_z(z) \ge 0$ — проводимость
- $\kappa_x(x), \kappa_y(y), \kappa_z(z) \ge 1$ — растяжение координат  
- $\alpha_x, \alpha_y, \alpha_z \ge 0$ — стабилизация

**Вспомогательные поля** $\psi$ вводятся для каждого потенциала:
- $\psi_{\varphi,x}, \psi_{\varphi,y}, \psi_{\varphi,z}$ для $\varphi$
- $\psi_{A_i,x}, \psi_{A_i,y}, \psi_{A_i,z}$ для каждой компоненты $A_i$

---

## Конфигурация PML

### Структура PMLConfig

```julia
mutable struct PMLConfig
    enabled::Bool              # Включить/выключить PML
    thickness::Float64         # Толщина слоя PML (в метрах)
    sigma_max::Float64        # Максимальная проводимость
    kappa_max::Float64        # Максимальное растяжение κ
    alpha_min::Float64        # Минимальный α для стабилизации
    alpha_max::Float64        # Максимальный α
    m::Int                    # Порядок полинома для σ (обычно 3-4)
    direction::Symbol         # :x, :y, :z, :xy, :xz, :yz, :xyz
end
```

### Параметры по умолчанию

```julia
default_pml_config() = PMLConfig(
    enabled = true,
    thickness = 0.1,        # 10 см PML слоя
    sigma_max = 1.0,       # σ_max = (m+1) * log(R) / (2 * d * pml_fraction)
    kappa_max = 5.0,       # Максимальное растяжение
    alpha_min = 0.1,       # Минимальная стабилизация
    alpha_max = 0.8,       # Максимальная стабилизация
    m = 3,                 # Полином 3-го порядка
    direction = :xyz       # Все направления
)
```

---

## Использование в Julia

### Базовое использование

```julia
using InverseNpde

# Использовать параметры по умолчанию
pml_config = default_pml_config()

# Или создать с пользовательскими параметрами
pml_config = PMLConfig(
    enabled = true,
    thickness = 0.2,        # 20 см
    sigma_max = 2.0,
    kappa_max = 10.0,
    alpha_min = 0.05,
    alpha_max = 0.5,
    m = 4,
    direction = :xyz
)

# Создать PDE систему с PML
pde_system = create_pde_system(
    domain, 
    physics_params; 
    pml_config = pml_config
)
```

### Отключение PML

```julia
# Полное отключение PML
pml_config = PMLConfig(enabled = false)
```

### Направления PML

```julia
# Только в направлении X
pml_config = PMLConfig(direction = :x)

# В направлениях X и Y (плоский случай)
pml_config = PMLConfig(direction = :xy)

# По всем направлениям (3D)
pml_config = PMLConfig(direction = :xyz)
```

---

## Использование в Python

### Базовое использование

```python
from inverse_npde import PMLConfig, create_pde_system

# Использовать параметры по умолчанию
pml_config = PMLConfig()

# С пользовательскими параметрами
pml_config = PMLConfig(
    enabled=True,
    thickness=0.2,      # 20 см
    sigma_max=2.0,
    kappa_max=10.0,
    alpha_min=0.05,
    alpha_max=0.5,
    m=4,
    direction="xyz"
)

# Создать PDE систему с PML
pde_system = create_pde_system(
    domain, 
    physics_params, 
    pml_config=pml_config
)
```

### Отключение PML

```python
# Полное отключение
pml_config = PMLConfig(enabled=False)
```

---

## Параметры профилей

### Профиль проводимости σ(x)

Проводимость нарастает от нуля на границе области к максимуму на внешней границе PML слоя:

$$\sigma_x(x) = \sigma_{\text{max}} \left( \frac{|x| - x_{\text{min}}}{d} \right)^m$$

где:
- $d$ — толщина PML слоя
- $m$ — порядок полинома (3-4)

### Профиль растяжения κ(x)

$$\kappa_x(x) = 1 + (\kappa_{\text{max}} - 1) \left( \frac{|x| - x_{\text{min}}}{d} \right)^m$$

### Профиль α(x)

$$\alpha_x(x) = \alpha_{\text{min}} + (\alpha_{\text{max}} - \alpha_{\text{min}}) \left( \frac{|x| - x_{\text{min}}}{d} \right)^m$$

---

## Рекомендации по настройке

### Выбор толщины слоя

- **Минимум**: 0.05-0.1 м (5-10 см)
- **Рекомендуется**: 0.1-0.2 м (10-20 см)
- **Для высоких частот**: большая толщина лучше

### Выбор sigma_max

Формула для расчёта:

$$\sigma_{\text{max}} = \frac{(m+1) \cdot |\ln(R)|}{2 \cdot d \cdot n}$$

где:
- $R$ — желаемый коэффициент отражения (например, $10^{-6}$)
- $d$ — толщина PML слоя
- $n$ — размерность (1, 2 или 3)

Примеры:
- $d = 0.1$, $m = 3$, $R = 10^{-6}$ → $\sigma_{\text{max}} \approx 0.7$
- $d = 0.2$, $m = 4$, $R = 10^{-8}$ → $\sigma_{\text{max}} \approx 1.0$

### Выбор kappa_max

- **Минимум**: 1.0 (без растяжения)
- **Рекомендуется**: 3.0-10.0
- **Слишком высокое**: может вызвать численную неустойчивость

### Выбор alpha

- **alpha_min**: 0.01-0.1 (небольшая стабилизация)
- **alpha_max**: 0.3-1.0
- **Для неустойчивых схем**: увеличить alpha

### Типичные конфигурации

#### Консервативная (низкие частоты, маленькая область)

```julia
PMLConfig(
    thickness = 0.1,
    sigma_max = 0.5,
    kappa_max = 3.0,
    alpha_min = 0.1,
    alpha_max = 0.5,
    m = 3
)
```

#### Агрессивная (высокие частоты, большая область)

```julia
PMLConfig(
    thickness = 0.2,
    sigma_max = 2.0,
    kappa_max = 10.0,
    alpha_min = 0.05,
    alpha_max = 0.8,
    m = 4
)
```

#### Баланс (рекомендуемая по умолчанию)

```julia
PMLConfig(
    thickness = 0.15,
    sigma_max = 1.0,
    kappa_max = 5.0,
    alpha_min = 0.1,
    alpha_max = 0.8,
    m = 3
)
```

---

## Диагностика проблем

### Признаки недостаточного PML

1. **Отражения на границах**: волны возвращаются в область интереса
2. **Нестабильность решения**: решение "взрывается" со временем
3. **Паразитные осцилляции**: высокочастотный шум вблизи границ

### Признаки слишком агрессивного PML

1. **Затухание в области интереса**: волны затухают слишком быстро
2. **Нарушение калибровки Лоренца**: $\nabla \cdot \mathbf{A} + \frac{1}{c^2}\partial_t \varphi \neq 0$
3. **Численные артефакты**: искажения формы волны

---

## Ссылки

- [Теоретическое описание PML](https://doi.org/10.1109/8.618268)
- [CPML для волновых уравнений](https://doi.org/10.1109/TAP.2002.1017661)
- [Документация NeuralPDE.jl](https://neuralpde.sciml.ai/stable/)

---

**Последнее обновление**: 2026-02-23  
**Версия**: 1.0.0
