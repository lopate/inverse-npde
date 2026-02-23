# План исправления масштабирования sigma_max в PML

## Диагностика проблемы

### Симптомы
При запуске эксперимента PDE loss имеет аномально большое значение:
```
PDE: 498005865363559940096.000  (~5×10^20)
Data: 0.786
λ: 1267461413192272445440.0    (~1.3×10^21)
```

### Корневая причина
Формула расчёта `sigma_max` в [`PML.jl`](src/neural_pde_solver/PML.jl:148) (строка 148):
```julia
sigma_max = -(m + 1) * c / (2.0 * d) * log(reflection_coefficient)
```

Эта формула Gedney предназначена для высокочастотных электромагнитных задач (СВЧ, оптика), но **не подходит** для квазистатических задач ЭЭГ.

### Численный анализ

**Текущие параметры:**
- c = 2.99792458e10 см/с (скорость света в СГС)
- domain_size = 20 см (от -10 до 10)
- pml_thickness_ratio = 0.15 → d = 3 см
- polynomial_order m = 3
- reflection_coefficient R = 0.001

**Текущее значение sigma_max:**
```
sigma_max = -(3 + 1) * 2.99792458e10 / (2.0 * 3) * log(0.001)
         = -4 * 2.99792458e10 / 6 * (-6.907)
         ≈ 1.38e11 1/с
```

**Правильное значение для ЭЭГ:**
Характерная частота ЭЭГ: f_char = 100 Гц
```
sigma_max = 2π * f_char * (m + 1) / 2
         = 2π * 100 * 2
         ≈ 1256 1/с
```

**Разница: 8 порядков!**

### Почему формула Gedney не подходит для ЭЭГ?

Формула Gedney выводится из условия:
- Минимизация отражения на границе PML для волны с частотой f
- При этом используется соотношение: σ_max ~ c / d

Для высокочастотных ЭМ задач:
- f ~ 1-100 ГГц
- λ = c/f ~ 0.003-0.3 м
- λ сопоставимо с размером домена

Для ЭЭГ:
- f ~ 1-100 Гц
- λ = c/f ~ 3000-300000 км
- λ >> размера домена ~0.2 м

**Вывод:** ЭЭГ работает в квазистатическом режиме, где волновые эффекты пренебрежимо малы. Формула Gedney даёт завышенное sigma_max.

---

## План исправления

### Шаг 1: Добавить параметр characteristic_frequency в PMLConfig

**Файл:** [`src/neural_pde_solver/PML.jl`](src/neural_pde_solver/PML.jl) (строки 22-66)

**Изменения:**
```julia
struct PMLConfig
    pml_thickness_ratio::Float64
    polynomial_order::Int
    sigma_max::Union{Float64, Nothing}
    kappa_max::Union{Float64, Nothing}
    alpha_max::Union{Float64, Nothing}
    inner_damping::Float64
    enabled::Bool
    characteristic_frequency::Union{Float64, Nothing}  # НОВОЕ ПОЛЕ
    
    function PMLConfig(; 
        pml_thickness_ratio::Float64 = 0.1, 
        polynomial_order::Int = 3, 
        sigma_max::Union{Float64, Nothing} = nothing, 
        kappa_max::Union{Float64, Nothing} = nothing, 
        alpha_max::Union{Float64, Nothing} = nothing,
        inner_damping::Float64 = 100.0,
        enabled::Bool = true,
        characteristic_frequency::Union{Float64, Nothing} = nothing  # НОВЫЙ ПАРАМЕТР
    )
        # ... валидация ...
        new(pml_thickness_ratio, polynomial_order, sigma_max, kappa_max, alpha_max, 
            inner_damping, enabled, characteristic_frequency)
    end
end
```

### Шаг 2: Изменить функцию compute_optimal_sigma_max

**Файл:** [`src/neural_pde_solver/PML.jl`](src/neural_pde_solver/PML.jl) (строки 133-151)

**Изменения:**
```julia
function compute_optimal_sigma_max(pml_config, domains, c; reflection_coefficient::Float64=0.001)
    # Если задана характерная частота - используем формулу для квазистатических задач
    if pml_config.characteristic_frequency !== nothing
        f_char = pml_config.characteristic_frequency
        m = pml_config.polynomial_order
        # sigma_max = 2π * f_char * (m + 1) / 2
        # Это обеспечивает затухание на характерной частоте
        return 2π * f_char * (m + 1) / 2
    end
    
    # Иначе используем оригинальную формулу Gedney для высокочастотных задач
    x_domain = domains[1]
    x_min, x_max = infimum(x_domain.domain), supremum(x_domain.domain)
    domain_size = x_max - x_min
    d = domain_size * pml_config.pml_thickness_ratio
    m = pml_config.polynomial_order
    
    sigma_max = -(m + 1) * c / (2.0 * d) * log(reflection_coefficient)
    return sigma_max
end
```

### Шаг 3: Обновить resolve_pml_config

**Файл:** [`src/neural_pde_solver/PML.jl`](src/neural_pde_solver/PML.jl) (строки 208-234)

**Изменения:**
```julia
function resolve_pml_config(pml_config::PMLConfig, domains, c)
    if !pml_config.enabled
        return pml_config
    end
    
    sigma_max_resolved = pml_config.sigma_max === nothing ? 
        compute_optimal_sigma_max(pml_config, domains, c) : pml_config.sigma_max
    
    kappa_max_resolved = pml_config.kappa_max === nothing ? 
        compute_optimal_kappa_max(pml_config, domains) : pml_config.kappa_max
    
    alpha_max_resolved = pml_config.alpha_max === nothing ? 
        compute_optimal_alpha_max(pml_config, domains) : pml_config.alpha_max
    
    return PMLConfig(
        pml_thickness_ratio = pml_config.pml_thickness_ratio,
        polynomial_order = pml_config.polynomial_order,
        sigma_max = sigma_max_resolved,
        kappa_max = kappa_max_resolved,
        alpha_max = alpha_max_resolved,
        inner_damping = pml_config.inner_damping,
        enabled = pml_config.enabled,
        characteristic_frequency = pml_config.characteristic_frequency  # НОВОЕ ПОЛЕ
    )
end
```

### Шаг 4: Обновить Python обёртку

**Файл:** [`src/inverse_npde/julia_interface.py`](src/inverse_npde/julia_interface.py)

**Изменения:**
1. Добавить параметр `characteristic_frequency` в функцию создания PMLConfig
2. Установить `characteristic_frequency = 100.0` для ЭЭГ

### Шаг 5: Обновить ноутбук

**Файл:** [`examples/eeg_realistic_demo.ipynb`](examples/eeg_realistic_demo.ipynb)

**Изменения:**
Передать `characteristic_frequency = 100.0` в PML конфигурацию.

---

## Ожидаемый результат

| Параметр | До исправления | После исправления |
|----------|----------------|-------------------|
| sigma_max | ~1.38e11 1/с | ~1256 1/с |
| PDE loss | ~10^20 | ~10^12 (на 8 порядков меньше) |
| Data loss | ~0.8 | ~0.8 (без изменений) |

После исправления PDE loss станет сопоставимым с Data loss, что позволит адаптивному балансу lambda_data работать корректно.

---

## Альтернативный подход: Явное задание sigma_max

Вместо добавления нового параметра можно просто явно задать sigma_max при создании PMLConfig:

```python
pml_config = jl.seval("""
    PMLConfig(
        pml_thickness_ratio = 0.15,
        polynomial_order = 3,
        sigma_max = 1256.0,  # Явное задание для ЭЭГ
        kappa_max = 2.5,
        alpha_max = 0.0,
        inner_damping = 100.0,
        enabled = true
    )
""")
```

Этот подход проще, но менее гибкий.

---

## Рекомендация

Рекомендую **добавить параметр characteristic_frequency**, так как:
1. Это более гибкое решение
2. Позволяет автоматически рассчитывать sigma_max для разных задач
3. Сохраняет обратную совместимость (characteristic_frequency = nothing использует формулу Gedney)
