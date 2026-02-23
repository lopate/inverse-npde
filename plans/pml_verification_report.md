# Отчёт о проверке реализации PML для уравнений Максвелла

## 1. Проверка уравнений Максвелла в СГС

### 1.1 Теоретические уравнения в СГС (Гауссова система единиц)

В СГС уравнения Максвелла через потенциалы в калибровке Лоренца:

**Поля через потенциалы:**
```
B = ∇×A
E = -∇φ - (1/c)∂A/∂t
```

**Калибровка Лоренца в СГС:**
```
∇·A + (1/c)∂φ/∂t = 0
```

**Уравнения Даламбера в СГС:**
```
□φ = -4πρ        (для скалярного потенциала)
□A = -(4π/c)J    (для векторного потенциала)

где □ = Δ - (1/c²)∂²/∂t²
```

### 1.2 Проверка реализации в коде

**PhysicalConstants (строка 81 PDEDefinitions.jl):**
```julia
PhysicalConstants(; c=2.99792458e10, ε₀=1.0, ε=1.0, μ₀=1.0, μ=1.0)
```
✅ **ПРАВИЛЬНО**: В СГС ε₀=μ₀=1

**Уравнение для φ (строка 160):**
```julia
eq_φ = ... ~ -4 * pi * variables.ρ(...) / constants.ε
```
При ε=1: `-4πρ` ✅ **ПРАВИЛЬНО** для СГС!

**Уравнение для A (строка 196):**
```julia
... ~ -constants.μ * 4 * pi / constants.c * variables.j[i](...)
```
При μ=1: `-(4π/c)J` ✅ **ПРАВИЛЬНО** для СГС!

**Калибровка Лоренца (строки 202-205):**
```julia
(∇·A + (constants.ε * constants.μ / constants.c) * ∂φ/∂t) ~ 0
```
При ε=μ=1: `∇·A + (1/c)∂φ/∂t = 0` ✅ **ПРАВИЛЬНО** для СГС!

**Оператор Даламбера (строка 128):**
```julia
Δ - (constants.ε * constants.μ / constants.c^2) * ∂²F/∂t²
```
При ε=μ=1: `Δ - (1/c²)∂²F/∂t²` ✅ **ПРАВИЛЬНО**!

### 1.3 Вывод по уравнениям Максвелла

✅ **Уравнения Максвелла в СГС реализованы ПРАВИЛЬНО!**

---

## 2. Проверка реализации PML

### 2.1 Формулы пользователя для PML

**Основное уравнение:**
```
(1/c²)∂²u/∂t² = ∂/∂x((1/κx)∂u/∂x + ψ_x) + 
                 ∂/∂y((1/κy)∂u/∂y + ψ_y) + 
                 ∂/∂z((1/κz)∂u/∂z + ψ_z) + S
```

**Эволюция ψ:**
```
∂ψ_x/∂t = -(σx/κx + αx)ψ_x + (σx/κx²)∂u/∂x
```

### 2.2 Проверка реализации в PML.jl

**apply_pml_to_dalembert (строки 347-361):**

Реализовано:
```julia
term_x = Differential(variables.x)( (1/κx) * dFdx )
term_y = Differential(variables.y)( (1/κy) * dFdy )
term_z = Differential(variables.z)( (1/κz) * dFdz )

if ψ_x !== nothing
    term_x = term_x + Differential(variables.x)(in_pml * ψ_x)
end
```

⚠️ **ПРОБЛЕМА**: ψ добавляется отдельно от основного члена!

**Правильно по формуле пользователя:**
```julia
term_x = Differential(variables.x)( (1/κx) * dFdx + in_pml * ψ_x )
```

**Текущая реализация:**
```julia
term_x = Differential(variables.x)( (1/κx) * dFdx ) + Differential(variables.x)(in_pml * ψ_x)
```

Это **математически эквивалентно** только если in_pml=const, но in_pml - функция координат!

По правилу производной произведения:
```
∂/∂x(in_pml * ψ_x) = in_pml * ∂ψ_x/∂x + ψ_x * ∂(in_pml)/∂x
```

**Рекомендация**: Изменить на формулу пользователя для корректности.

### 2.3 Проверка уравнений для ψ

**create_pml_auxiliary_equations (строки 428-469):**

```julia
damping_x = in_pml * (σx/κx + αx) + (1 - in_pml) * inner_damping
source_φx = in_pml * (σx/κx²) * ∂φ/∂x

eq_ψ_φx = ∂ψ_φx/∂t ~ -damping_x * ψ_φx + source_φx
```

✅ **ПРАВИЛЬНО**: Соответствует формуле пользователя с добавлением inner_damping для стабильности.

### 2.4 Проблема с sigma_max = 0

**PMLConfig (строки 70-77):**
```julia
function PMLConfig(; 
    sigma_max::Float64 = 0.0,  # ⚠️ ПРОБЛЕМА!
    ...
)
```

При sigma_max = 0:
- σ(d) = 0 для всех d
- PML НЕ поглощает волны!

**Рекомендация**: Вычислять sigma_max автоматически:
```julia
# Формула Gedney
sigma_max = (m + 1) / (150 * π * D * dx)
# где m - polynomial_order, D - толщина PML, dx - шаг сетки
```

Или задать разумное значение по умолчанию.

---

## 3. Проверка вычисления потенциалов на датчиках

### 3.1 Анализ analyze_results (InverseProblem.jl)

```julia
for point in measured_points
    x, y, z, t, phi_measured = point
    phi_pred = (phi([x, y, z, t], params|>cpud))[1]
    
    phi_measured_de = phi_measured * norm_factor
    phi_pred_de = phi_pred * norm_factor
end
```

✅ **ПРАВИЛЬНО**: 
- `phi` - функция нейросети
- `[1]` - берёт первый элемент (φ_pred)
- Деанормировка применяется корректно

### 3.2 Проверка размерности выхода нейросети

**get_output_dim (PML.jl строки 24-40):**
```julia
base_dim = 8  # φ, Ax, Ay, Az, ρ, jx, jy, jz
pml_dim = 12  # ψ_φx, ψ_φy, ψ_φz, ψ_Axx, ..., ψ_Azz
return base_dim + pml_dim  # = 20 при PML enabled
```

✅ **ПРАВИЛЬНО**: Размерность соответствует количеству переменных.

---

## 4. Найденные проблемы и рекомендации

### 4.1 Критические проблемы

| # | Проблема | Файл | Строка | Решение |
|---|----------|------|--------|---------|
| 1 | sigma_max = 0 по умолчанию | PML.jl | 73 | Вычислять автоматически или задать ~0.1-1.0 |
| 2 | ψ добавляется вне производной | PML.jl | 354-361 | Объединить с основным членом |

### 4.2 Рекомендации по улучшению сходимости

1. **Автоматический расчёт sigma_max:**
   ```julia
   function compute_sigma_max(pml_config, domain_size, wavelength)
       m = pml_config.polynomial_order
       D = domain_size * pml_config.pml_thickness_ratio
       return (m + 1) / (150 * π * D) * c / wavelength
   end
   ```

2. **Баланс весов в loss-функции:**
   - Увеличить lambda_data для лучшего соответствия данным
   - Добавить отдельный вес для калибровки Лоренца

3. **Профиль поглощения:**
   - Использовать grad-profile вместо polynomial для лучшей стабильности
   - Добавить α-profile для низкочастотных компонент

4. **Архитектура нейросети:**
   - Увеличить количество скрытых слоёв
   - Использовать остаточные связи (residual connections)

---

## 5. План исправлений

### 5.1 Исправление PML.jl

1. Изменить `apply_pml_to_dalembert`:
   ```julia
   # Было:
   term_x = Differential(variables.x)( (1/κx) * dFdx )
   if ψ_x !== nothing
       term_x = term_x + Differential(variables.x)(in_pml * ψ_x)
   end
   
   # Должно быть:
   if ψ_x !== nothing
       term_x = Differential(variables.x)( (1/κx) * dFdx + in_pml * ψ_x )
   else
       term_x = Differential(variables.x)( (1/κx) * dFdx )
   end
   ```

2. Добавить автоматический расчёт sigma_max:
   ```julia
   function compute_optimal_sigma_max(pml_config, domains, constants)
       # Характерная длина волны
       domain_size = supremum(domains[1].domain) - infimum(domains[1].domain)
       D = domain_size * pml_config.pml_thickness_ratio
       m = pml_config.polynomial_order
       # Формула Gedney
       return (m + 1) * constants.c / (150 * π * D)
   end
   ```

3. Изменить значение по умолчанию:
   ```julia
   sigma_max::Float64 = -1.0  # -1 означает автоматический расчёт
   ```

### 5.2 Создание тестов

1. Тест профилей PML (σ, κ, α)
2. Тест индикатора PML-области
3. Тест уравнений для ψ
4. Интеграционный тест с простой волной

---

## 6. Заключение

### Что реализовано ПРАВИЛЬНО:
- ✅ Уравнения Максвелла в СГС
- ✅ Калибровка Лоренца
- ✅ Оператор Даламбера
- ✅ Уравнения эволюции ψ
- ✅ Вычисление потенциалов на датчиках
- ✅ Размерность выхода нейросети

### Что требует ИСПРАВЛЕНИЯ:
- ⚠️ sigma_max = 0 по умолчанию (критично!)
- ⚠️ Формула добавления ψ в оператор Даламбера

### Рекомендации по улучшению:
- Автоматический расчёт параметров PML
- Баланс весов loss-функции
- Улучшение архитектуры нейросети
- Добавление тестов
