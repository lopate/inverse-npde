# План реализации предсказания производных для улучшения сходимости

## Цель

Модифицировать оператор Даламбера для использования предсказанных производных первого порядка вместо автоматического дифференцирования второго порядка. Это улучшит сходимость, так как вместо шумного автоматического дифференцирования второго порядка сеть будет предсказывать первые производные напрямую.

## Ключевое изменение

### Текущая реализация (автоматическое дифференцирование):
```
□φ = ∂²φ/∂x² + ∂²φ/∂y² + ∂²φ/∂z² - (1/c²)∂²φ/∂t² - γ∂φ/∂t - αφ
```
Где вторые производные вычисляются через `Differential(variables.t)^2)(F)`.

### Новая реализация (предсказанные производные + autodiff для PML):
```
□φ = ∂(DφDx)/∂x + ∂(DφDy)/∂y + ∂(DφDz)/∂z - (1/c²)∂(DφDt)/∂t - γ*autodiff(φ) - α*φ
```

**Важное уточнение:**
- Для члена PML затухания `γ*∂φ/∂t` используется autodiff на выходе нейросети (это критично для корректной работы PML)
- Для вторых производных в операторе Даламбера используются предсказанные первые производные

## Текущее состояние

- Нейросеть предсказывает 8 переменных: φ, Ax, Ay, Az, ρ, jx, jy, jz
- Выходная размерность: 8
- Оператор Даламбера использует автоматическое дифференцирование для ∂²/∂t², ∂²/∂x² и т.д.

## Целевое состояние

- Нейросеть предсказывает 24 переменные (базовые + производные)
- Оператор Даламбера использует предсказанные первые производные
- PML член γ*∂φ/∂t вычисляется через autodiff на φ
- Выходная размерность: 24

## Mermaid: Новая архитектура

```mermaid
graph TD
    A[Вход: x, y, z, t] --> B[Neural Network 24 outputs]
    
    B --> C1[φ 1]
    B --> C2[Ax 2]
    B --> C3[Ay 3]
    B --> C4[Az 4]
    B --> C5[ρ 5]
    B --> C6[jx 6]
    B --> C7[jy 7]
    B --> C8[jz 8]
    B --> C9[DφDt 9]
    B --> C10[DφDx 10]
    B --> C11[DφDy 11]
    B --> C12[DφDz 12]
    B --> C13[DAxDt 13]
    B --> C14[DAxDx 14]
    B --> C15[DAxDy 15]
    B --> C16[DAxDz 16]
    B --> C17[DAyDt 17]
    B --> C18[DAyDx 18]
    B --> C19[DAyDy 19]
    B --> C20[DAyDz 20]
    B --> C21[DAzDt 21]
    B --> C22[DAzDx 22]
    B --> C23[DAzDy 23]
    B --> C24[DAzDz 24]
    
    C1 --> D[PML: γ * autodiff]
    C1 --> E[α * φ]
    
    C9 --> F[Оператор Даламбера: ∂Dx + ∂Dy + ∂Dz - (1/c²)∂Dt]
    C10 --> F
    C11 --> F
    C12 --> F
    
    D --> G[PDE: □φ]
    E --> G
    F --> G
    
    G --> H[Loss Function]
```

## Порядок выходов нейросети (24)

```
[1]  φ       - скалярный потенциал
[2]  Ax      - x-компонента векторного потенциала
[3]  Ay      - y-компонента векторного потенциала
[4]  Az      - z-компонента векторного потенциала
[5]  ρ       - плотность заряда
[6]  jx      - x-компонента плотности тока
[7]  jy      - y-компонента плотности тока
[8]  jz      - z-компонента плотности тока
[9]  DφDt    - ∂φ/∂t (предсказанная)
[10] DφDx    - ∂φ/∂x (предсказанная)
[11] DφDy    - ∂φ/∂y (предсказанная)
[12] DφDz    - ∂φ/∂z (предсказанная)
[13] DAxDt   - ∂Ax/∂t (предсказанная)
[14] DAxDx   - ∂Ax/∂x (предсказанная)
[15] DAxDy   - ∂Ax/∂y (предсказанная)
[16] DAxDz   - ∂Ax/∂z (предсказанная)
[17] DAyDt   - ∂Ay/∂t (предсказанная)
[18] DAyDx   - ∂Ay/∂x (предсказанная)
[19] DAyDy   - ∂Ay/∂y (предсказанная)
[20] DAyDz   - ∂Ay/∂z (предсказанная)
[21] DAzDt   - ∂Az/∂t (предсказанная)
[22] DAzDx   - ∂Az/∂x (предсказанная)
[23] DAzDy   - ∂Az/∂y (предсказанная)
[24] DAzDz   - ∂Az/∂z (предсказанная)
```

## Пошаговый план реализации

### Этап 1: Обновление NeuralNetwork.jl

1.1. **Добавить новую структуру DerivativeAwareNetworkConfig**
   - Наследует основную функциональность NeuralNetworkConfig
   - Добавить флаг `include_derivatives::Bool` (по умолчанию true)
   - Автоматически вычислять output_dim = 24 при включённых производных

1.2. **Обновить create_neural_network()**
   - Поддержка output_dim=24
   - Убедиться что скрытые слои достаточного размера для 24 выходов

1.3. **Обновить create_output_splitter()**
   - Обработка 24 выходов
   - Разделение на базовые переменные и производные

### Этап 2: Обновление PDEDefinitions.jl (КЛЮЧЕВОЙ)

2.1. **Расширить VariableSet**
   - Добавить производные переменные для φ: DφDt, DφDx, DφDy, DφDz
   - Добавить производные переменные для A (3 компоненты × 4 производные = 12)

2.2. **Модифицировать dalembert_operator()**
   - Текущая реализация:
     ```julia
     Δ = laplacian(F, params)  # ∂²F/∂x² + ∂²F/∂y² + ∂²F/∂z²
     d2F_dt2 = (Differential(variables.t)^2)(F)  # ∂²F/∂t²
     ```
   - Новая реализация:
     ```julia
     # Используем предсказанные производные первого порядка для Даламбера
     dF_dt_pred = variables.DφDt(...)  # Предсказанная ∂φ/∂t
     dF_dx_pred = variables.DφDx(...)  # Предсказанная ∂φ/∂x
     dF_dy_pred = variables.DφDy(...)  # Предсказанная ∂φ/∂y
     dF_dz_pred = variables.DφDz(...)  # Предсказанная ∂φ/∂z
     
     # Применяем дифференцирование к предсказанным производным
     d2F_dt2_term = Differential(variables.t)(dF_dt_pred)  # ∂(∂φ/∂t)/∂t
     d2F_dx2_term = Differential(variables.x)(dF_dx_pred)  # ∂(∂φ/∂x)/∂x
     # и т.д.
     
     # Итого: □φ = ∂x + ∂y + ∂z - (1/c²)∂t - γ*Dt - α*φ
     
     # НО: для члена PML затухания используем autodiff на φ
     dF_dt_autodiff = Differential(variables.t)(F)  # ∂φ/∂t через autodiff
     pml_term = params.γ * dF_dt_autodiff  # γ * ∂φ/∂t (PML)
     screening_term = params.α * F  # α * φ (экранирование)
     ```

2.3. **Обновить create_pde_system()**
   - Добавить производные переменные в список vars
   - Система теперь имеет 8 + 16 = 24 неизвестные функции

2.4. **Обновить create_boundary_conditions()**
   - Добавить граничные условия для производных (нулевые на границах)

### Этап 3: Обновление InverseProblem.jl

3.1. **Обновить _create_complete_setup_standard()**
   - output_dim = 24 вместо 8
   - Обновить информационное сообщение

3.2. **Обновить analyze_results()**
   - Обновить индексацию для извлечения φ из 24 выходов

3.3. **Обновить нормализацию**
   - Проверить совместимость с новым размером выхода

### Этап 4: Опционально - Temporal-Aware архитектура

4.1. **Обновить TemporalAwareNetworkConfig**
   - output_dim = 24

4.2. **Обновить create_temporal_aware_network()**
   - Поддержка 24 выходов

## Ключевая формула оператора Даламбера с PML

```julia
function dalembert_operator(F, variables, params)
    # Предсказанные производные первого порядка для вторых производных
    dF_dt_pred = variables.DφDt(variables.t, variables.x, variables.y, variables.z)
    dF_dx_pred = variables.DφDx(variables.t, variables.x, variables.y, variables.z)
    dF_dy_pred = variables.DφDy(variables.t, variables.x, variables.y, variables.z)
    dF_dz_pred = variables.DφDz(variables.t, variables.x, variables.y, variables.z)
    
    # Применяем дифференцирование к предсказанным производным
    d2F_dt2 = Differential(variables.t)(dF_dt_pred)
    d2F_dx2 = Differential(variables.x)(dF_dx_pred)
    d2F_dy2 = Differential(variables.y)(dF_dy_pred)
    d2F_dz2 = Differential(variables.z)(dF_dz_pred)
    
    # Оператор Даламбера с предсказанными производными
    dAlembert = d2F_dx2 + d2F_dy2 + d2F_dz2 - (1/params.c^2) * d2F_dt2
    
    # PML члены - используем autodiff на φ для корректного затухания
    dF_dt_autodiff = Differential(variables.t)(F)
    pml_damping = params.γ * dF_dt_autodiff  # γ * ∂φ/∂t (autodiff)
    screening = params.α * F  # α * φ
    
    return dAlembert - pml_damping - screening
end
```

## Дополнительные уравнения для PDE системы

Как в примере пользователя, нужно добавить уравнения связывания:

```julia
# Уравнения связывания для φ
Dt(φ(x,y,z,t)) ~ DφDt(x,y,z,t)
Dx(φ(x,y,z,t)) ~ DφDx(x,y,z,t)
Dy(φ(x,y,z,t)) ~ DφDy(x,y,z,t)
Dz(φ(x,y,z,t)) ~ DφDz(x,y,z,t)

# Уравнения связывания для Ax
Dt(Ax(x,y,z,t)) ~ DAxDt(x,y,z,t)
Dx(Ax(x,y,z,t)) ~ DAxDx(x,y,z,t)
Dy(Ax(x,y,z,t)) ~ DAxDy(x,y,z,t)
Dz(Ax(x,y,z,t)) ~ DAxDz(x,y,z,t)

# Аналогично для Ay, Az
```

## Файлы для изменения

1. `src/neural_pde_solver/NeuralNetwork.jl`
2. `src/neural_pde_solver/PDEDefinitions.jl` - основные изменения в `dalembert_operator()`
3. `src/neural_pde_solver/InverseProblem.jl`
4. `src/neural_pde_solver/Optimization.jl` (проверить совместимость)

## Метрики успеха

- Оператор Даламбера использует предсказанные производные для вторых производных
- PML член γ*∂φ/∂t использует autodiff на φ
- Нейросеть успешно обучается с 24 выходами
- Сходимость улучшается по сравнению с автоматическим дифференцированием
- Потери на валидации уменьшаются
- Физические ограничения (уравнения связывания) выполняются
