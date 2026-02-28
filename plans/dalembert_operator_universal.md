# Универсальный оператор Даламбера для φ и A

## Новая реализация с универсальными параметрами

```julia
function dalembert_operator(F, dF_dt_pred, dF_dx_pred, dF_dy_pred, dF_dz_pred, 
                            params, constants::PhysicalConstants, variables::VariableSet, domains; 
                            pml_config::PMLConfig=default_pml_config())
    # Применяем дифференцирование к предсказанным производным
    d2F_dt2 = Differential(variables.t)(dF_dt_pred)  # ∂(∂F/∂t)/∂t
    d2F_dx2 = Differential(variables.x)(dF_dx_pred)  # ∂(∂F/∂x)/∂x
    d2F_dy2 = Differential(variables.y)(dF_dy_pred)  # ∂(∂F/∂y)/∂y
    d2F_dz2 = Differential(variables.z)(dF_dz_pred)  # ∂(∂F/∂z)/∂z
    
    Δ = d2F_dx2 + d2F_dy2 + d2F_dz2
    
    if !pml_config.enabled
        return Δ - (constants.ε * constants.μ / constants.c^2) * d2F_dt2
    end
    
    γ = compute_gamma(pml_config, variables.x, variables.y, variables.z, domains)
    α = compute_alpha(pml_config, variables.x, variables.y, variables.z, domains)
    
    # Для PML используем autodiff на F (важно для корректного затухания)
    dF_dt_autodiff = Differential(variables.t)(F)  # ∂F/∂t через autodiff
    
    return Δ - (constants.ε * constants.μ / constants.c^2) * d2F_dt2 - γ * dF_dt_autodiff - α * F
end
```

## Пример использования для φ и A

```julia
# Для скалярного потенциала φ
eq_φ = dalembert_operator(
    variables.φ(variables.x, variables.y, variables.z, variables.t), 
    variables.DφDt(variables.x, variables.y, variables.z, variables.t),
    variables.DφDx(variables.x, variables.y, variables.z, variables.t),
    variables.DφDy(variables.x, variables.y, variables.z, variables.t),
    variables.DφDz(variables.x, variables.y, variables.z, variables.t),
    [variables.x, variables.y, variables.z], 
    constants, variables, domains; 
    pml_config=resolved_pml_config
) ~ -4 * pi * variables.ρ(variables.x, variables.y, variables.z, variables.t) / constants.ε

# Для компоненты векторного потенциала Ax
eq_Ax = dalembert_operator(
    variables.A[1](variables.x, variables.y, variables.z, variables.t), 
    variables.DAxDt(variables.x, variables.y, variables.z, variables.t),
    variables.DAxDx(variables.x, variables.y, variables.z, variables.t),
    variables.DAxDy(variables.x, variables.y, variables.z, variables.t),
    variables.DAxDz(variables.x, variables.y, variables.z, variables.t),
    [variables.x, variables.y, variables.z], 
    constants, variables, domains; 
    pml_config=resolved_pml_config
) ~ -constants.μ * 4 * pi / constants.c * variables.j[1](variables.x, variables.y, variables.z, variables.t)

# Аналогично для Ay и Az
```

## Преимущества новой реализации:

1. **Универсальность**: Оператор работает для любого поля (φ, Ax, Ay, Az)
2. **Чистота кода**: Не нужно дублировать логику для разных переменных
3. **Поддерживаемость**: Изменения в алгоритме затрагивают только один место
4. **Прозрачность**: Аргументы явно указывают, какие производные используются

## Обновленные требования к VariableSet:

Теперь `VariableSet` должен содержать для каждой компоненты (φ, Ax, Ay, Az):
- Само поле: `φ`, `A[1]`, `A[2]`, `A[3]`
- Производные: `DφDt`, `DφDx`, `DφDy`, `DφDz`, `DAxDt`, `DAxDx`, `DAxDy`, `DAxDz`, etc.

## Связывающие уравнения для производных

Для каждой переменной нужно добавить уравнения связывания:
```julia
# Для φ
Differential(variables.t)(variables.φ) ~ variables.DφDt
Differential(variables.x)(variables.φ) ~ variables.DφDx
Differential(variables.y)(variables.φ) ~ variables.DφDy
Differential(variables.z)(variables.φ) ~ variables.DφDz

# Для Ax
Differential(variables.t)(variables.A[1]) ~ variables.DAxDt
Differential(variables.x)(variables.A[1]) ~ variables.DAxDx
Differential(variables.y)(variables.A[1]) ~ variables.DAxDy
Differential(variables.z)(variables.A[1]) ~ variables.DAxDz

# Аналогично для Ay и Az
```

Эти уравнения обеспечивают согласованность между предсказанными производными и фактическими значениями через autodiff.
