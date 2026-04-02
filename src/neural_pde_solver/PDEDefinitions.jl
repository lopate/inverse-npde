"""
    PDEDefinitions

Модуль содержит определения уравнений в частных производных для решения 
обратной задачи ЭЭГ, включая операторы Лапласа и Даламбера, а также 
физические константы.

Основные компоненты:
- Оператор Лапласа
- Оператор Даламбера с PML через затухание и экранирование
- Физические константы (скорость света, диэлектрическая проницаемость и т.д.)
- Определения переменных (скалярный потенциал, векторный потенциал, вектор поляризации)
- Граничные условия и области определения
- Поддержка PML (Perfectly Matched Layer) через γ и α

Новый подход PML:
Модифицированное волновое уравнение:
    ∇²u - (1/c²)∂²u/∂t² - γ(r)∂u/∂t - α(r)u = S

где:
- γ(r) — динамическое затухание (вязкое трение), поглощает энергию волн
- α(r) — статический экранирующий член, обеспечивает экспоненциальное убывание потенциала
"""

module PDEDefinitions

using ModelingToolkit: @parameters, @variables, Differential, Interval, infimum, supremum
using NeuralPDE, ComponentArrays, Symbolics
using ..PML: PMLConfig, compute_gamma, compute_alpha, default_pml_config, resolve_pml_config

# Экспортируем основные типы и функции
export laplacian, dalembert_operator, create_pde_system, PhysicalConstants
export create_variables, create_domains, create_boundary_conditions
export analytic_sol_func, generate_measured_points, VariableSet
export PMLConfig, default_pml_config

# Структура для хранения переменных PDE системы (без ψ-полей)
# Теперь с поддержкой производных для улучшения сходимости
struct VariableSet
    x::Any
    y::Any
    z::Any
    t::Any
    φ::Any
    A::Vector{Any}
    P::Vector{Any}
    
    # Производные для φ: DφDt, DφDx, DφDy, DφDz
    DφDt::Any
    DφDx::Any
    DφDy::Any
    DφDz::Any
    
    # Производные для A: по 4 производные на каждую компоненту (Dt, Dx, Dy, Dz)
    DA_dt::Vector{Any}
    DA_dx::Vector{Any}
    DA_dy::Vector{Any}
    DA_dz::Vector{Any}
    
    # Производные для P: по 4 производные на каждую компоненту (Dt, Dx, Dy, Dz)
    DP_dt::Vector{Any}
    DP_dx::Vector{Any}
    DP_dy::Vector{Any}
    DP_dz::Vector{Any}
    
    # Флаг использования предсказанных производных
    use_derivatives::Bool
    
    function VariableSet(;x=nothing, y=nothing, z=nothing, t=nothing, 
                        φ=nothing, A=nothing, P=nothing,
                        use_derivatives::Bool=true)
        if x === nothing
            @parameters x, y, z, t
            @variables φ(..), Ax(..), Ay(..), Az(..), Px(..), Py(..), Pz(..)
            
            @variables DφDt(..) DφDx(..) DφDy(..) DφDz(..)
            
            @variables DAxDt(..) DAxDx(..) DAxDy(..) DAxDz(..)
            @variables DAyDt(..) DAyDx(..) DAyDy(..) DAyDz(..)
            @variables DAzDt(..) DAzDx(..) DAzDy(..) DAzDz(..)
            
            @variables DPxDt(..) DPxDx(..) DPxDy(..) DPxDz(..)
            @variables DPyDt(..) DPyDx(..) DPyDy(..) DPyDz(..)
            @variables DPzDt(..) DPzDx(..) DPzDy(..) DPzDz(..)
            
            A = [Ax, Ay, Az]
            P = [Px, Py, Pz]
            
            DA_dt = [DAxDt, DAyDt, DAzDt]
            DA_dx = [DAxDx, DAyDx, DAzDx]
            DA_dy = [DAxDy, DAyDy, DAzDy]
            DA_dz = [DAxDz, DAyDz, DAzDz]
            
            DP_dt = [DPxDt, DPyDt, DPzDt]
            DP_dx = [DPxDx, DPyDx, DPzDx]
            DP_dy = [DPxDy, DPyDy, DPzDy]
            DP_dz = [DPxDz, DPyDz, DPzDz]
            
            return new(x, y, z, t, φ, A, P,
                      DφDt, DφDx, DφDy, DφDz,
                      DA_dt, DA_dx, DA_dy, DA_dz,
                      DP_dt, DP_dx, DP_dy, DP_dz,
                      use_derivatives)
        else
            DA_dt = [nothing, nothing, nothing]
            DA_dx = [nothing, nothing, nothing]
            DA_dy = [nothing, nothing, nothing]
            DA_dz = [nothing, nothing, nothing]
            DP_dt = [nothing, nothing, nothing]
            DP_dx = [nothing, nothing, nothing]
            DP_dy = [nothing, nothing, nothing]
            DP_dz = [nothing, nothing, nothing]
            return new(x, y, z, t, φ, A, P,
                      nothing, nothing, nothing, nothing,
                      DA_dt, DA_dx, DA_dy, DA_dz,
                      DP_dt, DP_dx, DP_dy, DP_dz,
                      use_derivatives)
        end
    end
end

# Определение структуры для физических констант
"""
    PhysicalConstants

Структура для хранения физических констант.

ВАЖНО: Используем безразмерные единицы с c = 1!
Это обеспечивает численную устойчивость и упрощает уравнения.

При c = 1:
- Волновое уравнение: ∇²u - ∂²u/∂t² = S
- Все величины нормированы на характерные масштабы задачи
"""
struct PhysicalConstants
    c::Float32      # Скорость света (безразмерная, c = 1)
    ε₀::Float32     # Диэлектрическая постоянная вакуума (безразмерная)
    ε::Float32      # Диэлектрическая проницаемость (отн.)
    μ₀::Float32     # Магнитная постоянная вакуума (безразмерная)
    μ::Float32      # Магнитная проницаемость (отн.)
    
    # Конструктор с параметрами по умолчанию (c = 1 для безразмерных единиц)
    function PhysicalConstants(; c=1.0f0, ε₀=1.0f0, ε=1.0f0, μ₀=1.0f0, μ=1.0f0)
        return new(c, ε₀, ε, μ₀, μ)
    end
end

"""
    create_variables(; use_derivatives::Bool=true)

Создает символьные переменные для PDE системы:
- φ(..) - скалярный потенциал
- Ax(..), Ay(..), Az(..) - компоненты векторного потенциала
- Px(..), Py(..), Pz(..) - компоненты вектора поляризации
- DφDt, DφDx, DφDy, DφDz - производные потенциала (если use_derivatives=true)
- DAxDt, DAxDx, DAxDy, DAxDz и т.д. - производные векторного потенциала
- DPxDt, DPxDx, DPxDy, DPxDz и т.д. - производные вектора поляризации
- x, y, z, t - пространственные и временная координаты

# Аргументы
- `use_derivatives::Bool`: включить производные (по умолчанию true)

Примечание: Новый подход PML не требует дополнительных ψ-переменных.
"""
function create_variables(; use_derivatives::Bool=true)
    return VariableSet(use_derivatives=use_derivatives)
end

"""
    laplacian(F, params)

Вычисляет оператор Лапласа для функции F по заданным параметрам params.
"""
function laplacian(F, params)
    return sum((Differential(param)^2)(F) for param in params)
end

"""
    dalembert_operator(F, dF_dt_pred, dF_dx_pred, dF_dy_pred, dF_dz_pred, 
                      params, constants::PhysicalConstants, variables::VariableSet, domains; 
                      pml_config::PMLConfig=default_pml_config())

Универсальный оператор Даламбера с поддержкой предсказанных производных.

Модифицированное волновое уравнение:
    ΔF - (ε * μ / c²) * ∂²F/∂t² - γ(r) * ∂F/∂t - α(r) * F

Ключевое отличие от предыдущей версии:
- Вторые производные вычисляются через дифференцирование предсказанных первых производных
- Для PML члена γ*∂F/∂t используется autodiff на F (критично для корректного затухания)

# Аргументы
- `F`: символьная функция (φ или A_i)
- `dF_dt_pred`, `dF_dx_pred`, `dF_dy_pred`, `dF_dz_pred`: предсказанные первые производные
- `params`: пространственные параметры [x, y, z]
- `constants::PhysicalConstants`: физические константы
- `variables::VariableSet`: набор переменных
- `domains`: области определения
- `pml_config::PMLConfig`: конфигурация PML

При отключённом PML (pml_config.enabled = false):
    ΔF - (ε * μ / c²) * ∂²F/∂t²
"""
function dalembert_operator(F, dF_dt_pred, dF_dx_pred, dF_dy_pred, dF_dz_pred, 
                           params, constants::PhysicalConstants, variables::VariableSet, domains; 
                           pml_config::PMLConfig=default_pml_config())
    # Применяем дифференцирование к предсказанным производным для получения вторых производных
    d2F_dt2 = Differential(variables.t)(dF_dt_pred)  # ∂(∂F/∂t)/∂t
    d2F_dx2 = Differential(variables.x)(dF_dx_pred)  # ∂(∂F/∂x)/∂x
    d2F_dy2 = Differential(variables.y)(dF_dy_pred)  # ∂(∂F/∂y)/∂y
    d2F_dz2 = Differential(variables.z)(dF_dz_pred)  # ∂(∂F/∂z)/∂z
    
    Δ = d2F_dx2 + d2F_dy2 + d2F_dz2
    
    if !pml_config.enabled
        # Без PML - обычный оператор Даламбера
        return Δ - (constants.ε * constants.μ / constants.c^2) * d2F_dt2
    end
    
    # С PML - добавляем члены затухания и экранирования
    γ = compute_gamma(pml_config, variables.x, variables.y, variables.z, domains)
    α = compute_alpha(pml_config, variables.x, variables.y, variables.z, domains)
    
    # Для PML используем autodiff на F (критично для корректного затухания)
    dF_dt_autodiff = Differential(variables.t)(F)  # ∂F/∂t через autodiff
    
    return Δ - (constants.ε * constants.μ / constants.c^2) * d2F_dt2 - γ * dF_dt_autodiff - α * F
end

"""
    dalembert_operator(F, params, constants, variables, domains; pml_config)

Устаревшая версия оператора Даламбера без предсказанных производных.
Использует автоматическое дифференцирование для вторых производных.

Оставлена для обратной совместимости.
"""
function dalembert_operator(F, params, constants::PhysicalConstants, variables::VariableSet, domains; 
                            pml_config::PMLConfig=default_pml_config())
    # Базовый оператор Даламбера - используем автоматическое дифференцирование
    Δ = laplacian(F, params)
    d2F_dt2 = (Differential(variables.t)^2)(F)
    
    if !pml_config.enabled
        # Без PML - обычный оператор Даламбера
        return Δ - (constants.ε * constants.μ / constants.c^2) * d2F_dt2
    end
    
    # С PML - добавляем члены затухания и экранирования
    γ = compute_gamma(pml_config, variables.x, variables.y, variables.z, domains)
    α = compute_alpha(pml_config, variables.x, variables.y, variables.z, domains)
    
    # Временная производная для затухания
    dF_dt = Differential(variables.t)(F)
    
    # Модифицированный оператор: ΔF - (1/c²)∂²F/∂t² - γ∂F/∂t - αF
    return Δ - (constants.ε * constants.μ / constants.c^2) * d2F_dt2 - γ * dF_dt - α * F
end

"""
    create_derivative_equations(variables::VariableSet)

Создает уравнения связывания между производными и их символьными производными.
Эти уравнения обеспечивают согласованность между предсказанными производными
и фактическими значениями через autodiff.

Уравнения связывания:
- Dt(φ) ~ DφDt
- Dx(φ) ~ DφDx  
- Dy(φ) ~ DφDy
- Dz(φ) ~ DφDz
- Dt(Ax) ~ DAxDt
- Dx(Ax) ~ DAxDx
- и т.д.
"""
function create_derivative_equations(variables::VariableSet)
    coords = (variables.x, variables.y, variables.z, variables.t)
    
    der_eqs = []
    
    push!(der_eqs, Differential(variables.t)(variables.φ(coords...)) ~ variables.DφDt(coords...))
    push!(der_eqs, Differential(variables.x)(variables.φ(coords...)) ~ variables.DφDx(coords...))
    push!(der_eqs, Differential(variables.y)(variables.φ(coords...)) ~ variables.DφDy(coords...))
    push!(der_eqs, Differential(variables.z)(variables.φ(coords...)) ~ variables.DφDz(coords...))
    
    for i in 1:3
        push!(der_eqs, Differential(variables.t)(variables.A[i](coords...)) ~ variables.DA_dt[i](coords...))
        push!(der_eqs, Differential(variables.x)(variables.A[i](coords...)) ~ variables.DA_dx[i](coords...))
        push!(der_eqs, Differential(variables.y)(variables.A[i](coords...)) ~ variables.DA_dy[i](coords...))
        push!(der_eqs, Differential(variables.z)(variables.A[i](coords...)) ~ variables.DA_dz[i](coords...))
    end
    
    for i in 1:3
        push!(der_eqs, Differential(variables.t)(variables.P[i](coords...)) ~ variables.DP_dt[i](coords...))
        push!(der_eqs, Differential(variables.x)(variables.P[i](coords...)) ~ variables.DP_dx[i](coords...))
        push!(der_eqs, Differential(variables.y)(variables.P[i](coords...)) ~ variables.DP_dy[i](coords...))
        push!(der_eqs, Differential(variables.z)(variables.P[i](coords...)) ~ variables.DP_dz[i](coords...))
    end
    
    return der_eqs
end

"""
    create_pde_system(constants, variables, bcs, domains; pml_config)

Создает систему PDE уравнений для обратной задачи ЭЭГ с поддержкой PML
и опциональным использованием предсказанных производных.

Уравнения:
1. □φ = 4π·div P / ε = 4π·(∂Px/∂x + ∂Py/∂y + ∂Pz/∂z) / ε (скалярный потенциал)
2. □A_i = -4πμ/c · ∂P_i/∂t (компоненты векторного потенциала)
3. ∇·A + (εμ/c)∂φ/∂t = 0 (калибровка Лоренца)
4. Уравнения связывания производных (der_) если use_derivatives=true

PML добавляет члены затухания и экранирования в оператор Даламбера.

Автоматически вычисляет параметры PML (gamma_max, alpha_max), если они не заданы.

# Аргументы
- `constants::PhysicalConstants`: физические константы
- `variables::VariableSet`: набор переменных
- `bcs`: граничные условия
- `domains`: области определения
- `pml_config::PMLConfig`: конфигурация PML
- `include_derivatives::Bool`: включить предсказанные производные (по умолчанию true)
"""
function create_pde_system(constants::PhysicalConstants, variables::VariableSet, bcs, domains; 
                           pml_config::PMLConfig=default_pml_config(),
                           include_derivatives::Bool=true)
    resolved_pml_config = resolve_pml_config(pml_config, domains, constants.c)
    
    coords = (variables.x, variables.y, variables.z, variables.t)
    
    if include_derivatives && variables.use_derivatives
        eq_φ = dalembert_operator(
            variables.φ(coords...),
            variables.DφDt(coords...),
            variables.DφDx(coords...),
            variables.DφDy(coords...),
            variables.DφDz(coords...),
            [variables.x, variables.y, variables.z],
            constants, variables, domains;
            pml_config=resolved_pml_config
        ) ~ 4 * pi * (
            Differential(variables.x)(variables.P[1](coords...)) +
            Differential(variables.y)(variables.P[2](coords...)) +
            Differential(variables.z)(variables.P[3](coords...))
        ) / constants.ε
    else
        eq_φ = dalembert_operator(
            variables.φ(coords...),
            [variables.x, variables.y, variables.z],
            constants, variables, domains;
            pml_config=resolved_pml_config
        ) ~ 4 * pi * (
            Differential(variables.x)(variables.P[1](coords...)) +
            Differential(variables.y)(variables.P[2](coords...)) +
            Differential(variables.z)(variables.P[3](coords...))
        ) / constants.ε
    end
    
    eq_A = []
    for i in 1:3
        if include_derivatives && variables.use_derivatives
            push!(eq_A, dalembert_operator(
                variables.A[i](coords...),
                variables.DA_dt[i](coords...),
                variables.DA_dx[i](coords...),
                variables.DA_dy[i](coords...),
                variables.DA_dz[i](coords...),
                [variables.x, variables.y, variables.z],
                constants, variables, domains;
                pml_config=resolved_pml_config
            ) ~ -constants.μ * 4 * pi / constants.c * variables.DP_dt[i](coords...))
        else
            push!(eq_A, dalembert_operator(
                variables.A[i](coords...),
                [variables.x, variables.y, variables.z],
                constants, variables, domains;
                pml_config=resolved_pml_config
            ) ~ -constants.μ * 4 * pi / constants.c * Differential(variables.t)(variables.P[i](coords...)))
        end
    end
    
    eq_gauge = (Differential(variables.x)(variables.A[1](coords...)) + 
                Differential(variables.y)(variables.A[2](coords...)) + 
                Differential(variables.z)(variables.A[3](coords...)) + 
                (constants.ε * constants.μ / constants.c) * Differential(variables.t)(variables.φ(coords...))) ~ 0.0
    
    eq = [eq_φ; eq_A; eq_gauge]
    
    if include_derivatives && variables.use_derivatives
        derivs_phi = [variables.DφDt(coords...), variables.DφDx(coords...), variables.DφDy(coords...), variables.DφDz(coords...)]
        derivs_A = vcat(
            [variables.DA_dt[i](coords...) for i in 1:3],
            [variables.DA_dx[i](coords...) for i in 1:3],
            [variables.DA_dy[i](coords...) for i in 1:3],
            [variables.DA_dz[i](coords...) for i in 1:3]
        )
        derivs_P = vcat(
            [variables.DP_dt[i](coords...) for i in 1:3],
            [variables.DP_dx[i](coords...) for i in 1:3],
            [variables.DP_dy[i](coords...) for i in 1:3],
            [variables.DP_dz[i](coords...) for i in 1:3]
        )
        allvars = vcat(
            [variables.φ(coords...)],
            [A_(coords...) for A_ in variables.A],
            [P_(coords...) for P_ in variables.P],
            derivs_phi,
            derivs_A,
            derivs_P
        )
    else
        allvars = vcat(
            [variables.φ(coords...)],
            [A_(coords...) for A_ in variables.A],
            [P_(coords...) for P_ in variables.P]
        )
    end
    
    @named pde_system = PDESystem(eq, bcs, domains, [variables.x, variables.y, variables.z, variables.t], allvars)
    return pde_system
end

"""
    create_domains(x_range, y_range, z_range, t_range)

Создает области определения для PDE системы.
"""
function create_domains(variables::VariableSet, x_range, y_range, z_range, t_range)
    domains = [variables.x ∈ Interval(x_range[1], x_range[2]),
               variables.y ∈ Interval(y_range[1], y_range[2]), 
               variables.z ∈ Interval(z_range[1], z_range[2]),
               variables.t ∈ Interval(t_range[1], t_range[2])]
    return domains
end

"""
    create_boundary_conditions(constants, variables, domains; pml_config)

Создает граничные условия для PDE системы.

Граничные условия:
- φ = 0 на всех пространственных границах
- A_i = 0 на всех пространственных границах
- φ = 0 при t = t_min (начальный момент)
- φ = 0 при t = t_max (конечный момент)
- A_i = 0 при t = t_min (начальный момент)
- A_i = 0 при t = t_max (конечный момент)

Примечание: Новый подход PML не требует граничных условий для ψ-полей.

Автоматически вычисляет параметры PML (gamma_max, alpha_max), если они не заданы.
"""
function create_boundary_conditions(constants::PhysicalConstants, variables::VariableSet, domains; pml_config::PMLConfig=default_pml_config())
    # Разрешаем параметры PML (вычисляем автоматические значения, если нужно)
    resolved_pml_config = resolve_pml_config(pml_config, domains, constants.c)
    
    x_min, x_max = infimum(domains[1].domain), supremum(domains[1].domain)
    y_min, y_max = infimum(domains[2].domain), supremum(domains[2].domain)
    z_min, z_max = infimum(domains[3].domain), supremum(domains[3].domain)
    t_min, t_max = infimum(domains[4].domain), supremum(domains[4].domain)
    
    # Граничные условия только для φ и A (без ψ-полей)
    bcs = [
        # Граничные условия для φ - пространственные границы
        variables.φ(x_min, variables.y, variables.z, variables.t) ~ 0.0f0,
        variables.φ(x_max, variables.y, variables.z, variables.t) ~ 0.0f0,
        variables.φ(variables.x, y_min, variables.z, variables.t) ~ 0.0f0,
        variables.φ(variables.x, y_max, variables.z, variables.t) ~ 0.0f0,
        variables.φ(variables.x, variables.y, z_min, variables.t) ~ 0.0f0,
        variables.φ(variables.x, variables.y, z_max, variables.t) ~ 0.0f0,
        # Граничные условия для φ - временные границы (t = t_min и t = t_max)
        variables.φ(variables.x, variables.y, variables.z, t_min) ~ 0.0f0,
        variables.φ(variables.x, variables.y, variables.z, t_max) ~ 0.0f0,
        # Граничные условия для A_x
        variables.A[1](x_min, variables.y, variables.z, variables.t) ~ 0.0f0,
        variables.A[1](x_max, variables.y, variables.z, variables.t) ~ 0.0f0,
        variables.A[1](variables.x, y_min, variables.z, variables.t) ~ 0.0f0,
        variables.A[1](variables.x, y_max, variables.z, variables.t) ~ 0.0f0,
        variables.A[1](variables.x, variables.y, z_min, variables.t) ~ 0.0f0,
        variables.A[1](variables.x, variables.y, z_max, variables.t) ~ 0.0f0,
        # Граничные условия для A_x - временные границы (t = t_min и t = t_max)
        variables.A[1](variables.x, variables.y, variables.z, t_min) ~ 0.0f0,
        variables.A[1](variables.x, variables.y, variables.z, t_max) ~ 0.0f0,
        # Граничные условия для A_y
        variables.A[2](x_min, variables.y, variables.z, variables.t) ~ 0.0f0,
        variables.A[2](x_max, variables.y, variables.z, variables.t) ~ 0.0f0,
        variables.A[2](variables.x, y_min, variables.z, variables.t) ~ 0.0f0,
        variables.A[2](variables.x, y_max, variables.z, variables.t) ~ 0.0f0,
        variables.A[2](variables.x, variables.y, z_min, variables.t) ~ 0.0f0,
        variables.A[2](variables.x, variables.y, z_max, variables.t) ~ 0.0f0,
        # Граничные условия для A_y - временные границы (t = t_min и t = t_max)
        variables.A[2](variables.x, variables.y, variables.z, t_min) ~ 0.0f0,
        variables.A[2](variables.x, variables.y, variables.z, t_max) ~ 0.0f0,
        # Граничные условия для A_z
        variables.A[3](x_min, variables.y, variables.z, variables.t) ~ 0.0f0,
        variables.A[3](x_max, variables.y, variables.z, variables.t) ~ 0.0f0,
        variables.A[3](variables.x, y_min, variables.z, variables.t) ~ 0.0f0,
        variables.A[3](variables.x, y_max, variables.z, variables.t) ~ 0.0f0,
        variables.A[3](variables.x, variables.y, z_min, variables.t) ~ 0.0f0,
        variables.A[3](variables.x, variables.y, z_max, variables.t) ~ 0.0f0,
        # Граничные условия для A_z - временные границы (t = t_min и t = t_max)
        variables.A[3](variables.x, variables.y, variables.z, t_min) ~ 0.0f0,
        variables.A[3](variables.x, variables.y, variables.z, t_max) ~ 0.0f0,
    ]

    return bcs
end

"""
    analytic_sol_func(t, x, y, z)

Аналитическое решение для тестирования.
"""
function analytic_sol_func(t, x, y, z)
    r = sqrt(x^2 + y^2 + z^2)
    return (t + 1)^2 / r
end

"""
    generate_measured_points(domains, num_points, constants, analytic_func)

Генерирует измеренные точки в области определения.
"""
function generate_measured_points(domains, num_points, constants::PhysicalConstants, analytic_func)
    x_min, x_max = infimum(domains[1].domain), supremum(domains[1].domain)
    y_min, y_max = infimum(domains[2].domain), supremum(domains[2].domain)
    z_min, z_max = infimum(domains[3].domain), supremum(domains[3].domain)
    t_min, t_max = infimum(domains[4].domain), supremum(domains[4].domain)
    
    measured_points = []
    
    # Случайные точки внутри домена
    for _ in 1:num_points
        x_p = rand(x_min/2:x_max/2)
        y_p = rand(y_min/2:y_max/2)
        z_p = rand(z_min/2:z_max/2)
        t_p = rand(t_min/2:t_max/2)
        
        phi_p = analytic_func(t_p, x_p, y_p, z_p)
        push!(measured_points, [x_p, y_p, z_p, t_p, phi_p])
    end
    
    # Точки на границе t = t_min
    for _ in 1:num_points
        x_p = rand(x_min/2:x_max/2)
        y_p = rand(y_min/2:y_max/2)
        z_p = rand(z_min/2:z_max/2)
        t_p = t_min
        
        phi_p = analytic_func(t_p, x_p, y_p, z_p)
        push!(measured_points, [x_p, y_p, z_p, t_p, phi_p])
    end
    
    return measured_points
end

end # module
