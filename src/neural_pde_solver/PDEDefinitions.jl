"""
    PDEDefinitions

Модуль содержит определения уравнений в частных производных для решения 
обратной задачи ЭЭГ, включая операторы Лапласа и Даламбера, а также 
физические константы.

Основные компоненты:
- Оператор Лапласа
- Оператор Даламбера с PML через затухание и экранирование
- Физические константы (скорость света, диэлектрическая проницаемость и т.д.)
- Определения переменных (скалярный потенциал, векторный потенциал, плотность заряда, плотность тока)
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
struct VariableSet
    x::Any
    y::Any
    z::Any
    t::Any
    φ::Any
    A::Vector{Any}
    ρ::Any
    j::Vector{Any}
    
    function VariableSet(;x=nothing, y=nothing, z=nothing, t=nothing, φ=nothing, A=nothing, ρ=nothing, j=nothing)
        if x === nothing
            @parameters x, y, z, t
            @variables φ(..), Ax(..), Ay(..), Az(..), ρ(..), jx(..), jy(..), jz(..)
            A = [Ax, Ay, Az]
            j = [jx, jy, jz]
            return new(x, y, z, t, φ, A, ρ, j)
        else
            return new(x, y, z, t, φ, A, ρ, j)
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
    c::Float64      # Скорость света (безразмерная, c = 1)
    ε₀::Float64     # Диэлектрическая постоянная вакуума (безразмерная)
    ε::Float64      # Диэлектрическая проницаемость (отн.)
    μ₀::Float64     # Магнитная постоянная вакуума (безразмерная)
    μ::Float64      # Магнитная проницаемость (отн.)
    
    # Конструктор с параметрами по умолчанию (c = 1 для безразмерных единиц)
    function PhysicalConstants(; c=1.0, ε₀=1.0, ε=1.0, μ₀=1.0, μ=1.0)
        return new(c, ε₀, ε, μ₀, μ)
    end
end

"""
    create_variables()

Создает символьные переменные для PDE системы:
- φ(..) - скалярный потенциал
- Ax(..), Ay(..), Az(..) - компоненты векторного потенциала
- ρ(..) - плотность заряда
- jx(..), jy(..), jz(..) - компоненты плотности тока
- x, y, z, t - пространственные и временная координаты

Примечание: Новый подход PML не требует дополнительных ψ-переменных.
"""
function create_variables()
    return VariableSet()
end

"""
    laplacian(F, params)

Вычисляет оператор Лапласа для функции F по заданным параметрам params.
"""
function laplacian(F, params)
    return sum((Differential(param)^2)(F) for param in params)
end

"""
    dalembert_operator(F, params, constants, variables, domains; pml_config)

Вычисляет оператор Даламбера для функции F с PML через затухание и экранирование.

Модифицированное волновое уравнение:
    ΔF - (ε * μ / c²) * ∂²F/∂t² - γ(r) * ∂F/∂t - α(r) * F

где:
- γ(r) — динамическое затухание (вязкое трение)
- α(r) — статический экранирующий член

При отключённом PML (pml_config.enabled = false):
    ΔF - (ε * μ / c²) * ∂²F/∂t²
"""
function dalembert_operator(F, params, constants::PhysicalConstants, variables::VariableSet, domains; 
                            pml_config::PMLConfig=default_pml_config())
    # Базовый оператор Даламбера
    Δ = laplacian(F, params)
    d2F_dt2 = (Differential(variables.t)^2)(F)
    
    if !pml_config.enabled
        # Без PML - обычный оператор Даламбера
        return Δ - (constants.ε * constants.μ / constants.c^2) * d2F_dt2
    end
    
    # С PML - добавляем члены затухания и экранирования
    # Вычисляем γ и α для текущих координат
    γ = compute_gamma(pml_config, variables.x, variables.y, variables.z, domains)
    α = compute_alpha(pml_config, variables.x, variables.y, variables.z, domains)
    
    # Временная производная для затухания
    dF_dt = Differential(variables.t)(F)
    
    # Модифицированный оператор: ΔF - (1/c²)∂²F/∂t² - γ∂F/∂t - αF
    return Δ - (constants.ε * constants.μ / constants.c^2) * d2F_dt2 - γ * dF_dt - α * F
end

"""
    create_pde_system(constants, variables, bcs, domains; pml_config)

Создает систему PDE уравнений для обратной задачи ЭЭГ с поддержкой PML.

Уравнения:
1. □φ = -4πρ/ε (скалярный потенциал)
2. □A_i = -4πμj_i/c (компоненты векторного потенциала)
3. ∇·A + (εμ/c)∂φ/∂t = 0 (калибровка Лоренца)

PML добавляет члены затухания и экранирования в оператор Даламбера.

Автоматически вычисляет параметры PML (gamma_max, alpha_max), если они не заданы.
"""
function create_pde_system(constants::PhysicalConstants, variables::VariableSet, bcs, domains; pml_config::PMLConfig=default_pml_config())
    # Разрешаем параметры PML (вычисляем автоматические значения, если нужно)
    resolved_pml_config = resolve_pml_config(pml_config, domains, constants.c)
    
    # Уравнение для скалярного потенциала φ
    eq_φ = dalembert_operator(
        variables.φ(variables.x, variables.y, variables.z, variables.t), 
        [variables.x, variables.y, variables.z], 
        constants, variables, domains; 
        pml_config=resolved_pml_config
    ) ~ -4 * pi * variables.ρ(variables.x, variables.y, variables.z, variables.t) / constants.ε
    
    # Уравнения для компонент векторного потенциала A
    eq_A = [
        dalembert_operator(
            variables.A[i](variables.x, variables.y, variables.z, variables.t), 
            [variables.x, variables.y, variables.z], 
            constants, variables, domains; 
            pml_config=resolved_pml_config
        ) ~ -constants.μ * 4 * pi / constants.c * variables.j[i](variables.x, variables.y, variables.z, variables.t)
        for i in 1:3
    ]
    
    # Условие калибровки Лоренца
    eq_gauge = (Differential(variables.x)(variables.A[1](variables.x, variables.y, variables.z, variables.t)) + 
                Differential(variables.y)(variables.A[2](variables.x, variables.y, variables.z, variables.t)) + 
                Differential(variables.z)(variables.A[3](variables.x, variables.y, variables.z, variables.t)) + 
                (constants.ε * constants.μ / constants.c) * Differential(variables.t)(variables.φ(variables.x, variables.y, variables.z, variables.t))) ~ 0.0
    
    # Объединяем все уравнения (только 5 уравнений, без ψ-уравнений)
    eq = [eq_φ; eq_A; eq_gauge]
    
    # Переменные системы (только 8 переменных, без ψ-полей)
    allvars = [
        variables.φ(variables.x, variables.y, variables.z, variables.t); 
        [A_(variables.x, variables.y, variables.z, variables.t) for A_ in variables.A]; 
        variables.ρ(variables.x, variables.y, variables.z, variables.t); 
        [j_(variables.x, variables.y, variables.z, variables.t) for j_ in variables.j]
    ]
    
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
- φ = 0 на всех границах
- A_i = 0 на всех границах

Примечание: Новый подход PML не требует граничных условий для ψ-полей.

Автоматически вычисляет параметры PML (gamma_max, alpha_max), если они не заданы.
"""
function create_boundary_conditions(constants::PhysicalConstants, variables::VariableSet, domains; pml_config::PMLConfig=default_pml_config())
    # Разрешаем параметры PML (вычисляем автоматические значения, если нужно)
    resolved_pml_config = resolve_pml_config(pml_config, domains, constants.c)
    
    x_min, x_max = infimum(domains[1].domain), supremum(domains[1].domain)
    y_min, y_max = infimum(domains[2].domain), supremum(domains[2].domain)
    z_min, z_max = infimum(domains[3].domain), supremum(domains[3].domain)
    
    # Граничные условия только для φ и A (без ψ-полей)
    bcs = [
        # Граничные условия для φ
        variables.φ(x_min, variables.y, variables.z, variables.t) ~ 0.0,
        variables.φ(x_max, variables.y, variables.z, variables.t) ~ 0.0,
        variables.φ(variables.x, y_min, variables.z, variables.t) ~ 0.0,
        variables.φ(variables.x, y_max, variables.z, variables.t) ~ 0.0,
        variables.φ(variables.x, variables.y, z_min, variables.t) ~ 0.0,
        variables.φ(variables.x, variables.y, z_max, variables.t) ~ 0.0,
        # Граничные условия для A_x
        variables.A[1](x_min, variables.y, variables.z, variables.t) ~ 0.0,
        variables.A[1](x_max, variables.y, variables.z, variables.t) ~ 0.0,
        variables.A[1](variables.x, y_min, variables.z, variables.t) ~ 0.0,
        variables.A[1](variables.x, y_max, variables.z, variables.t) ~ 0.0,
        variables.A[1](variables.x, variables.y, z_min, variables.t) ~ 0.0,
        variables.A[1](variables.x, variables.y, z_max, variables.t) ~ 0.0,
        # Граничные условия для A_y
        variables.A[2](x_min, variables.y, variables.z, variables.t) ~ 0.0,
        variables.A[2](x_max, variables.y, variables.z, variables.t) ~ 0.0,
        variables.A[2](variables.x, y_min, variables.z, variables.t) ~ 0.0,
        variables.A[2](variables.x, y_max, variables.z, variables.t) ~ 0.0,
        variables.A[2](variables.x, variables.y, z_min, variables.t) ~ 0.0,
        variables.A[2](variables.x, variables.y, z_max, variables.t) ~ 0.0,
        # Граничные условия для A_z
        variables.A[3](x_min, variables.y, variables.z, variables.t) ~ 0.0,
        variables.A[3](x_max, variables.y, variables.z, variables.t) ~ 0.0,
        variables.A[3](variables.x, y_min, variables.z, variables.t) ~ 0.0,
        variables.A[3](variables.x, y_max, variables.z, variables.t) ~ 0.0,
        variables.A[3](variables.x, variables.y, z_min, variables.t) ~ 0.0,
        variables.A[3](variables.x, variables.y, z_max, variables.t) ~ 0.0,
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
