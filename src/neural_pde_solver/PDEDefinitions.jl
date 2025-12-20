"""
    PDEDefinitions

Модуль содержит определения уравнений в частных производных для решения 
обратной задачи ЭЭГ, включая операторы Лапласа и Даламбера, а также 
физические константы.

Основные компоненты:
- Оператор Лапласа
- Оператор Даламбера  
- Физические константы (скорость света, диэлектрическая проницаемость и т.д.)
- Определения переменных (скалярный потенциал, векторный потенциал, плотность заряда, плотность тока)
- Граничные условия и области определения
"""

module PDEDefinitions

using ModelingToolkit: @parameters, @variables, Differential, Interval, infimum, supremum
using NeuralPDE, ComponentArrays, Symbolics

# Экспортируем основные типы и функции
export laplacian, dalembert_operator, create_pde_system, PhysicalConstants
export create_variables, create_domains, create_boundary_conditions
export analytic_sol_func, generate_measured_points, VariableSet

# Структура для хранения переменных PDE системы
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
            @variables φ(..), Ax(..),Ay(..),Az(..), ρ(..), jx(..), jy(..), jz(..)
            A = [Ax, Ay, Az]
            j = [jx, jy, jz]
            return new(x, y, z, t, φ, A, ρ, j)
        else
            return new(x, y, z, t, φ, A, ρ, j)
        end
    end
end

# Определение структуры для физических констант
struct PhysicalConstants
    c::Float64      # Скорость света в вакууме (см/с)
    ε₀::Float64     # Диэлектрическая постоянная вакуума в СГС
    ε::Float64      # Диэлектрическая проницаемость (отн.)
    μ₀::Float64     # Магнитная постоянная вакуума в СГС
    μ::Float64      # Магнитная проницаемость (отн.)
    
    # Конструктор с параметрами по умолчанию
    function PhysicalConstants(; c=2.99792458e10, ε₀=1.0, ε=1.0, μ₀=1.0, μ=1.0)
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
    dalembert_operator(F, params, constants)

Вычисляет оператор Даламбера для функции F.
ΔF - (ε * μ / c²) * ∂²F/∂t²
"""
function dalembert_operator(F, params, constants::PhysicalConstants, variables::VariableSet)
    Δ = laplacian(F, params)
    return Δ - (constants.ε * constants.μ / constants.c^2) * (Differential(variables.t)^2)(F)
end

"""
    create_pde_system(constants, domains)

Создает систему PDE уравнений для обратной задачи ЭЭГ.
"""
function create_pde_system(constants::PhysicalConstants, variables::VariableSet, bcs, domains)
    # Уравнение для скалярного потенциала
    eq = [
    dalembert_operator(variables.φ(variables.x, variables.y, variables.z, variables.t), [variables.x, variables.y, variables.z], constants, variables) ~ -4 * π * variables.ρ(variables.x, variables.y, variables.z, variables.t) / constants.ε;
    # Уравнения для векторного потенциала
    [dalembert_operator(variables.A[i](variables.x, variables.y, variables.z, variables.t), [variables.x, variables.y, variables.z], constants, variables) ~ -constants.μ * 4 * π / constants.c * variables.j[i](variables.x, variables.y, variables.z, variables.t) for i in 1:3];
    # Условие калибровки (условие Лоренца)
    (Differential(variables.x)(variables.A[1](variables.x, variables.y, variables.z, variables.t)) + Differential(variables.y)(variables.A[2](variables.x, variables.y, variables.z, variables.t)) + 
                        Differential(variables.z)(variables.A[3](variables.x, variables.y, variables.z, variables.t)) + (constants.ε * constants.μ / constants.c) * Differential(variables.t)(variables.φ(variables.x, variables.y, variables.z, variables.t))) ~ 0.0
    ]
    allvars = [
        variables.φ(variables.x, variables.y, variables.z, variables.t); 
        [A_(variables.x, variables.y, variables.z, variables.t) for A_ in variables.A]; 
        variables.ρ(variables.x, variables.y, variables.z, variables.t); 
        [j_(variables.x, variables.y, variables.z, variables.t) for j_ in variables.j]]
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
    create_boundary_conditions(constants, variables, domains)

Создает граничные условия для PDE системы.
"""
function create_boundary_conditions(constants::PhysicalConstants, variables::VariableSet, domains)
    x_min, x_max = infimum(domains[1].domain), supremum(domains[1].domain)
    y_min, y_max = infimum(domains[2].domain), supremum(domains[2].domain)
    z_min, z_max = infimum(domains[3].domain), supremum(domains[3].domain)
    
    bcs = [
    [variables.φ(x_min, variables.y, variables.z, variables.t) ~ 0.0, variables.φ(x_max, variables.y, variables.z, variables.t) ~ 0.0,
    variables.φ(variables.x, y_min, variables.z, variables.t) ~ 0.0, variables.φ(variables.x, y_max, variables.z, variables.t) ~ 0.0,
    variables.φ(variables.x, variables.y, z_min, variables.t) ~ 0.0, variables.φ(variables.x, variables.y, z_max, variables.t) ~ 0.0];
    [variables.A[i](x_min, variables.y, variables.z, variables.t)  ~ 0.0 for i in 1:3]
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