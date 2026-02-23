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
- Поддержка PML (Perfectly Matched Layer)
"""

module PDEDefinitions

using ModelingToolkit: @parameters, @variables, Differential, Interval, infimum, supremum
using NeuralPDE, ComponentArrays, Symbolics
using ..PML: PMLConfig, create_pml_variables, compute_pml_profiles, apply_pml_to_dalembert, create_pml_auxiliary_equations, create_pml_boundary_conditions, default_pml_config, resolve_pml_config

# Экспортируем основные типы и функции
export laplacian, dalembert_operator, create_pde_system, PhysicalConstants
export create_variables, create_domains, create_boundary_conditions
export analytic_sol_func, generate_measured_points, VariableSet
export PMLConfig, default_pml_config

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
    # PML variables
    ψ_φx::Any
    ψ_φy::Any
    ψ_φz::Any
    ψ_Axx::Any
    ψ_Axy::Any
    ψ_Axz::Any
    ψ_Ayx::Any
    ψ_Ayy::Any
    ψ_Ayz::Any
    ψ_Azx::Any
    ψ_Azy::Any
    ψ_Azz::Any
    
    function VariableSet(;x=nothing, y=nothing, z=nothing, t=nothing, φ=nothing, A=nothing, ρ=nothing, j=nothing,
                          ψ_φx=nothing, ψ_φy=nothing, ψ_φz=nothing,
                          ψ_Axx=nothing, ψ_Axy=nothing, ψ_Axz=nothing,
                          ψ_Ayx=nothing, ψ_Ayy=nothing, ψ_Ayz=nothing,
                          ψ_Azx=nothing, ψ_Azy=nothing, ψ_Azz=nothing)
        if x === nothing
            @parameters x, y, z, t
            @variables φ(..), Ax(..),Ay(..),Az(..), ρ(..), jx(..), jy(..), jz(..)
            A = [Ax, Ay, Az]
            j = [jx, jy, jz]
            # Create PML variables
            ψ_φx, ψ_φy, ψ_φz, ψ_Axx, ψ_Axy, ψ_Axz, ψ_Ayx, ψ_Ayy, ψ_Ayz, ψ_Azx, ψ_Azy, ψ_Azz = create_pml_variables()
            return new(x, y, z, t, φ, A, ρ, j, ψ_φx, ψ_φy, ψ_φz, ψ_Axx, ψ_Axy, ψ_Axz, ψ_Ayx, ψ_Ayy, ψ_Ayz, ψ_Azx, ψ_Azy, ψ_Azz)
        else
            return new(x, y, z, t, φ, A, ρ, j, ψ_φx, ψ_φy, ψ_φz, ψ_Axx, ψ_Axy, ψ_Axz, ψ_Ayx, ψ_Ayy, ψ_Ayz, ψ_Azx, ψ_Azy, ψ_Azz)
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
- ψ_φx, ψ_φy, ψ_φz - вспомогательные переменные для φ в PML
- ψ_Axx, ψ_Axy, ψ_Axz - вспомогательные переменные для A_x в PML
- ψ_Ayx, ψ_Ayy, ψ_Ayz - вспомогательные переменные для A_y в PML
- ψ_Azx, ψ_Azy, ψ_Azz - вспомогательные переменные для A_z в PML
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

При включённом PML добавляет ψ-члены с индикатором PML-области.
"""
function dalembert_operator(F, params, constants::PhysicalConstants, variables::VariableSet, domains; 
                            pml_config::PMLConfig=default_pml_config(), 
                            ψ_x=nothing, ψ_y=nothing, ψ_z=nothing)
    if pml_config.enabled
        return apply_pml_to_dalembert(F, constants, variables, pml_config, domains; ψ_x=ψ_x, ψ_y=ψ_y, ψ_z=ψ_z)
    else
        Δ = laplacian(F, params)
        return Δ - (constants.ε * constants.μ / constants.c^2) * (Differential(variables.t)^2)(F)
    end
end

"""
    create_pde_system(constants, domains, pml_config)

Создает систему PDE уравнений для обратной задачи ЭЭГ с поддержкой PML.

При включённом PML:
- ψ-поля передаются в dalembert_operator и умножаются на индикатор PML-области
- Это гарантирует, что вклад ψ равен нулю во внутренней области

Автоматически вычисляет параметры PML (sigma_max, kappa_max, alpha_max), если они не заданы.
"""
function create_pde_system(constants::PhysicalConstants, variables::VariableSet, bcs, domains; pml_config::PMLConfig=default_pml_config())
    # Разрешаем параметры PML (вычисляем автоматические значения, если нужно)
    resolved_pml_config = resolve_pml_config(pml_config, domains, constants.c)
    
    # Подготавливаем ψ-поля для скалярного потенциала φ
    ψ_φ = if resolved_pml_config.enabled
        (
            variables.ψ_φx(variables.x, variables.y, variables.z, variables.t),
            variables.ψ_φy(variables.x, variables.y, variables.z, variables.t),
            variables.ψ_φz(variables.x, variables.y, variables.z, variables.t)
        )
    else
        (nothing, nothing, nothing)
    end
    
    # Уравнение для скалярного потенциала с ψ-полями
    eq_φ = dalembert_operator(
        variables.φ(variables.x, variables.y, variables.z, variables.t), 
        [variables.x, variables.y, variables.z], 
        constants, variables, domains; 
        pml_config=resolved_pml_config,
        ψ_x=ψ_φ[1], ψ_y=ψ_φ[2], ψ_z=ψ_φ[3]
    ) ~ -4 * pi * variables.ρ(variables.x, variables.y, variables.z, variables.t) / constants.ε
    
    # Уравнения для векторного потенциала с соответствующими ψ-полями
    eq_A = [
        begin
            # Для каждой компоненты A[i] свои ψ-поля
            ψ_Ai = if resolved_pml_config.enabled
                if i == 1  # Ax
                    (
                        variables.ψ_Axx(variables.x, variables.y, variables.z, variables.t),
                        variables.ψ_Axy(variables.x, variables.y, variables.z, variables.t),
                        variables.ψ_Axz(variables.x, variables.y, variables.z, variables.t)
                    )
                elseif i == 2  # Ay
                    (
                        variables.ψ_Ayx(variables.x, variables.y, variables.z, variables.t),
                        variables.ψ_Ayy(variables.x, variables.y, variables.z, variables.t),
                        variables.ψ_Ayz(variables.x, variables.y, variables.z, variables.t)
                    )
                else  # Az
                    (
                        variables.ψ_Azx(variables.x, variables.y, variables.z, variables.t),
                        variables.ψ_Azy(variables.x, variables.y, variables.z, variables.t),
                        variables.ψ_Azz(variables.x, variables.y, variables.z, variables.t)
                    )
                end
            else
                (nothing, nothing, nothing)
            end
            
            dalembert_operator(
                variables.A[i](variables.x, variables.y, variables.z, variables.t), 
                [variables.x, variables.y, variables.z], 
                constants, variables, domains; 
                pml_config=resolved_pml_config,
                ψ_x=ψ_Ai[1], ψ_y=ψ_Ai[2], ψ_z=ψ_Ai[3]
            ) ~ -constants.μ * 4 * pi / constants.c * variables.j[i](variables.x, variables.y, variables.z, variables.t)
        end
        for i in 1:3
    ]
    
    # Условие калибровки (условие Лоренца)
    eq_gauge = (Differential(variables.x)(variables.A[1](variables.x, variables.y, variables.z, variables.t)) + 
                Differential(variables.y)(variables.A[2](variables.x, variables.y, variables.z, variables.t)) + 
                Differential(variables.z)(variables.A[3](variables.x, variables.y, variables.z, variables.t)) + 
                (constants.ε * constants.μ / constants.c) * Differential(variables.t)(variables.φ(variables.x, variables.y, variables.z, variables.t))) ~ 0.0
    
    eq = [eq_φ; eq_A; eq_gauge]
    
    # Добавляем уравнения для вспомогательных полей PML
    if resolved_pml_config.enabled
        append!(eq, create_pml_auxiliary_equations(variables, constants, resolved_pml_config, domains))
    end
    
    allvars = [
        variables.φ(variables.x, variables.y, variables.z, variables.t); 
        [A_(variables.x, variables.y, variables.z, variables.t) for A_ in variables.A]; 
        variables.ρ(variables.x, variables.y, variables.z, variables.t); 
        [j_(variables.x, variables.y, variables.z, variables.t) for j_ in variables.j];
        # Добавляем PML переменные
        variables.ψ_φx(variables.x, variables.y, variables.z, variables.t);
        variables.ψ_φy(variables.x, variables.y, variables.z, variables.t);
        variables.ψ_φz(variables.x, variables.y, variables.z, variables.t);
        variables.ψ_Axx(variables.x, variables.y, variables.z, variables.t);
        variables.ψ_Axy(variables.x, variables.y, variables.z, variables.t);
        variables.ψ_Axz(variables.x, variables.y, variables.z, variables.t);
        variables.ψ_Ayx(variables.x, variables.y, variables.z, variables.t);
        variables.ψ_Ayy(variables.x, variables.y, variables.z, variables.t);
        variables.ψ_Ayz(variables.x, variables.y, variables.z, variables.t);
        variables.ψ_Azx(variables.x, variables.y, variables.z, variables.t);
        variables.ψ_Azy(variables.x, variables.y, variables.z, variables.t);
        variables.ψ_Azz(variables.x, variables.y, variables.z, variables.t)
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
    create_boundary_conditions(constants, variables, domains, pml_config)

Создает граничные условия для PDE системы.

Автоматически вычисляет параметры PML (sigma_max, kappa_max, alpha_max), если они не заданы.
"""
function create_boundary_conditions(constants::PhysicalConstants, variables::VariableSet, domains; pml_config::PMLConfig=default_pml_config())
    # Разрешаем параметры PML (вычисляем автоматические значения, если нужно)
    resolved_pml_config = resolve_pml_config(pml_config, domains, constants.c)
    
    x_min, x_max = infimum(domains[1].domain), supremum(domains[1].domain)
    y_min, y_max = infimum(domains[2].domain), supremum(domains[2].domain)
    z_min, z_max = infimum(domains[3].domain), supremum(domains[3].domain)
    
    bcs = [
    [variables.φ(x_min, variables.y, variables.z, variables.t) ~ 0.0, variables.φ(x_max, variables.y, variables.z, variables.t) ~ 0.0,
    variables.φ(variables.x, y_min, variables.z, variables.t) ~ 0.0, variables.φ(variables.x, y_max, variables.z, variables.t) ~ 0.0,
    variables.φ(variables.x, variables.y, z_min, variables.t) ~ 0.0, variables.φ(variables.x, variables.y, z_max, variables.t) ~ 0.0];
    [variables.A[i](x_min, variables.y, variables.z, variables.t)  ~ 0.0 for i in 1:3];
    [variables.A[i](x_max, variables.y, variables.z, variables.t)  ~ 0.0 for i in 1:3];
    [variables.A[i](variables.x, y_min, variables.z, variables.t)  ~ 0.0 for i in 1:3];
    [variables.A[i](variables.x, y_max, variables.z, variables.t)  ~ 0.0 for i in 1:3];
    [variables.A[i](variables.x, variables.y, z_min, variables.t)  ~ 0.0 for i in 1:3];
    [variables.A[i](variables.x, variables.y, z_max, variables.t)  ~ 0.0 for i in 1:3];
    ]

    # Добавляем граничные условия для PML
    if resolved_pml_config.enabled
        append!(bcs, create_pml_boundary_conditions(variables, resolved_pml_config, domains))
    end

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
