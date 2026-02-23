module PML

using ModelingToolkit: @parameters, @variables, Differential, infimum, supremum
using Symbolics

# Экспорты
export PMLConfig, create_pml_variables, compute_pml_profiles, compute_pml_indicator
export apply_pml_to_dalembert, create_pml_auxiliary_equations
export default_pml_config, get_output_dim
export compute_optimal_sigma_max, compute_optimal_kappa_max, compute_optimal_alpha_max
export resolve_pml_config

"""
    PMLConfig

Конфигурация для Perfectly Matched Layer (PML) - поглощающих граничных условий.

Параметры sigma_max, kappa_max, alpha_max могут быть:
- nothing: автоматический расчёт оптимального значения
- Float64: явное задание значения
"""
struct PMLConfig
    pml_thickness_ratio::Float64           # Толщина PML слоя в долях от размеров домена (0-1)
    polynomial_order::Int                  # Порядок полиномиального профиля поглощения
    sigma_max::Union{Float64, Nothing}     # Максимальное значение коэффициента поглощения (nothing = авто)
    kappa_max::Union{Float64, Nothing}     # Максимальное значение коэффициента растяжения (nothing = авто)
    alpha_max::Union{Float64, Nothing}     # Максимальное значение коэффициента дисперсии (nothing = авто)
    inner_damping::Float64                 # Коэффициент затухания ψ во внутренней области
    enabled::Bool                          # Флаг включения/отключения PML

    """
        PMLConfig(; pml_thickness_ratio=0.1, polynomial_order=3, sigma_max=nothing, kappa_max=nothing, alpha_max=nothing, inner_damping=100.0, enabled=true)

    Создаёт конфигурацию PML. По умолчанию PML включён, все параметры профилей рассчитываются автоматически.

    Параметры:
    - pml_thickness_ratio: Толщина PML слоя (0.1 = 10% от домена)
    - polynomial_order: Порядок полиномиального профиля (обычно 3)
    - sigma_max: Максимальное поглощение (nothing = автоматический расчёт по формуле Gedney)
    - kappa_max: Максимальное растяжение координат (nothing = автоматический расчёт: 2.5)
    - alpha_max: Максимальное дисперсионное число (nothing = автоматический расчёт: 0.0)
    - inner_damping: Коэффициент затухания ψ во внутренней области (принудительно зануляет ψ)
    - enabled: Включить/отключить PML
    """
    function PMLConfig(; 
        pml_thickness_ratio::Float64 = 0.1, 
        polynomial_order::Int = 3, 
        sigma_max::Union{Float64, Nothing} = nothing, 
        kappa_max::Union{Float64, Nothing} = nothing, 
        alpha_max::Union{Float64, Nothing} = nothing,
        inner_damping::Float64 = 100.0,
        enabled::Bool = true
    )
        @assert pml_thickness_ratio > 0.0 && pml_thickness_ratio < 0.5 "PML thickness ratio must be between 0 and 0.5"
        @assert polynomial_order ≥ 1 "Polynomial order must be at least 1"
        if kappa_max !== nothing
            @assert kappa_max ≥ 1.0 "Kappa must be ≥ 1"
        end
        if alpha_max !== nothing
            @assert alpha_max ≥ 0.0 "Alpha must be ≥ 0"
        end
        @assert inner_damping ≥ 0.0 "Inner damping must be ≥ 0"
        
        new(pml_thickness_ratio, polynomial_order, sigma_max, kappa_max, alpha_max, inner_damping, enabled)
    end
end

"""
    default_pml_config()

Возвращает стандартную конфигурацию PML (включён по умолчанию).
"""
function default_pml_config()
    return PMLConfig()
end

"""
    get_output_dim(pml_config::PMLConfig)

Вычисляет размерность выхода нейросети в зависимости от конфигурации PML.

Без PML: 8 переменных (φ, Ax, Ay, Az, ρ, jx, jy, jz)
С PML: 8 + 12 = 20 переменных (добавляются ψ_φx, ψ_φy, ψ_φz, ψ_Axx, ψ_Axy, ψ_Axz, ψ_Ayx, ψ_Ayy, ψ_Ayz, ψ_Azx, ψ_Azy, ψ_Azz)

Параметры:
- pml_config: Конфигурация PML

Возвращает количество выходов нейросети.
"""
function get_output_dim(pml_config::PMLConfig)
    # Базовые переменные: φ, Ax, Ay, Az, ρ, jx, jy, jz = 8
    base_dim = 8
    
    if !pml_config.enabled
        return base_dim
    end
    
    # PML добавляет 12 ψ-полей:
    # ψ_φx, ψ_φy, ψ_φz (3)
    # ψ_Axx, ψ_Axy, ψ_Axz (3)
    # ψ_Ayx, ψ_Ayy, ψ_Ayz (3)
    # ψ_Azx, ψ_Azy, ψ_Azz (3)
    pml_dim = 12
    
    return base_dim + pml_dim
end

"""
    compute_optimal_sigma_max(pml_config, domains, c; reflection_coefficient=0.001)

Вычисляет оптимальное значение sigma_max для PML по формуле, учитывающей
коэффициент отражения на границе PML.

Параметры:
- pml_config: Конфигурация PML
- domains: Массив доменов для каждой переменной
- c: Скорость света в используемых единицах
- reflection_coefficient: Желаемый коэффициент отражения R (по умолчанию 0.001 = 0.1%)

Формула:
    σ_max ≈ -(m + 1) * c / (2 * d) * ln(R)

где:
- m - порядок полинома
- d - толщина PML слоя
- R - коэффициент отражения (типично 10^-3 до 10^-6)

При R = 0.001 (0.1% отражения) получаем:
    σ_max ≈ (m + 1) * c * 3.45 / d

Возвращает оптимальное значение sigma_max.
"""
function compute_optimal_sigma_max(pml_config, domains, c; reflection_coefficient::Float64=0.001)
    # Получаем размер домена
    x_domain = domains[1]
    x_min, x_max = infimum(x_domain.domain), supremum(x_domain.domain)
    domain_size = x_max - x_min
    
    # Толщина PML слоя (d в формуле)
    d = domain_size * pml_config.pml_thickness_ratio
    
    # Порядок полинома
    m = pml_config.polynomial_order
    
    # Формула для оптимального sigma_max с учётом коэффициента отражения
    # σ_max ≈ -(m + 1) * c / (2 * d) * ln(R)
    # При R = 0.001: -ln(0.001) ≈ 6.907
    sigma_max = -(m + 1) * c / (2.0 * d) * log(reflection_coefficient)
    
    return sigma_max
end

"""
    compute_optimal_kappa_max(pml_config, domains; default_value=2.5)

Вычисляет оптимальное значение kappa_max для PML.

Параметры:
- pml_config: Конфигурация PML
- domains: Массив доменов для каждой переменной
- default_value: Значение по умолчанию (2.5 - середина диапазона [2,3] для улучшения поглощения широкого спектра)

Возвращает оптимальное значение kappa_max.

Примечание: κ отвечает за растяжение координаты без потерь.
Формула: κ(x) = 1 + (κ_max - 1) * (x/d)^m
При κ_max ∈ [2,3] улучшается поглощение для широкого спектра частот.
"""
function compute_optimal_kappa_max(pml_config, domains; default_value::Float64=2.5)
    # kappa_max выбирается в диапазоне [2, 3] для улучшения поглощения широкого спектра
    # Значение 2.5 - середина диапазона, обеспечивает хороший баланс
    return default_value
end

"""
    compute_optimal_alpha_max(pml_config, domains; default_value=0.0)

Вычисляет оптимальное значение alpha_max для PML.

Параметры:
- pml_config: Конфигурация PML
- domains: Массив доменов для каждой переменной
- default_value: Значение по умолчанию (0.0 - без дополнительной дисперсии)

Возвращает оптимальное значение alpha_max.

Примечание: α отвечает за подавление низкочастотных отражений.
В первой реализации используем α = 0 для упрощения.
"""
function compute_optimal_alpha_max(pml_config, domains; default_value::Float64=0.0)
    # alpha_max = 0 в первой реализации для упрощения
    # Может быть увеличено до 0.05-0.5 для подавления низкочастотных отражений
    return default_value
end

"""
    resolve_pml_config(pml_config::PMLConfig, domains, c)

Разрешает все параметры PML конфигурации, вычисляя автоматические значения там, где они равны nothing.

Параметры:
- pml_config: Конфигурация PML (может содержать nothing для автоматического расчёта)
- domains: Массив доменов для каждой переменной
- c: Скорость света в используемых единицах

Возвращает новую PMLConfig с вычисленными значениями.
"""
function resolve_pml_config(pml_config::PMLConfig, domains, c)
    if !pml_config.enabled
        return pml_config
    end
    
    # Вычисляем sigma_max если не задан
    sigma_max_resolved = pml_config.sigma_max === nothing ? 
        compute_optimal_sigma_max(pml_config, domains, c) : pml_config.sigma_max
    
    # Вычисляем kappa_max если не задан
    kappa_max_resolved = pml_config.kappa_max === nothing ? 
        compute_optimal_kappa_max(pml_config, domains) : pml_config.kappa_max
    
    # Вычисляем alpha_max если не задан
    alpha_max_resolved = pml_config.alpha_max === nothing ? 
        compute_optimal_alpha_max(pml_config, domains) : pml_config.alpha_max
    
    return PMLConfig(
        pml_thickness_ratio = pml_config.pml_thickness_ratio,
        polynomial_order = pml_config.polynomial_order,
        sigma_max = sigma_max_resolved,
        kappa_max = kappa_max_resolved,
        alpha_max = alpha_max_resolved,
        inner_damping = pml_config.inner_damping,
        enabled = pml_config.enabled
    )
end

"""
    sigma_profile(d, D, sigma_max, m)

Вычисляет коэффициент поглощения σ по полиномиальному профилю степени m.

Параметры:
- d: Расстояние от границы внутренней области до точки в PML
- D: Толщина PML слоя
- sigma_max: Максимальное значение поглощения
- m: Порядок полинома

Формула: σ(d) = σ_max * (d/D)^m
"""
function sigma_profile(d, D, sigma_max, m)
    return sigma_max * (d / D)^m
end

"""
    kappa_profile(d, D, kappa_max, m)

Вычисляет коэффициент растяжения κ по полиномиальному профилю степени m.

Параметры:
- d: Расстояние от границы внутренней области до точки в PML
- D: Толщина PML слоя
- kappa_max: Максимальное значение растяжения
- m: Порядок полинома

Формула: κ(d) = 1 + (κ_max - 1) * (d/D)^m
"""
function kappa_profile(d, D, kappa_max, m)
    return 1.0 + (kappa_max - 1.0) * (d / D)^m
end

"""
    alpha_profile(d, D, alpha_max)

Вычисляет коэффициент дисперсии α по линейному профилю.

Параметры:
- d: Расстояние от границы внутренней области до точки в PML
- D: Толщина PML слоя
- alpha_max: Максимальное значение дисперсии

Формула: α(d) = α_max * (1 - d/D)
"""
function alpha_profile(d, D, alpha_max)
    return alpha_max * (1.0 - d / D)
end

"""
    compute_pml_indicator(pml_config::PMLConfig, x, y, z, domains)

Вычисляет индикатор PML-области: 1 если точка в PML, 0 если во внутренней области.
Использует GPU-дружественную функцию sign() для совместимости с Zygote AD.

Параметры:
- pml_config: Конфигурация PML
- x, y, z: Координаты точки
- domains: Массив доменов для каждой переменной

Возвращает значение от 0 (внутренняя область) до 1 (PML область)
"""
function compute_pml_indicator(pml_config::PMLConfig, x, y, z, domains)
    if !pml_config.enabled
        return 0.0
    end

    # Получаем размеры домена
    x_domain = domains[1]
    y_domain = domains[2]
    z_domain = domains[3]

    x_min, x_max = infimum(x_domain.domain), supremum(x_domain.domain)
    y_min, y_max = infimum(y_domain.domain), supremum(y_domain.domain)
    z_min, z_max = infimum(z_domain.domain), supremum(z_domain.domain)

    # Толщины PML слоёв
    Dx = (x_max - x_min) * pml_config.pml_thickness_ratio
    Dy = (y_max - y_min) * pml_config.pml_thickness_ratio
    Dz = (z_max - z_min) * pml_config.pml_thickness_ratio

    # Границы внутренней области
    x_inner_min = x_min + Dx
    x_inner_max = x_max - Dx
    y_inner_min = y_min + Dy
    y_inner_max = y_max - Dy
    z_inner_min = z_min + Dz
    z_inner_max = z_max - Dz

    # GPU-дружественная индикаторная функция: 0 если x < x0, 1 если x > x0
    step_indicator(x, x0) = (sign(x - x0) + 1) * 0.5

    # Определяем, находимся ли мы в PML по каждой координате
    # in_pml_x = 1 если x > x_inner_max ИЛИ x < x_inner_min
    # Используем формулу логического ИЛИ через умножение: a OR b = 1 - (1-a)*(1-b)
    in_pml_x_pos = step_indicator(x, x_inner_max)  # 1 если x > x_inner_max
    in_pml_x_neg = step_indicator(x_inner_min, x)  # 1 если x < x_inner_min (инвертированный порядок)
    in_pml_x = 1.0 - (1.0 - in_pml_x_pos) * (1.0 - in_pml_x_neg)  # Логическое ИЛИ без min/max

    in_pml_y_pos = step_indicator(y, y_inner_max)
    in_pml_y_neg = step_indicator(y_inner_min, y)
    in_pml_y = 1.0 - (1.0 - in_pml_y_pos) * (1.0 - in_pml_y_neg)

    in_pml_z_pos = step_indicator(z, z_inner_max)
    in_pml_z_neg = step_indicator(z_inner_min, z)
    in_pml_z = 1.0 - (1.0 - in_pml_z_pos) * (1.0 - in_pml_z_neg)

    # Точка в PML если хотя бы одна координата в PML слое
    # Используем формулу логического ИЛИ через умножение вместо max
    # in_pml = in_pml_x OR in_pml_y OR in_pml_z
    in_pml_xy = 1.0 - (1.0 - in_pml_x) * (1.0 - in_pml_y)
    in_pml = 1.0 - (1.0 - in_pml_xy) * (1.0 - in_pml_z)
    
    return in_pml
end

"""
    compute_pml_profiles(pml_config::PMLConfig, x, y, z, domains)

Вычисляет профили σ, κ и α для каждой координаты в зависимости от положения точки
с использованием обычного ifelse для динамического расчёта.

Параметры:
- pml_config: Конфигурация PML
- x, y, z: Координаты точки
- domains: Массив доменов для каждой переменной

Возвращает кортеж (σx, σy, σz, κx, κy, κz, αx, αy, αz)
"""
function compute_pml_profiles(pml_config::PMLConfig, x, y, z, domains)
    if !pml_config.enabled
        return (0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0)
    end

    # Получаем размеры домена (структура доменов известна: x, y, z, t)
    x_domain = domains[1]
    y_domain = domains[2]
    z_domain = domains[3]

    x_min, x_max = infimum(x_domain.domain), supremum(x_domain.domain)
    y_min, y_max = infimum(y_domain.domain), supremum(y_domain.domain)
    z_min, z_max = infimum(z_domain.domain), supremum(z_domain.domain)

    # Толщины PML слоёв
    Dx = (x_max - x_min) * pml_config.pml_thickness_ratio
    Dy = (y_max - y_min) * pml_config.pml_thickness_ratio
    Dz = (z_max - z_min) * pml_config.pml_thickness_ratio

    # Определяем границы внутренней области (без PML)
    x_inner_min = x_min + Dx
    x_inner_max = x_max - Dx
    y_inner_min = y_min + Dy
    y_inner_max = y_max - Dy
    z_inner_min = z_min + Dz
    z_inner_max = z_max - Dz

    # GPU-дружественная индикаторная функция на основе sign: 0 если x < x0, 1 если x > x0
    step_indicator(x, x0) = (sign(x - x0) + 1) * 0.5
    
    # Вычисляем расстояние до внутренней границы
    # (x - x_inner_max) активируется когда x > x_inner_max
    # (x_inner_min - x) активируется когда x < x_inner_min
    dx = (x - x_inner_max) * step_indicator(x, x_inner_max) + (x_inner_min - x) * step_indicator(x_inner_min, x)
    dy = (y - y_inner_max) * step_indicator(y, y_inner_max) + (y_inner_min - y) * step_indicator(y_inner_min, y)
    dz = (z - z_inner_max) * step_indicator(z, z_inner_max) + (z_inner_min - z) * step_indicator(z_inner_min, z)

    # Вычисляем профили с использованием обычных операций
    σx = sigma_profile(dx, Dx, pml_config.sigma_max, pml_config.polynomial_order)
    σy = sigma_profile(dy, Dy, pml_config.sigma_max, pml_config.polynomial_order)
    σz = sigma_profile(dz, Dz, pml_config.sigma_max, pml_config.polynomial_order)

    κx = kappa_profile(dx, Dx, pml_config.kappa_max, pml_config.polynomial_order)
    κy = kappa_profile(dy, Dy, pml_config.kappa_max, pml_config.polynomial_order)
    κz = kappa_profile(dz, Dz, pml_config.kappa_max, pml_config.polynomial_order)

    αx = alpha_profile(dx, Dx, pml_config.alpha_max)
    αy = alpha_profile(dy, Dy, pml_config.alpha_max)
    αz = alpha_profile(dz, Dz, pml_config.alpha_max)

    return (σx, σy, σz, κx, κy, κz, αx, αy, αz)
end

"""
    create_pml_variables()

Создаёт вспомогательные переменные для PML.

Возвращает кортеж с переменными для скалярного потенциала φ и векторного потенциала A:
- ψ_φx, ψ_φy, ψ_φz: Для φ
- ψ_Axx, ψ_Axy, ψ_Axz: Для A_x
- ψ_Ayx, ψ_Ayy, ψ_Ayz: Для A_y
- ψ_Azx, ψ_Azy, ψ_Azz: Для A_z
"""
function create_pml_variables()
    @variables ψ_φx(..), ψ_φy(..), ψ_φz(..)
    @variables ψ_Axx(..), ψ_Axy(..), ψ_Axz(..)
    @variables ψ_Ayx(..), ψ_Ayy(..), ψ_Ayz(..)
    @variables ψ_Azx(..), ψ_Azy(..), ψ_Azz(..)
    
    return (
        ψ_φx, ψ_φy, ψ_φz,
        ψ_Axx, ψ_Axy, ψ_Axz,
        ψ_Ayx, ψ_Ayy, ψ_Ayz,
        ψ_Azx, ψ_Azy, ψ_Azz
    )
end

"""
    apply_pml_to_dalembert(F, constants, variables, pml_config, domains; ψ_x=nothing, ψ_y=nothing, ψ_z=nothing)

Применяет PML к оператору Даламбера для волнового уравнения с использованием
обычного ifelse для динамического расчёта коэффициентов.

Важное дополнение: ψ-поля умножаются на индикатор PML-области (in_pml), чтобы
их вклад был равен нулю во внутренней области.

Параметры:
- F: Функция для которой применяется PML
- constants: Физические константы
- variables: Набор переменных
- pml_config: Конфигурация PML
- domains: Области определения
- ψ_x, ψ_y, ψ_z: Вспомогательные PML-поля для соответствующего потенциала

Возвращает модифицированный оператор Даламбера с PML.

Математическая формулировка:
    (1/c²)∂²u/∂t² = ∂/∂x((1/κx)∂u/∂x + in_pml*ψ_x) + ∂/∂y((1/κy)∂u/∂y + in_pml*ψ_y) + ∂/∂z((1/κz)∂u/∂z + in_pml*ψ_z) + S
"""
function apply_pml_to_dalembert(F, constants, variables, pml_config, domains; ψ_x=nothing, ψ_y=nothing, ψ_z=nothing)
    if !pml_config.enabled
        # Без PML - обычный оператор Даламбера
        Δ = Differential(variables.x)^2(F) + Differential(variables.y)^2(F) + Differential(variables.z)^2(F)
        return Δ - (constants.ε * constants.μ / constants.c^2) * (Differential(variables.t)^2)(F)
    end

    # Получаем профили для текущих координат
    σx, σy, σz, κx, κy, κz, αx, αy, αz = compute_pml_profiles(pml_config, variables.x, variables.y, variables.z, domains)
    
    # Вычисляем индикатор PML-области: 1 в PML, 0 во внутренней области
    in_pml = compute_pml_indicator(pml_config, variables.x, variables.y, variables.z, domains)

    # Модифицированный оператор с учетом PML для каждого направления
    dFdx = Differential(variables.x)(F)
    dFdy = Differential(variables.y)(F)
    dFdz = Differential(variables.z)(F)

    # Основные члены с растяжением координат и ψ-членами внутри производной
    # Формула пользователя: ∂/∂x((1/κx)∂u/∂x + in_pml*ψ_x)
    # ψ активны только в PML-области (in_pml = 1), во внутренней области (in_pml = 0) они зануляются
    if ψ_x !== nothing
        term_x = Differential(variables.x)( (1/κx) * dFdx + in_pml * ψ_x )
    else
        term_x = Differential(variables.x)( (1/κx) * dFdx )
    end
    if ψ_y !== nothing
        term_y = Differential(variables.y)( (1/κy) * dFdy + in_pml * ψ_y )
    else
        term_y = Differential(variables.y)( (1/κy) * dFdy )
    end
    if ψ_z !== nothing
        term_z = Differential(variables.z)( (1/κz) * dFdz + in_pml * ψ_z )
    else
        term_z = Differential(variables.z)( (1/κz) * dFdz )
    end

    return (term_x + term_y + term_z) - (constants.ε * constants.μ / constants.c^2) * (Differential(variables.t)^2)(F)
end

"""
    create_pml_auxiliary_equations(variables, constants, pml_config, domains)

Создаёт вспомогательные уравнения для эволюции полей ψ в PML с использованием
обычного ifelse для динамического расчёта коэффициентов.

Важное дополнение: добавлен член затухания ψ во внутренней области через
индикаторную функцию. Это гарантирует, что ψ = 0 во внутренней области,
где PML не активен.

Параметры:
- variables: Набор переменных (включая ψ)
- constants: Физические константы
- pml_config: Конфигурация PML
- domains: Области определения

Возвращает массив уравнений для эволюции ψ-полей.
"""
function create_pml_auxiliary_equations(variables, constants, pml_config, domains)
    if !pml_config.enabled
        return []
    end

    # Получаем доступ к переменным через прямые поля VariableSet
    # В VariableSet: A - это вектор [Ax, Ay, Az], поэтому используем индексирование
    ψ_φx_val = variables.ψ_φx
    ψ_φy_val = variables.ψ_φy
    ψ_φz_val = variables.ψ_φz
    ψ_Axx_val = variables.ψ_Axx
    ψ_Axy_val = variables.ψ_Axy
    ψ_Axz_val = variables.ψ_Axz
    ψ_Ayx_val = variables.ψ_Ayx
    ψ_Ayy_val = variables.ψ_Ayy
    ψ_Ayz_val = variables.ψ_Ayz
    ψ_Azx_val = variables.ψ_Azx
    ψ_Azy_val = variables.ψ_Azy
    ψ_Azz_val = variables.ψ_Azz
    
    # Получаем компоненты векторного потенциала A = [Ax, Ay, Az]
    Ax_val = variables.A[1]
    Ay_val = variables.A[2]
    Az_val = variables.A[3]
    
    φ_val = variables.φ
    x_val = variables.x
    y_val = variables.y
    z_val = variables.z
    t_val = variables.t
    
    # Рассчитываем профили
    σx, σy, σz, κx, κy, κz, αx, αy, αz = compute_pml_profiles(pml_config, x_val, y_val, z_val, domains)
    
    # Вычисляем индикатор PML-области: 1 в PML, 0 во внутренней области
    in_pml = compute_pml_indicator(pml_config, x_val, y_val, z_val, domains)
    
    # Коэффициент затухания ψ во внутренней области
    # inner_damping_factor = in_pml * (коэффициент PML) + (1 - in_pml) * inner_damping
    # Во внутренней области (in_pml = 0): затухание = inner_damping (сильное затухание → ψ → 0)
    # В PML области (in_pml = 1): затухание = σ/κ + α (обычное PML уравнение)
    inner_damping = pml_config.inner_damping
    
    # Эффективные коэффициенты затухания для каждого направления
    # Формула: damping_eff = in_pml * (σ/κ + α) + (1 - in_pml) * inner_damping
    damping_x = in_pml * (σx/κx + αx) + (1.0 - in_pml) * inner_damping
    damping_y = in_pml * (σy/κy + αy) + (1.0 - in_pml) * inner_damping
    damping_z = in_pml * (σz/κz + αz) + (1.0 - in_pml) * inner_damping
    
    # Источниковый член: активен только в PML области
    # source = in_pml * (σ/κ²) * ∂F/∂x
    source_φx = in_pml * (σx/κx^2) * Differential(x_val)(φ_val(x_val, y_val, z_val, t_val))
    source_φy = in_pml * (σy/κy^2) * Differential(y_val)(φ_val(x_val, y_val, z_val, t_val))
    source_φz = in_pml * (σz/κz^2) * Differential(z_val)(φ_val(x_val, y_val, z_val, t_val))

    source_Axx = in_pml * (σx/κx^2) * Differential(x_val)(Ax_val(x_val, y_val, z_val, t_val))
    source_Axy = in_pml * (σy/κy^2) * Differential(y_val)(Ax_val(x_val, y_val, z_val, t_val))
    source_Axz = in_pml * (σz/κz^2) * Differential(z_val)(Ax_val(x_val, y_val, z_val, t_val))

    source_Ayx = in_pml * (σx/κx^2) * Differential(x_val)(Ay_val(x_val, y_val, z_val, t_val))
    source_Ayy = in_pml * (σy/κy^2) * Differential(y_val)(Ay_val(x_val, y_val, z_val, t_val))
    source_Ayz = in_pml * (σz/κz^2) * Differential(z_val)(Ay_val(x_val, y_val, z_val, t_val))

    source_Azx = in_pml * (σx/κx^2) * Differential(x_val)(Az_val(x_val, y_val, z_val, t_val))
    source_Azy = in_pml * (σy/κy^2) * Differential(y_val)(Az_val(x_val, y_val, z_val, t_val))
    source_Azz = in_pml * (σz/κz^2) * Differential(z_val)(Az_val(x_val, y_val, z_val, t_val))

    # Уравнения для ψ_φx, ψ_φy, ψ_φz
    eq_ψ_φx = Differential(t_val)(ψ_φx_val(x_val, y_val, z_val, t_val)) ~ -damping_x * ψ_φx_val(x_val, y_val, z_val, t_val) + source_φx
    eq_ψ_φy = Differential(t_val)(ψ_φy_val(x_val, y_val, z_val, t_val)) ~ -damping_y * ψ_φy_val(x_val, y_val, z_val, t_val) + source_φy
    eq_ψ_φz = Differential(t_val)(ψ_φz_val(x_val, y_val, z_val, t_val)) ~ -damping_z * ψ_φz_val(x_val, y_val, z_val, t_val) + source_φz

    # Уравнения для ψ_Ax*
    eq_ψ_Axx = Differential(t_val)(ψ_Axx_val(x_val, y_val, z_val, t_val)) ~ -damping_x * ψ_Axx_val(x_val, y_val, z_val, t_val) + source_Axx
    eq_ψ_Axy = Differential(t_val)(ψ_Axy_val(x_val, y_val, z_val, t_val)) ~ -damping_y * ψ_Axy_val(x_val, y_val, z_val, t_val) + source_Axy
    eq_ψ_Axz = Differential(t_val)(ψ_Axz_val(x_val, y_val, z_val, t_val)) ~ -damping_z * ψ_Axz_val(x_val, y_val, z_val, t_val) + source_Axz

    # Уравнения для ψ_Ay*
    eq_ψ_Ayx = Differential(t_val)(ψ_Ayx_val(x_val, y_val, z_val, t_val)) ~ -damping_x * ψ_Ayx_val(x_val, y_val, z_val, t_val) + source_Ayx
    eq_ψ_Ayy = Differential(t_val)(ψ_Ayy_val(x_val, y_val, z_val, t_val)) ~ -damping_y * ψ_Ayy_val(x_val, y_val, z_val, t_val) + source_Ayy
    eq_ψ_Ayz = Differential(t_val)(ψ_Ayz_val(x_val, y_val, z_val, t_val)) ~ -damping_z * ψ_Ayz_val(x_val, y_val, z_val, t_val) + source_Ayz

    # Уравнения для ψ_Az*
    eq_ψ_Azx = Differential(t_val)(ψ_Azx_val(x_val, y_val, z_val, t_val)) ~ -damping_x * ψ_Azx_val(x_val, y_val, z_val, t_val) + source_Azx
    eq_ψ_Azy = Differential(t_val)(ψ_Azy_val(x_val, y_val, z_val, t_val)) ~ -damping_y * ψ_Azy_val(x_val, y_val, z_val, t_val) + source_Azy
    eq_ψ_Azz = Differential(t_val)(ψ_Azz_val(x_val, y_val, z_val, t_val)) ~ -damping_z * ψ_Azz_val(x_val, y_val, z_val, t_val) + source_Azz

    return [
        eq_ψ_φx, eq_ψ_φy, eq_ψ_φz,
        eq_ψ_Axx, eq_ψ_Axy, eq_ψ_Axz,
        eq_ψ_Ayx, eq_ψ_Ayy, eq_ψ_Ayz,
        eq_ψ_Azx, eq_ψ_Azy, eq_ψ_Azz
    ]
end

"""
    create_pml_boundary_conditions(variables, pml_config, domains)

Создаёт граничные условия для PML, гарантирующие, что вспомогательные поля ψ равны
0 на внутренней границе PML (где начинается слой поглощения).

Параметры:
- variables: Набор переменных (включая ψ)
- pml_config: Конфигурация PML
- domains: Области определения

Возвращает массив граничных условий для ψ-полей.
"""
function create_pml_boundary_conditions(variables, pml_config, domains)
    if !pml_config.enabled
        return []
    end

    # Получаем доступ к переменным через прямые поля VariableSet
    ψ_φx_val = variables.ψ_φx
    ψ_φy_val = variables.ψ_φy
    ψ_φz_val = variables.ψ_φz
    ψ_Axx_val = variables.ψ_Axx
    ψ_Axy_val = variables.ψ_Axy
    ψ_Axz_val = variables.ψ_Axz
    ψ_Ayx_val = variables.ψ_Ayx
    ψ_Ayy_val = variables.ψ_Ayy
    ψ_Ayz_val = variables.ψ_Ayz
    ψ_Azx_val = variables.ψ_Azx
    ψ_Azy_val = variables.ψ_Azy
    ψ_Azz_val = variables.ψ_Azz
    
    x_val = variables.x
    y_val = variables.y
    z_val = variables.z
    t_val = variables.t

    # Получаем размеры домена (структура доменов известна: x, y, z, t)
    x_domain = domains[1]
    y_domain = domains[2]
    z_domain = domains[3]

    x_min, x_max = infimum(x_domain.domain), supremum(x_domain.domain)
    y_min, y_max = infimum(y_domain.domain), supremum(y_domain.domain)
    z_min, z_max = infimum(z_domain.domain), supremum(z_domain.domain)

    # Толщины PML слоёв
    Dx = (x_max - x_min) * pml_config.pml_thickness_ratio
    Dy = (y_max - y_min) * pml_config.pml_thickness_ratio
    Dz = (z_max - z_min) * pml_config.pml_thickness_ratio

    # Определяем границы внутренней области (где начинается PML)
    x_inner_min = x_min + Dx
    x_inner_max = x_max - Dx
    y_inner_min = y_min + Dy
    y_inner_max = y_max - Dy
    z_inner_min = z_min + Dz
    z_inner_max = z_max - Dz

    # Граничные условия для ψ-полей на внутренней границе PML (ψ = 0)
    bcs = [
        # Для ψ_φx
        ψ_φx_val(x_inner_min, y_val, z_val, t_val) ~ 0.0,
        ψ_φx_val(x_inner_max, y_val, z_val, t_val) ~ 0.0,
        # Для ψ_φy
        ψ_φy_val(x_val, y_inner_min, z_val, t_val) ~ 0.0,
        ψ_φy_val(x_val, y_inner_max, z_val, t_val) ~ 0.0,
        # Для ψ_φz
        ψ_φz_val(x_val, y_val, z_inner_min, t_val) ~ 0.0,
        ψ_φz_val(x_val, y_val, z_inner_max, t_val) ~ 0.0,
        # Для ψ_Axx
        ψ_Axx_val(x_inner_min, y_val, z_val, t_val) ~ 0.0,
        ψ_Axx_val(x_inner_max, y_val, z_val, t_val) ~ 0.0,
        # Для ψ_Axy
        ψ_Axy_val(x_val, y_inner_min, z_val, t_val) ~ 0.0,
        ψ_Axy_val(x_val, y_inner_max, z_val, t_val) ~ 0.0,
        # Для ψ_Axz
        ψ_Axz_val(x_val, y_val, z_inner_min, t_val) ~ 0.0,
        ψ_Axz_val(x_val, y_val, z_inner_max, t_val) ~ 0.0,
        # Для ψ_Ayx
        ψ_Ayx_val(x_inner_min, y_val, z_val, t_val) ~ 0.0,
        ψ_Ayx_val(x_inner_max, y_val, z_val, t_val) ~ 0.0,
        # Для ψ_Ayy
        ψ_Ayy_val(x_val, y_inner_min, z_val, t_val) ~ 0.0,
        ψ_Ayy_val(x_val, y_inner_max, z_val, t_val) ~ 0.0,
        # Для ψ_Ayz
        ψ_Ayz_val(x_val, y_val, z_inner_min, t_val) ~ 0.0,
        ψ_Ayz_val(x_val, y_val, z_inner_max, t_val) ~ 0.0,
        # Для ψ_Azx
        ψ_Azx_val(x_inner_min, y_val, z_val, t_val) ~ 0.0,
        ψ_Azx_val(x_inner_max, y_val, z_val, t_val) ~ 0.0,
        # Для ψ_Azy
        ψ_Azy_val(x_val, y_inner_min, z_val, t_val) ~ 0.0,
        ψ_Azy_val(x_val, y_inner_max, z_val, t_val) ~ 0.0,
        # Для ψ_Azz
        ψ_Azz_val(x_val, y_val, z_inner_min, t_val) ~ 0.0,
        ψ_Azz_val(x_val, y_val, z_inner_max, t_val) ~ 0.0,
    ]

    return bcs
end

end # module PML
