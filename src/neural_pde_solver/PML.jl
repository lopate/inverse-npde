"""
    PML

Модуль для Perfectly Matched Layer (PML) - поглощающих граничных условий.

Подход: PML через затухание (γ) и экранирование (α).

Модифицированное волновое уравнение:
    ∇²u - (1/c²)∂²u/∂t² - γ(r)∂u/∂t - α(r)u = S

где:
- γ(r) — динамическое затухание (вязкое трение), поглощает энергию волн
- α(r) — статический экранирующий член, обеспечивает экспоненциальное убывание потенциала

ВАЖНО: Используем безразмерные единицы с c = 1!

Формула для gamma_max (на основе характерной частоты):
    γ_max = π * f_char * (m+1) * ln(1/R)

где:
- f_char — характерная частота сигнала (для ЭЭГ ~10-100 Гц)
- m — порядок профиля (обычно 2)
- R — коэффициент отражения
"""
module PML

using ModelingToolkit: @parameters, @variables, Differential, infimum, supremum
using Symbolics

# Экспорты
export PMLConfig, compute_gamma, compute_alpha
export compute_gamma_max, compute_alpha_max
export default_pml_config, get_output_dim
export resolve_pml_config, step_indicator

"""
    PMLConfig

Конфигурация для Perfectly Matched Layer (PML) - поглощающих граничных условий.

Параметры:
- pml_thickness_ratio: Толщина PML слоя в долях от размеров домена (0-1)
- reflection_coefficient: Коэффициент отражения R (по умолчанию 1e-4 = 0.01%)
- characteristic_frequency: Характерная частота сигнала в Гц (для ЭЭГ ~10-100 Гц)
- profile_order: Порядок профиля PML (обычно 2)
- gamma_max: Максимальное затухание (nothing = автоматический расчёт)
- alpha_max: Максимальное экранирование (nothing = автоматический расчёт)
- enabled: Включить/отключить PML
"""
struct PMLConfig
    pml_thickness_ratio::Float64           # Толщина PML слоя в долях от размеров домена (0-1)
    reflection_coefficient::Float64        # Коэффициент отражения R (по умолчанию 1e-4 = 0.01%)
    characteristic_frequency::Float64      # Характерная частота сигнала в Гц (для ЭЭГ ~10-100 Гц)
    profile_order::Int                     # Порядок профиля PML (обычно 2)
    gamma_max::Union{Float64, Nothing}     # Максимальное значение коэффициента затухания (nothing = авто)
    alpha_max::Union{Float64, Nothing}     # Максимальное значение коэффициента экранирования (nothing = авто)
    enabled::Bool                          # Флаг включения/отключения PML

    """
        PMLConfig(; kwargs...)

    Создаёт конфигурацию PML.

    # Параметры:
    - `pml_thickness_ratio::Float64 = 0.1`: Толщина PML слоя (0.1 = 10% от домена)
    - `reflection_coefficient::Float64 = 1e-4`: Коэффициент отражения R (1e-4 = 0.01% отражения)
    - `characteristic_frequency::Float64 = 100.0`: Характерная частота сигнала в Гц
    - `profile_order::Int = 2`: Порядок профиля PML
    - `gamma_max::Union{Float64, Nothing} = nothing`: Максимальное затухание (nothing = авто)
    - `alpha_max::Union{Float64, Nothing} = nothing`: Максимальное экранирование (nothing = авто)
    - `enabled::Bool = true`: Включить/отключить PML
    """
    function PMLConfig(; 
        pml_thickness_ratio::Float64 = 0.1, 
        reflection_coefficient::Float64 = 1e-4, 
        characteristic_frequency::Float64 = 100.0,
        profile_order::Int = 2,
        gamma_max::Union{Float64, Nothing} = nothing, 
        alpha_max::Union{Float64, Nothing} = nothing, 
        enabled::Bool = true
    )
        @assert pml_thickness_ratio > 0.0 && pml_thickness_ratio < 0.5 "PML thickness ratio must be between 0 and 0.5"
        @assert reflection_coefficient > 0.0 && reflection_coefficient < 1.0 "Reflection coefficient must be between 0 and 1"
        @assert characteristic_frequency > 0.0 "Characteristic frequency must be > 0"
        @assert profile_order >= 1 "Profile order must be >= 1"
        if gamma_max !== nothing
            @assert gamma_max >= 0.0 "Gamma max must be >= 0"
        end
        if alpha_max !== nothing
            @assert alpha_max >= 0.0 "Alpha max must be >= 0"
        end
        
        new(pml_thickness_ratio, reflection_coefficient, characteristic_frequency, profile_order, gamma_max, alpha_max, enabled)
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

Вычисляет размерность выхода нейросети. Всегда 8 переменных: φ, Ax, Ay, Az, ρ, jx, jy, jz
"""
function get_output_dim(pml_config::PMLConfig)
    return 8
end

"""
    step_indicator(x, x0)

GPU-дружественная индикаторная функция: 0 если x < x0, 1 если x > x0.

Формула: step_indicator(x, x0) = (sign(x - x0) + 1) * 0.5
"""
function step_indicator(x, x0)
    return (sign(x - x0) + 1) * 0.5
end

"""
    compute_gamma_max(f_char::Float64, m::Int, R::Float64)

Вычисляет оптимальное значение gamma_max для PML по формуле на основе частоты:

    γ_max = π * f_char * (m+1) * ln(1/R)

# Параметры:
- `f_char::Float64`: Характерная частота сигнала в Гц
- `m::Int`: Порядок профиля PML
- `R::Float64`: Коэффициент отражения

# Возвращает:
- `Float64`: Оптимальное значение gamma_max

# Пример:
```julia
gamma_max = compute_gamma_max(100.0, 2, 1e-4)  # ≈ 8680 для ЭЭГ
```
"""
function compute_gamma_max(f_char::Float64, m::Int, R::Float64)
    return π * f_char * (m + 1) * log(1.0 / R)
end

"""
    compute_gamma_max(pml_config::PMLConfig)

Вычисляет gamma_max используя параметры PMLConfig.
"""
function compute_gamma_max(pml_config::PMLConfig)
    return compute_gamma_max(
        pml_config.characteristic_frequency,
        pml_config.profile_order,
        pml_config.reflection_coefficient
    )
end

"""
    compute_alpha_max(L::Float64, R::Float64)

Вычисляет оптимальное значение alpha_max для PML по формуле:

    α_max = (2 * ln(1/R) / L)²

# Параметры:
- `L::Float64`: Характерный размер домена
- `R::Float64`: Коэффициент отражения

# Возвращает:
- `Float64`: Оптимальное значение alpha_max
"""
function compute_alpha_max(L::Float64, R::Float64)
    return (2.0 * log(1.0 / R) / L)^2
end

"""
    compute_alpha_max(pml_config::PMLConfig, domain_size::Float64)

Вычисляет alpha_max используя параметры PMLConfig и размер домена.
"""
function compute_alpha_max(pml_config::PMLConfig, domain_size::Float64)
    return compute_alpha_max(domain_size, pml_config.reflection_coefficient)
end

"""
    resolve_pml_config(pml_config::PMLConfig, domain_size::Float64)

Разрешает все параметры PML конфигурации, вычисляя автоматические значения.

# Параметры:
- `pml_config::PMLConfig`: Конфигурация PML
- `domain_size::Float64`: Размер домена (L)

# Возвращает:
- `PMLConfig`: Новая PMLConfig с вычисленными значениями gamma_max и alpha_max
"""
function resolve_pml_config(pml_config::PMLConfig, domain_size::Float64)
    if !pml_config.enabled
        return pml_config
    end
    
    # Вычисляем gamma_max если не задан (на основе частоты)
    gamma_max_resolved = pml_config.gamma_max === nothing ? 
        compute_gamma_max(pml_config) : pml_config.gamma_max
    
    # Вычисляем alpha_max если не задан
    alpha_max_resolved = pml_config.alpha_max === nothing ? 
        compute_alpha_max(pml_config, domain_size) : pml_config.alpha_max
    
    return PMLConfig(
        pml_thickness_ratio = pml_config.pml_thickness_ratio,
        reflection_coefficient = pml_config.reflection_coefficient,
        characteristic_frequency = pml_config.characteristic_frequency,
        profile_order = pml_config.profile_order,
        gamma_max = gamma_max_resolved,
        alpha_max = alpha_max_resolved,
        enabled = pml_config.enabled
    )
end

"""
    resolve_pml_config(pml_config::PMLConfig, domains, c)

Разрешает все параметры PML конфигурации, вычисляя автоматические значения.

# Параметры:
- `pml_config::PMLConfig`: Конфигурация PML
- `domains`: Массив доменов для каждой переменной
- `c`: Скорость света (игнорируется, используется c=1)

# Возвращает:
- `PMLConfig`: Новая PMLConfig с вычисленными значениями
"""
function resolve_pml_config(pml_config::PMLConfig, domains, c)
    if !pml_config.enabled
        return pml_config
    end
    
    # Получаем размер домена
    x_domain = domains[1]
    x_min, x_max = infimum(x_domain.domain), supremum(x_domain.domain)
    domain_size = x_max - x_min
    
    return resolve_pml_config(pml_config, domain_size)
end

"""
    compute_gamma(pml_config::PMLConfig, x, y, z, domains)

Вычисляет коэффициент затухания γ(r) для заданной точки.

Профиль затухания — полиномиальный порядка m:
    γ(x) = γ_max * ((x - x_pml) / d)^m

# Параметры:
- `pml_config::PMLConfig`: Конфигурация PML
- `x, y, z`: Координаты точки
- `domains`: Массив доменов для каждой переменной

# Возвращает:
- Значение γ(r) — суммарный коэффициент затухания по всем направлениям
"""
function compute_gamma(pml_config::PMLConfig, x, y, z, domains)
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

    # Порядок профиля
    m = pml_config.profile_order
    
    # Вычисляем расстояние до внутренней границы для каждого направления
    dx = (x - x_inner_max) * step_indicator(x, x_inner_max) + (x_inner_min - x) * step_indicator(x_inner_min, x)
    dy = (y - y_inner_max) * step_indicator(y, y_inner_max) + (y_inner_min - y) * step_indicator(y_inner_min, y)
    dz = (z - z_inner_max) * step_indicator(z, z_inner_max) + (z_inner_min - z) * step_indicator(z_inner_min, z)

    # Полиномиальный профиль затухания: γ = γ_max * (d/D)^m
    gamma_x = pml_config.gamma_max * (dx / Dx)^m
    gamma_y = pml_config.gamma_max * (dy / Dy)^m
    gamma_z = pml_config.gamma_max * (dz / Dz)^m

    # Суммарное затухание
    return gamma_x + gamma_y + gamma_z
end

"""
    compute_alpha(pml_config::PMLConfig, x, y, z, domains)

Вычисляет коэффициент экранирования α(r) для заданной точки.

Профиль экранирования — полиномиальный порядка m:
    α(x) = α_max * ((x - x_pml) / d)^m

# Параметры:
- `pml_config::PMLConfig`: Конфигурация PML
- `x, y, z`: Координаты точки
- `domains`: Массив доменов для каждой переменной

# Возвращает:
- Значение α(r) — суммарный коэффициент экранирования по всем направлениям
"""
function compute_alpha(pml_config::PMLConfig, x, y, z, domains)
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

    # Порядок профиля
    m = pml_config.profile_order
    
    # Вычисляем расстояние до внутренней границы для каждого направления
    dx = (x - x_inner_max) * step_indicator(x, x_inner_max) + (x_inner_min - x) * step_indicator(x_inner_min, x)
    dy = (y - y_inner_max) * step_indicator(y, y_inner_max) + (y_inner_min - y) * step_indicator(y_inner_min, y)
    dz = (z - z_inner_max) * step_indicator(z, z_inner_max) + (z_inner_min - z) * step_indicator(z_inner_min, z)

    # Полиномиальный профиль экранирования: α = α_max * (d/D)^m
    alpha_x = pml_config.alpha_max * (dx / Dx)^m
    alpha_y = pml_config.alpha_max * (dy / Dy)^m
    alpha_z = pml_config.alpha_max * (dz / Dz)^m

    # Суммарное экранирование
    return alpha_x + alpha_y + alpha_z
end

end # module PML
