"""
    HeadConstraints

Модуль для ограничения плотности заряда ρ внутри эллипсоида головы.

Ограничение: ρ ограничивается так, что ρ = 0 вне эллипсоида головы.
Используется формула эллипсоида:
    ((x-cx)/rx)² + ((y-cy)/ry)² + ((z-cz)/rz)² <= 1

Для GPU-совместимости используется step_indicator формула.

Экспорты:
- HeadConfig: конфигурация эллипсоида
- head_indicator: индикаторная функция (1 внутри, 0 снаружи)
- is_inside_ellipsoid: проверка точки внутри эллипсоида
- step_indicator: GPU-дружественная ступенчатая функция
"""
module HeadConstraints

# Экспорты
export HeadConfig, head_indicator, is_inside_ellipsoid, step_indicator

"""
    step_indicator(x, x0)

GPU-дружественная индикаторная функция: 0 если x < x0, 1 если x > x0.

Формула: step_indicator(x, x0) = (sign(x - x0) + 1) * 0.5

# Параметры:
- x: значение
- x0: порог

# Возвращает:
- 0.0 если x < x0, 1.0 если x > x0
"""
function step_indicator(x, x0)
    return (sign(x - x0) + 1) * 0.5f0
end

"""
    HeadConfig

Конфигурация эллипсоида головы для ограничения плотности заряда ρ.

Параметры:
- rx, ry, rz: Радиусы эллипсоида по осям x, y, z
- cx, cy, cz: Центр эллипсоида (по умолчанию 0,0,0)
- enabled: Включить/отключить ограничение

Ограничение: ρ ограничивается так, что ρ = 0 вне эллипсоида головы.
Используется формула эллипсоида:
    ((x-cx)/rx)² + ((y-cy)/ry)² + ((z-cz)/rz)² <= 1
"""
struct HeadConfig
    rx::Float32  # Радиус по x
    ry::Float32  # Радиус по y
    rz::Float32  # Радиус по z
    cx::Float32  # Центр x
    cy::Float32  # Центр y
    cz::Float32  # Центр z
    enabled::Bool  # Включить ограничение
    
    function HeadConfig(;
        rx::Float32=9.5f0,
        ry::Float32=9.0f0,
        rz::Float32=9.5f0,
        cx::Float32=0.0f0,
        cy::Float32=0.0f0,
        cz::Float32=0.0f0,
        enabled::Bool=true
    )
        @assert rx > 0 "rx must be positive"
        @assert ry > 0 "ry must be positive"
        @assert rz > 0 "rz must be positive"
        new(rx, ry, rz, cx, cy, cz, enabled)
    end
end

"""
    is_inside_ellipsoid(x, y, z, head_config::HeadConfig)

Проверяет, находится ли точка внутри эллипсоида.

Формула: (x-cx)²/rx² + (y-cy)²/ry² + (z-cz)²/rz² <= 1

# Параметры:
- x, y, z: Координаты точки
- head_config::HeadConfig: Конфигурация головы

# Возвращает:
- true если точка внутри эллипсоида, false иначе
"""
function is_inside_ellipsoid(x, y, z, head_config::HeadConfig)
    if !head_config.enabled
        return true  # Если ограничение отключено, считаем что точка внутри
    end
    
    dx = (x - head_config.cx) / head_config.rx
    dy = (y - head_config.cy) / head_config.ry
    dz = (z - head_config.cz) / head_config.rz
    
    return dx*dx + dy*dy + dz*dz <= 1.0f0
end

"""
    head_indicator(x, y, z, head_config::HeadConfig)

GPU-дружественная индикаторная функция эллипсоида.

Возвращает 1.0 если точка внутри эллипсоида, 0.0 иначе.

Использует step_indicator для каждого направления:
    indicator = step_indicator(1 - (dx² + dy² + dz²), 0)

Это даёт 1.0 внутри эллипсоида (где dx² + dy² + dz² <= 1) и 0.0 снаружи.

# Параметры:
- x, y, z: Координаты точки
- head_config::HeadConfig: Конфигурация головы

# Возвращает:
- 1.0 если точка внутри, 0.0 если снаружи
"""
function head_indicator(x, y, z, head_config::HeadConfig)
    if !head_config.enabled
        return 1.0f0  # Если ограничение отключено, возвращаем 1
    end
    
    # Вычисляем квадрат расстояния до центра, нормированный на радиусы
    dx = (x - head_config.cx) / head_config.rx
    dy = (y - head_config.cy) / head_config.ry
    dz = (z - head_config.cz) / head_config.rz
    
    dist_sq = dx*dx + dy*dy + dz*dz
    
    # Используем step_indicator: 1 если dist_sq <= 1, 0 иначе
    # step_indicator(x, x0) = (sign(x - x0) + 1) * 0.5
    # Нам нужно: 1 когда dist_sq <= 1, т.е. когда (1 - dist_sq) >= 0
    return step_indicator(1.0f0 - dist_sq, 0.0f0)
end

end  # module HeadConstraints
