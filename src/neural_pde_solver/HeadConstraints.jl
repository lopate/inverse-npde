"""
    HeadConstraints

Модуль для ограничения вектора поляризации P внутри эллипсоида головы.

Ограничение: P ограничивается так, что P = 0 вне эллипсоида головы.
Это влечёт:
    ρ = -div P = 0 вне головы
    j = ∂P/∂t = 0 вне головы

Используется формула эллипсоида:
    ((x-cx)/rx)² + ((y-cy)/ry)² + ((z-cz)/rz)² <= 1

Для GPU-совместимости используется гладкая аппроксимация сигмоидой.
Экспорты:
- HeadConfig: конфигурация эллипсоида
- head_indicator: индикаторная функция (1 внутри, 0 снаружи)
- is_inside_ellipsoid: проверка точки внутри эллипсоида
- compute_head_indicator_gpu: GPU-совместимая версия
"""
module HeadConstraints

using CUDA

# Экспорты
export HeadConfig, head_indicator, is_inside_ellipsoid, compute_head_indicator_gpu, head_indicator_batched

"""
    sigmoid_smooth(x, k=10.0)

Гладкая сигмоидная аппроксимация ступенчатой функции.
Возвращает значения от 0 до 1 плавно.

# Параметры:
- x: входное значение
- k: крутизна перехода (больше = резче)
"""
sigmoid_smooth(x, k=10.0f0) = 1.0f0 ./ (1.0f0 .+ exp.(-k .* x))

"""
    HeadConfig

Конфигурация эллипсоида головы для ограничения вектора поляризации P.

Параметры:
- rx, ry, rz: Радиусы эллипсоида по осям x, y, z
- cx, cy, cz: Центр эллипсоида (по умолчанию 0,0,0)
- enabled: Включить/отключить ограничение
- smooth_k: крутизна перехода для гладкой версии
"""
struct HeadConfig
    rx::Float32
    ry::Float32
    rz::Float32
    cx::Float32
    cy::Float32
    cz::Float32
    enabled::Bool
    smooth_k::Float32
    
    function HeadConfig(;
        rx::Float32=9.5f0,
        ry::Float32=9.0f0,
        rz::Float32=9.5f0,
        cx::Float32=0.0f0,
        cy::Float32=0.0f0,
        cz::Float32=0.0f0,
        enabled::Bool=false,  # Disabled by default for GPU training stability
        smooth_k::Float32=10.0f0
    )
        @assert rx > 0 "rx must be positive"
        @assert ry > 0 "ry must be positive"
        @assert rz > 0 "rz must be positive"
        @assert smooth_k > 0 "smooth_k must be positive"
        new(rx, ry, rz, cx, cy, cz, enabled, smooth_k)
    end
end

"""
    compute_head_indicator_gpu(x, y, z, rx, ry, rz, cx, cy, cz, smooth_k)

GPU-совместимая функция вычисления индикатора эллипсоида.
Использует гладкую сигмоидную аппроксимацию для избежания скалярного индексирования.

# Параметры:
- x, y, z: координаты (могут быть скалярами или массивами)
- rx, ry, rz: радиусы эллипсоида
- cx, cy, cz: центр эллипсоида
- smooth_k: крутизна перехода

# Возвращает:
- 1.0 внутри эллипсоида, 0.0 снаружи (плавный переход)
"""
function compute_head_indicator_gpu(x, y, z, rx, ry, rz, cx, cy, cz, smooth_k)
    # Нормированные расстояния до центра
    dx = (x .- cx) ./ rx
    dy = (y .- cy) ./ ry
    dz = (z .- cz) ./ rz
    
    # Квадрат нормированного расстояния
    dist_sq = dx .* dx .+ dy .* dy .+ dz .* dz
    
    # Гладкая аппроксимация: 1 внутри, 0 снаружи
    # Используем сигмоиду от (1 - dist_sq)
    return sigmoid_smooth(1.0f0 .- dist_sq, smooth_k)
end

"""
    is_inside_ellipsoid(x, y, z, head_config::HeadConfig)

Проверяет, находится ли точка внутри эллипсоида.

Формула: (x-cx)²/rx² + (y-cy)²/ry² + (z-cz)²/rz² <= 1
"""
function is_inside_ellipsoid(x, y, z, head_config::HeadConfig)
    if !head_config.enabled
        return true
    end
    
    dx = (x - head_config.cx) / head_config.rx
    dy = (y - head_config.cy) / head_config.ry
    dz = (z - head_config.cz) / head_config.rz
    
    return dx*dx + dy*dy + dz*dz <= 1.0f0
end

"""
    head_indicator(x, y, z, head_config::HeadConfig)

GPU-совместимая индикаторная функция эллипсоида.
Работает как с скалярами, так и с массивами (GPU/CPU).

Возвращает 1.0 если точка внутри эллипсоида, 0.0 иначе (с плавным переходом).
"""
function head_indicator(x, y, z, head_config::HeadConfig)
    if !head_config.enabled
        # Возвращаем единицу для всех входов - используем broadcasting
        if CUDA.functional() && (isa(x, CuArray) || isa(y, CuArray) || isa(z, CuArray))
            return CUDA.fill(1.0f0, size(x))
        else
            return 1.0f0
        end
    end
    
    return compute_head_indicator_gpu(
        x, y, z,
        head_config.rx, head_config.ry, head_config.rz,
        head_config.cx, head_config.cy, head_config.cz,
        head_config.smooth_k
    )
end

"""
    head_indicator_batched(coords::AbstractMatrix, head_config::HeadControl)

GPU-совместимая версия для батчей координат [3, N].

coords: матрица [3, N] где N - количество точек
"""
function head_indicator_batched(coords::AbstractMatrix, head_config::HeadConfig)
    if size(coords, 1) != 3
        throw(ArgumentError("coords must have 3 rows (x, y, z)"))
    end
    
    if !head_config.enabled
        return CUDA.fill(1.0f0, size(coords, 2))
    end
    
    x = coords[1:1, :]  # [1, N]
    y = coords[2:2, :]  # [1, N]
    z = coords[3:3, :]  # [1, N]
    
    return compute_head_indicator_gpu(
        x, y, z,
        head_config.rx, head_config.ry, head_config.rz,
        head_config.cx, head_config.cy, head_config.cz,
        head_config.smooth_k
    )
end

end  # module HeadConstraints