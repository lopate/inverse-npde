"""
    Optimization

Модуль для создания функций потерь, настройки оптимизации и callback функций
для решения обратной задачи ЭЭГ с использованием NeuralPDE.

Основные компоненты:
- Создание дополнительных функций потерь для измеренных данных
- Callback функции для логирования и мониторинга
- Настройка оптимизаторов
- Создание discretization объекта для NeuralPDE
"""

module Optimization

using NeuralPDE: GradientScaleAdaptiveLoss, NonAdaptiveLoss, PhysicsInformedNN, symbolic_discretize, QuasiRandomTraining, LogOptions
using Optimization, OptimizationOptimJL, OptimizationOptimisers, OptimizationOptimisers, LineSearches, TensorBoardLogger
using Integrals
using ProgressBars, Printf, CUDA, Lux, LuxCUDA, ComponentArrays
using CUDA: cpu_device, gpu_device, allowscalar
using Printf
using Zygote
using Statistics
using Dierckx
using ChainRulesCore

# Разрешаем скалярное индексирование для обхода проблем с GPU массивами
allowscalar(true)
using ..PDEDefinitions: PhysicalConstants
using ..NeuralNetwork: NeuralNetworkConfig

# Экспортируем основные функции
export OptimizationConfig, LossFunctionConfig
export create_additional_loss, create_optimization_callback
export setup_optimization, create_discretization
export log_training_progress, validate_optimization_config
export mc_integral_gpu, compute_field_energy_loss
export data_loss, derivative_loss

"""
    mc_integral_gpu(f_batch, θ, lb, ub, N)

Простой метод Монте-Карло на GPU для вычисления интеграла с параметрами нейросети.
Генерирует N случайных точек в гиперкубе [0,1]^d и масштабирует их в домен [lb, ub].

Использует ignore_derivatives() для исключения CUDA.rand из AD графа Zygote.

# Аргументы
- `f_batch`: Функция, принимающая матрицу [d, N] и параметры θ, возвращающая вектор размера N
- `θ`: Параметры нейросети
- `lb`: Нижние границы домена (вектор размера d)
- `ub`: Верхние границы домена (вектор размера d)  
- `N`: Количество точек для семплирования

# Возвращает
- Приближение интеграла (скаляр)
"""
function mc_integral_gpu(f_batch, θ, lb::Vector, ub::Vector, N::Int)
    d = length(lb)
    dev = Lux.gpu_device()
    
    # Генерируем случайные точки в [0,1]^d на GPU
    # Используем ignore_derivatives() для исключения из AD графа Zygote
    X = ignore_derivatives() do
        CUDA.rand(Float32, d, N)
    end
    
    # Масштабируем в домен [lb, ub]
    # Явно переносим на GPU - используем |> dev для переноса
    scale_vec = (ub .- lb) |> dev
    offset_vec = lb |> dev
    X_scaled = offset_vec .+ X .* scale_vec
    
    # Применяем функцию-батч ко всем точкам сразу
    # f_batch принимает [d, N] и θ, возвращает [N]
    vals = f_batch(X_scaled, θ)
    
    # Вычисляем среднее и умножаем на объём домена
    volume = prod(ub .- lb)
    return mean(vals) * volume
end

"""
    compute_field_energy_loss(phi_pred_fun, θ, measured_time, inner_lb, inner_ub, field_energy_scale, num_field_time_samples, N_mc)

Вычисляет field energy loss с нормировкой на объём домена.

Эта функция:
1. Сама создаёт time_points на основе measured_time и num_field_time_samples
2. Вычисляет интеграл энергии E_field = ∫ (ρ·φ + A·j) dV используя mc_integral_gpu
3. Нормирует на объём: E_field_normalized = E_field / volume
4. Вычисляет L_field = exp(-E_field_normalized / field_energy_scale)

Нормировка предотвращает переполнение градиента при больших объёмах домена.

# Аргументы
- `phi_pred_fun`: Функция предсказания сети (x, θ) -> pred
- `θ`: Параметры сети
- `measured_time`: Вектор временных точек измеренных данных (для определения диапазона)
- `inner_lb`: Нижние границы внутренней области [x, y, z]
- `inner_ub`: Верхние границы внутренней области [x, y, z]
- `field_energy_scale`: Масштаб для экспоненциального лосса
- `num_field_time_samples`: Количество временных точек для семплирования (по умолчанию 5)
- `N_mc`: Количество точек для Monte Carlo интегрирования (по умолчанию 1000)

# Возвращает
- NamedTuple с полями:
  - `E_field`: Интеграл энергии (без нормировки)
  - `E_field_normalized`: Нормированная плотность энергии (на единицу объёма)
  - `L_field`: Экспоненциальный лосс
"""
function compute_field_energy_loss(phi_pred_fun, θ, measured_time, inner_lb::Vector, inner_ub::Vector, 
                                   field_energy_scale::T, num_field_time_samples::Int=5, N_mc::Int=1000) where T <: Real
    dev = Lux.gpu_device()
    
    # Сами создаём time_points на основе measured_time и num_field_time_samples
    if length(measured_time) < 2
        error("measured_time must contain at least 2 distinct time points")
    end
    t_min = minimum(measured_time)
    t_max = maximum(measured_time)
    time_points = collect(range(t_min, t_max, length=num_field_time_samples))
    
    # Создаём обёртку которая принимает x_spatial и θ_inner - это работает с Zygote
    # θ передаётся явно как аргумент, а не захватывается замыканием
    field_energy_mc = let times = time_points, n_times_val = length(time_points)
        function(x_spatial::AbstractArray, θ_inner)
            batch_size = size(x_spatial, 2)
            n_times = n_times_val
            
            # Vectorized создание временной координаты - без цикла for!
            # time_idx = [1,1,1,...,1, 2,2,2,...,2, ..., n_times] (batch_size раз для каждого времени)
            # Используем repeat на CPU, затем переносим на GPU
            time_idx_cpu = repeat(1:n_times, inner=batch_size)
            time_idx_gpu = time_idx_cpu |> dev
            
            # Создаём массив временных значений используя vectorized индексацию
            # times - это Vector{Float64}, преобразуем в GPU массив
            times_gpu = Float32.(times) |> dev
            # Индексируем вектор times по вектору индексов - полностью vectorized
            t_vals = ignore_derivatives() do
                times_gpu[time_idx_gpu]
            end
            
            # Создаём полный батч: [x,y,z,t] - используем vectorized reshape и broadcast
            # x_spatial имеет размер [3, batch_size]
            # Нужно повторить каждую пространственную точку n_times раз
            X_full = ignore_derivatives() do
                # Повторяем пространственные координаты n_times раз
                x_spatial_expanded = repeat(x_spatial, 1, n_times)
                # Добавляем временную координату четвёртой строкой
                CUDA.vcat(x_spatial_expanded, reshape(t_vals, 1, :))
            end
            
            θ_gpu = θ_inner |> dev
            pred = phi_pred_fun(X_full, θ_gpu)
            
            # pred имеет размер [8, batch_size * n_times]
            # reshape для усреднения по времени
            phi = reshape(pred[1, :], n_times, batch_size)
            Ax = reshape(pred[2, :], n_times, batch_size)
            Ay = reshape(pred[3, :], n_times, batch_size)
            Az = reshape(pred[4, :], n_times, batch_size)
            rho = reshape(pred[5, :], n_times, batch_size)
            jx = reshape(pred[6, :], n_times, batch_size)
            jy = reshape(pred[7, :], n_times, batch_size)
            jz = reshape(pred[8, :], n_times, batch_size)
            
            # Энергия: ρ·φ + A·j
            energy_per_time = rho .* phi .+ Ax.*jx .+ Ay.*jy .+ Az.*jz
            energy_mean = mean(energy_per_time, dims=1)
            
            return vec(energy_mean)
        end
    end
    
    # Вычисляем интеграл - передаём θ как часть параметров
    E_field = mc_integral_gpu(field_energy_mc, θ, inner_lb, inner_ub, N_mc)
    
    # Нормировка на объём: получаем среднюю плотность энергии
    volume = prod(Float32.(inner_ub) .- Float32.(inner_lb))
    E_field_normalized = E_field / volume
    
    # Экспоненциальный лосс с нормированной энергией
    # L_field = exp(-E_field_normalized / scale)
    # При большой энергии -> маленький лосс (стимулируем максимизацию энергии)
    scale_Float = Float32(convert(T, field_energy_scale))
    L_field = exp(-E_field_normalized / scale_Float)
    
    return (E_field=E_field, E_field_normalized=E_field_normalized, L_field=L_field)
end

# Структура конфигурации оптимизации
struct OptimizationConfig
    optimizer::Symbol     # Тип оптимизатора (:adam, :lbfgs, :adamw)
    learning_rate::Float64 # Скорость обучения
    max_iterations::Int   # Максимальное количество итераций
    log_frequency::Int    # Частота логирования
    use_tensorboard::Bool # Использовать ли TensorBoard
    log_directory::String # Директория для логов
    
    function OptimizationConfig(; optimizer=:adam, learning_rate=0.001, max_iterations=3000, 
                               log_frequency=50, use_tensorboard=true, log_directory="logs/inverse_npde_exp")
        return new(Symbol(optimizer), learning_rate, max_iterations, log_frequency, use_tensorboard, log_directory)
    end
end

# Структура конфигурации функции потерь
struct LossFunctionConfig
    lambda_pde::Float64                           # Вес потерь PDE
    lambda_bc::Float64                           # Вес потерь граничных условий
    lambda_data_init::Float64                    # Начальное значение веса данных
    lambda_min::Union{Float64, Nothing}          # Минимальное значение lambda_data
    lambda_max::Union{Float64, Nothing}          # Максимальное значение lambda_data
    measured_points::Vector                      # Исходные измеренные точки (для совместимости)
    measured_points_coords::Matrix{Float32}      # Батчированные координаты [4, N]
    measured_points_values::Vector{Float32}      # Батчированные значения [N]
    measured_time::Vector{Float32}               # Уникальные временные координаты, упорядоченные
    n_measured_points::Int                       # Количество измеренных точек
    num_sensors::Int                              # Количество датчиков (временной группировки)
    num_time_steps::Int                           # Количество временных шагов в измерениях
    derivatives::Vector{Float32}                  # Вектор производных (diff по времени, vec of [num_sensors, num_time_steps-1])
    t_mid_for_deriv::Vector{Float32}             # Временные точки в середине интервалов (где вычисляются производные)
    deriv_coords::Matrix{Float32}                # [x,y,z,t_mid] координаты для оценки производных [4, num_sensors*(num_time_steps-1)]
    lambda_schedule_type::Symbol                  # :improvement (default)
    lambda_schedule::Dict{String,Any}             # Dict with scheduling params
    lambda_time::Float64                          # Вес для временной производной
    
    # Поля для регуляризации энергии поля (L_field)
    lambda_field::Float64                        # Вес регуляризации поля (default: 1.0)
    field_energy_scale::Float64                  # Масштаб энергии для экспоненциального лосса (default: 3.0)
    num_field_time_samples::Int                  # Количество временных срезов для интегрирования энергии поля
    inner_domain::Tuple{Vector{Float64}, Vector{Float64}}  # (lb, ub) для внутренней области интегрирования (x, y, z)
    
    # Поля для адаптивного взвешивания (GradientScaleAdaptiveLoss)
    enable_adaptive_loss::Bool                    # Включить адаптивное взвешивание
    adaptive_loss_reweight_every::Int             # Как часто пересчитывать веса (в итерациях)
    adaptive_weight_inertia::Float64              # Инерция для экспоненциального сглаживания весов
    
    # refs для хранения промежуточных значений лосса (избегаем пересчёта в callback)
    field_loss_ref::Ref{NamedTuple}               # Ref для E_field, E_field_normalized, L_field
    data_loss_ref::Ref{NamedTuple}                # Ref для mse, deriv_loss, total

    function LossFunctionConfig(; 
        lambda_pde::Float64=1.0, 
        lambda_bc::Float64=1.0, 
        lambda_data_init::Float64=1.0,
        lambda_min::Union{Float64, Nothing}=nothing, 
        lambda_max::Union{Float64, Nothing}=nothing,
        measured_points::Vector=Vector{Any}[],
        lambda_schedule_type::Symbol = :improvement,
        lambda_schedule::Dict{String,Any} = Dict(),
        lambda_time::Float64=1.0,
        # Параметры регуляризации энергии поля
        lambda_field::Float64=1.0,
        field_energy_scale::Float64=3.0,
        num_field_time_samples::Int=5,  # Количество временных срезов для интегрирования
        # Параметры домена - теперь напрямую для автоматического вычисления inner_domain
        x_range::Vector{Float64}=[-10.0, 10.0],
        y_range::Vector{Float64}=[-10.0, 10.0],
        z_range::Vector{Float64}=[-10.0, 10.0],
        t_range::Vector{Float64}=[0.0, 1.0],
        pml_thickness_ratio::Float64=0.1,  # Толщина PML слоя
        # Параметры адаптивного взвешивания (GradientScaleAdaptiveLoss)
        enable_adaptive_loss::Bool=false,  # По умолчанию отключено
        adaptive_loss_reweight_every::Int=100,  # Пересчитывать веса каждые 100 итераций
        adaptive_weight_inertia::Float64=0.9  # Инерция для сглаживания весов
    )
        if lambda_min !== nothing && lambda_max !== nothing && lambda_min >= lambda_max
            throw(ArgumentError("lambda_min должно быть меньше lambda_max"))
        end
        
        # Конвертируем измеренные точки в батчированную форму для GPU операций
        n_points = length(measured_points)

        # Инициализации по умолчанию
        coords = Matrix{Float32}(undef, 4, 0)
        values = Vector{Float32}(undef, 0)
        num_sensors = 0
        num_time_steps = 0
        derivatives = Vector{Float32}(undef, 0)
        measured_time = Vector{Float32}(undef, 0)
        t_mid_for_deriv = Vector{Float32}(undef, 0)
        deriv_coords = Matrix{Float32}(undef, 4, 0)

        if n_points > 0
            # Объединяем все точки в матрицу [5, N]
            all_data = hcat(measured_points...)

            # Извлекаем координаты [4, N] и значения [N]
            coords = Float32.(all_data[1:4, :])
            values = vec(Float32.(all_data[5, :]))

            # Для анализа структуры (уникальные позиции, вычисление производных) работаем с CPU-версией данных
            all_data_cpu = all_data isa CuArray ? (all_data |> cpu_device()) : all_data
            
            # Переносим coords на CPU для безопасного скалярного индексирования
            coords = coords isa CuArray ? (coords |> cpu_device()) : coords

            # Попытка вывести структуру измерений: число датчиков и число временных шагов
            # Определяем уникальные позиции датчиков (по x,y,z) в порядке первого появления
            pos = all_data_cpu[1:3, :]
            uniq_map = Dict{Tuple{Float32,Float32,Float32}, Int}()
            sensors = Tuple{Float32,Float32,Float32}[]
            for j in 1:size(pos, 2)
                key = (Float32(pos[1, j]), Float32(pos[2, j]), Float32(pos[3, j]))
                if !haskey(uniq_map, key)
                    uniq_map[key] = length(sensors) + 1
                    push!(sensors, key)
                end
            end
            num_sensors = length(sensors)
            if num_sensors == 0
                num_sensors = 1
            end

            # Вычисляем число временных шагов при условии упорядочения данных (sensors fastest)
            if n_points % num_sensors == 0
                num_time_steps = Int(n_points ÷ num_sensors)
            else
                num_time_steps = 1
            end

            # Если есть хотя бы два временных шага, вычисляем интерполяцию и производные по времени
            if num_time_steps > 1
                # Для аккуратности используем CPU-значения для вычисления интерполяции и diff
                values_cpu = vec(Float32.(all_data_cpu[5, :]))
                measured_mat = reshape(values_cpu, num_sensors, num_time_steps)

                # Получаем уникальные временные координаты (CPU)
                t_vals = vec(all_data_cpu[4, :])
                t_unique = collect(unique(t_vals))
                sort!(t_unique)
                measured_time = Float32.(t_unique)

                # Если есть больше одного временного шага, используем точный кубический сплайн
                # на неравномерной сетке (Dierckx.jl) и вычисляем производную сплайна
                if length(t_unique) > 1
                    # средние точки интервалов, где будет оцениваться производная
                    t_mid = Float64.((Float32.(t_unique[1:end-1]) .+ Float32.(t_unique[2:end])) ./ 2f0)
                    t_mid_for_deriv = Float32.(t_mid)

                    # Создаем матрицу производных (num_sensors, num_time_steps-1)
                    deriv_mat = zeros(Float32, num_sensors, num_time_steps - 1)
                    x = Float64.(t_unique)
                    for si in 1:num_sensors
                        y = Float64.(measured_mat[si, :])
                        # Построение кубического сплайна на неравномерной сетке
                        s = Dierckx.Spline1D(x, y, k=3)
                        # Оценка первой производной в средних точках интервалов
                        deriv_vals = Dierckx.derivative(s, t_mid)
                        deriv_mat[si, :] = Float32.(deriv_vals)
                    end

                    derivatives = vec(deriv_mat)
                    
                    # Создаем координаты для оценки сети в точках производных (t_mid)
                    # Эти координаты используются для вычисления производных на стороне предсказания
                    n_deriv_points = num_sensors * (num_time_steps - 1)
                    deriv_coords = zeros(Float32, 4, n_deriv_points)
                    
                    # Группируем координаты по датчикам и интервалам
                    for i in 1:n_deriv_points
                        # Какой датчик и какой интервал
                        sensor_idx = ((i-1) % num_sensors) + 1
                        interval_idx = div(i-1, num_sensors) + 1
                        
                        # Получаем пространственные координаты из батча (coords = measured_points_coords)
                        # (они одинаковы для всех временных интервалов)
                        sensor_point_idx = (interval_idx - 1) * num_sensors + sensor_idx
                        if sensor_point_idx <= size(coords, 2)
                            deriv_coords[1:3, i] = coords[1:3, sensor_point_idx]
                        else
                            # Fallback: используем координаты первого датчика этого интервала
                            base_point_idx = max(1, min((interval_idx - 1) * num_sensors + 1, size(coords, 2)))
                            deriv_coords[1:3, i] = coords[1:3, base_point_idx]
                        end
                        deriv_coords[4, i] = t_mid_for_deriv[interval_idx]
                    end
                else
                    derivatives = Vector{Float32}(undef, 0)
                    t_mid_for_deriv = Vector{Float32}(undef, 0)
                    deriv_coords = Matrix{Float32}(undef, 4, 0)
                end
            else
                derivatives = Vector{Float32}(undef, 0)
            end
        end

        inner_domain = (Float64[], Float64[])  # По умолчанию: пустой домен
        
        # Автоматически вычисляем inner_domain из x_range, y_range, z_range и pml_thickness_ratio
        x_min, x_max = x_range[1], x_range[2]
        y_min, y_max = y_range[1], y_range[2]
        z_min, z_max = z_range[1], z_range[2]
        
        # Толщина PML слоя
        Dx = (x_max - x_min) * pml_thickness_ratio
        Dy = (y_max - y_min) * pml_thickness_ratio
        Dz = (z_max - z_min) * pml_thickness_ratio
        
        # Внутренняя область
        x_inner_min = x_min + Dx
        x_inner_max = x_max - Dx
        y_inner_min = y_min + Dy
        y_inner_max = y_max - Dy
        z_inner_min = z_min + Dz
        z_inner_max = z_max - Dz
        
        # Сохраняем границы внутренней области для IntegralProblem (только x, y, z - без времени)
        inner_domain = ([x_inner_min, y_inner_min, z_inner_min],
                      [x_inner_max, y_inner_max, z_inner_max])

        # Инициализируем Refs для хранения промежуточных значений лосса
        # Это позволяет избежать пересчёта в callback
        field_loss_ref = Ref{NamedTuple}((E_field=0.0f0, E_field_normalized=0.0f0, L_field=0.0f0))
        data_loss_ref = Ref{NamedTuple}((mse=0.0f0, deriv=0.0f0, total=0.0f0))

        return new(lambda_pde, lambda_bc, lambda_data_init, lambda_min, lambda_max,
                   measured_points, coords, values, measured_time, n_points,
                   num_sensors, num_time_steps, derivatives, t_mid_for_deriv, deriv_coords, 
                    lambda_schedule_type, lambda_schedule, lambda_time,
                    lambda_field, field_energy_scale, num_field_time_samples, inner_domain,
                    enable_adaptive_loss, adaptive_loss_reweight_every, adaptive_weight_inertia,
                    field_loss_ref, data_loss_ref)
    end
end


function create_additional_loss(loss_config::LossFunctionConfig, lambda_data_ref::Ref{Float64})
    n_points = loss_config.n_measured_points
    coords_batch = loss_config.measured_points_coords
    values_batch = loss_config.measured_points_values
    deriv_meas = loss_config.derivatives
    deriv_coords = loss_config.deriv_coords
    num_sensors = loss_config.num_sensors
    num_time_steps = loss_config.num_time_steps
    lambda_time = loss_config.lambda_time
    measured_time = loss_config.measured_time
    
    # Параметры для регуляризации энергии поля
    lambda_field = loss_config.lambda_field
    field_energy_scale = loss_config.field_energy_scale
    inner_domain = loss_config.inner_domain
    num_field_time_samples = loss_config.num_field_time_samples
    
    # refs для хранения значений лосса
    field_loss_ref = loss_config.field_loss_ref
    data_loss_ref = loss_config.data_loss_ref
    
    # Проверяем, нужно ли вычислять field loss (домен не пустой)
    compute_field = lambda_field > 0 && length(inner_domain[1]) > 0
    
    function additional_loss(phi_pred_fun, θ, p_)
        # Получаем GPU устройство
        dev = Lux.gpu_device()
        
        # Переносим батчированные данные на GPU
        coords_gpu = coords_batch |> dev
        values_gpu = values_batch |> dev
        θ_gpu = θ |> dev
        
        # Используем функции data_loss и derivative_loss вместо дублирования кода
        data_loss_computed = data_loss(phi_pred_fun, θ_gpu, coords_gpu, values_gpu, n_points)
        
        # Вычисляем derivative loss если доступны производные
        deriv_loss = Float32(0.0)
        if size(deriv_coords, 2) > 0 && length(deriv_meas) > 0
            deriv_loss = derivative_loss(
                phi_pred_fun, θ_gpu,
                coords_gpu, values_gpu, n_points,
                deriv_meas, deriv_coords,
                num_sensors, num_time_steps, measured_time
            )
        end

        # Объединяем MSE и derivative loss, затем умножаем на detached lambda_data
        lambda_detached = Zygote.dropgrad(lambda_data_ref[])
        alpha_data_constraint = Float32(10.0)  # Масштаб для превращения data_loss в ограничение
        #Превращаем data_loss в ограничение
        data_constraint = data_loss_computed + alpha_data_constraint * (data_loss_computed^2) # Это превращает data_loss в мягкое ограничение, которое растёт экспоненциально при увеличении data_loss, но сохраняет градиенты даже при больших значениях. 
        # Это позволяет сохранять градиенты даже при больших значениях data_loss, так как экспонента будет расти, но градиент будет сохраняться.
        total = data_constraint + lambda_time * deriv_loss
        result = total * lambda_detached
        
        # Записываем data loss в ref для callback (без пересчёта)
        data_loss_ref[] = (mse=Float32(data_constraint), deriv=Float32(deriv_loss), total=Float32(total))
        
        # === ДОБАВЛЕНО: Регуляризация энергии поля через интеграл ===
        if compute_field
            # Используем IntegralProblem с VEGAS для правильного интегрирования
            # Интеграл: E_field = ∫ ρ·φ dV + ∫ A·j dV
            # 
            # Мы интегрируем по пространству (x, y, z) и усредняем по времени
            
            # Получаем границы внутренней области
            inner_lb, inner_ub = loss_config.inner_domain
            
            # Проверяем, что домен 3D
            if length(inner_lb) != 3 || length(inner_ub) != 3
                @warn "inner_domain должен быть 3D (x, y, z), получен $(length(inner_lb))D"
            else
                # Функция compute_field_energy_loss сама создаёт time_points
                # на основе measured_time и num_field_time_samples
                
                # Используем новую функцию compute_field_energy_loss
                # Она автоматически:
                # 1. Создаёт time_points на основе measured_time и num_field_time_samples
                # 2. Нормирует энергию на объём
                field_result = compute_field_energy_loss(
                    phi_pred_fun, 
                    θ_gpu, 
                    measured_time,
                    inner_lb, 
                    inner_ub, 
                    field_energy_scale,
                    num_field_time_samples,
                    1000  # N_mc
                )
                
                # Используем L_field с нормированной энергией
                L_field = field_result.L_field
                
                # Записываем field loss в ref для callback (без пересчёта)
                field_loss_ref[] = (E_field=Float32(field_result.E_field), 
                                   E_field_normalized=Float32(field_result.E_field_normalized), 
                                   L_field=Float32(L_field))
                
                # Добавляем к общему результату с весом lambda_field
                result = result + lambda_field * L_field
            end
        end

        return result
    end
    return additional_loss
end

"""
    data_loss(phi_pred_fun, θ, coords_gpu, values_gpu, n_points)

Вычисляет MSE loss между предсказанными и измеренными значениями потенциала.

# Аргументы
- `phi_pred_fun`: Функция предсказания сети
- `θ`: Параметры сети
- `coords_gpu`: GPU массив координат [4, N]
- `values_gpu`: GPU массив измеренных значений [N]
- `n_points`: Количество точек

# Возвращает
- MSE значение
"""
function data_loss(phi_pred_fun, θ, coords_gpu, values_gpu, n_points)
    # Векторизованный вызов сети для всех точек
    # Сеть выводит [φ, Ax, Ay, Az, ρ, jx, jy, jz], нам нужен только φ (первая строка)
    pred_all = phi_pred_fun(coords_gpu, θ)
    phi_pred = pred_all[1, :]  # Извлекаем только потенциал φ
    
    # Батчированное вычисление среднеквадратичной ошибки
    # Полностью на GPU без скалярных операций
    mse = sum(abs2.(phi_pred .- values_gpu)) / max(n_points, 1)
    
    return mse
end

"""
    derivative_loss(phi_pred_fun, θ, coords_gpu, values_gpu, n_points, deriv_meas, deriv_coords, 
                   num_sensors, num_time_steps, measured_time)

Вычисляет loss по производной по времени.

# Аргументы
- `phi_pred_fun`: Функция предсказания сети
- `θ`: Параметры сети
- `coords_gpu`: GPU массив координат [4, N]
- `values_gpu`: GPU массив измеренных значений [N]
- `n_points`: Количество точек
- `deriv_meas`: Вектор измеренных производных
- `deriv_coords`: Координаты для оценки производных [4, M]
- `num_sensors`: Количество датчиков
- `num_time_steps`: Количество временных шагов
- `measured_time`: Вектор временных точек

# Возвращает
- Значение derivative loss
"""
function derivative_loss(phi_pred_fun, θ, coords_gpu, values_gpu, n_points, 
                         deriv_meas, deriv_coords, num_sensors, num_time_steps, measured_time)
    # Производная по времени: требуется предварительно вычисленная производная
    if length(deriv_meas) == 0
        throw(ArgumentError("Measured temporal derivatives must be provided in loss_config.derivatives before training."))
    end

    if num_time_steps < 2 || num_sensors < 1
        throw(ArgumentError("Insufficient temporal structure in measured data to compute derivative loss (num_time_steps=$(num_time_steps), num_sensors=$(num_sensors))"))
    end

    dev = Lux.gpu_device()
    
    # === Оценка производных на точках t_mid ===
    # Вместо использования конечных разностей на граничных точках,
    # используем сеть для вычисления производных в средних точках интервалов
    if size(deriv_coords, 2) > 0
        deriv_coords_gpu = deriv_coords |> dev
        deriv_gpu = Float32.(deriv_meas) |> dev
        
        # Оцениваем сеть в точках t_mid
        pred_at_mid = phi_pred_fun(deriv_coords_gpu, θ)
        phi_pred_at_mid = pred_at_mid[1, :]
        
        # Получаем предсказания для всех точек
        pred_all = phi_pred_fun(coords_gpu, θ)
        phi_pred = pred_all[1, :]
        
        # reshape predicted into (num_sensors, num_time_steps)
        pred_mat = reshape(phi_pred, num_sensors, num_time_steps)

        if length(measured_time) < 2
            throw(ArgumentError("Measured time vector must contain at least two distinct times to compute dt"))
        end
        dt_vec = Float32.(diff(measured_time))
        dt_gpu = dt_vec |> dev
        pred_deriv = diff(pred_mat; dims=2) ./ reshape(dt_gpu, 1, :)
        pred_deriv_vec = vec(pred_deriv)

        # compare with measured derivatives
        deriv_loss_val = mean((pred_deriv_vec .- deriv_gpu).^2)
    else
        deriv_loss_val = Float32(0.0)
    end
    
    return deriv_loss_val
end


"""
Создает discretization объект для NeuralPDE с адаптивным весом данных.

# Возвращает
- Кортеж (discretization, lambda_data_ref) где lambda_data_ref - ссылка на текущий вес данных
"""
function create_discretization(chain, ps, loss_config::LossFunctionConfig, 
                              opt_config::OptimizationConfig, domain_config,
                              lambda_data_ref::Ref{Float64}=Ref{Float64}(loss_config.lambda_data_init))
    
    # Создаем стратегию обучения
    strategy = QuasiRandomTraining(domain_config["num_points"])
    
    # Дополнительная функция потерь с адаптивным весом
    additional_loss_func = create_additional_loss(loss_config, lambda_data_ref)
    
    # === Создаём адаптивный лосс для PDE и BC ===
    # NeuralPDE поддерживает GradientScaleAdaptiveLoss и NonAdaptiveLoss
    if loss_config.enable_adaptive_loss
        # Используем адаптивное взвешивание с пересчётом весов
        # Сигнатура: GradientScaleAdaptiveLoss(reweight_every::Int64; weight_change_inertia, pde_loss_weights, bc_loss_weights, additional_loss_weights)
        adaptive_loss = GradientScaleAdaptiveLoss(
            loss_config.adaptive_loss_reweight_every;
            weight_change_inertia = loss_config.adaptive_weight_inertia,
            pde_loss_weights = loss_config.lambda_pde,
            bc_loss_weights = loss_config.lambda_bc,
            additional_loss_weights = Float32(1.0)  # Для data loss
        )
        @info "Using GradientScaleAdaptiveLoss with reweight_every=$(loss_config.adaptive_loss_reweight_every), inertia=$(loss_config.adaptive_weight_inertia)"
    else
        # Используем статическое взвешивание с явными весами
        adaptive_loss = NonAdaptiveLoss(
            pde_loss_weights = loss_config.lambda_pde,
            bc_loss_weights = loss_config.lambda_bc
        )
        @info "Using NonAdaptiveLoss with lambda_pde=$(loss_config.lambda_pde), lambda_bc=$(loss_config.lambda_bc)"
    end
    
    # GPU-first construction: expect chain and params to be on the device desired by the user
    @info "Constructing PhysicsInformedNN (GPU-first). Ensure chain and params are on desired device"
    
    # Обработка настроек логирования - NeuralPDE не поддерживает nothing в log_options
    if opt_config.use_tensorboard
        log_options = LogOptions(; log_frequency = opt_config.log_frequency)
        discretization = PhysicsInformedNN(chain, strategy; 
                                          init_params = ps, 
                                          additional_loss = additional_loss_func,
                                          adaptive_loss = adaptive_loss,
                                          log_options = log_options)
    else
        # При use_tensorboard=false создаем без параметра log_options
        # Это избегает ошибки MethodError при передаче nothing
        discretization = PhysicsInformedNN(chain, strategy; 
                                          init_params = ps, 
                                          additional_loss = additional_loss_func,
                                          adaptive_loss = adaptive_loss)
    end

    return discretization, lambda_data_ref
end

"""
    create_optimization_callback(opt_config::OptimizationConfig, discretization, pde_system, bcs, domains,
                                 loss_config::LossFunctionConfig, lambda_data_ref::Ref{Float64},
                                 data_loss_raw_func::Function)

Создает callback функцию для мониторинга обучения с адаптивным балансом lambda_data.

# Аргументы
- `opt_config`: Конфигурация оптимизации
- `discretization`: Объект дискретизации NeuralPDE
- `pde_system`: Система PDE
- `bcs`: Граничные условия
- `domains`: Области
- `loss_config`: Конфигурация функции потерь
- `lambda_data_ref`: Ссылка на текущее значение веса данных
- `data_loss_raw_func`: Функция для вычисления "сырого" data loss без веса

# Адаптивный баланс
# Использует improvement-based планировщик: увеличивает λ при стагнации data loss.
"""
function create_optimization_callback(opt_config::OptimizationConfig, discretization, pde_system, bcs, domains,
                                      loss_config::LossFunctionConfig, lambda_data_ref::Ref{Float64},
                                      data_loss_raw_func::Function)
    maxiters = opt_config.max_iterations
    
    # Создаем логгер для TensorBoard
    logger = opt_config.use_tensorboard ? 
        TBLogger(opt_config.log_directory) : nothing
    
    # Получаем функции потерь
    sym_prob = symbolic_discretize(pde_system, discretization)
    pde_inner_loss_functions = sym_prob.loss_functions.pde_loss_functions
    bcs_inner_loss_functions = sym_prob.loss_functions.bc_loss_functions
    
    iter = 0
    pbar = ProgressBar(1:maxiters, printing_delay=1.)

    # Параметры для регуляризации энергии поля
    lambda_field = loss_config.lambda_field
    
    # refs для чтения значений лосса (заполняются в create_additional_loss)
    field_loss_ref = loss_config.field_loss_ref
    data_loss_ref = loss_config.data_loss_ref
    
    # State for improvement-based lambda scheduling
    # Initialized here so the closure can mutate them across iterations
    window_size = get(loss_config.lambda_schedule, "window_size", 10)
    increase_factor = get(loss_config.lambda_schedule, "increase_factor", 1.2)
    decrease_factor = get(loss_config.lambda_schedule, "decrease_factor", 0.98)
    patience = get(loss_config.lambda_schedule, "patience", 3)
    min_improvement = get(loss_config.lambda_schedule, "min_improvement", 1e-8)

    data_buffer = fill(Inf, window_size)
    buffer_pos = Ref(1)
    no_improve = Ref(0)
    lambda_action = Ref(0) # -1 decrease, 0 hold, 1 increase

    # field_loss_func НЕ используется - читаем из refs
    # field_loss_func = nothing

    return function (p, l)
        iter += 1
        
        # Логируем общую потерю
        if logger !== nothing
            log_value(logger, "Loss/Total", l; step=iter)
        end
        
        # Вычисляем потери по PDE
        pde_losses = map(l_ -> l_(p.u), pde_inner_loss_functions)
        L_pde = sum(pde_losses)
        
        if logger !== nothing
            for (i, pde_loss) in enumerate(pde_losses)
                log_value(logger, "Loss/PDE_$i", pde_loss; step=iter)
            end
            # Логируем суммарный PDE loss
            log_value(logger, "Loss/L_pde", L_pde; step=iter)
        end
        
        # Логируем потери по граничным условиям
        bcs_losses = map(l_ -> l_(p.u), bcs_inner_loss_functions)
        if logger !== nothing
            for (i, bc_loss) in enumerate(bcs_losses)
                log_value(logger, "Loss/BC_$i", bc_loss; step=iter)
            end
        end
        
        
        L_data_raw = data_loss_ref[].mse
        deriv_loss_val = data_loss_ref[].deriv
        
        # Improvement-based scheduling: increase λ when data loss stagnates
        if length(data_buffer) > 0
            # best observed in window BEFORE adding current value
            best_in_window = minimum(data_buffer)
        
            # if current is significantly better than best -> improvement
            if L_data_raw < best_in_window - min_improvement
                no_improve[] = 0
                # optionally decrease lambda slightly to allow flexibility
                lambda_data_new = lambda_data_ref[] * decrease_factor
                lambda_action[] = -1
            else
                no_improve[] += 1
                lambda_action[] = 0
            end
        
            # If stagnated for enough consecutive iterations -> increase lambda
            if no_improve[] >= patience
                lambda_data_new = lambda_data_ref[] * increase_factor
                no_improve[] = 0
                lambda_action[] = 1
            end
        
            # Add current value to buffer AFTER checking for improvement
            data_buffer[buffer_pos[]] = L_data_raw
            buffer_pos[] = (buffer_pos[] % length(data_buffer)) + 1

            # Apply bounds (consider both loss_config and schedule-specified bounds)
            schedule_min = get(loss_config.lambda_schedule, "lambda_min", nothing)
            schedule_max = get(loss_config.lambda_schedule, "lambda_max", nothing)
            lower = loss_config.lambda_min !== nothing ? loss_config.lambda_min : (schedule_min !== nothing ? schedule_min : 1e-8)
            upper = loss_config.lambda_max !== nothing ? loss_config.lambda_max : (schedule_max !== nothing ? schedule_max : 1e8)

            if @isdefined(lambda_data_new)
                lambda_data_new = clamp(lambda_data_new, lower, upper)
                lambda_data_ref[] = lambda_data_new
            end
        end
        
        # Логируем data loss и lambda_data
        if logger !== nothing
            log_value(logger, "Loss/L_data_raw", L_data_raw; step=iter)
            log_value(logger, "Loss/L_data_weighted", L_data_raw * lambda_data_ref[]; step=iter)
            # Log derivative loss component
            log_value(logger, "Loss/L_data_derivative", deriv_loss_val; step=iter)
            log_value(logger, "Loss/L_data_derivative_weighted", deriv_loss_val * loss_config.lambda_time; step=iter)
            log_value(logger, "Params/lambda_data", lambda_data_ref[]; step=iter)
            # iterations without improvement
            iter_no_improve = no_improve[]
            log_value(logger, "Params/iter_no_improve", iter_no_improve; step=iter)
            # action taken this iteration: -1 decrease, 0 hold, 1 increase
            log_value(logger, "Params/lambda_action", lambda_action[]; step=iter)
            
            # Логируем energy field loss из ref (без пересчёта)
            if field_loss_ref[] != nothing && haskey(field_loss_ref[], :L_field)
                log_value(logger, "Loss/E_field", field_loss_ref[].E_field; step=iter)
                log_value(logger, "Loss/E_field_normalized", field_loss_ref[].E_field_normalized; step=iter)
                log_value(logger, "Loss/L_field", field_loss_ref[].L_field; step=iter)
                log_value(logger, "Loss/L_field_weighted", field_loss_ref[].L_field * lambda_field; step=iter)
            end
        end
        
        # Читаем field metrics из ref (без пересчёта)
        E_field_val = 0.0f0
        E_field_norm_val = 0.0f0
        L_field_val = 0.0f0
        if field_loss_ref[] != nothing && haskey(field_loss_ref[], :L_field)
            E_field_val = field_loss_ref[].E_field
            E_field_norm_val = field_loss_ref[].E_field_normalized
            L_field_val = field_loss_ref[].L_field
        end
        
        # Обновляем прогресс бар - сначала продвигаем итератор, затем устанавливаем postfix
        ProgressBars.update(pbar)
        set_postfix(pbar, 
               Loss = round(l, sigdigits=3), 
               PDE = round(L_pde, sigdigits=3), 
               Data = round(L_data_raw, sigdigits=3),
               Deriv = round(deriv_loss_val, sigdigits=3),
               E_fld = round(E_field_norm_val, sigdigits=3),
               L_fld = round(L_field_val, sigdigits=3),
               λ = round(lambda_data_ref[], sigdigits=2))
        
        return false  # Продолжаем оптимизацию
    end
end

"""
    setup_optimization(opt_config::OptimizationConfig)

Настраивает оптимизатор согласно конфигурации.
"""
function setup_optimization(opt_config::OptimizationConfig)
    optimizers = Dict(
        :adam => OptimizationOptimisers.Adam(opt_config.learning_rate),
        :adamw => OptimizationOptimisers.AdamW(opt_config.learning_rate),
        :lbfgs => OptimizationOptimJL.LBFGS()
    )
    
    if !haskey(optimizers, opt_config.optimizer)
        throw(ArgumentError("Неизвестный оптимизатор: $(opt_config.optimizer). Доступны: $(keys(optimizers))"))
    end
    
    return optimizers[opt_config.optimizer]
end

"""
    log_training_progress(logger, iteration, losses, metrics)

Логирует прогресс обучения с дополнительными метриками.
"""
function log_training_progress(logger, iteration, losses, metrics::Dict{String, Any}=Dict{String, Any}())
    # Основные потери
    log_value(logger, "Loss/Total", losses.total; step=iteration)
    
    # Потери по компонентам
    if haskey(losses, :pde)
        log_value(logger, "Loss/PDE", losses.pde; step=iteration)
    end
    
    if haskey(losses, :bc)
        log_value(logger, "Loss/BC", losses.bc; step=iteration)
    end
    
    if haskey(losses, :data)
        log_value(logger, "Loss/Data", losses.data; step=iteration)
    end
    
    # Дополнительные метрики
    for (key, value) in metrics
        log_value(logger, "Metrics/$key", value; step=iteration)
    end
end

"""
    validate_optimization_config(opt_config::OptimizationConfig)

Проверяет корректность конфигурации оптимизации.
"""
function validate_optimization_config(opt_config::OptimizationConfig)
    if opt_config.learning_rate <= 0
        throw(ArgumentError("Скорость обучения должна быть положительной"))
    end
    
    if opt_config.max_iterations <= 0
        throw(ArgumentError("Максимальное количество итераций должно быть положительным"))
    end
    
    if opt_config.log_frequency <= 0
        throw(ArgumentError("Частота логирования должна быть положительной"))
    end
    
    valid_optimizers = [:adam, :adamw, :lbfgs]
    if !(opt_config.optimizer in valid_optimizers)
        throw(ArgumentError("Неизвестный оптимизатор: $(opt_config.optimizer)"))
    end
    
    return true
end

"""
    create_loss_callback(opt_config::OptimizationConfig, logger, loss_history)

Создает callback для сохранения истории потерь.
"""
function create_loss_callback(opt_config::OptimizationConfig, logger, loss_history::Vector{Float64}=Float64[])
    return function (p, l)
        push!(loss_history, l)
        
        # Логируем каждые log_frequency итераций
        if length(loss_history) % opt_config.log_frequency == 0
            log_value(logger, "Loss/History", l; step=length(loss_history))
        end
        
        return false
    end
end

"""
    create_early_stopping_patience(patience::Int, min_delta::Float64=1e-6)

Создает callback для ранней остановки.
"""
function create_early_stopping_patience(patience::Int, min_delta::Float64=1e-6)
    best_loss = Inf
    wait = 0
    
    # Используем ссылочные типы для сохранения состояния
    state = Ref{Float64}(best_loss)
    wait_ref = Ref{Int}(wait)
    
    return function (p, l)
        if l < state[] - min_delta
            state[] = l
            wait_ref[] = 0
        else
            wait_ref[] += 1
        end
        
        if wait_ref[] >= patience
            println("Ранняя остановка после $patience итераций без улучшения")
            return true  # Останавливаем оптимизацию
        end
        
        return false  # Продолжаем оптимизацию
    end
end

"""
    combine_callbacks(callbacks::Vector{Function})

Комбинирует несколько callback функций в одну.
"""
function combine_callbacks(callbacks::Vector{Function})
    return function (p, l)
        should_stop = false
        for callback in callbacks
            stop = callback(p, l)
            should_stop = should_stop || stop
        end
        return should_stop
    end
end

end # module