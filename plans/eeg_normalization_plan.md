# План реализации нормировки данных в EEGInverseSolver

## Цель
Реализовать нормировку значений на датчиках по среднему модулю всех измерений для обеспечения одинаковой масштабности при решении обратной задачи ЭЭГ.

## Архитектура проекта
- **Python интерфейс**: `src/inverse_npde/julia_interface.py` — класс `EEGInverseSolver`
- **Julia модуль**: `src/neural_pde_solver/InverseProblem.jl` — функции `create_complete_setup`, `run_eeg_inverse_problem`, `analyze_results`
- **Генератор данных**: `src/inverse_npde/eeg_data_generator.py` — класс `EEGDataGenerator`

## План действий

### 1. Реализация нормировки в Julia (InverseProblem.jl)

#### 1.1 Добавить функцию нормировки
В модуле `InverseProblem` создать функцию для вычисления нормы и нормализации данных:

```julia
"""
    normalize_measured_points(measured_points)

Нормирует значения на датчиках по среднему модулю всех измерений.

Args:
    measured_points: Массив измеренных точек [x, y, z, t, phi_measured]

Returns:
    (normalized_points, norm_factor): Нормированные точки и фактор нормировки
"""
function normalize_measured_points(measured_points)
    # Извлекаем все измеренные значения φ
    phi_values = [point[5] for point in measured_points]
    
    # Вычисляем норму - средний модуль всех значений
    norm_factor = mean(abs.(phi_values))
    
    # Нормируем все значения
    normalized_points = map(measured_points) do point
        [point[1], point[2], point[3], point[4], point[5] / norm_factor]
    end
    
    return normalized_points, norm_factor
end
```

#### 1.2 Изменить функцию create_complete_setup
Модифицировать функцию для применения нормировки при создании настроек эксперимента:

```julia
function create_complete_setup(; measured_points, nn_config::NeuralNetworkConfig, 
                               opt_config::OptimizationConfig,
                               loss_config::LossFunctionConfig,
                               domain_config::Dict{String, Any}=Dict(
                                   "x_range" => [-10.0, 10.0],
                                   "y_range" => [-10.0, 10.0], 
                                   "z_range" => [-10.0, 10.0],
                                   "t_range" => [0.0, 1.0],
                                   "num_points" => 100
                               ))
    
    # Нормируем измеренные точки
    normalized_points, norm_factor = normalize_measured_points(measured_points)
    println("✓ Данные нормированы, фактор: $(round(norm_factor, digits=6))")
    
    # Валидация конфигураций
    validate_config(nn_config)
    validate_optimization_config(opt_config)
    
    # Создаем физические константы
    constants = PhysicalConstants()
    
    # Создаем переменные и области
    variables = create_variables()
    domains = create_domains(variables, domain_config["x_range"], domain_config["y_range"], 
                            domain_config["z_range"], domain_config["t_range"])
    
    # Создаем граничные условия
    bcs = create_boundary_conditions(constants, variables, domains)

    # Создаем PDE систему
    pde_system = create_pde_system(constants, variables, bcs, domains)

    normalized_points = normalized_points |> gpu_device()
    println("✓ Используем нормированные измеренные точки")
    
    # Обновляем loss_config с измеренными точками
    loss_config = LossFunctionConfig(; 
                lambda_pde = loss_config.lambda_pde,
                lambda_bc = loss_config.lambda_bc,
                lambda_data=loss_config.lambda_data,
                measured_points=normalized_points)
    
    # Создаем нейронную сеть
    chain = create_neural_network(nn_config)
    ps = initialize_parameters(chain, Random.default_rng(), nn_config.use_gpu)
    
    return (chain=chain, ps=ps, constants=constants, variables=variables,
            domains=domains, pde_system=pde_system, bcs=bcs, 
            measured_points=normalized_points, configs=(nn_config=nn_config, opt_config=opt_config, loss_config=loss_config),
            norm_factor=norm_factor)  # Добавляем норму в результат
end
```

### 2. Сохранение нормы в результатах эксперимента

#### 2.1 Изменить функцию run_eeg_inverse_problem
Добавить сохранение нормы в структуру результатов:

```julia
function run_eeg_inverse_problem(;measured_points, nn_config::NeuralNetworkConfig,
                                opt_config::OptimizationConfig,
                                loss_config::LossFunctionConfig,
                                domain_config::Dict{Any, Any}=Dict(
                                    "x_range" => [-10.0, 10.0],
                                    "y_range" => [-10.0, 10.0], 
                                    "z_range" => [-10.0, 10.0],
                                    "t_range" => [0.0, 1.0]
                                ),
                                )
    domain_config= Dict{String, Any}(domain_config)
    println("🚀 Запуск эксперимента обратной задачи ЭЭГ...")
    
    # Создаем полную настройку
    setup = create_complete_setup(; measured_points, nn_config, opt_config, loss_config, domain_config)
    
    println("✓ Настройка создана")
    
    # Создаем discretization
    discretization = create_discretization(setup.chain, setup.ps, setup.configs.loss_config, 
                                          setup.configs.opt_config)

    println("✓ Discretization создан")
    
    # Дискретизация PDE системы
    prob = discretize(setup.pde_system, discretization)

    
    println("✓ PDE система дискретизирована")
    
    # Создаем callback функцию
    callback = create_optimization_callback(setup.configs.opt_config, discretization, 
                                          setup.pde_system, setup.bcs, setup.domains)
    
    # Настраиваем оптимизатор
    opt = setup_optimization(setup.configs.opt_config)
    
    println("🔄 Начинаем оптимизацию...")
    
    # Запускаем оптимизацию
    res = solve(prob, opt; maxiters = setup.configs.opt_config.max_iterations, callback)
    
    println("✓ Оптимизация завершена")
    
    # Извлекаем результаты
    phi = discretization.phi
    final_params = res.u
    
    # Анализируем результаты с учетом нормы
    results = analyze_results(phi, final_params, setup, domain_config, setup.norm_factor)
    
    return (solution=res, discretization=discretization, phi=phi, 
            params=final_params, results=results, setup=setup)
end
```

### 3. Обратное преобразование к предсказаниям

#### 3.1 Изменить функцию analyze_results
Модифицировать функцию для обратного преобразования предсказанных значений:

```julia
function analyze_results(phi, params, setup, domain_config, norm_factor::Float64)
    cpud = cpu_device()
    
    # Получаем измеренные точки с датчиков
    measured_points = setup.measured_points
    
    # Преобразуем в массивы для удобства работы
    n_points = length(measured_points)
    x_coords = Float64[]
    y_coords = Float64[]
    z_coords = Float64[]
    t_coords = Float64[]
    measured_phi = Float64[]
    predicted_phi = Float64[]
    
    # Анализируем каждую измеренную точку
    for point in measured_points
        x, y, z, t, phi_measured = point
        
        # Предсказанное значение потенциала (нормированное)
        phi_pred_normalized = (phi([x, y, z, t], params|>cpud))[1]
        
        # Обратное преобразование
        phi_pred = phi_pred_normalized * norm_factor
        
        push!(x_coords, x)
        push!(y_coords, y) 
        push!(z_coords, z)
        push!(t_coords, t)
        push!(measured_phi, phi_measured * norm_factor)  # Обратное преобразование для измеренных данных
        push!(predicted_phi, phi_pred)
    end
    
    # ... (остальная часть функции остается неизменной)
    
    # Добавляем информацию о нормировке в результаты
    results["norm_factor"] = norm_factor
    
    return results
end
```

### 4. Обновление Python интерфейса

#### 4.1 Изменить метод solve в EEGInverseSolver
Добавить поддержку получения и сохранения нормы в Python:

```python
    def solve(
        self,
        measured_points: List[List[float]],
        save_results: bool = True,
        save_path: str = "results/eeg_inverse_results.jld2",
    ) -> Dict[str, Any]:
        """
        Запускает решение обратной задачи ЭЭГ.

        Args:
            measured_points: Предварительно созданные измеренные точки (ОБЯЗАТЕЛЬНО)
            save_results: Сохранять ли результаты
            save_path: Путь для сохранения результатов

        Returns:
            Словарь с результатами эксперимента
        """

        # Проверяем, что измеренные точки предоставлены
        if not measured_points:
            raise ValueError(
                "Измеренные точки должны быть предоставлены в параметре 'measured_points'. "
                "Создайте точки самостоятельно и передайте их в solve()."
            )

        logger.info("Запуск решения обратной задачи ЭЭГ...")

        try:
            # Создаем конфигурации в формате Julia
            configs = self._create_julia_configs()

            # Создаем конфигурацию функции потерь с измеренными точками

            logger.info(f"Передано измеренных точек: {len(measured_points)}")

            # Запускаем эксперимент через JuliaCall
            results_jl = jl.InverseNpde.run_eeg_inverse_problem(
                measured_points=measured_points,
                nn_config=configs["neural_config"],
                opt_config=configs["optimization_config"],
                loss_config=configs["loss_config"],
                domain_config=configs["domain_config"],
            )

            # Сохраняем результаты
            self._results = results_jl

            # Извлекаем ключевые данные
            python_results = self._extract_results()

            # Сохраняем результаты если требуется
            if save_results:
                self.save_results(save_path)

            logger.info("Решение обратной задачи ЭЭГ завершено")
            return python_results

        except Exception as e:
            logger.error(f"Ошибка при решении обратной задачи: {e}")
            raise
```

#### 4.2 Изменить метод _extract_results
Добавить извлечение нормы из результатов Julia:

```python
    def _extract_results(self) -> Dict[str, Any]:
        """Извлекает результаты из Julia в Python формат с новой структурой анализа на датчиках."""
        results = self._results

        logger.info(f"Тип результатов: {type(results)}")

        # Извлекаем основные метрики из новой структуры анализа
        overall_mse = float(results.results["overall_mse"])
        overall_mae = float(results.results["overall_mae"])
        overall_max_error = float(results.results["overall_max_error"])
        avg_time_mse = float(results.results["avg_time_mse"])
        avg_time_mae = float(results.results["avg_time_mae"])
        avg_time_max_error = float(results.results["avg_time_max_error"])
        final_loss = float(results.results["final_loss"])
        
        # Извлекаем фактор нормировки
        norm_factor = float(results.results["norm_factor"])

        logger.info(f"Извлечены метрики: MSE={overall_mse:.6f}, MAE={overall_mae:.6f}")
        logger.info(f"Фактор нормировки: {norm_factor:.6f}")

        # ... (остальная часть функции остается неизменной)

        # Возвращаем новую структуру результатов с конвертированными данными
        return {
            "metrics": metrics,
            "time_step_metrics": time_step_metrics,
            "sensor_data": sensor_data,
            "uniform_grid_data": uniform_grid_data,
            "grid_info": grid_info,
            "time_steps": time_steps,
            "statistics": {
                "num_sensors": num_sensors,
                "num_time_steps": num_time_steps,
            },
            "norm_factor": norm_factor,  # Добавляем норму в результат
            "solved": True,
            "experiment_config": {
                "constants": self.constants,
                "neural_config": self.neural_config,
                "optimization_config": self.optimization_config,
                "domain": self.domain,
            },
        }
```

#### 4.3 Обновить метод predict_signal
Модифицировать метод для обратного преобразования предсказаний:

```python
    def predict_signal(
        self,
        sources: Dict[str, np.ndarray],
        x: Union[float, np.ndarray],
        y: Union[float, np.ndarray],
        z: Union[float, np.ndarray],
        t: float = 0.0,
    ) -> np.ndarray:
        """
        Восстанавливает сигнал по источникам в заданных точках.

        Args:
            sources: Источники (rho, jx, jy, jz)
            x, y, z: Координаты точек где восстанавливаем сигнал
            t: Временная координата

        Returns:
            Восстановленный скалярный потенциал
        """
        if self._results is None:
            raise RuntimeError("Сначала запустите solve()")

        try:
            # Конвертируем входные данные в Julia формат
            if isinstance(x, (int, float)):
                x = [x]
            if isinstance(y, (int, float)):
                y = [y]
            if isinstance(z, (int, float)):
                z = [z]

            # Создаем массив координат для Julia
            coords = [
                [float(xi), float(yi), float(zi), float(t)]
                for xi, yi, zi in zip(x, y, z)
            ]

            # Восстанавливаем сигнал через JuliaCall
            predicted_signals_jl = jl.seval(f"""
                let
                    params = results.params
                    phi_func = results.phi
                    points = {coords}
                    
                    predicted_signals = []
                    for point in points
                        pred = phi_func(point, params)
                        phi = pred[1]  # скалярный потенциал (нормированный)
                        push!(predicted_signals, phi)
                    end
                    predicted_signals
                end
            """)

            # Конвертируем результат в Python и применяем обратное преобразование
            normalized_signals = np.array(predicted_signals_jl)
            norm_factor = self._results.results["norm_factor"]
            return normalized_signals * norm_factor

        except Exception as e:
            logger.error(f"Ошибка восстановления сигнала: {e}")
            raise
```

### 5. Тестирование и валидация

#### 5.1 Добавить тесты в tests/test_juliacall_simple.py
```python
def test_normalization():
    """Тестирование нормировки данных"""
    from inverse_npde.julia_interface import EEGInverseSolver
    from inverse_npde.eeg_data_generator import create_eeg_experiment
    
    # Создаем тестовые данные
    measured_points, generator = create_eeg_experiment(
        num_sensors=8,
        sampling_rate=100.0,
        duration=0.1,
        noise_level=0.0,
        head_type="adult",
        save_visualization=False
    )
    
    # Создаем решатель
    solver = EEGInverseSolver()
    
    # Запускаем решение
    results = solver.solve(measured_points, save_results=False)
    
    # Проверяем, что норма была извлечена
    assert "norm_factor" in results
    assert isinstance(results["norm_factor"], float)
    assert results["norm_factor"] > 0
    
    logger.info(f"Тест нормировки пройден, фактор: {results['norm_factor']:.6f}")
```

### 6. Документация

#### 6.1 Обновить пример использования в README.md
Добавить информацию о нормировке в примерах кода.

#### 6.2 Обновить docstrings
Обновить комментарии в коде для описания измененных функций и добавленной функциональности.

## Важные замечания

- Нормировка одинакова для всех датчиков, что сохраняет относительные отношения между значениями
- Для предсказаний используется обратное преобразование с сохраненным фактором нормировки
- Норма вычисляется как средний модуль всех измеренных значений φ
- Изменения требуются как в Julia модуле, так и в Python интерфейсе

## Проверка результатов

После реализации необходимо запустить следующие тесты:
1. Проверить, что эксперимент запускается без ошибок
2. Проверить, что норма правильно сохраняется в результатах
3. Проверить, что предсказания правильно денормированы
4. Проверить, что метрики (MSE, MAE) корректны
