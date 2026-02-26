"""
    Julia Interface для модуля обратной задачи ЭЭГ

Интерфейс Python для вызова Julia функций из модуля InverseNpde.
Использует современный JuliaCall API для взаимодействия с Julia.

Пример использования:
```python
from julia_interface import EEGInverseSolver

# Создаем экземпляр решателя
solver = EEGInverseSolver(
    constants={'c': 2.99e10, 'epsilon': 1.0},
    neural_config={'layers': [32, 32], 'activation': 'σ'},
    domain={'x': [-10, 10], 'y': [-10, 10], 'z': [-10, 10], 't': [0, 1]},
    pml_config={'enabled': True, 'pml_thickness_ratio': 0.1}
)

# Создаем измеренные точки
measured_points = [[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.1]]

# Запускаем эксперимент
results = solver.solve(measured_points)
```
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import sys


# Используем только современный JuliaCall
from juliacall import Main as jl

JULIA_AVAILABLE = True
PATH_TO_PROJECT = Path(__file__).parent.parent.parent
# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PMLConfig:
    """
    Конфигурация для Perfectly Matched Layer (PML) - поглощающих граничных условий.

    Новый подход PML через затухание (γ) и экранирование (α):
    - γ(r) — динамическое затухание (вязкое трение), поглощает энергию волн
    - α(r) — статический экранирующий член, обеспечивает экспоненциальное убывание

    ВАЖНО: Используем безразмерные единицы с c = 1!

    Формула для gamma_max (на основе характерной частоты):
        γ_max = π * f_char * (m+1) * ln(1/R)

    Параметры gamma_max, alpha_max могут быть:
    - None: автоматический расчёт оптимального значения (рекомендуется)
    - float: явное задание значения
    """

    def __init__(
        self,
        enabled: bool = True,
        pml_thickness_ratio: float = 0.1,
        reflection_coefficient: float = 1e-4,
        characteristic_frequency: float = 100.0,
        profile_order: int = 2,
        gamma_max: Optional[float] = None,
        alpha_max: Optional[float] = None,
    ):
        """
        Создаёт конфигурацию PML.

        Параметры:
        - enabled: Включить/отключить PML (по умолчанию включен)
        - pml_thickness_ratio: Толщина PML слоя в долях от размеров домена (0-0.5)
        - reflection_coefficient: Коэффициент отражения от PML (по умолчанию 1e-4 = 0.01%)
        - characteristic_frequency: Характерная частота сигнала в Гц (для ЭЭГ ~10-100 Гц)
        - profile_order: Порядок профиля PML (обычно 2)
        - gamma_max: Максимальное значение коэффициента затухания
          (None = авто: γ_max = π * f_char * (m+1) * ln(1/R))
        - alpha_max: Максимальное значение коэффициента экранирования
          (None = авто: α_max = (2*ln(1/R)/L)²)
        """
        if not (0.0 < pml_thickness_ratio < 0.5):
            raise ValueError("pml_thickness_ratio must be between 0 and 0.5")
        if not (0.0 < reflection_coefficient < 1.0):
            raise ValueError("reflection_coefficient must be between 0 and 1")
        if characteristic_frequency <= 0.0:
            raise ValueError("characteristic_frequency must be > 0")
        if profile_order < 1:
            raise ValueError("profile_order must be >= 1")
        if gamma_max is not None and gamma_max < 0.0:
            raise ValueError("gamma_max must be ≥ 0")
        if alpha_max is not None and alpha_max < 0.0:
            raise ValueError("alpha_max must be ≥ 0")

        self.enabled = enabled
        self.pml_thickness_ratio = pml_thickness_ratio
        self.reflection_coefficient = reflection_coefficient
        self.characteristic_frequency = characteristic_frequency
        self.profile_order = profile_order
        self.gamma_max = gamma_max
        self.alpha_max = alpha_max

    def to_dict(self) -> Dict[str, Any]:
        """Конвертирует в словарь для передачи в Julia."""
        return {
            "enabled": self.enabled,
            "pml_thickness_ratio": self.pml_thickness_ratio,
            "reflection_coefficient": self.reflection_coefficient,
            "characteristic_frequency": self.characteristic_frequency,
            "profile_order": self.profile_order,
            "gamma_max": self.gamma_max,
            "alpha_max": self.alpha_max,
        }


class EEGInverseSolver:
    """
    Основной класс для решения обратной задачи ЭЭГ.

    Инкапсулирует настройки и обеспечивает простой интерфейс
    для взаимодействия с Julia модулем InverseNpde.
    """

    def __init__(
        self,
        constants: Optional[Dict[str, float]] = None,
        neural_config: Optional[Dict[str, Any]] = None,
        optimization_config: Optional[Dict[str, Any]] = None,
        domain: Optional[Dict[str, List[float]]] = None,
        loss_config: Optional[Dict[str, Any]] = None,
        pml_config: Optional[Union[Dict[str, Any], PMLConfig]] = None,
    ):
        """
        Инициализация решателя обратной задачи ЭЭГ.

        Args:
            constants: Физические константы (c, epsilon, mu и т.д.)
            neural_config: Конфигурация нейронной сети
            optimization_config: Конфигурация оптимизации
            domain: Область определения (x, y, z, t диапазоны)
            pml_config: Конфигурация PML (Perfectly Matched Layer)

        """
        if not JULIA_AVAILABLE:
            raise RuntimeError(
                "JuliaCall не установлен. Установите через: pip install juliacall"
            )

        # Инициализация Julia модуля
        self._init_julia_module()

        # Настройки по умолчанию
        self.constants = constants or self._default_constants()
        self.neural_config = neural_config or self._default_neural_config()
        self.optimization_config = (
            optimization_config or self._default_optimization_config()
        )
        self.domain = domain or self._default_domain()
        self.loss_config = loss_config or self._default_loss_config()
        self.pml_config = self._parse_pml_config(pml_config)

        # Внутренние переменные
        self._results = None
        self._solver_instance = None

        logger.info("EEGInverseSolver инициализирован с JuliaCall")

    def _parse_pml_config(
        self, pml_config: Optional[Union[Dict[str, Any], PMLConfig]]
    ) -> PMLConfig:
        """Парсит конфигурацию PML."""
        if pml_config is None:
            return PMLConfig()
        elif isinstance(pml_config, PMLConfig):
            return pml_config
        elif isinstance(pml_config, dict):
            return PMLConfig(**pml_config)
        else:
            raise ValueError("pml_config must be a dict or PMLConfig instance")

    def _preload_critical_modules(self):
        """Предварительно загружает критические модули для избежания конфликтов."""
        try:
            logger.info("Предварительная загрузка критических модулей...")

            # Проверяем, а не загружены ли уже модули
            try:
                # Проверяем, определен ли ModelingToolkit в текущем модуле
                result = jl.seval("isdefined(@__MODULE__, :ModelingToolkit)")
                if result:
                    logger.info(
                        "Модули уже загружены, пропускаем предварительную загрузку"
                    )
                    return
            except Exception:
                # Модули не загружены, продолжаем
                pass

            # Используем корректную активацию среды без временной среды
            try:
                jl.seval(
                    f'import Pkg; Pkg.activate("{PATH_TO_PROJECT}"); Pkg.instantiate();'
                )
                jl.seval("Pkg.status()")
                logger.info("Julia среда активирована")
            except Exception as e:
                logger.warning(f"Ошибка активации Julia среды: {e}")

            # Загружаем OpenSSL_jll
            logger.info("Загрузка OpenSSL_jll...")
            try:
                jl.seval("using OpenSSL_jll")
                logger.info("OpenSSL_jll успешно загружен")
            except Exception as e:
                logger.error(f"Ошибка загрузки OpenSSL_jll: {e}")
                raise

            # Загружаем остальные модули
            logger.info("Загрузка ModelingToolkit...")
            try:
                jl.seval("using ModelingToolkit")
                logger.info("ModelingToolkit успешно загружен")
            except Exception as e:
                logger.error(f"Ошибка загрузки ModelingToolkit: {e}")
                raise

            logger.info("Загрузка базовых библиотек...")
            jl.seval("using LinearAlgebra")
            jl.seval("using Statistics")

            logger.info("Критические модули успешно загружены")

        except Exception as e:
            logger.error(f"ОШИБКА при предварительной загрузке модулей: {e}")
            logger.info("Продолжаем без предварительной загрузки модулей...")
            # Не прерываем выполнение, позволяем основной инициализации продолжиться
            pass

    def _init_julia_module(self):
        """Инициализирует Julia модуль InverseNpde через JuliaCall."""
        try:
            # Предварительная загрузка критических модулей
            self._preload_critical_modules()

            # Загружаем Julia модуль с правильным путем
            module_path = str(
                PATH_TO_PROJECT / "src" / "neural_pde_solver" / "InverseNpde.jl"
            )

            # Добавляем путь к модулю и загружаем его
            jl.seval(
                f'push!(LOAD_PATH, "{PATH_TO_PROJECT / "src" / "neural_pde_solver"}")'
            )
            jl.seval("using InverseNpde")
            jl.seval("using LuxCUDA, CUDA")  # Для GPU поддержки
            logger.info("Julia модуль InverseNpde загружен через JuliaCall")
        except Exception as e:
            logger.error(f"Ошибка загрузки Julia модуля: {e}")
            raise

    def _default_constants(self) -> Dict[str, float]:
        """Возвращает константы по умолчанию (c = 1 для безразмерных единиц)."""
        return {
            "c": 1.0,  # Безразмерная скорость (c = 1)
            "epsilon": 1.0,  # Диэлектрическая проницаемость
            "mu": 1.0,  # Магнитная проницаемость
            "epsilon_0": 1.0,  # Диэлектрическая постоянная вакуума
            "mu_0": 1.0,  # Магнитная постоянная вакуума
        }

    def _default_loss_config(self) -> Dict[str, Any]:
        """
        Возвращает конфигурацию функции потерь по умолчанию.

        Адаптивный баланс lambda_data:
        - lambda_data_init: начальное значение веса данных
        - lambda_min/lambda_max: ограничения (None = без ограничений)
        - lambda_schedule_type: тип планировщика ("improvement" по умолчанию)
        - lambda_schedule: словарь параметров планировщика (например window_size, patience)
        """
        return {
            "lambda_pde": 1.0,
            "lambda_bc": 1.0,
            "lambda_data_init": 1.0,
            "lambda_min": None,
            "lambda_max": None,
            "lambda_schedule_type": "improvement",
            "lambda_schedule": {},
        }

    def _default_neural_config(self) -> Dict[str, Any]:
        """Возвращает конфигурацию нейронной сети по умолчанию."""
        return {
            "input_dim": 4,  # x, y, z, t
            "hidden_layers": [32, 32],  # Скрытые слои
            "output_dim": 8,  # φ, Ax, Ay, Az, ρ, jx, jy, jz
            "activation": "σ",  # sigmoid activation
            "use_gpu": True,  # Использовать GPU если доступен
        }

    def _default_optimization_config(self) -> Dict[str, Any]:
        """Возвращает конфигурацию оптимизации по умолчанию."""
        return {
            "optimizer": "adam",  # Оптимизатор
            "learning_rate": 0.001,  # Скорость обучения
            "max_iterations": 3000,  # Максимальное количество итераций
            "log_frequency": 50,  # Частота логирования
            "use_tensorboard": True,  # Использовать TensorBoard
            "log_directory": "logs/eeg_inverse_exp",
        }

    def _default_domain(self) -> Dict[str, List[float]]:
        """Возвращает область определения по умолчанию."""
        return {
            "x_range": [-10.0, 10.0],
            "y_range": [-10.0, 10.0],
            "z_range": [-10.0, 10.0],
            "t_range": [0.0, 1.0],
            "num_points": 100,  # Количество измеренных точек
        }

    def _convert_julia_to_python(self, obj):
        """
        Рекурсивно конвертирует Julia объекты в Python типы для JSON сериализации.

        Args:
            obj: Объект для конвертации (может быть Julia объектом или Python объектом)

        Returns:
            Конвертированный Python объект
        """
        # Если это None или простой Python тип, возвращаем как есть
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return obj

        # Если это список (Python list), рекурсивно конвертируем элементы
        if isinstance(obj, list):
            return [self._convert_julia_to_python(item) for item in obj]

        # Если это словарь (Python dict), рекурсивно конвертируем ключи и значения
        if isinstance(obj, dict):
            return {
                self._convert_julia_to_python(k): self._convert_julia_to_python(v)
                for k, v in obj.items()
            }

        # Если это numpy массив, конвертируем в Python list
        if hasattr(obj, "tolist"):  # numpy arrays
            return obj.tolist()

        # Если это Julia объект с определенными методами
        try:
            # Проверяем, есть ли метод keys() (аналог dict.keys())
            if hasattr(obj, "keys"):
                try:
                    # Пытаемся получить ключи как Python list
                    keys = list(obj.keys())
                    # Проверяем, есть ли доступ по ключу
                    first_key = keys[0] if keys else None
                    if first_key is not None:
                        # Это похоже на Julia Dict, конвертируем в Python dict
                        result = {}
                        for key in keys:
                            try:
                                result[self._convert_julia_to_python(key)] = (
                                    self._convert_julia_to_python(obj[key])
                                )
                            except:
                                # Если не удалось получить значение, пропускаем
                                pass
                        return result
                except Exception:
                    pass

            # Проверяем, есть ли метод length или __len__
            if hasattr(obj, "__len__") or hasattr(obj, "length"):
                try:
                    length = len(obj)
                    # Это похоже на Julia Vector/Array, конвертируем в Python list
                    result = []
                    for i in range(length):
                        try:
                            result.append(self._convert_julia_to_python(obj[i]))
                        except:
                            # Если не удалось получить элемент, пропускаем
                            pass
                    return result
                except Exception:
                    pass

            # Если это скалярный Julia объект, пытаемся конвертировать в Python тип
            if hasattr(obj, "__float__"):
                return float(obj)
            elif hasattr(obj, "__int__"):
                return int(obj)
            elif hasattr(obj, "__str__"):
                return str(obj)
            else:
                # Последняя попытка - str() представление
                return str(obj)

        except Exception as e:
            logger.warning(f"Не удалось конвертировать Julia объект: {e}")
            return str(obj)

    def _create_julia_configs(self) -> Dict[str, Any]:
        """Создает конфигурации в формате Julia."""
        try:
            # Создаем физические константы
            const_dict = self.constants
            constants_jl = jl.seval(f"""
                PhysicalConstants(; c={const_dict.get("c", 2.99e10)}, 
                                  ε={const_dict.get("epsilon", 1.0)},
                                  μ={const_dict.get("mu", 1.0)},
                                  ε₀={const_dict.get("epsilon_0", 1.0)},
                                  μ₀={const_dict.get("mu_0", 1.0)})
            """)

            # Создаем конфигурацию нейронной сети через seval для корректной передачи именованных аргументов
            nn_config = self.neural_config
            hidden_layers_str = str(nn_config.get("hidden_layers", [32, 32]))
            neural_config_jl = jl.seval(f"""
                NeuralNetworkConfig(; input_dim={nn_config.get("input_dim", 4)}, 
                                       hidden_layers={hidden_layers_str},
                                       output_dim={nn_config.get("output_dim", 8)},
                                       activation=Symbol("{nn_config.get("activation", "σ")}"),
                                       use_gpu={str(nn_config.get("use_gpu", True)).lower()})
            """)

            # Создаем конфигурацию оптимизации через seval для корректной передачи именованных аргументов
            opt_config = self.optimization_config
            optimization_config_jl = jl.seval(f"""
                OptimizationConfig(; optimizer=Symbol("{opt_config.get("optimizer", "adam")}"),
                                      learning_rate={opt_config.get("learning_rate", 0.001)},
                                      max_iterations={opt_config.get("max_iterations", 3000)},
                                      log_frequency={opt_config.get("log_frequency", 50)},
                                      use_tensorboard={str(opt_config.get("use_tensorboard", True)).lower()},
                                      log_directory="{opt_config.get("log_directory", "logs/eeg_inverse_exp")}")
            """)

            # Создаем конфигурацию домена
            domain = self.domain
            domain_config_jl = jl.Dict(
                {
                    "x_range": domain.get("x_range", [-10.0, 10.0]),
                    "y_range": domain.get("y_range", [-10.0, 10.0]),
                    "z_range": domain.get("z_range", [-10.0, 10.0]),
                    "t_range": domain.get("t_range", [0.0, 1.0]),
                    "num_points": domain.get("num_points", 100),
                }
            )

            # Создаем конфигурацию функции потерь через seval для корректной передачи именованных аргументов
            loss_config = self.loss_config
            # Преобразуем None в nothing для Julia
            lambda_min_str = (
                "nothing"
                if loss_config.get("lambda_min") is None
                else str(loss_config.get("lambda_min"))
            )
            lambda_max_str = (
                "nothing"
                if loss_config.get("lambda_max") is None
                else str(loss_config.get("lambda_max"))
            )

            # Построим описание словаря lambda_schedule для передачи в Julia
            schedule = loss_config.get("lambda_schedule", {}) or {}
            if schedule:
                entries = []
                for k, v in schedule.items():
                    if isinstance(v, str):
                        val = f'"{v}"'
                    elif isinstance(v, bool):
                        val = "true" if v else "false"
                    else:
                        val = repr(v)
                    entries.append(f'"{k}" => {val}')
                schedule_jl = "Dict(" + ", ".join(entries) + ")"
            else:
                schedule_jl = "Dict()"

            loss_config_jl = jl.seval(f"""
                LossFunctionConfig(; lambda_pde={loss_config.get("lambda_pde", 1.0)},
                                      lambda_bc={loss_config.get("lambda_bc", 1.0)},
                                      lambda_data_init={loss_config.get("lambda_data_init", 1.0)},
                                      lambda_min={lambda_min_str},
                                      lambda_max={lambda_max_str},
                                      lambda_time={loss_config.get("lambda_time", 1.0)},
                                      lambda_schedule_type=Symbol("{loss_config.get("lambda_schedule_type", "improvement")}"),
                                      lambda_schedule={schedule_jl})
            """)

            # Создаем конфигурацию PML через seval для корректной передачи именованных аргументов
            pml_config = self.pml_config.to_dict()
            # Преобразуем None в nothing для Julia
            gamma_max_str = (
                "nothing"
                if pml_config.get("gamma_max") is None
                else str(pml_config.get("gamma_max"))
            )
            alpha_max_str = (
                "nothing"
                if pml_config.get("alpha_max") is None
                else str(pml_config.get("alpha_max"))
            )

            pml_config_jl = jl.seval(f"""
                PMLConfig(; pml_thickness_ratio={pml_config.get("pml_thickness_ratio", 0.1)},
                            reflection_coefficient={pml_config.get("reflection_coefficient", 1e-4)},
                            characteristic_frequency={pml_config.get("characteristic_frequency", 100.0)},
                            profile_order={pml_config.get("profile_order", 2)},
                            gamma_max={gamma_max_str},
                            alpha_max={alpha_max_str},
                            enabled={str(pml_config.get("enabled", True)).lower()})
            """)

            return {
                "constants": constants_jl,
                "neural_config": neural_config_jl,
                "optimization_config": optimization_config_jl,
                "domain_config": domain_config_jl,
                "loss_config": loss_config_jl,
                "pml_config": pml_config_jl,
            }

        except Exception as e:
            logger.error(f"Ошибка создания Julia конфигураций: {e}")
            raise

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
                pml_config=configs["pml_config"],
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

        logger.info(f"Извлечены метрики: MSE={overall_mse:.6f}, MAE={overall_mae:.6f}")

        # Извлекаем метрики по временным шагам и КОНВЕРТИРУЕМ в Python типы
        time_step_metrics_julia = results.results["time_step_metrics"]
        time_step_metrics = self._convert_julia_to_python(time_step_metrics_julia)
        logger.info(
            f"Извлечены и конвертированы метрики по {len(time_step_metrics)} временным шагам"
        )

        # Извлекаем данные датчиков и КОНВЕРТИРУЕМ в Python типы
        sensor_data_julia = results.results["sensor_data"]
        sensor_data = self._convert_julia_to_python(sensor_data_julia)
        logger.info(
            f"Извлечены и конвертированы данные датчиков: {len(sensor_data['x_coords'])} точек"
        )

        # Извлекаем данные равномерной сетки и КОНВЕРТИРУЕМ в Python типы
        uniform_grid_data_julia = results.results["uniform_grid_data"]
        uniform_grid_data = self._convert_julia_to_python(uniform_grid_data_julia)
        logger.info(
            f"Извлечены и конвертированы данные равномерной сетки для {len(uniform_grid_data)} временных шагов"
        )

        # Извлекаем информацию о сетке и КОНВЕРТИРУЕМ в Python типы
        grid_info_julia = results.results["grid_info"]
        grid_info = self._convert_julia_to_python(grid_info_julia)

        # Извлекаем временные шаги и КОНВЕРТИРУЕМ в Python типы
        time_steps_julia = results.results["time_steps"]
        time_steps = self._convert_julia_to_python(time_steps_julia)

        # Извлекаем статистику
        num_sensors = results.results["num_sensors"]
        num_time_steps = results.results["num_time_steps"]

        # Извлекаем норму
        norm_factor = float(results.results.get("norm_factor", 1.0))
        logger.info(f"Извлечен фактор нормировки: {norm_factor:.6f}")

        # Извлекаем финальное значение lambda_data (адаптивный баланс)
        final_lambda_data = float(results.final_lambda_data)
        logger.info(f"Финальное lambda_data: {final_lambda_data:.6f}")

        # Формируем итоговые метрики
        metrics = {
            "final_loss": final_loss,
            "overall_mse": overall_mse,
            "overall_mae": overall_mae,
            "overall_max_error": overall_max_error,
            "avg_time_mse": avg_time_mse,
            "avg_time_mae": avg_time_mae,
            "avg_time_max_error": avg_time_max_error,
            "final_lambda_data": final_lambda_data,
        }

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
            "solved": True,
            "norm_factor": norm_factor,  # Добавляем норму
            "experiment_config": {
                "constants": self.constants,
                "neural_config": self.neural_config,
                "optimization_config": self.optimization_config,
                "domain": self.domain,
                "pml_config": self.pml_config.to_dict(),
            },
        }

    def get_sources(
        self,
        x: Union[float, np.ndarray],
        y: Union[float, np.ndarray],
        z: Union[float, np.ndarray],
        t: float = 0.0,
    ) -> Dict[str, np.ndarray]:
        """
        Получает источники сигнала (плотность заряда и тока) в заданных точках.

        Args:
            x, y, z: Координаты точек
            t: Временная координата

        Returns:
            Словарь с компонентами источников
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

            # Вычисляем источники через JuliaCall
            sources_jl = jl.seval(f"""
                let
                    params = results.params
                    phi_func = results.phi
                    points = {coords}
                    
                    sources = []
                    for point in points
                        pred = phi_func(point, params)
                        rho = pred[5]  # плотность заряда
                        jx = pred[6]   # компоненты плотности тока
                        jy = pred[7]
                        jz = pred[8]
                        push!(sources, (rho, jx, jy, jz))
                    end
                    sources
                end
            """)

            # Конвертируем результаты в Python
            sources_python = {
                "rho": np.array([s[0] for s in sources_jl]),
                "jx": np.array([s[1] for s in sources_jl]),
                "jy": np.array([s[2] for s in sources_jl]),
                "jz": np.array([s[3] for s in sources_jl]),
            }

            return sources_python

        except Exception as e:
            logger.error(f"Ошибка вычисления источников: {e}")
            raise

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
                        phi = pred[1]  # скалярный потенциал
                        push!(predicted_signals, phi)
                    end
                    predicted_signals
                end
            """)

            # Конвертируем результат в Python
            return np.array(predicted_signals_jl)

        except Exception as e:
            logger.error(f"Ошибка восстановления сигнала: {e}")
            raise

    def visualize_results(
        self, save_path: str = "figures/eeg_inverse_results.png"
    ) -> None:
        """
        Создает визуализацию результатов.

        Args:
            save_path: Путь для сохранения графиков
        """
        if self._results is None:
            raise RuntimeError("Сначала запустите solve()")

        try:
            # Создаем визуализацию через JuliaCall
            jl.seval(f'create_visualization_plots(results["results"], "{save_path}")')

            logger.info(f"Визуализация сохранена в {save_path}")

        except Exception as e:
            logger.error(f"Ошибка создания визуализации: {e}")
            raise

    def save_results(self, filepath: str) -> None:
        """
        Сохраняет результаты эксперимента в файл.

        Args:
            filepath: Путь для сохранения
        """
        if self._results is None:
            raise RuntimeError("Нет результатов для сохранения")

        try:
            # Сохраняем через JuliaCall
            jl.seval(f'save_results(results, "{filepath}")')

            logger.info(f"Результаты сохранены в {filepath}")

        except Exception as e:
            logger.error(f"Ошибка сохранения результатов: {e}")
            raise

    def load_results(self, filepath: str) -> Dict[str, Any]:
        """
        Загружает результаты эксперимента из файла.

        Args:
            filepath: Путь к файлу с результатами

        Returns:
            Словарь с результатами
        """
        try:
            # Загружаем через JuliaCall
            loaded_results_jl = jl.seval(f'load_results("{filepath}")')

            self._results = loaded_results_jl
            return self._extract_results()

        except Exception as e:
            logger.error(f"Ошибка загрузки результатов: {e}")
            raise

    def benchmark_configs(self, configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Запускает бенчмарк с разными конфигурациями.

        Args:
            configs: Список конфигураций для тестирования

        Returns:
            Список результатов для каждой конфигурации
        """
        logger.info(f"Запуск бенчмарка с {len(configs)} конфигурациями...")

        benchmark_results = []

        for i, config in enumerate(configs):
            try:
                logger.info(f"Тестирование конфигурации {i + 1}/{len(configs)}")

                # Создаем временный решатель с новой конфигурацией
                temp_solver = EEGInverseSolver(
                    constants=config.get("constants", self.constants),
                    neural_config=config.get("neural_config", self.neural_config),
                    optimization_config=config.get(
                        "optimization_config", self.optimization_config
                    ),
                    domain=config.get("domain", self.domain),
                    pml_config=config.get("pml_config", self.pml_config),
                )

                # Запускаем эксперимент
                # Создаем простые измеренные точки для бенчмарка
                measured_points = [[0.0, 0.0, 0.0, 0.0, 0.0]]
                result = temp_solver.solve(measured_points, save_results=False)
                result["config_id"] = i + 1
                result["success"] = True

                benchmark_results.append(result)

            except Exception as e:
                logger.error(f"Ошибка в конфигурации {i + 1}: {e}")
                benchmark_results.append(
                    {"config_id": i + 1, "success": False, "error": str(e)}
                )

        logger.info("Бенчмарк завершен")
        return benchmark_results

    def get_config_summary(self) -> Dict[str, Any]:
        """Возвращает сводку текущей конфигурации."""
        return {
            "constants": self.constants,
            "neural_config": self.neural_config,
            "optimization_config": self.optimization_config,
            "domain": self.domain,
            "pml_config": self.pml_config.to_dict(),
            "results_available": self._results is not None,
        }


def create_default_solver() -> EEGInverseSolver:
    """
    Создает решатель с конфигурацией по умолчанию.

    Returns:
        EEGInverseSolver с настройками по умолчанию
    """
    return EEGInverseSolver()


def create_gpu_solver() -> EEGInverseSolver:
    """
    Создает решатель оптимизированный для GPU.

    Returns:
        EEGInverseSolver с GPU оптимизацией
    """
    return EEGInverseSolver(
        neural_config={
            "input_dim": 4,
            "hidden_layers": [64, 64, 32],  # Более глубокая сеть
            "output_dim": 8,
            "activation": "relu",  # Быстрая активация
            "use_gpu": True,
        },
        optimization_config={
            "optimizer": "adam",
            "learning_rate": 0.001,
            "max_iterations": 5000,
            "log_frequency": 100,
        },
    )


def create_fast_solver() -> EEGInverseSolver:
    """
    Создает решатель для быстрых экспериментов.

    Returns:
        EEGInverseSolver для быстрых тестов
    """
    return EEGInverseSolver(
        neural_config={
            "input_dim": 4,
            "hidden_layers": [16, 16],  # Меньшая сеть
            "output_dim": 8,
            "activation": "σ",
            "use_gpu": False,  # CPU для стабильности
        },
        optimization_config={
            "optimizer": "adam",
            "learning_rate": 0.01,  # Большая скорость обучения
            "max_iterations": 500,  # Меньше итераций
            "log_frequency": 25,
            "use_tensorboard": False,
        },
        domain={
            "x_range": [-5.0, 5.0],  # Меньшая область
            "y_range": [-5.0, 5.0],
            "z_range": [-5.0, 5.0],
            "t_range": [0.0, 0.5],
            "num_points": 50,  # Меньше точек
        },
    )


if __name__ == "__main__":
    # Пример использования
    print("Создание решателя обратной задачи ЭЭГ...")

    # Создаем быстрый решатель для демонстрации
    solver = create_fast_solver()

    # Создаем измеренные точки для демонстрации
    measured_points = [
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 0.1, 0.0],
        [-1.0, -1.0, -1.0, 0.2, 0.0],
    ]

    print("Запуск эксперимента...")
    try:
        results = solver.solve(measured_points, save_results=False)
        print(
            f"Эксперимент завершен. Финальная ошибка: {results['metrics']['final_loss']:.6f}"
        )

        # Демонстрация получения источников
        sources = solver.get_sources([0.0, 1.0, -1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        print(f"Источники в точках: rho = {sources['rho']}")

    except Exception as e:
        print(f"Ошибка: {e}")
        print("Убедитесь, что JuliaCall установлен и Julia модуль загружен правильно.")
