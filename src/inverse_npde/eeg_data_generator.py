"""
Генератор реалистичных данных для обратной задачи ЭЭГ

Создает измеренные точки, имитирующие EEG шапочку с датчиками,
и генерирует сигналы от множественных источников с временными пиками.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any
from pathlib import Path
import json


class EEGSensorGenerator:
    """
    Генератор координат датчиков EEG на поверхности головы (эллипсоид)
    """

    def __init__(self, head_type: str = "adult", num_sensors: int = 64):
        """
        Инициализация генератора датчиков

        Args:
            head_type: Тип головы ('child', 'adult', 'large_adult', 'female', 'male')
            num_sensors: Количество датчиков на шапочке
        """
        self.head_type = head_type
        self.num_sensors = num_sensors
        self.head_dimensions = self._get_head_dimensions(head_type)
        self.sensor_positions = self._generate_sensor_positions()

    def _get_head_dimensions(self, head_type: str) -> Dict[str, float]:
        """
        Возвращает размеры головы для разных типов

        Args:
            head_type: Тип головы

        Returns:
            Словарь с размерами (rx, ry, rz) - радиусы по осям X, Y, Z
        """
        dimensions = {
            "child": {"rx": 7.0, "ry": 8.0, "rz": 9.0},  # Детская голова
            "adult": {"rx": 8.5, "ry": 9.5, "rz": 10.5},  # Стандартная взрослая
            "large_adult": {"rx": 9.5, "ry": 10.5, "rz": 11.5},  # Крупная голова
            "female": {"rx": 8.0, "ry": 9.0, "rz": 10.0},  # Женская (более округлая)
            "male": {"rx": 9.0, "ry": 10.0, "rz": 11.0},  # Мужская (более вытянутая)
        }

        return dimensions.get(head_type, dimensions["adult"])

    def _generate_sensor_positions(self) -> np.ndarray:
        """
        Генерирует координаты датчиков на верхней части эллипсоидальной поверхности головы

        Использует модифицированное распределение Fibonacci sphere
        с адаптацией под форму эллипсоида и ограничением только верхней части
        """
        points = []
        phi = np.pi * (3.0 - np.sqrt(5.0))  # Золотое сечение

        # Получаем размеры головы
        rx, ry, rz = (
            self.head_dimensions["rx"],
            self.head_dimensions["ry"],
            self.head_dimensions["rz"],
        )

        # Генерируем больше точек, чем нужно, чтобы отфильтровать нижние
        temp_points = []
        for i in range(self.num_sensors * 3):  # Генерируем в 3 раза больше
            # Создаем точки на единичной сфере
            y = 1 - (i / float(self.num_sensors * 3 - 1)) * 2  # y от 1 до -1
            radius_at_y = np.sqrt(1 - y * y)

            theta = phi * i

            x = np.cos(theta) * radius_at_y
            z = np.sin(theta) * radius_at_y

            # ФИЛЬТР: оставляем только верхнюю часть (z >= 0)
            if z >= 0:  # Только верхняя часть головы
                # Применяем эллипсоидальную деформацию
                # Учитываем асимметрию головы (затылок выступает больше)
                if z < 0:  # Задняя часть головы
                    z *= 1.1  # Немного увеличиваем

                if x < 0:  # Левая сторона (обычно меньше)
                    x *= 0.95

                # Масштабируем к размерам головы
                head_x = x * rx
                head_y = y * ry
                head_z = z * rz

                temp_points.append([head_x, head_y, head_z])

        # Берем первые num_sensors точек из отфильтрованного списка
        points = temp_points[: self.num_sensors]

        return np.array(points)

    def get_sensor_positions(self) -> np.ndarray:
        """Возвращает координаты датчиков"""
        return self.sensor_positions

    def get_head_dimensions(self) -> Dict[str, float]:
        """Возвращает размеры головы"""
        return self.head_dimensions.copy()

    def plot_sensors(self, save_path: str = None) -> None:
        """Визуализирует расположение датчиков на эллипсоиде головы с сеткой"""
        fig = plt.figure(figsize=(18, 12))

        # Получаем размеры головы
        rx, ry, rz = (
            self.head_dimensions["rx"],
            self.head_dimensions["ry"],
            self.head_dimensions["rz"],
        )

        # Создаем сетку для поверхности головы
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)

        # Параметрические уравнения эллипсоида
        x_surf = rx * np.outer(np.cos(u), np.sin(v))
        y_surf = ry * np.outer(np.sin(u), np.sin(v))
        z_surf = rz * np.outer(np.ones(np.size(u)), np.cos(v))

        # Применяем асимметрию
        y_surf_modified = y_surf.copy()
        y_surf_modified[y_surf < 0] *= 1.1  # Затылок (y < 0) выступает больше
        x_surf_modified = x_surf.copy()
        x_surf_modified[x_surf < 0] *= 0.95  # Левая сторона (x < 0) меньше

        # 1. 3D вид с головой и датчиками
        ax1 = fig.add_subplot(221, projection="3d")

        # Рисуем поверхность головы (полупрозрачная)
        ax1.plot_surface(
            x_surf_modified,
            y_surf_modified,
            z_surf,
            alpha=0.3,
            color="lightblue",
            linewidth=0.5,
            edgecolors="navy",
        )

        # Рисуем датчики с нумерацией
        scatter = ax1.scatter(
            self.sensor_positions[:, 0],
            self.sensor_positions[:, 1],
            self.sensor_positions[:, 2],
            c="red",
            s=80,
            alpha=0.9,
            edgecolors="darkred",
            linewidth=1,
        )

        # Добавляем номера датчиков
        for i, (x, y, z) in enumerate(self.sensor_positions):
            ax1.text(
                x,
                y,
                z,
                str(i + 1),
                fontsize=8,
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
            )

        ax1.set_title(
            f"3D вид головы {self.head_type} с датчиками (n={self.num_sensors})"
        )
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")

        # Устанавливаем правильные масштабы осей на основе размеров головы
        x_range = rx  # Полный радиус по X
        y_range = ry  # Полный радиус по Y
        z_range = rz  # Полный радиус по Z

        max_range = max(x_range, y_range, z_range)

        # Центрируем в начале координат
        ax1.set_xlim(-max_range, max_range)
        ax1.set_ylim(-max_range, max_range)
        ax1.set_zlim(-max_range, max_range)

        # 2. Вид спереди (XZ проекция)
        ax2 = fig.add_subplot(222)

        # Рисуем контур головы
        theta = np.linspace(0, 2 * np.pi, 100)
        x_contour = rx * np.cos(theta)
        z_contour = rz * np.sin(theta)
        ax2.plot(
            x_contour, z_contour, "b-", linewidth=2, alpha=0.7, label="Контур головы"
        )
        ax2.fill(x_contour, z_contour, alpha=0.1, color="lightblue")

        # Датчики с номерами
        ax2.scatter(
            self.sensor_positions[:, 0],
            self.sensor_positions[:, 2],
            c="red",
            s=60,
            alpha=0.8,
            edgecolors="darkred",
            linewidth=1,
            zorder=5,
        )

        # Номера датчиков
        for i, (x, z) in enumerate(self.sensor_positions[:, [0, 2]]):
            ax2.text(
                x,
                z,
                str(i + 1),
                fontsize=7,
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.9),
            )

        ax2.set_title(f"Вид спереди (XZ) - {self.head_type}")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Z")
        ax2.set_aspect("equal")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # 3. Вид сверху (XY проекция)
        ax3 = fig.add_subplot(223)

        # Контур головы сверху
        x_top = rx * np.cos(theta)
        y_top = ry * np.sin(theta)
        ax3.plot(x_top, y_top, "g-", linewidth=2, alpha=0.7, label="Контур головы")
        ax3.fill(x_top, y_top, alpha=0.1, color="lightgreen")

        # Датчики
        ax3.scatter(
            self.sensor_positions[:, 0],
            self.sensor_positions[:, 1],
            c="red",
            s=60,
            alpha=0.8,
            edgecolors="darkred",
            linewidth=1,
            zorder=5,
        )

        # Номера
        for i, (x, y) in enumerate(self.sensor_positions[:, [0, 1]]):
            ax3.text(
                x,
                y,
                str(i + 1),
                fontsize=7,
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.9),
            )

        ax3.set_title(f"Вид сверху (XY) - {self.head_type}")
        ax3.set_xlabel("X")
        ax3.set_ylabel("Y")
        ax3.set_aspect("equal")
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        # 4. Боковой вид (YZ проекция)
        ax4 = fig.add_subplot(224)

        # Контур головы сбоку
        y_side = ry * np.cos(theta)
        z_side = rz * np.sin(theta)
        ax4.plot(y_side, z_side, "m-", linewidth=2, alpha=0.7, label="Контур головы")
        ax4.fill(y_side, z_side, alpha=0.1, color="pink")

        # Датчики
        ax4.scatter(
            self.sensor_positions[:, 1],
            self.sensor_positions[:, 2],
            c="red",
            s=60,
            alpha=0.8,
            edgecolors="darkred",
            linewidth=1,
            zorder=5,
        )

        # Номера
        for i, (y, z) in enumerate(self.sensor_positions[:, [1, 2]]):
            ax4.text(
                y,
                z,
                str(i + 1),
                fontsize=7,
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.9),
            )

        ax4.set_title(f"Боковой вид (YZ) - {self.head_type}")
        ax4.set_xlabel("Y")
        ax4.set_ylabel("Z")
        ax4.set_aspect("equal")
        ax4.grid(True, alpha=0.3)
        ax4.legend()

        # Общая информация
        dims = self.get_head_dimensions()
        info_text = f"""Размеры головы: X={dims["rx"]:.1f}, Y={dims["ry"]:.1f}, Z={dims["rz"]:.1f}
Тип: {self.head_type}, Датчиков: {self.num_sensors}
Распределение: Только верхняя часть головы (z ≥ 0)"""

        fig.text(
            0.02,
            0.02,
            info_text,
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            verticalalignment="bottom",
        )

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)  # Место для информации

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Визуализация датчиков сохранена: {save_path}")

        plt.show()


class EEGSignalSource:
    """
    Источник сигнала для EEG с временным пиком
    """

    def __init__(
        self,
        position: List[float],
        amplitude: float = 1.0,
        peak_time: float = 0.5,
        sigma_time: float = 0.1,
    ):
        """
        Инициализация источника сигнала

        Args:
            position: [x, y, z] координаты источника
            amplitude: Амплитуда сигнала
            peak_time: Время пика сигнала
            sigma_time: Ширина пика (стандартное отклонение)
        """
        self.position = np.array(position)
        self.amplitude = amplitude
        self.peak_time = peak_time
        self.sigma_time = sigma_time

    def signal_at_point(self, t: float, point: np.ndarray) -> float:
        """
        Вычисляет сигнал источника в точке space-time

        Args:
            t: Время
            point: [x, y, z] координаты точки

        Returns:
            Значение сигнала
        """
        # Расстояние от источника до точки
        r = np.linalg.norm(point - self.position)

        # Временной гауссовский пик
        time_signal = self.amplitude * np.exp(
            -0.5 * ((t - self.peak_time) / self.sigma_time) ** 2
        )

        # Пространственное затухание (подобно электромагнитному полю)
        spatial_decay = 1.0 / (1.0 + r)

        return time_signal * spatial_decay

    def total_signal(self, t: float, points: np.ndarray) -> np.ndarray:
        """
        Вычисляет сигнал во множестве точек

        Args:
            t: Время
            points: массив точек [N, 3]

        Returns:
            Массив значений сигнала [N]
        """
        return np.array([self.signal_at_point(t, point) for point in points])


class EEGDataGenerator:
    """
    Основной генератор данных для обратной задачи ЭЭГ
    """

    def __init__(
        self,
        num_sensors: int = 64,
        sampling_rate: float = 100.0,
        duration: float = 1.0,
        noise_level: float = 0.01,
        head_type: str = "adult",
    ):
        """
        Инициализация генератора данных

        Args:
            num_sensors: Количество датчиков
            sampling_rate: Частота дискретизации в Гц (например, 100.0 = 100 Гц)
            duration: Период времени в секундах (например, 1.0 = 1 секунда)
            noise_level: Уровень шума (стандартное отклонение)
            head_type: Тип головы для генерации датчиков
        """
        self.num_sensors = num_sensors
        self.sampling_rate = sampling_rate
        self.duration = duration
        self.noise_level = noise_level
        self.head_type = head_type

        # Вычисляем количество временных точек
        self.num_time_points = int(sampling_rate * duration)

        print(
            f"📊 Параметры сигнала: {sampling_rate} Гц × {duration} сек = {self.num_time_points} отсчетов"
        )

        # Создаем генератор датчиков
        self.sensor_gen = EEGSensorGenerator(
            head_type=head_type, num_sensors=num_sensors
        )

        # Создаем источники сигнала
        self.sources = []

    def add_source(
        self,
        position: List[float],
        amplitude: float = 1.0,
        peak_time: float = 0.5,
        sigma_time: float = 0.1,
    ) -> None:
        """
        Добавляет источник сигнала

        Args:
            position: [x, y, z] координаты источника
            amplitude: Амплитуда сигнала
            peak_time: Время пика сигнала
            sigma_time: Ширина пика
        """
        source = EEGSignalSource(position, amplitude, peak_time, sigma_time)
        self.sources.append(source)
        print(f"Добавлен источник в позиции {position}")

    def create_default_sources(self) -> None:
        """
        Создает два источника по умолчанию для демонстрации
        """
        # Источник 1: левый участок коры
        self.add_source(
            position=[-3.0, 2.0, 1.0], amplitude=10.0, peak_time=0.3, sigma_time=0.08
        )

        # Источник 2: правый участок коры
        self.add_source(
            position=[3.0, -1.0, 2.0], amplitude=10.0, peak_time=0.7, sigma_time=0.12
        )

    def generate_measured_points(self) -> List[List[float]]:
        """
        Генерирует измеренные точки для обратной задачи

        Returns:
            Список точек [x, y, z, t, measurement]
        """
        if not self.sources:
            self.create_default_sources()

        # Получаем позиции датчиков
        sensor_positions = self.sensor_gen.get_sensor_positions()

        # Временные точки
        time_points = np.linspace(0.0, self.duration, self.num_time_points)

        measured_points = []

        for t in time_points:
            # Вычисляем суммарный сигнал от всех источников
            total_signal = np.zeros(self.num_sensors)

            for source in self.sources:
                source_signal = source.total_signal(t, sensor_positions)
                total_signal += source_signal

            # Добавляем шум
            noise = np.random.normal(0, self.noise_level, self.num_sensors)
            measured_signal = total_signal + noise

            # Сохраняем каждую точку измерения
            for i, (pos, signal_value) in enumerate(
                zip(sensor_positions, measured_signal)
            ):
                measured_points.append(
                    [
                        float(pos[0]),
                        float(pos[1]),
                        float(pos[2]),
                        float(t),
                        float(signal_value),
                    ]
                )

        print(
            f"Сгенерировано {len(measured_points)} измеренных точек для головы {self.head_type}"
        )
        return measured_points

    def visualize_data(
        self, measured_points: List[List[float]], save_dir: str = "figures"
    ) -> None:
        """
        Визуализирует сгенерированные данные

        Args:
            measured_points: Список измеренных точек
            save_dir: Директория для сохранения
        """
        Path(save_dir).mkdir(exist_ok=True)

        # Преобразуем в numpy массив для удобства
        data = np.array(measured_points)
        positions = data[:, :3]  # x, y, z
        times = data[:, 3]  # t
        signals = data[:, 4]  # measurement

        # 1. Визуализация датчиков
        self.sensor_gen.plot_sensors(f"{save_dir}/eeg_sensors_{self.head_type}.png")

        # 2. Временные ряды для некоторых датчиков
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Выбираем несколько датчиков для визуализации
        sensor_indices = [
            0,
            self.num_sensors // 4,
            self.num_sensors // 2,
            3 * self.num_sensors // 4,
        ]

        for idx, sensor_idx in enumerate(sensor_indices):
            ax = axes[idx // 2, idx % 2]

            # Фильтруем данные для этого датчика
            sensor_mask = np.where(
                np.isin(
                    positions[:, 0], [self.sensor_gen.sensor_positions[sensor_idx, 0]]
                )
            )[0]

            if len(sensor_mask) > 0:
                sensor_times = times[sensor_mask]
                sensor_signals = signals[sensor_mask]

                # Сортируем по времени
                sort_idx = np.argsort(sensor_times)
                sensor_times = sensor_times[sort_idx]
                sensor_signals = sensor_signals[sort_idx]

                ax.plot(sensor_times, sensor_signals, "o-", linewidth=2, markersize=6)
                ax.set_title(f"Датчик {sensor_idx + 1} (голова: {self.head_type})")
                ax.set_xlabel("Время")
                ax.set_ylabel("Сигнал")
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            f"{save_dir}/eeg_time_series_{self.head_type}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

        # 3. Пространственное распределение в разные моменты времени
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        time_indices = np.linspace(0, len(times) - 1, 6, dtype=int)

        for i, time_idx in enumerate(time_indices):
            ax = axes[i // 3, i % 3]

            # Фильтруем точки для этого времени
            time_mask = np.abs(times - times[time_idx]) < 1e-6
            time_positions = positions[time_mask]
            time_signals = signals[time_mask]

            scatter = ax.scatter(
                time_positions[:, 0],
                time_positions[:, 1],
                c=time_signals,
                cmap="viridis",
                s=50,
                alpha=0.8,
            )
            ax.set_title(f"t = {times[time_idx]:.2f} (голова: {self.head_type})")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_aspect("equal")
            plt.colorbar(scatter, ax=ax)

        plt.tight_layout()
        plt.savefig(
            f"{save_dir}/eeg_spatial_distribution_{self.head_type}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

        print(f"Визуализация сохранена в {save_dir}")

    def save_data(
        self,
        measured_points: List[List[float]],
        filename: str = "eeg_measured_data.json",
    ) -> None:
        """
        Сохраняет данные в JSON файл

        Args:
            measured_points: Список измеренных точек
            filename: Имя файла
        """
        data_dict = {
            "num_sensors": self.num_sensors,
            "sampling_rate": self.sampling_rate,
            "duration": self.duration,
            "num_time_points": self.num_time_points,
            "noise_level": self.noise_level,
            "head_type": self.head_type,
            "head_dimensions": self.sensor_gen.get_head_dimensions(),
            "sensor_positions": self.sensor_gen.get_sensor_positions().tolist(),
            "sources": [
                {
                    "position": source.position.tolist(),
                    "amplitude": source.amplitude,
                    "peak_time": source.peak_time,
                    "sigma_time": source.sigma_time,
                }
                for source in self.sources
            ],
            "measured_points": measured_points,
        }

        with open(filename, "w") as f:
            json.dump(data_dict, f, indent=2)

        print(f"Данные сохранены в {filename}")


def create_eeg_experiment(
    num_sensors: int = 64,
    sampling_rate: float = 100.0,
    duration: float = 1.0,
    noise_level: float = 0.01,
    head_type: str = "adult",
    save_visualization: bool = True,
) -> Tuple[List[List[float]], EEGDataGenerator]:
    """
    Создает полный EEG эксперимент с данными

    Args:
        num_sensors: Количество датчиков
        sampling_rate: Частота дискретизации в Гц (например, 100.0 = 100 Гц)
        duration: Период времени в секундах (например, 1.0 = 1 секунда)
        noise_level: Уровень шума
        head_type: Тип головы ('child', 'adult', 'large_adult', 'female', 'male')
        save_visualization: Сохранять ли визуализацию

    Returns:
        Кортеж (measured_points, generator)
    """
    print(f"🧠 Создание EEG эксперимента для головы {head_type}...")
    print(f"📊 Параметры сигнала: {sampling_rate} Гц × {duration} сек")

    # Создаем генератор
    generator = EEGDataGenerator(
        num_sensors, sampling_rate, duration, noise_level, head_type
    )

    # Добавляем источники сигнала
    generator.create_default_sources()

    # Генерируем данные
    measured_points = generator.generate_measured_points()

    # Создаем визуализацию
    if save_visualization:
        generator.visualize_data(measured_points)

    print(f"✅ EEG эксперимент для головы {head_type} создан!")
    return measured_points, generator


# Пример использования
if __name__ == "__main__":
    # Создаем эксперимент с разными типами головы
    for head_type in ["child", "adult", "female"]:
        print(f"\n=== Эксперимент с головой типа: {head_type} ===")

        measured_points, generator = create_eeg_experiment(
            num_sensors=24,  # Меньше датчиков для быстрого теста
            sampling_rate=100.0,  # 100 Гц
            duration=0.6,  # 0.6 секунды = 60 отсчетов
            noise_level=0.005,
            head_type=head_type,
            save_visualization=False,  # Отключаем для скорости
        )

        # Сохраняем данные
        filename = f"test_eeg_data_{head_type}.json"
        generator.save_data(measured_points, filename)

    print("\n🔬 Демонстрация использования с решателем:")
    print("measured_points =", measured_points[:3], "...")  # Показываем первые 3 точки
    print(f"Всего точек: {len(measured_points)}")
    print("\nТеперь можно использовать:")
    print("solver = EEGInverseSolver()")
    print("results = solver.solve(measured_points)")
