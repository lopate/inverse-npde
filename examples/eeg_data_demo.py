#!/usr/bin/env python3
"""
Демо скрипт для генерации и визуализации EEG данных

Этот скрипт демонстрирует основные возможности модуля генерации EEG данных:
- Создание датчиков на сферической поверхности
- Генерацию множественных источников сигнала
- Визуализацию результатов
"""

import sys
import argparse
from pathlib import Path

# Добавляем путь к src для импорта модулей
sys.path.append(str(Path(__file__).parent.parent / "src"))

from inverse_npde.eeg_data_generator import EEGDataGenerator, create_eeg_experiment


def demo_basic():
    """Базовый пример создания EEG данных"""
    print("🧠 Создание базового EEG эксперимента...")

    # Создаем эксперимент
    measured_points, generator = create_eeg_experiment(
        num_sensors=16,  # Мало датчиков для быстрого теста
        num_time_points=8,  # 8 временных точек
        noise_level=0.01,  # 1% шума
        save_visualization=True,
    )

    print(f"✓ Создано {len(measured_points)} измеренных точек")
    print(f"✓ {generator.num_sensors} датчиков на сфере")
    print(f"✓ {len(generator.sources)} источников сигнала")

    # Сохраняем данные
    output_file = "demo_eeg_data.json"
    generator.save_data(measured_points, output_file)
    print(f"✓ Данные сохранены в {output_file}")


def demo_head_types():
    """Демонстрация разных типов головы"""
    print("\n👤 Демонстрация разных типов головы...")

    head_types = ["child", "adult", "female", "male", "large_adult"]

    for head_type in head_types:
        print(f"\n🔧 Создание эксперимента с головой типа: {head_type}")

        try:
            measured_points, generator = create_eeg_experiment(
                num_sensors=16,  # Мало датчиков для скорости
                num_time_points=8,
                noise_level=0.01,
                head_type=head_type,
                save_visualization=False,  # Отключаем для скорости
            )

            # Показываем размеры головы
            dimensions = generator.sensor_gen.get_head_dimensions()
            print(
                f"  ✓ Размеры: X={dimensions['rx']:.1f}, Y={dimensions['ry']:.1f}, Z={dimensions['rz']:.1f}"
            )
            print(f"  ✓ {len(measured_points)} измеренных точек создано")

        except Exception as e:
            print(f"  ❌ Ошибка: {e}")


def demo_custom():
    """Пример с кастомными параметрами"""
    print("\n🔬 Создание кастомного EEG эксперимента...")

    # Создаем генератор
    generator = EEGDataGenerator(
        num_sensors=32,
        num_time_points=12,
        noise_level=0.02,  # Больше шума
    )

    # Добавляем кастомные источники
    print("Добавляем источники:")

    # Источник 1: Левый участок коры
    generator.add_source(
        position=[-4.0, 3.0, 1.0], amplitude=1.2, peak_time=0.3, sigma_time=0.06
    )
    print("  ✓ Левый источник")

    # Источник 2: Правый участок коры
    generator.add_source(
        position=[4.0, -2.0, 2.0], amplitude=0.9, peak_time=0.7, sigma_time=0.1
    )
    print("  ✓ Правый источник")

    # Источник 3: Задний участок
    generator.add_source(
        position=[1.0, -5.0, 0.0], amplitude=0.7, peak_time=0.5, sigma_time=0.08
    )
    print("  ✓ Задний источник")

    # Генерируем данные
    measured_points = generator.generate_measured_points()

    # Создаем визуализацию
    generator.visualize_data(measured_points, "demo_custom")

    # Сохраняем
    generator.save_data(measured_points, "demo_custom_eeg.json")

    print(f"✓ Создано {len(measured_points)} измеренных точек")
    print("✓ Визуализация сохранена в папке demo_custom")


def demo_comparison():
    """Сравнение разных конфигураций"""
    print("\n📊 Сравнение разных конфигураций...")

    configs = [
        {"name": "Мало датчиков", "sensors": 16, "times": 8},
        {"name": "Стандарт", "sensors": 32, "times": 12},
        {"name": "Много датчиков", "sensors": 64, "times": 15},
    ]

    results = []

    for config in configs:
        print(f"\n🔧 Тестирование: {config['name']} ({config['sensors']} датчиков)")

        try:
            measured_points, generator = create_eeg_experiment(
                num_sensors=config["sensors"],
                num_time_points=config["times"],
                noise_level=0.01,
                save_visualization=False,  # Не сохраняем графики для сравнения
            )

            results.append(
                {"config": config, "num_points": len(measured_points), "success": True}
            )

            print(f"  ✓ {len(measured_points)} точек успешно создано")

        except Exception as e:
            print(f"  ❌ Ошибка: {e}")
            results.append({"config": config, "success": False, "error": str(e)})

    # Показываем сводку
    print("\n📈 Результаты сравнения:")
    print("-" * 60)
    print(f"{'Конфигурация':<20} {'Датчики':<10} {'Точки':<8} {'Статус':<10}")
    print("-" * 60)

    for result in results:
        config = result["config"]
        status = "✓ OK" if result["success"] else "❌ Ошибка"
        points = result.get("num_points", "-")

        print(f"{config['name']:<20} {config['sensors']:<10} {points:<8} {status:<10}")

    print("-" * 60)


def analyze_data(filename):
    """Анализ сохраненных данных"""
    print(f"\n🔍 Анализ данных из файла {filename}...")

    import json
    import numpy as np

    try:
        with open(filename, "r") as f:
            data = json.load(f)

        measured_points = data["measured_points"]
        print(f"✓ Загружено {len(measured_points)} измеренных точек")

        # Анализируем структуру
        coords = np.array(measured_points)[:, :3]
        times = np.array(measured_points)[:, 3]
        signals = np.array(measured_points)[:, 4]

        print("\n📊 Статистика данных:")
        print(f"  Координаты X: [{coords[:, 0].min():.2f}, {coords[:, 0].max():.2f}]")
        print(f"  Координаты Y: [{coords[:, 1].min():.2f}, {coords[:, 1].max():.2f}]")
        print(f"  Координаты Z: [{coords[:, 2].min():.2f}, {coords[:, 2].max():.2f}]")
        print(f"  Время: [{times.min():.2f}, {times.max():.2f}]")
        print(f"  Сигнал: [{signals.min():.6f}, {signals.max():.6f}]")
        print(f"  Средний сигнал: {signals.mean():.6f}")
        print(f"  Стандартное отклонение: {signals.std():.6f}")

        print(f"\n📍 Конфигурация:")
        print(f"  Датчиков: {data.get('num_sensors', 'N/A')}")
        print(f"  Временных точек: {data.get('num_time_points', 'N/A')}")
        print(f"  Источников: {len(data.get('sources', []))}")

        return True

    except FileNotFoundError:
        print(f"❌ Файл {filename} не найден")
        return False
    except Exception as e:
        print(f"❌ Ошибка анализа: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Демо скрипт для генерации EEG данных")
    parser.add_argument(
        "command",
        choices=["basic", "custom", "comparison", "analyze"],
        help="Команда для выполнения",
    )
    parser.add_argument(
        "--file", "-f", type=str, help="Файл для анализа (для команды analyze)"
    )

    args = parser.parse_args()

    print("🧠 Демо скрипт генерации EEG данных")
    print("=" * 50)

    if args.command == "basic":
        demo_basic()
    elif args.command == "custom":
        demo_custom()
    elif args.command == "comparison":
        demo_comparison()
    elif args.command == "analyze":
        if not args.file:
            print("❌ Для команды 'analyze' необходимо указать файл через --file")
            sys.exit(1)
        analyze_data(args.file)

    print("\n✅ Демо завершено!")
    print("\n💡 Подсказки:")
    print("  - Для просмотра созданных графиков откройте папки figures/, demo_custom/")
    print(
        "  - Для анализа данных используйте: python eeg_data_demo.py analyze <файл.json>"
    )
    print(
        "  - Для интеграции с решателем см. примеры в examples/eeg_realistic_demo.ipynb"
    )


if __name__ == "__main__":
    main()
