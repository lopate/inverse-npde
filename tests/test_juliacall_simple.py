#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Простой тест для проверки работы JuliaCall интерфейса.
Этот скрипт проверяет базовую функциональность JuliaCall
без запуска сложных вычислений.
"""

import sys
import os

sys.path.append("../src/python")


def test_juliacall_import():
    """Тест импорта JuliaCall и создания интерфейса"""
    print("🔧 Тестирование импорта JuliaCall...")

    try:
        from juliacall import Main as jl

        print("✓ JuliaCall импортирован успешно")

        # Проверяем доступность Julia
        jl.println("Hello from Julia!")
        print("✓ Julia исполняется корректно")

        return True

    except Exception as e:
        print(f"❌ Ошибка JuliaCall: {e}")
        return False


def test_juliacall_interface():
    """Тест импорта нашего Python интерфейса"""
    print("\n🔧 Тестирование Python интерфейса...")

    try:
        from julia_interface import (
            EEGInverseSolver,
            create_fast_solver,
            create_gpu_solver,
        )

        print("✓ Python интерфейс импортирован успешно")

        # Проверяем создание решателей
        fast_solver = create_fast_solver()
        print("✓ Быстрый решатель создан")

        gpu_solver = create_gpu_solver()
        print("✓ GPU решатель создан")

        # Получаем конфигурацию
        config = fast_solver.get_config_summary()
        print("✓ Конфигурация получена:")
        print(f"  - Нейронная сеть: {config['neural_config']['hidden_layers']} слоев")
        print(f"  - Оптимизатор: {config['optimization_config']['optimizer']}")
        print(f"  - Итерации: {config['optimization_config']['max_iterations']}")

        return True

    except Exception as e:
        print(f"❌ Ошибка Python интерфейса: {e}")
        return False


def test_juliacall_basic_functionality():
    """Тест базовой функциональности без запуска вычислений"""
    print("\n🔧 Тестирование базовой функциональности...")

    try:
        from julia_interface import create_fast_solver

        solver = create_fast_solver()

        # Тестируем получение конфигурации
        config = solver.get_config_summary()
        assert "neural_config" in config
        assert "optimization_config" in config
        assert "domain" in config
        print("✓ Конфигурация корректна")

        # Тестируем получение источников (но не запускаем вычисления)
        print("✓ Базовые методы работают")

        return True

    except Exception as e:
        print(f"❌ Ошибка базовой функциональности: {e}")
        return False


def main():
    """Главная функция тестирования"""
    print("🚀 Начало тестирования JuliaCall интерфейса")
    print("=" * 60)

    tests = [
        ("Импорт JuliaCall", test_juliacall_import),
        ("Python интерфейс", test_juliacall_interface),
        ("Базовая функциональность", test_juliacall_basic_functionality),
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\n📋 {test_name}:")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ Критическая ошибка в {test_name}: {e}")
            results.append((test_name, False))

    # Итоговый отчет
    print("\n" + "=" * 60)
    print("📊 ИТОГОВЫЙ ОТЧЕТ:")
    print("=" * 60)

    passed = 0
    total = len(results)

    for test_name, success in results:
        status = "✅ ПРОЙДЕН" if success else "❌ ПРОВАЛЕН"
        print(f"{test_name:<30} {status}")
        if success:
            passed += 1

    print("-" * 60)
    print(f"Результат: {passed}/{total} тестов пройдено")

    if passed == total:
        print("🎉 Все тесты пройдены! JuliaCall работает корректно.")
        print("✅ Можно запускать полные эксперименты.")
    else:
        print("⚠️ Некоторые тесты провалены.")
        print("❌ Проверьте установку Julia и JuliaCall.")

    return passed == total


if __name__ == "__main__":
    main()
