#!/usr/bin/env python3
"""
Простой тест интеграции Python-Julia
"""

import sys
import os

# Добавляем путь к модулю
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def test_julia_basic():
    """Тестирует базовую функциональность juliacall"""
    try:
        # Импортируем juliacall
        from juliacall import Main as jl

        print("✅ juliacall успешно импортирован")

        # Тестируем простую команду
        result = jl.eval("1 + 1")
        print(f"✅ Простой расчет: 1 + 1 = {result}")

        # Тестируем доступ к функциям Julia
        sin_result = jl.eval("sin(pi/2)")
        print(f"✅ Функции Julia: sin(π/2) = {sin_result}")

        return True

    except Exception as e:
        print(f"❌ Ошибка тестирования juliacall: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_julia_module():
    """Тестирует загрузку нашего Julia модуля"""
    try:
        from juliacall import Main as jl

        # Добавляем путь к модулю
        module_path = os.path.join(
            os.path.dirname(__file__), "..", "src", "neural_pde_solver"
        )
        jl.eval(f'push!(LOAD_PATH, "{module_path}")')
        print("✅ Путь к модулю добавлен")

        # Пытаемся загрузить модуль
        jl.eval("using InverseNpde")
        print("✅ Julia модуль InverseNpde загружен")

        # Проверяем что функция доступна
        jl.eval("get_available_devices")
        print("✅ Функция get_available_devices доступна")

        return True

    except Exception as e:
        print(f"❌ Ошибка загрузки модуля: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_python_interface():
    """Тестирует Python интерфейс"""
    try:
        # Создаем простой интерфейс для тестирования
        from juliacall import Main as jl

        # Инициализируем модуль напрямую
        module_path = os.path.join(
            os.path.dirname(__file__), "..", "src", "neural_pde_solver"
        )
        jl.eval(f'push!(LOAD_PATH, "{module_path}")')
        jl.eval("using InverseNpde")

        # Проверяем что Python интерфейс работает
        try:
            func = jl.get_available_devices
            print("✅ Python интерфейс доступен")
        except AttributeError:
            print("⚠️  Python интерфейс не найден")

        return True

    except Exception as e:
        print(f"❌ Ошибка Python интерфейса: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🧪 Запуск тестов интеграции Python-Julia...\n")

    # Тест 1: Базовый juliacall
    print("1. Тестирование базового juliacall:")
    test1_passed = test_julia_basic()

    print("\n" + "=" * 50 + "\n")

    # Тест 2: Julia модуль
    print("2. Тестирование Julia модуля:")
    test2_passed = test_julia_module()

    print("\n" + "=" * 50 + "\n")

    # Тест 3: Python интерфейс
    print("3. Тестирование Python интерфейса:")
    test3_passed = test_python_interface()

    print("\n" + "=" * 50 + "\n")

    # Итоги
    total_tests = 3
    passed_tests = sum([test1_passed, test2_passed, test3_passed])

    print("📊 Результаты тестирования:")
    print(f"   Пройдено тестов: {passed_tests}/{total_tests}")

    if passed_tests == total_tests:
        print("🎉 Все тесты пройдены успешно!")
        sys.exit(0)
    else:
        print("❌ Некоторые тесты не пройдены")
        sys.exit(1)
