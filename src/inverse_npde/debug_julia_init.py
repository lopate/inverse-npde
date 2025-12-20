#!/usr/bin/env python3
"""
Диагностический скрипт для выявления причины проблемы с OpenSSL в Julia интерфейсе.
Проверяет версии OpenSSL, состояние Julia среды и пытается воспроизвести ошибку.
"""

import logging
import subprocess
import sys
from pathlib import Path

# Настройка логирования
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def check_system_openssl():
    """Проверяет версию OpenSSL в системе."""
    try:
        result = subprocess.run(
            ["openssl", "version"], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            logger.info(f"✅ Системная версия OpenSSL: {result.stdout.strip()}")
            return result.stdout.strip()
        else:
            logger.error(f"❌ Ошибка проверки OpenSSL: {result.stderr}")
            return None
    except Exception as e:
        logger.error(f"❌ Не удалось проверить системную версию OpenSSL: {e}")
        return None


def check_julia_environment():
    """Проверяет состояние Julia среды."""
    try:
        # Проверяем Julia статус
        result = subprocess.run(
            ["julia", "--project=.", "-e", "using Pkg; Pkg.status()"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            logger.info("✅ Julia среда доступна")
            logger.info(f"Julia статус:\n{result.stdout}")
            return True
        else:
            logger.error(f"❌ Ошибка доступа к Julia среде: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"❌ Ошибка проверки Julia среды: {e}")
        return False


def test_julia_openssl_loading():
    """Тестирует загрузку OpenSSL_jll в Julia."""
    try:
        logger.info("🔄 Тестирование загрузки OpenSSL_jll...")

        # Создаем простой Julia скрипт для тестирования
        julia_test_script = """
try
    using OpenSSL_jll
    println("SUCCESS: OpenSSL_jll loaded successfully")
    println("OpenSSL version available: ", get(ENV, "OPENSSL_JLL_VERSION", "unknown"))
catch e
    println("ERROR: Failed to load OpenSSL_jll")
    println("Error message: ", e)
    println("Error type: ", typeof(e))
end
"""

        # Записываем скрипт во временный файл
        test_script_path = Path("/tmp/test_openssl_jl.jl")
        with open(test_script_path, "w") as f:
            f.write(julia_test_script)

        # Запускаем скрипт
        result = subprocess.run(
            ["julia", "--project=.", str(test_script_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )

        logger.info(f"Julia вывод:\n{result.stdout}")
        if result.stderr:
            logger.error(f"Julia ошибки:\n{result.stderr}")

        # Удаляем временный файл
        test_script_path.unlink(missing_ok=True)

        return "SUCCESS" in result.stdout

    except Exception as e:
        logger.error(f"❌ Ошибка тестирования OpenSSL_jll: {e}")
        return False


def test_python_julia_interface():
    """Тестирует инициализацию Python-Julia интерфейса."""
    try:
        logger.info("🔄 Тестирование Python-Julia интерфейса...")

        # Добавляем src в Python path
        src_path = Path(__file__).parent
        sys.path.insert(0, str(src_path))

        # Пытаемся импортировать и инициализировать интерфейс
        from inverse_npde.julia_interface import EEGInverseSolver

        logger.info("✅ Модуль julia_interface импортирован успешно")

        # Создаем экземпляр (это должно вызвать _preload_critical_modules)
        logger.info("🔄 Создание экземпляра EEGInverseSolver...")
        solver = EEGInverseSolver()
        logger.info("✅ EEGInverseSolver создан успешно")

        return True

    except Exception as e:
        logger.error(f"❌ Ошибка тестирования Python-Julia интерфейса: {e}")
        import traceback

        logger.error(f"Полная ошибка:\n{traceback.format_exc()}")
        return False


def diagnose_openssl_issue():
    """Основная функция диагностики проблемы с OpenSSL."""
    logger.info("🔍 Начинаем диагностику проблемы OpenSSL в Julia интерфейсе")
    logger.info("=" * 60)

    # 1. Проверяем системную версию OpenSSL
    logger.info("\n1️⃣ Проверка системной версии OpenSSL")
    openssl_version = check_system_openssl()

    # 2. Проверяем Julia среду
    logger.info("\n2️⃣ Проверка Julia среды")
    julia_available = check_julia_environment()

    # 3. Тестируем загрузку OpenSSL_jll
    logger.info("\n3️⃣ Тестирование загрузки OpenSSL_jll")
    openssl_jl_ok = test_julia_openssl_loading()

    # 4. Тестируем Python-Julia интерфейс
    logger.info("\n4️⃣ Тестирование Python-Julia интерфейса")
    interface_ok = test_python_julia_interface()

    # Подводим итоги
    logger.info("\n" + "=" * 60)
    logger.info("📊 ИТОГИ ДИАГНОСТИКИ:")
    logger.info(f"• Системная версия OpenSSL: {openssl_version}")
    logger.info(f"• Julia среда доступна: {'✅' if julia_available else '❌'}")
    logger.info(f"• OpenSSL_jll загружается: {'✅' if openssl_jl_ok else '❌'}")
    logger.info(f"• Python-Julia интерфейс работает: {'✅' if interface_ok else '❌'}")

    # Диагноз
    logger.info("\n🔬 ПРЕДВАРИТЕЛЬНЫЙ ДИАГНОЗ:")
    if not openssl_jl_ok:
        logger.error("❌ ПРОБЛЕМА: OpenSSL_jll не может загрузиться")
        if openssl_version and "3.0." in openssl_version:
            logger.error(
                "🔍 ПРИЧИНА: Система имеет OpenSSL 3.0.x, но Julia требует 3.3.0"
            )
            logger.error("💡 РЕШЕНИЕ: Обновите OpenSSL до версии 3.3.0 или выше")
    else:
        logger.info("✅ OpenSSL_jll работает корректно")

    if not interface_ok:
        logger.error("❌ ПРОБЛЕМА: Python-Julia интерфейс не инициализируется")
        logger.error(
            "🔍 ПРИЧИНА: Возможна проблема с предварительной загрузкой модулей"
        )
    else:
        logger.info("✅ Python-Julia интерфейс работает корректно")


if __name__ == "__main__":
    diagnose_openssl_issue()
