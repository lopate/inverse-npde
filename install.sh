#!/bin/bash

# Скрипт автоматической установки Julia и зависимостей для проекта inverse-npde
# Целевая платформа: Linux
# Автор: Исследовательская группа

set -e  # Прекратить выполнение при любой ошибке

echo "=== Скрипт установки Julia и зависимостей для inverse-npde ==="
echo

# Проверка операционной системы
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo "Ошибка: Этот скрипт предназначен только для Linux"
    exit 1
fi

echo "Шаг 1: Обновление системы..."
sudo apt update && sudo apt upgrade -y

echo "Шаг 2: Обновление OpenSSL (критично для Julia интеграции)..."
# Проверяем и обновляем OpenSSL для совместимости с Julia
CURRENT_OPENSSL=$(openssl version 2>/dev/null | cut -d' ' -f2 || echo "unknown")
echo "Текущая версия OpenSSL: $CURRENT_OPENSSL"

# Если версия старая или неизвестна, обновляем
if [[ "$CURRENT_OPENSSL" == "unknown" ]] || [[ "$(echo $CURRENT_OPENSSL | cut -d. -f1)" -lt "3" ]] || [[ "$(echo $CURRENT_OPENSSL | cut -d. -f2)" -lt "3" ]]; then
    echo "Обновляем OpenSSL для совместимости с Julia..."
    sudo apt install -y openssl libssl-dev
    echo "Перезагружаем библиотеки..."
    sudo ldconfig
else
    echo "✅ OpenSSL уже обновлен ($CURRENT_OPENSSL)"
fi

# Проверяем, что Julia пакеты будут работать
julia --project=. -e "using Pkg; Pkg.precompile()" > /dev/null 2>&1 || {
    echo "⚠️  Обнаружена проблема с OpenSSL совместимостью Julia"
    echo "📖 Попробуйте запустить: ./fix_openssl.sh"
    echo "📖 Подробности в docs/openssl_fix.md"
}

echo "Шаг 3: Установка системных зависимостей..."
sudo apt install -y curl wget git build-essential libopenblas-dev liblapack-dev libfftw3-dev

echo "Шаг 4: Установка Julia через juliaup (рекомендуемый метод)..."
if ! command -v juliaup &> /dev/null; then
    curl -fsSL https://install.julialang.org | sh
    source ~/.bashrc
    # Перезагружаем PATH для текущей сессии
    export PATH="$HOME/.juliaup/bin:$PATH"
else
    echo "juliaup уже установлен"
fi

# Проверяем, что julia доступен
if ! command -v julia &> /dev/null; then
    echo "Ошибка: Julia не найдена в PATH"
    exit 1
fi

echo "Шаг 5: Установка последней стабильной версии Julia..."
juliaup update

echo "Шаг 6: Установка IJulia для работы с ноутбуками..."
julia -e 'using Pkg; Pkg.add("IJulia")'

echo "Шаг 7: Создание и активация Julia окружения проекта..."
julia --project=. -e 'using Pkg; Pkg.instantiate()'

echo "Шаг 8: Установка Python и настройка интеграции (опционально)..."
# Устанавливаем Python если его нет
if ! command -v python3 &> /dev/null; then
    sudo apt install -y python3 python3-pip python3-venv
fi

echo "Шаг 9: Создание Python виртуального окружения для проекта..."
python3 -m venv .venv
source .venv/bin/activate

echo "Шаг 10: Установка Python зависимостей проекта в редактируемом режиме..."
pip install -e .

echo "Шаг 11: Установка Python зависимостей..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Шаг 12: Установка JuliaCall для интеграции Julia-Python..."
# Получаем путь к исполняемому файлу Julia
JULIA_PATH=$(which julia)

# Настройка переменных окружения для JuliaCall
echo 'export PYTHON_JULIACALL_PROJECT="$(pwd)"' >> ~/.bashrc
echo 'export PYTHON_JULIACALL_EXE="'"$JULIA_PATH"'"' >> ~/.bashrc

# Настройка переменной в активационном скрипте Python (правильный путь к корню проекта)
echo 'export PYTHON_JULIACALL_PROJECT="$(cd $(dirname "${BASH_SOURCE[0]}")/../.. && pwd)"' >> .venv/bin/activate
echo 'export PYTHON_JULIACALL_EXE="'"$JULIA_PATH"'"' >> .venv/bin/activate

echo "Шаг 13: Проверка установки и совместимости..."
echo "Проверяем Julia пакеты..."
julia --project=. -e 'using InverseNpde; println("✅ Julia пакеты установлены корректно")' || {
    echo "⚠️  Ошибка: Julia пакеты не загружаются"
    echo "📖 Попробуйте: ./fix_openssl.sh"
    echo "📖 Подробности в docs/openssl_fix.md"
}

echo "Проверяем Python-Julia интеграцию..."
python -c 'from julia_interface import EEGInverseSolver; print("✅ JuliaCall интерфейс работает корректно")' || {
    echo "⚠️  Ошибка: JuliaCall интеграция не работает"
    echo "📖 Попробуйте: ./fix_openssl.sh"
    echo "📖 Подробности в docs/openssl_fix.md"
}

echo
echo "=== Установка завершена успешно! ==="
echo
echo "Для использования проекта:"
echo "1. Активируйте Python окружение: source .venv/bin/activate"
echo "2. Перейдите в директорию experiments/base"
echo "3. Запустите Jupyter: julia --project=. -e 'using IJulia; notebook()'"
echo "   или для Python: source .venv/bin/activate && jupyter notebook"
echo
echo "Если возникают проблемы с OpenSSL:"
echo "- Запустите: ./fix_openssl.sh"
echo "- Читайте: docs/openssl_fix.md"
echo
echo "Для проверки установки (повторно):"
echo "julia --project=. -e 'using InverseNpde; println("Julia пакеты установлены корректно")'"
echo "python -c 'from julia_interface import EEGInverseSolver; print("JuliaCall интерфейс работает корректно")'"