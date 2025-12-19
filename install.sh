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

echo "Шаг 2: Установка системных зависимостей..."
sudo apt install -y curl wget git build-essential libopenblas-dev liblapack-dev libfftw3-dev

echo "Шаг 3: Установка Julia через juliaup (рекомендуемый метод)..."
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

echo "Шаг 4: Установка последней стабильной версии Julia..."
juliaup update

echo "Шаг 5: Установка IJulia для работы с ноутбуками..."
julia -e 'using Pkg; Pkg.add("IJulia")'

echo "Шаг 6: Создание и активация Julia окружения проекта..."
cd experiments/base
julia --project=. -e 'using Pkg; Pkg.instantiate()'

echo "Шаг 7: Установка Python и настройка интеграции (опционально)..."
# Устанавливаем Python если его нет
if ! command -v python3 &> /dev/null; then
    sudo apt install -y python3 python3-pip python3-venv
fi

echo "Шаг 8: Создание Python виртуального окружения для проекта..."
cd ../..
python3 -m venv .venv
source .venv/bin/activate

echo "Шаг 9: Установка Python зависимостей..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Шаг 10: Установка PythonCall для интеграции Julia-Python..."
julia --project=. -e 'using Pkg; Pkg.add("PythonCall")'

echo "=== Установка завершена успешно! ==="
echo
echo "Для использования проекта:"
echo "1. Активируйте Python окружение: source .venv/bin/activate"
echo "2. Перейдите в директорию experiments/base"
echo "3. Запустите Jupyter: julia --project=. -e 'using IJulia; notebook()'"
echo "   или для Python: source .venv/bin/activate && jupyter notebook"
echo
echo "Для проверки установки:"
echo "julia --project=. -e 'using InverseNpde; println("Julia пакеты установлены корректно")'"
echo "python -c 'import mne; print("Python пакеты установлены корректно")'"