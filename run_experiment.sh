#!/bin/bash

# Скрипт для запуска эксперимента inverse-npde
# Автоматически настраивает окружения и запускает ноутбуки

set -e

echo "=== Запуск эксперимента inverse-npde ==="
echo

# Проверяем наличие Julia
if ! command -v julia &> /dev/null; then
    echo "Ошибка: Julia не установлена. Запустите сначала install.sh"
    exit 1
fi

# Проверяем наличие Python
if ! command -v python3 &> /dev/null; then
    echo "Ошибка: Python3 не установлен"
    exit 1
fi

# Функция для активации Python окружения
activate_python_env() {
    if [ -f ".venv/bin/activate" ]; then
        echo "Активирую Python виртуальное окружение..."
        source .venv/bin/activate
    else
        echo "Создаю Python виртуальное окружение..."
        python3 -m venv .venv
        source .venv/bin/activate
        pip install --upgrade pip
        pip install -r requirements.txt
    fi
}

# Функция для проверки Julia пакетов
check_julia_packages() {
    echo "Проверяю Julia пакеты..."
    cd experiments/base
    
    # Проверяем наличие Project.toml
    if [ ! -f "Project.toml" ]; then
        echo "Создаю Julia проект..."
        julia --project=. -e 'using Pkg; Pkg.instantiate()'
    fi
    
    # Тестируем основные пакеты
    julia --project=. -e '
        try
            using NeuralPDE
            using Lux
            using ModelingToolkit
            using CUDA
            println("✓ Julia пакеты установлены корректно")
        catch e
            println("✗ Ошибка загрузки Julia пакетов: ", e)
            exit(1)
        end
    '
    cd ../..
}

# Функция для проверки Python пакетов
check_python_packages() {
    echo "Проверяю Python пакеты..."
    source .venv/bin/activate
    python3 -c "
        try:
            import mne
            import numpy
            import pandas
            import xarray
            print('✓ Python пакеты установлены корректно')
        except ImportError as e:
            print(f'✗ Ошибка импорта Python пакетов: {e}')
            exit(1)
    "
}

# Главное меню
echo "Выберите действие:"
echo "1. Проверить установку и запустить Jupyter для Julia"
echo "2. Проверить установку и запустить Jupyter для Python"
echo "3. Только проверить установку"
echo "4. Запустить конкретный ноутбук"
echo

read -p "Введите номер действия (1-4): " choice

case $choice in
    1)
        echo "Запуск Jupyter для Julia..."
        activate_python_env
        check_julia_packages
        check_python_packages
        cd experiments/base
        echo "Запускаю Jupyter с Julia ядром..."
        julia --project=. -e 'using IJulia; notebook(; dir=pwd())'
        ;;
    2)
        echo "Запуск Jupyter для Python..."
        activate_python_env
        check_julia_packages
        check_python_packages
        echo "Запускаю Jupyter с Python ядром..."
        source .venv/bin/activate
        jupyter notebook
        ;;
    3)
        echo "Проверка установки..."
        check_julia_packages
        check_python_packages
        echo "Все проверки пройдены успешно!"
        ;;
    4)
        echo "Доступные ноутбуки:"
        echo "1. inverse_problem_eeg.ipynb (Julia)"
        echo "2. brunton_uw_bio.ipynb (Python)"
        echo "3. pde_exp_julia.ipynb (Julia)"
        read -p "Введите номер ноутбука (1-3): " notebook_choice
        
        case $notebook_choice in
            1)
                echo "Запускаю inverse_problem_eeg.ipynb..."
                activate_python_env
                check_julia_packages
                cd experiments/base
                julia --project=. -e 'using IJulia; notebook(; dir=pwd())'
                ;;
            2)
                echo "Запускаю brunton_uw_bio.ipynb..."
                activate_python_env
                check_python_packages
                source .venv/bin/activate
                cd datasets
                jupyter notebook brunton_uw_bio.ipynb
                ;;
            3)
                echo "Запускаю pde_exp_julia.ipynb..."
                activate_python_env
                check_julia_packages
                cd experiments/base
                julia --project=. -e 'using IJulia; notebook(; dir=pwd())'
                ;;
            *)
                echo "Неверный выбор"
                exit 1
                ;;
        esac
        ;;
    *)
        echo "Неверный выбор"
        exit 1
        ;;
esac

echo "Эксперимент завершен"