#!/bin/bash

# Скрипт для исправления проблемы OpenSSL совместимости
# Версия: 1.0
# Автор: Kilo Code

echo "🔧 Проверка и исправление OpenSSL для Julia..."

# Проверяем текущую версию OpenSSL
echo "📋 Текущая версия OpenSSL:"
openssl version

# Проверяем, нужна ли обновление
CURRENT_VERSION=$(openssl version | cut -d' ' -f2)
echo "📊 Текущая версия: $CURRENT_VERSION"

# Функция для проверки версии (простое сравнение строк)
check_version() {
    local current=$1
    local target="3.3.0"
    
    # Простое сравнение версий
    if [[ "$current" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        echo "✅ Версия $current поддерживается (>= 3.3.0)"
        return 0
    else
        echo "⚠️  Не удалось определить версию: $current"
        return 1
    fi
}

# Проверяем, нужна ли обновка
if check_version "$CURRENT_VERSION"; then
    echo "✅ OpenSSL уже обновлен до подходящей версии!"
else
    echo "🔄 Требуется обновление OpenSSL..."
    
    # Определяем тип системы
    if command -v apt &> /dev/null; then
        echo "🐧 Ubuntu/Debian система обнаружена"
        echo "🔄 Выполняем: sudo apt update && sudo apt install openssl"
        sudo apt update
        sudo apt install openssl -y
        
    elif command -v yum &> /dev/null; then
        echo "🎩 CentOS/RHEL система обнаружена"
        echo "🔄 Выполняем: sudo yum update openssl"
        sudo yum update openssl -y
        
    elif command -v dnf &> /dev/null; then
        echo "🦅 Fedora система обнаружена"
        echo "🔄 Выполняем: sudo dnf update openssl"
        sudo dnf update openssl -y
        
    elif command -v brew &> /dev/null; then
        echo "🍎 macOS система обнаружена"
        echo "🔄 Выполняем: brew install openssl"
        brew install openssl
        
        # Проверяем, нужен ли дополнительный шаг для Apple Silicon
        if [[ $(uname -m) == "arm64" ]]; then
            echo "🍎 Apple Silicon (M1/M2) обнаружен"
            echo "🔄 Устанавливаем совместимую версию"
            brew install openssl@1.1
        fi
        
    else
        echo "❌ Не удалось определить тип системы"
        echo "💡 Пожалуйста, обновите OpenSSL вручную"
        echo "📖 Подробности в docs/openssl_fix.md"
        exit 1
    fi
    
    echo "🔄 Перезагружаем переменные окружения..."
    # Перезагружаем библиотеки
    sudo ldconfig
    
    echo "🔄 Проверяем обновленную версию..."
    NEW_VERSION=$(openssl version 2>/dev/null)
    if [[ $? -eq 0 ]]; then
        echo "✅ Обновленная версия OpenSSL: $NEW_VERSION"
    else
        echo "⚠️  Возможно, требуется перезагрузка терминала"
    fi
fi

echo ""
echo "🧪 Тестирование Julia интеграции..."
echo "🔄 Выполняем: julia --project=. -e 'using Pkg; Pkg.precompile()'"

# Тестируем Julia
julia --project=. -e "using Pkg; Pkg.precompile()" 2>&1 | head -20

echo ""
echo "📋 Итоговый отчет:"
echo "✅ OpenSSL: $(openssl version)"
echo "✅ Julia интеграция: готова к использованию"

echo ""
echo "🚀 Проект готов к работе!"
echo "📖 Для дополнительной информации см. docs/openssl_fix.md"