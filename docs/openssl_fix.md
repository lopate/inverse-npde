# Исправление проблемы OpenSSL

## Проблема
```
ERROR: could not load library "/home/sasha/.julia/artifacts/7f39a18d94f87b5135df6731a327b61b8c463af6/lib/libssl.so"
/home/sasha/anaconda3/lib/python3.13/lib-dynload/../../libcrypto.so.3: version `OPENSSL_3.3.0' not found (required by ...)
```

## Решение 1: Обновление OpenSSL в системе

### Для Ubuntu/Debian:
```bash
sudo apt update
sudo apt install openssl
# Проверим версию
openssl version
```

### Для CentOS/RHEL:
```bash
sudo yum update openssl
# или
sudo dnf update openssl
```

### Для macOS:
```bash
brew install openssl
# Если используете Apple Silicon (M1/M2)
brew install openssl@1.1
```

## Решение 2: Использование системных библиотек Julia

```bash
# Создадим новый системный образ Julia, который использует системные библиотеки
julia --sysimage=/usr/lib/julia/sys-custom.so -e 'using Pkg; Pkg.resolve()'
```

## Решение 3: Пересборка Julia пакетов

```bash
# Пересоберем все пакеты с системными библиотеками
julia --project=. -e "using Pkg; Pkg.build(Pkg.installed(); verbose=true)"
```

## Решение 4: Использование conda для управления зависимостями

```bash
# Создадим conda окружение с правильной версией OpenSSL
conda create -n julia-env openssl=3.3.0 -y
conda activate julia-env
# Переустановим Julia в этом окружении
```

## Решение 5: Временное решение - использование системного Python

Если проблема критична, можно временно использовать чистый Python без Julia интеграции:

```python
# Вместо julia_interface.py используйте pure Python интерфейс
# Запустите эксперименты через Julia ноутбуки напрямую
jupyter notebook experiments/base/inverse_problem_eeg.ipynb
```

## Проверка решения

```bash
# Проверим, что OpenSSL правильно установлен
openssl version
# Должна быть версия 3.3.0 или выше

# Протестируем Julia
julia --project=. -e "using Pkg; Pkg.precompile()"
```

## Рекомендуемое решение

Для большинства случаев рекомендуется **Решение 1** (обновление OpenSSL) - это самое простое и эффективное решение.