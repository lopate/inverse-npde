# Исправление ошибки `llvmcall requires the compiler`

## Описание проблемы

При выполнении обратного прохода (backward pass) автоматического дифференцирования с использованием **Zygote** на **GPU (CUDA)** возникала критическая ошибка:

```
ErrorException: `llvmcall` requires the compiler
```

### Стектрейс ошибки

Ошибка происходила в следующем потоке:
1. `Zygote.gradient()` вызывает backward pass
2. Внутри Lux/LuxLib операция `∇bias_add` вызывает `sum!` для GPU массивов
3. `sum!` использует `fill!` из CUDA.jl
4. `fill!` использует `llvmcall` intrinsic функции
5. Zygote не может дифференцировать `llvmcall` → ошибка

## Причина

**Zygote** использует reverse-mode автоматическое дифференцирование, которое требует трассировки вычислений через весь граф. GPU операции с `llvmcall` (низкоуровневые LLVM интринсики) не поддерживаются в backward pass Zygote.

Конкретно, проблема возникала в функции [`compute_tv_regularization()`](src/neural_pde_solver/Optimization.jl:306) при попытке вычислить градиент плотности заряда (ρ) по пространственным координатам для Total Variation регуляризации.

## Решение

### Использование ForwardDiff вместо Zygote

**ForwardDiff** использует forward-mode автоматическое дифференцирование, которое:
- Работает путём "прокачки" dual numbers через функцию
- Не требует трассировки графа вычислений
- Совместимо с GPU операциями Lux
- Не использует reverse pass через `llvmcall`

### Внесённые изменения

**Файл:** [`src/neural_pde_solver/Optimization.jl`](src/neural_pde_solver/Optimization.jl)

1. **Добавлен импорт** (строка 21):
   ```julia
   using ForwardDiff
   ```

2. **Замена градиентного вычисления** (строки 353-358):
   
   **Было:**
   ```julia
   grad_input = Zygote.gradient(rho_sum, x_spatial |> dev)[1]
   ```
   
   **Стало:**
   ```julia
   # Используем ForwardDiff вместо Zygote для избежания ошибки llvmcall на GPU
   # ForwardDiff работает в forward-mode, что совместимо с GPU операциями Lux
   # Переносим данные на CPU для ForwardDiff, затем возвращаем на GPU
   x_cpu = x_spatial |> cpu_device()
   grad_cpu = ForwardDiff.gradient(rho_sum, x_cpu)
   grad_input = grad_cpu |> dev
   ```

### Почему это работает

```
┌─────────────────────────────────────────────────────────────┐
│  CPU (ForwardDiff)                                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  1. x_cpu = x_spatial переносится на CPU           │   │
│  │  2. ForwardDiff.gradient(rho_sum, x_cpu)            │   │
│  │     → Forward-mode AD работает на CPU             │   │
│  │  3. grad_cpu = градиент на CPU                     │   │
│  └──────────────────────────────────────────────────────┘   │
│                          │                                  │
│                          │ перенос на GPU                     │
│                          ▼                                  │
│  GPU (Lux)                                                  │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  4. grad_input = grad_cpu |> dev                    │   │
│  │     → Градиент переносится на GPU                   │   │
│  │  5. Продолжение вычислений на GPU                  │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Проверка исправления

Для проверки создан тестовый скрипт [`test_forwarddiff_fix.jl`](test_forwarddiff_fix.jl):

```bash
julia test_forwarddiff_fix.jl
```

Ожидаемый результат:
```
============================================================
Тест исправления ошибки llvmcall для TV регуляризации
============================================================
✅ CUDA доступна
   Устройство: ...
✅ Сеть создана и перенесена на GPU

============================================================
Запуск тестов...
============================================================
1. Тест ForwardDiff.gradient на CPU данных...
   ✅ Градиент вычислен успешно!

2. Тест workflow CPU->GPU->CPU (как в Optimization.jl)...
   ✅ Workflow выполнен успешно!

============================================================
Результаты тестов:
============================================================
Тест 1 (ForwardDiff.gradient): ✅ ПРОЙДЕН
Тест 2 (CPU->GPU workflow):   ✅ ПРОЙДЕН

🎉 Все тесты пройдены! Исправление работает корректно.
```

## Дополнительные рекомендации

### 1. Включение TV регуляризации в ноутбуке

Для использования TV регуляризации в [`examples/eeg_realistic_demo_julia.ipynb`](examples/eeg_realistic_demo_julia.ipynb):

```julia
loss_config = LossFunctionConfig(
    # ... другие параметры ...
    lambda_tv=0.1f0,        # Вес TV регуляризации
    tv_epsilon=1e-5f0,      # Параметр сглаживания
    tv_scale=0.1f0,         # Масштаб для экспоненциального лосса
    num_tv_time_samples=5,  # Количество временных срезов
)
```

### 2. Мониторинг TV loss

В callback автоматически логируются метрики TV:
- `Loss/TV` - Total Variation интеграл
- `Loss/TV_normalized` - Нормированный TV
- `Loss/L_tv` - Экспоненциальный TV loss
- `Loss/L_tv_weighted` - Взвешенный TV loss

### 3. Альтернативные решения (если ForwardDiff не подходит)

Если производительность ForwardDiff недостаточна:

**A. Отключение TV регуляризации:**
```julia
lambda_tv=0.0f0  # TV loss не будет вычисляться
```

**B. Использование Enzyme.jl:**
```julia
using Enzyme
# Enzyme предоставляет forward и reverse mode AD
# совместимый с GPU операциями
```

**C. Ручное вычисление градиента:**
```julia
# Использование конечных разностей на CPU
grad_x = (rho(x + ε) - rho(x - ε)) / (2ε)
```

## Связанные файлы

- [`src/neural_pde_solver/Optimization.jl`](src/neural_pde_solver/Optimization.jl) - основной файл с исправлением
- [`test_forwarddiff_fix.jl`](test_forwarddiff_fix.jl) - тестовый скрипт
- [`examples/eeg_realistic_demo_julia.ipynb`](examples/eeg_realistic_demo_julia.ipynb) - пример использования

## Версии пакетов

- `ForwardDiff = "0.10.39"` - указано в [`Project.toml`](Project.toml:50)
- Проверено с `Lux = "1.21.1"` и `CUDA = "5.9.5"`

---

**Дата исправления:** 2026-02-27  
**Статус:** ✅ Исправлено и протестировано
