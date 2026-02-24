# План: Improvement-based scheduler для lambda_data

## Цель

Внедрить event-driven (improvement-based) планировщик для `lambda_data`, который
увеличивает вес данных, когда "сырой" data loss (L_data_raw) не улучшается
в течение заданного `patience`, и опционально слегка уменьшает λ при улучшении.

## Алгоритм

- Поддерживать кольцевой буфер последних `window_size` значений `L_data_raw`.
- На каждой итерации:
  - Если `L_data_raw` < best_in_window - `min_improvement` → считаем, что есть улучшение;
    сбрасываем счётчик `no_improve` и уменьшаем λ умножением на `decrease_factor`.
  - Иначе увеличиваем `no_improve`.
  - Если `no_improve` >= `patience` → увеличиваем λ умножением на `increase_factor` и
    сбрасываем `no_improve`.
  - Применяем ограничения `lambda_min`/`lambda_max` (с учётом как глобальных, так и
    schedule-специфичных границ).

## Псевдокод

```julia
# Параметры
window_size = 10
patience = 3
increase_factor = 1.25
decrease_factor = 0.99
min_improvement = 1e-8

data_buffer = fill(Inf, window_size)
buffer_pos = 1
no_improve = 0

for each iteration
    data_buffer[buffer_pos] = L_data_raw
    buffer_pos = buffer_pos % window_size + 1
    best = minimum(data_buffer)
    if L_data_raw < best - min_improvement
        no_improve = 0
        lambda *= decrease_factor
    else
        no_improve += 1
    end
    if no_improve >= patience
        lambda *= increase_factor
        no_improve = 0
    end
    lambda = clamp(lambda, lambda_min, lambda_max)
end
```

## Преимущества

- Управляемые, локальные адаптации вместо резких глобальных пересчётов.
- Простая интерпретируемость (`patience`, `increase_factor`).
- Совместимость с глобальными ограничениями `lambda_min`/`lambda_max`.

## Параметры конфигурации

- `lambda_data_init` — начальное значение λ (используется планировщиком).
- `lambda_schedule_type` — сейчас поддерживается `:improvement` (по умолчанию).
- `lambda_schedule` — Dict со значениями: `window_size`, `patience`, `increase_factor`, `decrease_factor`, `min_improvement`, `lambda_min`, `lambda_max`.

## Файлы для изменения (кратко)

1. `src/neural_pde_solver/Optimization.jl` — реализовать improvement-scheduler и логирование.
2. `src/neural_pde_solver/InverseProblem.jl` — корректно передавать `lambda_schedule` при создании `LossFunctionConfig`.
3. `src/inverse_npde/julia_interface.py` — обновить Python обёртку для передачи `lambda_schedule`.

## Тестирование

1. Запустить короткий эксперимент (несколько сотен итераций) с активированным планировщиком.
2. Проверить логи TensorBoard: `Params/lambda_data`, `Params/iter_no_improve`, `Params/lambda_action`.
3. Подтвердить, что λ увеличивается при стагнации data loss и опционально уменьшается при улучшении.
