# Подробный план исправления: применение весов lambda_pde и lambda_bc

## Диагноз

**Проблема:** Параметры `lambda_pde` и `lambda_bc` в `LossFunctionConfig` не влияют на соответствующие потери.

**Корневая причина:** В NeuralPDE суммарная функция потерь формируется как:
```
total_loss = L_pde + L_bc + L_additional
```
где `L_pde` и `L_bc` вычисляются **внутри NeuralPDE без применения весов** из `LossFunctionConfig`. Веса просто хранятся в структуре, но не используются при расчёте градиентов.

---

## Подробный план изменений

### Шаг 1: Модификация `create_additional_loss` (Optimization.jl:409)

**Текущий код:**
```julia
function create_additional_loss(loss_config::LossFunctionConfig, lambda_data_ref::Ref{Float64})
    # ... только data loss
end
```

**Новый код:**
```julia
function create_additional_loss(
    loss_config::LossFunctionConfig, 
    lambda_data_ref::Ref{Float64};
    pde_inner_loss_functions::Union{Vector{Function}, Nothing}=nothing,
    bcs_inner_loss_functions::Union{Vector{Function}, Nothing}=nothing
)
    # ... существующий код data loss
    
    # НОВОЕ: Добавляем вычисление PDE и BC потерь с весами
    if pde_inner_loss_functions !== nothing
        lambda_pde = loss_config.lambda_pde
        pde_losses = map(l_ -> l_(θ), pde_inner_loss_functions)
        L_pde_weighted = lambda_pde * sum(pde_losses)
    else
        L_pde_weighted = Float32(0.0)
    end
    
    if bcs_inner_loss_functions !== nothing
        lambda_bc = loss_config.lambda_bc
        bcs_losses = map(l_ -> l_(θ), bcs_inner_loss_functions)
        L_bc_weighted = lambda_bc * sum(bcs_losses)
    else
        L_bc_weighted = Float32(0.0)
    end
    
    # Обновляем результат
    result = result + L_pde_weighted + L_bc_weighted
end
```

---

### Шаг 2: Модификация `create_discretization` (Optimization.jl:605)

**Текущий код:**
```julia
function create_discretization(chain, ps, loss_config::LossFunctionConfig, 
                              opt_config::OptimizationConfig, domain_config,
                              lambda_data_ref::Ref{Float64}=Ref{Float64}(loss_config.lambda_data_init))
    
    strategy = QuasiRandomTraining(domain_config["num_points"])
    additional_loss_func = create_additional_loss(loss_config, lambda_data_ref)
    # ...
end
```

**Новый код:**
```julia
function create_discretization(chain, ps, loss_config::LossFunctionConfig, 
                              opt_config::OptimizationConfig, domain_config,
                              lambda_data_ref::Ref{Float64}=Ref{Float64}(loss_config.lambda_data_init);
                              pde_inner_loss_functions::Union{Vector{Function}, Nothing}=nothing,
                              bcs_inner_loss_functions::Union{Vector{Function}, Nothing}=nothing)
    
    strategy = QuasiRandomTraining(domain_config["num_points"])
    
    # ПЕРЕДАЁМ loss functions в create_additional_loss
    additional_loss_func = create_additional_loss(
        loss_config, 
        lambda_data_ref;
        pde_inner_loss_functions=pde_inner_loss_functions,
        bcs_inner_loss_functions=bcs_inner_loss_functions
    )
    # ...
end
```

---

### Шаг 3: Обновление вызова в InverseProblem.jl

**Текущий код:**
```julia
discretization, lambda_data_ref = Optimization.create_discretization(
    chain, ps, loss_config, opt_config, domain_config, lambda_data_ref
)
```

**Новый код:**
```julia
# Сначала создаём discretization БЕЗ loss functions (они станут доступны после symbolic_discretize)
discretization, lambda_data_ref = Optimization.create_discretization(
    chain, ps, loss_config, opt_config, domain_config, lambda_data_ref
)

# Получаем loss functions через symbolic_discretize
sym_prob = NeuralPDE.symbolic_discretize(pde_system, discretization)
pde_inner_loss_functions = sym_prob.loss_functions.pde_loss_functions
bcs_inner_loss_functions = sym_prob.loss_functions.bc_loss_functions

# Создаём НОВЫЙ additional_loss_func С весами
additional_loss_func_with_weights = Optimization.create_additional_loss(
    loss_config, 
    lambda_data_ref;
    pde_inner_loss_functions=pde_inner_loss_functions,
    bcs_inner_loss_functions=bcs_inner_loss_functions
)

# Пересоздаём discretization с весами (альтернативный подход)
# ИЛИ: просто обновляем loss functions в существующем discretization
```

---

### Шаг 4: Альтернативный подход - полная реконструкция

**Более чистый подход:** Изменить порядок создания, чтобы `symbolic_discretize` вызывался ДО создания `additional_loss_func`.

```julia
# 1. Создаём базовый discretization БЕЗ additional_loss
discretization_base = PhysicsInformedNN(chain, strategy; init_params = ps)

# 2. Получаем loss functions
sym_prob = symbolic_discretize(pde_system, discretization_base)
pde_inner_loss_functions = sym_prob.loss_functions.pde_loss_functions
bcs_inner_loss_functions = sym_prob.loss_functions.bc_loss_functions

# 3. Создаём additional_loss с весами
additional_loss_func = create_additional_loss(
    loss_config, 
    lambda_data_ref;
    pde_inner_loss_functions=pde_inner_loss_functions,
    bcs_inner_loss_functions=bcs_inner_loss_functions
)

# 4. Создаём ФИНАЛЬНЫЙ discretization с additional_loss
discretization = PhysicsInformedNN(chain, strategy; 
    init_params = ps, 
    additional_loss = additional_loss_func,
    log_options = log_options)
```

---

## Ключевые изменения в коде

### Файл: src/neural_pde_solver/Optimization.jl

1. **Сигнатура `create_additional_loss`** (строка ~409):
   - Добавить опциональные параметры `pde_inner_loss_functions` и `bcs_inner_loss_functions`
   - Добавить логику вычисления `L_pde_weighted` и `L_bc_weighted`
   - Интегрировать их в `result`

2. **Сигнатура `create_discretization`** (строка ~605):
   - Принять новые параметры loss functions
   - Передать их в `create_additional_loss`

3. **Логирование** (строка ~742-760):
   - Добавить логирование взвешенных потерь
   - Показать влияние весов на общий loss

---

## Ожидаемый результат

После применения исправлений:
- `lambda_pde` будет умножаться на PDE loss
- `lambda_bc` будет умножаться на BC loss
- Общий loss: `total = lambda_pde * L_pde + lambda_bc * L_bc + lambda_data * L_data + lambda_field * L_field`

---

## Тестирование

1. Создать тестовый скрипт `test_lambda_weights.jl`
2. Установить `lambda_pde = 2.0`, `lambda_bc = 3.0`
3. Проверить что PDE и BC потери в 2 и 3 раза больше соответственно
4. Проверить что градиенты корректно проходят через веса

---

## Примечание

Это изменение затрагивает архитектуру NeuralPDE - мы "подменяем" внутреннюю функцию потерь на свою версию с весами. Это общий паттерн для кастомизации NeuralPDE.
