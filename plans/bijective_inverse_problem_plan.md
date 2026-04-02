# План превращения обратной задачи ЭЭГ в биективное отображение

## Анализ текущей ситуации

### Существующая архитектура
- **Прямая задача**: PDE система → φ, A по P (вектору поляризации)
- **Обратная задача**: оптимизация P → измеренные φ на датчиках
- **Проблема**: неединство решения (разные P могут давать одинаковые φ на поверхности)

### Текущие компоненты
- `PDEDefinitions.jl`: волновые уравнения для φ, A
- `HeadConstraints.jl`: ограничение P внутри эллипсоида
- `PML.jl`: поглощение на границах
- `InverseProblem.jl`: оптимизация для восстановления P

---

## План достижения биективности

### Этап 1: Создание явного оператора отображения (ключевой)

**1.1. Явное интегральное представление**
- Вывести оператор Грина для волнового уравнения в ограниченной области
- Создать явное отображение: `G: P(x,t) → φ(sensor, t)` где G — интегральный оператор
- Реализовать в `PDEDefinitions.jl` функцию `green_function_operator`

**1.2. Дискретизация оператора**
- Построить матрицу Грина `G_ij = G(x_j, x_i)` где x_j — точки источников, x_i — датчики
- Сделать G обратимой через добавление регуляризации (Tikhonov)
- Сохранить в `src/neural_pde_solver/GreenOperator.jl`

### Этап 2: Биективная параметризация источников

**2.1. Параметризация через базисные функции**
- Использовать компактные базисные функции с минимальной поддержкой
- Параметры: центры, амплитуды, ширины (вместо поля P на всей сетке)
- Количество параметров << количества степеней свободы сетки

**2.2. Sparsity-inducing регуляризация**
- L0/L1 регуляризация на коэффициенты базисных функций
- Принудительная концентрация источников в минимуме (few active sources)
- Добавить в `Optimization.jl`: `sparsity_loss`, `elastic_net_loss`

### Этап 3: Ограничение на физические свойства

**3.1. Энергетическое ограничение**
- Ограничение на L2 норму P: `||P||₂² < E_max`
- Ограничение на энергию: `∫|∂P/∂t|² dt < J_max`
- Добавить в `HeadConstraints.jl`: `energy_constraint`

**3.2. Временная компактность**
- Ограничение на количество временных интервалов активности
- Потребовать, чтобы P было сосредоточено в K << T временных точек
- Добавить в `Optimization.jl`: `temporal_sparsity_loss`

### Этап 4: Двунаправленная архитектура (опционально)

**4.1. Инвертируемая нейросеть (нGAN/realNVP)**
- Создать обратимую архитектуру для G
- Обучить: forward (P → φ) + inverse (φ → P) одновременно
- Гарантирует биективность через конструкцию

**4.2 Совместное обучение**
- Добавить cycle consistency loss: `G(G⁻¹(φ)) ≈ φ`
- Добавить identity loss: `G⁻¹(G(P)) ≈ P`

---

## Реализация (приоритеты)

### Приоритет 1: Явный оператор Грина
```
src/neural_pde_solver/GreenOperator.jl (новый)
├── GreenMatrix struct
├── compute_green_matrix(pde_system, sensor_positions, source_positions)
├── apply_green_operator(G, P)
└── inverse_green_operator(G, regularization)
```

### Приоритет 2: Базисная параметризация источников
```
src/neural_pde_solver/SourceBasis.jl (новый)
├── BasisConfig (num_sources, basis_type)
├── GaussianBasis, WaveletBasis, BumpBasis
├── source_parameters_to_P(params) -> P field
└── P_to_source_parameters(P) -> params (через G)
```

### Приоритет 3: Sparsity регуляризация
```
src/neural_pde_solver/SparsityRegularizers.jl (новый)
├── L0_loss(params, threshold)
├── L1_loss(params)
├── GroupLasso_loss(params, group_indices)
└── TotalVariation_spatial(P_grid)
```

### Приоритет 4: Обновление обратной задачи
```
src/neural_pde_solver/BijectiveInverseProblem.jl (новый на основе InverseProblem.jl)
├── bijective_loss(P_params, measured_phi, G)
├── forward_map(P_params, G) -> predicted_phi
└── inverse_map(measured_phi, G_inv) -> P_params
```

---

## Ожидаемые результаты

1. **Биективность**: каждое измерение φ однозначно соответствует набору параметров источников
2. **Концентрация**: P сконцентрирован в минимуме источников (< 10 активных вместо сетки 1000+ точек)
3. **Устойчивость**: матрица G⁻¹ хорошо обусловлена с Tikhonov регуляризацией
4. **Скорость**: прямая задача — умножение G, обратная — G⁻¹ (без итеративной оптимизации)

---

## Связь с существующим кодом

- `PDEDefinitions.jl`: добавить `green_function` вычисление
- `HeadConstraints.jl`: расширить `HeadConfig` с параметрами sparsity
- `Optimization.jl`: добавить sparsity-aware optimizers (ISTA, FISTA)
- `InverseProblem.jl`: рефакторить в `BijectiveInverseProblem.jl`