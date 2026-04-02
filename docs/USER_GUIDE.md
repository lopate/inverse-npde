# Руководство пользователя: inverse-npde

Это подробное руководство поможет вам эффективно использовать проект inverse-npde для решения обратных задач ЭЭГ.

## 📋 Содержание

1. [Введение](#введение)
2. [Начало работы](#начало-работы)
3. [Работа с данными EEG](#работа-с-данными-eeg)
4. [Выполнение экспериментов](#выполнение-экспериментов)
5. [Интерпретация результатов](#интерпретация-результатов)
6. [Настройка параметров](#настройка-параметров)
7. [PML граничные условия](#pml-граничные-условия)
8. [Интеграция Julia-Python](#интеграция-julia-python)
9. [Часто задаваемые вопросы](#часто-задаваемые-вопросы)

## Введение

Проект inverse-npde предназначен для решения обратных задач электроэнцефалографии (ЭЭГ) с использованием современных методов машинного обучения и нейронных дифференциальных уравнений.

### Основные возможности

- 🔬 **Решение обратных задач ЭЭГ** - восстановление источников мозговой активности
- 🧠 **Нейронные PDE** - использование физически информированных нейронных сетей
- 📊 **Обработка данных EEG** - загрузка, предобработка и анализ электрофизиологических данных
- 🔗 **Интеграция языков** - бесшовная работа между Julia и Python
- 📈 **Визуализация результатов** - создание информативных графиков и отчетов

## Начало работы

### Предварительные требования

Перед началом работы убедитесь, что у вас установлены:

- **Julia 1.9+** с пакетным менеджером Pkg
- **Python 3.8+** с виртуальными окружениями
- **Jupyter** для работы с ноутбуками
- **CUDA** (опционально) для ускорения вычислений на GPU

### Первый запуск

1. **Клонируйте репозиторий**:
```bash
git clone https://github.com/your-username/inverse-npde.git
cd inverse-npde
```

2. **Запустите автоматическую установку**:
```bash
chmod +x install.sh
./install.sh
```

3. **Проверьте установку**:
```bash
./run_experiment.sh
# Выберите опцию 3 для проверки
```

4. **Запустите первый эксперимент**:
```bash
./run_experiment.sh
# Выберите опцию 1 для запуска Julia ноутбуков
```

## Работа с данными EEG

### Поддерживаемые форматы

Проект поддерживает следующие форматы данных EEG:

- **EDF/EDF+** - European Data Format
- **BrainVision** - формат компании Brain Products
- **MATLAB** - файлы .mat с данными EEG
- **NetCDF** - формат для больших массивов данных

### Загрузка данных

Используйте ноутбук `datasets/brunton_uw_bio.ipynb` для загрузки и предобработки данных:

```python
import mne
import numpy as np
import pandas as pd
import xarray as xr

# Загрузка данных
raw = mne.io.read_raw_edf('path/to/data.edf', preload=True)

# Предобработка
raw.filter(1.0, 50.0)  # Фильтрация
raw.set_eeg_reference('average')  # Референс

# Создание эпох
events = mne.find_events(raw)
epochs = mne.Epochs(raw, events, tmin=-0.2, tmax=0.8)
```

### Структура данных

После предобработки данные имеют следующую структуру:

```
DataArray dimensions:
- events: количество эпох
- channels: количество электродов (обычно 61)
- time: временные точки (зависит от частоты дискретизации)
```

## Выполнение экспериментов

### Основной эксперимент

Файл: `experiments/base/inverse_problem_eeg.ipynb`

Этот ноутбук содержит полный пайплайн для решения обратной задачи ЭЭГ:

#### 1. Настройка окружения
```julia
using NeuralPDE, Lux, ModelingToolkit, Optimization
using CUDA, Plots, TensorBoardLogger
```

#### 2. Определение физической модели
```julia
@parameters x, y, z, t
@variables φ(..), Ax(..),Ay(..),Az(..), Px(..), Py(..), Pz(..)

# Определение уравнений Максвелла в терминах поляризации P:
# ρ = -div P = -(∂Px/∂x + ∂Py/∂y + ∂Pz/∂z)
# j = ∂P/∂t = (∂Px/∂t, ∂Py/∂t, ∂Pz/∂t)
eqs = [
    dalembert_operator(φ(x, y, z, t), [x, y, z], ε, μ, c) ~ 4 * pi * (Differential(x)(Px(x, y, z, t)) + Differential(y)(Py(x, y, z, t)) + Differential(z)(Pz(x, y, z, t))) / ε
    # ... дополнительные уравнения
]
```

#### 3. Создание нейронной сети
```julia
input_dim = 4  # x, y, z, t
hidden_dim = 32
output_dim = 7  # φ, Ax, Ay, Az, Px, Py, Pz

chain = Chain(
    Dense(input_dim, hidden_dim, σ),
    Dense(hidden_dim, hidden_dim, σ),
    Dense(hidden_dim, hidden_dim, σ),
    Dense(hidden_dim, output_dim)
)
```

#### 4. Обучение модели
```julia
strategy = QuasiRandomTraining(4096)
discretization = PhysicsInformedNN(chain, strategy; 
    init_params = ps, 
    additional_loss = additional_loss_weightened(lambda))

prob = discretize(pde_system, discretization)
res = solve(prob, opt; maxiters = 3000, callback)
```

### Эксперименты с PDE

Файл: `experiments/base/pde_exp_julia.ipynb`

Содержит дополнительные эксперименты с частными дифференциальными уравнениями:

- Тестирование различных архитектур нейронных сетей
- Сравнение методов оптимизации
- Исследование влияния гиперпараметров

## Интерпретация результатов

### Выходные данные

После выполнения экспериментов вы получите:

1. **Графики распределений**:
   - `plot_phi.png` - распределение электрического потенциала
   - `plot_rho.png` - распределение плотности заряда
   - Дополнительные графики для компонент магнитного поля

2. **Логи обучения**:
   - TensorBoard логи для мониторинга процесса обучения
   - CSV файлы с метриками

3. **Сохраненные модели**:
   - Обученные параметры нейронной сети
   - Конфигурация эксперимента

### Анализ результатов

#### Оценка качества решения

```julia
# Вычисление ошибки
error = norm(analytic_solution .- predicted_solution)

# Визуализация
p1 = heatmap(xs, ys, real_solution, title="Аналитическое решение")
p2 = heatmap(xs, ys, predicted_solution, title="Предсказанное решение")
p3 = heatmap(xs, ys, abs.(real_solution .- predicted_solution), title="Ошибка")

plot(p1, p2, p3, layout=(1, 3))
```

#### Статистический анализ

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Вычисление корреляций
correlation = np.corrcoef(analytic.flatten(), predicted.flatten())[0, 1]

# Статистические тесты
stat, p_value = stats.pearsonr(analytic.flatten(), predicted.flatten())

print(f"Корреляция: {correlation:.4f}")
print(f"P-value: {p_value:.4e}")
```

## Настройка параметров

### Гиперпараметры нейронной сети

Основные параметры для настройки в файле `inverse_problem_eeg.ipynb`:

```julia
# Размерность входа и выхода
input_ = 4  # x, y, z, t
output_ = 7 # φ, Ax, Ay, Az, Px, Py, Pz

# Архитектура сети
n = 32      # число нейронов в скрытых слоях
depth = 3   # число скрытых слоев

# Параметры обучения
lambda = 10 # вес дополнительной потери
maxiters = 3000  # максимальное число итераций
learning_rate = 0.001  # скорость обучения
```

### Параметры физической модели

```julia
# Физические постоянные
const c = 2.99792458e10    # Скорость света
const ε = 1.0              # Диэлектрическая проницаемость
const μ = 1.0              # Магнитная проницаемость

# Область решения
x_min, x_max = -10.0, 10.0
y_min, y_max = -10.0, 10.0
z_min, z_max = -10.0, 10.0
t_min, t_max = 0.0, 1.0
```

## PML граничные условия

### Введение

PML (Perfectly Matched Layer) — это метод поглощающих граничных условий, который используется для:

- Предотвращения отражений волн от границ расчётной области
- Обеспечения численной устойчивости решения
- Корректного моделирования распространения электромагнитных волн

### Быстрое включение

```julia
using InverseNpde

# Использовать PML с параметрами по умолчанию
pml_config = default_pml_config()

# Создать PDE систему с PML
pde_system = create_pde_system(
    domain, 
    physics_params; 
    pml_config = pml_config
)
```

### Настройка PML

Подробная документация по всем параметрам PML доступна в файле [PML.md](PML.md).

#### Основные параметры

| Параметр | Описание | Рекомендуемое значение |
|----------|----------|------------------------|
| `enabled` | Включить/выключить PML | `true` |
| `thickness` | Толщина слоя PML (м) | 0.1-0.2 |
| `sigma_max` | Максимальная проводимость | 0.5-2.0 |
| `kappa_max` | Максимальное растяжение κ | 3.0-10.0 |
| `m` | Порядок полинома | 3-4 |

#### Примеры конфигураций

```julia
# Консервативная (низкие частоты)
pml_config = PMLConfig(
    enabled = true,
    thickness = 0.1,
    sigma_max = 0.5,
    kappa_max = 3.0,
    alpha_min = 0.1,
    alpha_max = 0.5,
    m = 3,
    direction = :xyz
)

# Агрессивная (высокие частоты)
pml_config = PMLConfig(
    enabled = true,
    thickness = 0.2,
    sigma_max = 2.0,
    kappa_max = 10.0,
    alpha_min = 0.05,
    alpha_max = 0.8,
    m = 4,
    direction = :xyz
)
```

### Отключение PML

```julia
# Полное отключение PML
pml_config = PMLConfig(enabled = false)
```

### Диагностика проблем

**Признаки недостаточного PML:**
- Отражения на границах
- Нестабильность решения
- Паразитные осцилляции

**Признаки слишком агрессивного PML:**
- Затухание в области интереса
- Нарушение калибровки Лоренца
- Численные артефакты

### Оптимизация производительности

#### Использование GPU

```julia
# Проверка доступности CUDA
if CUDA.functional()
    println("CUDA доступна, используем GPU")
    ps = ps |> gpu
    gpud = gpu_device()
else
    println("CUDA недоступна, используем CPU")
    cpud = cpu_device()
end
```

#### Настройка памяти

```julia
# Для больших задач увеличьте лимит памяти
CUDA.allowscalar(false)

# Используйте смешанную точность для экономии памяти
ps = ps .|> Float32
```

## Интеграция Julia-Python

### Вызов Julia из Python

```python
from juliacall import Main as jl

# Загрузка Julia модуля
jl.eval('include("path/to/julia_script.jl")')

# Вызов функции
result = jl.julia_function(python_data)
```

### Вызов Python из Julia

```julia
using PythonCall

# Импорт Python библиотеки
np = pyimport("numpy")
pd = pyimport("pandas")

# Работа с данными
py_data = np.array([1, 2, 3, 4, 5])
jl_data = pyconvert(Array{Float32}, py_data)

# Обратный вызов
py_result = pycall(py_function, Py, jl_data)
```

### Обмен данными

#### Форматы обмена

1. **NumPy массивы** - для численных данных
2. **Pandas DataFrame** - для табличных данных
3. **xarray DataArray** - для многомерных научных данных
4. **NetCDF** - для больших массивов данных

#### Пример обмена данными

```julia
# Julia: Подготовка данных
julia_data = rand(100, 61, 500)  # события, каналы, время

# Передача в Python
py_data = pyconvert(PyArray, julia_data)

# Python: Обработка
processed_data = process_eeg_data(py_data)

# Возврат в Julia
jl_processed = pyconvert(Array{Float32}, processed_data)
```

## Часто задаваемые вопросы

### Q: Как изменить архитектуру нейронной сети?

**A:** Измените параметры в секции "Нейросеть" ноутбука:

```julia
# Изменение числа слоев
chain = Chain(
    Dense(input_, n, σ),
    Dense(n, n, σ),
    Dense(n, n, σ),
    Dense(n, n, σ),  # Дополнительный слой
    Dense(n, output_)
)

# Изменение активационной функции
chain = Chain(
    Dense(input_, n, relu),  # Вместо σ
    Dense(n, n, tanh),
    Dense(n, output_)
)
```

### Q: Как добавить собственные данные EEG?

**A:** Подготовьте данные в формате, совместимом с MNE-Python:

```python
# Создание объекта MNE
info = mne.create_info(ch_names, sfreq, ch_types)
raw = mne.io.RawArray(data, info)

# Сохранение в стандартном формате
raw.save('my_data.fif')
```

Затем загрузите данные в ноутбуке `brunton_uw_bio.ipynb`.

### Q: Как ускорить обучение модели?

**A:** Используйте следующие стратегии:

1. **GPU ускорение**:
```julia
if CUDA.functional()
    ps = ps |> gpu
end
```

2. **Смешанная точность**:
```julia
ps = ps .|> Float32  # Вместо Float32
```

3. **Батчинг данных**:
```julia
strategy = QuasiRandomTraining(8192)  # Увеличьте размер батча
```

4. **Предобученные модели**:
```julia
# Загрузите предобученные веса
ps = load_pretrained_weights("path/to/model.bson")
```

### Q: Как сохранить промежуточные результаты?

**A:** Используйте функции сохранения:

```julia
# Сохранение модели
using BSON
bson("model_weights.bson", ps=res.u)

# Сохранение результатов
using JLD2
@save "experiment_results.jld2" res discretization

# Сохранение графиков
savefig(plot_results, "results.png")
```

### Q: Как настроить логирование?

**A:** Настройте TensorBoardLogger:

```julia
using TensorBoardLogger
logger = TBLogger("logs/inverse_npde_exp")

# Логирование метрик
log_value(logger, "Loss/Total", loss; step=iteration)
log_value(logger, "Validation/Accuracy", accuracy; step=iteration)
```

Запустите TensorBoard для мониторинга:
```bash
tensorboard --logdir=logs/
```

### Q: Как обработать большие датасеты?

**A:** Используйте следующие подходы:

1. **Потоковая обработка**:
```python
def process_large_dataset(file_list, batch_size=100):
    for i in range(0, len(file_list), batch_size):
        batch = file_list[i:i+batch_size]
        yield process_batch(batch)
```

2. **Параллельная обработка**:
```julia
using Distributed
addprocs(4)
@everywhere using NeuralPDE

results = pmap(process_subject, subjects)
```

3. **Внешние хранилища**:
```python
import dask.array as da
data = da.from_array('large_file.h5', chunks=(1000, 61, 500))
```

### Q: Как интерпретировать результаты визуально?

**A:** Используйте различные типы визуализации:

1. **Тепловые карты**:
```julia
heatmap(xs, ys, solution, title="Распределение потенциала")
```

2. **3D визуализация**:
```julia
using PlotlyJS
plot([
    mesh3d(x=xs, y=ys, z=zs, intensity=values)
])
```

3. **Временные ряды**:
```julia
plot(times, time_series, title="Временная динамика")
```

---

## Дополнительные ресурсы

- [Документация NeuralPDE.jl](https://neuralpde.sciml.ai/stable/)
- [MNE-Python Tutorial](https://mne.tools/stable/auto_tutorials/index.html)
- [Julia for Data Science](https://juliadatascience.io/)
- [CUDA.jl Documentation](https://cuda.juliagpu.org/stable/)

---

**Последнее обновление**: 2025-12-19  
**Версия руководства**: 1.0.0