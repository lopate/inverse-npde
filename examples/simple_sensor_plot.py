# Простой график изменения потенциала на определенном датчике
import numpy as np
import matplotlib.pyplot as plt

# Выбираем центральный датчик для анализа
central_sensor_idx = n_sensors // 2

# Извлекаем данные из исходного датасета (реальные значения)
# Ищем точки, соответствующие выбранному датчику
sensor_x = np.mean([p[0] for p in measured_points])  # Средняя x координата датчика
sensor_y = np.mean([p[1] for p in measured_points])  # Средняя y координата датчика
sensor_z = np.mean([p[2] for p in measured_points])  # Средняя z координата датчика

# Группируем данные по времени для этого датчика
time_real = []
potential_real = []

# Проходим по всем точкам и ищем те, что близки к выбранному датчику
for point in measured_points:
    x, y, z, t, phi = point
    # Проверяем, близка ли точка к нашему датчику (допуск 1.0)
    if abs(x - sensor_x) < 1.0 and abs(y - sensor_y) < 1.0 and abs(z - sensor_z) < 1.0:
        time_real.append(t)
        potential_real.append(phi)

# Сортируем по времени
sorted_indices = np.argsort(time_real)
time_real = np.array(time_real)[sorted_indices]
potential_real = np.array(potential_real)[sorted_indices]
