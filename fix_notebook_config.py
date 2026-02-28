#!/usr/bin/env python3
"""
Исправляет размещение параметров three-stage optimization:
- Перемещает параметры из LossFunctionConfig в OptimizationConfig
"""

import json
import sys

# Читаем notebook
with open("examples/eeg_realistic_demo_julia.ipynb", "r") as f:
    nb = json.load(f)

# Параметры three-stage optimization для перемещения
THREE_STAGE_PARAMS = [
    "three_stage_optimization",
    "adam1_lr",
    "adam2_lr",
    "adam1_ratio",
    "lbfgs_ratio",
    "adam2_ratio",
    "scheduler_type",
    "scheduler_warmup",
]

# Находим и обновляем ячейки
for cell in nb["cells"]:
    if cell.get("cell_type") != "code":
        continue

    source_lines = cell.get("source", [])
    source = "".join(source_lines)

    # Обновляем OptimizationConfig
    if "opt_config = OptimizationConfig" in source:
        print("Обновляем OptimizationConfig...")

        # Находим строку с log_directory и добавляем параметры после неё
        new_lines = []
        inserted = False

        for i, line in enumerate(source_lines):
            new_lines.append(line)

            # После строки с log_directory вставляем новые параметры
            if "log_directory" in line and not inserted:
                new_lines.append("\n")
                new_lines.append("    # === НОВОЕ: Three-stage optimization ===\n")
                new_lines.append(
                    "    three_stage_optimization = true,  # Включить трёхэтапную оптимизацию\n"
                )
                new_lines.append(
                    "    adam1_lr = 0.01,               # LR для первого Adam этапа\n"
                )
                new_lines.append(
                    "    adam2_lr = 0.001,              # LR для второго Adam этапа\n"
                )
                new_lines.append(
                    "    adam1_ratio = 0.2,             # 20% итераций для первого Adam\n"
                )
                new_lines.append(
                    "    lbfgs_ratio = 0.1,             # 10% итераций для LBFGS\n"
                )
                new_lines.append(
                    "    adam2_ratio = 0.7,             # 70% итераций для второго Adam\n"
                )
                new_lines.append(
                    "    scheduler_type = :cosine,       # Тип scheduler: :cosine, :warmup_cosine, :step\n"
                )
                new_lines.append(
                    "    scheduler_warmup = 100          # Warmup итерации\n"
                )
                inserted = True

        cell["source"] = new_lines
        print("  ✓ Добавлены параметры three-stage optimization")

    # Обновляем LossFunctionConfig - удаляем three-stage параметры
    if "loss_config = LossFunctionConfig" in source:
        print("Обновляем LossFunctionConfig...")

        new_lines = []
        skip_mode = False
        skip_count = 0

        for line in source_lines:
            # Пропускаем комментарий о three-stage
            if "=== НОВОЕ: Three-stage optimization ===" in line:
                skip_mode = True
                skip_count = len(THREE_STAGE_PARAMS) + 1  # +1 для комментария
                continue

            # Пропускаем строки с three-stage параметрами
            if skip_mode:
                if any(param in line for param in THREE_STAGE_PARAMS):
                    continue
                # Если это не параметр three-stage, выходим из режима пропуска
                skip_mode = False

            new_lines.append(line)

        cell["source"] = new_lines
        print("  ✓ Удалены параметры three-stage optimization")

# Сохраняем исправленный notebook
with open("examples/eeg_realistic_demo_julia.ipynb", "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("\n✅ Notebook успешно обновлён!")
print("\nИзменения:")
print("  1. Параметры three-stage optimization добавлены в OptimizationConfig")
print("  2. Параметры three-stage optimization удалены из LossFunctionConfig")
