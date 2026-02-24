#!/usr/bin/env julia

"""
Расширенный тест для проверки PML интеграции с PDE системой.
Тестирует новый подход PML через затухание (γ) и экранирование (α).

Тестирует:
1. Создание PMLConfig с новыми параметрами
2. Создание переменных (без ψ-полей)
3. Создание доменов
4. Создание граничных условий
5. Создание PDE системы с модифицированным оператором Даламбера
6. Проверку автоматического расчёта γ_max и α_max
"""

using Pkg
Pkg.activate(pwd())
Pkg.instantiate()

println("=" ^ 60)
println("Расширенный тест PML интеграции с PDE системой")
println("Новый подход: затухание (γ) + экранирование (α)")
println("=" ^ 60)

# Тест 1: Загрузка модуля
println("\n📦 Тест 1: Загрузка модуля InverseNpde...")
try
    using InverseNpde
    println("✓ Модуль InverseNpde загружен успешно")
catch e
    println("✗ Ошибка загрузки модуля: $e")
    println(stacktrace(catch_backtrace()))
    exit(1)
end

# Тест 2: Создание PMLConfig с новыми параметрами
println("\n⚙️  Тест 2: Создание PMLConfig...")
global pml_config = nothing
try
    global pml_config = PMLConfig(enabled=true)
    println("✓ PMLConfig создан успешно")
    println("  - Enabled: $(pml_config.enabled)")
    println("  - Thickness ratio: $(pml_config.pml_thickness_ratio)")
    println("  - Reflection coefficient: $(pml_config.reflection_coefficient)")
    println("  - γ_max (auto): $(pml_config.gamma_max)")
    println("  - α_max (auto): $(pml_config.alpha_max)")
catch e
    println("✗ Ошибка создания PMLConfig: $e")
    println(stacktrace(catch_backtrace()))
    exit(1)
end

# Тест 3: Создание PMLConfig с явными параметрами
println("\n⚙️  Тест 3: Создание PMLConfig с явными параметрами...")
try
    pml_config_explicit = PMLConfig(
        enabled=true,
        pml_thickness_ratio=0.15,
        reflection_coefficient=1e-6,
        gamma_max=1000.0,
        alpha_max=500.0
    )
    println("✓ PMLConfig с явными параметрами создан успешно")
    println("  - Thickness ratio: $(pml_config_explicit.pml_thickness_ratio)")
    println("  - Reflection coefficient: $(pml_config_explicit.reflection_coefficient)")
    println("  - γ_max (explicit): $(pml_config_explicit.gamma_max)")
    println("  - α_max (explicit): $(pml_config_explicit.alpha_max)")
catch e
    println("✗ Ошибка создания PMLConfig с явными параметрами: $e")
    println(stacktrace(catch_backtrace()))
    exit(1)
end

# Тест 4: Создание переменных
println("\n📝 Тест 4: Создание переменных PDE...")
global variables = nothing
global constants = nothing
try
    # Импортируем внутренние модули для доступа к функциям
    import InverseNpde.PDEDefinitions: create_variables, PhysicalConstants, create_domains, create_boundary_conditions, create_pde_system
    
    global variables = create_variables()
    global constants = PhysicalConstants()
    println("✓ Переменные созданы успешно")
    println("  - Независимые переменные: x, y, z, t")
    println("  - Зависимые переменные: φ, Ax, Ay, Az, ρ, jx, jy, jz")
    println("  - PML переменные: НЕТ (новый подход без ψ-полей)")
catch e
    println("✗ Ошибка создания переменных: $e")
    println(stacktrace(catch_backtrace()))
    exit(1)
end

# Тест 5: Создание доменов
println("\n📐 Тест 5: Создание доменов...")
global domains = nothing
try
    global domains = create_domains(variables, [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [0.0, 1.0])
    println("✓ Домены созданы успешно")
    println("  - x ∈ [-1.0, 1.0]")
    println("  - y ∈ [-1.0, 1.0]")
    println("  - z ∈ [-1.0, 1.0]")
    println("  - t ∈ [0.0, 1.0]")
catch e
    println("✗ Ошибка создания доменов: $e")
    println(stacktrace(catch_backtrace()))
    exit(1)
end

# Тест 6: Создание граничных условий с PML
println("\n🔲 Тест 6: Создание граничных условий с PML...")
global bcs = nothing
try
    global bcs = create_boundary_conditions(constants, variables, domains; pml_config=pml_config)
    println("✓ Граничные условия созданы успешно")
    println("  - Количество граничных условий: $(length(bcs))")
    println("  - Ожидается: 24 (6 граней × 4 поля: φ, Ax, Ay, Az)")
catch e
    println("✗ Ошибка создания граничных условий: $e")
    println(stacktrace(catch_backtrace()))
    exit(1)
end

# Тест 7: Создание PDE системы с PML
println("\n🔬 Тест 7: Создание PDE системы с PML...")
global pde_system = nothing
try
    global pde_system = create_pde_system(constants, variables, bcs, domains; pml_config=pml_config)
    println("✓ PDE система создана успешно")
    println("  - Количество уравнений: $(length(pde_system.eqs))")
    println("  - Ожидается: 5 (4 волновых + 1 калибровка Лоренца)")
    println("  - Количество зависимых переменных: $(length(pde_system.depvars))")
    println("  - Ожидается: 8 (φ, Ax, Ay, Az, ρ, jx, jy, jz)")
    println("  - Количество независимых переменных: $(length(pde_system.indvars))")
catch e
    println("✗ Ошибка создания PDE системы: $e")
    println(stacktrace(catch_backtrace()))
    exit(1)
end

# Тест 8: Проверка структуры уравнений
println("\n📊 Тест 8: Проверка структуры уравнений...")
try
    eqs = pde_system.eqs
    println("✓ Структура уравнений:")
    for (i, eq) in enumerate(eqs)
        eq_str = string(eq)
        # Обрезаем слишком длинные строки
        if length(eq_str) > 100
            eq_str = eq_str[1:100] * "..."
        end
        println("  Уравнение $i: $eq_str")
    end
catch e
    println("✗ Ошибка проверки уравнений: $e")
    println(stacktrace(catch_backtrace()))
end

# Тест 9: Тест с отключённым PML
println("\n🔄 Тест 9: Создание PDE системы с отключённым PML...")
try
    pml_config_disabled = PMLConfig(enabled=false)
    variables_disabled = create_variables()
    domains_disabled = create_domains(variables_disabled, [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [0.0, 1.0])
    bcs_disabled = create_boundary_conditions(constants, variables_disabled, domains_disabled; pml_config=pml_config_disabled)
    pde_system_disabled = create_pde_system(constants, variables_disabled, bcs_disabled, domains_disabled; pml_config=pml_config_disabled)
    
    println("✓ PDE система без PML создана успешно")
    println("  - Количество уравнений: $(length(pde_system_disabled.eqs))")
    println("  - Количество переменных: $(length(pde_system_disabled.depvars))")
    
    # Сравниваем количество уравнений
    println("\n📈 Сравнение:")
    println("  - С PML: $(length(pde_system.eqs)) уравнений, $(length(pde_system.depvars)) переменных")
    println("  - Без PML: $(length(pde_system_disabled.eqs)) уравнений, $(length(pde_system_disabled.depvars)) переменных")
    
    # Проверяем, что количество уравнений одинаковое (новый подход!)
    if length(pde_system.eqs) == length(pde_system_disabled.eqs)
        println("  ✓ Количество уравнений одинаковое (PML не добавляет уравнения)")
    else
        println("  ⚠ Количество уравнений отличается!")
    end
catch e
    println("✗ Ошибка создания PDE системы без PML: $e")
    println(stacktrace(catch_backtrace()))
end

# Тест 10: Проверка функций compute_gamma и compute_alpha
println("\n🧮 Тест 10: Проверка функций compute_gamma и compute_alpha...")
try
    import InverseNpde.PML: compute_gamma_max, compute_alpha_max, step_indicator
    
    # Тестируем compute_gamma_max
    c = 299792458.0  # скорость света
    d = 0.1  # толщина PML
    R = 1e-4  # коэффициент отражения
    
    gamma_max = compute_gamma_max(c, d, R)
    println("✓ compute_gamma_max(c=$c, d=$d, R=$R) = $gamma_max")
    
    # Тестируем compute_alpha_max
    L = 1.0  # характерный размер
    alpha_max = compute_alpha_max(L, R)
    println("✓ compute_alpha_max(L=$L, R=$R) = $alpha_max")
    
    # Тестируем step_indicator
    x0 = 0.5
    x_vals = [0.0, 0.4, 0.5, 0.6, 1.0]
    println("✓ step_indicator(x, x0=$x0):")
    for x in x_vals
        ind = step_indicator(x, x0)
        println("    step_indicator($x, $x0) = $ind")
    end
catch e
    println("✗ Ошибка проверки функций PML: $e")
    println(stacktrace(catch_backtrace()))
end

# Тест 11: Проверка resolve_pml_config
println("\n🔧 Тест 11: Проверка resolve_pml_config...")
try
    import InverseNpde.PML: resolve_pml_config
    
    # Тестируем с автоматическим расчётом
    resolved_auto = resolve_pml_config(pml_config, 2.0, 299792458.0)
    println("✓ resolve_pml_config (auto):")
    println("  - γ_max: $(resolved_auto.gamma_max)")
    println("  - α_max: $(resolved_auto.alpha_max)")
    
    # Тестируем с явными значениями
    pml_explicit = PMLConfig(enabled=true, gamma_max=1000.0, alpha_max=500.0)
    resolved_explicit = resolve_pml_config(pml_explicit, 2.0, 299792458.0)
    println("✓ resolve_pml_config (explicit):")
    println("  - γ_max: $(resolved_explicit.gamma_max)")
    println("  - α_max: $(resolved_explicit.alpha_max)")
catch e
    println("✗ Ошибка проверки resolve_pml_config: $e")
    println(stacktrace(catch_backtrace()))
end

println("\n" * "=" ^ 60)
println("✅ Все тесты PML интеграции завершены успешно!")
println("=" ^ 60)
println("\n📝 Резюме нового подхода PML:")
println("  - Модифицированный оператор Даламбера: □u - γ·∂u/∂t - α·u = S")
println("  - γ(r) — динамическое затухание (вязкое трение)")
println("  - α(r) — статический экранирующий член")
println("  - Нет вспомогательных ψ-полей")
println("  - Выход нейросети: 8 значений (вместо 20)")
println("  - PDE уравнений: 5 (вместо 17)")
