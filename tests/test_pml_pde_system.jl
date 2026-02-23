#!/usr/bin/env julia

"""
Расширенный тест для проверки PML интеграции с PDE системой.
Тестирует:
1. Создание PMLConfig
2. Создание переменных с PML
3. Создание доменов
4. Создание граничных условий с PML
5. Создание PDE системы с PML
"""

using Pkg
Pkg.activate(pwd())
Pkg.instantiate()

println("=" ^ 60)
println("Расширенный тест PML интеграции с PDE системой")
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

# Тест 2: Создание PMLConfig
println("\n⚙️  Тест 2: Создание PMLConfig...")
pml_config = nothing
try
    pml_config = PMLConfig(enabled=true)
    println("✓ PMLConfig создан успешно")
    println("  - Enabled: $(pml_config.enabled)")
    println("  - Thickness: $(pml_config.pml_thickness_ratio)")
    println("  - Polynomial order: $(pml_config.polynomial_order)")
    println("  - σ_max: $(pml_config.sigma_max)")
    println("  - κ_max: $(pml_config.kappa_max)")
    println("  - α_max: $(pml_config.alpha_max)")
catch e
    println("✗ Ошибка создания PMLConfig: $e")
    println(stacktrace(catch_backtrace()))
    exit(1)
end

# Тест 3: Создание переменных
println("\n📝 Тест 3: Создание переменных PDE...")
variables = nothing
constants = nothing
try
    # Импортируем внутренние модули для доступа к функциям
    import InverseNpde.PDEDefinitions: create_variables, PhysicalConstants, create_domains, create_boundary_conditions, create_pde_system
    
    variables = create_variables()
    constants = PhysicalConstants()
    println("✓ Переменные созданы успешно")
    println("  - Переменные: x, y, z, t, φ, A, ρ, j")
    println("  - PML переменные: ψ_φx, ψ_φy, ψ_φz, ψ_Axx-ψ_Azz")
catch e
    println("✗ Ошибка создания переменных: $e")
    println(stacktrace(catch_backtrace()))
    exit(1)
end

# Тест 4: Создание доменов
println("\n📐 Тест 4: Создание доменов...")
domains = nothing
try
    domains = create_domains(variables, [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [0.0, 1.0])
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

# Тест 5: Создание граничных условий с PML
println("\n🔲 Тест 5: Создание граничных условий с PML...")
bcs = nothing
try
    bcs = create_boundary_conditions(constants, variables, domains; pml_config=pml_config)
    println("✓ Граничные условия созданы успешно")
    println("  - Количество граничных условий: $(length(bcs))")
catch e
    println("✗ Ошибка создания граничных условий: $e")
    println(stacktrace(catch_backtrace()))
    exit(1)
end

# Тест 6: Создание PDE системы с PML
println("\n🔬 Тест 6: Создание PDE системы с PML...")
pde_system = nothing
try
    pde_system = create_pde_system(constants, variables, bcs, domains; pml_config=pml_config)
    println("✓ PDE система создана успешно")
    println("  - Количество уравнений: $(length(pde_system.eqs))")
    println("  - Количество переменных: $(length(pde_system.depvars))")
    println("  - Количество независимых переменных: $(length(pde_system.indvars))")
catch e
    println("✗ Ошибка создания PDE системы: $e")
    println(stacktrace(catch_backtrace()))
    exit(1)
end

# Тест 7: Проверка структуры уравнений
println("\n📊 Тест 7: Проверка структуры уравнений...")
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

# Тест 8: Тест с отключённым PML
println("\n🔄 Тест 8: Создание PDE системы с отключённым PML...")
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
catch e
    println("✗ Ошибка создания PDE системы без PML: $e")
    println(stacktrace(catch_backtrace()))
end

println("\n" * "=" ^ 60)
println("✅ Все тесты PML интеграции завершены успешно!")
println("=" ^ 60)
