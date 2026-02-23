#!/usr/bin/env julia

using Pkg
Pkg.activate(pwd())
Pkg.instantiate()

using InverseNpde

# Create a simple test to verify the PML functionality
println("Testing PML implementation...")

# Test 1: Basic PML configuration
println("\n1. Testing PMLConfig creation:")
try
    pml_config = PMLConfig()
    println("✓ PMLConfig created successfully")
    println("  - Enabled: $(pml_config.enabled)")
    println("  - Thickness: $(pml_config.pml_thickness_ratio)")
    println("  - Polynomial order: $(pml_config.polynomial_order)")
    println("  - σ_max: $(pml_config.sigma_max)")
    println("  - κ_max: $(pml_config.kappa_max)")
    println("  - α_max: $(pml_config.alpha_max)")
catch e
    println("✗ Error creating PMLConfig: $e")
    println(stacktrace(catch_backtrace()))
end

# Test 2: Disable PML
println("\n2. Testing disabled PML:")
try
    pml_config = PMLConfig(enabled=false)
    println("✓ Disabled PMLConfig created successfully")
    println("  - Enabled: $(pml_config.enabled)")
catch e
    println("✗ Error creating disabled PMLConfig: $e")
    println(stacktrace(catch_backtrace()))
end

# Test 3: Custom PML configuration
println("\n3. Testing custom PML parameters:")
try
    pml_config = PMLConfig(
        enabled=true,
        pml_thickness_ratio=0.15,
        polynomial_order=4,
        sigma_max=0.8,
        kappa_max=8.0,
        alpha_max=0.15
    )
    println("✓ Custom PMLConfig created successfully")
    println("  - Enabled: $(pml_config.enabled)")
    println("  - Thickness: $(pml_config.pml_thickness_ratio)")
    println("  - Polynomial order: $(pml_config.polynomial_order)")
    println("  - σ_max: $(pml_config.sigma_max)")
    println("  - κ_max: $(pml_config.kappa_max)")
    println("  - α_max: $(pml_config.alpha_max)")
catch e
    println("✗ Error creating custom PMLConfig: $e")
    println(stacktrace(catch_backtrace()))
end

# Test 4: Test sigma_max=0 for automatic calculation
println("\n4. Testing automatic sigma_max calculation:")
try
    pml_config = PMLConfig(sigma_max=0.0)
    println("✓ PMLConfig with sigma_max=0 created successfully")
    println("  - σ_max: $(pml_config.sigma_max) (will be calculated automatically)")
catch e
    println("✗ Error: $e")
    println(stacktrace(catch_backtrace()))
end

# Test 5: Test default_pml_config
println("\n5. Testing default_pml_config function:")
try
    default_config = default_pml_config()
    println("✓ default_pml_config() works")
    println("  - Enabled: $(default_config.enabled)")
    println("  - Thickness: $(default_config.pml_thickness_ratio)")
catch e
    println("✗ Error: $e")
    println(stacktrace(catch_backtrace()))
end

println("\n✅ All PML configuration tests completed!")
