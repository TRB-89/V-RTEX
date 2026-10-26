cmake_minimum_required(VERSION 3.20)
project(VORTEX VERSION 3.0.0 LANGUAGES CXX)

# ==============================================================================
# Global C++20 Standard Configuration
# ==============================================================================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Common compiler warnings & optimization flags
if (MSVC)
    add_compile_options(/W4 /permissive- /O2)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -O3)
endif()

# ==============================================================================
# Core Library Target: vortex_core
# ==============================================================================
add_library(vortex_core STATIC
    src/simplex.cpp
    src/revised_simplex.cpp
)

target_include_directories(vortex_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(vortex_core PUBLIC cxx_std_20)

# ==============================================================================
# CLI Demo Executable: vortex_cli
# ==============================================================================
add_executable(vortex_cli src/main.cpp)
target_link_libraries(vortex_cli PRIVATE vortex_core)

# ==============================================================================
# Automated Test Suites
# ==============================================================================
enable_testing()

# Matrix Test Suite (Phase 1)
add_executable(vortex_tests tests/test_matrix.cpp)
target_link_libraries(vortex_tests PRIVATE vortex_core)
add_test(NAME MatrixUnitTests COMMAND vortex_tests)

# Simplex LP Solver Test Suite (Phase 2 Standard)
add_executable(vortex_simplex_tests tests/test_simplex.cpp)
target_link_libraries(vortex_simplex_tests PRIVATE vortex_core)
add_test(NAME SimplexUnitTests COMMAND vortex_simplex_tests)

# General Two-Phase Simplex Test Suite (Phase 3)
add_executable(vortex_two_phase_tests tests/test_two_phase_simplex.cpp)
target_link_libraries(vortex_two_phase_tests PRIVATE vortex_core)
add_test(NAME TwoPhaseSimplexUnitTests COMMAND vortex_two_phase_tests)

# LU Decomposition Test Suite (Phase 4)
add_executable(vortex_lu_tests tests/test_lu_decomposition.cpp)
target_link_libraries(vortex_lu_tests PRIVATE vortex_core)
add_test(NAME LUDecompositionUnitTests COMMAND vortex_lu_tests)

# Revised Simplex LP Solver Test Suite (Phase 4)
add_executable(vortex_revised_simplex_tests tests/test_revised_simplex.cpp)
target_link_libraries(vortex_revised_simplex_tests PRIVATE vortex_core)
add_test(NAME RevisedSimplexUnitTests COMMAND vortex_revised_simplex_tests)

# ==============================================================================
# Benchmarks Executable: vortex_benchmarks
# ==============================================================================
add_executable(vortex_benchmarks benchmarks/benchmark_matrix.cpp)
target_link_libraries(vortex_benchmarks PRIVATE vortex_core)

