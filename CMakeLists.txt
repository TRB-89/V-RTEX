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
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/W4;/permissive-;/O2>")
else()
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic;-O3>")
endif()

# ==============================================================================
# Build Options & Feature Toggles
# ==============================================================================
option(VORTEX_ENABLE_CUDA "Enable CUDA GPU acceleration" OFF)
option(VORTEX_BUILD_TESTS "Build automated unit and integration tests" ON)
option(VORTEX_BUILD_BENCHMARKS "Build benchmark executables" ON)

# Respect standard CMake BUILD_TESTING if provided
if (DEFINED BUILD_TESTING AND NOT BUILD_TESTING)
    set(VORTEX_BUILD_TESTS OFF)
endif()

if (VORTEX_ENABLE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)
    set(CMAKE_CUDA_ARCHITECTURES 86)

    find_package(CUDAToolkit REQUIRED)

    # CUDA Backend Library: vortex_cuda
    add_library(vortex_cuda STATIC
        src/cuda_backend.cu
    )

    target_include_directories(vortex_cuda PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    )

    target_compile_features(vortex_cuda PUBLIC cxx_std_20)
    target_compile_definitions(vortex_cuda PUBLIC VORTEX_ENABLE_CUDA)

    set_target_properties(vortex_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        CUDA_ARCHITECTURES 86
    )

    target_link_libraries(vortex_cuda PUBLIC
        CUDA::cublas
        CUDA::cusparse
    )
endif()

# ==============================================================================
# Core Library Target: vortex_core
# ==============================================================================
add_library(vortex_core STATIC
    src/simplex.cpp
    src/revised_simplex.cpp
    src/sparse_revised_simplex.cpp
    src/milp.cpp
    src/qp.cpp
)

target_include_directories(vortex_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(vortex_core PUBLIC cxx_std_20)

if (VORTEX_ENABLE_CUDA)
    target_link_libraries(vortex_core PUBLIC vortex_cuda)
    target_compile_definitions(vortex_core PUBLIC VORTEX_ENABLE_CUDA)
endif()

# ==============================================================================
# CLI Demo Executable: vortex_cli
# ==============================================================================
add_executable(vortex_cli src/main.cpp)
target_link_libraries(vortex_cli PRIVATE vortex_core)

# ==============================================================================
# Machine-Readable JSON Solver Runner: vortex_solver_runner (Phase 9)
# ==============================================================================
add_executable(vortex_solver_runner src/solver_runner.cpp)
target_link_libraries(vortex_solver_runner PRIVATE vortex_core)
if (VORTEX_ENABLE_CUDA)
    target_link_libraries(vortex_solver_runner PRIVATE vortex_cuda)
endif()

# ==============================================================================
# Automated Test Suites
# ==============================================================================
if (VORTEX_BUILD_TESTS)
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

    # Sparse Matrix Engine Test Suite (Phase 5)
    add_executable(vortex_sparse_matrix_tests tests/test_sparse_matrix.cpp)
    target_link_libraries(vortex_sparse_matrix_tests PRIVATE vortex_core)
    add_test(NAME SparseMatrixUnitTests COMMAND vortex_sparse_matrix_tests)

    # Sparse Revised Simplex Test Suite (Phase 5)
    add_executable(vortex_sparse_revised_simplex_tests tests/test_sparse_revised_simplex.cpp)
    target_link_libraries(vortex_sparse_revised_simplex_tests PRIVATE vortex_core)
    add_test(NAME SparseRevisedSimplexUnitTests COMMAND vortex_sparse_revised_simplex_tests)

    # Numerical Correctness and Memory Audit Suite (Phase 5.1)
    add_executable(vortex_numerical_audit_tests tests/test_numerical_audit.cpp)
    target_link_libraries(vortex_numerical_audit_tests PRIVATE vortex_core)
    add_test(NAME NumericalAuditTests COMMAND vortex_numerical_audit_tests)

    # Mixed-Integer Linear Programming Test Suite (Phase 6)
    add_executable(vortex_milp_tests tests/test_milp.cpp)
    target_link_libraries(vortex_milp_tests PRIVATE vortex_core)
    add_test(NAME MILPUnitTests COMMAND vortex_milp_tests)

    # Convex Quadratic Programming Test Suite (Phase 7)
    add_executable(vortex_qp_tests tests/test_qp.cpp)
    target_link_libraries(vortex_qp_tests PRIVATE vortex_core)
    add_test(NAME QPUnitTests COMMAND vortex_qp_tests)

    # CUDA GPU Acceleration Test Suites (Phase 8A, 8B & 8C)
    if (VORTEX_ENABLE_CUDA)
        add_executable(vortex_cuda_tests tests/test_cuda.cpp)
        target_link_libraries(vortex_cuda_tests PRIVATE vortex_cuda vortex_core)
        add_test(NAME CudaBackendUnitTests COMMAND vortex_cuda_tests)

        add_executable(vortex_cuda_matrix_tests tests/test_cuda_matrix.cpp)
        target_link_libraries(vortex_cuda_matrix_tests PRIVATE vortex_cuda vortex_core)
        add_test(NAME CudaMatrixOperationsTests COMMAND vortex_cuda_matrix_tests)

        # GPU Numerical Accuracy Validation Suite (Phase 8C)
        add_executable(vortex_cuda_accuracy_tests tests/test_cuda_accuracy.cpp)
        target_link_libraries(vortex_cuda_accuracy_tests PRIVATE vortex_cuda vortex_core)
        add_test(NAME GpuNumericalAccuracyTests COMMAND vortex_cuda_accuracy_tests)
    endif()

    # Hybrid CPU/GPU Solver Integration Test Suite (Phase 8D)
    add_executable(vortex_hybrid_solver_tests tests/test_hybrid_solver.cpp)
    target_link_libraries(vortex_hybrid_solver_tests PRIVATE vortex_core)
    add_test(NAME HybridSolverIntegrationTests COMMAND vortex_hybrid_solver_tests)
endif()

# ==============================================================================
# Benchmarks Executables
# ==============================================================================
if (VORTEX_BUILD_BENCHMARKS)
    add_executable(vortex_benchmarks benchmarks/benchmark_matrix.cpp)
    target_link_libraries(vortex_benchmarks PRIVATE vortex_core)

    add_executable(vortex_sparse_benchmarks benchmarks/benchmark_sparse.cpp)
    target_link_libraries(vortex_sparse_benchmarks PRIVATE vortex_core)

    # GPU Performance Benchmarking Suite (Phase 8E)
    if (VORTEX_ENABLE_CUDA)
        add_executable(vortex_gpu_benchmarks benchmarks/benchmark_gpu_performance.cpp)
        target_link_libraries(vortex_gpu_benchmarks PRIVATE vortex_cuda vortex_core)
        if (VORTEX_BUILD_TESTS)
            add_test(NAME GpuBenchmarkCorrectnessTests COMMAND vortex_gpu_benchmarks --verify-only)
        endif()
    endif()
endif()



