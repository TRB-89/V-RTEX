#pragma once

#include <string>
#include <cstddef>
#include <iostream>

namespace vortex {

/**
 * @brief Requested execution backend for mathematical optimization solvers.
 */
enum class SolverBackend {
    CPU,   ///< Explicitly force CPU execution.
    GPU,   ///< Explicitly force GPU acceleration. Reports error/unsupported if GPU is unavailable.
    AUTO   ///< Heuristically selects CPU or GPU based on problem dimension, density, and hardware detection.
};

/**
 * @brief Actually executed backend reported in solver diagnostics.
 */
enum class ExecutedBackend {
    CPU,     ///< Pure CPU execution.
    GPU,     ///< GPU-accelerated execution.
    HYBRID   ///< Hybrid execution: iterative matrix operations on GPU, basis/KKT factorizations on CPU.
};

/**
 * @brief Diagnostic telemetry for hybrid solver device execution and transfers.
 */
struct SolverDeviceDiagnostics {
    SolverBackend requested_backend{SolverBackend::CPU};
    ExecutedBackend executed_backend{ExecutedBackend::CPU};
    std::size_t gpu_iterations{0};       ///< Number of actual solver iterations utilizing GPU kernels
    std::size_t gpu_gemv_calls{0};       ///< Number of cuBLAS GEMV operations executed
    std::size_t gpu_spmv_calls{0};       ///< Number of cuSPARSE SpMV operations executed
    std::size_t h2d_transfers{0};        ///< Number of Host-to-Device data transfers
    std::size_t d2h_transfers{0};        ///< Number of Device-to-Host data transfers
    std::string fallback_reason;         ///< Reason for CPU fallback (if requested != CPU and executed == CPU)
};

[[nodiscard]] inline std::string to_string(SolverBackend backend) {
    switch (backend) {
        case SolverBackend::CPU:  return "CPU";
        case SolverBackend::GPU:  return "GPU";
        case SolverBackend::AUTO: return "AUTO";
        default: return "UNKNOWN";
    }
}

[[nodiscard]] inline std::string to_string(ExecutedBackend backend) {
    switch (backend) {
        case ExecutedBackend::CPU:    return "CPU";
        case ExecutedBackend::GPU:    return "GPU";
        case ExecutedBackend::HYBRID: return "HYBRID (GPU Iterations / CPU Factorization)";
        default: return "UNKNOWN";
    }
}

} // namespace vortex
