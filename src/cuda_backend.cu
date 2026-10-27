#include "vortex/cuda_backend.hpp"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>

#include <iostream>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <chrono>
#include <limits>
#include <numeric>

namespace vortex::cuda {

// ==============================================================================
// Error Checking Macros and Status String Helpers
// ==============================================================================
#define VORTEX_CUDA_CHECK(call)                                                      \
    do {                                                                             \
        const cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                                  \
            std::ostringstream ss__;                                                 \
            ss__ << "CUDA Runtime Error: " << cudaGetErrorString(err__)              \
                 << " [" << cudaGetErrorName(err__) << "] "                          \
                 << "in call '" << #call << "' at " << __FILE__ << ":" << __LINE__; \
            throw std::runtime_error(ss__.str());                                    \
        }                                                                            \
    } while (0)

static const char* cublas_get_error_string(cublasStatus_t status) noexcept {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
        default: return "CUBLAS_STATUS_UNKNOWN";
    }
}

#define VORTEX_CUBLAS_CHECK(call)                                                    \
    do {                                                                             \
        const cublasStatus_t stat__ = (call);                                        \
        if (stat__ != CUBLAS_STATUS_SUCCESS) {                                       \
            std::ostringstream ss__;                                                 \
            ss__ << "cuBLAS Error: " << cublas_get_error_string(stat__)              \
                 << " (" << static_cast<int>(stat__) << ") "                        \
                 << "in call '" << #call << "' at " << __FILE__ << ":" << __LINE__; \
            throw std::runtime_error(ss__.str());                                    \
        }                                                                            \
    } while (0)

#define VORTEX_CUSPARSE_CHECK(call)                                                  \
    do {                                                                             \
        const cusparseStatus_t stat__ = (call);                                      \
        if (stat__ != CUSPARSE_STATUS_SUCCESS) {                                     \
            std::ostringstream ss__;                                                 \
            ss__ << "cuSPARSE Error: " << cusparseGetErrorString(stat__)             \
                 << " (" << static_cast<int>(stat__) << ") "                        \
                 << "in call '" << #call << "' at " << __FILE__ << ":" << __LINE__; \
            throw std::runtime_error(ss__.str());                                    \
        }                                                                            \
    } while (0)

// ==============================================================================
// CUDA Device Query Implementations
// ==============================================================================
bool is_cuda_available() noexcept {
    int count = 0;
    const cudaError_t err = cudaGetDeviceCount(&count);
    return (err == cudaSuccess && count > 0);
}

CudaDeviceInfo get_cuda_device_info(int device_id) {
    if (!is_cuda_available()) {
        throw std::runtime_error("get_cuda_device_info: No CUDA-capable GPU detected or driver unavailable.");
    }

    int count = 0;
    VORTEX_CUDA_CHECK(cudaGetDeviceCount(&count));
    if (device_id < 0 || device_id >= count) {
        std::ostringstream ss;
        ss << "get_cuda_device_info: Invalid device ID " << device_id
           << " (Available devices: " << count << ")";
        throw std::invalid_argument(ss.str());
    }

    cudaDeviceProp prop{};
    VORTEX_CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));

    CudaDeviceInfo info;
    info.device_id = device_id;
    info.name = prop.name;
    info.major = prop.major;
    info.minor = prop.minor;
    info.total_memory_bytes = prop.totalGlobalMem;
    info.multiprocessor_count = prop.multiProcessorCount;
    info.max_threads_per_block = prop.maxThreadsPerBlock;
    info.warp_size = prop.warpSize;
    info.is_available = true;

    size_t free_mem = 0;
    size_t total_mem = 0;
    if (cudaSetDevice(device_id) == cudaSuccess &&
        cudaMemGetInfo(&free_mem, &total_mem) == cudaSuccess) {
        info.free_memory_bytes = free_mem;
    } else {
        info.free_memory_bytes = 0;
    }

    return info;
}

// ==============================================================================
// Low-Level Device Memory Primitives with OOM Verification
// ==============================================================================
void* cuda_allocate_device_bytes(std::size_t bytes) {
    if (bytes == 0) return nullptr;

    size_t free_mem = 0, total_mem = 0;
    if (cudaMemGetInfo(&free_mem, &total_mem) == cudaSuccess) {
        if (bytes > free_mem) {
            std::ostringstream ss;
            ss << "CUDA Out of Memory: requested " << bytes << " bytes ("
               << (static_cast<double>(bytes) / (1024.0 * 1024.0)) << " MB), but only "
               << free_mem << " bytes (" << (static_cast<double>(free_mem) / (1024.0 * 1024.0))
               << " MB) free out of " << total_mem << " bytes total VRAM.";
            throw std::runtime_error(ss.str());
        }
    }

    void* ptr = nullptr;
    const cudaError_t err = cudaMalloc(&ptr, bytes);
    if (err == cudaErrorMemoryAllocation) {
        std::ostringstream ss;
        ss << "CUDA Out of Memory in cudaMalloc: failed to allocate "
           << bytes << " bytes.";
        throw std::runtime_error(ss.str());
    } else if (err != cudaSuccess) {
        std::ostringstream ss;
        ss << "CUDA Runtime Error: " << cudaGetErrorString(err)
           << " [" << cudaGetErrorName(err) << "] in cudaMalloc";
        throw std::runtime_error(ss.str());
    }
    return ptr;
}

void cuda_free_device_memory(void* ptr) noexcept {
    if (ptr != nullptr) {
        cudaFree(ptr);
    }
}

void cuda_copy_host_to_device(void* dst, const void* src, std::size_t bytes) {
    if (bytes == 0) return;
    if (dst == nullptr || src == nullptr) {
        throw std::invalid_argument("cuda_copy_host_to_device: null buffer pointer encountered");
    }
    VORTEX_CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice));
}

void cuda_copy_device_to_host(void* dst, const void* src, std::size_t bytes) {
    if (bytes == 0) return;
    if (dst == nullptr || src == nullptr) {
        throw std::invalid_argument("cuda_copy_device_to_host: null buffer pointer encountered");
    }
    VORTEX_CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost));
}

// ==============================================================================
// Module C: cuBLAS & cuSPARSE RAII Handle Implementations
// ==============================================================================
CublasHandle::CublasHandle() {
    VORTEX_CUBLAS_CHECK(cublasCreate(&handle_));
}

CublasHandle::~CublasHandle() {
    if (handle_ != nullptr) {
        cublasDestroy(handle_);
        handle_ = nullptr;
    }
}

CublasHandle::CublasHandle(CublasHandle&& other) noexcept : handle_(other.handle_) {
    other.handle_ = nullptr;
}

CublasHandle& CublasHandle::operator=(CublasHandle&& other) noexcept {
    if (this != &other) {
        if (handle_ != nullptr) {
            cublasDestroy(handle_);
        }
        handle_ = other.handle_;
        other.handle_ = nullptr;
    }
    return *this;
}

CusparseHandle::CusparseHandle() {
    VORTEX_CUSPARSE_CHECK(cusparseCreate(&handle_));
}

CusparseHandle::~CusparseHandle() {
    if (handle_ != nullptr) {
        cusparseDestroy(handle_);
        handle_ = nullptr;
    }
}

CusparseHandle::CusparseHandle(CusparseHandle&& other) noexcept : handle_(other.handle_) {
    other.handle_ = nullptr;
}

CusparseHandle& CusparseHandle::operator=(CusparseHandle&& other) noexcept {
    if (this != &other) {
        if (handle_ != nullptr) {
            cusparseDestroy(handle_);
        }
        handle_ = other.handle_;
        other.handle_ = nullptr;
    }
    return *this;
}

// RAII wrappers for cuSPARSE descriptors
namespace {

class CusparseSpMatWrapper {
public:
    CusparseSpMatWrapper() = default;
    explicit CusparseSpMatWrapper(cusparseSpMatDescr_t descr) : descr_(descr) {}
    ~CusparseSpMatWrapper() {
        if (descr_ != nullptr) {
            cusparseDestroySpMat(descr_);
            descr_ = nullptr;
        }
    }
    CusparseSpMatWrapper(const CusparseSpMatWrapper&) = delete;
    CusparseSpMatWrapper& operator=(const CusparseSpMatWrapper&) = delete;
    CusparseSpMatWrapper(CusparseSpMatWrapper&& other) noexcept : descr_(other.descr_) {
        other.descr_ = nullptr;
    }
    CusparseSpMatWrapper& operator=(CusparseSpMatWrapper&&) = delete;
    [[nodiscard]] cusparseSpMatDescr_t get() const noexcept { return descr_; }
private:
    cusparseSpMatDescr_t descr_{nullptr};
};

class CusparseDnVecWrapper {
public:
    CusparseDnVecWrapper() = default;
    explicit CusparseDnVecWrapper(cusparseDnVecDescr_t descr) : descr_(descr) {}
    ~CusparseDnVecWrapper() {
        if (descr_ != nullptr) {
            cusparseDestroyDnVec(descr_);
            descr_ = nullptr;
        }
    }
    CusparseDnVecWrapper(const CusparseDnVecWrapper&) = delete;
    CusparseDnVecWrapper& operator=(const CusparseDnVecWrapper&) = delete;
    CusparseDnVecWrapper(CusparseDnVecWrapper&& other) noexcept : descr_(other.descr_) {
        other.descr_ = nullptr;
    }
    CusparseDnVecWrapper& operator=(CusparseDnVecWrapper&&) = delete;
    [[nodiscard]] cusparseDnVecDescr_t get() const noexcept { return descr_; }
private:
    cusparseDnVecDescr_t descr_{nullptr};
};

} // namespace

// ==============================================================================
// CUDA Vector Addition Kernels & Implementations (Phase 8A)
// ==============================================================================
template <typename T>
__global__ void vector_add_kernel(const T* __restrict__ a,
                                  const T* __restrict__ b,
                                  T* __restrict__ c,
                                  std::size_t n) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

template <typename T>
static void launch_vector_add_pipeline(const T* a, const T* b, T* c, std::size_t n, int block_size) {
    if (a == nullptr || b == nullptr || c == nullptr) {
        throw std::invalid_argument("vector_add: null input/output pointer provided");
    }
    if (n == 0) {
        throw std::invalid_argument("vector_add: vector size n must be strictly positive (n > 0)");
    }
    if (block_size <= 0 || block_size > 1024) {
        std::ostringstream ss;
        ss << "vector_add: block_size (" << block_size << ") must be between 1 and 1024";
        throw std::invalid_argument(ss.str());
    }

    CudaDeviceBuffer<T> d_a(n);
    CudaDeviceBuffer<T> d_b(n);
    CudaDeviceBuffer<T> d_c(n);

    d_a.copy_from_host(a, n);
    d_b.copy_from_host(b, n);

    const unsigned int threads_per_block = static_cast<unsigned int>(block_size);
    const unsigned int num_blocks = static_cast<unsigned int>((n + threads_per_block - 1) / threads_per_block);

    vector_add_kernel<T><<<num_blocks, threads_per_block>>>(d_a.data(), d_b.data(), d_c.data(), n);

    VORTEX_CUDA_CHECK(cudaGetLastError());
    VORTEX_CUDA_CHECK(cudaDeviceSynchronize());

    d_c.copy_to_host(c, n);
}

void vector_add(const float* a, const float* b, float* c, std::size_t n, int block_size) {
    launch_vector_add_pipeline<float>(a, b, c, n, block_size);
}

void vector_add(const double* a, const double* b, double* c, std::size_t n, int block_size) {
    launch_vector_add_pipeline<double>(a, b, c, n, block_size);
}

std::vector<float> vector_add(const std::vector<float>& a, const std::vector<float>& b, int block_size) {
    if (a.size() != b.size()) {
        std::ostringstream ss;
        ss << "vector_add: mismatched vector dimensions (a.size=" << a.size() << ", b.size=" << b.size() << ")";
        throw std::invalid_argument(ss.str());
    }
    if (a.empty()) {
        throw std::invalid_argument("vector_add: input vectors cannot be empty");
    }
    std::vector<float> c(a.size());
    vector_add(a.data(), b.data(), c.data(), a.size(), block_size);
    return c;
}

std::vector<double> vector_add(const std::vector<double>& a, const std::vector<double>& b, int block_size) {
    if (a.size() != b.size()) {
        std::ostringstream ss;
        ss << "vector_add: mismatched vector dimensions (a.size=" << a.size() << ", b.size=" << b.size() << ")";
        throw std::invalid_argument(ss.str());
    }
    if (a.empty()) {
        throw std::invalid_argument("vector_add: input vectors cannot be empty");
    }
    std::vector<double> c(a.size());
    vector_add(a.data(), b.data(), c.data(), a.size(), block_size);
    return c;
}

VectorAddExecutionReport execute_vector_add_verified(const std::vector<double>& a,
                                                    const std::vector<double>& b,
                                                    int block_size) {
    VectorAddExecutionReport report;
    report.vector_size = a.size();
    report.block_size = block_size;
    report.grid_size = (block_size > 0 && a.size() > 0)
                           ? static_cast<int>((a.size() + block_size - 1) / block_size)
                           : 0;

    try {
        CudaDeviceInfo dev = get_cuda_device_info();
        report.device_name = dev.name;
    } catch (const std::exception& e) {
        report.device_name = "Unknown / Error: " + std::string(e.what());
    }

    try {
        std::vector<double> c_gpu = vector_add(a, b, block_size);
        double max_err = 0.0;
        for (std::size_t i = 0; i < a.size(); ++i) {
            const double ref_val = a[i] + b[i];
            const double diff = std::abs(c_gpu[i] - ref_val);
            if (diff > max_err) {
                max_err = diff;
            }
        }
        report.max_numerical_error = max_err;
        report.success = true;
        report.status_message = "Execution completed successfully; verified against CPU reference.";
    } catch (const std::exception& e) {
        report.success = false;
        report.status_message = std::string("Execution failed: ") + e.what();
    }

    return report;
}

// ==============================================================================
// Numerical Error Calculation Helper
// ==============================================================================
template <typename T>
static NumericalErrorMetrics compute_error_metrics_impl(const T* actual, const T* expected, std::size_t n, double tol) {
    NumericalErrorMetrics metrics;
    if (n == 0 || actual == nullptr || expected == nullptr) {
        return metrics;
    }

    double max_abs = 0.0;
    double max_expected_norm = 0.0;
    std::size_t mismatches = 0;

    for (std::size_t i = 0; i < n; ++i) {
        const double act = static_cast<double>(actual[i]);
        const double exp = static_cast<double>(expected[i]);
        const double abs_err = std::abs(act - exp);
        const double exp_abs = std::abs(exp);

        if (abs_err > max_abs) {
            max_abs = abs_err;
        }
        if (exp_abs > max_expected_norm) {
            max_expected_norm = exp_abs;
        }
        if (abs_err > tol) {
            mismatches++;
        }
    }

    metrics.max_absolute_error = max_abs;
    metrics.scaled_relative_error = max_abs / std::max(1.0, max_expected_norm);
    metrics.mismatch_count = mismatches;
    return metrics;
}

NumericalErrorMetrics compute_error_metrics(const double* actual, const double* expected, std::size_t n, double tol) {
    return compute_error_metrics_impl<double>(actual, expected, n, tol);
}

NumericalErrorMetrics compute_error_metrics(const float* actual, const float* expected, std::size_t n, float tol) {
    return compute_error_metrics_impl<float>(actual, expected, n, tol);
}

// ==============================================================================
// Phase 8C: Componentwise Accuracy Validation Implementations
// ==============================================================================
template <typename T>
static ComponentwiseAccuracyReport evaluate_componentwise_accuracy_impl(
    const T* actual,
    const T* expected,
    const T* componentwise_scale,
    std::size_t n,
    double atol,
    double rtol) {

    ComponentwiseAccuracyReport report;
    report.atol = atol;
    report.rtol = rtol;
    report.total_elements = n;

    if (n == 0) {
        report.passed = true;
        report.diagnostic_summary = "Zero elements to evaluate; passed vacuously.";
        return report;
    }
    if (actual == nullptr || expected == nullptr) {
        throw std::invalid_argument("evaluate_componentwise_accuracy: actual or expected pointer is null");
    }

    double max_abs = 0.0;
    double max_exp_norm = 0.0;
    double max_ratio = 0.0;
    std::size_t mismatches = 0;

    for (std::size_t i = 0; i < n; ++i) {
        const double act = static_cast<double>(actual[i]);
        const double exp = static_cast<double>(expected[i]);
        const double scale = (componentwise_scale != nullptr) ? static_cast<double>(componentwise_scale[i]) : std::abs(exp);
        const double abs_err = std::abs(act - exp);
        const double bound = atol + rtol * scale;
        const double ratio = (bound > 0.0) ? (abs_err / bound) : ((abs_err == 0.0) ? 0.0 : std::numeric_limits<double>::infinity());

        if (abs_err > max_abs) {
            max_abs = abs_err;
        }
        if (std::abs(exp) > max_exp_norm) {
            max_exp_norm = std::abs(exp);
        }
        if (ratio > max_ratio) {
            max_ratio = ratio;
            report.worst_row = i;
            report.worst_actual = act;
            report.worst_expected = exp;
            report.worst_abs_err = abs_err;
            report.worst_scale = scale;
            report.worst_bound = bound;
        }
        if (abs_err > bound) {
            mismatches++;
        }
    }

    report.max_absolute_error = max_abs;
    report.infinity_norm_residual = max_abs;
    report.scaled_relative_error = max_abs / std::max(1.0, max_exp_norm);
    report.max_componentwise_ratio = max_ratio;
    report.mismatch_count = mismatches;
    report.passed = (mismatches == 0);

    std::ostringstream ss;
    if (report.passed) {
        ss << "Componentwise accuracy verified across " << n << " elements. "
           << "Max abs err: " << max_abs << ", Max ratio: " << max_ratio << " <= 1.0.";
    } else {
        ss << "Accuracy verification FAILED: " << mismatches << "/" << n << " elements exceeded bound. "
           << "Worst at index " << report.worst_row << ": actual=" << report.worst_actual
           << ", expected=" << report.worst_expected << ", diff=" << report.worst_abs_err
           << ", bound=" << report.worst_bound << " (scale=" << report.worst_scale << ", ratio=" << max_ratio << ").";
    }
    report.diagnostic_summary = ss.str();
    return report;
}

ComponentwiseAccuracyReport evaluate_componentwise_accuracy(
    const double* actual,
    const double* expected,
    const double* componentwise_scale,
    std::size_t n,
    double atol,
    double rtol) {
    return evaluate_componentwise_accuracy_impl<double>(actual, expected, componentwise_scale, n, atol, rtol);
}

ComponentwiseAccuracyReport evaluate_componentwise_accuracy(
    const float* actual,
    const float* expected,
    const float* componentwise_scale,
    std::size_t n,
    float atol,
    float rtol) {
    return evaluate_componentwise_accuracy_impl<float>(actual, expected, componentwise_scale, n, static_cast<double>(atol), static_cast<double>(rtol));
}

ComponentwiseAccuracyReport validate_dense_gemv_accuracy(
    const Matrix<double>& A,
    const std::vector<double>& x,
    bool transpose,
    double alpha,
    double beta,
    const std::vector<double>& y_initial,
    double atol,
    double rtol) {

    const std::size_t expected_x_len = transpose ? A.rows() : A.cols();
    const std::size_t output_len     = transpose ? A.cols() : A.rows();

    if (x.size() != expected_x_len) {
        throw std::invalid_argument("validate_dense_gemv_accuracy: vector x dimension mismatch");
    }
    if (!y_initial.empty() && y_initial.size() != output_len) {
        throw std::invalid_argument("validate_dense_gemv_accuracy: y_initial dimension mismatch");
    }

    // 1. CPU Reference calculation
    std::vector<double> y_cpu(output_len, 0.0);
    if (!transpose) {
        y_cpu = A.multiply(x);
    } else {
        y_cpu = A.transpose().multiply(x);
    }
    for (std::size_t i = 0; i < output_len; ++i) {
        y_cpu[i] = alpha * y_cpu[i] + (y_initial.empty() ? 0.0 : (beta * y_initial[i]));
    }

    // 2. Componentwise scale calculation: scale_i = |alpha| * sum_j (|A_ij| * |x_j|) + |beta| * |y_init_i|
    std::vector<double> scale(output_len, 0.0);
    const double abs_alpha = std::abs(alpha);
    const double abs_beta  = std::abs(beta);

    if (!transpose) {
        for (std::size_t r = 0; r < A.rows(); ++r) {
            double dot_abs = 0.0;
            for (std::size_t c = 0; c < A.cols(); ++c) {
                dot_abs += std::abs(A(r, c)) * std::abs(x[c]);
            }
            scale[r] = abs_alpha * dot_abs + (y_initial.empty() ? 0.0 : (abs_beta * std::abs(y_initial[r])));
        }
    } else {
        for (std::size_t c = 0; c < A.cols(); ++c) {
            double dot_abs = 0.0;
            for (std::size_t r = 0; r < A.rows(); ++r) {
                dot_abs += std::abs(A(r, c)) * std::abs(x[r]);
            }
            scale[c] = abs_alpha * dot_abs + (y_initial.empty() ? 0.0 : (abs_beta * std::abs(y_initial[c])));
        }
    }

    // 3. GPU execution
    std::vector<double> y_gpu = y_initial.empty() ? std::vector<double>(output_len, 0.0) : y_initial;
    dense_gemv(A, x, y_gpu, transpose, alpha, beta);

    return evaluate_componentwise_accuracy(y_gpu.data(), y_cpu.data(), scale.data(), output_len, atol, rtol);
}

ComponentwiseAccuracyReport validate_sparse_spmv_accuracy(
    const CSRMatrix<double>& A,
    const std::vector<double>& x,
    bool transpose,
    double alpha,
    double beta,
    const std::vector<double>& y_initial,
    double atol,
    double rtol) {

    const std::size_t expected_x_len = transpose ? A.rows() : A.cols();
    const std::size_t output_len     = transpose ? A.cols() : A.rows();

    if (x.size() != expected_x_len) {
        throw std::invalid_argument("validate_sparse_spmv_accuracy: vector x dimension mismatch");
    }
    if (!y_initial.empty() && y_initial.size() != output_len) {
        throw std::invalid_argument("validate_sparse_spmv_accuracy: y_initial dimension mismatch");
    }

    // 1. CPU Reference calculation
    std::vector<double> y_cpu(output_len, 0.0);
    if (!transpose) {
        y_cpu = A.multiply(x);
    } else {
        y_cpu = A.multiply_transpose(x);
    }
    for (std::size_t i = 0; i < output_len; ++i) {
        y_cpu[i] = alpha * y_cpu[i] + (y_initial.empty() ? 0.0 : (beta * y_initial[i]));
    }

    // 2. Componentwise scale calculation
    std::vector<double> scale(output_len, 0.0);
    const double abs_alpha = std::abs(alpha);
    const double abs_beta  = std::abs(beta);
    const auto& row_ptrs = A.row_ptrs();
    const auto& col_indices = A.col_indices();
    const auto& values = A.values();

    if (!transpose) {
        for (std::size_t r = 0; r < A.rows(); ++r) {
            double dot_abs = 0.0;
            for (std::size_t idx = row_ptrs[r]; idx < row_ptrs[r + 1]; ++idx) {
                dot_abs += std::abs(values[idx]) * std::abs(x[col_indices[idx]]);
            }
            scale[r] = abs_alpha * dot_abs + (y_initial.empty() ? 0.0 : (abs_beta * std::abs(y_initial[r])));
        }
    } else {
        for (std::size_t r = 0; r < A.rows(); ++r) {
            const double xr_abs = std::abs(x[r]);
            for (std::size_t idx = row_ptrs[r]; idx < row_ptrs[r + 1]; ++idx) {
                scale[col_indices[idx]] += abs_alpha * std::abs(values[idx]) * xr_abs;
            }
        }
        if (!y_initial.empty()) {
            for (std::size_t c = 0; c < output_len; ++c) {
                scale[c] += abs_beta * std::abs(y_initial[c]);
            }
        }
    }

    // 3. GPU execution
    std::vector<double> y_gpu = y_initial.empty() ? std::vector<double>(output_len, 0.0) : y_initial;
    sparse_spmv(A, x, y_gpu, transpose, alpha, beta);

    return evaluate_componentwise_accuracy(y_gpu.data(), y_cpu.data(), scale.data(), output_len, atol, rtol);
}

// ==============================================================================
// Module A: Dense Matrix-Vector Multiplication (cuBLAS GEMV)
// ==============================================================================
// Row-major CPU input vs Column-major cuBLAS storage:
// In C++, matrix A of shape (rows x cols) is stored row-major with leading dimension cols.
// Viewed in column-major order (as cuBLAS expects), this same array represents a matrix
// C of shape (cols x rows) with leading dimension lda = cols, where C = A^T.
//
// 1. Non-transposed multiply: y = A * x (A is rows x cols, x is cols x 1, y is rows x 1).
//    Since C = A^T, we have A = C^T.
//    Therefore, y = C^T * x.
//    In cuBLAS: trans = CUBLAS_OP_T, m = cols, n = rows, lda = cols.
//    Result: y = alpha * C^T * x + beta * y = alpha * A * x + beta * y.
//
// 2. Transposed multiply: y = A^T * x (A^T is cols x rows, x is rows x 1, y is cols x 1).
//    Since C = A^T, we have A^T = C.
//    Therefore, y = C * x.
//    In cuBLAS: trans = CUBLAS_OP_N, m = cols, n = rows, lda = cols.
//    Result: y = alpha * C * x + beta * y = alpha * A^T * x + beta * y.
//
// This formulation avoids explicit transpositions or memory reallocations.

void dense_gemv(const double* A, std::size_t rows, std::size_t cols,
                const double* x, double* y,
                bool transpose,
                double alpha, double beta,
                cublasHandle_t handle) {
    if (A == nullptr || x == nullptr || y == nullptr) {
        throw std::invalid_argument("dense_gemv: null matrix/vector pointer encountered");
    }
    if (rows == 0 || cols == 0) {
        throw std::invalid_argument("dense_gemv: matrix dimensions (rows and cols) must be strictly positive");
    }
    if (rows > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        cols > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("dense_gemv: dimensions exceed 32-bit integer limits supported by cuBLAS");
    }

    const std::size_t x_len = transpose ? rows : cols;
    const std::size_t y_len = transpose ? cols : rows;

    // Validate finite numerical values
    for (std::size_t i = 0; i < rows * cols; ++i) {
        if (!std::isfinite(A[i])) {
            throw std::invalid_argument("dense_gemv: non-finite (NaN/Inf) element detected in matrix A at index " + std::to_string(i));
        }
    }
    for (std::size_t i = 0; i < x_len; ++i) {
        if (!std::isfinite(x[i])) {
            throw std::invalid_argument("dense_gemv: non-finite (NaN/Inf) element detected in vector x at index " + std::to_string(i));
        }
    }
    if (beta != 0.0) {
        for (std::size_t i = 0; i < y_len; ++i) {
            if (!std::isfinite(y[i])) {
                throw std::invalid_argument("dense_gemv: non-finite (NaN/Inf) element detected in initial vector y at index " + std::to_string(i));
            }
        }
    }

    // Handle lifecycle: use caller-supplied handle or construct an RAII handle
    std::unique_ptr<CublasHandle> local_handle;
    cublasHandle_t cublas_h = handle;
    if (cublas_h == nullptr) {
        local_handle = std::make_unique<CublasHandle>();
        cublas_h = local_handle->get();
    }

    // Allocate GPU buffers via RAII
    CudaDeviceBuffer<double> d_A(rows * cols);
    CudaDeviceBuffer<double> d_x(x_len);
    CudaDeviceBuffer<double> d_y(y_len);

    d_A.copy_from_host(A, rows * cols);
    d_x.copy_from_host(x, x_len);
    if (beta != 0.0) {
        d_y.copy_from_host(y, y_len);
    } else {
        // Zero-initialize device output memory when beta == 0
        VORTEX_CUDA_CHECK(cudaMemset(d_y.data(), 0, y_len * sizeof(double)));
    }

    const int m_cu = static_cast<int>(cols);
    const int n_cu = static_cast<int>(rows);
    const int lda  = static_cast<int>(cols);
    const cublasOperation_t op = transpose ? CUBLAS_OP_N : CUBLAS_OP_T;

    VORTEX_CUBLAS_CHECK(cublasDgemv(cublas_h,
                                   op,
                                   m_cu,
                                   n_cu,
                                   &alpha,
                                   d_A.data(),
                                   lda,
                                   d_x.data(),
                                   1,
                                   &beta,
                                   d_y.data(),
                                   1));

    VORTEX_CUDA_CHECK(cudaDeviceSynchronize());
    d_y.copy_to_host(y, y_len);
}

void dense_gemv(const float* A, std::size_t rows, std::size_t cols,
                const float* x, float* y,
                bool transpose,
                float alpha, float beta,
                cublasHandle_t handle) {
    if (A == nullptr || x == nullptr || y == nullptr) {
        throw std::invalid_argument("dense_gemv: null matrix/vector pointer encountered");
    }
    if (rows == 0 || cols == 0) {
        throw std::invalid_argument("dense_gemv: matrix dimensions (rows and cols) must be strictly positive");
    }
    if (rows > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        cols > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("dense_gemv: dimensions exceed 32-bit integer limits supported by cuBLAS");
    }

    const std::size_t x_len = transpose ? rows : cols;
    const std::size_t y_len = transpose ? cols : rows;

    for (std::size_t i = 0; i < rows * cols; ++i) {
        if (!std::isfinite(A[i])) {
            throw std::invalid_argument("dense_gemv: non-finite (NaN/Inf) element detected in matrix A at index " + std::to_string(i));
        }
    }
    for (std::size_t i = 0; i < x_len; ++i) {
        if (!std::isfinite(x[i])) {
            throw std::invalid_argument("dense_gemv: non-finite (NaN/Inf) element detected in vector x at index " + std::to_string(i));
        }
    }
    if (beta != 0.0f) {
        for (std::size_t i = 0; i < y_len; ++i) {
            if (!std::isfinite(y[i])) {
                throw std::invalid_argument("dense_gemv: non-finite (NaN/Inf) element detected in initial vector y at index " + std::to_string(i));
            }
        }
    }

    std::unique_ptr<CublasHandle> local_handle;
    cublasHandle_t cublas_h = handle;
    if (cublas_h == nullptr) {
        local_handle = std::make_unique<CublasHandle>();
        cublas_h = local_handle->get();
    }

    CudaDeviceBuffer<float> d_A(rows * cols);
    CudaDeviceBuffer<float> d_x(x_len);
    CudaDeviceBuffer<float> d_y(y_len);

    d_A.copy_from_host(A, rows * cols);
    d_x.copy_from_host(x, x_len);
    if (beta != 0.0f) {
        d_y.copy_from_host(y, y_len);
    } else {
        VORTEX_CUDA_CHECK(cudaMemset(d_y.data(), 0, y_len * sizeof(float)));
    }

    const int m_cu = static_cast<int>(cols);
    const int n_cu = static_cast<int>(rows);
    const int lda  = static_cast<int>(cols);
    const cublasOperation_t op = transpose ? CUBLAS_OP_N : CUBLAS_OP_T;

    VORTEX_CUBLAS_CHECK(cublasSgemv(cublas_h,
                                   op,
                                   m_cu,
                                   n_cu,
                                   &alpha,
                                   d_A.data(),
                                   lda,
                                   d_x.data(),
                                   1,
                                   &beta,
                                   d_y.data(),
                                   1));

    VORTEX_CUDA_CHECK(cudaDeviceSynchronize());
    d_y.copy_to_host(y, y_len);
}

std::vector<double> dense_gemv(const Matrix<double>& A,
                               const std::vector<double>& x,
                               bool transpose,
                               double alpha, double beta,
                               cublasHandle_t handle) {
    const std::size_t expected_x_size = transpose ? A.rows() : A.cols();
    if (x.size() != expected_x_size) {
        std::ostringstream ss;
        ss << "dense_gemv: input vector x size (" << x.size()
           << ") does not match matrix requirement (" << expected_x_size
           << ") [transpose=" << (transpose ? "true" : "false") << "]";
        throw std::invalid_argument(ss.str());
    }

    const std::size_t y_size = transpose ? A.cols() : A.rows();
    std::vector<double> y(y_size, 0.0);
    dense_gemv(A.data(), A.rows(), A.cols(), x.data(), y.data(), transpose, alpha, beta, handle);
    return y;
}

std::vector<float> dense_gemv(const Matrix<float>& A,
                              const std::vector<float>& x,
                              bool transpose,
                              float alpha, float beta,
                              cublasHandle_t handle) {
    const std::size_t expected_x_size = transpose ? A.rows() : A.cols();
    if (x.size() != expected_x_size) {
        std::ostringstream ss;
        ss << "dense_gemv: input vector x size (" << x.size()
           << ") does not match matrix requirement (" << expected_x_size
           << ") [transpose=" << (transpose ? "true" : "false") << "]";
        throw std::invalid_argument(ss.str());
    }

    const std::size_t y_size = transpose ? A.cols() : A.rows();
    std::vector<float> y(y_size, 0.0f);
    dense_gemv(A.data(), A.rows(), A.cols(), x.data(), y.data(), transpose, alpha, beta, handle);
    return y;
}

void dense_gemv(const Matrix<double>& A,
                const std::vector<double>& x,
                std::vector<double>& y,
                bool transpose,
                double alpha, double beta,
                cublasHandle_t handle) {
    const std::size_t expected_x_size = transpose ? A.rows() : A.cols();
    const std::size_t expected_y_size = transpose ? A.cols() : A.rows();

    if (x.size() != expected_x_size) {
        throw std::invalid_argument("dense_gemv: input vector x size mismatch");
    }
    if (y.size() != expected_y_size) {
        throw std::invalid_argument("dense_gemv: output vector y size mismatch");
    }

    dense_gemv(A.data(), A.rows(), A.cols(), x.data(), y.data(), transpose, alpha, beta, handle);
}

void dense_gemv(const Matrix<float>& A,
                const std::vector<float>& x,
                std::vector<float>& y,
                bool transpose,
                float alpha, float beta,
                cublasHandle_t handle) {
    const std::size_t expected_x_size = transpose ? A.rows() : A.cols();
    const std::size_t expected_y_size = transpose ? A.cols() : A.rows();

    if (x.size() != expected_x_size) {
        throw std::invalid_argument("dense_gemv: input vector x size mismatch");
    }
    if (y.size() != expected_y_size) {
        throw std::invalid_argument("dense_gemv: output vector y size mismatch");
    }

    dense_gemv(A.data(), A.rows(), A.cols(), x.data(), y.data(), transpose, alpha, beta, handle);
}

// ==============================================================================
// Module B: Sparse Matrix-Vector Multiplication (cuSPARSE SpMV)
// ==============================================================================
void sparse_spmv_csr(std::size_t rows, std::size_t cols, std::size_t nnz,
                     const double* values, const int* col_indices, const int* row_ptrs,
                     const double* x, double* y,
                     bool transpose,
                     double alpha, double beta,
                     cusparseHandle_t handle) {
    if (row_ptrs == nullptr || x == nullptr || y == nullptr) {
        throw std::invalid_argument("sparse_spmv_csr: null pointer encountered in row_ptrs, x, or y");
    }
    if (nnz > 0 && (values == nullptr || col_indices == nullptr)) {
        throw std::invalid_argument("sparse_spmv_csr: null pointer encountered in values or col_indices with nnz > 0");
    }
    if (rows == 0 || cols == 0) {
        throw std::invalid_argument("sparse_spmv_csr: matrix dimensions must be strictly positive");
    }
    if (rows > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        cols > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        nnz  > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("sparse_spmv_csr: dimensions or non-zero count exceed 32-bit integer limits");
    }

    const std::size_t x_len = transpose ? rows : cols;
    const std::size_t y_len = transpose ? cols : rows;

    // Validate CSR row pointers structure
    if (row_ptrs[0] != 0) {
        throw std::invalid_argument("sparse_spmv_csr: row_ptrs[0] must be exactly 0 (0-based indexing)");
    }
    if (static_cast<std::size_t>(row_ptrs[rows]) != nnz) {
        std::ostringstream ss;
        ss << "sparse_spmv_csr: row_ptrs[rows] (" << row_ptrs[rows]
           << ") must match nnz (" << nnz << ")";
        throw std::invalid_argument(ss.str());
    }
    for (std::size_t r = 0; r < rows; ++r) {
        if (row_ptrs[r] > row_ptrs[r + 1]) {
            std::ostringstream ss;
            ss << "sparse_spmv_csr: non-monotonic row_ptrs at row " << r
               << " (start=" << row_ptrs[r] << ", end=" << row_ptrs[r + 1] << ")";
            throw std::invalid_argument(ss.str());
        }
    }

    // Validate column indices and finite values
    for (std::size_t i = 0; i < nnz; ++i) {
        if (col_indices[i] < 0 || static_cast<std::size_t>(col_indices[i]) >= cols) {
            std::ostringstream ss;
            ss << "sparse_spmv_csr: col_indices[" << i << "] = " << col_indices[i]
               << " is out of bounds for cols = " << cols;
            throw std::invalid_argument(ss.str());
        }
        if (!std::isfinite(values[i])) {
            throw std::invalid_argument("sparse_spmv_csr: non-finite (NaN/Inf) element in sparse matrix at index " + std::to_string(i));
        }
    }
    for (std::size_t i = 0; i < x_len; ++i) {
        if (!std::isfinite(x[i])) {
            throw std::invalid_argument("sparse_spmv_csr: non-finite (NaN/Inf) element in vector x at index " + std::to_string(i));
        }
    }
    if (beta != 0.0) {
        for (std::size_t i = 0; i < y_len; ++i) {
            if (!std::isfinite(y[i])) {
                throw std::invalid_argument("sparse_spmv_csr: non-finite (NaN/Inf) element in vector y at index " + std::to_string(i));
            }
        }
    }

    // Safe zero-matrix handling: if nnz == 0, y = beta * y (or zeros if beta == 0)
    if (nnz == 0) {
        for (std::size_t i = 0; i < y_len; ++i) {
            y[i] = (beta == 0.0) ? 0.0 : (beta * y[i]);
        }
        return;
    }

    // Handle lifecycle
    std::unique_ptr<CusparseHandle> local_handle;
    cusparseHandle_t cusparse_h = handle;
    if (cusparse_h == nullptr) {
        local_handle = std::make_unique<CusparseHandle>();
        cusparse_h = local_handle->get();
    }

    // Allocate Device Buffers via RAII
    CudaDeviceBuffer<int> d_row_ptrs(rows + 1);
    CudaDeviceBuffer<int> d_col_indices(nnz);
    CudaDeviceBuffer<double> d_values(nnz);
    CudaDeviceBuffer<double> d_x(x_len);
    CudaDeviceBuffer<double> d_y(y_len);

    d_row_ptrs.copy_from_host(row_ptrs, rows + 1);
    d_col_indices.copy_from_host(col_indices, nnz);
    d_values.copy_from_host(values, nnz);
    d_x.copy_from_host(x, x_len);
    if (beta != 0.0) {
        d_y.copy_from_host(y, y_len);
    } else {
        VORTEX_CUDA_CHECK(cudaMemset(d_y.data(), 0, y_len * sizeof(double)));
    }

    // Create cuSPARSE Descriptors wrapped in RAII
    cusparseSpMatDescr_t raw_matA = nullptr;
    VORTEX_CUSPARSE_CHECK(cusparseCreateCsr(&raw_matA,
                                           static_cast<int64_t>(rows),
                                           static_cast<int64_t>(cols),
                                           static_cast<int64_t>(nnz),
                                           d_row_ptrs.data(),
                                           d_col_indices.data(),
                                           d_values.data(),
                                           CUSPARSE_INDEX_32I,
                                           CUSPARSE_INDEX_32I,
                                           CUSPARSE_INDEX_BASE_ZERO,
                                           CUDA_R_64F));
    CusparseSpMatWrapper matA(raw_matA);

    cusparseDnVecDescr_t raw_vecX = nullptr;
    VORTEX_CUSPARSE_CHECK(cusparseCreateDnVec(&raw_vecX,
                                             static_cast<int64_t>(x_len),
                                             d_x.data(),
                                             CUDA_R_64F));
    CusparseDnVecWrapper vecX(raw_vecX);

    cusparseDnVecDescr_t raw_vecY = nullptr;
    VORTEX_CUSPARSE_CHECK(cusparseCreateDnVec(&raw_vecY,
                                             static_cast<int64_t>(y_len),
                                             d_y.data(),
                                             CUDA_R_64F));
    CusparseDnVecWrapper vecY(raw_vecY);

    const cusparseOperation_t op = transpose ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;

    // Query workspace buffer size
    size_t buffer_size = 0;
    VORTEX_CUSPARSE_CHECK(cusparseSpMV_bufferSize(cusparse_h,
                                                 op,
                                                 &alpha,
                                                 matA.get(),
                                                 vecX.get(),
                                                 &beta,
                                                 vecY.get(),
                                                 CUDA_R_64F,
                                                 CUSPARSE_SPMV_ALG_DEFAULT,
                                                 &buffer_size));

    CudaDeviceBuffer<uint8_t> d_workspace(buffer_size);

    // Execute cuSPARSE SpMV
    VORTEX_CUSPARSE_CHECK(cusparseSpMV(cusparse_h,
                                       op,
                                       &alpha,
                                       matA.get(),
                                       vecX.get(),
                                       &beta,
                                       vecY.get(),
                                       CUDA_R_64F,
                                       CUSPARSE_SPMV_ALG_DEFAULT,
                                       d_workspace.data()));

    VORTEX_CUDA_CHECK(cudaDeviceSynchronize());
    d_y.copy_to_host(y, y_len);
}

void sparse_spmv_csr(std::size_t rows, std::size_t cols, std::size_t nnz,
                     const float* values, const int* col_indices, const int* row_ptrs,
                     const float* x, float* y,
                     bool transpose,
                     float alpha, float beta,
                     cusparseHandle_t handle) {
    if (row_ptrs == nullptr || x == nullptr || y == nullptr) {
        throw std::invalid_argument("sparse_spmv_csr: null pointer encountered in row_ptrs, x, or y");
    }
    if (nnz > 0 && (values == nullptr || col_indices == nullptr)) {
        throw std::invalid_argument("sparse_spmv_csr: null pointer encountered in values or col_indices with nnz > 0");
    }
    if (rows == 0 || cols == 0) {
        throw std::invalid_argument("sparse_spmv_csr: matrix dimensions must be strictly positive");
    }
    if (rows > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        cols > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        nnz  > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("sparse_spmv_csr: dimensions or non-zero count exceed 32-bit integer limits");
    }

    const std::size_t x_len = transpose ? rows : cols;
    const std::size_t y_len = transpose ? cols : rows;

    if (row_ptrs[0] != 0) {
        throw std::invalid_argument("sparse_spmv_csr: row_ptrs[0] must be exactly 0 (0-based indexing)");
    }
    if (static_cast<std::size_t>(row_ptrs[rows]) != nnz) {
        throw std::invalid_argument("sparse_spmv_csr: row_ptrs[rows] must match nnz");
    }
    for (std::size_t r = 0; r < rows; ++r) {
        if (row_ptrs[r] > row_ptrs[r + 1]) {
            throw std::invalid_argument("sparse_spmv_csr: non-monotonic row_ptrs detected");
        }
    }

    for (std::size_t i = 0; i < nnz; ++i) {
        if (col_indices[i] < 0 || static_cast<std::size_t>(col_indices[i]) >= cols) {
            throw std::invalid_argument("sparse_spmv_csr: col_indices is out of bounds");
        }
        if (!std::isfinite(values[i])) {
            throw std::invalid_argument("sparse_spmv_csr: non-finite element in sparse matrix values");
        }
    }
    for (std::size_t i = 0; i < x_len; ++i) {
        if (!std::isfinite(x[i])) {
            throw std::invalid_argument("sparse_spmv_csr: non-finite element in vector x");
        }
    }
    if (beta != 0.0f) {
        for (std::size_t i = 0; i < y_len; ++i) {
            if (!std::isfinite(y[i])) {
                throw std::invalid_argument("sparse_spmv_csr: non-finite element in vector y");
            }
        }
    }

    if (nnz == 0) {
        for (std::size_t i = 0; i < y_len; ++i) {
            y[i] = (beta == 0.0f) ? 0.0f : (beta * y[i]);
        }
        return;
    }

    std::unique_ptr<CusparseHandle> local_handle;
    cusparseHandle_t cusparse_h = handle;
    if (cusparse_h == nullptr) {
        local_handle = std::make_unique<CusparseHandle>();
        cusparse_h = local_handle->get();
    }

    CudaDeviceBuffer<int> d_row_ptrs(rows + 1);
    CudaDeviceBuffer<int> d_col_indices(nnz);
    CudaDeviceBuffer<float> d_values(nnz);
    CudaDeviceBuffer<float> d_x(x_len);
    CudaDeviceBuffer<float> d_y(y_len);

    d_row_ptrs.copy_from_host(row_ptrs, rows + 1);
    d_col_indices.copy_from_host(col_indices, nnz);
    d_values.copy_from_host(values, nnz);
    d_x.copy_from_host(x, x_len);
    if (beta != 0.0f) {
        d_y.copy_from_host(y, y_len);
    } else {
        VORTEX_CUDA_CHECK(cudaMemset(d_y.data(), 0, y_len * sizeof(float)));
    }

    cusparseSpMatDescr_t raw_matA = nullptr;
    VORTEX_CUSPARSE_CHECK(cusparseCreateCsr(&raw_matA,
                                           static_cast<int64_t>(rows),
                                           static_cast<int64_t>(cols),
                                           static_cast<int64_t>(nnz),
                                           d_row_ptrs.data(),
                                           d_col_indices.data(),
                                           d_values.data(),
                                           CUSPARSE_INDEX_32I,
                                           CUSPARSE_INDEX_32I,
                                           CUSPARSE_INDEX_BASE_ZERO,
                                           CUDA_R_32F));
    CusparseSpMatWrapper matA(raw_matA);

    cusparseDnVecDescr_t raw_vecX = nullptr;
    VORTEX_CUSPARSE_CHECK(cusparseCreateDnVec(&raw_vecX,
                                             static_cast<int64_t>(x_len),
                                             d_x.data(),
                                             CUDA_R_32F));
    CusparseDnVecWrapper vecX(raw_vecX);

    cusparseDnVecDescr_t raw_vecY = nullptr;
    VORTEX_CUSPARSE_CHECK(cusparseCreateDnVec(&raw_vecY,
                                             static_cast<int64_t>(y_len),
                                             d_y.data(),
                                             CUDA_R_32F));
    CusparseDnVecWrapper vecY(raw_vecY);

    const cusparseOperation_t op = transpose ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;

    size_t buffer_size = 0;
    VORTEX_CUSPARSE_CHECK(cusparseSpMV_bufferSize(cusparse_h,
                                                 op,
                                                 &alpha,
                                                 matA.get(),
                                                 vecX.get(),
                                                 &beta,
                                                 vecY.get(),
                                                 CUDA_R_32F,
                                                 CUSPARSE_SPMV_ALG_DEFAULT,
                                                 &buffer_size));

    CudaDeviceBuffer<uint8_t> d_workspace(buffer_size);

    VORTEX_CUSPARSE_CHECK(cusparseSpMV(cusparse_h,
                                       op,
                                       &alpha,
                                       matA.get(),
                                       vecX.get(),
                                       &beta,
                                       vecY.get(),
                                       CUDA_R_32F,
                                       CUSPARSE_SPMV_ALG_DEFAULT,
                                       d_workspace.data()));

    VORTEX_CUDA_CHECK(cudaDeviceSynchronize());
    d_y.copy_to_host(y, y_len);
}

// Convert VORTEX CSRMatrix<T> (which uses std::size_t indices) to 32-bit int vectors for cuSPARSE
template <typename T>
static void convert_csr_indices_to_int(const CSRMatrix<T>& A,
                                       std::vector<int>& col_indices_int,
                                       std::vector<int>& row_ptrs_int) {
    const auto& col_indices = A.col_indices();
    const auto& row_ptrs = A.row_ptrs();

    col_indices_int.resize(col_indices.size());
    for (std::size_t i = 0; i < col_indices.size(); ++i) {
        if (col_indices[i] > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            throw std::overflow_error("convert_csr_indices_to_int: column index exceeds 32-bit integer range");
        }
        col_indices_int[i] = static_cast<int>(col_indices[i]);
    }

    row_ptrs_int.resize(row_ptrs.size());
    for (std::size_t i = 0; i < row_ptrs.size(); ++i) {
        if (row_ptrs[i] > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            throw std::overflow_error("convert_csr_indices_to_int: row pointer exceeds 32-bit integer range");
        }
        row_ptrs_int[i] = static_cast<int>(row_ptrs[i]);
    }
}

std::vector<double> sparse_spmv(const CSRMatrix<double>& A,
                                const std::vector<double>& x,
                                bool transpose,
                                double alpha, double beta,
                                cusparseHandle_t handle) {
    const std::size_t expected_x_size = transpose ? A.rows() : A.cols();
    if (x.size() != expected_x_size) {
        std::ostringstream ss;
        ss << "sparse_spmv: input vector x size (" << x.size()
           << ") does not match matrix dimension (" << expected_x_size
           << ") [transpose=" << (transpose ? "true" : "false") << "]";
        throw std::invalid_argument(ss.str());
    }

    const std::size_t y_size = transpose ? A.cols() : A.rows();
    std::vector<double> y(y_size, 0.0);

    std::vector<int> col_indices_int;
    std::vector<int> row_ptrs_int;
    convert_csr_indices_to_int(A, col_indices_int, row_ptrs_int);

    sparse_spmv_csr(A.rows(), A.cols(), A.non_zeros(),
                    A.values().data(), col_indices_int.data(), row_ptrs_int.data(),
                    x.data(), y.data(), transpose, alpha, beta, handle);
    return y;
}

std::vector<float> sparse_spmv(const CSRMatrix<float>& A,
                               const std::vector<float>& x,
                               bool transpose,
                               float alpha, float beta,
                               cusparseHandle_t handle) {
    const std::size_t expected_x_size = transpose ? A.rows() : A.cols();
    if (x.size() != expected_x_size) {
        std::ostringstream ss;
        ss << "sparse_spmv: input vector x size (" << x.size()
           << ") does not match matrix dimension (" << expected_x_size
           << ") [transpose=" << (transpose ? "true" : "false") << "]";
        throw std::invalid_argument(ss.str());
    }

    const std::size_t y_size = transpose ? A.cols() : A.rows();
    std::vector<float> y(y_size, 0.0f);

    std::vector<int> col_indices_int;
    std::vector<int> row_ptrs_int;
    convert_csr_indices_to_int(A, col_indices_int, row_ptrs_int);

    sparse_spmv_csr(A.rows(), A.cols(), A.non_zeros(),
                    A.values().data(), col_indices_int.data(), row_ptrs_int.data(),
                    x.data(), y.data(), transpose, alpha, beta, handle);
    return y;
}

void sparse_spmv(const CSRMatrix<double>& A,
                 const std::vector<double>& x,
                 std::vector<double>& y,
                 bool transpose,
                 double alpha, double beta,
                 cusparseHandle_t handle) {
    const std::size_t expected_x_size = transpose ? A.rows() : A.cols();
    const std::size_t expected_y_size = transpose ? A.cols() : A.rows();

    if (x.size() != expected_x_size) {
        throw std::invalid_argument("sparse_spmv: input vector x size mismatch");
    }
    if (y.size() != expected_y_size) {
        throw std::invalid_argument("sparse_spmv: output vector y size mismatch");
    }

    std::vector<int> col_indices_int;
    std::vector<int> row_ptrs_int;
    convert_csr_indices_to_int(A, col_indices_int, row_ptrs_int);

    sparse_spmv_csr(A.rows(), A.cols(), A.non_zeros(),
                    A.values().data(), col_indices_int.data(), row_ptrs_int.data(),
                    x.data(), y.data(), transpose, alpha, beta, handle);
}

void sparse_spmv(const CSRMatrix<float>& A,
                 const std::vector<float>& x,
                 std::vector<float>& y,
                 bool transpose,
                 float alpha, float beta,
                 cusparseHandle_t handle) {
    const std::size_t expected_x_size = transpose ? A.rows() : A.cols();
    const std::size_t expected_y_size = transpose ? A.cols() : A.rows();

    if (x.size() != expected_x_size) {
        throw std::invalid_argument("sparse_spmv: input vector x size mismatch");
    }
    if (y.size() != expected_y_size) {
        throw std::invalid_argument("sparse_spmv: output vector y size mismatch");
    }

    std::vector<int> col_indices_int;
    std::vector<int> row_ptrs_int;
    convert_csr_indices_to_int(A, col_indices_int, row_ptrs_int);

    sparse_spmv_csr(A.rows(), A.cols(), A.non_zeros(),
                    A.values().data(), col_indices_int.data(), row_ptrs_int.data(),
                    x.data(), y.data(), transpose, alpha, beta, handle);
}

// ==============================================================================
// Module E: Benchmarking Infrastructure
// ==============================================================================
namespace {

double calculate_median(std::vector<double>& values) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const std::size_t mid = values.size() / 2;
    if (values.size() % 2 == 0) {
        return (values[mid - 1] + values[mid]) * 0.5;
    }
    return values[mid];
}

class CudaScopedEventTimer {
public:
    CudaScopedEventTimer() {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }
    ~CudaScopedEventTimer() {
        if (start_ != nullptr) cudaEventDestroy(start_);
        if (stop_ != nullptr) cudaEventDestroy(stop_);
    }
    void record_start(cudaStream_t stream = nullptr) {
        cudaEventRecord(start_, stream);
    }
    void record_stop(cudaStream_t stream = nullptr) {
        cudaEventRecord(stop_, stream);
    }
    [[nodiscard]] double elapsed_ms() {
        cudaEventSynchronize(stop_);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start_, stop_);
        return static_cast<double>(ms);
    }
private:
    cudaEvent_t start_{nullptr};
    cudaEvent_t stop_{nullptr};
};

} // namespace

GemvBenchmarkResult benchmark_dense_gemv(const Matrix<double>& A,
                                         const std::vector<double>& x,
                                         bool transpose,
                                         int warmup_runs,
                                         int timed_runs) {
    GemvBenchmarkResult res;
    res.name = "Dense GEMV (" + std::to_string(A.rows()) + "x" + std::to_string(A.cols()) +
               (transpose ? " Transposed" : " Non-Transposed") + ")";
    res.rows = A.rows();
    res.cols = A.cols();
    res.transpose = transpose;

    const std::size_t x_len = transpose ? A.rows() : A.cols();
    const std::size_t y_len = transpose ? A.cols() : A.rows();

    if (x.size() != x_len) {
        throw std::invalid_argument("benchmark_dense_gemv: input vector x size mismatch");
    }

    // 1. CPU Reference Benchmark
    std::vector<double> y_cpu(y_len, 0.0);
    // Warmup CPU
    for (int i = 0; i < warmup_runs; ++i) {
        if (transpose) {
            y_cpu = A.transpose().multiply(x);
        } else {
            y_cpu = A.multiply(x);
        }
    }
    // Timed CPU runs
    std::vector<double> cpu_times;
    cpu_times.reserve(timed_runs);
    for (int i = 0; i < timed_runs; ++i) {
        const auto t0 = std::chrono::high_resolution_clock::now();
        if (transpose) {
            y_cpu = A.transpose().multiply(x);
        } else {
            y_cpu = A.multiply(x);
        }
        const auto t1 = std::chrono::high_resolution_clock::now();
        cpu_times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    res.cpu_median_ms = calculate_median(cpu_times);

    // 2. GPU Warmup (ensure driver & context initialized)
    CublasHandle handle;
    std::vector<double> y_gpu(y_len, 0.0);
    for (int i = 0; i < warmup_runs; ++i) {
        dense_gemv(A.data(), A.rows(), A.cols(), x.data(), y_gpu.data(), transpose, 1.0, 0.0, handle.get());
    }

    // 3. Isolated Phase Benchmarking (H2D, Kernel, D2H, E2E)
    CudaDeviceBuffer<double> d_A(A.rows() * A.cols());
    CudaDeviceBuffer<double> d_x(x_len);
    CudaDeviceBuffer<double> d_y(y_len);

    std::vector<double> h2d_times;
    std::vector<double> kernel_times;
    std::vector<double> d2h_times;
    std::vector<double> e2e_times;

    h2d_times.reserve(timed_runs);
    kernel_times.reserve(timed_runs);
    d2h_times.reserve(timed_runs);
    e2e_times.reserve(timed_runs);

    CudaScopedEventTimer event_timer;
    const int m_cu = static_cast<int>(A.cols());
    const int n_cu = static_cast<int>(A.rows());
    const int lda  = static_cast<int>(A.cols());
    const cublasOperation_t op = transpose ? CUBLAS_OP_N : CUBLAS_OP_T;
    const double alpha = 1.0;
    const double beta  = 0.0;

    for (int i = 0; i < timed_runs; ++i) {
        // Full E2E timer
        const auto e2e_start = std::chrono::high_resolution_clock::now();

        // Phase 1: H2D Transfer
        const auto h2d_start = std::chrono::high_resolution_clock::now();
        d_A.copy_from_host(A.data(), A.rows() * A.cols());
        d_x.copy_from_host(x.data(), x_len);
        VORTEX_CUDA_CHECK(cudaDeviceSynchronize());
        const auto h2d_end = std::chrono::high_resolution_clock::now();
        h2d_times.push_back(std::chrono::duration<double, std::milli>(h2d_end - h2d_start).count());

        // Phase 2: GPU Kernel (cuBLAS GEMV)
        event_timer.record_start();
        VORTEX_CUBLAS_CHECK(cublasDgemv(handle.get(),
                                       op,
                                       m_cu,
                                       n_cu,
                                       &alpha,
                                       d_A.data(),
                                       lda,
                                       d_x.data(),
                                       1,
                                       &beta,
                                       d_y.data(),
                                       1));
        event_timer.record_stop();
        kernel_times.push_back(event_timer.elapsed_ms());

        // Phase 3: D2H Transfer
        const auto d2h_start = std::chrono::high_resolution_clock::now();
        d_y.copy_to_host(y_gpu.data(), y_len);
        VORTEX_CUDA_CHECK(cudaDeviceSynchronize());
        const auto d2h_end = std::chrono::high_resolution_clock::now();
        d2h_times.push_back(std::chrono::duration<double, std::milli>(d2h_end - d2h_start).count());

        const auto e2e_end = std::chrono::high_resolution_clock::now();
        e2e_times.push_back(std::chrono::duration<double, std::milli>(e2e_end - e2e_start).count());
    }

    res.gpu_h2d_ms    = calculate_median(h2d_times);
    res.gpu_kernel_ms = calculate_median(kernel_times);
    res.gpu_d2h_ms    = calculate_median(d2h_times);
    res.gpu_e2e_ms    = calculate_median(e2e_times);

    res.speedup_e2e     = (res.gpu_e2e_ms > 0.0)    ? (res.cpu_median_ms / res.gpu_e2e_ms)    : 0.0;
    res.speedup_compute = (res.gpu_kernel_ms > 0.0) ? (res.cpu_median_ms / res.gpu_kernel_ms) : 0.0;

    const auto err = compute_error_metrics(y_gpu.data(), y_cpu.data(), y_len);
    res.max_abs_error    = err.max_absolute_error;
    res.scaled_rel_error = err.scaled_relative_error;

    return res;
}

SpmvBenchmarkResult benchmark_sparse_spmv(const CSRMatrix<double>& A,
                                          const std::vector<double>& x,
                                          bool transpose,
                                          int warmup_runs,
                                          int timed_runs) {
    SpmvBenchmarkResult res;
    res.name = "Sparse SpMV (" + std::to_string(A.rows()) + "x" + std::to_string(A.cols()) +
               ", nnz=" + std::to_string(A.non_zeros()) +
               (transpose ? " Transposed" : " Non-Transposed") + ")";
    res.rows = A.rows();
    res.cols = A.cols();
    res.nnz = A.non_zeros();
    res.density = (A.rows() > 0 && A.cols() > 0)
                      ? static_cast<double>(A.non_zeros()) / static_cast<double>(A.rows() * A.cols())
                      : 0.0;
    res.transpose = transpose;

    const std::size_t x_len = transpose ? A.rows() : A.cols();
    const std::size_t y_len = transpose ? A.cols() : A.rows();

    if (x.size() != x_len) {
        throw std::invalid_argument("benchmark_sparse_spmv: input vector x size mismatch");
    }

    // 1. CPU Reference Benchmark
    std::vector<double> y_cpu(y_len, 0.0);
    for (int i = 0; i < warmup_runs; ++i) {
        if (transpose) {
            y_cpu = A.multiply_transpose(x);
        } else {
            y_cpu = A.multiply(x);
        }
    }
    std::vector<double> cpu_times;
    cpu_times.reserve(timed_runs);
    for (int i = 0; i < timed_runs; ++i) {
        const auto t0 = std::chrono::high_resolution_clock::now();
        if (transpose) {
            y_cpu = A.multiply_transpose(x);
        } else {
            y_cpu = A.multiply(x);
        }
        const auto t1 = std::chrono::high_resolution_clock::now();
        cpu_times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    res.cpu_median_ms = calculate_median(cpu_times);

    // Prepare CSR 32-bit int vectors
    std::vector<int> col_indices_int;
    std::vector<int> row_ptrs_int;
    convert_csr_indices_to_int(A, col_indices_int, row_ptrs_int);

    // 2. GPU Warmup
    CusparseHandle handle;
    std::vector<double> y_gpu(y_len, 0.0);
    for (int i = 0; i < warmup_runs; ++i) {
        sparse_spmv_csr(A.rows(), A.cols(), A.non_zeros(),
                        A.values().data(), col_indices_int.data(), row_ptrs_int.data(),
                        x.data(), y_gpu.data(), transpose, 1.0, 0.0, handle.get());
    }

    // 3. Isolated Timings
    const std::size_t nnz = A.non_zeros();
    CudaDeviceBuffer<int> d_row_ptrs(A.rows() + 1);
    CudaDeviceBuffer<int> d_col_indices(nnz > 0 ? nnz : 1);
    CudaDeviceBuffer<double> d_values(nnz > 0 ? nnz : 1);
    CudaDeviceBuffer<double> d_x(x_len);
    CudaDeviceBuffer<double> d_y(y_len);

    std::vector<double> h2d_times;
    std::vector<double> kernel_times;
    std::vector<double> d2h_times;
    std::vector<double> e2e_times;

    h2d_times.reserve(timed_runs);
    kernel_times.reserve(timed_runs);
    d2h_times.reserve(timed_runs);
    e2e_times.reserve(timed_runs);

    // Pre-create descriptors for kernel timing isolation
    cusparseSpMatDescr_t raw_matA = nullptr;
    if (nnz > 0) {
        VORTEX_CUSPARSE_CHECK(cusparseCreateCsr(&raw_matA,
                                               static_cast<int64_t>(A.rows()),
                                               static_cast<int64_t>(A.cols()),
                                               static_cast<int64_t>(nnz),
                                               d_row_ptrs.data(),
                                               d_col_indices.data(),
                                               d_values.data(),
                                               CUSPARSE_INDEX_32I,
                                               CUSPARSE_INDEX_32I,
                                               CUSPARSE_INDEX_BASE_ZERO,
                                               CUDA_R_64F));
    }
    CusparseSpMatWrapper matA(raw_matA);

    cusparseDnVecDescr_t raw_vecX = nullptr;
    VORTEX_CUSPARSE_CHECK(cusparseCreateDnVec(&raw_vecX, static_cast<int64_t>(x_len), d_x.data(), CUDA_R_64F));
    CusparseDnVecWrapper vecX(raw_vecX);

    cusparseDnVecDescr_t raw_vecY = nullptr;
    VORTEX_CUSPARSE_CHECK(cusparseCreateDnVec(&raw_vecY, static_cast<int64_t>(y_len), d_y.data(), CUDA_R_64F));
    CusparseDnVecWrapper vecY(raw_vecY);

    const cusparseOperation_t op = transpose ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;
    const double alpha = 1.0;
    const double beta  = 0.0;

    size_t buffer_size = 0;
    if (nnz > 0) {
        VORTEX_CUSPARSE_CHECK(cusparseSpMV_bufferSize(handle.get(),
                                                     op,
                                                     &alpha,
                                                     matA.get(),
                                                     vecX.get(),
                                                     &beta,
                                                     vecY.get(),
                                                     CUDA_R_64F,
                                                     CUSPARSE_SPMV_ALG_DEFAULT,
                                                     &buffer_size));
    }
    CudaDeviceBuffer<uint8_t> d_workspace(buffer_size);
    CudaScopedEventTimer event_timer;

    for (int i = 0; i < timed_runs; ++i) {
        const auto e2e_start = std::chrono::high_resolution_clock::now();

        // Phase 1: H2D Transfer
        const auto h2d_start = std::chrono::high_resolution_clock::now();
        d_row_ptrs.copy_from_host(row_ptrs_int.data(), A.rows() + 1);
        if (nnz > 0) {
            d_col_indices.copy_from_host(col_indices_int.data(), nnz);
            d_values.copy_from_host(A.values().data(), nnz);
        }
        d_x.copy_from_host(x.data(), x_len);
        VORTEX_CUDA_CHECK(cudaMemset(d_y.data(), 0, y_len * sizeof(double)));
        VORTEX_CUDA_CHECK(cudaDeviceSynchronize());
        const auto h2d_end = std::chrono::high_resolution_clock::now();
        h2d_times.push_back(std::chrono::duration<double, std::milli>(h2d_end - h2d_start).count());

        // Phase 2: Kernel
        if (nnz > 0) {
            event_timer.record_start();
            VORTEX_CUSPARSE_CHECK(cusparseSpMV(handle.get(),
                                               op,
                                               &alpha,
                                               matA.get(),
                                               vecX.get(),
                                               &beta,
                                               vecY.get(),
                                               CUDA_R_64F,
                                               CUSPARSE_SPMV_ALG_DEFAULT,
                                               d_workspace.data()));
            event_timer.record_stop();
            kernel_times.push_back(event_timer.elapsed_ms());
        } else {
            kernel_times.push_back(0.0);
        }

        // Phase 3: D2H Transfer
        const auto d2h_start = std::chrono::high_resolution_clock::now();
        d_y.copy_to_host(y_gpu.data(), y_len);
        VORTEX_CUDA_CHECK(cudaDeviceSynchronize());
        const auto d2h_end = std::chrono::high_resolution_clock::now();
        d2h_times.push_back(std::chrono::duration<double, std::milli>(d2h_end - d2h_start).count());

        const auto e2e_end = std::chrono::high_resolution_clock::now();
        e2e_times.push_back(std::chrono::duration<double, std::milli>(e2e_end - e2e_start).count());
    }

    res.gpu_h2d_ms    = calculate_median(h2d_times);
    res.gpu_kernel_ms = calculate_median(kernel_times);
    res.gpu_d2h_ms    = calculate_median(d2h_times);
    res.gpu_e2e_ms    = calculate_median(e2e_times);

    res.speedup_e2e     = (res.gpu_e2e_ms > 0.0)    ? (res.cpu_median_ms / res.gpu_e2e_ms)    : 0.0;
    res.speedup_compute = (res.gpu_kernel_ms > 0.0) ? (res.cpu_median_ms / res.gpu_kernel_ms) : 0.0;

    const auto err = compute_error_metrics(y_gpu.data(), y_cpu.data(), y_len);
    res.max_abs_error    = err.max_absolute_error;
    res.scaled_rel_error = err.scaled_relative_error;

    return res;
}

// ==============================================================================
// Phase 8D: Hybrid Solver Iteration Context Implementations
// ==============================================================================

// --- GpuDensePricingContext ---
GpuDensePricingContext::GpuDensePricingContext(const Matrix<double>& A_std)
    : m_(A_std.rows()), n_(A_std.cols()),
      cublas_(std::make_unique<CublasHandle>()),
      d_A_(m_ * n_), d_pi_(m_), d_r_(n_) {
    if (m_ == 0 || n_ == 0) {
        throw std::invalid_argument("GpuDensePricingContext: Matrix dimensions cannot be zero");
    }
    // Upload static constraint matrix A_std to persistent device buffer once
    d_A_.copy_from_host(A_std.data(), m_ * n_);
    h2d_transfers_ += 1;
}

GpuDensePricingContext::~GpuDensePricingContext() = default;

void GpuDensePricingContext::compute_reduced_costs(
    const std::vector<double>& c_std,
    const std::vector<double>& pi,
    std::vector<double>& reduced_costs) {

    if (pi.size() != m_) {
        throw std::invalid_argument("GpuDensePricingContext: Dual vector pi size mismatch");
    }
    if (c_std.size() != n_) {
        throw std::invalid_argument("GpuDensePricingContext: Cost vector c_std size mismatch");
    }
    if (reduced_costs.size() != n_) {
        reduced_costs.resize(n_);
    }

    // Transfer dual vector pi and initialize output buffer with c_std
    d_pi_.copy_from_host(pi.data(), m_);
    d_r_.copy_from_host(c_std.data(), n_);
    h2d_transfers_ += 2;

    // A_std is row-major m x n (cuBLAS column-major n x m, lda = n).
    // CUBLAS_OP_N evaluates: d_r = alpha * A^T * d_pi + beta * d_r
    // With alpha = -1.0, beta = 1.0 => d_r = c_std - A^T * pi.
    const double alpha = -1.0;
    const double beta  = 1.0;
    const int m_cu = static_cast<int>(n_);
    const int n_cu = static_cast<int>(m_);
    const int lda  = static_cast<int>(n_);

    VORTEX_CUBLAS_CHECK(cublasDgemv(
        cublas_->get(),
        CUBLAS_OP_N,
        m_cu,
        n_cu,
        &alpha,
        d_A_.data(),
        lda,
        d_pi_.data(),
        1,
        &beta,
        d_r_.data(),
        1));

    VORTEX_CUDA_CHECK(cudaDeviceSynchronize());
    gemv_calls_ += 1;

    // Transfer reduced costs back to host
    d_r_.copy_to_host(reduced_costs.data(), n_);
    d2h_transfers_ += 1;
}

// --- GpuSparsePricingContext ---
GpuSparsePricingContext::GpuSparsePricingContext(const CSRMatrix<double>& A_csr)
    : m_(A_csr.rows()), n_(A_csr.cols()), nnz_(A_csr.non_zeros()),
      cusparse_(std::make_unique<CusparseHandle>()),
      d_row_ptrs_(m_ + 1), d_col_indices_(nnz_ > 0 ? nnz_ : 1),
      d_values_(nnz_ > 0 ? nnz_ : 1), d_pi_(m_), d_r_(n_) {

    if (m_ == 0 || n_ == 0) {
        throw std::invalid_argument("GpuSparsePricingContext: Matrix dimensions cannot be zero");
    }

    // Convert size_t CSR pointers to 32-bit int
    std::vector<int> h_row_ptrs(m_ + 1);
    for (std::size_t i = 0; i <= m_; ++i) {
        h_row_ptrs[i] = static_cast<int>(A_csr.row_ptrs()[i]);
    }
    std::vector<int> h_col_indices(nnz_ > 0 ? nnz_ : 1, 0);
    for (std::size_t i = 0; i < nnz_; ++i) {
        h_col_indices[i] = static_cast<int>(A_csr.col_indices()[i]);
    }

    d_row_ptrs_.copy_from_host(h_row_ptrs.data(), m_ + 1);
    if (nnz_ > 0) {
        d_col_indices_.copy_from_host(h_col_indices.data(), nnz_);
        d_values_.copy_from_host(A_csr.values().data(), nnz_);
    }
    h2d_transfers_ += 3;

    if (nnz_ > 0) {
        VORTEX_CUSPARSE_CHECK(cusparseCreateCsr(
            &mat_descr_,
            static_cast<int64_t>(m_),
            static_cast<int64_t>(n_),
            static_cast<int64_t>(nnz_),
            d_row_ptrs_.data(),
            d_col_indices_.data(),
            d_values_.data(),
            CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO,
            CUDA_R_64F));

        VORTEX_CUSPARSE_CHECK(cusparseCreateDnVec(
            &vec_x_descr_,
            static_cast<int64_t>(m_),
            d_pi_.data(),
            CUDA_R_64F));

        VORTEX_CUSPARSE_CHECK(cusparseCreateDnVec(
            &vec_y_descr_,
            static_cast<int64_t>(n_),
            d_r_.data(),
            CUDA_R_64F));

        double alpha = -1.0;
        double beta  = 1.0;
        std::size_t bufferSize = 0;
        VORTEX_CUSPARSE_CHECK(cusparseSpMV_bufferSize(
            cusparse_->get(),
            CUSPARSE_OPERATION_TRANSPOSE,
            &alpha,
            mat_descr_,
            vec_x_descr_,
            &beta,
            vec_y_descr_,
            CUDA_R_64F,
            CUSPARSE_SPMV_ALG_DEFAULT,
            &bufferSize));

        if (bufferSize > 0) {
            d_buffer_.allocate(bufferSize);
        }
    }
}

GpuSparsePricingContext::~GpuSparsePricingContext() {
    if (mat_descr_ != nullptr) {
        cusparseDestroySpMat(mat_descr_);
        mat_descr_ = nullptr;
    }
    if (vec_x_descr_ != nullptr) {
        cusparseDestroyDnVec(vec_x_descr_);
        vec_x_descr_ = nullptr;
    }
    if (vec_y_descr_ != nullptr) {
        cusparseDestroyDnVec(vec_y_descr_);
        vec_y_descr_ = nullptr;
    }
}

void GpuSparsePricingContext::compute_reduced_costs(
    const std::vector<double>& c_std,
    const std::vector<double>& pi,
    std::vector<double>& reduced_costs) {

    if (pi.size() != m_) {
        throw std::invalid_argument("GpuSparsePricingContext: Dual vector pi size mismatch");
    }
    if (c_std.size() != n_) {
        throw std::invalid_argument("GpuSparsePricingContext: Cost vector c_std size mismatch");
    }
    if (reduced_costs.size() != n_) {
        reduced_costs.resize(n_);
    }

    if (nnz_ == 0) {
        reduced_costs = c_std;
        return;
    }

    d_pi_.copy_from_host(pi.data(), m_);
    d_r_.copy_from_host(c_std.data(), n_);
    h2d_transfers_ += 2;

    double alpha = -1.0;
    double beta  = 1.0;

    VORTEX_CUSPARSE_CHECK(cusparseSpMV(
        cusparse_->get(),
        CUSPARSE_OPERATION_TRANSPOSE,
        &alpha,
        mat_descr_,
        vec_x_descr_,
        &beta,
        vec_y_descr_,
        CUDA_R_64F,
        CUSPARSE_SPMV_ALG_DEFAULT,
        d_buffer_.empty() ? nullptr : d_buffer_.data()));

    VORTEX_CUDA_CHECK(cudaDeviceSynchronize());
    spmv_calls_ += 1;

    d_r_.copy_to_host(reduced_costs.data(), n_);
    d2h_transfers_ += 1;
}

// --- GpuQPIterationContext ---
GpuQPIterationContext::GpuQPIterationContext(const Matrix<double>& Q, const Matrix<double>& A)
    : n_(Q.rows()), m_(A.rows()),
      cublas_(std::make_unique<CublasHandle>()) {

    if (n_ > 0) {
        d_Q_.allocate(n_ * n_);
        d_Q_.copy_from_host(Q.data(), n_ * n_);
        d_x_n_.allocate(n_);
        d_y_n_.allocate(n_);
        h2d_transfers_ += 1;
    }
    if (m_ > 0 && n_ > 0) {
        d_A_.allocate(m_ * n_);
        d_A_.copy_from_host(A.data(), m_ * n_);
        d_x_m_.allocate(m_);
        d_y_m_.allocate(m_);
        h2d_transfers_ += 1;
    }
}

GpuQPIterationContext::~GpuQPIterationContext() = default;

void GpuQPIterationContext::compute_Qx(const std::vector<double>& x, std::vector<double>& Qx) {
    if (n_ == 0) return;
    if (x.size() != n_) {
        throw std::invalid_argument("GpuQPIterationContext::compute_Qx: Vector dimension mismatch");
    }
    if (Qx.size() != n_) {
        Qx.resize(n_);
    }

    d_x_n_.copy_from_host(x.data(), n_);
    h2d_transfers_ += 1;

    VORTEX_CUDA_CHECK(cudaMemset(d_y_n_.data(), 0, n_ * sizeof(double)));

    const double alpha = 1.0;
    const double beta  = 0.0;
    const int n_int = static_cast<int>(n_);

    // Q is symmetric n x n
    VORTEX_CUBLAS_CHECK(cublasDgemv(
        cublas_->get(),
        CUBLAS_OP_N,
        n_int,
        n_int,
        &alpha,
        d_Q_.data(),
        n_int,
        d_x_n_.data(),
        1,
        &beta,
        d_y_n_.data(),
        1));

    VORTEX_CUDA_CHECK(cudaDeviceSynchronize());
    gemv_calls_ += 1;

    d_y_n_.copy_to_host(Qx.data(), n_);
    d2h_transfers_ += 1;
}

void GpuQPIterationContext::compute_Ax(const std::vector<double>& x, std::vector<double>& Ax) {
    if (m_ == 0 || n_ == 0) return;
    if (x.size() != n_) {
        throw std::invalid_argument("GpuQPIterationContext::compute_Ax: Vector dimension mismatch");
    }
    if (Ax.size() != m_) {
        Ax.resize(m_);
    }

    d_x_n_.copy_from_host(x.data(), n_);
    h2d_transfers_ += 1;

    VORTEX_CUDA_CHECK(cudaMemset(d_y_m_.data(), 0, m_ * sizeof(double)));

    const double alpha = 1.0;
    const double beta  = 0.0;
    const int m_cu = static_cast<int>(n_);
    const int n_cu = static_cast<int>(m_);
    const int lda  = static_cast<int>(n_);

    // A is m x n row-major. CUBLAS_OP_T produces A * x
    VORTEX_CUBLAS_CHECK(cublasDgemv(
        cublas_->get(),
        CUBLAS_OP_T,
        m_cu,
        n_cu,
        &alpha,
        d_A_.data(),
        lda,
        d_x_n_.data(),
        1,
        &beta,
        d_y_m_.data(),
        1));

    VORTEX_CUDA_CHECK(cudaDeviceSynchronize());
    gemv_calls_ += 1;

    d_y_m_.copy_to_host(Ax.data(), m_);
    d2h_transfers_ += 1;
}

void GpuQPIterationContext::compute_AT_vec(const std::vector<double>& v, std::vector<double>& ATv) {
    if (m_ == 0 || n_ == 0) return;
    if (v.size() != m_) {
        throw std::invalid_argument("GpuQPIterationContext::compute_AT_vec: Vector dimension mismatch");
    }
    if (ATv.size() != n_) {
        ATv.resize(n_);
    }

    d_x_m_.copy_from_host(v.data(), m_);
    h2d_transfers_ += 1;

    VORTEX_CUDA_CHECK(cudaMemset(d_y_n_.data(), 0, n_ * sizeof(double)));

    const double alpha = 1.0;
    const double beta  = 0.0;
    const int m_cu = static_cast<int>(n_);
    const int n_cu = static_cast<int>(m_);
    const int lda  = static_cast<int>(n_);

    // A is m x n row-major. CUBLAS_OP_N produces A^T * v
    VORTEX_CUBLAS_CHECK(cublasDgemv(
        cublas_->get(),
        CUBLAS_OP_N,
        m_cu,
        n_cu,
        &alpha,
        d_A_.data(),
        lda,
        d_x_m_.data(),
        1,
        &beta,
        d_y_n_.data(),
        1));

    VORTEX_CUDA_CHECK(cudaDeviceSynchronize());
    gemv_calls_ += 1;

    d_y_n_.copy_to_host(ATv.data(), n_);
    d2h_transfers_ += 1;
}

} // namespace vortex::cuda
