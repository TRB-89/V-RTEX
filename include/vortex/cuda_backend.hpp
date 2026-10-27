#pragma once

#include <cstddef>
#include <string>
#include <vector>
#include <stdexcept>
#include <memory>
#include <utility>
#include <cmath>
#include <algorithm>

#include "vortex/matrix.hpp"
#include "vortex/sparse_matrix.hpp"

#if defined(VORTEX_ENABLE_CUDA) || defined(__CUDACC__)
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#else
using cublasHandle_t = void*;
using cusparseHandle_t = void*;
#endif

namespace vortex::cuda {

// ==============================================================================
// CUDA Device Information
// ==============================================================================
struct CudaDeviceInfo {
    int device_id{-1};
    std::string name;
    int major{0};
    int minor{0};
    std::size_t total_memory_bytes{0};
    std::size_t free_memory_bytes{0};
    int multiprocessor_count{0};
    int max_threads_per_block{0};
    int warp_size{32};
    bool is_available{false};
};

#if defined(VORTEX_ENABLE_CUDA) || defined(__CUDACC__)
// Check if a CUDA-capable device is accessible
bool is_cuda_available() noexcept;

// Query properties of the specified device (default 0)
CudaDeviceInfo get_cuda_device_info(int device_id = 0);
#else
inline bool is_cuda_available() noexcept { return false; }
inline CudaDeviceInfo get_cuda_device_info(int device_id = 0) {
    (void)device_id;
    return CudaDeviceInfo{};
}
#endif

// ==============================================================================
// Low-level CUDA Memory Primitives (Defined in cuda_backend.cu)
// ==============================================================================
void* cuda_allocate_device_bytes(std::size_t bytes);
void cuda_free_device_memory(void* ptr) noexcept;
void cuda_copy_host_to_device(void* dst, const void* src, std::size_t bytes);
void cuda_copy_device_to_host(void* dst, const void* src, std::size_t bytes);

// ==============================================================================
// RAII Device Buffer Template
// ==============================================================================
template <typename T>
class CudaDeviceBuffer {
public:
    CudaDeviceBuffer() noexcept : ptr_(nullptr), count_(0) {}

    explicit CudaDeviceBuffer(std::size_t count) : ptr_(nullptr), count_(0) {
        allocate(count);
    }

    ~CudaDeviceBuffer() {
        reset();
    }

    // Non-copyable to prevent double-free
    CudaDeviceBuffer(const CudaDeviceBuffer&) = delete;
    CudaDeviceBuffer& operator=(const CudaDeviceBuffer&) = delete;

    // Move-constructible
    CudaDeviceBuffer(CudaDeviceBuffer&& other) noexcept
        : ptr_(other.ptr_), count_(other.count_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }

    // Move-assignable
    CudaDeviceBuffer& operator=(CudaDeviceBuffer&& other) noexcept {
        if (this != &other) {
            reset();
            ptr_ = other.ptr_;
            count_ = other.count_;
            other.ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    void allocate(std::size_t count) {
        if (count == 0) {
            reset();
            return;
        }
        if (count == count_ && ptr_ != nullptr) {
            return;
        }
        reset();
        ptr_ = static_cast<T*>(cuda_allocate_device_bytes(count * sizeof(T)));
        count_ = count;
    }

    void reset() noexcept {
        if (ptr_ != nullptr) {
            cuda_free_device_memory(ptr_);
            ptr_ = nullptr;
        }
        count_ = 0;
    }

    void copy_from_host(const T* host_src, std::size_t count) {
        if (count == 0) return;
        if (host_src == nullptr) {
            throw std::invalid_argument("CudaDeviceBuffer::copy_from_host: host_src is null");
        }
        if (count > count_) {
            allocate(count);
        }
        cuda_copy_host_to_device(ptr_, host_src, count * sizeof(T));
    }

    void copy_to_host(T* host_dst, std::size_t count) const {
        if (count == 0) return;
        if (host_dst == nullptr) {
            throw std::invalid_argument("CudaDeviceBuffer::copy_to_host: host_dst is null");
        }
        if (count > count_ || ptr_ == nullptr) {
            throw std::runtime_error("CudaDeviceBuffer::copy_to_host: requested count exceeds allocated device memory");
        }
        cuda_copy_device_to_host(host_dst, ptr_, count * sizeof(T));
    }

    [[nodiscard]] T* data() noexcept { return ptr_; }
    [[nodiscard]] const T* data() const noexcept { return ptr_; }
    [[nodiscard]] std::size_t size() const noexcept { return count_; }
    [[nodiscard]] std::size_t bytes() const noexcept { return count_ * sizeof(T); }
    [[nodiscard]] bool empty() const noexcept { return count_ == 0; }

private:
    T* ptr_{nullptr};
    std::size_t count_{0};
};

// ==============================================================================
// Module C: Safe RAII cuBLAS and cuSPARSE Handle Ownership
// ==============================================================================
class CublasHandle {
public:
    CublasHandle();
    ~CublasHandle();

    CublasHandle(const CublasHandle&) = delete;
    CublasHandle& operator=(const CublasHandle&) = delete;

    CublasHandle(CublasHandle&& other) noexcept;
    CublasHandle& operator=(CublasHandle&& other) noexcept;

    [[nodiscard]] cublasHandle_t get() const noexcept { return handle_; }
    operator cublasHandle_t() const noexcept { return handle_; }
    [[nodiscard]] bool is_valid() const noexcept { return handle_ != nullptr; }

private:
    cublasHandle_t handle_{nullptr};
};

class CusparseHandle {
public:
    CusparseHandle();
    ~CusparseHandle();

    CusparseHandle(const CusparseHandle&) = delete;
    CusparseHandle& operator=(const CusparseHandle&) = delete;

    CusparseHandle(CusparseHandle&& other) noexcept;
    CusparseHandle& operator=(CusparseHandle&& other) noexcept;

    [[nodiscard]] cusparseHandle_t get() const noexcept { return handle_; }
    operator cusparseHandle_t() const noexcept { return handle_; }
    [[nodiscard]] bool is_valid() const noexcept { return handle_ != nullptr; }

private:
    cusparseHandle_t handle_{nullptr};
};

// ==============================================================================
// Vector Addition CUDA Operations (Phase 8A)
// ==============================================================================
struct VectorAddExecutionReport {
    bool success{false};
    std::string device_name;
    std::size_t vector_size{0};
    int block_size{256};
    int grid_size{0};
    double max_numerical_error{0.0};
    std::string status_message;
};

void vector_add(const float* a, const float* b, float* c, std::size_t n, int block_size = 256);
void vector_add(const double* a, const double* b, double* c, std::size_t n, int block_size = 256);

std::vector<float> vector_add(const std::vector<float>& a, const std::vector<float>& b, int block_size = 256);
std::vector<double> vector_add(const std::vector<double>& a, const std::vector<double>& b, int block_size = 256);

VectorAddExecutionReport execute_vector_add_verified(const std::vector<double>& a, const std::vector<double>& b, int block_size = 256);

// ==============================================================================
// Numerical Error Metrics Structure
// ==============================================================================
struct NumericalErrorMetrics {
    double max_absolute_error{0.0};
    double scaled_relative_error{0.0};
    std::size_t mismatch_count{0};
};

NumericalErrorMetrics compute_error_metrics(const double* actual, const double* expected, std::size_t n, double tol = 1e-12);
NumericalErrorMetrics compute_error_metrics(const float* actual, const float* expected, std::size_t n, float tol = 1e-5f);

// ==============================================================================
// Phase 8C: Comprehensive Componentwise Accuracy Validation Framework
// ==============================================================================
struct ComponentwiseAccuracyReport {
    bool passed{false};
    double max_absolute_error{0.0};
    double scaled_relative_error{0.0};
    double infinity_norm_residual{0.0};
    double max_componentwise_ratio{0.0}; // max_i ( |actual[i] - expected[i]| / bound[i] )
    std::size_t mismatch_count{0};
    std::size_t total_elements{0};
    std::size_t worst_row{0};
    double worst_actual{0.0};
    double worst_expected{0.0};
    double worst_abs_err{0.0};
    double worst_scale{0.0};
    double worst_bound{0.0};
    double atol{1e-12};
    double rtol{1e-12};
    std::string diagnostic_summary;
};

// Evaluates defensible componentwise bound: |actual[i] - expected[i]| <= atol + rtol * scale[i]
ComponentwiseAccuracyReport evaluate_componentwise_accuracy(
    const double* actual,
    const double* expected,
    const double* componentwise_scale,
    std::size_t n,
    double atol = 1e-12,
    double rtol = 1e-12);

ComponentwiseAccuracyReport evaluate_componentwise_accuracy(
    const float* actual,
    const float* expected,
    const float* componentwise_scale,
    std::size_t n,
    float atol = 1e-5f,
    float rtol = 1e-5f);

// Validates Dense GEMV against CPU reference with defensible componentwise bound:
// scale[i] = |alpha| * sum_j (|A_ij| * |x_j|) + |beta| * |y_initial_i|
ComponentwiseAccuracyReport validate_dense_gemv_accuracy(
    const Matrix<double>& A,
    const std::vector<double>& x,
    bool transpose = false,
    double alpha = 1.0,
    double beta = 0.0,
    const std::vector<double>& y_initial = {},
    double atol = 1e-12,
    double rtol = 1e-12);

// Validates Sparse SpMV against CPU reference with defensible componentwise bound:
// scale[i] = |alpha| * sum_{j in row i} (|val_ij| * |x_j|) + |beta| * |y_initial_i|
ComponentwiseAccuracyReport validate_sparse_spmv_accuracy(
    const CSRMatrix<double>& A,
    const std::vector<double>& x,
    bool transpose = false,
    double alpha = 1.0,
    double beta = 0.0,
    const std::vector<double>& y_initial = {},
    double atol = 1e-12,
    double rtol = 1e-12);

// ==============================================================================
// Module A: Dense Matrix-Vector Multiplication (cuBLAS GEMV)
// ==============================================================================
// Computes y = alpha * op(A) * x + beta * y
// A is M x N row-major matrix.
// If transpose == false:
//   op(A) = A (dimensions M x N). x must have length N, y must have length M.
// If transpose == true:
//   op(A) = A^T (dimensions N x M). x must have length M, y must have length N.
void dense_gemv(const double* A, std::size_t rows, std::size_t cols,
                const double* x, double* y,
                bool transpose = false,
                double alpha = 1.0, double beta = 0.0,
                cublasHandle_t handle = nullptr);

void dense_gemv(const float* A, std::size_t rows, std::size_t cols,
                const float* x, float* y,
                bool transpose = false,
                float alpha = 1.0f, float beta = 0.0f,
                cublasHandle_t handle = nullptr);

std::vector<double> dense_gemv(const Matrix<double>& A,
                               const std::vector<double>& x,
                               bool transpose = false,
                               double alpha = 1.0, double beta = 0.0,
                               cublasHandle_t handle = nullptr);

std::vector<float> dense_gemv(const Matrix<float>& A,
                              const std::vector<float>& x,
                              bool transpose = false,
                              float alpha = 1.0f, float beta = 0.0f,
                              cublasHandle_t handle = nullptr);

void dense_gemv(const Matrix<double>& A,
                const std::vector<double>& x,
                std::vector<double>& y,
                bool transpose = false,
                double alpha = 1.0, double beta = 0.0,
                cublasHandle_t handle = nullptr);

void dense_gemv(const Matrix<float>& A,
                const std::vector<float>& x,
                std::vector<float>& y,
                bool transpose = false,
                float alpha = 1.0f, float beta = 0.0f,
                cublasHandle_t handle = nullptr);

// ==============================================================================
// Module B: Sparse Matrix-Vector Multiplication (cuSPARSE SpMV)
// ==============================================================================
// Computes y = alpha * op(A) * x + beta * y using cuSPARSE CSR representation.
// A is M x N sparse matrix in CSR format (0-based indexing).
// If transpose == false:
//   op(A) = A (dimensions M x N). x must have length N, y must have length M.
// If transpose == true:
//   op(A) = A^T (dimensions N x M). x must have length M, y must have length N.
void sparse_spmv_csr(std::size_t rows, std::size_t cols, std::size_t nnz,
                     const double* values, const int* col_indices, const int* row_ptrs,
                     const double* x, double* y,
                     bool transpose = false,
                     double alpha = 1.0, double beta = 0.0,
                     cusparseHandle_t handle = nullptr);

void sparse_spmv_csr(std::size_t rows, std::size_t cols, std::size_t nnz,
                     const float* values, const int* col_indices, const int* row_ptrs,
                     const float* x, float* y,
                     bool transpose = false,
                     float alpha = 1.0f, float beta = 0.0f,
                     cusparseHandle_t handle = nullptr);

std::vector<double> sparse_spmv(const CSRMatrix<double>& A,
                                const std::vector<double>& x,
                                bool transpose = false,
                                double alpha = 1.0, double beta = 0.0,
                                cusparseHandle_t handle = nullptr);

std::vector<float> sparse_spmv(const CSRMatrix<float>& A,
                               const std::vector<float>& x,
                               bool transpose = false,
                               float alpha = 1.0f, float beta = 0.0f,
                               cusparseHandle_t handle = nullptr);

void sparse_spmv(const CSRMatrix<double>& A,
                 const std::vector<double>& x,
                 std::vector<double>& y,
                 bool transpose = false,
                 double alpha = 1.0, double beta = 0.0,
                 cusparseHandle_t handle = nullptr);

void sparse_spmv(const CSRMatrix<float>& A,
                 const std::vector<float>& x,
                 std::vector<float>& y,
                 bool transpose = false,
                 float alpha = 1.0f, float beta = 0.0f,
                 cusparseHandle_t handle = nullptr);

// ==============================================================================
// Module E: Benchmarking Data Structures and Routines
// ==============================================================================
struct GemvBenchmarkResult {
    std::string name;
    std::size_t rows{0};
    std::size_t cols{0};
    bool transpose{false};
    double cpu_median_ms{0.0};
    double gpu_h2d_ms{0.0};
    double gpu_kernel_ms{0.0};
    double gpu_d2h_ms{0.0};
    double gpu_e2e_ms{0.0};
    double speedup_e2e{0.0};
    double speedup_compute{0.0};
    double max_abs_error{0.0};
    double scaled_rel_error{0.0};
};

struct SpmvBenchmarkResult {
    std::string name;
    std::size_t rows{0};
    std::size_t cols{0};
    std::size_t nnz{0};
    double density{0.0};
    bool transpose{false};
    double cpu_median_ms{0.0};
    double gpu_h2d_ms{0.0};
    double gpu_kernel_ms{0.0};
    double gpu_d2h_ms{0.0};
    double gpu_e2e_ms{0.0};
    double speedup_e2e{0.0};
    double speedup_compute{0.0};
    double max_abs_error{0.0};
    double scaled_rel_error{0.0};
};

GemvBenchmarkResult benchmark_dense_gemv(const Matrix<double>& A,
                                         const std::vector<double>& x,
                                         bool transpose = false,
                                         int warmup_runs = 5,
                                         int timed_runs = 20);

SpmvBenchmarkResult benchmark_sparse_spmv(const CSRMatrix<double>& A,
                                          const std::vector<double>& x,
                                          bool transpose = false,
                                          int warmup_runs = 5,
                                          int timed_runs = 20);

// ==============================================================================
// Phase 8D: Hybrid Solver Iteration Contexts
// ==============================================================================
#if defined(VORTEX_ENABLE_CUDA) || defined(__CUDACC__)

class GpuDensePricingContext {
public:
    explicit GpuDensePricingContext(const Matrix<double>& A_std);
    ~GpuDensePricingContext();

    GpuDensePricingContext(const GpuDensePricingContext&) = delete;
    GpuDensePricingContext& operator=(const GpuDensePricingContext&) = delete;

    // Computes reduced costs: reduced_costs = c_std - A_std^T * pi
    void compute_reduced_costs(const std::vector<double>& c_std,
                               const std::vector<double>& pi,
                               std::vector<double>& reduced_costs);

    [[nodiscard]] std::size_t gemv_calls() const noexcept { return gemv_calls_; }
    [[nodiscard]] std::size_t h2d_transfers() const noexcept { return h2d_transfers_; }
    [[nodiscard]] std::size_t d2h_transfers() const noexcept { return d2h_transfers_; }

private:
    std::size_t m_{0};
    std::size_t n_{0};
    std::unique_ptr<CublasHandle> cublas_;
    CudaDeviceBuffer<double> d_A_;
    CudaDeviceBuffer<double> d_pi_;
    CudaDeviceBuffer<double> d_r_;
    std::size_t gemv_calls_{0};
    std::size_t h2d_transfers_{0};
    std::size_t d2h_transfers_{0};
};

class GpuSparsePricingContext {
public:
    explicit GpuSparsePricingContext(const CSRMatrix<double>& A_csr);
    ~GpuSparsePricingContext();

    GpuSparsePricingContext(const GpuSparsePricingContext&) = delete;
    GpuSparsePricingContext& operator=(const GpuSparsePricingContext&) = delete;

    // Computes reduced costs: reduced_costs = c_std - A_std^T * pi
    void compute_reduced_costs(const std::vector<double>& c_std,
                               const std::vector<double>& pi,
                               std::vector<double>& reduced_costs);

    [[nodiscard]] std::size_t spmv_calls() const noexcept { return spmv_calls_; }
    [[nodiscard]] std::size_t h2d_transfers() const noexcept { return h2d_transfers_; }
    [[nodiscard]] std::size_t d2h_transfers() const noexcept { return d2h_transfers_; }

private:
    std::size_t m_{0};
    std::size_t n_{0};
    std::size_t nnz_{0};
    std::unique_ptr<CusparseHandle> cusparse_;
    CudaDeviceBuffer<int> d_row_ptrs_;
    CudaDeviceBuffer<int> d_col_indices_;
    CudaDeviceBuffer<double> d_values_;
    CudaDeviceBuffer<double> d_pi_;
    CudaDeviceBuffer<double> d_r_;
    CudaDeviceBuffer<char> d_buffer_;
    cusparseSpMatDescr_t mat_descr_{nullptr};
    cusparseDnVecDescr_t vec_x_descr_{nullptr};
    cusparseDnVecDescr_t vec_y_descr_{nullptr};
    std::size_t spmv_calls_{0};
    std::size_t h2d_transfers_{0};
    std::size_t d2h_transfers_{0};
};

class GpuQPIterationContext {
public:
    GpuQPIterationContext(const Matrix<double>& Q, const Matrix<double>& A);
    ~GpuQPIterationContext();

    GpuQPIterationContext(const GpuQPIterationContext&) = delete;
    GpuQPIterationContext& operator=(const GpuQPIterationContext&) = delete;

    // Computes Q * x
    void compute_Qx(const std::vector<double>& x, std::vector<double>& Qx);

    // Computes A * x
    void compute_Ax(const std::vector<double>& x, std::vector<double>& Ax);

    // Computes A^T * v
    void compute_AT_vec(const std::vector<double>& v, std::vector<double>& ATv);

    [[nodiscard]] std::size_t gemv_calls() const noexcept { return gemv_calls_; }
    [[nodiscard]] std::size_t h2d_transfers() const noexcept { return h2d_transfers_; }
    [[nodiscard]] std::size_t d2h_transfers() const noexcept { return d2h_transfers_; }

private:
    std::size_t n_{0};
    std::size_t m_{0};
    std::unique_ptr<CublasHandle> cublas_;
    CudaDeviceBuffer<double> d_Q_;
    CudaDeviceBuffer<double> d_A_;
    CudaDeviceBuffer<double> d_x_n_;
    CudaDeviceBuffer<double> d_y_n_;
    CudaDeviceBuffer<double> d_x_m_;
    CudaDeviceBuffer<double> d_y_m_;
    std::size_t gemv_calls_{0};
    std::size_t h2d_transfers_{0};
    std::size_t d2h_transfers_{0};
};

#endif

} // namespace vortex::cuda
