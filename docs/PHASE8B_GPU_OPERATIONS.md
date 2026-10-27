# VØRTEX Phase 8B: GPU Matrix Operations Documentation
## cuBLAS Dense GEMV & cuSPARSE Generic CSR SpMV Foundations

---

## 1. Executive Summary & Architectural Overview

Phase 8B expands the VØRTEX optimization engine from basic scalar vector-addition GPU acceleration into high-throughput, hardware-accelerated linear algebraic kernels for mathematical optimization:
1. **Module A — Dense Matrix-Vector Multiplication (GEMV)**: Powered by NVIDIA cuBLAS (`cublasDgemv` for double precision, `cublasSgemv` for single precision), featuring an implicit zero-copy row-major to column-major transposition mapping.
2. **Module B — Sparse Matrix-Vector Multiplication (SpMV)**: Powered by NVIDIA cuSPARSE Generic API (`cusparseSpMV`), directly consuming VØRTEX Compressed Sparse Row (`CSRMatrix<T>`) representations with automatic workspace buffer management and 32-bit index translation.
3. **Module C — Resource Management & Exception Safety**: RAII device memory allocation (`CudaDeviceBuffer<T>`), explicit handle lifecycle management (`CublasHandle`, `CusparseHandle`), device query primitives, and strict out-of-memory verification.
4. **Module D — Numerical Validation**: Precision certification against CPU reference implementations across rectangular, square, ill-conditioned, catastrophic cancellation, empty-row, and degenerate zero-matrix structures.
5. **Module E — Benchmarking Profiling**: Isolated phase measurement separating Host-to-Device (H2D) PCIe transfer, GPU kernel computation, Device-to-Host (D2H) transfer, and total End-to-End (E2E) elapsed time.

---

## 2. Mathematical Formulations & Storage Layouts

### 2.1 Dense GEMV Formulation & Storage Mapping

Dense matrix-vector multiplication computes the linear combination:
$$y = \alpha \cdot \text{op}(A) \cdot x + \beta \cdot y$$
where $\alpha, \beta \in \mathbb{R}$ are scalar multipliers, $A \in \mathbb{R}^{M \times N}$, and:
- If $\text{op}(A) = A$ (non-transpose): $x \in \mathbb{R}^N, y \in \mathbb{R}^M$.
- If $\text{op}(A) = A^T$ (transpose): $x \in \mathbb{R}^M, y \in \mathbb{R}^N$.

#### The Row-Major vs. Column-Major Challenge
- **C++ CPU Memory**: VØRTEX stores dense matrices in contiguous **row-major order** ($M \times N$ entries). The element at row $i$ and column $j$ is located at memory offset:
  $$\text{Offset}_{\text{row}}(i, j) = i \cdot N + j$$
- **cuBLAS Convention**: Fortran/cuBLAS natively assumes **column-major order**. A matrix $C$ of dimensions $m_{\text{cu}} \times n_{\text{cu}}$ with leading dimension $\text{lda} = m_{\text{cu}}$ locates element $(r, c)$ at:
  $$\text{Offset}_{\text{col}}(r, c) = c \cdot \text{lda} + r$$

#### Mathematical Zero-Copy Transposition Mapping
Notice that if we interpret the contiguous flat buffer $A$ as a column-major matrix $C$:
$$\text{Offset}_{\text{row}}(i, j) = i \cdot N + j \equiv j \cdot 1 + i \cdot N = \text{Offset}_{\text{col}}(j, i)$$
where $m_{\text{cu}} = N = \text{cols}$, $n_{\text{cu}} = M = \text{rows}$, and $\text{lda} = N = \text{cols}$.
In other words, the flat buffer $A$ viewed in column-major order mathematically represents $C = A^T$!

Consequently:
1. **To compute $y = A x$ (Non-Transposed)**:
   Since $A = C^T$, we have $A x = C^T x$.
   We instruct cuBLAS to execute transposition on $C$:
   $$\text{cuBLAS Call:} \quad \text{trans} = \texttt{CUBLAS\_OP\_T}, \quad m_{\text{cu}} = \text{cols}, \quad n_{\text{cu}} = \text{rows}, \quad \text{lda} = \text{cols}$$
2. **To compute $y = A^T x$ (Transposed)**:
   Since $A^T = C$, we have $A^T x = C x$.
   We instruct cuBLAS to execute non-transposition on $C$:
   $$\text{cuBLAS Call:} \quad \text{trans} = \texttt{CUBLAS\_OP\_N}, \quad m_{\text{cu}} = \text{cols}, \quad n_{\text{cu}} = \text{rows}, \quad \text{lda} = \text{cols}$$

**Key Benefit**: Zero CPU memory shuffling, zero GPU transposing kernels, and zero temporary buffer allocations.

---

### 2.2 Sparse SpMV Formulation & cuSPARSE Generic API

Sparse matrix-vector multiplication computes:
$$y = \alpha \cdot \text{op}(A) \cdot x + \beta \cdot y$$
where $A$ is stored in **Compressed Sparse Row (CSR)** format with:
- `values`: Array of non-zero numerical values of length $\text{nnz}$.
- `col_indices`: Array of column coordinates of length $\text{nnz}$.
- `row_ptrs`: Array of row start/end offsets of length $M + 1$, where $\text{row\_ptrs}[0] = 0$ and $\text{row\_ptrs}[M] = \text{nnz}$.

#### cuSPARSE Generic Pipeline
VØRTEX uses the modern cuSPARSE generic API rather than deprecated legacy routines (`cusparseDcsrmv`):
1. **Sparse Matrix Descriptor**: Created via `cusparseCreateCsr()` using `CUSPARSE_INDEX_32I` (32-bit signed integer indices) and `CUDA_R_64F` (double precision) or `CUDA_R_32F` (single precision).
2. **Dense Vector Descriptors**: Created via `cusparseCreateDnVec()` wrapping input $x$ and output $y$.
3. **Workspace Query**: Calls `cusparseSpMV_bufferSize()` with algorithm `CUSPARSE_SPMV_ALG_DEFAULT` to determine device memory requirements.
4. **Execution**: Allocates dynamic workspace buffer via RAII `CudaDeviceBuffer<uint8_t>` and executes `cusparseSpMV()`.
5. **Zero-Matrix Safeguard**: If $\text{nnz} = 0$, cuSPARSE calls are bypassed completely, setting $y = \beta \cdot y$ directly on host/device to avoid driver errors.

---

## 3. Public API Specification

The GPU matrix operations API resides in namespace `vortex::cuda` in header `include/vortex/cuda_backend.hpp`.

### 3.1 Dense Matrix-Vector API
```cpp
// Raw pointer API (Double Precision)
void dense_gemv(const double* A, std::size_t rows, std::size_t cols,
                const double* x, double* y,
                bool transpose = false,
                double alpha = 1.0, double beta = 0.0,
                cublasHandle_t handle = nullptr);

// Raw pointer API (Single Precision)
void dense_gemv(const float* A, std::size_t rows, std::size_t cols,
                const float* x, float* y,
                bool transpose = false,
                float alpha = 1.0f, float beta = 0.0f,
                cublasHandle_t handle = nullptr);

// High-level VØRTEX Matrix<T> Overloads
std::vector<double> dense_gemv(const Matrix<double>& A, const std::vector<double>& x,
                               bool transpose = false, double alpha = 1.0, double beta = 0.0,
                               cublasHandle_t handle = nullptr);

void dense_gemv(const Matrix<double>& A, const std::vector<double>& x, std::vector<double>& y,
                bool transpose = false, double alpha = 1.0, double beta = 0.0,
                cublasHandle_t handle = nullptr);
```

### 3.2 Sparse Matrix-Vector API
```cpp
// Raw CSR pointer API (Double Precision)
void sparse_spmv_csr(std::size_t rows, std::size_t cols, std::size_t nnz,
                     const double* values, const int* col_indices, const int* row_ptrs,
                     const double* x, double* y,
                     bool transpose = false,
                     double alpha = 1.0, double beta = 0.0,
                     cusparseHandle_t handle = nullptr);

// High-level VØRTEX CSRMatrix<T> Overloads
std::vector<double> sparse_spmv(const CSRMatrix<double>& A, const std::vector<double>& x,
                                bool transpose = false, double alpha = 1.0, double beta = 0.0,
                                cusparseHandle_t handle = nullptr);

void sparse_spmv(const CSRMatrix<double>& A, const std::vector<double>& x, std::vector<double>& y,
                 bool transpose = false, double alpha = 1.0, double beta = 0.0,
                 cusparseHandle_t handle = nullptr);
```

### 3.3 RAII Handles & Error Metrics
```cpp
// RAII Handles managing cuBLAS and cuSPARSE context lifetimes
class CublasHandle;   // wraps cublasHandle_t (cublasCreate / cublasDestroy)
class CusparseHandle; // wraps cusparseHandle_t (cusparseCreate / cusparseDestroy)

// Numerical validation metric structure
struct NumericalErrorMetrics {
    double max_absolute_error{0.0};
    double scaled_relative_error{0.0};
    std::size_t mismatch_count{0};
};

NumericalErrorMetrics compute_error_metrics(const double* actual, const double* expected,
                                           std::size_t n, double tol = 1e-12);
```

---

## 4. Resource Management & Exception Safety

1. **RAII Device Buffers (`CudaDeviceBuffer<T>`)**:
   - Manages device memory lifetime via `cudaMalloc` and `cudaFree`.
   - Deleted copy constructor and copy assignment prevents accidental double-free.
   - Non-throwing move constructor and move assignment transfers ownership cleanly.
2. **Handle Ownership (`CublasHandle` / `CusparseHandle`)**:
   - Manages library context allocation and automatic teardown upon stack unwinding.
   - Enables callers to reuse persistent handles across successive solver iterations, eliminating handle creation overhead (~20-50 ms).
3. **Out-of-Memory Safeguards**:
   - Before device allocation, queries available global memory via `cudaMemGetInfo()`. If the requested allocation exceeds free VRAM, raises descriptive `std::runtime_error` rather than crashing the driver.
4. **Input Sanitization**:
   - Validates dimension compatibility ($x$ length matches matrix columns or rows for transpose).
   - Scans inputs with `std::isfinite()`, throwing descriptive `std::invalid_argument` upon encountering `NaN` or `Inf`.
   - Validates CSR monotonic row pointers ($row\_ptrs[r] \le row\_ptrs[r+1]$), $row\_ptrs[0] = 0$, $row\_ptrs[M] = \text{nnz}$, and column index bounds ($0 \le col < N$).

---

## 5. Numerical Validation & Error Analysis

Tests executed on **NVIDIA GeForce RTX 3050 Laptop GPU (4095 MB VRAM, Compute Capability 8.6, CUDA 13.4, MSVC x64)**:

### 5.1 Error Formulas
- **Maximum Absolute Error**:
  $$\|y_{\text{gpu}} - y_{\text{cpu}}\|_\infty = \max_{i} |y_{\text{gpu}}[i] - y_{\text{cpu}}[i]|$$
- **Scaled Relative Error**:
  $$\epsilon_{\text{rel}} = \frac{\|y_{\text{gpu}} - y_{\text{cpu}}\|_\infty}{\max(1.0, \|y_{\text{cpu}}\|_\infty)}$$

### 5.2 Summary of Test Results

| Test Category | Configuration / Shape | Precision | Max Abs Error | Scaled Rel Error | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Dense GEMV Core** | $1 \times 1, 1 \times 64, 64 \times 1$ | Double | $0.00 \times 10^0$ | $0.00 \times 10^0$ | **PASS** |
| **Dense GEMV Core** | Square $4 \times 4, 67 \times 67, 128 \times 128$ | Double | $< 1.0 \times 10^{-14}$ | $< 1.0 \times 10^{-15}$ | **PASS** |
| **Dense GEMV Core** | Tall ($300 \times 50$) & Wide ($50 \times 300$) | Double | $< 1.0 \times 10^{-14}$ | $< 1.0 \times 10^{-15}$ | **PASS** |
| **Dense Transpose** | Tall ($300 \times 50$) & Wide ($50 \times 300$) | Double | $< 1.0 \times 10^{-14}$ | $< 1.0 \times 10^{-15}$ | **PASS** |
| **Dense Alpha/Beta** | $\alpha = 2.5, \beta = -1.5$ ($40 \times 60$) | Double | $1.42 \times 10^{-14}$ | $2.31 \times 10^{-15}$ | **PASS** |
| **Dense Hilbert** | Ill-Conditioned $H_{ij} = \frac{1}{i+j+1}$ ($15 \times 15$) | Double | $2.22 \times 10^{-16}$ | $2.22 \times 10^{-16}$ | **PASS** |
| **Dense Subnormal** | Diagonal $10^{-16}$ | Double | $0.00 \times 10^0$ | $0.00 \times 10^0$ | **PASS** |
| **Dense Cancellation**| Alternating $\pm 10^8 + \delta$ ($100 \times 100$) | Double | $8.94 \times 10^{-8}$ | $6.02 \times 10^{-8}$ | **PASS** |
| **Dense Float Path** | Square & Rectangular ($80 \times 60$) | Float | $1.19 \times 10^{-6}$ | $2.41 \times 10^{-7}$ | **PASS** |
| **Sparse SpMV Core** | Square ($100 \times 100$), Tall ($250 \times 40$) | Double | $< 1.0 \times 10^{-14}$ | $< 1.0 \times 10^{-15}$ | **PASS** |
| **Sparse SpMV Core** | Wide ($40 \times 250$), Large ($500 \times 500$) | Double | $< 1.0 \times 10^{-14}$ | $< 1.0 \times 10^{-15}$ | **PASS** |
| **Sparse Transpose** | All Core Shapes | Double | $< 1.0 \times 10^{-14}$ | $< 1.0 \times 10^{-15}$ | **PASS** |
| **Sparse Alpha/Beta** | $\alpha = -1.75, \beta = 2.25$ ($50 \times 80$) | Double | $7.11 \times 10^{-15}$ | $1.12 \times 10^{-15}$ | **PASS** |
| **Sparse Zero Matrix**| $10 \times 15$ with $\text{nnz} = 0$ | Double | $0.00 \times 10^0$ | $0.00 \times 10^0$ | **PASS** |
| **Sparse Empty Rows** | Interleaved zero rows | Double | $0.00 \times 10^0$ | $0.00 \times 10^0$ | **PASS** |
| **Sparse Float Path** | Rectangular ($75 \times 45, 10\%$ density) | Float | $2.38 \times 10^{-6}$ | $3.51 \times 10^{-7}$ | **PASS** |

### 5.3 Mathematical Justification for Cancellation Tolerances
In the catastrophic cancellation benchmark ($A_{ij} \approx \pm 10^8$ alternating, $N = 100$ terms):
- **CPU sequential summation** adds pairs $(10^8 - 10^8 + \delta) = \delta$ sequentially.
- **GPU cuBLAS parallel tree reduction** accumulates intermediate sums across SIMD lanes of magnitude $10^9$ before subtracting.
- In IEEE-754 double precision ($\epsilon_{\text{mach}} \approx 2.22 \times 10^{-16}$), a sum of order $10^9$ has unit in the last place (ulp) $10^9 \times 2.22 \times 10^{-16} \approx 2.22 \times 10^{-7}$.
- Standard numerical analysis bounds parallel vs. sequential summation by:
  $$|y_{\text{gpu}} - y_{\text{cpu}}| \le \mathcal{O}(N \cdot \epsilon_{\text{mach}} \cdot \max |A_{ij}|) = 100 \times 2.22 \times 10^{-16} \times 10^8 = 2.22 \times 10^{-6}$$
- The observed maximum absolute error of $8.94 \times 10^{-8}$ is well below this theoretical floating-point bound.

---

## 6. Benchmarking & Performance Profiling

Timings were measured with 5 warm-up iterations followed by 20 timed measurement cycles using `std::chrono::high_resolution_clock` and CUDA events (`cudaEventElapsedTime`), reporting **median values**:

### 6.1 Dense GEMV Benchmark ($1000 \times 1000$ Double Precision, 8 MB Data)

| Metric | Non-Transpose ($y = A x$) | Transpose ($y = A^T x$) |
| :--- | :--- | :--- |
| **CPU Median** | $0.7150\text{ ms}$ | $4.0956\text{ ms}$ |
| **GPU Host-to-Device (H2D)** | $2.4805\text{ ms}$ | $2.4805\text{ ms}$ |
| **GPU Kernel (`cublasDgemv`)** | **$0.4301\text{ ms}$** | **$0.4260\text{ ms}$** |
| **GPU Device-to-Host (D2H)** | $0.0166\text{ ms}$ | $0.0166\text{ ms}$ |
| **GPU End-to-End (E2E)** | $2.9390\text{ ms}$ | $2.9664\text{ ms}$ |
| **Pure Compute Speedup** | **$1.66\times$** | **$9.61\times$** |
| **End-to-End Speedup (with PCIe)**| $0.24\times$ | $1.38\times$ |
| **Max Absolute Error** | $1.63 \times 10^{-13}$ | $1.63 \times 10^{-13}$ |
| **Scaled Relative Error** | $2.55 \times 10^{-15}$ | $2.55 \times 10^{-15}$ |

**Key Performance Insight**:
- For non-transposed CPU multiplication, CPU cache prefetching is optimal along row-major rows ($0.715\text{ ms}$). GPU pure compute kernel is **$1.66\times$ faster** ($0.430\text{ ms}$).
- For transposed CPU multiplication, row-major CPU access requires strided column traversal, causing significant cache misses ($4.096\text{ ms}$). cuBLAS achieves a **$9.61\times$ compute speedup** ($0.426\text{ ms}$) and a **$1.38\times$ End-to-End speedup** even including entire PCIe host-device data transfers!

### 6.2 Sparse SpMV Benchmark ($2000 \times 2000$, $\text{nnz} = 79,797$, $2.0\%$ Density)

| Metric | Non-Transpose ($y = A x$) | Transpose ($y = A^T x$) |
| :--- | :--- | :--- |
| **CPU Median** | $0.0326\text{ ms}$ | $0.0683\text{ ms}$ |
| **GPU Host-to-Device (H2D)** | $0.3902\text{ ms}$ | $0.3902\text{ ms}$ |
| **GPU Kernel (`cusparseSpMV`)** | **$0.0379\text{ ms}$** | **$0.0328\text{ ms}$** |
| **GPU Device-to-Host (D2H)** | $0.0176\text{ ms}$ | $0.0176\text{ ms}$ |
| **GPU End-to-End (E2E)** | $0.4585\text{ ms}$ | $0.4604\text{ ms}$ |
| **Pure Compute Speedup** | **$0.86\times$** | **$2.08\times$** |
| **End-to-End Speedup (with PCIe)**| $0.07\times$ | $0.15\times$ |
| **Max Absolute Error** | $3.55 \times 10^{-14}$ | $3.55 \times 10^{-14}$ |
| **Scaled Relative Error** | $5.13 \times 10^{-16}$ | $5.13 \times 10^{-16}$ |

**PCIe Transfer Overhead vs. Dedicated Device Memory**:
- In one-off SpMV operations where the matrix and vector are transferred across the PCIe bus for a single multiplication, the transfer time ($\approx 0.39\text{ ms}$) dominates the actual computation ($\approx 0.03\text{ ms}$).
- Consequently, true GPU acceleration in optimization solvers (e.g., Interior-Point or Simplex iterations) is achieved when matrices remain resident in GPU device memory across hundreds of solver iterations, amortizing the transfer cost to zero.

---

## 7. Known Limitations & Future Roadmap

1. **VRAM Capacity Limit**:
   - The NVIDIA RTX 3050 Laptop GPU has 4 GB of total VRAM (approximately 3.3 GB free for application allocations). Dense double-precision matrices larger than $20,000 \times 20,000$ (requiring $\approx 3.2\text{ GB}$) will trigger the `CUDA Out of Memory` guard.
2. **32-bit cuSPARSE Indices**:
   - cuSPARSE Generic API uses 32-bit signed integer offsets (`CUSPARSE_INDEX_32I`). Sparse matrices with more than $2^{31} - 1 \approx 2.14$ billion non-zeros will exceed this format and must use 64-bit indices in future phases.
3. **Phase 8C Transition**:
   - Phase 8C will implement GPU-accelerated Cholesky factorizations (`cusolverDnDpotrf`) and dense linear solvers for normal equations ($A D A^T \Delta y = r$).
