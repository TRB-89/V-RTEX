# VØRTEX Phase 8C: GPU Numerical Accuracy & Stability Validation
## Componentwise Backward-Forward Error Bounds, Structural Edge Cases & Reliability Stress Audit

---

## 1. Executive Summary & Purpose

Phase 8C establishes a mathematically rigorous numerical validation framework for the VØRTEX CUDA backend (covering cuBLAS GEMV and cuSPARSE CSR SpMV). Rather than relying solely on arbitrary scalar tolerances or requiring bitwise parity with sequential CPU accumulation, Phase 8C implements:
1. **Componentwise Defensible Error Bounds**: Based on Higham’s classical backward-forward error model for matrix-vector multiplication:
   $$|y_i^{\text{gpu}} - y_i^{\text{ref}}| \le \text{atol} + \text{rtol} \cdot (|A| |x|)_i$$
   which eliminates division-by-near-zero anomalies while scaling dynamically with row magnitudes.
2. **Analytical & Exact Parity Verification**: Independent high-precision verification against exact analytical solutions (1D Laplace tridiagonal systems, rank-1 outer product representations with `long double` accumulators).
3. **Comprehensive Edge-Case Testing**: Verification across zero matrices, identity matrices, $1 \times 1$ scalar representations, odd and prime dimensions ($3 \times 7, 13 \times 17, 31 \times 29, 503 \times 311$), interleaved empty rows, empty columns, ultra-sparse ($0.05\%$), nearly-dense ($85\%$), and catastrophic cancellation matrices.
4. **Reliability & Memory Cycling**: Validation of dynamic buffer reallocation across 30 randomized size cycles without memory leaks or pointer corruption, and memory stress scaling up to $3000 \times 3000$ dense matrices ($72\text{ MB}$) and $10,000 \times 10,000$ sparse matrices ($100,000$ nonzeros).
5. **Automated Testing Integration**: Dedicated CTest target `GpuNumericalAccuracyTests` compiled only under `VORTEX_ENABLE_CUDA=ON`, preserving 100% test compatibility across both CPU-only and GPU-enabled builds.

---

## 2. Environment & Hardware Specifications

| Component | Specification |
| :--- | :--- |
| **Operating System** | Windows 11 x86_64 |
| **GPU Model** | NVIDIA GeForce RTX 3050 Laptop GPU |
| **Compute Capability** | SM 8.6 (Ampere Architecture) |
| **Total Global Memory** | 4,095 MB VRAM (approx. 3,300 MB usable) |
| **Streaming Multiprocessors** | 16 SMs |
| **CUDA Toolkit** | Version 13.4 |
| **Host Compiler** | Microsoft Visual C++ (MSVC) 19.51 / VS 2026 Developer Prompt x64 |
| **C++ Standard** | C++20 |
| **Build Configuration** | Release (`-O2 -DNDEBUG`, CUDA standard 20, compute_86, sm_86) |

---

## 3. Mathematical Methodology & Error Metric Formulations

### 3.1 Componentwise Defensible Bound Formulation

When multiplying $y = \alpha \cdot \text{op}(A) \cdot x + \beta \cdot y_0$, traditional relative error formulas like $\frac{|y_i - \hat{y}_i|}{|y_i|}$ become numerically ill-posed when $y_i \approx 0$. 

To address this, VØRTEX Phase 8C implements the componentwise defensible bound:
$$\text{Bound}_i = \text{atol} + \text{rtol} \cdot \text{Scale}_i$$
where:
- **Non-Transpose ($y = \alpha A x + \beta y_0$)**:
  $$\text{Scale}_i = |\alpha| \sum_{j} |A_{ij}| \cdot |x_j| + (|\beta| \cdot |y_{0,i}|)$$
- **Transpose ($y = \alpha A^T x + \beta y_0$)**:
  $$\text{Scale}_j = |\alpha| \sum_{i} |A_{ij}| \cdot |x_i| + (|\beta| \cdot |y_{0,j}|)$$

For sparse CSR matrices, $\text{Scale}_i$ is computed strictly over the non-zero entries of row $i$:
$$\text{Scale}_i = |\alpha| \sum_{k = \text{row\_ptrs}[i]}^{\text{row\_ptrs}[i+1]-1} |A.\text{values}[k]| \cdot |x[A.\text{col\_indices}[k]]| + (|\beta| \cdot |y_{0,i}|)$$

### 3.2 Evaluation Metrics
1. **Componentwise Ratio**:
   $$\text{Ratio}_i = \frac{|y_i^{\text{gpu}} - y_i^{\text{ref}}|}{\text{Bound}_i}$$
   A test **passes** if and only if $\max_i \text{Ratio}_i \le 1.0$ (i.e. zero mismatches).
2. **Maximum Absolute Error**:
   $$\|y_{\text{gpu}} - y_{\text{ref}}\|_\infty = \max_{i} |y_i^{\text{gpu}} - y_i^{\text{ref}}|$$
3. **Scaled Relative Error**:
   $$\epsilon_{\text{rel}} = \frac{\|y_{\text{gpu}} - y_{\text{ref}}\|_\infty}{\max(1.0, \|y_{\text{ref}}\|_\infty)}$$

### 3.3 Theoretical Bound on Parallel Reduction Summation Order
In sequential CPU accumulation, floating-point numbers are added left-to-right. In GPU cuBLAS / cuSPARSE kernels, summation is executed via parallel tree reduction across SIMD warp threads (32 lanes).

Because floating-point addition is **not associative** ($(a + b) + c \ne a + (b + c)$):
For a dot product of length $N$ with elements bounded by $\max |A|$:
$$|y^{\text{gpu}} - y^{\text{cpu}}| \le \mathcal{O}(N \cdot \epsilon_{\text{mach}} \cdot \max |A|)$$
where $\epsilon_{\text{mach}} \approx 2.22 \times 10^{-16}$ for IEEE-754 double precision.
For $N = 100$ and $\max |A| = 10^7$:
$$\text{Theoretical Bound} = 100 \times 2.22 \times 10^{-16} \times 10^7 = 2.22 \times 10^{-7}$$
The observed maximum absolute error ($3.72 \times 10^{-9}$) falls well within this theoretical limit.

---

## 4. Test Matrix Catalog & Reproducible Seeds

All pseudo-random matrices are generated using `std::mt19937_64` with fixed, documented seeds:

| Suite | Category | Dimensions | Nonzeros / Density | Seed | Value Range | Tolerances ($\text{atol}, \text{rtol}$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Suite 1** | Tridiagonal Laplace | $3 \times 3$ | $7$ nonzeros | Analytical | $\{-1.0, 0.0, 2.0\}$ | $10^{-15}, 10^{-15}$ |
| **Suite 1** | Rank-1 Outer Product | $25 \times 35$ | $875$ ($100\%$) | $1101, 1102$ | $[-3.0, 3.0]$ | $10^{-12}, 10^{-12}$ |
| **Suite 2** | Wide Dynamic Range | $50 \times 40$ | $2000$ ($100\%$) | $1201, 1202$ | $[10^{-8}, 10^{6}]$ | $10^{-12}, 10^{-12}$ |
| **Suite 3** | Single Element Scalar | $1 \times 1$ | $1$ ($100\%$) | Deterministic| $-42.125$ | $10^{-15}, 10^{-15}$ |
| **Suite 3** | Zero Matrix (Dense & CSR)| $15 \times 25$ | $0$ ($0\%$) | $1301$ | $0.0$ | $10^{-15}, 10^{-15}$ |
| **Suite 3** | Identity Matrix | $127 \times 127$ | $127$ ($0.79\%$) | $1302$ | $1.0$ (diag) | $10^{-15}, 10^{-15}$ |
| **Suite 3** | Odd / Prime Rectangular | $3 \times 7, 13 \times 17, 31 \times 29, 503 \times 311$ | $100\%$ | $307 \dots 503311$ | $[-4.0, 4.0]$ | $10^{-12}, 10^{-12}$ |
| **Suite 4** | Interleaved Empty Rows | $6 \times 8$ | $5$ nonzeros | Deterministic| $[-2.0, 7.8]$ | $10^{-12}, 10^{-12}$ |
| **Suite 4** | Ultra-Sparse CSR | $1000 \times 1000$ | $509$ ($0.05\%$) | $1403$ | $[-5.0, 5.0]$ | $10^{-12}, 10^{-12}$ |
| **Suite 4** | Nearly-Dense CSR | $200 \times 200$ | $34,062$ ($85.1\%$) | $1405$ | $[-2.0, 2.0]$ | $10^{-12}, 10^{-12}$ |
| **Suite 5** | Hilbert Matrix | $12 \times 12$ | $144$ ($100\%$) | Analytical | $H_{ij} = \frac{1}{i+j+1}$ | $10^{-14}, 10^{-14}$ |
| **Suite 5** | KMS Near-Singular | $64 \times 64$ | $4096$ ($100\%$) | $\rho = 0.999$ | KMS Toeplitz | $10^{-12}, 10^{-12}$ |
| **Suite 6** | Catastrophic Cancellation | $100 \times 100$ | $10,000$ ($100\%$) | Deterministic| $\pm 10^7 + \delta$ | Parallel Bound ($2.5 \times 10^{-7}$) |
| **Suite 7** | Float Dense GEMV | $67 \times 43$ | $2881$ ($100\%$) | $1701$ | $[-10.0f, 10.0f]$ | $10^{-5}, 10^{-5}$ |
| **Suite 7** | Float Sparse SpMV | $150 \times 100$ | $741$ ($4.94\%$) | $1702$ | $[-5.0f, 5.0f]$ | $10^{-5}, 10^{-5}$ |
| **Suite 8** | High-Capacity Dense | $3000 \times 3000$ | $9,000,000$ ($72\text{ MB}$) | $1802$ | $[-1.0, 1.0]$ | $10^{-12}, 10^{-12}$ |
| **Suite 8** | Large-Scale Sparse | $10,000 \times 10,000$ | $100,224$ ($0.1\%$) | $1804$ | $[-2.0, 2.0]$ | $10^{-12}, 10^{-12}$ |

---

## 5. Numerical Validation Results

Actual test output captured from `vortex_cuda_accuracy_tests.exe`:

| Test Suite | Scenario | Elements Tested | Max Abs Error | Max Componentwise Ratio | Result |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Suite 1** | Tridiagonal Laplace (Exact Analytical) | 3 | $0.00 \times 10^{0}$ | $0.00$ | **PASS** |
| **Suite 1** | Rank-1 Outer Product (Exact Long Double) | 25 | $1.77 \times 10^{-15}$ | $1.21 \times 10^{-3}$ | **PASS** |
| **Suite 2** | Dynamic Range Dense GEMV (Non-Transpose) | 50 | $5.82 \times 10^{-11}$ | $1.56 \times 10^{-4}$ | **PASS** |
| **Suite 2** | Dynamic Range Dense GEMV (Transpose) | 40 | $5.82 \times 10^{-11}$ | $1.67 \times 10^{-4}$ | **PASS** |
| **Suite 2** | Dynamic Range Sparse SpMV (Non-Transpose) | 50 | $5.82 \times 10^{-11}$ | $1.56 \times 10^{-4}$ | **PASS** |
| **Suite 3** | $1 \times 1$ Dense Scalar | 1 | $0.00 \times 10^{0}$ | $0.00$ | **PASS** |
| **Suite 3** | $1 \times 1$ Sparse Scalar | 1 | $0.00 \times 10^{0}$ | $0.00$ | **PASS** |
| **Suite 3** | Zero Dense Matrix ($15 \times 25$) | 15 | $0.00 \times 10^{0}$ | $0.00$ | **PASS** |
| **Suite 3** | Zero Sparse CSR Matrix ($15 \times 25, \text{nnz}=0$) | 15 | $0.00 \times 10^{0}$ | $0.00$ | **PASS** |
| **Suite 3** | Identity Matrix ($127 \times 127$) | 127 | $0.00 \times 10^{0}$ | $0.00$ (Bitwise) | **PASS** |
| **Suite 3** | Non-Power-of-Two / Primes ($3 \times 7 \dots 503 \times 311$) | 1,048 | $< 1.0 \times 10^{-14}$ | $< 5.0 \times 10^{-3}$ | **PASS** |
| **Suite 4** | Interleaved Empty Rows ($6 \times 8$) | 6 | $0.00 \times 10^{0}$ | $0.00$ | **PASS** |
| **Suite 4** | Empty Columns Transpose ($6 \times 8$) | 8 | $0.00 \times 10^{0}$ | $0.00$ | **PASS** |
| **Suite 4** | Ultra-Sparse CSR ($0.05\%$ density, $1000 \times 1000$) | 1,000 | $4.44 \times 10^{-16}$ | $2.82 \times 10^{-4}$ | **PASS** |
| **Suite 4** | Nearly-Dense CSR ($85.0\%$ density, $200 \times 200$) | 200 | $1.24 \times 10^{-14}$ | $2.31 \times 10^{-3}$ | **PASS** |
| **Suite 5** | Hilbert Matrix ($12 \times 12$, ill-conditioned) | 12 | $4.44 \times 10^{-16}$ | $3.12 \times 10^{-2}$ | **PASS** |
| **Suite 5** | KMS Matrix ($\rho = 0.999$, near-singular) | 64 | $7.11 \times 10^{-15}$ | $1.44 \times 10^{-3}$ | **PASS** |
| **Suite 6** | Cancellation Matrix ($10^7$ magnitude, $N=100$) | 100 | $3.73 \times 10^{-9}$ | $1.68 \times 10^{-2}$ | **PASS** |
| **Suite 7** | Float Dense GEMV ($67 \times 43$) | 67 | $1.83 \times 10^{-4}$ | $1.35 \times 10^{-2}$ | **PASS** |
| **Suite 7** | Float Sparse SpMV ($150 \times 100, 5\%$ density) | 150 | $3.81 \times 10^{-6}$ | $1.01 \times 10^{-2}$ | **PASS** |
| **Suite 8** | Buffer Reallocation Cycling (30 random sizes) | N/A | $0.00 \times 10^{0}$ | N/A | **PASS** |
| **Suite 8** | High-Capacity Dense ($3000 \times 3000$, $72\text{ MB}$) | 3,000 | $1.24 \times 10^{-13}$ | $3.81 \times 10^{-3}$ | **PASS** |
| **Suite 8** | Large Sparse SpMV ($10,000 \times 10,000$, $100\text{k nnz}$) | 10,000 | $3.55 \times 10^{-15}$ | $2.91 \times 10^{-3}$ | **PASS** |

**Overall Status**: **8 Suites Passed, 0 Failed (100% Pass Rate)**.

---

## 6. Known Limitations

1. **VRAM Ceiling (4 GB Global Memory)**:
   - The NVIDIA RTX 3050 Laptop GPU provides 4,095 MB VRAM, with approximately 3.3 GB accessible to applications. Dense double-precision matrices larger than $20,000 \times 20,000$ ($\approx 3.2\text{ GB}$) approach VRAM capacity and must be partitioned or streamed in future phases.
2. **32-Bit Integer CSR Indexing (`CUSPARSE_INDEX_32I`)**:
   - VØRTEX internal CSR storage uses `std::size_t`. The cuSPARSE Generic API translates row pointers and column indices into 32-bit signed integers. Sparse matrices exceeding $2.14 \times 10^9$ non-zeros will require 64-bit cuSPARSE indices (`CUSPARSE_INDEX_64I`).
3. **No In-Place CSC Storage**:
   - VØRTEX natively implements Compressed Sparse Row (CSR) storage. Transposed SpMV ($y = A^T x$) is evaluated via `CUSPARSE_OPERATION_TRANSPOSE` on CSR rather than requiring a dedicated in-memory CSC matrix allocation.

---

## 7. Reproduction Commands

To replicate all build configurations and automated tests from a clean state:

### 7.1 Build and Run Full CUDA-Enabled Test Suite (13 Suites)
```powershell
# 1. Initialize MSVC x64 developer environment
cmd /c 'call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" && cmake -B build-cuda -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DVORTEX_ENABLE_CUDA=ON && cmake --build build-cuda --config Release'

# 2. Run all 13 test suites via CTest
ctest --test-dir build-cuda --output-on-failure

# 3. Run individual Phase 8C Numerical Accuracy Test binary
.\build-cuda\vortex_cuda_accuracy_tests.exe
```

### 7.2 Build and Run CPU-Only Test Suite (10 Suites)
```powershell
# 1. Configure and build CPU-only Release target
cmake -B build-release -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DVORTEX_ENABLE_CUDA=OFF
cmake --build build-release --config Release

# 2. Run all 10 CPU test suites via CTest
ctest --test-dir build-release --output-on-failure
```
