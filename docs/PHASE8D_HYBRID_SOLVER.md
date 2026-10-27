# VØRTEX Phase 8D: Hybrid CPU/GPU Optimization Solver Integration

## 1. Executive Summary

Phase 8D establishes a true **Hybrid CPU/GPU Optimization Solver Architecture** for the VØRTEX optimization framework in modern C++20 and CUDA. Rather than executing GPU routines as isolated one-off operations outside the solver, GPU matrix-vector multiplications (`cuBLAS GEMV` and `cuSPARSE SpMV`) are directly embedded **inside the inner loops of actual solver iterations** for both Linear Programming (Revised Simplex and Sparse Revised Simplex) and Quadratic Programming (Primal-Dual Interior-Point Method).

### Key Architectural Tenets
1. **True Iterative Execution**: In every solver iteration, pricing (LP reduced costs) and constraint/Hessian multiplications (QP primal residuals, dual residuals, affine predictor steps, corrector centering steps) execute directly on the NVIDIA GPU via persistent device memory.
2. **Persistent Matrix Buffers**: The static constraint matrix $A$ and QP Hessian $Q$ are transferred to GPU device memory **once** at the beginning of the solve. Subsequent iterations transfer only minimal vector data ($\pi, c, x, \lambda, \Delta x$), avoiding PCIe transfer saturation.
3. **Numerically Defensible Factorization Separation**: LU factorizations for basis updates in Revised Simplex and KKT systems in QP are maintained on the CPU via `LUDecomposition<double>` with partial pivoting. No unverified GPU factorizations are claimed.
4. **Strict Backend Integrity & Graceful Degeneration**:
   - `SolverBackend::CPU`: Strictly forces pure CPU execution (`executed_backend == ExecutedBackend::CPU`, 0 GPU calls).
   - `SolverBackend::GPU`: Strictly requires GPU execution (`executed_backend == ExecutedBackend::HYBRID`). If CUDA is disabled or no CUDA-capable hardware is present, the solver rejects execution with `INVALID_INPUT` / `UNSUPPORTED` and provides explicit diagnostic telemetry without silent CPU fallback.
   - `SolverBackend::AUTO`: Applies empirical, mathematically documented problem-size and density heuristics ($m \ge 32, n \ge 32, m \cdot n \ge 4096$ for Dense LP; $m \ge 32, \text{nnz} \ge 2000$ for Sparse LP; $n \ge 32, m \ge 16$ for QP). Small instances fall back gracefully to CPU with explicit diagnostic reason tracking.
5. **Full Diagnostics & Telemetry**: Every result struct exposes requested backend, actually executed backend, GPU iterations count, cuBLAS/cuSPARSE call counts, host-to-device (H2D) and device-to-host (D2H) transfer counts, and fallback explanations.

---

## 2. Solver Architecture & GPU Integration Points

```
                               +-----------------------------------+
                               |        VØRTEX Solver API          |
                               | (Revised Simplex / QP IPM / Sparse)|
                               +-----------------+-----------------+
                                                 |
                                     Backend Selection Check
                                     (CPU, GPU, or AUTO Heuristic)
                                                 |
                         +-----------------------+-----------------------+
                         |                                               |
             [CPU Execution Path]                            [Hybrid GPU Path]
                         |                                               |
          +--------------+-------------+                 +---------------+---------------+
          |  Inner Solver Iterations   |                 | Setup Persistent Context      |
          |  - Reduced costs on CPU    |                 | - Upload A/Q once via RAII    |
          |  - LU Factorization on CPU |                 | - Pre-allocate vector buffers |
          |  - Status & Convergence   |                 +---------------+---------------+
          +----------------------------+                                 |
                                                         +---------------+---------------+
                                                         | Inner Solver Iterations Loop  |
                                                         | 1. LU Basis Factorize (CPU)   |
                                                         | 2. Solve Basis/KKT (CPU)      |
                                                         | 3. Copy vectors to GPU (H2D)  |
                                                         | 4. GPU GEMV/SpMV (cuBLAS/cuSP)|
                                                         | 5. Copy results to Host (D2H) |
                                                         | 6. Update Iteration Telemetry |
                                                         +---------------+---------------+
```

### A. Dense Revised Simplex Integration (`GpuDensePricingContext`)
In each iteration of the Revised Simplex method:
1. Primal basic values $x_B = B^{-1} b$ and dual simplex multipliers $\pi = B^{-T} c_B$ are solved via CPU LU backward/forward substitution.
2. **GPU Pricing Step**: Simplex reduced costs are evaluated by:
   $$\bar{c} = c_{\text{std}} - A_{\text{std}}^T \pi$$
   Because $A_{\text{std}}$ is row-major ($m \times n$), it appears to column-major cuBLAS as an $n \times m$ matrix with leading dimension $n$. The multiplication:
   $$\text{cublasDgemv}(\text{CUBLAS\_OP\_N}, n, m, \alpha = -1.0, d_A, n, d_\pi, 1, \beta = 1.0, d_r, 1)$$
   computes $d_r = 1.0 \cdot c_{\text{std}} - 1.0 \cdot A_{\text{std}}^T \pi$ in parallel across GPU streaming multiprocessors.
3. Only the $m$-element dual vector $\pi$ and $n$-element cost vector $c_{\text{std}}$ are copied Host-to-Device each iteration; reduced costs $\bar{c}$ ($n$ elements) are copied Device-to-Host.
4. Entering and leaving variable selections (Dantzig or Bland tie-break) execute on CPU.

### B. Sparse Revised Simplex Integration (`GpuSparsePricingContext`)
1. Before the simplex loop, $A_{\text{std}}^{\text{CSC}}$ is converted to CSR format in $O(\text{nnz} + m + n)$ time and uploaded to persistent device buffers ($d_{\text{row\_ptrs}}$, $d_{\text{col\_indices}}$, $d_{\text{values}}$).
2. cuSPARSE matrix descriptor `cusparseSpMatDescr_t` and dense vector descriptors `cusparseDnVecDescr_t` are constructed once and cached.
3. In each iteration, `cusparseSpMV` with `CUSPARSE_OPERATION_TRANSPOSE` computes:
   $$\bar{c} = c_{\text{std}} - A_{\text{std}}^T \pi$$
   leveraging cuSPARSE CSR transposed SpMV.

### C. Convex Quadratic Programming (`GpuQPIterationContext`)
Inside each interior-point iteration:
1. **Primal Residual**: $r_p = A x + s - b$ calls `gpu_ctx.compute_Ax(x, ax)` via cuBLAS `CUBLAS_OP_T`.
2. **Dual Residual**: $r_d = Q x + c + A^T \lambda$ calls:
   - `gpu_ctx.compute_Qx(x, qx)` on symmetric Hessian $Q$ via cuBLAS.
   - `gpu_ctx.compute_AT_vec(lambda, atl)` via cuBLAS `CUBLAS_OP_N`.
3. **Affine Predictor Direction**:
   - $A^T (D r_p)$ calls `gpu_ctx.compute_AT_vec(d_rp, at_drp)`.
   - $\Delta s_{\text{aff}} = -r_p - A \Delta x_{\text{aff}}$ calls `gpu_ctx.compute_Ax(dx_aff, a_dx_aff)`.
4. **Centering Corrector Direction**:
   - $A^T (w + D r_p)$ calls `gpu_ctx.compute_AT_vec(w_plus_drp, at_w)`.
   - $\Delta s = -r_p - A \Delta x$ calls `gpu_ctx.compute_Ax(dx, a_dx)`.
5. **KKT System Solution**: The reduced KKT system $H = Q + A^T D A + \delta I$ is formed and factorized using CPU `LUDecomposition<double>` with dynamic Tikhonov regularization $\delta$.
6. **Objective Evaluation**: Final quadratic form $(1/2) x^T Q x + c^T x$ calls `gpu_ctx.compute_Qx(x, qx)`.

---

## 3. Telemetry and Diagnostic Metrics

Every solver result structure incorporates `SolverDeviceDiagnostics`:

| Field | Type | Description |
| :--- | :--- | :--- |
| `requested_backend` | `SolverBackend` | `CPU`, `GPU`, or `AUTO` as requested in options |
| `executed_backend` | `ExecutedBackend` | `CPU` or `HYBRID` (GPU iterations + CPU factorization) |
| `gpu_iterations` | `std::size_t` | Number of solver iterations executing GPU kernels |
| `gpu_gemv_calls` | `std::size_t` | Total `cuBLAS` GEMV operations executed |
| `gpu_spmv_calls` | `std::size_t` | Total `cuSPARSE` SpMV operations executed |
| `h2d_transfers` | `std::size_t` | Total Host-to-Device memory copy operations |
| `d2h_transfers` | `std::size_t` | Total Device-to-Host memory copy operations |
| `fallback_reason` | `std::string` | Detailed rationale if CPU execution occurred under `AUTO` or rejection occurred under `GPU` |

---

## 4. Test Results and Validation

### A. CUDA Release Build (`build-cuda`)
**Hardware**: NVIDIA GeForce RTX 3050 Laptop GPU (Compute Capability 8.6, 4095 MB VRAM)
**Toolchain**: MSVC 19.44 / CUDA 12.6, C++20

```text
Test project C:/Users/TRB/modified mindspace sdg/CrediAI/vortex/build-cuda
      Start  1: MatrixUnitTests
 1/14 Test  #1: MatrixUnitTests ..................   Passed    0.03 sec
      Start  2: SimplexUnitTests
 2/14 Test  #2: SimplexUnitTests .................   Passed    0.04 sec
      Start  3: TwoPhaseSimplexUnitTests
 3/14 Test  #3: TwoPhaseSimplexUnitTests .........   Passed    0.03 sec
      Start  4: LUDecompositionUnitTests
 4/14 Test  #4: LUDecompositionUnitTests .........   Passed    0.03 sec
      Start  5: RevisedSimplexUnitTests
 5/14 Test  #5: RevisedSimplexUnitTests ..........   Passed    1.67 sec
      Start  6: SparseMatrixUnitTests
 6/14 Test  #6: SparseMatrixUnitTests ............   Passed    0.04 sec
      Start  7: SparseRevisedSimplexUnitTests
 7/14 Test  #7: SparseRevisedSimplexUnitTests ....   Passed    0.28 sec
      Start  8: NumericalAuditTests
 8/14 Test  #8: NumericalAuditTests ..............   Passed    0.29 sec
      Start  9: MILPUnitTests
 9/14 Test  #9: MILPUnitTests ....................   Passed    0.03 sec
      Start 10: QPUnitTests
10/14 Test #10: QPUnitTests ......................   Passed    0.25 sec
      Start 11: CudaBackendUnitTests
11/14 Test #11: CudaBackendUnitTests .............   Passed    0.13 sec
      Start 12: CudaMatrixOperationsTests
12/14 Test #12: CudaMatrixOperationsTests ........   Passed    0.68 sec
      Start 13: GpuNumericalAccuracyTests
13/14 Test #13: GpuNumericalAccuracyTests ........   Passed    0.86 sec
      Start 14: HybridSolverIntegrationTests
14/14 Test #14: HybridSolverIntegrationTests .....   Passed    0.46 sec

100% tests passed, 0 tests failed out of 14 (Total Test time: 4.85 sec)
```

### B. CPU-Only Release Build (`build-release`)
**Toolchain**: GCC 14.2 / MinGW-w64, C++20 (No CUDA)

```text
Test project C:/Users/TRB/modified mindspace sdg/CrediAI/vortex/build-release
      Start  1: MatrixUnitTests
 1/11 Test  #1: MatrixUnitTests ..................   Passed    0.01 sec
      Start  2: SimplexUnitTests
 2/11 Test  #2: SimplexUnitTests .................   Passed    0.01 sec
      Start  3: TwoPhaseSimplexUnitTests
 3/11 Test  #3: TwoPhaseSimplexUnitTests .........   Passed    0.01 sec
      Start  4: LUDecompositionUnitTests
 4/11 Test  #4: LUDecompositionUnitTests .........   Passed    0.01 sec
      Start  5: RevisedSimplexUnitTests
 5/11 Test  #5: RevisedSimplexUnitTests ..........   Passed    0.01 sec
      Start  6: SparseMatrixUnitTests
 6/11 Test  #6: SparseMatrixUnitTests ............   Passed    0.01 sec
      Start  7: SparseRevisedSimplexUnitTests
 7/11 Test  #7: SparseRevisedSimplexUnitTests ....   Passed    0.01 sec
      Start  8: NumericalAuditTests
 8/11 Test  #8: NumericalAuditTests ..............   Passed    0.01 sec
      Start  9: MILPUnitTests
 9/11 Test  #9: MILPUnitTests ....................   Passed    0.01 sec
      Start 10: QPUnitTests
10/11 Test #10: QPUnitTests ......................   Passed    0.01 sec
      Start 11: HybridSolverIntegrationTests
11/11 Test #11: HybridSolverIntegrationTests .....   Passed    0.04 sec

100% tests passed, 0 tests failed out of 11 (Total Test time: 0.17 sec)
```

### C. HybridSolverIntegrationTests Detailed Trace (48 / 48 Tests Passed)
- **Backend Selection & Safety**:
  - Explicit `CPU`: passes with `executed_backend == CPU`, 0 GPU iterations.
  - `AUTO` thresholding: 2x4 small LP falls back to CPU with documented dimension heuristic.
  - Explicit `GPU`: successfully initializes device context, sets `executed_backend == HYBRID`, and tracks iterations, GEMV calls, H2D, and D2H transfers.
- **Dense Revised Simplex Cross-Validation**:
  - Objective value matches CPU reference within $10^{-6}$ ($|z_{\text{gpu}} - z_{\text{cpu}}| < 10^{-6}$).
  - Decision variables match CPU reference within $10^{-5}$ ($||x_{\text{gpu}} - x_{\text{cpu}}||_\infty < 10^{-5}$).
  - Pricing operations verified: `gpu_gemv_calls == gpu_iterations == iterations + 1`.
  - Memory transfers verified: `h2d_transfers == 1 + 2 * (iters + 1)`, `d2h_transfers == iters + 1`.
- **Two-Phase Revised Simplex**:
  - Mixed constraint LP ($\le, \ge, =$) solved to optimality.
  - Phase I and Phase II pricing calls accumulated cleanly in diagnostic counters.
- **Sparse Revised Simplex Cross-Validation**:
  - Bounded sparse LP solved to optimality with identical objective value ($< 10^{-6}$ discrepancy).
  - cuSPARSE SpMV calls matched pricing iterations; H2D and D2H transfers tracked.
- **Convex QP IPM Integration**:
  - Constrained QP ($n = 35, m = 20$) converges to optimal solution matching CPU within $10^{-5}$.
  - Primal residual $\|Ax + s - b\|_\infty \le 10^{-6}$.
  - Dual residual $\|Qx + c + A^T \lambda\|_\infty \le 10^{-6}$.
  - Complementarity duality gap $\mu \le 10^{-6}$.
  - cuBLAS calls $\ge 6 \times \text{iterations}$ verifying active acceleration in every IPM iteration.
- **Edge Cases & Robustness**:
  - Unbounded LP on GPU correctly terminates with `SolverStatus::UNBOUNDED`.
  - Infeasible LP on GPU correctly terminates with `SolverStatus::INFEASIBLE`.
  - Degenerate Beale cycling problem terminates with `SolverStatus::OPTIMAL` under Bland's rule.
  - Dimension mismatches and invalid inputs safely rejected with `INVALID_INPUT` without device resource leaks.

---

## 5. Limitations & Future Work (Phase 8E)
1. **Factorization Boundary**: LU factorizations for LP bases and QP KKT systems currently remain on the CPU. While cuSOLVER LU factorizations are technically feasible, dense LU factorizations for small to medium matrices ($m \le 200$) incur higher device synchronization latency than CPU LU substitution with partial pivoting.
2. **Dense QP Assumption**: The current `GpuQPIterationContext` targets dense QP Hessian and constraint matrices. Sparse QP formulation using cuSPARSE for $Q$ and $A$ will be considered in future extensions.
3. **Formal Benchmarking (Phase 8E)**: Comprehensive microbenchmarks comparing end-to-end solve time across varying sparsity ratios, matrix scales, and problem families will be implemented in Phase 8E.
