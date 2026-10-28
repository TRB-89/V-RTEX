# VØRTEX (SIH26119) — Indigenous GPU-Accelerated Optimization Engine

```
 ██╗   ██╗ ██████╗ ██████╗ ████████╗███████╗██╗  ██╗
 ██║   ██║██╔═══██╗██╔══██╗╚══██╔══╝██╔════╝╚██╗██╔╝
 ██║   ██║██║   ██║██████╔╝   ██║   █████╗   ╚███╔╝ 
 ╚██╗ ██╔╝██║   ██║██╔══██╗   ██║   ██╔══╝   ██╔██╗ 
  ╚████╔╝ ╚██████╔╝██║  ██║   ██║   ███████╗██╔╝ ██╗
   ╚═══╝   ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚══════╝╚═╝  ╚═╝
              v1.0.0-RESEARCH-PROTOTYPE
```

> **An Indigenous, GPU-Accelerated Mathematical Optimization Solver for Linear Programming (LP), Mixed-Integer Linear Programming (MILP), and Convex Quadratic Programming (QP).**

Developed for **Smart India Hackathon (SIH26119)**, VØRTEX bridges mathematical optimization theory and high-performance heterogeneous scientific computing. It combines an optimized C++20 core engine, native CUDA kernel pipelines, a safe FastAPI machine-readable REST interface, and an editorial React 19/TypeScript web workspace.

---

## Architecture Overview

```
                                      +---------------------------------------------+
                                      |         VØRTEX Web Interface (v1.0)         |
                                      |     React 19 / TypeScript / Vite / GSAP     |
                                      +----------------------+----------------------+
                                                             |
                                           REST API (HTTP / JSON / FP64)
                                                             |
                                      +----------------------v----------------------+
                                      |            FastAPI Python Backend           |
                                      |     Validation / Concurrency / Telemetry    |
                                      +----------------------+----------------------+
                                                             |
                                              Subprocess Exec (No Shell / Pipes)
                                                             |
                                      +----------------------v----------------------+
                                      |         Native Solver Runner (CLI)          |
                                      |         vortex_solver_runner.exe            |
                                      +----------------------+----------------------+
                                                             |
                      +--------------------------------------+--------------------------------------+
                      |                                                                             |
       +--------------v---------------+                                              +--------------v---------------+
       |       CPU Solver Engine      |                                              |      CUDA Acceleration       |
       |                              |                                              |                              |
       | - Two-Phase Primal Simplex   |                                              | - cuBLAS FP64 GEMV           |
       | - Revised Simplex + LU (PP)  | <================ Hybrid Scheduler ========> | - cuSPARSE CSR SpMV          |
       | - Sparse CSR Simplex Pricing |            Dynamic Fallback Thresholds       | - Warp-Level Pricing Reduct. |
       | - MILP Branch-and-Bound      |                                              | - Unified Device Memory Pool |
       | - Mehrotra Interior-Point QP |                                              | - cudaEvent High-Res Clocks  |
       +------------------------------+                                              +------------------------------+
```

---

## Supported Mathematical Capabilities

| Formulation | Mathematical Specification | Algorithms & Implementations | Backends Supported |
| :--- | :--- | :--- | :---: |
| **Linear Programming (LP)** | $\max/\min\ c^T x \text{ s.t. } A x \{\le, \ge, =\} b,\ x \ge 0$ | - Standard Simplex (Dantzig/Bland rule)<br>- Two-Phase Simplex (Artificial variables)<br>- Revised Simplex with LU Decomposition & Partial Pivoting<br>- Sparse Revised Simplex with $O(nnz)$ column pricing | **CPU**, **GPU**, **AUTO** |
| **Mixed-Integer Linear Programming (MILP)** | $\max/\min\ c^T x \text{ s.t. } A x \le b,\ x_j \in \mathbb{Z} \text{ for } j \in I$ | - Branch-and-Bound tree search<br>- Continuous, Integer, and Binary variables<br>- Arbitrary variable bounds $[l_j, u_j]$<br>- Most-fractional variable branching | **CPU** (Native AVX2) |
| **Convex Quadratic Programming (QP)** | $\min\ \frac{1}{2} x^T Q x + c^T x \text{ s.t. } A x \le b$ | - Mehrotra Predictor-Corrector Primal-Dual Interior-Point Method (IPM)<br>- Jacobi eigenvalue PSD verification ($Q \succeq 0$)<br>- Dynamic KKT system regularization ($10^{-12} \le \delta \le 10^{-6}$)<br>- Unconstrained stationarity analytic fast-path | **CPU**, **GPU**, **AUTO** |

---

## Hardware & Toolchain Requirements

### Recommended Hardware Environment
- **Host CPU**: AMD Ryzen 5 6600H / Intel Core i5-12500H or equivalent (6 physical cores, 12 threads).
- **Physical GPU**: NVIDIA GeForce RTX 3050 Laptop GPU (16 SMs, Compute Capability 8.6, 4 GB GDDR6 VRAM).
- **Host RAM**: 16 GB DDR5 / DDR4.

### Software Stack
- **Compiler**: MSVC 19.51+ (Visual Studio 2022) with C++20 support.
- **CUDA Toolkit**: CUDA 12.0+ / 13.4+ (`nvcc`, `cuBLAS`, `cuSPARSE`).
- **Build System**: CMake 3.20+ and Ninja.
- **Backend Runtime**: Python 3.11+ (FastAPI, Uvicorn, Pydantic, Pytest).
- **Frontend Runtime**: Node.js 20+ (React 19, TypeScript 5.8, Tailwind CSS, Recharts, Three.js, jsPDF).

---

## Build & Installation Guide (Windows)

### 1. Build Native C++20 / CUDA Optimization Engine
Open a **Visual Studio 2022 x64 Native Tools Command Prompt**:

```powershell
# Clone repository
git clone https://github.com/TRB-89/V-RTEX.git
cd V-RTEX

# Configure CMake with CUDA acceleration enabled
cmake -B build-cuda -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DVORTEX_ENABLE_CUDA=ON

# Compile the native solver runner, benchmark harness, and test targets
cmake --build build-cuda --config Release

# Verify native unit tests via CTest (14 test suites)
ctest --test-dir build-cuda -E GpuBenchmarkCorrectnessTests --output-on-failure
```

### 2. Configure and Launch FastAPI Backend
```powershell
# Navigate to repository root and create virtual environment
python -m venv .venv
.\.venv\Scripts\activate

# Install backend dependencies
pip install fastapi uvicorn pydantic pytest

# Launch FastAPI production adapter
python -m uvicorn server.main:app --host 127.0.0.1 --port 8000 --reload
```
API Documentation will be live at: `http://127.0.0.1:8000/api/docs`.

### 3. Launch Web User Interface
```powershell
# Navigate to web directory
cd web

# Install npm dependencies
npm install

# Run Vite development server
npm run dev
```
Open browser at: `http://127.0.0.1:5173`.

---

## Native CLI Solver Usage

The native solver executable accepts machine-readable JSON via file or standard I/O:

```powershell
# Solve a Linear Program from JSON file
.\build-cuda\vortex_solver_runner.exe --input problem.json --output solution.json

# Query compiled engine capabilities
.\build-cuda\vortex_solver_runner.exe --capabilities

# Query physical GPU hardware telemetry
.\build-cuda\vortex_solver_runner.exe --device-info
```

### Sample Problem JSON (`problem.json`)
```json
{
  "solver_type": "QP",
  "sense": "MINIMIZE",
  "backend": "AUTO",
  "Q": [[2.0, 0.0], [0.0, 2.0]],
  "c": [-2.0, -5.0],
  "A": [[1.0, 1.0]],
  "b": [2.0],
  "constraint_types": ["LEQ"]
}
```

---

## Empirical Benchmark Findings (Phase 8E)

Measured on physical hardware (**NVIDIA GeForce RTX 3050 Laptop GPU** and **AMD Ryzen 5 6600H**) across 33 empirical experiment records:

1. **Extreme Simplex Pricing Scaling**:
   - Column pricing on a $1024 \times 4096$ matrix achieved **`904.88x` speedup** ($226.66\text{ ms}$ on CPU down to $0.25\text{ ms}$ on GPU).
2. **Buffer Reuse Imperative**:
   - Reusing persistent GPU memory buffers across iterations eliminates PCIe overhead, yielding a **`31.19x` speedup**.
3. **Truthful Small-Problem Slowdown**:
   - For small matrices ($< 256 \times 256$), GPU end-to-end execution is slower ($0.20\times\text{ to }0.72\times$) due to PCIe transfer latency. VØRTEX's dynamic `AUTO` scheduler routes these workloads to CPU cache.
4. **Numerical Precision**:
   - Double-precision FP64 elementwise errors remain strictly bounded below $10^{-11}$.

---

## Test Verification Summary

```
================================================================================
VØRTEX Automated Test Suite Execution Status
================================================================================
Native CTest Suites (14/14 Passed):
  [PASS] MatrixUnitTests (0.01s)
  [PASS] SimplexUnitTests (0.01s)
  [PASS] TwoPhaseSimplexUnitTests (0.01s)
  [PASS] LUDecompositionUnitTests (0.01s)
  [PASS] RevisedSimplexUnitTests (1.53s)
  [PASS] SparseMatrixUnitTests (0.01s)
  [PASS] SparseRevisedSimplexUnitTests (0.27s)
  [PASS] NumericalAuditTests (0.12s)
  [PASS] MILPUnitTests (0.01s)
  [PASS] QPUnitTests (0.13s)
  [PASS] CudaBackendUnitTests (0.10s)
  [PASS] CudaMatrixOperationsTests (0.52s)
  [PASS] GpuNumericalAccuracyTests (0.93s)
  [PASS] HybridSolverIntegrationTests (0.27s)

Python Integration & Security Tests (64/64 Passed):
  [PASS] tests/test_api_v1.py (10/10)
  [PASS] tests/test_native_json_interface.py (5/5)
  [PASS] tests/test_phase9c_workspace.py (9/9)
  [PASS] tests/test_phase9d_visualization.py (8/8)
  [PASS] tests/test_phase9e_benchmarks.py (8/8)
  [PASS] tests/test_phase9f_final_validation.py (14/14)
  [PASS] tests/test_full_stack_integration.py (10/10)

Frontend Build (0 Errors):
  [PASS] tsc -b && vite build (2752 modules transformed, gzip: 728.77 kB)
================================================================================
```

---

## Production Cloud Deployment (Render Free Tier)

VØRTEX provides a production-ready, multi-stage Docker deployment optimized for **Render Free Web Services** (512 MB RAM, headless Linux C++20 optimization solver, pure CPU LP/MILP/QP execution):

- **Zero-Config Blueprint**: Deploy directly via [`render.yaml`](render.yaml).
- **Deployment Documentation**: Complete deployment, configuration, and verification guide in [`docs/RENDER_DEPLOYMENT.md`](docs/RENDER_DEPLOYMENT.md).
- **Safe Hardware Limits**: Preconfigured with process-level memory limits (`MAX_SOLVER_MEMORY_MB=300`), concurrency semaphore (`MAX_CONCURRENT_SOLVES=1`), and solver timeouts (`SOLVER_TIMEOUT_SECONDS=15`) tailored for 512 MB memory footprints.

---

## Known Limitations

- **Branch-and-Bound on GPU**: MILP branch-and-bound tree search is currently restricted to CPU host execution due to warp divergence and irregular stack memory access.
- **Single SpMV Transfer Bottleneck**: Single isolated Sparse Matrix-Vector multiplications show PCIe latency dominance. SpMV acceleration requires persistent iterative memory residence.
- **Non-Convex QP**: Indefinite or concave Hessian matrices are rejected by eigenvalue analysis ($Q \succeq 0$ required).

---

## License

Developed as part of the **Smart India Hackathon (SIH26119)** project initiative. Released under the MIT License.
