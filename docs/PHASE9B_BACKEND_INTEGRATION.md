# VØRTEX — Phase 9B: Native C++/CUDA Solver API Integration

## 1. Overview & System Architecture

Phase 9B establishes the secure, machine-readable backend interface connecting external applications and the React frontend to the native **VØRTEX C++20 / CUDA** optimization engine.

```mermaid
graph TD
    Client[React Frontend / External Client] -->|POST /api/v1/solve| FastAPI[FastAPI Adapter Process (server/main.py)]
    Client -->|GET /api/v1/capabilities| FastAPI
    Client -->|GET /api/v1/health| FastAPI
    FastAPI -->|Argument Array IPC (shell=False)| NativeRunner[vortex_solver_runner.exe]
    NativeRunner -->|FP64 Matrix Engine| CoreLib[vortex_core (C++20)]
    NativeRunner -->|cuBLAS & cuSPARSE| CudaLib[vortex_cuda (CUDA 13.4 / SM 8.6)]
    CudaLib -->|Direct GPU Kernels| Hardware[NVIDIA GeForce RTX 3050 Laptop GPU]
    NativeRunner -->|Structured JSON via File/Pipe| FastAPI
    FastAPI -->|Typed JSON Response| Client
```

### Security & Process Isolation Guarantees
- **No User Shell Execution**: Processes are invoked using argument arrays (`shell=False`) with strict executable paths. User inputs never form shell strings.
- **Process Concurrency Limit**: Controlled via `asyncio.Semaphore(4)` to prevent server resource thrashing.
- **Hard Timeout**: 30.0-second execution cutoff via `asyncio.wait_for`. On timeout, the subprocess is terminated and all temporary files are purged.
- **Controlled Working Directory**: Subprocess execution is restricted strictly to the repository root.
- **CORS Restriction**: Explicitly limited to authorized origins (`http://localhost:5173`, `http://127.0.0.1:5173`, `http://localhost:3000`).

---

## 2. Real API Contracts (/api/v1/*)

### 2.1 `GET /api/v1/health`
Checks API operational status and native executable existence.

#### Response (200 OK)
```json
{
  "status": "online",
  "engine": "VØRTEX C++20 / CUDA Optimization Engine",
  "version": "3.0.0",
  "executable_exists": true,
  "executable_path": "C:\\Users\\TRB\\modified mindspace sdg\\CrediAI\\vortex\\build-cuda\\vortex_solver_runner.exe"
}
```

---

### 2.2 `GET /api/v1/capabilities`
Directly queries the native binary with `--capabilities` to report verified solver algorithms, physical CUDA hardware, supported file formats, and limits.

#### Response (200 OK)
```json
{
  "engine": "VØRTEX Optimization Engine",
  "version": "3.0.0",
  "cxx_standard": "C++20",
  "precision": "IEEE 754 FP64 (Double Precision)",
  "solvers": {
    "LP": {
      "algorithms": [
        "Dense Revised Simplex (LU Factorization)",
        "Sparse Revised Simplex (CSR/CSC)",
        "Two-Phase Standard Simplex"
      ],
      "senses": ["MAXIMIZE", "MINIMIZE"],
      "constraints": ["LEQ (<=)", "GEQ (>=)", "EQ (==)"],
      "backends": ["CPU", "GPU", "AUTO"],
      "status_codes": ["OPTIMAL", "INFEASIBLE", "UNBOUNDED", "ITERATION_LIMIT", "NUMERICAL_ERROR", "INVALID_INPUT"]
    },
    "MILP": {
      "algorithms": ["Branch-and-Bound with Linear Relaxations"],
      "variable_types": ["CONTINUOUS", "INTEGER", "BINARY"],
      "branching_strategies": ["MOST_FRACTIONAL", "FIRST_FRACTIONAL"],
      "node_selection": ["BEST_BOUND", "DEPTH_FIRST", "BREADTH_FIRST"],
      "backends": ["CPU"],
      "status_codes": ["OPTIMAL", "FEASIBLE_LIMIT", "INFEASIBLE", "UNBOUNDED", "NODE_LIMIT", "TIME_LIMIT", "NUMERICAL_ERROR"]
    },
    "QP": {
      "algorithms": ["Primal-Dual Predictor-Corrector Interior-Point Method with Dynamic Tikhonov Regularization"],
      "problem_classes": [
        "Convex strictly positive definite Q",
        "Convex positive semidefinite singular Q",
        "Linear objective Q=0",
        "Unconstrained QP (m=0)"
      ],
      "constraints": ["Linear Inequalities A*x <= b"],
      "backends": ["CPU", "GPU", "AUTO"],
      "status_codes": ["OPTIMAL", "MAX_ITERATIONS", "NUMERICAL_ERROR", "UNBOUNDED", "INFEASIBLE", "INVALID_INPUT", "UNSUPPORTED"]
    }
  },
  "supported_file_formats": {
    "JSON": true,
    "MPS": false,
    "LP_FILE": false
  },
  "cuda": {
    "available": true,
    "device_name": "NVIDIA GeForce RTX 3050 Laptop GPU",
    "compute_capability": "8.6",
    "multiprocessors": 16,
    "total_vram_bytes": 4294443008,
    "kernels": ["cuBLAS GEMV (FP64)", "cuSPARSE SpMV (CSR FP64)"]
  },
  "limits": {
    "max_variables": 50000,
    "max_constraints": 50000,
    "max_elements": 25000000
  }
}
```

---

### 2.3 `POST /api/v1/solve`
Submits an optimization problem instance for native solution.

#### Sample Request: Linear Program (LP)
```json
{
  "solver_type": "LP",
  "sense": "MAXIMIZE",
  "backend": "AUTO",
  "c": [3.0, 5.0],
  "A": [
    [1.0, 0.0],
    [0.0, 2.0],
    [3.0, 2.0]
  ],
  "b": [4.0, 12.0, 18.0],
  "constraint_types": ["LEQ", "LEQ", "LEQ"],
  "options": {
    "tolerance": 1e-9,
    "pivot_rule": "DANTZIG"
  }
}
```

#### Sample Response: LP (200 OK)
```json
{
  "success": true,
  "solver_type": "LP",
  "status": "OPTIMAL",
  "message": "Optimal solution found successfully via Two-Phase Revised Simplex.",
  "objective_value": 36.0,
  "variable_values": [2.0, 6.0],
  "slack_values": [2.0, 0.0, 0.0],
  "surplus_values": [],
  "dual_values": [0.0, 1.5, 1.0],
  "reduced_costs": [0.0, 0.0, 0.0, -1.5, -1.0],
  "basis_indices": [2, 1, 0],
  "iterations": 2,
  "phase1_iterations": 0,
  "phase2_iterations": 2,
  "max_primal_residual": 0.0,
  "max_dual_residual": 0.0,
  "basis_condition": 0.333333333333,
  "wall_time_ms": 15.77,
  "device_diagnostics": {
    "requested_backend": "AUTO",
    "executed_backend": "CPU",
    "gpu_iterations": 0,
    "gpu_gemv_calls": 0,
    "gpu_spmv_calls": 0,
    "h2d_transfers": 0,
    "d2h_transfers": 0,
    "fallback_reason": "Problem dimension (3x5, 15 elements) below GPU acceleration threshold (m >= 32, n >= 32, m*n >= 4096); AUTO selected CPU."
  },
  "matrix_info": {
    "rows": 3,
    "cols": 2,
    "nnz": 4,
    "density": 0.666666666667
  },
  "request_meta": {
    "solver_type": "LP",
    "requested_backend": "AUTO",
    "dimension_n": 2,
    "dimension_m": 3,
    "sense": "MAXIMIZE"
  }
}
```

#### Sample Request: Convex Quadratic Program (QP)
```json
{
  "solver_type": "QP",
  "backend": "AUTO",
  "Q": [
    [2.0, 0.0],
    [0.0, 2.0]
  ],
  "c": [-2.0, -5.0],
  "A": [
    [1.0, -1.0],
    [1.0, 2.0],
    [-1.0, 0.0],
    [0.0, -1.0]
  ],
  "b": [2.0, 6.0, 0.0, 0.0],
  "options": {
    "primal_tolerance": 1e-7,
    "dual_tolerance": 1e-7,
    "duality_gap_tolerance": 1e-7
  }
}
```

#### Sample Response: QP (200 OK)
```json
{
  "success": true,
  "solver_type": "QP",
  "status": "OPTIMAL",
  "message": "Optimality conditions (primal feasibility, dual feasibility, complementarity) satisfied.",
  "objective_value": -7.24999980415,
  "variable_values": [0.999802084082, 2.49960416815],
  "lambda": [1.11931630368e-11, 0.000395831864569, 3.91650116567e-11, 1.56672425139e-11],
  "slack": [3.49980208407, 0.000989579620626, 0.999802084082, 2.49960416815],
  "iterations": 12,
  "primal_residual": 0.0,
  "dual_residual": 3.97323233342e-14,
  "duality_gap": 9.79561598481e-08,
  "wall_time_ms": 16.87,
  "device_diagnostics": {
    "requested_backend": "AUTO",
    "executed_backend": "CPU",
    "gpu_iterations": 0,
    "gpu_gemv_calls": 0,
    "gpu_spmv_calls": 0,
    "h2d_transfers": 0,
    "d2h_transfers": 0,
    "fallback_reason": "QP dimensions (n=2, m=4) below GPU threshold (n >= 32, m >= 16); AUTO selected CPU."
  }
}
```

---

## 3. Build & Operational Commands

### 3.1 Build Native Solver Runner (`vortex_solver_runner.exe`)
```powershell
cmd /c 'call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" && cmake --build build-cuda --config Release --target vortex_solver_runner'
```

### 3.2 Launch FastAPI Backend Server
```powershell
cd server
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

### 3.3 Launch React Frontend (Vite)
```powershell
cd web
npm run dev -- --host 127.0.0.1 --port 5173
```

---

## 4. Test Suite Execution & Evidence

### 4.1 Native C++ & CUDA Test Suite (15 Test Executables)
```powershell
ctest --test-dir build-cuda --output-on-failure
# Result: 100% tests passed, 0 tests failed out of 15 (21.64s)
```

### 4.2 Native CLI Interface Unit Tests
```powershell
python tests/test_native_json_interface.py
# Result: 5/5 tests passed (2.28s)
```

### 4.3 API v1 Automated Integration Tests
```powershell
python tests/test_api_v1.py
# Result: 10/10 tests passed (1.37s)
```

---

## 5. CUDA Requirements & Fallback Behavior

1. **Hardware Requirement**: NVIDIA Pascal architecture (SM 6.0) or higher. Benchmarks and testing are performed on Ampere SM 8.6 (`NVIDIA GeForce RTX 3050 Laptop GPU`).
2. **CUDA Toolkit Requirement**: CUDA Toolkit 12.0+ (Tested on CUDA 13.4 with cuBLAS and cuSPARSE).
3. **Crossover & AUTO Fallback**:
   - For instances smaller than $32 \times 32$ or total elements $< 4096$, PCIe transfer latency exceeds GPU compute savings.
   - When `backend: "AUTO"` is selected, VØRTEX automatically falls back to CPU execution and documents this in `device_diagnostics.fallback_reason`.
   - When `backend: "GPU"` is explicitly requested on an unsupported dimension or non-CUDA hardware, the engine cleanly reports the exact reason without silent failures.

---

## 6. Known Limitations

- **File Parser Support**: Only structured JSON problem definitions are currently supported by the API. Industry standard MPS and CPLEX LP file formats are not yet natively implemented; attempting to submit MPS will return an explicit unsupported notice rather than fabricated results.
- **MILP GPU Acceleration**: Branch-and-Bound tree search currently runs on the CPU host, using the native LP engine for continuous node relaxations.
- **Maximum Problem Size**: Single-request matrix element allocation is capped at 25,000,000 elements to prevent host memory exhaustion.
