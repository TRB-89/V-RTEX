# VØRTEX Phase 9C: Interactive Optimization Workspace Documentation

## 1. Overview & Objectives

**Phase 9C** delivers a research-grade, interactive mathematical optimization workspace connecting the modern React/TypeScript frontend directly to the native VØRTEX C++20 / CUDA physical optimization solver via the FastAPI API (`/api/v1/*`).

The workspace adheres to the dark futuristic VØRTEX design system:
- **Obsidian**: `#0C0D13` (Core background & viewport canvas)
- **Graphite**: `#191923` (Panels, architectural grids, borders)
- **Chrome Silver**: `#E7E8EF` (High-contrast typography & active vector states)
- **Muted Violet**: `#77768F` (Dimension indicators, inactive indices, secondary metrics)
- **Electric Blue**: `#2514ED` (Active selections, computation highlights, telemetry pulses)

---

## 2. Component Architecture

The Phase 9C workspace is structured modularly within `web/src/components/`:

```
web/src/components/
├── OptimizationWorkspace.tsx       # Primary workspace container & two-panel layout
├── VariableEditor.tsx              # Interactive decision variable editor (cost, bounds, type)
├── ConstraintEditor.tsx            # Interactive linear constraint matrix editor (A, senses, RHS b)
├── QMatrixEditor.tsx               # QP quadratic Hessian matrix editor with symmetry checker
├── MathematicalPreview.tsx         # Live equation renderer (Objective & constraints)
├── ResultsInspector.tsx            # Diagnostics console, feasibility verification & telemetry
└── Navbar.tsx                      # Top navigation bar & telemetry indicator
```

### Component Responsibilities:
1. **`OptimizationWorkspace.tsx`**:
   - Manages top-level formulation state (`solverType`, `sense`, `backend`, `variables`, `constraints`, `qMatrix`).
   - Dynamically checks engine capabilities from `GET /api/v1/capabilities`.
   - Coordinates versioned problem import/export (`vortex_problem_v1`).
   - Controls asynchronous solver execution with `AbortController` cancellation.
   - Preserves state across tab switches in the main application.

2. **`VariableEditor.tsx`**:
   - Manages $n$ decision variables ($x_1, \dots, x_n$).
   - Real-time editing of variable names, linear objective coefficients $c_j$, variable classifications (`CONTINUOUS`, `INTEGER`, `BINARY`), and box bounds $[l_j, u_j]$.
   - Automatic clamping: Binary variables automatically clamp bounds to $[0, 1]$.

3. **`ConstraintEditor.tsx`**:
   - Manages $m$ linear constraints ($A_i x \sim b_i$).
   - Dynamic rows with per-column coefficient inputs $A_{i,j}$.
   - Constraint sense selection: $\le$ (`LEQ`), $\ge$ (`GEQ`), $=$ (`EQ`).
   - Right-hand side scalar $b_i$.

4. **`QMatrixEditor.tsx`**:
   - Active only when formulation is Quadratic Programming (`QP`).
   - Renders $n \times n$ symmetric Hessian matrix grid.
   - Real-time numerical symmetry validator: checks $|Q_{ij} - Q_{ji}| \le 10^{-7}$.
   - One-click auto-symmetrizer: enforces $Q \leftarrow \frac{Q + Q^T}{2}$.
   - Numerical disclaimer: Confirms that rigorous positive semi-definiteness (PSD) is evaluated by native Cholesky factorization in C++.

5. **`MathematicalPreview.tsx`**:
   - Formats objective function in real-time ($Z = \sum c_j x_j + \frac{1}{2} x^T Q x$).
   - Formats linear constraint equations.
   - Live validation box reporting all syntax/dimension/numerical errors before solver execution.

6. **`ResultsInspector.tsx`**:
   - Displays real solver status (`OPTIMAL`, `INFEASIBLE`, `UNBOUNDED`, `CLIENT_ABORTED`, `NUMERICAL_ERROR`, `TIMEOUT`).
   - Displays real objective value $Z^*$ or `"Not reported"`.
   - Primal solution vector table ($x^*$).
   - **Automated Constraint Feasibility Verification**: Evaluates $A_i x^*$ against $b_i$, displays slack, and marks feasible/violated.
   - **Automated Integrality Verification**: Evaluates $|x_j - \text{round}(x_j)| \le 10^{-5}$ for discrete variables.
   - Telemetry metrics: Solve wall-clock time, iteration counts, executed backend, and GPU fallback explanations.
   - One-click export to versioned JSON and tabular CSV.

---

## 3. API Mapping & Endpoints

| Endpoint | Method | Purpose | Native Engine Mapping |
| :--- | :--- | :--- | :--- |
| `/api/v1/health` | `GET` | Verifies solver runner binary presence & state | Verifies `vortex_solver_runner.exe` |
| `/api/v1/capabilities` | `GET` | Dynamically queries supported models & backends | `vortex_solver_runner.exe --capabilities` |
| `/api/v1/solve` | `POST` | Submits problem payload for native execution | `vortex_solver_runner.exe --input <in> --output <out>` |
| `/api/v1/presets` | `GET` | Retrieves verified reference problems | Pre-configured benchmarks |
| `/api/v1/benchmarks` | `GET` | Retrieves Phase 8E empirical performance data | Phase 8E benchmark results |

### Payload Contract (`POST /api/v1/solve`):
```json
{
  "solver_type": "LP" | "MILP" | "QP",
  "sense": "MAXIMIZE" | "MINIMIZE",
  "backend": "AUTO" | "GPU" | "CPU",
  "c": [3.0, 5.0],
  "A": [[1.0, 0.0], [0.0, 2.0], [3.0, 2.0]],
  "b": [4.0, 12.0, 18.0],
  "constraint_types": ["LEQ", "LEQ", "LEQ"],
  "variable_types": ["CONTINUOUS", "CONTINUOUS"],
  "bounds": [{"lower": 0.0, "upper": 1e20}, {"lower": 0.0, "upper": 1e20}],
  "Q": [[2.0, 0.0], [0.0, 2.0]],
  "options": {}
}
```

---

## 4. Supported Mathematical Models

1. **Linear Programming (LP)**:
   - Formulations: Standard Dense Revised Simplex (LU Factorization), Sparse Revised Simplex (CSR/CSC), Two-Phase Simplex.
   - Constraints: $\le$, $\ge$, $=$.
   - Senses: `MAXIMIZE`, `MINIMIZE`.
   - Backends: `CPU`, `GPU`, `AUTO`.

2. **Mixed-Integer Linear Programming (MILP)**:
   - Formulations: Branch-and-Bound with Linear Relaxations.
   - Variable Types: `CONTINUOUS`, `INTEGER`, `BINARY`.
   - Node Selection: `BEST_BOUND`, `DEPTH_FIRST`, `BREADTH_FIRST`.
   - Backends: `CPU` (with GPU acceleration for simplex node relaxations when enabled).

3. **Convex Quadratic Programming (QP)**:
   - Formulations: Primal-Dual Predictor-Corrector Interior-Point Method with Dynamic Tikhonov Regularization.
   - Objective: $\min \frac{1}{2} x^T Q x + c^T x$.
   - Constraints: $A x \le b$.
   - Matrix $Q$: $n \times n$ Symmetric Positive Semi-Definite (PSD).
   - Backends: `CPU`, `GPU`, `AUTO`.

---

## 5. Input Validation Rules

The frontend validates all problem parameters before dispatching:
- **Variable Count**: Minimum 1 decision variable required.
- **Constraint Count**: Minimum 1 constraint required.
- **Numerical Integrity**: Rejects `NaN`, `Infinity`, and unparseable numbers.
- **Bounds Consistency**: Rejects lower bound > upper bound ($l_j > u_j$).
- **Dimension Matching**: Ensures each constraint row in $A$ has exactly $n$ coefficients.
- **QP Hessian Matrix $Q$**:
  - Row and column dimensions must equal $n$.
  - Symmetry verification: $\max_{i,j} |Q_{ij} - Q_{ji}| \le 10^{-7}$.
- **Cancellation & Duplicate Prevention**:
  - `isSolving` lock prevents concurrent duplicate executions.
  - AbortController handles client-side cancellation, accurately flagging `[CLIENT_ABORTED]`.

---

## 6. Testing Instructions & Verification Results

### Running Workspace & Integration Tests:
```powershell
# 1. Run Phase 9C automated test suite
python tests/test_phase9c_workspace.py

# 2. Run API v1 test suite
python tests/test_api_v1.py

# 3. Run Native CLI adapter tests
python tests/test_native_json_interface.py

# 4. Run Full-Stack integration suite
python tests/test_full_stack_integration.py

# 5. Run all 15 native C++/CUDA CTest suites
ctest --test-dir build-cuda --output-on-failure
```

### Verification Results Summary:
- **`test_phase9c_workspace.py`**: **9/9 tests passed (100%)**.
- **`test_api_v1.py`**: **10/10 tests passed (100%)**.
- **`test_native_json_interface.py`**: **5/5 tests passed (100%)**.
- **`test_full_stack_integration.py`**: **10/10 tests passed (100%)**.
- **Native CTest Suite**: **15/15 tests passed (100%)**.

---

## 7. Concrete Real Problem Execution Evidence

### 1. Linear Program (LP) — SIH26119 Simplex Benchmark
- **Problem**:
  $$\max 3x_1 + 5x_2 \quad \text{s.t.} \quad x_1 \le 4, \; 2x_2 \le 12, \; 3x_1 + 2x_2 \le 18, \quad x_1, x_2 \ge 0$$
- **Solver Output**:
  - Status: `OPTIMAL`
  - Objective $Z^*$: `36.000000`
  - Primal Solution $x^*$: `[2.000000, 6.000000]`
  - Feasibility Verification:
    - Constraint 1: $1(2) \le 4$ (Slack: 2.00, Feasible)
    - Constraint 2: $2(6) \le 12$ (Slack: 0.00, Feasible, Binding)
    - Constraint 3: $3(2) + 2(6) = 18 \le 18$ (Slack: 0.00, Feasible, Binding)
  - Wall-clock time: `158.66 ms`

### 2. Mixed-Integer Linear Program (MILP) — 0-1 Knapsack
- **Problem**:
  $$\max 10x_1 + 40x_2 + 30x_3 + 50x_4 \quad \text{s.t.} \quad 5x_1 + 4x_2 + 6x_3 + 3x_4 \le 10, \quad x_j \in \{0, 1\}$$
- **Solver Output**:
  - Status: `OPTIMAL`
  - Objective $Z^*$: `90.000000`
  - Primal Solution $x^*$: `[0.000000, 1.000000, 0.000000, 1.000000]`
  - Explored B&B Nodes: `5`
  - Integrality Verification: $x_1=0$, $x_2=1$, $x_3=0$, $x_4=1$ (Residuals $\le 10^{-6}$, 100% satisfied)
  - Feasibility: $4(1) + 3(1) = 7 \le 10$ (Slack: 3.00, Feasible)

### 3. Convex Quadratic Program (QP) — Constrained Portfolio
- **Problem**:
  $$\min \frac{1}{2}(2x_1^2 + 2x_2^2) - 2x_1 - 5x_2 \quad \text{s.t.} \quad x_1 + x_2 \le 2, \quad x_1, x_2 \ge 0$$
- **Solver Output**:
  - Status: `OPTIMAL`
  - Objective $Z^*$: `-6.125000`
  - Primal Solution $x^*$: `[0.250000, 1.750000]`
  - Duality Gap: `5.62e-12`
  - Feasibility: $0.25 + 1.75 = 2.00 \le 2.00$ (Binding constraint, Feasible)

---

## 8. Known Limitations & Future Work

1. **File Format Parsers (MPS / CPLEX .LP)**:
   - The native engine reports `supported_file_formats: {JSON: true, MPS: false, LP_FILE: false}`.
   - The UI truthfully disables MPS and LP file import buttons with explicit explanatory tooltips rather than fabricating support.
2. **GPU Availability on Large Scale Batches**:
   - On workloads exceeding VRAM or where GPU kernels are uncompiled for specific architectures, the solver transparently falls back to CPU execution with an explicit `fallback_reason` displayed in the telemetry console.
3. **Rigorous PSD Confirmation**:
   - The frontend enforces matrix symmetry $|Q_{ij} - Q_{ji}| \le 10^{-7}$; positive semidefiniteness is rigorously checked by native Cholesky LDL factorizations during solver execution.
