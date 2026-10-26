# VORTEX (SIH26119) — High-Performance Optimization Engine

VORTEX is an optimization engine developed for **SIH26119**.
- **Phase 1**: Core Linear Algebra Engine in C++20 with contiguous 1D memory layout and cache-friendly matrix operations.
- **Phase 2**: Standard CPU Simplex LP Solver for $\le$ constraints with non-negative RHS.
- **Phase 3**: General Two-Phase Simplex Solver supporting $\le, \ge, =$ constraints, negative RHS normalization, minimization and maximization, slack, surplus, and artificial variables, and infeasibility detection.
- **Phase 4**: Revised Simplex Solver + LU Decomposition with partial pivoting, forward/backward substitution, basis matrix operations, reduced-cost calculations, and numerical stability checks.
- **Phase 5**: Sparse Matrix Engine (CSR and CSC), SpMV, Dense-to-Sparse conversions, Sparse-Aware Revised Simplex with $O(nnz)$ column pricing, and memory/performance benchmarks.
- **Phase 6**: Mixed-Integer Linear Programming (MILP) Branch-and-Bound solver supporting Continuous, Integer, and Binary variables, arbitrary bounds, best-bound/DFS/BFS exploration, most-fractional branching, and incumbent tracking.

---

## Project Structure
```
vortex/
├── CMakeLists.txt              # CMake 3.20+ build definitions
├── README.md                   # Project overview & quick start
├── include/
│   ├── simplex.hpp             # Simplex LP solver header (Standard & Two-Phase)
│   ├── revised_simplex.hpp     # Revised Simplex LP solver header
│   ├── lu_decomposition.hpp    # LU decomposition header
│   ├── sparse_matrix.hpp       # CSR and CSC sparse matrix templates
│   ├── sparse_revised_simplex.hpp # Sparse Revised Simplex LP solver
│   ├── milp.hpp                # Mixed-Integer Linear Programming (MILP) solver
│   └── vortex/
│       ├── matrix.hpp          # C++20 Matrix class template with concept constraints
│       ├── simplex.hpp         # Forwarding header for namespace inclusion
│       ├── revised_simplex.hpp # Revised Simplex LP solver header
│       ├── lu_decomposition.hpp# LU factorization with partial pivoting
│       ├── sparse_matrix.hpp   # CSR & CSC sparse matrix implementations
│       ├── sparse_revised_simplex.hpp # Sparse Revised Simplex solver
│       └── milp.hpp            # MILP Branch-and-Bound solver header
├── src/
│   ├── simplex.cpp             # Simplex tableau implementation & Two-Phase logic
│   ├── revised_simplex.cpp     # Revised Simplex solver & basis operations
│   ├── sparse_revised_simplex.cpp # Sparse Revised Simplex implementation
│   ├── milp.cpp                # MILP Branch-and-Bound solver implementation
│   └── main.cpp                # CLI demo (Phases 1-6)
├── tests/
│   ├── test_matrix.cpp         # 19 automated unit tests for Matrix engine (Phase 1)
│   ├── test_simplex.cpp        # 17 automated unit tests for Standard Simplex (Phase 2)
│   ├── test_two_phase_simplex.cpp # 17 automated unit tests for Two-Phase Simplex (Phase 3)
│   ├── test_lu_decomposition.cpp  # 24 automated unit tests for LU Decomposition (Phase 4)
│   ├── test_revised_simplex.cpp   # 31 automated unit tests for Revised Simplex (Phase 4)
│   ├── test_sparse_matrix.cpp     # 45 automated unit tests for Sparse Engine (Phase 5)
│   ├── test_sparse_revised_simplex.cpp # 21 automated unit tests for Sparse Simplex (Phase 5)
│   ├── test_numerical_audit.cpp   # 54 automated numerical & memory audit tests (Phase 5.1)
│   └── test_milp.cpp           # 27 automated unit tests for MILP solver (Phase 6)
├── benchmarks/
│   ├── benchmark_matrix.cpp    # Dense Matrix performance benchmarks
│   └── benchmark_sparse.cpp    # Sparse memory, SpMV, and pricing benchmarks
└── docs/
    ├── ARCHITECTURE.md         # Architecture, mathematical foundations & memory specs
    └── BEGINNER_GUIDE.md       # Beginner guide to C++20, Matrix math & Two-Phase Simplex
```

---

## Quick Start Commands

```powershell
# 1. Configure the build
cmake -B build -G "MinGW Makefiles"

# 2. Compile all targets
cmake --build build

# 3. Run all unit tests via CTest (255 total tests across 9 suites)
ctest --test-dir build-release --output-on-failure

# 4. Run individual test suites
.\build-release\vortex_milp_tests.exe
.\build-release\vortex_numerical_audit_tests.exe
.\build-release\vortex_sparse_revised_simplex_tests.exe
.\build-release\vortex_sparse_matrix_tests.exe
.\build-release\vortex_revised_simplex_tests.exe
.\build-release\vortex_lu_tests.exe
.\build-release\vortex_two_phase_tests.exe
.\build-release\vortex_simplex_tests.exe
.\build-release\vortex_tests.exe

# 5. Run performance benchmarks
.\build\vortex_sparse_benchmarks.exe
.\build\vortex_benchmarks.exe

# 6. Run the CLI demo
.\build\vortex_cli.exe
```

---

## Solved Benchmark Problems

### 1. SIH26119 Standard Form Benchmark (Phase 2)
$$\begin{aligned}
\text{Maximize } \quad & Z = 3x + 5y \\
\text{Subject to } \quad & x \le 4, \quad 2y \le 12, \quad 3x + 2y \le 18, \quad x, y \ge 0
\end{aligned}$$
- Status: `OPTIMAL`
- Iterations: `2`
- Solution: $x = 2.0, y = 6.0 \implies Z^* = 36.0$

### 2. General Mixed-Constraint Benchmark (Phase 3)
$$\begin{aligned}
\text{Maximize } \quad & Z = 3x_1 + 2x_2 \\
\text{Subject to } \quad & 2x_1 + x_2 \le 18 \quad (\text{Slack}) \\
& 2x_1 + 3x_2 \ge 12 \quad (\text{Surplus + Artificial}) \\
& x_1 + x_2 = 8 \quad (\text{Artificial}) \\
& x_1, x_2 \ge 0
\end{aligned}$$
- Status: `OPTIMAL` (via Two-Phase Simplex)
- Phase I Iterations: `3` (Eliminates artificial variables, $W = 0$)
- Phase II Iterations: `0` (Initial feasible BFS is optimal)
- Solution: $x_1 = 8.0, x_2 = 0.0 \implies Z^* = 24.0$

### 3. General Minimization with Surplus Variables
$$\begin{aligned}
\text{Minimize } \quad & Z = 4x_1 + 2x_2 \\
\text{Subject to } \quad & 3x_1 + x_2 \ge 27, \quad x_1 + x_2 \ge 21, \quad x_1 + 2x_2 \ge 30, \quad x_1, x_2 \ge 0
\end{aligned}$$
- Status: `OPTIMAL`
- Solution: $x_1 = 3.0, x_2 = 18.0 \implies Z^* = 48.0$

### 4. 0-1 Binary Knapsack Benchmark (Phase 6)
$$\begin{aligned}
\text{Maximize } \quad & Z = 10x_1 + 40x_2 + 30x_3 + 50x_4 \\
\text{Subject to } \quad & 5x_1 + 4x_2 + 6x_3 + 3x_4 \le 10 \\
& x_j \in \{0, 1\}
\end{aligned}$$
- Status: `OPTIMAL`
- Explored Nodes: `5`
- Solution: $x = [0, 1, 0, 1] \implies Z^* = 90.0$
