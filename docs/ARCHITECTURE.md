# VORTEX: High-Performance Optimization Engine (SIH26119)
## Architecture & Mathematical Foundations: Phase 1, Phase 2 & Phase 3

---

### 1. Overview
**VORTEX** is an optimization engine designed for **SIH26119**.
- **Phase 1**: Foundational linear algebra engine (C++20 Matrix class, contiguous row-major memory, $i \to k \to j$ cache optimization).
- **Phase 2**: Standard tableau-based Simplex linear programming solver for $\le$ constraints with $b \ge 0$.
- **Phase 3**: General Two-Phase Simplex solver supporting $\le, \ge, =$ constraints, negative RHS normalization, minimization and maximization, slack, surplus, and artificial variables, and infeasibility detection.

```
                      +-----------------------------+
                      |   SIH26119 Problem Layer    |
                      +-----------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|             Phase 3: General Two-Phase Simplex LP Solver Layer          |
|                                                                         |
|  +-----------------------+  +-------------------+  +-----------------+  |
|  | Negative RHS Normaliz |  | Slack, Surplus &  |  | Phase I / II    |  |
|  | (Flip inequalities)   |  | Artificial Vars   |  | State Machine   |  |
|  +-----------------------+  +-------------------+  +-----------------+  |
|                                                                         |
|  [SimplexSolver General API]                                            |
|    - <=, >=, == constraint types                                        |
|    - Maximization and Minimization objective senses                     |
|    - Infeasibility detection (Phase I artificial sum W > 0)             |
|    - Unboundedness detection & Bland's anti-cycling                     |
+-------------------------------------------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                Phase 1: VORTEX Matrix Core Engine                       |
|                                                                         |
|  +------------------------+  +-------------------+  +----------------+  |
|  | Contiguous Memory 1D   |  | C++20 Arithmetic  |  | Cache-Friendly |  |
|  | Row-Major Storage      |  | Concept Guard     |  | (i-k-j) Gemm   |  |
|  +------------------------+  +-------------------+  +----------------+  |
+-------------------------------------------------------------------------+
```

---

### 2. Phase 3 General Two-Phase Simplex Architecture

#### 2.1 Problem Formulation
VORTEX Phase 3 solves general linear programs of the form:
$$\begin{aligned}
\text{Optimize } \quad & Z = c^T x \quad (\text{Maximize or Minimize}) \\
\text{Subject to } \quad & \sum_{j=0}^{n-1} A_{i, j} x_j \sim_i b_i, \quad \forall i \in \{0, \dots, m-1\}, \quad \sim_i \in \{\le, \ge, =\} \\
& x_j \ge 0, \quad \forall j \in \{0, \dots, n-1\}
\end{aligned}$$

#### 2.2 Preprocessing & Negative RHS Normalization
Before constructing the initial tableau, any constraint with a negative right-hand side ($b_i < 0$) is normalized:
1. Multiply row $i$ by $-1$: $A_{i, \cdot} \leftarrow -A_{i, \cdot}$, $b_i \leftarrow -b_i$.
2. Invert inequality sense:
   - $\le$ becomes $\ge$
   - $\ge$ becomes $\le$
   - $=$ remains $=$
All $b_i \ge 0$ are now guaranteed non-negative.

#### 2.3 Variable Augmentation
- **Less-Than-or-Equal ($\le$)**:
  $$\sum A_{i, j} x_j + s_i = b_i \quad (s_i \ge 0 \text{ slack variable})$$
  Initial basic variable: $s_i$.
- **Greater-Than-or-Equal ($\ge$)**:
  $$\sum A_{i, j} x_j - e_i + a_i = b_i \quad (e_i \ge 0 \text{ surplus, } a_i \ge 0 \text{ artificial})$$
  Initial basic variable: $a_i$.
- **Equality ($=$)**:
  $$\sum A_{i, j} x_j + a_i = b_i \quad (a_i \ge 0 \text{ artificial})$$
  Initial basic variable: $a_i$.

#### 2.4 Phase I: Driving Artificial Variables to Zero
If $num\_artificial > 0$:
1. Objective: Minimize $W = \sum a_k \iff \text{Maximize } -W = -\sum a_k$.
2. Initialize objective row $m$ with $+1.0$ on artificial columns.
3. Bring to canonical form by subtracting each constraint row containing an artificial basic variable from row $m$:
   $$\text{Row}(m) \leftarrow \text{Row}(m) - \text{Row}(i)$$
4. Pivot until Phase I optimality.
5. **Infeasibility Check**:
   $$\text{Final } W = -Tableau(m, \text{RHS})$$
   - If $W > \varepsilon$: The problem is **INFEASIBLE**.
   - If $W \le \varepsilon$: A basic feasible solution to the original problem exists!
6. If any artificial variable remains basic at value $0$ (degenerate BFS), pivot it out with an available non-artificial variable.

#### 2.5 Phase II: Optimizing the Original Objective
1. Set up effective objective row:
   - For `MAXIMIZE`: $c_{eff} = c$.
   - For `MINIMIZE`: $c_{eff} = -c$.
2. Set $Tableau(m, j) = -c_{eff}[j]$ for $j < n$.
3. Bring to canonical form by adding $c_{eff}[var] \times \text{Row}(i)$ for each basic decision variable $var = basis[i]$.
4. Lock out artificial variables from entering the basis.
5. Pivot to optimality or detect unboundedness.

---

### 3. Build Targets
| Target | Type | Description |
|---|---|---|
| `vortex_core` | `STATIC` | Static C++20 library with Matrix engine & Two-Phase Simplex solver |
| `vortex_cli` | `EXECUTABLE` | Command-line demo showcasing Matrix math, Simplex, and Two-Phase solving |
| `vortex_tests` | `EXECUTABLE` | Phase 1 Matrix unit test suite (19 test cases) |
| `vortex_simplex_tests` | `EXECUTABLE` | Phase 2 Simplex unit test suite (17 test cases) |
| `vortex_two_phase_tests` | `EXECUTABLE` | Phase 3 General Two-Phase Simplex test suite (17 test cases) |
| `vortex_benchmarks` | `EXECUTABLE` | Performance benchmark suite |
