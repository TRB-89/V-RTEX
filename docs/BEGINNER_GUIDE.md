# Beginner's Guide to C++ in VORTEX (Phase 1, 2 & 3)

Welcome to **VORTEX**! This guide walks you through the concepts and implementations across Phase 1 (Matrix Engine), Phase 2 (Standard Simplex), and Phase 3 (General Two-Phase Simplex).

---

## 1. What is the Two-Phase Simplex Method?

Standard Simplex requires all constraints to be $\le$ with non-negative right-hand sides ($b \ge 0$). In the real world, optimization problems often have:
- Constraints with $\ge$ (e.g., minimum nutritional or production requirements)
- Constraints with $=$ (e.g., exact mass or flow conservation balances)
- Negative RHS numbers ($b_i < 0$)
- Minimization goals ($\min \text{cost}$)

The **Two-Phase Simplex Method** is an algorithm that can solve any linear program.

---

## 2. Step-by-Step Two-Phase Process

### Step 2.1: Preprocessing Negative RHS
If an equation has a negative RHS like $-x_1 - x_2 \le -4$:
Multiply by $-1$ and flip the sign:
$$x_1 + x_2 \ge 4$$
Now the RHS is $+4 \ge 0$.

### Step 2.2: Slack, Surplus, and Artificial Variables
1. **Slack variable ($s \ge 0$)**: Added to $\le$ constraints ($+s$). It measures unused capacity.
2. **Surplus variable ($e \ge 0$)**: Subtracted from $\ge$ constraints ($-e$). It measures excess over the minimum.
3. **Artificial variable ($a \ge 0$)**: When we have $\ge$ or $=$ constraints, setting $x=0$ would make $e = -b < 0$ (infeasible!). To get a starting basis, we add an artificial variable $+a$ with initial value $a = b \ge 0$.

### Step 2.3: Phase I — Finding a Feasible Starting Point
Artificial variables do not exist in the real problem. They are mathematical scaffolds.
In **Phase I**, our sole goal is to drive all artificial variables to zero:
$$\text{Minimize } W = \sum a_k \iff \text{Maximize } -W = -\sum a_k$$

- If Phase I ends with $W = 0$, we successfully removed the scaffolds and found a valid corner (BFS) of the real feasible region! We proceed to Phase II.
- If Phase I ends with $W > 0$, it is mathematically impossible to satisfy all constraints at once. The problem is **INFEASIBLE**!

### Step 2.4: Phase II — Finding the Optimal Solution
Once we have a valid feasible basis:
1. Drop the artificial variables.
2. Put back the original objective function ($Z = c^T x$).
3. Use Gaussian elimination pivots until no entering variable can improve the objective.
4. If an entering variable has no positive coefficients in any row, the objective can grow to infinity, meaning the problem is **UNBOUNDED**!

---

## 3. How to Build & Run with CMake

```powershell
# Configure build
cmake -B build -G "MinGW Makefiles"

# Compile all targets
cmake --build build

# Run Two-Phase Simplex unit tests
.\build\vortex_two_phase_tests.exe

# Run Standard Simplex unit tests
.\build\vortex_simplex_tests.exe

# Run Matrix unit tests
.\build\vortex_tests.exe

# Run all test suites via CTest
ctest --test-dir build --output-on-failure

# Run the full CLI demonstration
.\build\vortex_cli.exe
```
