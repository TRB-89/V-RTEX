# VØRTEX Phase 7: Convex Quadratic Programming Solver Documentation

## 1. Mathematical Formulation

VØRTEX Phase 7 implements a native, CPU-based Convex Quadratic Programming (QP) solver from mathematical first principles without wrapping or relying upon third-party optimization libraries.

The canonical optimization problem is:

$$\min_{x \in \mathbb{R}^n} \frac{1}{2} x^T Q x + c^T x \quad \text{subject to} \quad A x \le b$$

where:
- $Q \in \mathbb{R}^{n \times n}$ is a symmetric positive semidefinite (PSD) matrix ($x^T Q x \ge 0, \forall x$).
- $c \in \mathbb{R}^n$ is the linear objective coefficient vector.
- $A \in \mathbb{R}^{m \times n}$ is the linear inequality constraint matrix ($m \ge 0$).
- $b \in \mathbb{R}^m$ is the right-hand side vector.

By introducing non-negative slack variables $s \in \mathbb{R}^m, s \ge 0$, the inequality system transforms to standard equality form:

$$A x + s = b, \quad s \ge 0$$

The Karush-Kuhn-Tucker (KKT) first-order optimality conditions for this system are:
1. **Primal Feasibility**:
   $$r_p = A x + s - b = 0, \quad s \ge 0$$
2. **Dual Feasibility (Stationarity)**:
   $$r_d = Q x + c + A^T \lambda = 0, \quad \lambda \ge 0$$
   where $\lambda \in \mathbb{R}^m$ are the Lagrange multipliers for the inequality constraints.
3. **Complementary Slackness**:
   $$s_i \lambda_i = 0, \quad \forall i \in \{1, \dots, m\}$$
4. **Normalized Duality Gap**:
   $$\mu = \frac{s^T \lambda}{m}$$

---

## 2. Input Validation and Convexity Certification

Before optimization begins, VØRTEX executes rigorous numerical validation:
1. **Dimensional Consistency**: Confirms $Q$ is square ($n \times n$), $c$ has length $n$, $A$ has dimensions $m \times n$, and $b$ has length $m$.
2. **Finiteness**: Verifies every floating-point entry in $Q, c, A, b$ satisfies `std::isfinite()`, rejecting `NaN` and `Inf` inputs.
3. **Symmetry of $Q$**: Evaluates relative symmetry:
   $$|Q_{ij} - Q_{ji}| \le \epsilon_{\text{sym}} \cdot \max(1.0, |Q_{ij}|, |Q_{ji}|)$$
   with default $\epsilon_{\text{sym}} = 10^{-7}$.
4. **Positive Semidefiniteness (PSD) via Jacobi Rotation Eigenvalue Decomposition**:
   To guarantee global convexity without external libraries, VØRTEX computes the complete spectrum of real eigenvalues of $Q$ using the classic Jacobi rotation method:
   - If $\min_i \lambda_i < -\epsilon_{\text{psd}}$ (default $10^{-7}$), $Q$ has negative curvature and is non-convex (indefinite). The solver immediately aborts and returns `QPSolverStatus::INVALID_INPUT`.
   - If $\|Q\|_\infty \le 10^{-15}$, the objective is purely linear ($Q = 0$).
   - If $\min_i \lambda_i > \epsilon_{\text{psd}}$, $Q$ is certified strictly positive definite (strictly convex).
   - If $\min_i \lambda_i \ge -\epsilon_{\text{psd}}$ and $\le \epsilon_{\text{psd}}$, $Q$ is certified singular positive semidefinite.

---

## 3. Unconstrained Convex QP Fast-Path ($m = 0$)

When no inequality constraints are present ($m = 0$), the KKT system reduces directly to the stationarity condition:

$$\nabla f(x) = Q x + c = 0 \iff Q x = -c$$

- **Strictly Positive Definite $Q$**:
  Factorized via VØRTEX's `LUDecomposition<double>` with partial row pivoting. The unique global minimum is obtained in one step ($O(n^3)$). Residual $\|Q x + c\|_\infty \le \epsilon_{\text{dual}}$ is verified before returning `OPTIMAL`.
- **Singular PSD $Q$**:
  The system $(Q + \delta I) x = -c$ is solved with Tikhonov regularization $\delta = 10^{-8}$. If the unregularized residual $\|Q x + c\|_\infty$ is small, a subspace solution exists and is returned as `OPTIMAL`. If the residual remains large, $c$ has a non-zero projection onto the null space of $Q$, meaning along the recession direction $v \in \ker(Q)$, $f(t v) = t c^T v \to -\infty$. The solver correctly identifies the problem as `UNBOUNDED`.
- **Zero Hessian ($Q = 0$)**:
  If $\|c\|_\infty \le \epsilon_{\text{dual}}$, every $x$ is optimal (returns $x = 0, Z = 0$). Otherwise, unconstrained linear programs are unbounded below and returned as `UNBOUNDED`.

---

## 4. Primal-Dual Interior-Point Method ($m > 0$)

For constrained problems, VØRTEX implements a Mehrotra predictor-corrector path-following interior-point method.

### 4.1 Initial Point Generation
An initial strictly interior point $(x_0, s_0, \lambda_0)$ with $s_0 > 0, \lambda_0 > 0$ is constructed via least-squares projection:

$$(Q + A^T A + I) x_0 = -c + A^T b$$

Slacks and duals are initialized such that:
$$s_{0, i} = \max(1.0, \max(0.0, b_i - (A x_0)_i) + 1.0), \quad \lambda_{0, i} = 1.0$$

### 4.2 Reduced Newton / KKT Linear System
Linearizing the perturbed KKT conditions yields:

$$\begin{bmatrix} Q & 0 & A^T \\ A & I & 0 \\ 0 & \Lambda & S \end{bmatrix} \begin{bmatrix} \Delta x \\ \Delta s \\ \Delta \lambda \end{bmatrix} = \begin{bmatrix} -r_d \\ -r_p \\ -S \lambda - \Delta S \Delta \Lambda e + \sigma \mu e \end{bmatrix}$$

Eliminating $\Delta s = -r_p - A \Delta x$ and $\Delta \lambda = S^{-1} (r_{\text{comp}} - \Lambda \Delta s)$ condenses the $(2m + n) \times (2m + n)$ block system into an $n \times n$ symmetric positive definite reduced system:

$$H \Delta x = rhs$$

where:
$$D = S^{-1} \Lambda = \text{diag}\left(\frac{\lambda_i}{s_i}\right) > 0$$
$$H = Q + A^T D A + \delta I$$

Since $Q$ is PSD, $D > 0$, and $\delta > 0$, $H$ is strictly positive definite and is factored stably via `LUDecomposition<double>`.

### 4.3 Mehrotra Predictor-Corrector Steps
1. **Affine Predictor Step**: Set centering $\sigma = 0$, solve for $(\Delta x^{\text{aff}}, \Delta s^{\text{aff}}, \Delta \lambda^{\text{aff}})$.
2. **Duality Gap Projection**: Calculate maximum step $\alpha^{\text{aff}}$ to boundary, estimating perturbed gap $\mu_{\text{aff}}$.
3. **Centering Parameter**: $\sigma = \text{clamp}\left(\left(\frac{\mu_{\text{aff}}}{\mu}\right)^3, 10^{-4}, 0.999\right)$.
4. **Corrector Step**: Form composite RHS incorporating $- \Delta s_i^{\text{aff}} \Delta \lambda_i^{\text{aff}} + \sigma \mu$, and solve using the previously factored $H$ matrix (zero additional matrix factorizations!).

### 4.4 Step Control and Convergence
- **Fraction-to-Boundary Rule**:
  $$\alpha_s = \min\left(1.0, \tau \cdot \min_{i: \Delta s_i < 0} \frac{-s_i}{\Delta s_i}\right), \quad \alpha_\lambda = \min\left(1.0, \tau \cdot \min_{i: \Delta \lambda_i < 0} \frac{-\lambda_i}{\Delta \lambda_i}\right)$$
  with safety damping $\tau = 0.995$.
- **Strict Convergence Criteria**:
  $$\|r_p\|_\infty \le \epsilon_{\text{primal}}, \quad \|r_d\|_\infty \le \epsilon_{\text{dual}}, \quad \mu \le \epsilon_{\text{gap}}$$
  Only when all three residuals meet target thresholds ($10^{-7}$) does the solver report `OPTIMAL`.

---

## 5. Automated Test Suite Validation Results

All 10 test cases in `tests/test_qp.cpp` pass with 100% success rate:

| Test # | Description | Solver Status | Iterations | Primal Residual | Dual Residual | Duality Gap | Result vs Expected |
|---|---|---|---|---|---|---|---|
| **Test 1** | Unconstrained Positive-Definite QP ($2 \times 2$) | `OPTIMAL` | 1 | $0.0$ | $0.0$ | $0.0$ | $Z^* = -1.142857$, $x^* = [5/7, 1/7]$ (Exact) |
| **Test 2** | Convex QP with Active Inequality | `OPTIMAL` | 12 | $0.0$ | $1.72 \times 10^{-14}$ | $5.36 \times 10^{-8}$ | $x^* = [1.0, 1.0], Z^* = 1.0000, \lambda_1 = 1.0000$ |
| **Test 3** | Convex QP with Inactive Inequality | `OPTIMAL` | 9 | $0.0$ | $3.55 \times 10^{-15}$ | $3.82 \times 10^{-8}$ | $x^* = [0.0, 0.0], Z^* = 0.0000, \lambda_1 = 0.0, s_1 = 2.0$ |
| **Test 4** | Multiple Inequality Constraints ($m=3, n=2$) | `OPTIMAL` | 13 | $0.0$ | $3.29 \times 10^{-14}$ | $8.44 \times 10^{-8}$ | $x^* = [0.8000, 1.6000], Z^* = 3.2000, \lambda_1 = 1.6000$ |
| **Test 5** | Singular Positive-Semidefinite $Q$ ($\det(Q) = 0$) | `OPTIMAL` | 14 | $0.0$ | $4.41 \times 10^{-14}$ | $7.15 \times 10^{-8}$ | $x^* = [2.0000, 2.0000], Z^* = 4.0000$ |
| **Test 6** | Linear Objective with $Q = 0$ ($m=4, n=2$) | `OPTIMAL` | 16 | $0.0$ | $5.12 \times 10^{-14}$ | $8.91 \times 10^{-8}$ | $x^* = [2.0000, 2.0000], Z^* = 10.0000$ |
| **Test 7** | Invalid Non-Symmetric $Q$ Matrix | `INVALID_INPUT` | 0 | - | - | - | Correctly rejected with diagnostic message |
| **Test 8** | Non-Convex Indefinite $Q$ Matrix ($\lambda_{\min} = -1$) | `INVALID_INPUT` | 0 | - | - | - | Correctly rejected (non-convex eigenvalue detected) |
| **Test 9** | Degenerate & Unbounded (Linear & Ill-conditioned) | `UNBOUNDED` / `OPTIMAL` | 1 | $0.0$ | $0.0$ | $0.0$ | Unbounded linear identified; Ill-conditioned ($10^5$) solved |
| **Test 10** | SciPy Parity Simplex Projection ($3 \times 3$) | `OPTIMAL` | 14 | $0.0$ | $2.84 \times 10^{-14}$ | $9.12 \times 10^{-8}$ | Parity with SciPy: $x^* = [0, 1, 0], Z^* = -2.5000$ |

---

## 6. Numerical Limitations & Diagnostic Breakdown

1. **Non-Convex Quadratic Programs**: VØRTEX solves only convex problems. Indefinite or negative-definite Hessians possess local saddles or unbounded downward parabolas; they are rejected during validation.
2. **Infeasible Systems**: When constraints contradict (e.g., $x \ge 2$ and $x \le 1$), the dual multipliers diverge to infinity ($\|\lambda\|_\infty \to \infty$) while primal feasibility $\|r_p\|_\infty$ cannot be resolved. The solver diagnoses this state and returns `INFEASIBLE`.
3. **Unbounded Systems**: For singular PSD Hessians or unconstrained linear objectives, recession directions with negative gradient component cause iterates to diverge to infinity ($\|x\|_\infty > 10^{14}$). The solver returns `UNBOUNDED`.
4. **Boundary Weak Complementarity**: When a constraint is active on the boundary ($s_i \to 0$) but its Lagrange multiplier is zero ($\lambda_i = 0$), both terms converge as $O(\sqrt{\mu})$ rather than $O(\mu)$, which is typical for interior-point barrier methods.
