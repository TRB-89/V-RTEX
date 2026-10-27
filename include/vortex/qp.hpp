#pragma once

#include "vortex/matrix.hpp"
#include "vortex/solver_backend.hpp"
#include <vector>
#include <string>
#include <limits>
#include <optional>
#include <cmath>

namespace vortex {

/**
 * @brief Status resulting from solving a Quadratic Program.
 */
enum class QPSolverStatus {
    OPTIMAL,          ///< Karush-Kuhn-Tucker (KKT) optimality conditions satisfied within tolerances
    MAX_ITERATIONS,   ///< Maximum interior-point iterations reached without convergence
    NUMERICAL_ERROR,  ///< Linear system singularity, numerical breakdown, or invalid search direction
    UNBOUNDED,        ///< Objective can improve to -infinity along an unconstrained or feasible recession direction
    INFEASIBLE,       ///< Problem constraints cannot be satisfied simultaneously
    INVALID_INPUT,    ///< Dimension mismatch, non-finite coefficient, asymmetry, or non-convexity (indefinite Q)
    UNSUPPORTED       ///< Feature or problem formulation not supported by the solver
};

/**
 * @brief Model definition for a Convex Quadratic Program.
 *
 * Mathematical formulation:
 *   Minimize   (1/2) * x^T * Q * x + c^T * x
 *   Subject to   A * x <= b
 *
 * where:
 *   - Q is an (n x n) symmetric positive semidefinite (PSD) matrix.
 *   - c is an (n x 1) vector.
 *   - A is an (m x n) matrix (optional; if m = 0, problem is unconstrained).
 *   - b is an (m x 1) vector (optional).
 */
struct QuadraticProgram {
    Matrix<double> Q;             ///< (n x n) symmetric positive semidefinite Hessian matrix
    std::vector<double> c;        ///< (n x 1) linear objective vector
    Matrix<double> A;             ///< (m x n) linear inequality constraint matrix (A x <= b)
    std::vector<double> b;        ///< (m x 1) constraint right-hand side vector
};

/**
 * @brief Configuration parameters for the Primal-Dual Interior-Point QP Solver.
 */
struct QPOptions {
    double primal_tolerance{1e-7};       ///< Target primal feasibility tolerance: ||Ax + s - b||_inf <= tol
    double dual_tolerance{1e-7};         ///< Target dual feasibility tolerance: ||Qx + c + A^T lambda||_inf <= tol
    double duality_gap_tolerance{1e-7};   ///< Target complementarity duality gap: s^T lambda / max(1, m) <= tol
    std::size_t max_iterations{100};     ///< Maximum interior-point iterations
    double step_damping{0.995};          ///< Fraction-to-boundary parameter (tau in (0, 1))
    double regularization{1e-10};        ///< Tikhonov diagonal regularization parameter delta for KKT system
    double psd_tolerance{1e-7};          ///< Minimum eigenvalue threshold for positive semidefiniteness
    double symmetry_tolerance{1e-7};     ///< Threshold for symmetry check |Q_ij - Q_ji|
    bool verbose{false};                 ///< Print iteration progress to std::cout
    SolverBackend backend{SolverBackend::AUTO}; ///< Computation backend (CPU, GPU, AUTO)
};

/**
 * @brief Step-by-step diagnostic telemetry recorded per interior-point iteration.
 */
struct QPIterationRecord {
    std::size_t iteration{0};
    double objective_value{0.0};
    double primal_residual{0.0};
    double dual_residual{0.0};
    double duality_gap{0.0};
};

/**
 * @brief Solution and diagnostic result returned by the Convex QP Solver.
 */
struct QPResult {
    QPSolverStatus status{QPSolverStatus::NUMERICAL_ERROR};
    double objective_value{0.0};         ///< Optimal objective value (1/2) * x^T * Q * x + c^T * x
    std::vector<double> x;               ///< Optimal primal solution vector (n x 1)
    std::vector<double> lambda;          ///< Dual variables / Lagrange multipliers for A x <= b (m x 1)
    std::vector<double> slack;           ///< Slack variables s = b - A x >= 0 (m x 1)
    std::size_t iterations{0};           ///< Total interior-point iterations performed
    double primal_residual{0.0};         ///< Infinity norm of primal residual: ||Ax + s - b||_inf
    double dual_residual{0.0};           ///< Infinity norm of dual residual: ||Qx + c + A^T lambda||_inf
    double duality_gap{0.0};             ///< Normalized duality gap: s^T lambda / max(1, m)
    std::string message;                 ///< Detailed status or error description
    SolverDeviceDiagnostics device_diagnostics; ///< Telemetry on backend selection and GPU execution
    std::vector<QPIterationRecord> iteration_history; ///< Genuine per-iteration convergence telemetry
};

/**
 * @brief CPU-Based Convex Quadratic Programming (QP) Solver.
 *
 * Implements a primal-dual interior-point method with predictor-corrector
 * steps from mathematical first principles, solving:
 *
 *   Minimize   (1/2) * x^T * Q * x + c^T * x
 *   Subject to   A * x <= b
 *
 * Features:
 *  - Full numerical validation: symmetry, finiteness, and positive semidefiniteness verification.
 *  - Robust handling of strictly positive-definite (PD), positive semidefinite (PSD) singular,
 *    and linear (Q = 0) objective functions.
 *  - Dedicated fast-path for unconstrained convex QPs (m = 0).
 *  - Primal-dual interior-point algorithm with fraction-to-the-boundary step control.
 *  - Dynamic KKT regularization to prevent numerical breakdown near ill-conditioned boundaries.
 *  - Rigorous KKT residual tracking (primal, dual, and complementarity).
 */
class ConvexQPSolver {
public:
    explicit ConvexQPSolver(QPOptions options = QPOptions{});

    /**
     * @brief Solves a Quadratic Program given a QuadraticProgram struct.
     */
    [[nodiscard]] QPResult solve(const QuadraticProgram& qp);

    /**
     * @brief Convenience overload for solving a Quadratic Program.
     */
    [[nodiscard]] QPResult solve(
        const Matrix<double>& Q,
        const std::vector<double>& c,
        const Matrix<double>& A = Matrix<double>(),
        const std::vector<double>& b = std::vector<double>());

private:
    QPOptions options_;

    /**
     * @brief Validates input dimensions, finiteness, symmetry, and positive semidefiniteness of Q.
     */
    [[nodiscard]] std::optional<std::string> validate_input(
        const QuadraticProgram& qp,
        bool& out_is_strictly_pd,
        bool& out_is_zero_q) const;

    /**
     * @brief Solves an unconstrained convex QP (m = 0).
     */
    [[nodiscard]] QPResult solve_unconstrained(
        const QuadraticProgram& qp,
        bool is_strictly_pd,
        bool is_zero_q) const;

    /**
     * @brief Solves an inequality-constrained convex QP via Primal-Dual Interior-Point Method.
     */
    [[nodiscard]] QPResult solve_interior_point(const QuadraticProgram& qp) const;

    /**
     * @brief Computes eigenvalues of a symmetric matrix using Jacobi eigenvalue algorithm.
     */
    [[nodiscard]] std::vector<double> compute_eigenvalues_symmetric(
        const Matrix<double>& M) const;
};

} // namespace vortex
