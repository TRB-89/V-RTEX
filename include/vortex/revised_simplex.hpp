#pragma once

#include "vortex/matrix.hpp"
#include "vortex/lu_decomposition.hpp"
#include "simplex.hpp" // Reuse SolverStatus, PivotRule, ObjectiveSense, ConstraintType, LinearProgram
#include "vortex/solver_backend.hpp"
#include <vector>
#include <string>
#include <optional>
#include <iostream>

namespace vortex {

/**
 * @brief Configuration parameters for the Revised Simplex LP solver.
 */
struct RevisedSimplexOptions {
    double tolerance{1e-9};                     ///< Numerical zero tolerance (epsilon)
    double pivot_tolerance{1e-12};              ///< Threshold for singularity in LU decomposition
    std::size_t max_iterations{10000};          ///< Maximum pivot iterations allowed
    PivotRule pivot_rule{PivotRule::DANTZIG};   ///< Variable selection strategy (Dantzig or Bland)
    bool verbose{false};                        ///< Print iteration diagnostics to std::cout
    std::size_t refactor_interval{50};          ///< Re-factorize basis every N pivots
    double max_allowed_residual{1e-5};          ///< Numerical stability threshold
    SolverBackend backend{SolverBackend::AUTO}; ///< Computation backend (CPU, GPU, AUTO)
};

/**
 * @brief Result structure returned by the Revised Simplex solver.
 */
struct RevisedSimplexResult {
    SolverStatus status{SolverStatus::INVALID_INPUT};
    double objective_value{0.0};
    std::vector<double> variable_values;       ///< Values of original decision variables x
    std::vector<double> slack_values;          ///< Values of slack variables s
    std::vector<double> surplus_values;        ///< Values of surplus variables e
    std::vector<double> dual_values;           ///< Simplex multipliers / shadow prices pi
    std::vector<double> reduced_costs;         ///< Reduced costs for all variables
    std::vector<std::size_t> basis_indices;    ///< Column indices of basic variables
    std::size_t iterations{0};                 ///< Total pivot iterations
    std::size_t phase1_iterations{0};          ///< Iterations in Phase I
    std::size_t phase2_iterations{0};          ///< Iterations in Phase II
    double max_primal_residual{0.0};           ///< Max ||B * x_B - b||_inf observed
    double max_dual_residual{0.0};             ///< Max ||B^T * pi - c_B||_inf observed
    double basis_condition{0.0};               ///< Condition ratio of final basis matrix
    std::string message;                       ///< Status or diagnostic description
    SolverDeviceDiagnostics device_diagnostics; ///< Telemetry on backend selection and GPU execution
};

/**
 * @brief High-Performance Revised Simplex LP Solver using LU Decomposition with Partial Pivoting.
 *
 * Rather than maintaining and transforming an entire m x (n+m) tableau at every step,
 * the Revised Simplex method maintains only the m x m Basis matrix B and solves:
 *   1. Dual pricing system:   B^T * pi = c_B           (via LU backward/forward substitution)
 *   2. Reduced cost vector:   c_bar_j = c_j - pi^T * A_j
 *   3. Primal step direction: B * d = A_j_enter        (via LU forward/backward substitution)
 *   4. Primal basic values:   B * x_B = b
 *
 * This dramatically reduces memory footprint, computational complexity per pivot,
 * and eliminates accumulation of round-off errors via periodic or on-demand LU refactorization.
 */
class RevisedSimplexSolver {
public:
    explicit RevisedSimplexSolver(RevisedSimplexOptions options = RevisedSimplexOptions{});

    /**
     * @brief Phase 2 standard API: Solves standard maximization LP with <= constraints and b >= 0.
     */
    [[nodiscard]] RevisedSimplexResult solve(const std::vector<double>& c,
                                            const Matrix<double>& A,
                                            const std::vector<double>& b);

    /**
     * @brief General Two-Phase API: Solves general LP with <=, >=, =, negative RHS, and min/max.
     */
    [[nodiscard]] RevisedSimplexResult solve(const std::vector<double>& c,
                                            const Matrix<double>& A,
                                            const std::vector<double>& b,
                                            const std::vector<ConstraintType>& constraint_types,
                                            ObjectiveSense sense = ObjectiveSense::MAXIMIZE);

    /**
     * @brief General LinearProgram API: Solves general LinearProgram structure.
     */
    [[nodiscard]] RevisedSimplexResult solve(const LinearProgram& lp);

    // ==========================================
    // Basis Matrix Operations (Exposed for modularity & unit testing)
    // ==========================================

    /**
     * @brief Extracts the m x m Basis matrix B from standard-form constraint matrix A_std.
     */
    [[nodiscard]] Matrix<double> extract_basis_matrix(
        const Matrix<double>& A_std,
        const std::vector<std::size_t>& basis) const;

    /**
     * @brief Extracts the m-element cost vector c_B corresponding to basic variables.
     */
    [[nodiscard]] std::vector<double> extract_basis_costs(
        const std::vector<double>& c_std,
        const std::vector<std::size_t>& basis) const;

    /**
     * @brief Computes primal basic variable values by solving B * x_B = b.
     */
    [[nodiscard]] std::vector<double> compute_primal_solution(
        const LUDecomposition<double>& lu,
        const std::vector<double>& b) const;

    /**
     * @brief Computes dual simplex multipliers by solving B^T * pi = c_B.
     */
    [[nodiscard]] std::vector<double> compute_dual_multipliers(
        const LUDecomposition<double>& lu,
        const std::vector<double>& c_B) const;

    /**
     * @brief Computes reduced costs: c_bar_j = c_j - pi^T * A_j for all candidate columns.
     */
    [[nodiscard]] std::vector<double> compute_reduced_costs(
        const std::vector<double>& c_std,
        const Matrix<double>& A_std,
        const std::vector<double>& pi) const;

    /**
     * @brief Computes search direction d by solving B * d = A_j_enter.
     */
    [[nodiscard]] std::vector<double> compute_step_direction(
        const LUDecomposition<double>& lu,
        const Matrix<double>& A_std,
        std::size_t entering_col) const;

    /**
     * @brief Selects the entering column using Dantzig or Bland's rule.
     * @return index of entering column, or -1 if current basis is optimal.
     */
    [[nodiscard]] int select_entering_variable(
        const std::vector<double>& reduced_costs,
        const std::vector<std::size_t>& basis,
        const std::vector<bool>& allowed_variables) const;

    /**
     * @brief Selects the leaving basic row using minimum ratio test with Bland tie-break.
     * @return row index in basis (0 to m-1), or -1 if problem is unbounded.
     */
    [[nodiscard]] int select_leaving_variable(
        const std::vector<double>& x_B,
        const std::vector<double>& d,
        const std::vector<std::size_t>& basis) const;

    /**
     * @brief Validates primal residual ||B * x_B - b|| and dual residual ||B^T * pi - c_B||.
     */
    [[nodiscard]] bool check_numerical_stability(
        const LUDecomposition<double>& lu,
        const Matrix<double>& B_mat,
        const std::vector<double>& x_B,
        const std::vector<double>& b,
        const std::vector<double>& pi,
        const std::vector<double>& c_B,
        double& primal_res,
        double& dual_res) const;

    [[nodiscard]] const RevisedSimplexOptions& options() const noexcept { return options_; }
    void set_options(const RevisedSimplexOptions& options) noexcept { options_ = options; }

private:
    RevisedSimplexOptions options_;

    /**
     * @brief Core revised simplex optimization loop on a standard-form formulation:
     *   Maximize c_std^T * x
     *   Subject to A_std * x = b, x >= 0
     */
    [[nodiscard]] RevisedSimplexResult run_revised_simplex_loop(
        const std::vector<double>& c_std,
        const Matrix<double>& A_std,
        const std::vector<double>& b,
        std::vector<std::size_t>& basis,
        const std::vector<bool>& allowed_variables,
        std::size_t max_iters);

    /**
     * @brief Validates input dimensions for standard Phase 2 solver.
     */
    [[nodiscard]] std::optional<std::string> validate_input_standard(
        const std::vector<double>& c,
        const Matrix<double>& A,
        const std::vector<double>& b) const;

    /**
     * @brief Validates input dimensions for general LP solver.
     */
    [[nodiscard]] std::optional<std::string> validate_input_general(
        const std::vector<double>& c,
        const Matrix<double>& A,
        const std::vector<double>& b,
        const std::vector<ConstraintType>& constraint_types) const;
};

} // namespace vortex
