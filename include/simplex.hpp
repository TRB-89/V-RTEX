#pragma once

#include "vortex/matrix.hpp"
#include <vector>
#include <string>
#include <optional>
#include <iostream>

namespace vortex {

/**
 * @brief Status resulting from solving a linear programming problem.
 */
enum class SolverStatus {
    OPTIMAL,                 ///< Found a unique or non-unique optimal solution
    UNBOUNDED,               ///< Problem is unbounded (objective can improve to infinity)
    INFEASIBLE,              ///< Problem has no feasible solution (Phase I artificials sum > 0)
    MAX_ITERATIONS_REACHED,  ///< Iteration limit reached without convergence
    INVALID_INPUT            ///< Input dimension mismatch, negative RHS (in Phase 2 standard), or invalid parameters
};

/**
 * @brief Pivot selection rule for the Simplex algorithm.
 */
enum class PivotRule {
    DANTZIG, ///< Most negative reduced cost (steepest edge ascent) with Bland tie-break
    BLAND    ///< Smallest subscript rule (guarantees anti-cycling on degenerate problems)
};

/**
 * @brief Optimization direction sense.
 */
enum class ObjectiveSense {
    MAXIMIZE, ///< Maximize c^T x
    MINIMIZE  ///< Minimize c^T x
};

/**
 * @brief Type of constraint inequality or equality.
 */
enum class ConstraintType {
    LEQ, ///< Less-than-or-equal (<=): requires slack variable (+1)
    GEQ, ///< Greater-than-or-equal (>=): requires surplus variable (-1) and artificial variable (+1)
    EQ   ///< Equality (=): requires artificial variable (+1)
};

/**
 * @brief Configuration parameters for the Simplex solver.
 */
struct SimplexOptions {
    double tolerance{1e-9};               ///< Numerical zero tolerance (epsilon)
    std::size_t max_iterations{10000};    ///< Maximum pivot iterations allowed
    PivotRule pivot_rule{PivotRule::DANTZIG}; ///< Pivot variable selection strategy
    bool verbose{false};                   ///< Print tableau steps to std::cout
};

/**
 * @brief Result structure returned by the Simplex solver.
 */
struct SimplexResult {
    SolverStatus status{SolverStatus::INVALID_INPUT};
    double objective_value{0.0};
    std::vector<double> variable_values;  ///< Values of original decision variables x
    std::vector<double> slack_values;     ///< Values of slack variables s
    std::vector<double> surplus_values;   ///< Values of surplus variables e
    std::size_t iterations{0};            ///< Total pivot iterations performed across all phases
    std::size_t phase1_iterations{0};     ///< Pivot iterations performed during Phase I
    std::size_t phase2_iterations{0};     ///< Pivot iterations performed during Phase II
    std::string message;                  ///< Status or diagnostic description
    Matrix<double> final_tableau;         ///< Snapshot of the final Simplex tableau
};

/**
 * @brief General Linear Program container.
 */
struct LinearProgram {
    ObjectiveSense sense{ObjectiveSense::MAXIMIZE};
    std::vector<double> c;
    Matrix<double> A;
    std::vector<double> b;
    std::vector<ConstraintType> constraint_types;
};

/**
 * @brief CPU Tableau-based Two-Phase Simplex Solver for General Linear Programs:
 *
 *   Optimize   Z = c^T * x   (Maximize or Minimize)
 *   Subject to A * x (<=, >=, =) b
 *              x >= 0
 *              (Negative RHS b_i fully supported via normalization)
 *
 * Supports slack, surplus, and artificial variables, Phase I auxiliary optimization,
 * Phase II canonical recovery, Gaussian elimination pivoting, Bland's anti-cycling safeguards,
 * unboundedness, and infeasibility detection.
 */
class SimplexSolver {
public:
    explicit SimplexSolver(SimplexOptions options = SimplexOptions{});

    /**
     * @brief Phase 2 API: Solves standard maximization LP with <= constraints and b >= 0.
     * Preserves strict Phase 2 input validation.
     */
    [[nodiscard]] SimplexResult solve(const std::vector<double>& c,
                                     const Matrix<double>& A,
                                     const std::vector<double>& b);

    /**
     * @brief Phase 3 General API: Solves general LP with <=, >=, =, negative RHS, and min/max.
     */
    [[nodiscard]] SimplexResult solve(const std::vector<double>& c,
                                     const Matrix<double>& A,
                                     const std::vector<double>& b,
                                     const std::vector<ConstraintType>& constraint_types,
                                     ObjectiveSense sense = ObjectiveSense::MAXIMIZE);

    /**
     * @brief Phase 3 General API: Solves general LinearProgram structure.
     */
    [[nodiscard]] SimplexResult solve(const LinearProgram& lp);

    [[nodiscard]] const SimplexOptions& options() const noexcept { return options_; }
    void set_options(const SimplexOptions& options) noexcept { options_ = options; }

private:
    SimplexOptions options_;

    /**
     * @brief Validates input dimensions for Phase 2 standard solver.
     */
    [[nodiscard]] std::optional<std::string> validate_input_standard(
        const std::vector<double>& c,
        const Matrix<double>& A,
        const std::vector<double>& b) const;

    /**
     * @brief Validates input dimensions and types for General LP solver.
     */
    [[nodiscard]] std::optional<std::string> validate_input_general(
        const std::vector<double>& c,
        const Matrix<double>& A,
        const std::vector<double>& b,
        const std::vector<ConstraintType>& constraint_types) const;

    /**
     * @brief Constructs initial canonical tableau with slack variables (Phase 2).
     */
    [[nodiscard]] Matrix<double> build_initial_tableau_standard(
        const std::vector<double>& c,
        const Matrix<double>& A,
        const std::vector<double>& b,
        std::vector<std::size_t>& basis) const;

    /**
     * @brief Selects the entering column using the configured PivotRule.
     */
    [[nodiscard]] int select_entering_column(const Matrix<double>& tableau,
                                             std::size_t num_candidate_cols,
                                             const std::vector<bool>& allowed_cols) const;

    /**
     * @brief Selects the leaving row using the minimum ratio test with Bland's tie-break.
     */
    [[nodiscard]] int select_leaving_row(const Matrix<double>& tableau,
                                         std::size_t entering_col,
                                         std::size_t num_constraints,
                                         const std::vector<std::size_t>& basis) const;

    /**
     * @brief Performs a full Gaussian elimination pivot operation.
     */
    void pivot(Matrix<double>& tableau,
               std::size_t pivot_row,
               std::size_t pivot_col,
               std::vector<std::size_t>& basis);
};

/**
 * @brief String conversion helpers.
 */
std::string to_string(SolverStatus status);
std::string to_string(ObjectiveSense sense);
std::string to_string(ConstraintType type);

} // namespace vortex
