#include "simplex.hpp"
#include <limits>
#include <cmath>
#include <iomanip>
#include <sstream>

namespace vortex {

SimplexSolver::SimplexSolver(SimplexOptions options)
    : options_(options) {}

std::string to_string(SolverStatus status) {
    switch (status) {
        case SolverStatus::OPTIMAL:
            return "OPTIMAL";
        case SolverStatus::UNBOUNDED:
            return "UNBOUNDED";
        case SolverStatus::INFEASIBLE:
            return "INFEASIBLE";
        case SolverStatus::MAX_ITERATIONS_REACHED:
            return "MAX_ITERATIONS_REACHED";
        case SolverStatus::INVALID_INPUT:
            return "INVALID_INPUT";
    }
    return "UNKNOWN";
}

std::string to_string(ObjectiveSense sense) {
    return (sense == ObjectiveSense::MAXIMIZE) ? "MAXIMIZE" : "MINIMIZE";
}

std::string to_string(ConstraintType type) {
    switch (type) {
        case ConstraintType::LEQ: return "<=";
        case ConstraintType::GEQ: return ">=";
        case ConstraintType::EQ:  return "==";
    }
    return "?";
}

std::optional<std::string> SimplexSolver::validate_input_standard(
    const std::vector<double>& c,
    const Matrix<double>& A,
    const std::vector<double>& b) const {

    if (c.empty()) {
        return "Invalid input: objective vector c cannot be empty.";
    }
    if (A.rows() == 0 || A.cols() == 0) {
        return "Invalid input: constraint matrix A cannot have 0 rows or 0 columns.";
    }
    if (b.empty()) {
        return "Invalid input: right-hand side vector b cannot be empty.";
    }
    if (A.cols() != c.size()) {
        return "Dimension mismatch: A columns (" + std::to_string(A.cols()) +
               ") must equal c size (" + std::to_string(c.size()) + ").";
    }
    if (A.rows() != b.size()) {
        return "Dimension mismatch: A rows (" + std::to_string(A.rows()) +
               ") must equal b size (" + std::to_string(b.size()) + ").";
    }

    for (double val : c) {
        if (!std::isfinite(val)) return "Invalid input: objective vector c contains non-finite values (NaN or Inf).";
    }
    for (double val : b) {
        if (!std::isfinite(val)) return "Invalid input: right-hand side vector b contains non-finite values (NaN or Inf).";
    }
    if (!A.is_finite()) {
        return "Invalid input: constraint matrix A contains non-finite values (NaN or Inf).";
    }

    // Explicitly reject negative RHS values in standard Phase 2 solver
    for (std::size_t i = 0; i < b.size(); ++i) {
        if (b[i] < -options_.tolerance) {
            return "Unsupported input: negative RHS detected (b[" + std::to_string(i) +
                   "] = " + std::to_string(b[i]) + " < 0). Standard feasible tableau requires b >= 0.";
        }
    }

    return std::nullopt;
}

std::optional<std::string> SimplexSolver::validate_input_general(
    const std::vector<double>& c,
    const Matrix<double>& A,
    const std::vector<double>& b,
    const std::vector<ConstraintType>& constraint_types) const {

    if (c.empty()) {
        return "Invalid input: objective vector c cannot be empty.";
    }
    if (A.rows() == 0 || A.cols() == 0) {
        return "Invalid input: constraint matrix A cannot have 0 rows or 0 columns.";
    }
    if (b.empty()) {
        return "Invalid input: right-hand side vector b cannot be empty.";
    }
    if (A.cols() != c.size()) {
        return "Dimension mismatch: A columns (" + std::to_string(A.cols()) +
               ") must equal c size (" + std::to_string(c.size()) + ").";
    }
    if (A.rows() != b.size()) {
        return "Dimension mismatch: A rows (" + std::to_string(A.rows()) +
               ") must equal b size (" + std::to_string(b.size()) + ").";
    }
    if (constraint_types.size() != A.rows()) {
        return "Dimension mismatch: constraint_types size (" + std::to_string(constraint_types.size()) +
               ") must equal A rows (" + std::to_string(A.rows()) + ").";
    }

    for (double val : c) {
        if (!std::isfinite(val)) return "Invalid input: objective vector c contains non-finite values (NaN or Inf).";
    }
    for (double val : b) {
        if (!std::isfinite(val)) return "Invalid input: right-hand side vector b contains non-finite values (NaN or Inf).";
    }
    if (!A.is_finite()) {
        return "Invalid input: constraint matrix A contains non-finite values (NaN or Inf).";
    }

    return std::nullopt;
}

Matrix<double> SimplexSolver::build_initial_tableau_standard(
    const std::vector<double>& c,
    const Matrix<double>& A,
    const std::vector<double>& b,
    std::vector<std::size_t>& basis) const {

    const std::size_t m = A.rows();
    const std::size_t n = c.size();
    const std::size_t total_cols = n + m + 1;
    const std::size_t total_rows = m + 1;

    Matrix<double> tableau(total_rows, total_cols, 0.0);
    basis.resize(m);

    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            tableau(i, j) = A(i, j);
        }
        tableau(i, n + i) = 1.0;
        tableau(i, total_cols - 1) = (b[i] < 0.0) ? 0.0 : b[i];
        basis[i] = n + i;
    }

    for (std::size_t j = 0; j < n; ++j) {
        tableau(m, j) = -c[j];
    }
    tableau(m, total_cols - 1) = 0.0;

    return tableau;
}

int SimplexSolver::select_entering_column(const Matrix<double>& tableau,
                                         std::size_t num_candidate_cols,
                                         const std::vector<bool>& allowed_cols) const {
    const std::size_t obj_row = tableau.rows() - 1;
    int entering_col = -1;

    if (options_.pivot_rule == PivotRule::BLAND) {
        for (std::size_t j = 0; j < num_candidate_cols; ++j) {
            if (j < allowed_cols.size() && !allowed_cols[j]) continue;
            if (tableau(obj_row, j) < -options_.tolerance) {
                entering_col = static_cast<int>(j);
                break;
            }
        }
    } else {
        double most_negative = -options_.tolerance;
        for (std::size_t j = 0; j < num_candidate_cols; ++j) {
            if (j < allowed_cols.size() && !allowed_cols[j]) continue;
            const double val = tableau(obj_row, j);
            if (val < most_negative) {
                most_negative = val;
                entering_col = static_cast<int>(j);
            }
        }
    }

    return entering_col;
}

int SimplexSolver::select_leaving_row(const Matrix<double>& tableau,
                                      std::size_t entering_col,
                                      std::size_t num_constraints,
                                      const std::vector<std::size_t>& basis) const {
    const std::size_t rhs_col = tableau.cols() - 1;
    int leaving_row = -1;
    double min_ratio = std::numeric_limits<double>::infinity();

    for (std::size_t i = 0; i < num_constraints; ++i) {
        const double coeff = tableau(i, entering_col);

        if (coeff > options_.tolerance) {
            double rhs_val = tableau(i, rhs_col);
            if (rhs_val < 0.0) rhs_val = 0.0;

            const double ratio = rhs_val / coeff;

            if (leaving_row == -1 || ratio < (min_ratio - options_.tolerance)) {
                min_ratio = ratio;
                leaving_row = static_cast<int>(i);
            } else if (std::abs(ratio - min_ratio) <= options_.tolerance) {
                if (leaving_row != -1 && basis[i] < basis[leaving_row]) {
                    leaving_row = static_cast<int>(i);
                }
            }
        }
    }

    return leaving_row;
}

void SimplexSolver::pivot(Matrix<double>& tableau,
                          std::size_t pivot_row,
                          std::size_t pivot_col,
                          std::vector<std::size_t>& basis) {
    const std::size_t total_rows = tableau.rows();
    const std::size_t total_cols = tableau.cols();
    const double pivot_val = tableau(pivot_row, pivot_col);

    for (std::size_t j = 0; j < total_cols; ++j) {
        tableau(pivot_row, j) /= pivot_val;
    }
    tableau(pivot_row, pivot_col) = 1.0;

    for (std::size_t i = 0; i < total_rows; ++i) {
        if (i != pivot_row) {
            const double factor = tableau(i, pivot_col);
            if (std::abs(factor) > 1e-15) {
                for (std::size_t j = 0; j < total_cols; ++j) {
                    tableau(i, j) -= factor * tableau(pivot_row, j);
                }
                tableau(i, pivot_col) = 0.0;
            }
        }
    }

    basis[pivot_row] = pivot_col;

    for (std::size_t i = 0; i < total_rows; ++i) {
        for (std::size_t j = 0; j < total_cols; ++j) {
            if (std::abs(tableau(i, j)) < options_.tolerance * 1e-4) {
                tableau(i, j) = 0.0;
            }
        }
    }
}

// Phase 2 backward-compatible API
SimplexResult SimplexSolver::solve(const std::vector<double>& c,
                                   const Matrix<double>& A,
                                   const std::vector<double>& b) {
    SimplexResult result;

    auto validation_error = validate_input_standard(c, A, b);
    if (validation_error.has_value()) {
        result.status = SolverStatus::INVALID_INPUT;
        result.message = validation_error.value();
        return result;
    }

    const std::size_t m = A.rows();
    const std::size_t n = c.size();
    const std::size_t total_vars = n + m;
    const std::size_t rhs_col = total_vars;

    std::vector<std::size_t> basis;
    Matrix<double> tableau = build_initial_tableau_standard(c, A, b, basis);
    std::vector<bool> allowed_cols(total_vars, true);

    std::size_t iteration = 0;

    while (iteration < options_.max_iterations) {
        int entering_col = select_entering_column(tableau, total_vars, allowed_cols);

        if (entering_col == -1) {
            result.status = SolverStatus::OPTIMAL;
            result.message = "Optimal solution found successfully.";
            break;
        }

        int leaving_row = select_leaving_row(tableau, entering_col, m, basis);

        if (leaving_row == -1) {
            result.status = SolverStatus::UNBOUNDED;
            result.message = "Problem is unbounded: objective can increase indefinitely.";
            result.iterations = iteration;
            result.final_tableau = tableau;
            return result;
        }

        pivot(tableau, leaving_row, entering_col, basis);
        iteration++;
    }

    if (iteration >= options_.max_iterations && result.status != SolverStatus::OPTIMAL) {
        result.status = SolverStatus::MAX_ITERATIONS_REACHED;
        result.message = "Maximum iteration limit (" + std::to_string(options_.max_iterations) +
                         ") reached without achieving optimality.";
    }

    result.iterations = iteration;
    result.phase2_iterations = iteration;
    result.final_tableau = tableau;
    result.objective_value = tableau(m, rhs_col);

    result.variable_values.assign(n, 0.0);
    result.slack_values.assign(m, 0.0);

    for (std::size_t i = 0; i < m; ++i) {
        const std::size_t var_idx = basis[i];
        const double val = tableau(i, rhs_col);

        if (var_idx < n) {
            result.variable_values[var_idx] = val;
        } else if (var_idx < total_vars) {
            result.slack_values[var_idx - n] = val;
        }
    }

    return result;
}

SimplexResult SimplexSolver::solve(const LinearProgram& lp) {
    return solve(lp.c, lp.A, lp.b, lp.constraint_types, lp.sense);
}

// Phase 3 General Two-Phase Simplex API
SimplexResult SimplexSolver::solve(const std::vector<double>& c,
                                   const Matrix<double>& A,
                                   const std::vector<double>& b,
                                   const std::vector<ConstraintType>& constraint_types,
                                   ObjectiveSense sense) {
    SimplexResult result;

    auto val_err = validate_input_general(c, A, b, constraint_types);
    if (val_err.has_value()) {
        result.status = SolverStatus::INVALID_INPUT;
        result.message = val_err.value();
        return result;
    }

    const std::size_t m = A.rows();
    const std::size_t n = c.size();

    // Step 1: Normalize constraints so that all RHS values are non-negative
    Matrix<double> A_norm = A;
    std::vector<double> b_norm = b;
    std::vector<ConstraintType> types_norm = constraint_types;

    for (std::size_t i = 0; i < m; ++i) {
        if (b_norm[i] < -options_.tolerance) {
            for (std::size_t j = 0; j < n; ++j) {
                A_norm(i, j) = -A_norm(i, j);
            }
            b_norm[i] = -b_norm[i];
            if (types_norm[i] == ConstraintType::LEQ) {
                types_norm[i] = ConstraintType::GEQ;
            } else if (types_norm[i] == ConstraintType::GEQ) {
                types_norm[i] = ConstraintType::LEQ;
            }
        }
        if (std::abs(b_norm[i]) <= options_.tolerance) {
            b_norm[i] = 0.0;
        }
    }

    // Step 2: Determine Slack, Surplus, and Artificial variables
    std::vector<int> slack_indices(m, -1);
    std::vector<int> surplus_indices(m, -1);
    std::vector<int> artificial_indices(m, -1);

    std::size_t num_slack_surplus = 0;
    for (std::size_t i = 0; i < m; ++i) {
        if (types_norm[i] == ConstraintType::LEQ) {
            slack_indices[i] = static_cast<int>(n + num_slack_surplus);
            num_slack_surplus++;
        } else if (types_norm[i] == ConstraintType::GEQ) {
            surplus_indices[i] = static_cast<int>(n + num_slack_surplus);
            num_slack_surplus++;
        }
    }

    std::size_t num_artificial = 0;
    for (std::size_t i = 0; i < m; ++i) {
        if (types_norm[i] == ConstraintType::GEQ || types_norm[i] == ConstraintType::EQ) {
            artificial_indices[i] = static_cast<int>(n + num_slack_surplus + num_artificial);
            num_artificial++;
        }
    }

    const std::size_t total_vars = n + num_slack_surplus + num_artificial;
    const std::size_t total_cols = total_vars + 1; // last col is RHS
    const std::size_t rhs_col = total_vars;
    const std::size_t total_rows = m + 1; // constraints + obj row

    Matrix<double> tableau(total_rows, total_cols, 0.0);
    std::vector<std::size_t> basis(m);
    std::vector<bool> is_artificial(total_vars, false);

    for (std::size_t i = 0; i < m; ++i) {
        if (artificial_indices[i] != -1) {
            is_artificial[artificial_indices[i]] = true;
        }
    }

    // Populate constraint rows
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            tableau(i, j) = A_norm(i, j);
        }
        if (slack_indices[i] != -1) {
            tableau(i, slack_indices[i]) = 1.0;
            basis[i] = static_cast<std::size_t>(slack_indices[i]);
        }
        if (surplus_indices[i] != -1) {
            tableau(i, surplus_indices[i]) = -1.0;
        }
        if (artificial_indices[i] != -1) {
            tableau(i, artificial_indices[i]) = 1.0;
            basis[i] = static_cast<std::size_t>(artificial_indices[i]);
        }
        tableau(i, rhs_col) = b_norm[i];
    }

    std::size_t total_iterations = 0;

    // =========================================================================
    // PHASE I: Auxiliary problem to eliminate artificial variables
    // =========================================================================
    if (num_artificial > 0) {
        // Phase I Objective: Maximize -W = -sum(a_k)
        // Row m initially: +1 on artificial columns
        for (std::size_t i = 0; i < m; ++i) {
            if (artificial_indices[i] != -1) {
                tableau(m, artificial_indices[i]) = 1.0;
            }
        }

        // Canonical reduction: subtract rows with artificial basic variables
        for (std::size_t i = 0; i < m; ++i) {
            if (artificial_indices[i] != -1) {
                for (std::size_t j = 0; j < total_cols; ++j) {
                    tableau(m, j) -= tableau(i, j);
                }
            }
        }

        // Clean up small values
        for (std::size_t j = 0; j < total_cols; ++j) {
            if (std::abs(tableau(m, j)) < options_.tolerance * 1e-4) {
                tableau(m, j) = 0.0;
            }
        }

        // Phase I Simplex Pivoting
        std::vector<bool> phase1_allowed(total_vars, true);
        std::size_t p1_iter = 0;

        while (p1_iter < options_.max_iterations) {
            int entering_col = select_entering_column(tableau, total_vars, phase1_allowed);

            if (entering_col == -1) {
                break; // Phase I Optimal
            }

            int leaving_row = select_leaving_row(tableau, entering_col, m, basis);

            if (leaving_row == -1) {
                // If auxiliary is unbounded, artificials cannot be zeroed
                result.status = SolverStatus::INFEASIBLE;
                result.message = "Problem is infeasible: auxiliary Phase I problem is unbounded.";
                result.final_tableau = tableau;
                return result;
            }

            pivot(tableau, leaving_row, entering_col, basis);
            p1_iter++;
            total_iterations++;
        }

        result.phase1_iterations = p1_iter;

        // Check feasibility: W = -tableau(m, rhs_col)
        const double w_val = -tableau(m, rhs_col);
        if (w_val > options_.tolerance) {
            result.status = SolverStatus::INFEASIBLE;
            result.message = "Problem is infeasible: Phase I artificial sum W = " +
                             std::to_string(w_val) + " > 0.";
            result.final_tableau = tableau;
            result.iterations = total_iterations;
            return result;
        }

        // Drive any remaining artificial variables out of the basis (degenerate vertices)
        for (std::size_t i = 0; i < m; ++i) {
            if (is_artificial[basis[i]]) {
                for (std::size_t j = 0; j < n + num_slack_surplus; ++j) {
                    if (std::abs(tableau(i, j)) > options_.tolerance) {
                        pivot(tableau, i, j, basis);
                        break;
                    }
                }
            }
        }
    }

    // =========================================================================
    // PHASE II: Optimize original objective function
    // =========================================================================
    // Set up effective objective coefficients
    // If Minimizing: min c^T x <=> max -c^T x
    std::vector<double> c_eff = c;
    if (sense == ObjectiveSense::MINIMIZE) {
        for (auto& val : c_eff) val = -val;
    }

    // Reset objective row
    for (std::size_t j = 0; j < total_cols; ++j) {
        tableau(m, j) = 0.0;
    }

    // Insert original objective coefficients: Z - sum(c_j * x_j) = 0
    for (std::size_t j = 0; j < n; ++j) {
        tableau(m, j) = -c_eff[j];
    }

    // Bring objective row into canonical form for current basic variables
    for (std::size_t i = 0; i < m; ++i) {
        const std::size_t var = basis[i];
        if (var < n) {
            const double coeff = c_eff[var];
            if (std::abs(coeff) > options_.tolerance) {
                for (std::size_t j = 0; j < total_cols; ++j) {
                    tableau(m, j) += coeff * tableau(i, j);
                }
                tableau(m, var) = 0.0;
            }
        }
    }

    // Artificial variables must never re-enter during Phase II
    std::vector<bool> phase2_allowed(total_vars, true);
    for (std::size_t j = 0; j < total_vars; ++j) {
        if (is_artificial[j]) {
            phase2_allowed[j] = false;
        }
    }

    std::size_t p2_iter = 0;
    while (p2_iter < options_.max_iterations) {
        int entering_col = select_entering_column(tableau, total_vars, phase2_allowed);

        if (entering_col == -1) {
            result.status = SolverStatus::OPTIMAL;
            result.message = "Optimal solution found successfully via Two-Phase Simplex.";
            break;
        }

        int leaving_row = select_leaving_row(tableau, entering_col, m, basis);

        if (leaving_row == -1) {
            result.status = SolverStatus::UNBOUNDED;
            result.message = "Problem is unbounded: objective can improve indefinitely.";
            result.iterations = total_iterations;
            result.final_tableau = tableau;
            return result;
        }

        pivot(tableau, leaving_row, entering_col, basis);
        p2_iter++;
        total_iterations++;
    }

    result.phase2_iterations = p2_iter;
    result.iterations = total_iterations;

    if (p2_iter >= options_.max_iterations && result.status != SolverStatus::OPTIMAL) {
        result.status = SolverStatus::MAX_ITERATIONS_REACHED;
        result.message = "Maximum iteration limit reached during Phase II.";
    }

    // =========================================================================
    // Extract Solution
    // =========================================================================
    result.final_tableau = tableau;

    // Objective value
    double raw_obj = tableau(m, rhs_col);
    result.objective_value = (sense == ObjectiveSense::MINIMIZE) ? -raw_obj : raw_obj;

    // Decision variables
    result.variable_values.assign(n, 0.0);
    result.slack_values.assign(m, 0.0);
    result.surplus_values.assign(m, 0.0);

    for (std::size_t i = 0; i < m; ++i) {
        const std::size_t var_idx = basis[i];
        const double val = tableau(i, rhs_col);

        if (var_idx < n) {
            result.variable_values[var_idx] = val;
        }
    }

    // Extract slack and surplus values
    for (std::size_t i = 0; i < m; ++i) {
        if (slack_indices[i] != -1) {
            for (std::size_t r = 0; r < m; ++r) {
                if (basis[r] == static_cast<std::size_t>(slack_indices[i])) {
                    result.slack_values[i] = tableau(r, rhs_col);
                    break;
                }
            }
        }
        if (surplus_indices[i] != -1) {
            for (std::size_t r = 0; r < m; ++r) {
                if (basis[r] == static_cast<std::size_t>(surplus_indices[i])) {
                    result.surplus_values[i] = tableau(r, rhs_col);
                    break;
                }
            }
        }
    }

    return result;
}

} // namespace vortex
