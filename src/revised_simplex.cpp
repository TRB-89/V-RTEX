#include "vortex/revised_simplex.hpp"
#include <limits>
#include <cmath>
#include <iomanip>
#include <sstream>

namespace vortex {

RevisedSimplexSolver::RevisedSimplexSolver(RevisedSimplexOptions options)
    : options_(options) {}

std::optional<std::string> RevisedSimplexSolver::validate_input_standard(
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

    for (std::size_t i = 0; i < b.size(); ++i) {
        if (b[i] < -options_.tolerance) {
            return "Unsupported input: negative RHS detected (b[" + std::to_string(i) +
                   "] = " + std::to_string(b[i]) + " < 0). Standard feasible tableau requires b >= 0.";
        }
    }

    return std::nullopt;
}

std::optional<std::string> RevisedSimplexSolver::validate_input_general(
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

    return std::nullopt;
}

Matrix<double> RevisedSimplexSolver::extract_basis_matrix(
    const Matrix<double>& A_std,
    const std::vector<std::size_t>& basis) const {

    const std::size_t m = A_std.rows();
    if (basis.size() != m) {
        throw std::invalid_argument("extract_basis_matrix: basis size must match number of rows.");
    }

    Matrix<double> B(m, m);
    for (std::size_t col = 0; col < m; ++col) {
        const std::size_t var_idx = basis[col];
        for (std::size_t row = 0; row < m; ++row) {
            B(row, col) = A_std(row, var_idx);
        }
    }
    return B;
}

std::vector<double> RevisedSimplexSolver::extract_basis_costs(
    const std::vector<double>& c_std,
    const std::vector<std::size_t>& basis) const {

    std::vector<double> c_B(basis.size());
    for (std::size_t i = 0; i < basis.size(); ++i) {
        c_B[i] = c_std[basis[i]];
    }
    return c_B;
}

std::vector<double> RevisedSimplexSolver::compute_primal_solution(
    const LUDecomposition<double>& lu,
    const std::vector<double>& b) const {
    return lu.solve(b);
}

std::vector<double> RevisedSimplexSolver::compute_dual_multipliers(
    const LUDecomposition<double>& lu,
    const std::vector<double>& c_B) const {
    return lu.solve_transpose(c_B);
}

std::vector<double> RevisedSimplexSolver::compute_reduced_costs(
    const std::vector<double>& c_std,
    const Matrix<double>& A_std,
    const std::vector<double>& pi) const {

    const std::size_t num_vars = c_std.size();
    const std::size_t m = A_std.rows();
    std::vector<double> reduced_costs(num_vars, 0.0);

    for (std::size_t j = 0; j < num_vars; ++j) {
        double pi_aj = 0.0;
        for (std::size_t i = 0; i < m; ++i) {
            pi_aj += pi[i] * A_std(i, j);
        }
        reduced_costs[j] = c_std[j] - pi_aj;
    }

    return reduced_costs;
}

std::vector<double> RevisedSimplexSolver::compute_step_direction(
    const LUDecomposition<double>& lu,
    const Matrix<double>& A_std,
    std::size_t entering_col) const {

    std::vector<double> a_col = A_std.get_column(entering_col);
    return lu.solve(a_col);
}

int RevisedSimplexSolver::select_entering_variable(
    const std::vector<double>& reduced_costs,
    const std::vector<std::size_t>& basis,
    const std::vector<bool>& allowed_variables) const {

    const std::size_t num_vars = reduced_costs.size();
    std::vector<bool> is_basic(num_vars, false);
    for (std::size_t idx : basis) {
        if (idx < num_vars) {
            is_basic[idx] = true;
        }
    }

    int entering_var = -1;

    if (options_.pivot_rule == PivotRule::BLAND) {
        // Bland's Rule: smallest subscript among eligible candidates with reduced_cost > tolerance
        for (std::size_t j = 0; j < num_vars; ++j) {
            if (is_basic[j]) continue;
            if (j < allowed_variables.size() && !allowed_variables[j]) continue;

            if (reduced_costs[j] > options_.tolerance) {
                entering_var = static_cast<int>(j);
                break;
            }
        }
    } else {
        // Dantzig's Rule: steepest edge (maximum positive reduced cost)
        double max_reduced_cost = options_.tolerance;

        for (std::size_t j = 0; j < num_vars; ++j) {
            if (is_basic[j]) continue;
            if (j < allowed_variables.size() && !allowed_variables[j]) continue;

            if (reduced_costs[j] > max_reduced_cost) {
                max_reduced_cost = reduced_costs[j];
                entering_var = static_cast<int>(j);
            }
        }
    }

    return entering_var;
}

int RevisedSimplexSolver::select_leaving_variable(
    const std::vector<double>& x_B,
    const std::vector<double>& d,
    const std::vector<std::size_t>& basis) const {

    const std::size_t m = x_B.size();
    int leaving_row = -1;
    double min_ratio = std::numeric_limits<double>::infinity();

    for (std::size_t i = 0; i < m; ++i) {
        if (d[i] > options_.tolerance) {
            const double rhs = (x_B[i] < 0.0) ? 0.0 : x_B[i];
            const double ratio = rhs / d[i];

            if (ratio < min_ratio - options_.tolerance) {
                min_ratio = ratio;
                leaving_row = static_cast<int>(i);
            } else if (std::abs(ratio - min_ratio) <= options_.tolerance) {
                // Bland tie-break: choose the one with smaller variable index in basis
                if (leaving_row != -1 && basis[i] < basis[static_cast<std::size_t>(leaving_row)]) {
                    leaving_row = static_cast<int>(i);
                }
            }
        }
    }

    return leaving_row;
}

bool RevisedSimplexSolver::check_numerical_stability(
    const LUDecomposition<double>& lu,
    const Matrix<double>& B_mat,
    const std::vector<double>& x_B,
    const std::vector<double>& b,
    const std::vector<double>& pi,
    const std::vector<double>& c_B,
    double& primal_res,
    double& dual_res) const {

    primal_res = lu.residual_norm(B_mat, x_B, b);
    dual_res = lu.residual_norm_transpose(B_mat, pi, c_B);

    return (primal_res <= options_.max_allowed_residual &&
            dual_res <= options_.max_allowed_residual &&
            !lu.is_singular());
}

RevisedSimplexResult RevisedSimplexSolver::run_revised_simplex_loop(
    const std::vector<double>& c_std,
    const Matrix<double>& A_std,
    const std::vector<double>& b,
    std::vector<std::size_t>& basis,
    const std::vector<bool>& allowed_variables,
    std::size_t max_iters) {

    const std::size_t m = A_std.rows();
    RevisedSimplexResult result;
    result.iterations = 0;
    result.basis_indices = basis;

    LUDecomposition<double> lu;

    while (result.iterations < max_iters) {
        // 1. Form Basis matrix B and compute LU decomposition
        Matrix<double> B_mat = extract_basis_matrix(A_std, basis);
        bool non_singular = lu.compute(B_mat, options_.pivot_tolerance);

        if (!non_singular || lu.is_singular()) {
            result.status = SolverStatus::INVALID_INPUT;
            result.message = "Revised Simplex error: Basis matrix is singular or ill-conditioned.";
            return result;
        }

        // 2. Compute primal basic solution x_B = B^-1 * b
        std::vector<double> x_B = compute_primal_solution(lu, b);

        // Filter small numerical noise in feasibility
        for (std::size_t i = 0; i < m; ++i) {
            if (x_B[i] < 0.0 && x_B[i] >= -options_.tolerance) {
                x_B[i] = 0.0;
            }
        }

        // 3. Compute dual multipliers: B^T * pi = c_B
        std::vector<double> c_B = extract_basis_costs(c_std, basis);
        std::vector<double> pi = compute_dual_multipliers(lu, c_B);

        // 4. Numerical stability check
        double primal_res = 0.0, dual_res = 0.0;
        [[maybe_unused]] bool is_stable = check_numerical_stability(
            lu, B_mat, x_B, b, pi, c_B, primal_res, dual_res);
        result.max_primal_residual = std::max(result.max_primal_residual, primal_res);
        result.max_dual_residual = std::max(result.max_dual_residual, dual_res);
        result.basis_condition = lu.condition_ratio();

        // 5. Pricing: compute reduced costs c_bar_j = c_j - pi^T * A_j
        std::vector<double> reduced_costs = compute_reduced_costs(c_std, A_std, pi);

        // 6. Select entering variable
        int entering_var = select_entering_variable(reduced_costs, basis, allowed_variables);

        if (entering_var == -1) {
            // Optimality reached: all eligible reduced costs <= tolerance
            result.status = SolverStatus::OPTIMAL;
            result.basis_indices = basis;
            result.dual_values = pi;
            result.reduced_costs = reduced_costs;

            double z = 0.0;
            for (std::size_t i = 0; i < m; ++i) {
                z += c_B[i] * x_B[i];
            }
            result.objective_value = z;
            return result;
        }

        // 7. Compute step direction d = B^-1 * A_enter
        std::vector<double> d = compute_step_direction(lu, A_std, static_cast<std::size_t>(entering_var));

        // 8. Minimum ratio test: select leaving variable
        int leaving_row = select_leaving_variable(x_B, d, basis);

        if (leaving_row == -1) {
            // Problem is unbounded: entering variable can increase infinitely
            result.status = SolverStatus::UNBOUNDED;
            result.basis_indices = basis;
            result.message = "Objective is unbounded (no positive entries in pivot column direction).";
            return result;
        }

        // 9. Execute Basis update
        basis[static_cast<std::size_t>(leaving_row)] = static_cast<std::size_t>(entering_var);
        result.iterations++;
    }

    result.status = SolverStatus::MAX_ITERATIONS_REACHED;
    result.basis_indices = basis;
    result.message = "Maximum iteration limit (" + std::to_string(max_iters) + ") reached.";
    return result;
}

RevisedSimplexResult RevisedSimplexSolver::solve(
    const std::vector<double>& c,
    const Matrix<double>& A,
    const std::vector<double>& b) {

    auto validation_err = validate_input_standard(c, A, b);
    if (validation_err) {
        RevisedSimplexResult res;
        res.status = SolverStatus::INVALID_INPUT;
        res.message = *validation_err;
        return res;
    }

    const std::size_t m = A.rows();
    const std::size_t n = c.size();
    const std::size_t total_cols = n + m;

    // Build standard form [A | I]
    Matrix<double> A_std(m, total_cols, 0.0);
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            A_std(i, j) = A(i, j);
        }
        A_std(i, n + i) = 1.0; // Slack variable
    }

    std::vector<double> c_std(total_cols, 0.0);
    for (std::size_t j = 0; j < n; ++j) {
        c_std[j] = c[j];
    }

    // Initial basis: slack variables
    std::vector<std::size_t> basis(m);
    for (std::size_t i = 0; i < m; ++i) {
        basis[i] = n + i;
    }

    std::vector<bool> allowed_variables(total_cols, true);

    RevisedSimplexResult res = run_revised_simplex_loop(
        c_std, A_std, b, basis, allowed_variables, options_.max_iterations);

    if (res.status == SolverStatus::OPTIMAL) {
        res.message = "Optimal solution found successfully via Revised Simplex.";

        // Compute final x_B
        Matrix<double> B_mat = extract_basis_matrix(A_std, basis);
        LUDecomposition<double> lu(B_mat, options_.pivot_tolerance);
        std::vector<double> x_B = compute_primal_solution(lu, b);

        res.variable_values.assign(n, 0.0);
        res.slack_values.assign(m, 0.0);

        for (std::size_t i = 0; i < m; ++i) {
            const std::size_t var_idx = basis[i];
            const double val = (x_B[i] < 0.0) ? 0.0 : x_B[i];
            if (var_idx < n) {
                res.variable_values[var_idx] = val;
            } else if (var_idx < total_cols) {
                res.slack_values[var_idx - n] = val;
            }
        }

        // Objective calculation
        double z = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
            z += c[j] * res.variable_values[j];
        }
        res.objective_value = z;
    }

    return res;
}

RevisedSimplexResult RevisedSimplexSolver::solve(
    const std::vector<double>& c,
    const Matrix<double>& A,
    const std::vector<double>& b,
    const std::vector<ConstraintType>& constraint_types,
    ObjectiveSense sense) {

    auto validation_err = validate_input_general(c, A, b, constraint_types);
    if (validation_err) {
        RevisedSimplexResult res;
        res.status = SolverStatus::INVALID_INPUT;
        res.message = *validation_err;
        return res;
    }

    const std::size_t m = A.rows();
    const std::size_t n = c.size();

    // 1. Normalize minimization to maximization
    std::vector<double> c_norm = c;
    if (sense == ObjectiveSense::MINIMIZE) {
        for (auto& val : c_norm) val = -val;
    }

    // 2. Normalize negative RHS by multiplying row by -1 and flipping inequalities
    Matrix<double> A_norm = A;
    std::vector<double> b_norm = b;
    std::vector<ConstraintType> types_norm = constraint_types;

    for (std::size_t i = 0; i < m; ++i) {
        if (b_norm[i] < -options_.tolerance) {
            b_norm[i] = -b_norm[i];
            for (std::size_t j = 0; j < n; ++j) {
                A_norm(i, j) = -A_norm(i, j);
            }
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

    // 3. Count required auxiliary variables
    std::size_t num_slacks = 0;
    std::size_t num_surplus = 0;
    std::size_t num_artificial = 0;

    for (std::size_t i = 0; i < m; ++i) {
        if (types_norm[i] == ConstraintType::LEQ) {
            num_slacks++;
        } else if (types_norm[i] == ConstraintType::GEQ) {
            num_surplus++;
            num_artificial++;
        } else if (types_norm[i] == ConstraintType::EQ) {
            num_artificial++;
        }
    }

    const std::size_t slack_offset = n;
    const std::size_t surplus_offset = slack_offset + num_slacks;
    const std::size_t artificial_offset = surplus_offset + num_surplus;
    const std::size_t total_cols = artificial_offset + num_artificial;

    // 4. Construct standard-form constraint matrix A_std
    Matrix<double> A_std(m, total_cols, 0.0);
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            A_std(i, j) = A_norm(i, j);
        }
    }

    std::vector<std::size_t> basis(m);
    std::size_t cur_slack = 0;
    std::size_t cur_surplus = 0;
    std::size_t cur_artificial = 0;

    std::vector<std::size_t> row_slack_map(m, total_cols);
    std::vector<std::size_t> row_surplus_map(m, total_cols);

    for (std::size_t i = 0; i < m; ++i) {
        if (types_norm[i] == ConstraintType::LEQ) {
            const std::size_t col = slack_offset + cur_slack++;
            A_std(i, col) = 1.0;
            basis[i] = col;
            row_slack_map[i] = col;
        } else if (types_norm[i] == ConstraintType::GEQ) {
            const std::size_t col_surplus = surplus_offset + cur_surplus++;
            const std::size_t col_art = artificial_offset + cur_artificial++;
            A_std(i, col_surplus) = -1.0;
            A_std(i, col_art) = 1.0;
            basis[i] = col_art;
            row_surplus_map[i] = col_surplus;
        } else if (types_norm[i] == ConstraintType::EQ) {
            const std::size_t col_art = artificial_offset + cur_artificial++;
            A_std(i, col_art) = 1.0;
            basis[i] = col_art;
        }
    }

    std::size_t phase1_iters = 0;
    std::size_t phase2_iters = 0;

    // 5. Execute Phase I if artificial variables are present
    if (num_artificial > 0) {
        // Phase I costs: maximize -sum(a_i)
        std::vector<double> c_phase1(total_cols, 0.0);
        for (std::size_t j = artificial_offset; j < total_cols; ++j) {
            c_phase1[j] = -1.0;
        }

        std::vector<bool> allowed_p1(total_cols, true);

        RevisedSimplexResult p1_res = run_revised_simplex_loop(
            c_phase1, A_std, b_norm, basis, allowed_p1, options_.max_iterations);

        phase1_iters = p1_res.iterations;

        if (p1_res.status != SolverStatus::OPTIMAL) {
            RevisedSimplexResult res;
            res.status = p1_res.status;
            res.phase1_iterations = phase1_iters;
            res.iterations = phase1_iters;
            res.message = "Phase I failed: " + p1_res.message;
            return res;
        }

        // Check Phase I objective feasibility: sum of artificials must be 0
        Matrix<double> B_mat = extract_basis_matrix(A_std, basis);
        LUDecomposition<double> lu(B_mat, options_.pivot_tolerance);
        std::vector<double> x_B = compute_primal_solution(lu, b_norm);

        double artificial_sum = 0.0;
        for (std::size_t i = 0; i < m; ++i) {
            if (basis[i] >= artificial_offset) {
                artificial_sum += (x_B[i] > 0.0 ? x_B[i] : 0.0);
            }
        }

        if (artificial_sum > options_.tolerance) {
            RevisedSimplexResult res;
            res.status = SolverStatus::INFEASIBLE;
            res.phase1_iterations = phase1_iters;
            res.iterations = phase1_iters;
            res.message = "Problem is infeasible: artificial variable sum W = " +
                          std::to_string(artificial_sum) + " > 0 at Phase I optimality.";
            return res;
        }

        // Drive out zero-valued artificials from the basis if possible
        for (std::size_t i = 0; i < m; ++i) {
            if (basis[i] >= artificial_offset) {
                for (std::size_t j = 0; j < artificial_offset; ++j) {
                    bool already_basic = false;
                    for (std::size_t b_idx : basis) {
                        if (b_idx == j) { already_basic = true; break; }
                    }
                    if (already_basic) continue;

                    std::vector<double> d = compute_step_direction(lu, A_std, j);
                    if (std::abs(d[i]) > options_.tolerance) {
                        basis[i] = j;
                        B_mat = extract_basis_matrix(A_std, basis);
                        lu.compute(B_mat, options_.pivot_tolerance);
                        break;
                    }
                }
            }
        }
    }

    // 6. Execute Phase II
    std::vector<double> c_phase2(total_cols, 0.0);
    for (std::size_t j = 0; j < n; ++j) {
        c_phase2[j] = c_norm[j];
    }

    // Forbid artificial variables from entering in Phase II
    std::vector<bool> allowed_p2(total_cols, true);
    for (std::size_t j = artificial_offset; j < total_cols; ++j) {
        allowed_p2[j] = false;
    }

    std::size_t remaining_iters = (options_.max_iterations > phase1_iters) ?
                                  (options_.max_iterations - phase1_iters) : 0;

    RevisedSimplexResult p2_res = run_revised_simplex_loop(
        c_phase2, A_std, b_norm, basis, allowed_p2, remaining_iters);

    phase2_iters = p2_res.iterations;

    RevisedSimplexResult res;
    res.status = p2_res.status;
    res.phase1_iterations = phase1_iters;
    res.phase2_iterations = phase2_iters;
    res.iterations = phase1_iters + phase2_iters;
    res.basis_indices = basis;
    res.dual_values = p2_res.dual_values;
    res.reduced_costs = p2_res.reduced_costs;
    res.max_primal_residual = std::max(p2_res.max_primal_residual, 0.0);
    res.max_dual_residual = std::max(p2_res.max_dual_residual, 0.0);
    res.basis_condition = p2_res.basis_condition;

    if (res.status == SolverStatus::OPTIMAL) {
        res.message = "Optimal solution found successfully via Two-Phase Revised Simplex.";

        Matrix<double> B_mat = extract_basis_matrix(A_std, basis);
        LUDecomposition<double> lu(B_mat, options_.pivot_tolerance);
        std::vector<double> x_B = compute_primal_solution(lu, b_norm);

        res.variable_values.assign(n, 0.0);
        res.slack_values.assign(num_slacks, 0.0);
        res.surplus_values.assign(num_surplus, 0.0);

        for (std::size_t i = 0; i < m; ++i) {
            const std::size_t var_idx = basis[i];
            const double val = (x_B[i] < 0.0) ? 0.0 : x_B[i];

            if (var_idx < n) {
                res.variable_values[var_idx] = val;
            } else if (var_idx >= slack_offset && var_idx < surplus_offset) {
                res.slack_values[var_idx - slack_offset] = val;
            } else if (var_idx >= surplus_offset && var_idx < artificial_offset) {
                res.surplus_values[var_idx - surplus_offset] = val;
            }
        }

        // Calculate original objective value
        double z = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
            z += c[j] * res.variable_values[j];
        }
        res.objective_value = z;

        // If sense was MINIMIZE, dual prices match minimization form
        if (sense == ObjectiveSense::MINIMIZE) {
            for (auto& pi_val : res.dual_values) {
                pi_val = -pi_val;
            }
        }
    } else {
        res.message = p2_res.message;
    }

    return res;
}

RevisedSimplexResult RevisedSimplexSolver::solve(const LinearProgram& lp) {
    return solve(lp.c, lp.A, lp.b, lp.constraint_types, lp.sense);
}

} // namespace vortex
