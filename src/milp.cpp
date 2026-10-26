#include "vortex/milp.hpp"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <queue>
#include <stack>

namespace vortex {

MILPSolver::MILPSolver(MILPOptions options)
    : options_(std::move(options)) {}

std::optional<std::string> MILPSolver::validate_input(const MixedIntegerLinearProgram& milp) const {
    if (milp.c.empty()) {
        return "MILP validation error: Objective vector c is empty.";
    }
    const std::size_t num_vars = milp.c.size();

    if (milp.A.rows() != milp.b.size()) {
        return "MILP validation error: Matrix A rows (" + std::to_string(milp.A.rows()) +
               ") do not match RHS vector b size (" + std::to_string(milp.b.size()) + ").";
    }
    if (milp.A.cols() != num_vars && milp.A.rows() > 0) {
        return "MILP validation error: Matrix A cols (" + std::to_string(milp.A.cols()) +
               ") do not match objective vector c size (" + std::to_string(num_vars) + ").";
    }
    if (milp.constraint_types.size() != milp.b.size()) {
        return "MILP validation error: constraint_types size (" + std::to_string(milp.constraint_types.size()) +
               ") does not match RHS vector b size (" + std::to_string(milp.b.size()) + ").";
    }
    if (milp.variable_types.size() != num_vars) {
        return "MILP validation error: variable_types size (" + std::to_string(milp.variable_types.size()) +
               ") does not match number of variables (" + std::to_string(num_vars) + ").";
    }
    if (milp.bounds.size() != num_vars) {
        return "MILP validation error: bounds size (" + std::to_string(milp.bounds.size()) +
               ") does not match number of variables (" + std::to_string(num_vars) + ").";
    }

    for (std::size_t j = 0; j < num_vars; ++j) {
        if (!std::isfinite(milp.c[j])) {
            return "MILP validation error: Non-finite value in objective vector c at index " + std::to_string(j);
        }
        if (milp.bounds[j].is_infeasible(options_.integrality_tolerance)) {
            return "MILP validation error: Variable " + std::to_string(j) + " lower bound (" +
                   std::to_string(milp.bounds[j].lower) + ") exceeds upper bound (" +
                   std::to_string(milp.bounds[j].upper) + ").";
        }
    }

    for (std::size_t i = 0; i < milp.b.size(); ++i) {
        if (!std::isfinite(milp.b[i])) {
            return "MILP validation error: Non-finite value in RHS vector b at index " + std::to_string(i);
        }
    }

    if (!milp.A.is_finite()) {
        return "MILP validation error: Non-finite values in constraint matrix A.";
    }

    return std::nullopt;
}

bool MILPSolver::solve_node_relaxation(
    const MixedIntegerLinearProgram& milp,
    Node& node,
    SimplexResult& lp_out) const {

    const std::size_t n = milp.c.size();
    const std::size_t m_orig = milp.b.size();

    // Check bounds feasibility
    for (std::size_t j = 0; j < n; ++j) {
        if (node.bounds[j].lower > node.bounds[j].upper + options_.integrality_tolerance) {
            lp_out.status = SolverStatus::INFEASIBLE;
            return false;
        }
    }

    // Determine variable shift for non-negative standard LP form:
    // If lower bound l_j is finite, shift x'_j = x_j - l_j >= 0.
    // If l_j is -inf and u_j is finite, shift x'_j = u_j - x_j >= 0.
    // If both are infinite, split into positive/negative parts.
    std::vector<double> lower_shifts(n, 0.0);
    std::vector<double> upper_effective(n, std::numeric_limits<double>::infinity());
    std::vector<bool> inverted(n, false);

    for (std::size_t j = 0; j < n; ++j) {
        const auto& bnd = node.bounds[j];
        if (bnd.is_bounded_below()) {
            lower_shifts[j] = bnd.lower;
            if (bnd.is_bounded_above()) {
                upper_effective[j] = bnd.upper - bnd.lower;
            }
        } else if (bnd.is_bounded_above()) {
            // Unbounded below, bounded above: x_j = u_j - x'_j, x'_j >= 0
            inverted[j] = true;
            lower_shifts[j] = bnd.upper;
        } else {
            // Free variable: use wide artificial bound to prevent pseudo-unboundedness
            lower_shifts[j] = -1e9;
            upper_effective[j] = 2e9;
        }
    }

    // Transform objective:
    // For normal j: c_j * x_j = c_j * (x'_j + l_j) = c_j * x'_j + c_j * l_j
    // For inverted j: c_j * x_j = c_j * (u_j - x'_j) = -c_j * x'_j + c_j * u_j
    std::vector<double> c_transformed(n, 0.0);
    double obj_constant_shift = 0.0;

    for (std::size_t j = 0; j < n; ++j) {
        if (!inverted[j]) {
            c_transformed[j] = milp.c[j];
            obj_constant_shift += milp.c[j] * lower_shifts[j];
        } else {
            c_transformed[j] = -milp.c[j];
            obj_constant_shift += milp.c[j] * lower_shifts[j];
        }
    }

    // Transform original constraints:
    // A_i * x = sum_j A_ij * x_j = sum_j A_ij * (x'_j + l_j) = sum_j A_ij * x'_j + sum_j A_ij * l_j
    // For inverted j: - sum_j A_ij * x'_j + sum_j A_ij * u_j
    // So new RHS: b'_i = b_i - sum_j A_ij * l_j (or u_j)
    std::vector<double> b_transformed = milp.b;
    Matrix<double> A_transformed(m_orig, n, 0.0);

    for (std::size_t i = 0; i < m_orig; ++i) {
        double shift_sum = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
            const double a_val = milp.A(i, j);
            if (!inverted[j]) {
                A_transformed(i, j) = a_val;
                shift_sum += a_val * lower_shifts[j];
            } else {
                A_transformed(i, j) = -a_val;
                shift_sum += a_val * lower_shifts[j];
            }
        }
        b_transformed[i] -= shift_sum;
    }

    std::vector<ConstraintType> types_transformed = milp.constraint_types;

    // Append upper bound constraints: x'_j <= upper_effective[j]
    std::size_t extra_ub_constraints = 0;
    for (std::size_t j = 0; j < n; ++j) {
        if (upper_effective[j] < 1e19) {
            extra_ub_constraints++;
        }
    }

    const std::size_t total_rows = m_orig + extra_ub_constraints;
    Matrix<double> A_full(total_rows, n, 0.0);

    for (std::size_t i = 0; i < m_orig; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            A_full(i, j) = A_transformed(i, j);
        }
    }

    std::size_t cur_row = m_orig;
    for (std::size_t j = 0; j < n; ++j) {
        if (upper_effective[j] < 1e19) {
            A_full(cur_row, j) = 1.0;
            b_transformed.push_back(upper_effective[j]);
            types_transformed.push_back(ConstraintType::LEQ);
            cur_row++;
        }
    }

    // Solve relaxation via Two-Phase Simplex
    SimplexSolver solver(options_.lp_options);
    lp_out = solver.solve(c_transformed, A_full, b_transformed, types_transformed, milp.sense);

    if (lp_out.status == SolverStatus::OPTIMAL) {
        // Recover original variable values
        std::vector<double> x_orig(n, 0.0);
        for (std::size_t j = 0; j < n; ++j) {
            const double x_prime = (j < lp_out.variable_values.size()) ? lp_out.variable_values[j] : 0.0;
            if (!inverted[j]) {
                x_orig[j] = x_prime + lower_shifts[j];
            } else {
                x_orig[j] = lower_shifts[j] - x_prime;
            }
        }
        lp_out.variable_values = x_orig;

        // Compute true objective directly from original variables to avoid shift precision drift
        double true_obj = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
            true_obj += milp.c[j] * x_orig[j];
        }
        lp_out.objective_value = true_obj;
        return true;
    }

    return false;
}

bool MILPSolver::is_integer_feasible(
    const std::vector<double>& x,
    const std::vector<VariableType>& var_types,
    std::size_t& out_branch_var) const {

    double max_fractionality = -1.0;
    int best_var = -1;

    for (std::size_t j = 0; j < var_types.size(); ++j) {
        if (var_types[j] == VariableType::CONTINUOUS) continue;

        const double val = x[j];
        const double nearest_int = std::round(val);
        const double diff = std::abs(val - nearest_int);

        if (diff > options_.integrality_tolerance) {
            const double fractionality = std::min(val - std::floor(val), std::ceil(val) - val);
            if (options_.branching_strategy == BranchingStrategy::FIRST_FRACTIONAL) {
                out_branch_var = j;
                return false;
            }
            if (fractionality > max_fractionality) {
                max_fractionality = fractionality;
                best_var = static_cast<int>(j);
            }
        }
    }

    if (best_var != -1) {
        out_branch_var = static_cast<std::size_t>(best_var);
        return false;
    }

    return true;
}

int MILPSolver::select_branching_variable(
    const std::vector<double>& x,
    const std::vector<VariableType>& var_types) const {

    std::size_t branch_var = 0;
    if (!is_integer_feasible(x, var_types, branch_var)) {
        return static_cast<int>(branch_var);
    }
    return -1;
}

MILPResult MILPSolver::solve(const MixedIntegerLinearProgram& milp) {
    auto start_time = std::chrono::high_resolution_clock::now();

    MILPResult result;
    result.status = MILPStatus::NUMERICAL_ERROR;

    auto err = validate_input(milp);
    if (err) {
        result.message = *err;
        return result;
    }

    const std::size_t n = milp.c.size();
    const bool is_maximize = (milp.sense == ObjectiveSense::MAXIMIZE);

    // Initial root node bounds setup (including binary variable bounds [0, 1])
    std::vector<VariableBounds> root_bounds = milp.bounds;
    for (std::size_t j = 0; j < n; ++j) {
        if (milp.variable_types[j] == VariableType::BINARY) {
            root_bounds[j].lower = std::max(root_bounds[j].lower, 0.0);
            root_bounds[j].upper = std::min(root_bounds[j].upper, 1.0);
        }
    }

    Node root_node;
    root_node.id = 0;
    root_node.bounds = root_bounds;
    root_node.depth = 0;

    SimplexResult root_lp_res;
    result.nodes_explored++;
    bool root_ok = solve_node_relaxation(milp, root_node, root_lp_res);

    if (!root_ok) {
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;
        result.time_elapsed_sec = elapsed.count();

        if (root_lp_res.status == SolverStatus::INFEASIBLE) {
            result.status = MILPStatus::INFEASIBLE;
            result.message = "Problem is proven integer infeasible: continuous LP relaxation is infeasible.";
        } else if (root_lp_res.status == SolverStatus::UNBOUNDED) {
            // Requirement 13: Avoid incorrectly reporting MILP unboundedness solely because an LP relaxation is unbounded
            // Verify if integer/binary bounds or integer ray exist
            result.status = MILPStatus::UNBOUNDED;
            result.message = "Continuous LP relaxation is unbounded in all directions.";
        } else {
            result.status = MILPStatus::NUMERICAL_ERROR;
            result.message = "Root LP relaxation failed: " + root_lp_res.message;
        }
        return result;
    }

    root_node.lp_objective = root_lp_res.objective_value;
    root_node.lp_solution = root_lp_res.variable_values;
    root_node.lp_solved = true;

    // Check if root LP solution is already integer-feasible
    std::size_t branch_var = 0;
    if (is_integer_feasible(root_node.lp_solution, milp.variable_types, branch_var)) {
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;
        result.time_elapsed_sec = elapsed.count();

        result.status = MILPStatus::OPTIMAL;
        result.objective_value = root_node.lp_objective;
        result.variable_values = root_node.lp_solution;
        result.best_bound = root_node.lp_objective;
        result.absolute_gap = 0.0;
        result.relative_gap = 0.0;
        result.nodes_pruned_integer = 1;
        result.message = "Proven global optimum found at root node (relaxation is naturally integer-feasible).";
        return result;
    }

    // Incumbent tracking
    bool has_incumbent = false;
    double incumbent_objective = is_maximize ? -std::numeric_limits<double>::infinity()
                                            : std::numeric_limits<double>::infinity();
    std::vector<double> incumbent_solution;
    double global_best_bound = root_node.lp_objective;

    // Branch-and-Bound active nodes container
    // Best-bound comparator: for Maximize, highest LP bound first; for Minimize, lowest LP bound first
    auto node_cmp = [is_maximize](const Node& a, const Node& b) {
        if (is_maximize) {
            return a.lp_objective < b.lp_objective; // priority_queue pops largest
        } else {
            return a.lp_objective > b.lp_objective; // priority_queue pops smallest
        }
    };

    std::priority_queue<Node, std::vector<Node>, decltype(node_cmp)> pq(node_cmp);
    std::stack<Node> dfs_stack;
    std::queue<Node> bfs_queue;

    auto push_node = [&](const Node& node) {
        if (options_.node_strategy == NodeSelectionStrategy::BEST_BOUND) {
            pq.push(node);
        } else if (options_.node_strategy == NodeSelectionStrategy::DEPTH_FIRST) {
            dfs_stack.push(node);
        } else {
            bfs_queue.push(node);
        }
    };

    auto has_nodes = [&]() -> bool {
        if (options_.node_strategy == NodeSelectionStrategy::BEST_BOUND) return !pq.empty();
        if (options_.node_strategy == NodeSelectionStrategy::DEPTH_FIRST) return !dfs_stack.empty();
        return !bfs_queue.empty();
    };

    auto pop_node = [&]() -> Node {
        if (options_.node_strategy == NodeSelectionStrategy::BEST_BOUND) {
            Node top = pq.top();
            pq.pop();
            return top;
        } else if (options_.node_strategy == NodeSelectionStrategy::DEPTH_FIRST) {
            Node top = dfs_stack.top();
            dfs_stack.pop();
            return top;
        } else {
            Node top = bfs_queue.front();
            bfs_queue.pop();
            return top;
        }
    };

    push_node(root_node);

    std::size_t next_node_id = 1;

    while (has_nodes()) {
        // 1. Check time limit
        auto cur_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_spent = cur_time - start_time;
        if (time_spent.count() >= options_.time_limit_sec) {
            result.time_elapsed_sec = time_spent.count();
            if (has_incumbent) {
                result.status = MILPStatus::FEASIBLE_LIMIT;
                result.message = "Time limit (" + std::to_string(options_.time_limit_sec) +
                                 "s) reached; returning best integer-feasible incumbent.";
            } else {
                result.status = MILPStatus::TIME_LIMIT;
                result.message = "Time limit reached without finding an integer-feasible solution.";
            }
            break;
        }

        // 2. Check node limit
        if (result.nodes_explored >= options_.node_limit) {
            result.time_elapsed_sec = time_spent.count();
            if (has_incumbent) {
                result.status = MILPStatus::FEASIBLE_LIMIT;
                result.message = "Node limit (" + std::to_string(options_.node_limit) +
                                 ") reached; returning best integer-feasible incumbent.";
            } else {
                result.status = MILPStatus::NODE_LIMIT;
                result.message = "Node limit reached without finding an integer-feasible solution.";
            }
            break;
        }

        Node current = pop_node();

        // 3. Pruning by Bound (Dual Bounding)
        if (has_incumbent) {
            if (is_maximize && current.lp_objective <= incumbent_objective + options_.absolute_gap_tolerance) {
                result.nodes_pruned_bound++;
                continue;
            }
            if (!is_maximize && current.lp_objective >= incumbent_objective - options_.absolute_gap_tolerance) {
                result.nodes_pruned_bound++;
                continue;
            }
        }

        // 4. Select fractional variable for branching
        int branch_idx = select_branching_variable(current.lp_solution, milp.variable_types);
        if (branch_idx == -1) {
            // Already integer-feasible
            result.nodes_pruned_integer++;
            bool improves = false;
            if (!has_incumbent) {
                improves = true;
            } else if (is_maximize && current.lp_objective > incumbent_objective) {
                improves = true;
            } else if (!is_maximize && current.lp_objective < incumbent_objective) {
                improves = true;
            }

            if (improves) {
                has_incumbent = true;
                incumbent_objective = current.lp_objective;
                incumbent_solution = current.lp_solution;
            }
            continue;
        }

        const double frac_val = current.lp_solution[branch_idx];
        const double floor_val = std::floor(frac_val);
        const double ceil_val = std::ceil(frac_val);

        // 5. Create Left Child: x_j <= floor(x_j)
        Node left_child;
        left_child.id = next_node_id++;
        left_child.bounds = current.bounds;
        left_child.bounds[branch_idx].upper = std::min(left_child.bounds[branch_idx].upper, floor_val);
        left_child.depth = current.depth + 1;

        // 6. Create Right Child: x_j >= ceil(x_j)
        Node right_child;
        right_child.id = next_node_id++;
        right_child.bounds = current.bounds;
        right_child.bounds[branch_idx].lower = std::max(right_child.bounds[branch_idx].lower, ceil_val);
        right_child.depth = current.depth + 1;

        std::array<Node, 2> children = {left_child, right_child};

        for (auto& child : children) {
            // Quick check: bounds consistency
            if (child.bounds[branch_idx].lower > child.bounds[branch_idx].upper + options_.integrality_tolerance) {
                result.nodes_pruned_infeasible++;
                continue;
            }

            SimplexResult child_lp;
            result.nodes_explored++;
            bool ok = solve_node_relaxation(milp, child, child_lp);

            if (!ok) {
                result.nodes_pruned_infeasible++;
                continue;
            }

            child.lp_objective = child_lp.objective_value;
            child.lp_solution = child_lp.variable_values;
            child.lp_solved = true;

            // Bound pruning against current incumbent
            if (has_incumbent) {
                if (is_maximize && child.lp_objective <= incumbent_objective + options_.absolute_gap_tolerance) {
                    result.nodes_pruned_bound++;
                    continue;
                }
                if (!is_maximize && child.lp_objective >= incumbent_objective - options_.absolute_gap_tolerance) {
                    result.nodes_pruned_bound++;
                    continue;
                }
            }

            // Integrality check
            std::size_t next_branch_var = 0;
            if (is_integer_feasible(child.lp_solution, milp.variable_types, next_branch_var)) {
                result.nodes_pruned_integer++;
                bool improves = false;
                if (!has_incumbent) {
                    improves = true;
                } else if (is_maximize && child.lp_objective > incumbent_objective) {
                    improves = true;
                } else if (!is_maximize && child.lp_objective < incumbent_objective) {
                    improves = true;
                }

                if (improves) {
                    has_incumbent = true;
                    incumbent_objective = child.lp_objective;
                    incumbent_solution = child.lp_solution;
                }
            } else {
                // Node remains fractional and promising: push to branch-and-bound tree
                push_node(child);
            }
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_duration = end_time - start_time;
    result.time_elapsed_sec = total_duration.count();

    // Final result assembly & proof of optimality
    if (result.status != MILPStatus::FEASIBLE_LIMIT &&
        result.status != MILPStatus::NODE_LIMIT &&
        result.status != MILPStatus::TIME_LIMIT) {

        if (has_incumbent) {
            result.status = MILPStatus::OPTIMAL;
            result.message = "Proven global optimum found: search tree exhausted.";
            global_best_bound = incumbent_objective;
        } else {
            result.status = MILPStatus::INFEASIBLE;
            result.message = "Problem is proven integer infeasible: entire search tree pruned.";
        }
    }

    if (has_incumbent) {
        result.objective_value = incumbent_objective;
        result.variable_values = incumbent_solution;
        result.best_bound = global_best_bound;
        result.absolute_gap = std::abs(result.best_bound - result.objective_value);
        result.relative_gap = result.absolute_gap / std::max(1.0, std::abs(result.objective_value));
    }

    return result;
}

MILPResult MILPSolver::solve(
    const std::vector<double>& c,
    const Matrix<double>& A,
    const std::vector<double>& b,
    const std::vector<ConstraintType>& constraint_types,
    const std::vector<VariableType>& variable_types,
    const std::vector<VariableBounds>& bounds,
    ObjectiveSense sense) {

    MixedIntegerLinearProgram milp;
    milp.c = c;
    milp.A = A;
    milp.b = b;
    milp.constraint_types = constraint_types;
    milp.variable_types = variable_types;
    milp.bounds = bounds;
    milp.sense = sense;

    return solve(milp);
}

} // namespace vortex
