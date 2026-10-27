#pragma once

#include "vortex/matrix.hpp"
#include "simplex.hpp"
#include <vector>
#include <string>
#include <limits>
#include <cmath>
#include <memory>
#include <chrono>
#include <queue>
#include <optional>
#include <functional>

namespace vortex {

/**
 * @brief Variable type classification for Mixed-Integer Linear Programming.
 */
enum class VariableType {
    CONTINUOUS, ///< Real-valued variable in [lower, upper]
    INTEGER,    ///< Integer-valued variable in [lower, upper] cap Z
    BINARY      ///< Binary indicator variable in {0, 1}
};

/**
 * @brief Lower and upper bounds on a decision variable.
 */
struct VariableBounds {
    double lower{0.0};
    double upper{std::numeric_limits<double>::infinity()};

    VariableBounds() = default;
    VariableBounds(double lb, double ub) : lower(lb), upper(ub) {}

    [[nodiscard]] bool is_bounded_below() const noexcept {
        return lower > -1e19;
    }
    [[nodiscard]] bool is_bounded_above() const noexcept {
        return upper < 1e19;
    }
    [[nodiscard]] bool is_fixed(double tol = 1e-9) const noexcept {
        return std::abs(upper - lower) <= tol;
    }
    [[nodiscard]] bool is_infeasible(double tol = 1e-9) const noexcept {
        return lower > upper + tol;
    }
};

/**
 * @brief Node selection strategy for the Branch-and-Bound tree.
 */
enum class NodeSelectionStrategy {
    BEST_BOUND,     ///< Prioritize node with most promising dual bound (minimizes nodes explored)
    DEPTH_FIRST,    ///< LIFO order (rapidly explores deep branches to find incumbents)
    BREADTH_FIRST   ///< FIFO order (explores level by level)
};

/**
 * @brief Strategy for choosing which fractional integer variable to branch on.
 */
enum class BranchingStrategy {
    MOST_FRACTIONAL,  ///< Maximize min(x_j - floor(x_j), ceil(x_j) - x_j)
    FIRST_FRACTIONAL  ///< First variable index j that violates integrality
};

/**
 * @brief Configuration parameters for the VØRTEX MILP Branch-and-Bound solver.
 */
struct MILPOptions {
    double integrality_tolerance{1e-6};           ///< Tolerance for considering a value integer
    double absolute_gap_tolerance{1e-6};          ///< Target absolute optimality gap |BestBound - Incumbent|
    double relative_gap_tolerance{1e-4};          ///< Target relative optimality gap
    std::size_t node_limit{10000};                ///< Maximum branch-and-bound nodes explored
    double time_limit_sec{60.0};                  ///< Wall clock timeout in seconds
    NodeSelectionStrategy node_strategy{NodeSelectionStrategy::BEST_BOUND};
    BranchingStrategy branching_strategy{BranchingStrategy::MOST_FRACTIONAL};
    bool verbose{false};                          ///< Print progress of nodes to std::cout
    SimplexOptions lp_options{};                  ///< Configuration options passed to LP solver
};

/**
 * @brief Status resulting from solving a Mixed-Integer Linear Program.
 */
enum class MILPStatus {
    OPTIMAL,          ///< Proven global optimum (gap <= tolerance or search tree exhausted)
    FEASIBLE_LIMIT,   ///< Integer feasible solution found, but terminated by node or time limit
    INFEASIBLE,       ///< Proven integer infeasible
    UNBOUNDED,        ///< Proven unbounded
    NODE_LIMIT,       ///< Node limit reached without finding any integer solution
    TIME_LIMIT,       ///< Time limit reached without finding any integer solution
    NUMERICAL_ERROR   ///< Numerical failure or invalid formulation in underlying LP engine
};

/**
 * @brief Result structure returned by the MILP solver.
 */
struct MILPResult {
    MILPStatus status{MILPStatus::NUMERICAL_ERROR};
    double objective_value{0.0};
    std::vector<double> variable_values;          ///< Solution values for all decision variables
    double best_bound{0.0};                       ///< Best dual bound across active nodes
    double absolute_gap{0.0};                     ///< |best_bound - objective_value|
    double relative_gap{0.0};                     ///< absolute_gap / max(1.0, |objective_value|)
    std::size_t nodes_explored{0};                ///< Total branch-and-bound nodes solved
    std::size_t nodes_pruned_infeasible{0};       ///< Nodes pruned because LP relaxation was infeasible
    std::size_t nodes_pruned_bound{0};            ///< Nodes pruned because LP bound <= incumbent
    std::size_t nodes_pruned_integer{0};          ///< Nodes pruned because LP solution was integer-feasible
    double time_elapsed_sec{0.0};                 ///< Total solver execution time
    std::string message;                          ///< Informative status message
};

/**
 * @brief Complete Mixed-Integer Linear Program specification.
 */
struct MixedIntegerLinearProgram {
    std::vector<double> c;                                       ///< Objective coefficients
    Matrix<double> A;                                            ///< Constraint matrix (rows x cols)
    std::vector<double> b;                                       ///< Constraint right-hand side
    std::vector<ConstraintType> constraint_types;                ///< LEQ (<=), GEQ (>=), EQ (==)
    std::vector<VariableType> variable_types;                    ///< CONTINUOUS, INTEGER, or BINARY
    std::vector<VariableBounds> bounds;                          ///< Lower and upper bounds per variable
    ObjectiveSense sense{ObjectiveSense::MAXIMIZE};              ///< MAXIMIZE or MINIMIZE
};

/**
 * @brief Branch-and-Bound Mixed-Integer Linear Programming (MILP) Solver.
 *
 * Implements linear Branch-and-Bound using VØRTEX's core LP engine to solve continuous
 * relaxations at every tree node. Features:
 *  - Continuous, Integer, and Binary variables
 *  - Arbitrary lower/upper variable bounds (including negative bounds and unconstrained)
 *  - Most-Fractional and First-Fractional variable branching
 *  - Best-Bound, Depth-First, and Breadth-First node exploration
 *  - Strict incumbent tracking and dual bound bounding
 *  - Node-limit and time-limit handling
 *  - Sound unboundedness discrimination
 */
class MILPSolver {
public:
    explicit MILPSolver(MILPOptions options = MILPOptions{});

    /**
     * @brief Solves a Mixed-Integer Linear Program.
     * @param milp The complete MILP specification.
     * @return MILPResult containing status, solution, bound, gap, and tree metrics.
     */
    [[nodiscard]] MILPResult solve(const MixedIntegerLinearProgram& milp);

    /**
     * @brief Convenience overload for solving an MILP with explicit vectors.
     */
    [[nodiscard]] MILPResult solve(
        const std::vector<double>& c,
        const Matrix<double>& A,
        const std::vector<double>& b,
        const std::vector<ConstraintType>& constraint_types,
        const std::vector<VariableType>& variable_types,
        const std::vector<VariableBounds>& bounds,
        ObjectiveSense sense = ObjectiveSense::MAXIMIZE);

private:
    MILPOptions options_;

    /**
     * @brief Internal branch-and-bound node structure.
     */
    struct Node {
        std::size_t id{0};
        std::vector<VariableBounds> bounds;
        double lp_objective{0.0};
        std::vector<double> lp_solution;
        bool lp_solved{false};
        std::size_t depth{0};
    };

    /**
     * @brief Solves the continuous LP relaxation for a given node.
     */
    bool solve_node_relaxation(
        const MixedIntegerLinearProgram& milp,
        Node& node,
        SimplexResult& lp_out) const;

    /**
     * @brief Checks if a solution satisfies integer constraints within tolerance.
     */
    [[nodiscard]] bool is_integer_feasible(
        const std::vector<double>& x,
        const std::vector<VariableType>& var_types,
        std::size_t& out_branch_var) const;

    /**
     * @brief Selects the integer variable to branch on.
     */
    [[nodiscard]] int select_branching_variable(
        const std::vector<double>& x,
        const std::vector<VariableType>& var_types) const;

    /**
     * @brief Validates input MILP dimensions and parameters.
     */
    [[nodiscard]] std::optional<std::string> validate_input(
        const MixedIntegerLinearProgram& milp) const;
};

} // namespace vortex
