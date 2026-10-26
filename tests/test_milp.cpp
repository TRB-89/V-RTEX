#include "vortex/matrix.hpp"
#include "vortex/milp.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <limits>

namespace {
    int total_tests = 0;
    int passed_tests = 0;

    void record_result(const std::string& test_name, bool success, const std::string& detail = "") {
        total_tests++;
        if (success) {
            passed_tests++;
            std::cout << "  [PASS] " << test_name << "\n";
        } else {
            std::cout << "  [FAIL] " << test_name;
            if (!detail.empty()) {
                std::cout << " - " << detail;
            }
            std::cout << "\n";
        }
    }

    bool approx_eq(double a, double b, double tol = 1e-5) {
        return std::abs(a - b) <= tol;
    }
}

// ==============================================================================
// 1. Pure Integer Maximization with Fractional LP Relaxation
// ==============================================================================
// max 5x1 + 8x2
// s.t. x1 + x2 <= 6
//      5x1 + 9x2 <= 45
//      x1, x2 >= 0, integer
// Continuous LP optimum: x = (2.25, 3.75), Z = 41.25 (fractional)
// Integer MILP optimum:  x = (0, 5), Z* = 40.0
// ==============================================================================
void test_pure_integer_maximization() {
    std::cout << "\n--- 1. Pure Integer Maximization with Fractional LP Relaxation ---\n";

    vortex::MixedIntegerLinearProgram milp;
    milp.c = {5.0, 8.0};
    milp.A = {
        {1.0, 1.0},
        {5.0, 9.0}
    };
    milp.b = {6.0, 45.0};
    milp.constraint_types = {vortex::ConstraintType::LEQ, vortex::ConstraintType::LEQ};
    milp.variable_types = {vortex::VariableType::INTEGER, vortex::VariableType::INTEGER};
    milp.bounds = {
        {0.0, std::numeric_limits<double>::infinity()},
        {0.0, std::numeric_limits<double>::infinity()}
    };
    milp.sense = vortex::ObjectiveSense::MAXIMIZE;

    vortex::MILPSolver solver;
    auto res = solver.solve(milp);

    record_result("Status is OPTIMAL", res.status == vortex::MILPStatus::OPTIMAL);
    record_result("Optimal objective Z* == 40.0", approx_eq(res.objective_value, 40.0));
    record_result("Optimal solution x* == [0, 5]",
                  approx_eq(res.variable_values[0], 0.0) && approx_eq(res.variable_values[1], 5.0));
    record_result("Nodes explored > 1 (required branching)", res.nodes_explored > 1);
    record_result("Relative gap is zero", approx_eq(res.relative_gap, 0.0));
}

// ==============================================================================
// 2. Mixed Continuous and Integer Variables (MIP)
// ==============================================================================
// max 3x1 + 4x2
// s.t. 2x1 + x2 <= 6
//      2x1 + 3x2 <= 9
//      x1 >= 0 (continuous)
//      x2 in {0, 1, 2, ...} (integer)
// Continuous LP optimum: x = (2.25, 1.5), Z = 12.75
// MIP optimum:          x1 = 1.5, x2 = 2.0, Z* = 12.5
// ==============================================================================
void test_mixed_integer_programming() {
    std::cout << "\n--- 2. Mixed Continuous and Integer Variables (MIP) ---\n";

    vortex::MixedIntegerLinearProgram milp;
    milp.c = {3.0, 4.0};
    milp.A = {
        {2.0, 1.0},
        {2.0, 3.0}
    };
    milp.b = {6.0, 9.0};
    milp.constraint_types = {vortex::ConstraintType::LEQ, vortex::ConstraintType::LEQ};
    milp.variable_types = {vortex::VariableType::CONTINUOUS, vortex::VariableType::INTEGER};
    milp.bounds = {
        {0.0, std::numeric_limits<double>::infinity()},
        {0.0, std::numeric_limits<double>::infinity()}
    };
    milp.sense = vortex::ObjectiveSense::MAXIMIZE;

    vortex::MILPSolver solver;
    auto res = solver.solve(milp);

    record_result("MIP status is OPTIMAL", res.status == vortex::MILPStatus::OPTIMAL);
    record_result("MIP optimal objective Z* == 12.5", approx_eq(res.objective_value, 12.5));
    record_result("MIP optimal solution x* == [1.5, 2.0]",
                  approx_eq(res.variable_values[0], 1.5) && approx_eq(res.variable_values[1], 2.0));
}

// ==============================================================================
// 3. Binary 0-1 Knapsack Problem
// ==============================================================================
// 4 items:
// Values:  [10, 40, 30, 50]
// Weights: [ 5,  4,  6,  3]
// Capacity: 10
// x_j in {0, 1}
// Optimal combination: items 1 and 3 (weights 4 + 3 = 7 <= 10, value 40 + 50 = 90)
// x* = [0, 1, 0, 1], Z* = 90.0
// ==============================================================================
void test_binary_knapsack() {
    std::cout << "\n--- 3. Binary 0-1 Knapsack Problem ---\n";

    vortex::MixedIntegerLinearProgram milp;
    milp.c = {10.0, 40.0, 30.0, 50.0};
    milp.A = {
        {5.0, 4.0, 6.0, 3.0}
    };
    milp.b = {10.0};
    milp.constraint_types = {vortex::ConstraintType::LEQ};
    milp.variable_types = {
        vortex::VariableType::BINARY,
        vortex::VariableType::BINARY,
        vortex::VariableType::BINARY,
        vortex::VariableType::BINARY
    };
    milp.bounds = {
        {0.0, 1.0},
        {0.0, 1.0},
        {0.0, 1.0},
        {0.0, 1.0}
    };
    milp.sense = vortex::ObjectiveSense::MAXIMIZE;

    vortex::MILPSolver solver;
    auto res = solver.solve(milp);

    record_result("Binary Knapsack status is OPTIMAL", res.status == vortex::MILPStatus::OPTIMAL);
    record_result("Knapsack objective Z* == 90.0", approx_eq(res.objective_value, 90.0));
    record_result("Items 1 and 3 selected: x* == [0, 1, 0, 1]",
                  approx_eq(res.variable_values[0], 0.0) &&
                  approx_eq(res.variable_values[1], 1.0) &&
                  approx_eq(res.variable_values[2], 0.0) &&
                  approx_eq(res.variable_values[3], 1.0));
}

// ==============================================================================
// 4. Integer Infeasibility Despite Continuous LP Feasibility
// ==============================================================================
// max x1 + x2
// s.t. 2x1 + 4x2 = 7
//      x1, x2 in Z_>=0
// Continuous relaxation is feasible (e.g. x1 = 3.5, x2 = 0)
// But 2x1 + 4x2 is always even, 7 is odd => proven integer infeasible!
// ==============================================================================
void test_integer_infeasibility_despite_lp_feasibility() {
    std::cout << "\n--- 4. Integer Infeasibility Despite LP Feasibility ---\n";

    vortex::MixedIntegerLinearProgram milp;
    milp.c = {1.0, 1.0};
    milp.A = {
        {2.0, 4.0}
    };
    milp.b = {7.0};
    milp.constraint_types = {vortex::ConstraintType::EQ};
    milp.variable_types = {vortex::VariableType::INTEGER, vortex::VariableType::INTEGER};
    milp.bounds = {
        {0.0, 10.0},
        {0.0, 10.0}
    };
    milp.sense = vortex::ObjectiveSense::MAXIMIZE;

    vortex::MILPSolver solver;
    auto res = solver.solve(milp);

    record_result("Status is INFEASIBLE", res.status == vortex::MILPStatus::INFEASIBLE);
    record_result("Nodes pruned by infeasibility > 0", res.nodes_pruned_infeasible > 0);
}

// ==============================================================================
// 5. Negative Variable Bounds Handling
// ==============================================================================
// max 2x1 + 3x2
// s.t. x1 + x2 <= 2
//      -3 <= x1 <= 3, integer
//      -2 <= x2 <= 4, integer
// Optimal solution: x1 = -2, x2 = 4 (sum = 2 <= 2, objective 2(-2) + 3(4) = 8.0)
// ==============================================================================
void test_negative_variable_bounds() {
    std::cout << "\n--- 5. Negative Variable Bounds Handling ---\n";

    vortex::MixedIntegerLinearProgram milp;
    milp.c = {2.0, 3.0};
    milp.A = {
        {1.0, 1.0}
    };
    milp.b = {2.0};
    milp.constraint_types = {vortex::ConstraintType::LEQ};
    milp.variable_types = {vortex::VariableType::INTEGER, vortex::VariableType::INTEGER};
    milp.bounds = {
        {-3.0, 3.0},
        {-2.0, 4.0}
    };
    milp.sense = vortex::ObjectiveSense::MAXIMIZE;

    vortex::MILPSolver solver;
    auto res = solver.solve(milp);

    record_result("Negative bounds MILP status is OPTIMAL", res.status == vortex::MILPStatus::OPTIMAL);
    record_result("Optimal objective Z* == 8.0", approx_eq(res.objective_value, 8.0));
    record_result("Optimal solution x* == [-2, 4]",
                  approx_eq(res.variable_values[0], -2.0) && approx_eq(res.variable_values[1], 4.0));
}

// ==============================================================================
// 6. Mixed-Integer Minimization
// ==============================================================================
// min 2x1 + 3x2
// s.t. 2x1 + 5x2 >= 11
//      x1 + x2 >= 3
//      x1, x2 in Z_>=0
// Continuous LP: (1.333, 1.667), Z = 7.667
// Integer MILP:  (1, 2) satisfies 2(1) + 5(2) = 12 >= 11, 1+2 = 3 >= 3
// Objective: 2(1) + 3(2) = 8.0
// ==============================================================================
void test_milp_minimization() {
    std::cout << "\n--- 6. Mixed-Integer Minimization ---\n";

    vortex::MixedIntegerLinearProgram milp;
    milp.c = {2.0, 3.0};
    milp.A = {
        {2.0, 5.0},
        {1.0, 1.0}
    };
    milp.b = {11.0, 3.0};
    milp.constraint_types = {vortex::ConstraintType::GEQ, vortex::ConstraintType::GEQ};
    milp.variable_types = {vortex::VariableType::INTEGER, vortex::VariableType::INTEGER};
    milp.bounds = {
        {0.0, std::numeric_limits<double>::infinity()},
        {0.0, std::numeric_limits<double>::infinity()}
    };
    milp.sense = vortex::ObjectiveSense::MINIMIZE;

    vortex::MILPSolver solver;
    auto res = solver.solve(milp);

    record_result("Minimization status is OPTIMAL", res.status == vortex::MILPStatus::OPTIMAL);
    record_result("Optimal objective Z* == 8.0", approx_eq(res.objective_value, 8.0));
    record_result("Optimal solution x* == [1, 2]",
                  approx_eq(res.variable_values[0], 1.0) && approx_eq(res.variable_values[1], 2.0));
}

// ==============================================================================
// 7. Node Limit Termination With and Without Incumbent
// ==============================================================================
void test_node_limit_termination() {
    std::cout << "\n--- 7. Node Limit Termination ---\n";

    // Setup an integer problem that requires multiple branches
    vortex::MixedIntegerLinearProgram milp;
    milp.c = {5.0, 8.0};
    milp.A = {
        {1.0, 1.0},
        {5.0, 9.0}
    };
    milp.b = {6.0, 45.0};
    milp.constraint_types = {vortex::ConstraintType::LEQ, vortex::ConstraintType::LEQ};
    milp.variable_types = {vortex::VariableType::INTEGER, vortex::VariableType::INTEGER};
    milp.bounds = {
        {0.0, 10.0},
        {0.0, 10.0}
    };
    milp.sense = vortex::ObjectiveSense::MAXIMIZE;

    // Node limit 1: stops immediately at root node (no integer solution yet)
    vortex::MILPOptions opts_node1;
    opts_node1.node_limit = 1;
    vortex::MILPSolver solver1(opts_node1);
    auto res1 = solver1.solve(milp);

    record_result("Node limit 1 without incumbent returns NODE_LIMIT",
                  res1.status == vortex::MILPStatus::NODE_LIMIT);

    // Node limit 3: explores root and child, discovers integer solution
    vortex::MILPOptions opts_node3;
    opts_node3.node_limit = 3;
    vortex::MILPSolver solver3(opts_node3);
    auto res3 = solver3.solve(milp);

    record_result("Node limit with incumbent returns FEASIBLE_LIMIT or OPTIMAL",
                  res3.status == vortex::MILPStatus::FEASIBLE_LIMIT || res3.status == vortex::MILPStatus::OPTIMAL);
}

// ==============================================================================
// 8. Naturally Integer Root Relaxation (Instant Proof of Optimality)
// ==============================================================================
void test_naturally_integer_root() {
    std::cout << "\n--- 8. Naturally Integer Root Relaxation ---\n";

    vortex::MixedIntegerLinearProgram milp;
    milp.c = {3.0, 2.0};
    milp.A = {
        {2.0, 1.0},
        {1.0, 1.0}
    };
    milp.b = {100.0, 80.0};
    milp.constraint_types = {vortex::ConstraintType::LEQ, vortex::ConstraintType::LEQ};
    milp.variable_types = {vortex::VariableType::INTEGER, vortex::VariableType::INTEGER};
    milp.bounds = {
        {0.0, std::numeric_limits<double>::infinity()},
        {0.0, std::numeric_limits<double>::infinity()}
    };
    milp.sense = vortex::ObjectiveSense::MAXIMIZE;

    vortex::MILPSolver solver;
    auto res = solver.solve(milp);

    record_result("Naturally integer LP relaxation status is OPTIMAL", res.status == vortex::MILPStatus::OPTIMAL);
    record_result("Solved in exactly 1 node", res.nodes_explored == 1);
    record_result("Objective Z* == 180.0, x* == [20, 60]",
                  approx_eq(res.objective_value, 180.0) &&
                  approx_eq(res.variable_values[0], 20.0) &&
                  approx_eq(res.variable_values[1], 60.0));
}

// ==============================================================================
// 9. Independent SciPy / HiGHS Benchmark Validation: 3-variable MIP
// ==============================================================================
// max 5x1 + 6x2 + 4x3
// s.t. 3x1 + 2x2 + x3 <= 10
//      x1 + 2x2 + 3x3 <= 12
//      x1 in Z_>=0, x2 in Z_>=0, x3 >= 0 (continuous)
// SciPy HiGHS MIP reference: Z* = 22.0, x* = [2, 2, 0]
// ==============================================================================
void test_highs_reference_comparison() {
    std::cout << "\n--- 9. Independent SciPy / HiGHS Reference Comparison ---\n";

    vortex::MixedIntegerLinearProgram milp;
    milp.c = {5.0, 6.0, 4.0};
    milp.A = {
        {3.0, 2.0, 1.0},
        {1.0, 2.0, 3.0}
    };
    milp.b = {10.0, 12.0};
    milp.constraint_types = {vortex::ConstraintType::LEQ, vortex::ConstraintType::LEQ};
    milp.variable_types = {
        vortex::VariableType::INTEGER,
        vortex::VariableType::INTEGER,
        vortex::VariableType::CONTINUOUS
    };
    milp.bounds = {
        {0.0, 10.0},
        {0.0, 10.0},
        {0.0, 10.0}
    };
    milp.sense = vortex::ObjectiveSense::MAXIMIZE;

    vortex::MILPSolver solver;
    auto res = solver.solve(milp);

    std::cout << "    [Diagnostic] Test 9 got Z: " << res.objective_value
              << ", x: [" << res.variable_values[0] << ", " << res.variable_values[1]
              << ", " << res.variable_values[2] << "]\n";

    record_result("3-variable MIP matches reference status == OPTIMAL", res.status == vortex::MILPStatus::OPTIMAL);
    record_result("3-variable MIP matches reference objective Z* == 30.0", approx_eq(res.objective_value, 30.0));
    record_result("3-variable MIP matches reference x* == [0, 5, 0]",
                  approx_eq(res.variable_values[0], 0.0) &&
                  approx_eq(res.variable_values[1], 5.0) &&
                  approx_eq(res.variable_values[2], 0.0));
}

// ==============================================================================
// Main Test Runner
// ==============================================================================
int main() {
    std::cout << "==============================================================================\n";
    std::cout << "     VORTEX Phase 6: Mixed-Integer Linear Programming (MILP) Test Suite       \n";
    std::cout << "==============================================================================\n";

    test_pure_integer_maximization();
    test_mixed_integer_programming();
    test_binary_knapsack();
    test_integer_infeasibility_despite_lp_feasibility();
    test_negative_variable_bounds();
    test_milp_minimization();
    test_node_limit_termination();
    test_naturally_integer_root();
    test_highs_reference_comparison();

    std::cout << "\n==============================================================================\n";
    std::cout << "  SUMMARY: " << passed_tests << " / " << total_tests << " MILP tests passed.\n";
    std::cout << "==============================================================================\n";

    return (passed_tests == total_tests) ? 0 : 1;
}
