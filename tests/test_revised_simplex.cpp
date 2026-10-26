#include "vortex/matrix.hpp"
#include "vortex/revised_simplex.hpp"
#include "simplex.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <cmath>

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
// 1. SIH26119 Standard Benchmark Problem
//    Max Z = 3x + 5y
//    s.t.  x        <= 4
//              2y   <= 12
//         3x + 2y   <= 18
//         x, y >= 0
//    Expected: x = 2.0, y = 6.0, Z = 36.0
// ==============================================================================
void test_standard_sih_benchmark() {
    std::cout << "\n--- Running Suite: SIH26119 Standard Benchmark Problem ---\n";

    std::vector<double> c = {3.0, 5.0};
    vortex::Matrix<double> A = {
        {1.0, 0.0},
        {0.0, 2.0},
        {3.0, 2.0}
    };
    std::vector<double> b = {4.0, 12.0, 18.0};

    vortex::RevisedSimplexSolver solver;
    auto res = solver.solve(c, A, b);

    record_result("Status is OPTIMAL", res.status == vortex::SolverStatus::OPTIMAL);
    record_result("Objective Z == 36.0", approx_eq(res.objective_value, 36.0),
                  "Got: " + std::to_string(res.objective_value));
    record_result("Variables x == 2.0, y == 6.0",
                  approx_eq(res.variable_values[0], 2.0) && approx_eq(res.variable_values[1], 6.0),
                  "Got x=" + std::to_string(res.variable_values[0]) + ", y=" + std::to_string(res.variable_values[1]));
    record_result("Slack variable values: s1 == 2.0, s2 == 0.0, s3 == 0.0",
                  approx_eq(res.slack_values[0], 2.0) &&
                  approx_eq(res.slack_values[1], 0.0) &&
                  approx_eq(res.slack_values[2], 0.0));
    record_result("Total iterations <= 3", res.iterations <= 3,
                  "Iterations: " + std::to_string(res.iterations));
    record_result("Residual ||B*x_B - b|| < 1e-10", res.max_primal_residual < 1e-10);
}

// ==============================================================================
// 2. Basis Matrix Operations and Reduced Cost Unit Validation
// ==============================================================================
void test_basis_operations_and_reduced_costs() {
    std::cout << "\n--- Running Suite: Basis Matrix Operations & Reduced Costs ---\n";

    // Standard problem [A | I]:
    // A_std has 3 rows, 5 columns (x1, x2, s1, s2, s3)
    vortex::Matrix<double> A_std = {
        {1.0, 0.0, 1.0, 0.0, 0.0},
        {0.0, 2.0, 0.0, 1.0, 0.0},
        {3.0, 2.0, 0.0, 0.0, 1.0}
    };
    std::vector<double> c_std = {3.0, 5.0, 0.0, 0.0, 0.0};
    std::vector<double> b = {4.0, 12.0, 18.0};

    vortex::RevisedSimplexSolver solver;

    // Test initial basis: s1 (col 2), s2 (col 3), s3 (col 4)
    std::vector<std::size_t> basis = {2, 3, 4};
    auto B = solver.extract_basis_matrix(A_std, basis);

    record_result("Initial basis matrix is 3x3 Identity",
                  B.is_approx(vortex::Matrix<double>::identity(3)));

    auto c_B = solver.extract_basis_costs(c_std, basis);
    record_result("Initial basis costs are all zero",
                  c_B[0] == 0.0 && c_B[1] == 0.0 && c_B[2] == 0.0);

    vortex::LUDecomposition<double> lu(B);
    auto pi = solver.compute_dual_multipliers(lu, c_B);
    record_result("Initial dual multipliers pi are all zero",
                  approx_eq(pi[0], 0.0) && approx_eq(pi[1], 0.0) && approx_eq(pi[2], 0.0));

    auto reduced_costs = solver.compute_reduced_costs(c_std, A_std, pi);
    record_result("Initial reduced costs c_bar match c",
                  approx_eq(reduced_costs[0], 3.0) && approx_eq(reduced_costs[1], 5.0));

    // Test entering variable selection with Dantzig rule (col 1 has max cost 5.0)
    std::vector<bool> allowed = {true, true, true, true, true};
    int enter = solver.select_entering_variable(reduced_costs, basis, allowed);
    record_result("Dantzig selects entering column 1 (y with cost 5.0)", enter == 1,
                  "Got: " + std::to_string(enter));

    // Test step direction for entering col 1: B * d = A_std[:, 1] = [0, 2, 2]^T
    auto d = solver.compute_step_direction(lu, A_std, 1);
    record_result("Step direction d == [0, 2, 2]^T",
                  approx_eq(d[0], 0.0) && approx_eq(d[1], 2.0) && approx_eq(d[2], 2.0));

    // Test ratio test with x_B = b = [4, 12, 18]^T
    auto x_B = solver.compute_primal_solution(lu, b);
    int leave = solver.select_leaving_variable(x_B, d, basis);
    // Row 1 ratio: 12 / 2 = 6. Row 2 ratio: 18 / 2 = 9.
    record_result("Ratio test selects row 1 (ratio 6 < 9)", leave == 1,
                  "Got: " + std::to_string(leave));
}

// ==============================================================================
// 3. Mixed Constraints via Two-Phase Revised Simplex (<=, >=, ==)
// ==============================================================================
void test_two_phase_revised_simplex() {
    std::cout << "\n--- Running Suite: Mixed Constraints (Two-Phase Revised Simplex) ---\n";

    std::vector<double> c = {3.0, 2.0};
    vortex::Matrix<double> A = {
        {2.0, 1.0},
        {2.0, 3.0},
        {1.0, 1.0}
    };
    std::vector<double> b = {18.0, 12.0, 8.0};
    std::vector<vortex::ConstraintType> types = {
        vortex::ConstraintType::LEQ,
        vortex::ConstraintType::GEQ,
        vortex::ConstraintType::EQ
    };

    vortex::RevisedSimplexSolver solver;
    auto res = solver.solve(c, A, b, types, vortex::ObjectiveSense::MAXIMIZE);

    record_result("Status is OPTIMAL", res.status == vortex::SolverStatus::OPTIMAL);
    record_result("Optimal Z == 24.0", approx_eq(res.objective_value, 24.0),
                  "Got: " + std::to_string(res.objective_value));
    record_result("Variables x1 == 8.0, x2 == 0.0",
                  approx_eq(res.variable_values[0], 8.0) && approx_eq(res.variable_values[1], 0.0),
                  "Got x1=" + std::to_string(res.variable_values[0]) + ", x2=" + std::to_string(res.variable_values[1]));
    record_result("Phase I iterations > 0", res.phase1_iterations > 0);
    record_result("Total iterations == Phase I + Phase II",
                  res.iterations == (res.phase1_iterations + res.phase2_iterations));
}

// ==============================================================================
// 4. Minimization Problem
// ==============================================================================
void test_minimization_revised() {
    std::cout << "\n--- Running Suite: Minimization with >= Constraints ---\n";

    std::vector<double> c = {4.0, 2.0};
    vortex::Matrix<double> A = {
        {3.0, 1.0},
        {1.0, 1.0},
        {1.0, 2.0}
    };
    std::vector<double> b = {27.0, 21.0, 30.0};
    std::vector<vortex::ConstraintType> types = {
        vortex::ConstraintType::GEQ,
        vortex::ConstraintType::GEQ,
        vortex::ConstraintType::GEQ
    };

    vortex::RevisedSimplexSolver solver;
    auto res = solver.solve(c, A, b, types, vortex::ObjectiveSense::MINIMIZE);

    record_result("Status is OPTIMAL", res.status == vortex::SolverStatus::OPTIMAL);
    record_result("Minimum Z == 48.0", approx_eq(res.objective_value, 48.0),
                  "Got: " + std::to_string(res.objective_value));
    record_result("Variables x1 == 3.0, x2 == 18.0",
                  approx_eq(res.variable_values[0], 3.0) && approx_eq(res.variable_values[1], 18.0),
                  "Got x1=" + std::to_string(res.variable_values[0]) + ", x2=" + std::to_string(res.variable_values[1]));
}

// ==============================================================================
// 5. Negative RHS Normalization
// ==============================================================================
void test_negative_rhs_revised() {
    std::cout << "\n--- Running Suite: Negative RHS Normalization ---\n";

    std::vector<double> c = {2.0, 3.0};
    vortex::Matrix<double> A = {
        {-1.0, -1.0},
        { 1.0,  2.0}
    };
    std::vector<double> b = {-4.0, 10.0};
    std::vector<vortex::ConstraintType> types = {
        vortex::ConstraintType::LEQ,
        vortex::ConstraintType::LEQ
    };

    vortex::RevisedSimplexSolver solver;
    auto res = solver.solve(c, A, b, types, vortex::ObjectiveSense::MAXIMIZE);

    record_result("Status is OPTIMAL with negative RHS",
                  res.status == vortex::SolverStatus::OPTIMAL);
    record_result("Objective Z == 20.0", approx_eq(res.objective_value, 20.0),
                  "Got: " + std::to_string(res.objective_value));
    record_result("Variables x1 == 10.0, x2 == 0.0",
                  approx_eq(res.variable_values[0], 10.0) && approx_eq(res.variable_values[1], 0.0));
}

// ==============================================================================
// 6. Infeasibility and Unboundedness Detection
// ==============================================================================
void test_infeasible_and_unbounded() {
    std::cout << "\n--- Running Suite: Infeasibility & Unboundedness Detection ---\n";

    vortex::RevisedSimplexSolver solver;

    // Infeasible Problem
    std::vector<double> c1 = {5.0, 3.0};
    vortex::Matrix<double> A1 = {
        {1.0, 1.0},
        {1.0, 1.0}
    };
    std::vector<double> b1 = {2.0, 5.0};
    std::vector<vortex::ConstraintType> types1 = {
        vortex::ConstraintType::LEQ,
        vortex::ConstraintType::GEQ
    };

    auto res_inf = solver.solve(c1, A1, b1, types1, vortex::ObjectiveSense::MAXIMIZE);
    record_result("Correctly detects INFEASIBLE problem",
                  res_inf.status == vortex::SolverStatus::INFEASIBLE);

    // Unbounded Problem
    std::vector<double> c2 = {3.0, -1.0};
    vortex::Matrix<double> A2 = {
        {-1.0, 1.0},
        { 0.0, 1.0}
    };
    std::vector<double> b2 = {5.0, 2.0};
    std::vector<vortex::ConstraintType> types2 = {
        vortex::ConstraintType::LEQ,
        vortex::ConstraintType::GEQ
    };

    auto res_unb = solver.solve(c2, A2, b2, types2, vortex::ObjectiveSense::MAXIMIZE);
    record_result("Correctly detects UNBOUNDED problem",
                  res_unb.status == vortex::SolverStatus::UNBOUNDED);
}

// ==============================================================================
// 7. Bland's Anti-Cycling Rule on Degenerate LP
// ==============================================================================
void test_bland_anti_cycling() {
    std::cout << "\n--- Running Suite: Bland's Anti-Cycling Rule on Degenerate LP ---\n";

    // Degenerate problem with multiple zero RHS constraints
    std::vector<double> c = {10.0, -57.0, -9.0, -24.0};
    vortex::Matrix<double> A = {
        { 0.5, -5.5, -2.5,  9.0},
        { 0.5, -1.5, -0.5,  1.0},
        { 1.0,  0.0,  0.0,  0.0}
    };
    std::vector<double> b = {0.0, 0.0, 1.0};

    vortex::RevisedSimplexOptions opts;
    opts.pivot_rule = vortex::PivotRule::BLAND;
    opts.max_iterations = 200;

    vortex::RevisedSimplexSolver solver(opts);
    auto res = solver.solve(c, A, b);

    record_result("Bland's rule converges on degenerate problem without cycling",
                  res.status == vortex::SolverStatus::OPTIMAL);
    record_result("Objective Z >= 0", res.objective_value >= -1e-6);
}

// ==============================================================================
// 8. Cross-Validation: Revised Simplex vs. Tableau Simplex
// ==============================================================================
void test_cross_validation_with_tableau() {
    std::cout << "\n--- Running Suite: Cross-Validation: Revised Simplex vs. Tableau Simplex ---\n";

    // Problem 1: 3-variable, 3-constraint
    std::vector<double> c = {2.0, 4.0, 3.0};
    vortex::Matrix<double> A = {
        {3.0, 2.0, 1.0},
        {1.0, 1.0, 2.0},
        {2.0, 2.0, 5.0}
    };
    std::vector<double> b = {10.0, 8.0, 14.0};

    vortex::SimplexSolver tableau_solver;
    auto res_tab = tableau_solver.solve(c, A, b);

    vortex::RevisedSimplexSolver revised_solver;
    auto res_rev = revised_solver.solve(c, A, b);

    record_result("Both solvers return OPTIMAL status",
                  res_tab.status == vortex::SolverStatus::OPTIMAL &&
                  res_rev.status == vortex::SolverStatus::OPTIMAL);
    record_result("Objective values match exactly",
                  approx_eq(res_tab.objective_value, res_rev.objective_value),
                  "Tableau: " + std::to_string(res_tab.objective_value) +
                  ", Revised: " + std::to_string(res_rev.objective_value));

    bool vars_match = true;
    for (std::size_t i = 0; i < c.size(); ++i) {
        if (!approx_eq(res_tab.variable_values[i], res_rev.variable_values[i], 1e-4)) {
            vars_match = false;
        }
    }
    record_result("Optimal variables x match across both solvers", vars_match);
}

int main() {
    std::cout << "=========================================================\n";
    std::cout << "      VORTEX Revised Simplex LP Test Suite (Phase 4)     \n";
    std::cout << "=========================================================\n";

    test_standard_sih_benchmark();
    test_basis_operations_and_reduced_costs();
    test_two_phase_revised_simplex();
    test_minimization_revised();
    test_negative_rhs_revised();
    test_infeasible_and_unbounded();
    test_bland_anti_cycling();
    test_cross_validation_with_tableau();

    std::cout << "\n=========================================================\n";
    std::cout << "  SUMMARY: " << passed_tests << " / " << total_tests << " tests passed.\n";
    std::cout << "=========================================================\n";

    return (passed_tests == total_tests) ? 0 : 1;
}
