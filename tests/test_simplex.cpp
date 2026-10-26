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

    bool approx_eq(double a, double b, double tol = 1e-6) {
        return std::abs(a - b) <= tol;
    }
}

// ==============================================================================
// 1. Mandatory Problem: Max Z = 3x + 5y
//    s.t. x <= 4, 2y <= 12, 3x + 2y <= 18, x, y >= 0
//    Expected: x = 2, y = 6, Z = 36
// ==============================================================================
void test_mandatory_problem() {
    std::cout << "\n--- Running Suite: Mandatory Problem (SIH26119 Spec) ---\n";

    std::vector<double> c = {3.0, 5.0};
    vortex::Matrix<double> A = {
        {1.0, 0.0},
        {0.0, 2.0},
        {3.0, 2.0}
    };
    std::vector<double> b = {4.0, 12.0, 18.0};

    vortex::SimplexSolver solver;
    vortex::SimplexResult result = solver.solve(c, A, b);

    bool status_ok = (result.status == vortex::SolverStatus::OPTIMAL);
    bool obj_ok = approx_eq(result.objective_value, 36.0);
    bool x_ok = (result.variable_values.size() >= 2) &&
                approx_eq(result.variable_values[0], 2.0) &&
                approx_eq(result.variable_values[1], 6.0);

    record_result("Status is OPTIMAL", status_ok);
    record_result("Objective value Z == 36.0", obj_ok, "Got: " + std::to_string(result.objective_value));
    record_result("Variable values x == 2.0, y == 6.0", x_ok,
                  "Got x=" + std::to_string(result.variable_values[0]) +
                  ", y=" + std::to_string(result.variable_values[1]));
    record_result("Iteration count is positive", result.iterations > 0);
}

// ==============================================================================
// 2. Additional Bounded Optimal Problem
//    Max Z = 5x1 + 4x2
//    s.t. 6x1 + 4x2 <= 24
//         x1 + 2x2  <= 6
//        -x1 + x2   <= 1
//              x2   <= 2
//         x1, x2    >= 0
//    Optimal: x1 = 3, x2 = 1.5, Z = 21.0
// ==============================================================================
void test_additional_bounded_problem() {
    std::cout << "\n--- Running Suite: Additional Bounded Optimal Problem ---\n";

    std::vector<double> c = {5.0, 4.0};
    vortex::Matrix<double> A = {
        { 6.0, 4.0},
        { 1.0, 2.0},
        {-1.0, 1.0},
        { 0.0, 1.0}
    };
    std::vector<double> b = {24.0, 6.0, 1.0, 2.0};

    vortex::SimplexSolver solver;
    vortex::SimplexResult res = solver.solve(c, A, b);

    record_result("Status is OPTIMAL", res.status == vortex::SolverStatus::OPTIMAL);
    record_result("Objective value Z == 21.0", approx_eq(res.objective_value, 21.0),
                  "Got: " + std::to_string(res.objective_value));
    record_result("Variable values x1 == 3.0, x2 == 1.5",
                  approx_eq(res.variable_values[0], 3.0) && approx_eq(res.variable_values[1], 1.5),
                  "Got x1=" + std::to_string(res.variable_values[0]) + ", x2=" + std::to_string(res.variable_values[1]));
}

// ==============================================================================
// 3. Unbounded Problem Detection
//    Max Z = 2x1 + x2
//    s.t. x1 - x2 <= 10
//         2x1 - x2 <= 40
//         x1, x2 >= 0
//    x2 has negative coefficients in all rows, so x2 -> inf without constraint bounds.
// ==============================================================================
void test_unbounded_problem() {
    std::cout << "\n--- Running Suite: Unbounded Problem Detection ---\n";

    std::vector<double> c = {2.0, 1.0};
    vortex::Matrix<double> A = {
        {1.0, -1.0},
        {2.0, -1.0}
    };
    std::vector<double> b = {10.0, 40.0};

    vortex::SimplexSolver solver;
    vortex::SimplexResult res = solver.solve(c, A, b);

    record_result("Correctly detects UNBOUNDED status", res.status == vortex::SolverStatus::UNBOUNDED);
    record_result("Unbounded message populated", !res.message.empty());
}

// ==============================================================================
// 4. Degenerate Feasible Problem (Tests Anti-Cycling & Pivot Tie-breaking)
//    Max Z = 3x1 + 9x2
//    s.t. x1 + 4x2 <= 8
//         x1 + 2x2 <= 4
//         x1, x2 >= 0
//    At vertex (0, 2), both constraints intersect simultaneously.
//    Optimal: x1 = 0, x2 = 2, Z = 18.0
// ==============================================================================
void test_degenerate_problem() {
    std::cout << "\n--- Running Suite: Degenerate Feasible Problem ---\n";

    std::vector<double> c = {3.0, 9.0};
    vortex::Matrix<double> A = {
        {1.0, 4.0},
        {1.0, 2.0}
    };
    std::vector<double> b = {8.0, 4.0};

    // Test with Dantzig (default)
    {
        vortex::SimplexSolver solver;
        vortex::SimplexResult res = solver.solve(c, A, b);
        record_result("Degenerate LP solved with Dantzig + Bland tie-break",
                      res.status == vortex::SolverStatus::OPTIMAL &&
                      approx_eq(res.objective_value, 18.0) &&
                      approx_eq(res.variable_values[0], 0.0) &&
                      approx_eq(res.variable_values[1], 2.0));
    }

    // Test with Pure Bland's Rule
    {
        vortex::SimplexOptions opt;
        opt.pivot_rule = vortex::PivotRule::BLAND;
        vortex::SimplexSolver solver_bland(opt);
        vortex::SimplexResult res_bland = solver_bland.solve(c, A, b);
        record_result("Degenerate LP solved with pure Bland's Rule",
                      res_bland.status == vortex::SolverStatus::OPTIMAL &&
                      approx_eq(res_bland.objective_value, 18.0));
    }
}

// ==============================================================================
// 5. Invalid Matrix Dimensions
// ==============================================================================
void test_invalid_dimensions() {
    std::cout << "\n--- Running Suite: Invalid Dimensions Rejection ---\n";

    vortex::SimplexSolver solver;

    // Test 1: c size != A cols
    {
        std::vector<double> c = {1.0, 2.0, 3.0}; // 3 vars
        vortex::Matrix<double> A = {{1.0, 2.0}}; // 2 cols
        std::vector<double> b = {5.0};

        auto res = solver.solve(c, A, b);
        record_result("Reject dimension mismatch: c size != A cols",
                      res.status == vortex::SolverStatus::INVALID_INPUT);
    }

    // Test 2: b size != A rows
    {
        std::vector<double> c = {1.0, 2.0};
        vortex::Matrix<double> A = {
            {1.0, 2.0},
            {3.0, 4.0}
        }; // 2 rows
        std::vector<double> b = {5.0}; // 1 element

        auto res = solver.solve(c, A, b);
        record_result("Reject dimension mismatch: b size != A rows",
                      res.status == vortex::SolverStatus::INVALID_INPUT);
    }

    // Test 3: Empty inputs
    {
        std::vector<double> empty_c;
        vortex::Matrix<double> empty_A;
        std::vector<double> empty_b;

        auto res = solver.solve(empty_c, empty_A, empty_b);
        record_result("Reject empty input vectors and matrix",
                      res.status == vortex::SolverStatus::INVALID_INPUT);
    }
}

// ==============================================================================
// 6. Unsupported Input Cases (Negative RHS)
// ==============================================================================
void test_unsupported_inputs() {
    std::cout << "\n--- Running Suite: Unsupported Negative RHS Rejection ---\n";

    vortex::SimplexSolver solver;

    std::vector<double> c = {3.0, 5.0};
    vortex::Matrix<double> A = {
        {1.0, 1.0},
        {2.0, 1.0}
    };
    std::vector<double> b_negative = {4.0, -2.0}; // Negative RHS!

    auto res = solver.solve(c, A, b_negative);
    record_result("Explicitly reject negative RHS (b_i < 0)",
                  res.status == vortex::SolverStatus::INVALID_INPUT);
    record_result("Error message details negative RHS rejection",
                  res.message.find("negative RHS detected") != std::string::npos);
}

// ==============================================================================
// 7. Max Iterations Exceeded Check
// ==============================================================================
void test_max_iterations_limit() {
    std::cout << "\n--- Running Suite: Max Iterations Enforcement ---\n";

    std::vector<double> c = {3.0, 5.0};
    vortex::Matrix<double> A = {
        {1.0, 0.0},
        {0.0, 2.0},
        {3.0, 2.0}
    };
    std::vector<double> b = {4.0, 12.0, 18.0};

    vortex::SimplexOptions opt;
    opt.max_iterations = 0; // Force 0 iterations limit
    vortex::SimplexSolver solver(opt);
    auto res = solver.solve(c, A, b);

    record_result("Stops when max_iterations reached",
                  res.status == vortex::SolverStatus::MAX_ITERATIONS_REACHED);
}

int main() {
    std::cout << "=========================================================\n";
    std::cout << "        VORTEX Simplex Solver Test Suite (Phase 2)       \n";
    std::cout << "=========================================================\n";

    test_mandatory_problem();
    test_additional_bounded_problem();
    test_unbounded_problem();
    test_degenerate_problem();
    test_invalid_dimensions();
    test_unsupported_inputs();
    test_max_iterations_limit();

    std::cout << "\n=========================================================\n";
    std::cout << "  SUMMARY: " << passed_tests << " / " << total_tests << " tests passed.\n";
    std::cout << "=========================================================\n";

    return (passed_tests == total_tests) ? 0 : 1;
}
