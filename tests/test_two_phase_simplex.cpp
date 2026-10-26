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
// 1. Mixed Constraints: LEQ (<=), GEQ (>=), and EQ (==)
//    Max Z = 3x1 + 2x2
//    s.t.  2x1 +  x2 <= 18
//          2x1 + 3x2 >= 12
//           x1 +  x2 == 8
//          x1, x2 >= 0
//    Feasible segment: x1 in [0, 8], x2 = 8 - x1
//    Z = 3x1 + 2(8 - x1) = 16 + x1
//    Optimal: x1 = 8, x2 = 0, Z = 24.0
// ==============================================================================
void test_mixed_constraints() {
    std::cout << "\n--- Running Suite: Mixed Constraints (<=, >=, ==) ---\n";

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

    vortex::SimplexSolver solver;
    auto res = solver.solve(c, A, b, types, vortex::ObjectiveSense::MAXIMIZE);

    record_result("Status is OPTIMAL", res.status == vortex::SolverStatus::OPTIMAL);
    record_result("Objective Z == 24.0", approx_eq(res.objective_value, 24.0),
                  "Got: " + std::to_string(res.objective_value));
    record_result("Variables x1 == 8.0, x2 == 0.0",
                  approx_eq(res.variable_values[0], 8.0) && approx_eq(res.variable_values[1], 0.0),
                  "Got x1=" + std::to_string(res.variable_values[0]) + ", x2=" + std::to_string(res.variable_values[1]));
    record_result("Phase I performed pivots", res.phase1_iterations > 0,
                  "Phase I pivots: " + std::to_string(res.phase1_iterations));
    record_result("Total iterations equal Phase I + Phase II",
                  res.iterations == (res.phase1_iterations + res.phase2_iterations),
                  "Phase I: " + std::to_string(res.phase1_iterations) + ", Phase II: " + std::to_string(res.phase2_iterations));
}

// ==============================================================================
// 2. Negative RHS Normalization
//    Max Z = 2x1 + 3x2
//    s.t. -x1 - x2 <= -4   (Normalized to x1 + x2 >= 4)
//          x1 + 2x2 <= 10
//          x1, x2 >= 0
//    Optimal: x1 = 10, x2 = 0, Z = 20.0
// ==============================================================================
void test_negative_rhs() {
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

    vortex::SimplexSolver solver;
    auto res = solver.solve(c, A, b, types, vortex::ObjectiveSense::MAXIMIZE);

    record_result("Status is OPTIMAL with normalized negative RHS",
                  res.status == vortex::SolverStatus::OPTIMAL);
    record_result("Objective Z == 20.0", approx_eq(res.objective_value, 20.0),
                  "Got: " + std::to_string(res.objective_value));
    record_result("Variables x1 == 10.0, x2 == 0.0",
                  approx_eq(res.variable_values[0], 10.0) && approx_eq(res.variable_values[1], 0.0),
                  "Got x1=" + std::to_string(res.variable_values[0]) + ", x2=" + std::to_string(res.variable_values[1]));
}

// ==============================================================================
// 3. Minimization Problem with Surplus and Artificial Variables
//    Min Z = 4x1 + 2x2
//    s.t. 3x1 +  x2 >= 27
//          x1 +  x2 >= 21
//          x1 + 2x2 >= 30
//         x1, x2 >= 0
//    Optimal: x1 = 3, x2 = 18, Z = 48.0
// ==============================================================================
void test_minimization_problem() {
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

    vortex::SimplexSolver solver;
    auto res = solver.solve(c, A, b, types, vortex::ObjectiveSense::MINIMIZE);

    record_result("Status is OPTIMAL for Minimization",
                  res.status == vortex::SolverStatus::OPTIMAL);
    record_result("Minimum Objective Z == 48.0", approx_eq(res.objective_value, 48.0),
                  "Got: " + std::to_string(res.objective_value));
    record_result("Variables x1 == 3.0, x2 == 18.0",
                  approx_eq(res.variable_values[0], 3.0) && approx_eq(res.variable_values[1], 18.0),
                  "Got x1=" + std::to_string(res.variable_values[0]) + ", x2=" + std::to_string(res.variable_values[1]));
}

// ==============================================================================
// 4. Infeasibility Detection (Phase I W > 0)
//    Max Z = 5x1 + 3x2
//    s.t. x1 + x2 <= 2
//         x1 + x2 >= 5
//         x1, x2 >= 0
//    Result: INFEASIBLE
// ==============================================================================
void test_infeasibility_detection() {
    std::cout << "\n--- Running Suite: Infeasibility Detection ---\n";

    std::vector<double> c = {5.0, 3.0};
    vortex::Matrix<double> A = {
        {1.0, 1.0},
        {1.0, 1.0}
    };
    std::vector<double> b = {2.0, 5.0};
    std::vector<vortex::ConstraintType> types = {
        vortex::ConstraintType::LEQ,
        vortex::ConstraintType::GEQ
    };

    vortex::SimplexSolver solver;
    auto res = solver.solve(c, A, b, types, vortex::ObjectiveSense::MAXIMIZE);

    record_result("Correctly detects INFEASIBLE status",
                  res.status == vortex::SolverStatus::INFEASIBLE);
    record_result("Diagnostic message identifies artificial sum W > 0",
                  res.message.find("infeasible") != std::string::npos);
}

// ==============================================================================
// 5. Unboundedness Detection with General Constraints
//    Max Z = 3x1 - x2
//    s.t. -x1 + x2 <= 5
//               x2 >= 2
//         x1, x2 >= 0
//    Result: UNBOUNDED
// ==============================================================================
void test_unboundedness_general() {
    std::cout << "\n--- Running Suite: Unboundedness in General LP ---\n";

    std::vector<double> c = {3.0, -1.0};
    vortex::Matrix<double> A = {
        {-1.0, 1.0},
        { 0.0, 1.0}
    };
    std::vector<double> b = {5.0, 2.0};
    std::vector<vortex::ConstraintType> types = {
        vortex::ConstraintType::LEQ,
        vortex::ConstraintType::GEQ
    };

    vortex::SimplexSolver solver;
    auto res = solver.solve(c, A, b, types, vortex::ObjectiveSense::MAXIMIZE);

    record_result("Correctly detects UNBOUNDED status in Two-Phase LP",
                  res.status == vortex::SolverStatus::UNBOUNDED);
}

// ==============================================================================
// 6. LinearProgram Struct API Convenience
// ==============================================================================
void test_linear_program_struct() {
    std::cout << "\n--- Running Suite: LinearProgram Struct API ---\n";

    vortex::LinearProgram lp;
    lp.sense = vortex::ObjectiveSense::MAXIMIZE;
    lp.c = {3.0, 5.0};
    lp.A = {
        {1.0, 0.0},
        {0.0, 2.0},
        {3.0, 2.0}
    };
    lp.b = {4.0, 12.0, 18.0};
    lp.constraint_types = {
        vortex::ConstraintType::LEQ,
        vortex::ConstraintType::LEQ,
        vortex::ConstraintType::LEQ
    };

    vortex::SimplexSolver solver;
    auto res = solver.solve(lp);

    record_result("LinearProgram struct solves to OPTIMAL",
                  res.status == vortex::SolverStatus::OPTIMAL);
    record_result("Objective value Z == 36.0",
                  approx_eq(res.objective_value, 36.0));
    record_result("Variables x == 2.0, y == 6.0",
                  approx_eq(res.variable_values[0], 2.0) && approx_eq(res.variable_values[1], 6.0));
}

int main() {
    std::cout << "=========================================================\n";
    std::cout << "   VORTEX General Two-Phase Simplex Test Suite (Phase 3) \n";
    std::cout << "=========================================================\n";

    test_mixed_constraints();
    test_negative_rhs();
    test_minimization_problem();
    test_infeasibility_detection();
    test_unboundedness_general();
    test_linear_program_struct();

    std::cout << "\n=========================================================\n";
    std::cout << "  SUMMARY: " << passed_tests << " / " << total_tests << " tests passed.\n";
    std::cout << "=========================================================\n";

    return (passed_tests == total_tests) ? 0 : 1;
}
