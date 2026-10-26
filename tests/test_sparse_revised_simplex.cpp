#include "vortex/matrix.hpp"
#include "vortex/sparse_matrix.hpp"
#include "vortex/sparse_revised_simplex.hpp"
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
// 1. SIH26119 Standard Benchmark Problem with Sparse Solver
//    Max Z = 3x + 5y
//    s.t.  x        <= 4
//              2y   <= 12
//         3x + 2y   <= 18
//         x, y >= 0
//    Expected: x = 2.0, y = 6.0, Z = 36.0
// ==============================================================================
void test_sparse_standard_sih_benchmark() {
    std::cout << "\n--- Running Suite: SIH26119 Benchmark via Sparse Revised Simplex ---\n";

    std::vector<double> c = {3.0, 5.0};
    vortex::Matrix<double> A_dense = {
        {1.0, 0.0},
        {0.0, 2.0},
        {3.0, 2.0}
    };
    std::vector<double> b = {4.0, 12.0, 18.0};

    auto A_csc = vortex::CSCMatrix<double>::from_dense(A_dense);

    vortex::SparseRevisedSimplexSolver solver;
    auto res = solver.solve(c, A_csc, b);

    record_result("Status is OPTIMAL", res.status == vortex::SolverStatus::OPTIMAL);
    record_result("Optimal Z == 36.0", approx_eq(res.objective_value, 36.0),
                  "Got: " + std::to_string(res.objective_value));
    record_result("Variables x == 2.0, y == 6.0",
                  approx_eq(res.variable_values[0], 2.0) && approx_eq(res.variable_values[1], 6.0),
                  "Got x=" + std::to_string(res.variable_values[0]) + ", y=" + std::to_string(res.variable_values[1]));
    record_result("Slack values: s1 == 2.0, s2 == 0.0, s3 == 0.0",
                  approx_eq(res.slack_values[0], 2.0) &&
                  approx_eq(res.slack_values[1], 0.0) &&
                  approx_eq(res.slack_values[2], 0.0));
    record_result("Sparsity recorded correctly (> 0)", res.matrix_sparsity > 0.4);
    record_result("Residual ||B*x_B - b|| < 1e-10", res.max_primal_residual < 1e-10);
}

// ==============================================================================
// 2. Sparse Pricing / Reduced Costs Unit Validation
// ==============================================================================
void test_sparse_pricing() {
    std::cout << "\n--- Running Suite: Sparse Pricing (CSC Column Dot) vs Dense ---\n";

    std::vector<double> c = {2.0, 3.0, 1.0, 0.0, 0.0};
    vortex::Matrix<double> A_dense = {
        {1.0, 0.0, 2.0, 1.0, 0.0},
        {0.0, 4.0, 0.0, 0.0, 1.0}
    };
    auto A_csc = vortex::CSCMatrix<double>::from_dense(A_dense);

    std::vector<double> pi = {0.5, 1.5};

    vortex::SparseRevisedSimplexSolver sparse_solver;
    auto red_costs_sparse = sparse_solver.compute_sparse_reduced_costs(c, A_csc, pi);

    // Manual dense calculation:
    // c_bar[0] = 2.0 - (0.5*1.0 + 1.5*0.0) = 1.5
    // c_bar[1] = 3.0 - (0.5*0.0 + 1.5*4.0) = -3.0
    // c_bar[2] = 1.0 - (0.5*2.0 + 1.5*0.0) = 0.0
    // c_bar[3] = 0.0 - (0.5*1.0 + 1.5*0.0) = -0.5
    // c_bar[4] = 0.0 - (0.5*0.0 + 1.5*1.0) = -1.5

    record_result("Sparse reduced cost c_bar[0] == 1.5", approx_eq(red_costs_sparse[0], 1.5));
    record_result("Sparse reduced cost c_bar[1] == -3.0", approx_eq(red_costs_sparse[1], -3.0));
    record_result("Sparse reduced cost c_bar[2] == 0.0", approx_eq(red_costs_sparse[2], 0.0));
    record_result("Sparse reduced cost c_bar[3] == -0.5", approx_eq(red_costs_sparse[3], -0.5));
    record_result("Sparse reduced cost c_bar[4] == -1.5", approx_eq(red_costs_sparse[4], -1.5));
}

// ==============================================================================
// 3. Mixed Constraints via Two-Phase Sparse Revised Simplex
// ==============================================================================
void test_sparse_two_phase_mixed() {
    std::cout << "\n--- Running Suite: Mixed Constraints (Two-Phase Sparse Revised Simplex) ---\n";

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

    vortex::SparseRevisedSimplexSolver solver;
    auto res = solver.solve(c, A, b, types, vortex::ObjectiveSense::MAXIMIZE);

    record_result("Status is OPTIMAL", res.status == vortex::SolverStatus::OPTIMAL);
    record_result("Optimal Z == 24.0", approx_eq(res.objective_value, 24.0),
                  "Got: " + std::to_string(res.objective_value));
    record_result("Variables x1 == 8.0, x2 == 0.0",
                  approx_eq(res.variable_values[0], 8.0) && approx_eq(res.variable_values[1], 0.0));
    record_result("Phase I iterations > 0", res.phase1_iterations > 0);
}

// ==============================================================================
// 4. Large Sparse LP Cross-Validation (Sparse Revised vs Dense Revised vs Tableau)
// ==============================================================================
void test_large_sparse_cross_validation() {
    std::cout << "\n--- Running Suite: Large Sparse LP Cross-Validation ---\n";

    // 10 variables, 6 sparse constraints
    const std::size_t n = 10;
    const std::size_t m = 6;

    std::vector<double> c = {5.0, 4.0, 6.0, 3.0, 2.0, 4.0, 5.0, 3.0, 2.0, 1.0};
    vortex::Matrix<double> A_dense(m, n, 0.0);

    // Create sparse banded constraint matrix
    A_dense(0, 0) = 1.0; A_dense(0, 1) = 2.0;
    A_dense(1, 1) = 1.0; A_dense(1, 2) = 1.5; A_dense(1, 3) = 1.0;
    A_dense(2, 3) = 2.0; A_dense(2, 4) = 1.0;
    A_dense(3, 4) = 1.0; A_dense(3, 5) = 3.0; A_dense(3, 6) = 1.0;
    A_dense(4, 6) = 2.0; A_dense(4, 7) = 1.0; A_dense(4, 8) = 1.0;
    A_dense(5, 8) = 1.0; A_dense(5, 9) = 2.0;

    std::vector<double> b = {10.0, 15.0, 12.0, 18.0, 14.0, 8.0};

    auto A_csc = vortex::CSCMatrix<double>::from_dense(A_dense);

    vortex::SimplexSolver tableau_solver;
    auto res_tableau = tableau_solver.solve(c, A_dense, b);

    vortex::RevisedSimplexSolver dense_revised_solver;
    auto res_dense = dense_revised_solver.solve(c, A_dense, b);

    vortex::SparseRevisedSimplexSolver sparse_solver;
    auto res_sparse = sparse_solver.solve(c, A_csc, b);

    record_result("Tableau, Dense Revised, and Sparse Revised all report OPTIMAL",
                  res_tableau.status == vortex::SolverStatus::OPTIMAL &&
                  res_dense.status == vortex::SolverStatus::OPTIMAL &&
                  res_sparse.status == vortex::SolverStatus::OPTIMAL);

    record_result("Optimal Z matches across all three solvers",
                  approx_eq(res_tableau.objective_value, res_sparse.objective_value) &&
                  approx_eq(res_dense.objective_value, res_sparse.objective_value),
                  "Tableau: " + std::to_string(res_tableau.objective_value) +
                  ", Dense: " + std::to_string(res_dense.objective_value) +
                  ", Sparse: " + std::to_string(res_sparse.objective_value));

    bool vars_match = true;
    for (std::size_t i = 0; i < n; ++i) {
        if (!approx_eq(res_tableau.variable_values[i], res_sparse.variable_values[i], 1e-4)) {
            vars_match = false;
        }
    }
    record_result("All variable values x match identically across solvers", vars_match);
    record_result("Memory savings ratio is positive and tracked", res_sparse.memory_savings_ratio > 1.0,
                  "Ratio: " + std::to_string(res_sparse.memory_savings_ratio));
}

// ==============================================================================
// 5. Infeasibility and Unboundedness with Sparse Engine
// ==============================================================================
void test_sparse_infeasible_and_unbounded() {
    std::cout << "\n--- Running Suite: Infeasibility & Unboundedness in Sparse LP ---\n";

    vortex::SparseRevisedSimplexSolver solver;

    // Infeasible
    std::vector<double> c_inf = {3.0, 4.0};
    vortex::Matrix<double> A_inf = {
        {1.0, 1.0},
        {1.0, 1.0}
    };
    std::vector<double> b_inf = {2.0, 6.0};
    std::vector<vortex::ConstraintType> types_inf = {
        vortex::ConstraintType::LEQ,
        vortex::ConstraintType::GEQ
    };
    auto res_inf = solver.solve(c_inf, A_inf, b_inf, types_inf, vortex::ObjectiveSense::MAXIMIZE);
    record_result("Sparse solver correctly detects INFEASIBLE",
                  res_inf.status == vortex::SolverStatus::INFEASIBLE);

    // Unbounded
    std::vector<double> c_unb = {2.0, -1.0};
    vortex::Matrix<double> A_unb = {
        {-1.0, 1.0},
        { 0.0, 1.0}
    };
    std::vector<double> b_unb = {4.0, 2.0};
    std::vector<vortex::ConstraintType> types_unb = {
        vortex::ConstraintType::LEQ,
        vortex::ConstraintType::GEQ
    };
    auto res_unb = solver.solve(c_unb, A_unb, b_unb, types_unb, vortex::ObjectiveSense::MAXIMIZE);
    record_result("Sparse solver correctly detects UNBOUNDED",
                  res_unb.status == vortex::SolverStatus::UNBOUNDED);
}

int main() {
    std::cout << "=========================================================\n";
    std::cout << "   VORTEX Sparse Revised Simplex Unit Tests (Phase 5)    \n";
    std::cout << "=========================================================\n";

    test_sparse_standard_sih_benchmark();
    test_sparse_pricing();
    test_sparse_two_phase_mixed();
    test_large_sparse_cross_validation();
    test_sparse_infeasible_and_unbounded();

    std::cout << "\n=========================================================\n";
    std::cout << "  SUMMARY: " << passed_tests << " / " << total_tests << " tests passed.\n";
    std::cout << "=========================================================\n";

    return (passed_tests == total_tests) ? 0 : 1;
}
