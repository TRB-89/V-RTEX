#include "vortex/matrix.hpp"
#include "simplex.hpp"
#include "vortex/lu_decomposition.hpp"
#include "vortex/revised_simplex.hpp"
#include <iostream>
#include <iomanip>

void demo_linear_algebra() {
    std::cout << "=========================================================\n";
    std::cout << "  PART 1: High-Performance Matrix Engine (Phase 1)       \n";
    std::cout << "=========================================================\n\n";

    // Matrix Initialization
    vortex::Matrix<double> A = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0}
    };

    vortex::Matrix<double> B = {
        {9.0, 8.0, 7.0},
        {6.0, 5.0, 4.0},
        {3.0, 2.0, 1.0}
    };

    std::cout << "Matrix A (3x3):\n" << A << "\n";
    std::cout << "Matrix B (3x3):\n" << B << "\n";

    std::cout << "Matrix Addition (A + B):\n" << (A + B) << "\n";
    std::cout << "Matrix Transpose (A^T):\n" << A.transpose() << "\n";

    vortex::Matrix<double> M1 = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0}
    }; // 2x3

    vortex::Matrix<double> M2 = {
        {7.0, 8.0},
        {9.0, 1.0},
        {2.0, 3.0}
    }; // 3x2

    std::cout << "Rectangular Multiplication (2x3 * 3x2):\n" << (M1 * M2) << "\n";
}

void demo_simplex_optimization() {
    std::cout << "\n=========================================================\n";
    std::cout << "  PART 2: CPU Simplex LP Solver (Phase 2)                \n";
    std::cout << "=========================================================\n\n";

    std::cout << "Solving SIH26119 Benchmark Problem:\n";
    std::cout << "  Maximize   Z = 3x + 5y\n";
    std::cout << "  Subject to:   x          <= 4\n";
    std::cout << "                    2y     <= 12\n";
    std::cout << "               3x + 2y     <= 18\n";
    std::cout << "               x, y        >= 0\n\n";

    std::vector<double> c = {3.0, 5.0};
    vortex::Matrix<double> A = {
        {1.0, 0.0},
        {0.0, 2.0},
        {3.0, 2.0}
    };
    std::vector<double> b = {4.0, 12.0, 18.0};

    vortex::SimplexOptions options;
    options.verbose = true;
    options.pivot_rule = vortex::PivotRule::DANTZIG;

    vortex::SimplexSolver solver(options);
    vortex::SimplexResult result = solver.solve(c, A, b);

    std::cout << "---------------- Solver Results ----------------\n";
    std::cout << "Status:          " << vortex::to_string(result.status) << "\n";
    std::cout << "Message:         " << result.message << "\n";
    std::cout << "Total Pivots:    " << result.iterations << "\n";
    std::cout << "Objective Z*:    " << std::fixed << std::setprecision(4) << result.objective_value << "\n";
    std::cout << "Variables:\n";
    std::cout << "  x = " << result.variable_values[0] << " (Expected: 2.0)\n";
    std::cout << "  y = " << result.variable_values[1] << " (Expected: 6.0)\n";
    std::cout << "Slack Variables:\n";
    for (std::size_t i = 0; i < result.slack_values.size(); ++i) {
        std::cout << "  s" << (i + 1) << " = " << result.slack_values[i] << "\n";
    }

    std::cout << "\nFinal Tableau:\n";
    result.final_tableau.print(std::cout, 2);
}

void demo_two_phase_simplex() {
    std::cout << "\n=========================================================\n";
    std::cout << "  PART 3: General Two-Phase Simplex LP Solver (Phase 3)  \n";
    std::cout << "=========================================================\n\n";

    std::cout << "Solving Mixed-Constraint Problem with Artificials:\n";
    std::cout << "  Maximize   Z = 3x1 + 2x2\n";
    std::cout << "  Subject to:  2x1 +   x2  <= 18   (Slack)\n";
    std::cout << "               2x1 +  3x2  >= 12   (Surplus + Artificial)\n";
    std::cout << "                x1 +   x2  ==  8   (Artificial)\n";
    std::cout << "               x1, x2      >=  0\n\n";

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
    vortex::SimplexResult res = solver.solve(c, A, b, types, vortex::ObjectiveSense::MAXIMIZE);

    std::cout << "---------------- Two-Phase Results ---------------\n";
    std::cout << "Status:             " << vortex::to_string(res.status) << "\n";
    std::cout << "Message:            " << res.message << "\n";
    std::cout << "Phase I Iterations: " << res.phase1_iterations << "\n";
    std::cout << "Phase II Iterations:" << res.phase2_iterations << "\n";
    std::cout << "Total Iterations:   " << res.iterations << "\n";
    std::cout << "Optimal Z*:         " << std::fixed << std::setprecision(4) << res.objective_value << " (Expected: 24.0)\n";
    std::cout << "Variables:\n";
    std::cout << "  x1 = " << res.variable_values[0] << " (Expected: 8.0)\n";
    std::cout << "  x2 = " << res.variable_values[1] << " (Expected: 0.0)\n";
}

void demo_revised_simplex() {
    std::cout << "\n=========================================================\n";
    std::cout << "  PART 4: Revised Simplex + LU Decomposition (Phase 4)   \n";
    std::cout << "=========================================================\n\n";

    // Sub-part A: LU Decomposition
    std::cout << "[4A] LU Decomposition with Partial Pivoting (P*A = L*U):\n";
    vortex::Matrix<double> M = {
        {2.0, 1.0, 1.0},
        {4.0, -6.0, 0.0},
        {-2.0, 7.0, 2.0}
    };
    std::cout << "Matrix M (3x3):\n" << M << "\n";

    vortex::LUDecomposition<double> lu(M);
    std::cout << "Determinant:     " << lu.determinant() << " (Expected: -16.0)\n";
    std::cout << "Condition ratio: " << lu.condition_ratio() << "\n";
    std::cout << "Row Swaps:       " << lu.num_swaps() << "\n";

    std::vector<double> rhs = {5.0, -2.0, 9.0};
    auto sol = lu.solve(rhs);
    std::cout << "Solve M * x = [5, -2, 9]^T:\n";
    std::cout << "  x = [" << sol[0] << ", " << sol[1] << ", " << sol[2] << "]^T (Expected: [1, 1, 2])\n";
    std::cout << "Residual ||M*x - b||: " << lu.residual_norm(M, sol, rhs) << "\n\n";

    // Sub-part B: Revised Simplex Solver on SIH26119
    std::cout << "[4B] Solving Benchmark Problem via Revised Simplex:\n";
    std::cout << "  Maximize   Z = 3x + 5y\n";
    std::cout << "  Subject to:   x          <= 4\n";
    std::cout << "                    2y     <= 12\n";
    std::cout << "               3x + 2y     <= 18\n";
    std::cout << "               x, y        >= 0\n\n";

    std::vector<double> c = {3.0, 5.0};
    vortex::Matrix<double> A = {
        {1.0, 0.0},
        {0.0, 2.0},
        {3.0, 2.0}
    };
    std::vector<double> b = {4.0, 12.0, 18.0};

    vortex::RevisedSimplexSolver solver;
    auto res = solver.solve(c, A, b);

    std::cout << "---------------- Revised Simplex Results ----------------\n";
    std::cout << "Status:             " << vortex::to_string(res.status) << "\n";
    std::cout << "Message:            " << res.message << "\n";
    std::cout << "Total Pivots:       " << res.iterations << "\n";
    std::cout << "Optimal Z*:         " << std::fixed << std::setprecision(4) << res.objective_value << "\n";
    std::cout << "Variables:\n";
    std::cout << "  x = " << res.variable_values[0] << " (Expected: 2.0)\n";
    std::cout << "  y = " << res.variable_values[1] << " (Expected: 6.0)\n";
    std::cout << "Slack Variables:\n";
    for (std::size_t i = 0; i < res.slack_values.size(); ++i) {
        std::cout << "  s" << (i + 1) << " = " << res.slack_values[i] << "\n";
    }
    std::cout << "Numerical Stability Check:\n";
    std::cout << "  Max Primal Residual: " << res.max_primal_residual << "\n";
    std::cout << "  Max Dual Residual:   " << res.max_dual_residual << "\n";
    std::cout << "  Basis Condition:     " << res.basis_condition << "\n";
}

int main() {
    std::cout << "=========================================================\n";
    std::cout << "               VORTEX OPTIMIZATION ENGINE                 \n";
    std::cout << "      SIH26119 - Core, Simplex & Two-Phase Solver        \n";
    std::cout << "=========================================================\n\n";

    demo_linear_algebra();
    demo_simplex_optimization();
    demo_two_phase_simplex();
    demo_revised_simplex();

    std::cout << "\n=========================================================\n";
    std::cout << " VORTEX Execution & Validation Completed Successfully!   \n";
    std::cout << "=========================================================\n";

    return 0;
}

