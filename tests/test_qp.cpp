#include "vortex/qp.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <string>

// Simple assertion helper macro for tests
#define ASSERT_TRUE(cond) \
    do { \
        if (!(cond)) { \
            std::cerr << "Assertion failed: " #cond << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            return false; \
        } \
    } while (0)

#define ASSERT_NEAR(val, expected, tol) \
    do { \
        double diff = std::abs((val) - (expected)); \
        if (diff > (tol)) { \
            std::cerr << "Assertion failed: |" #val " - " #expected "| = " << diff \
                      << " > " << (tol) << " at " << __FILE__ << ":" << __LINE__ \
                      << " (val=" << (val) << ", expected=" << (expected) << ")" << std::endl; \
            return false; \
        } \
    } while (0)

namespace {

bool test_unconstrained_positive_definite() {
    std::cout << "[Test 1] Unconstrained Positive-Definite QP..." << std::endl;
    // Minimize 0.5 * x^T * Q * x + c^T * x
    // Q = [[4, 1], [1, 2]], c = [-3, -1]^T
    // Q * x = -c => 4 x1 + x2 = 3, x1 + 2 x2 = 1
    // det(Q) = 7 => x1 = 5/7 (~0.7142857), x2 = 1/7 (~0.1428571)
    // Z* = -8/7 (~-1.1428571)
    vortex::Matrix<double> Q(2, 2);
    Q(0, 0) = 4.0; Q(0, 1) = 1.0;
    Q(1, 0) = 1.0; Q(1, 1) = 2.0;

    std::vector<double> c = {-3.0, -1.0};

    vortex::ConvexQPSolver solver;
    vortex::QPResult result = solver.solve(Q, c);

    ASSERT_TRUE(result.status == vortex::QPSolverStatus::OPTIMAL);
    ASSERT_NEAR(result.x[0], 5.0 / 7.0, 1e-6);
    ASSERT_NEAR(result.x[1], 1.0 / 7.0, 1e-6);
    ASSERT_NEAR(result.objective_value, -8.0 / 7.0, 1e-6);
    ASSERT_TRUE(result.dual_residual <= 1e-6);
    return true;
}

bool test_active_inequality() {
    std::cout << "[Test 2] Convex QP with an Active Inequality..." << std::endl;
    // Minimize 0.5 * (x1^2 + x2^2)
    // Subject to x1 + x2 >= 2  =>  -x1 - x2 <= -2
    // Unconstrained min is (0, 0), which violates the constraint.
    // Optimal point: x* = (1, 1), Z* = 1.0, active lambda = 1.0
    vortex::Matrix<double> Q(2, 2);
    Q(0, 0) = 1.0; Q(0, 1) = 0.0;
    Q(1, 0) = 0.0; Q(1, 1) = 1.0;

    std::vector<double> c = {0.0, 0.0};

    vortex::Matrix<double> A(1, 2);
    A(0, 0) = -1.0; A(0, 1) = -1.0;
    std::vector<double> b = {-2.0};

    vortex::ConvexQPSolver solver;
    vortex::QPResult result = solver.solve(Q, c, A, b);

    ASSERT_TRUE(result.status == vortex::QPSolverStatus::OPTIMAL);
    ASSERT_NEAR(result.x[0], 1.0, 1e-5);
    ASSERT_NEAR(result.x[1], 1.0, 1e-5);
    ASSERT_NEAR(result.objective_value, 1.0, 1e-5);
    ASSERT_NEAR(result.lambda[0], 1.0, 1e-4);
    ASSERT_NEAR(result.slack[0], 0.0, 1e-5);
    ASSERT_TRUE(result.primal_residual <= 1e-5);
    ASSERT_TRUE(result.dual_residual <= 1e-5);
    return true;
}

bool test_inactive_inequality() {
    std::cout << "[Test 3] Convex QP with an Inactive Inequality..." << std::endl;
    // Minimize 0.5 * (x1^2 + x2^2)
    // Subject to x1 + x2 <= 2
    // Unconstrained min is (0, 0), where 0 + 0 = 0 <= 2 (strictly inactive).
    // Optimal point: x* = (0, 0), Z* = 0.0, lambda = 0.0, slack = 2.0
    vortex::Matrix<double> Q(2, 2);
    Q(0, 0) = 1.0; Q(0, 1) = 0.0;
    Q(1, 0) = 0.0; Q(1, 1) = 1.0;

    std::vector<double> c = {0.0, 0.0};

    vortex::Matrix<double> A(1, 2);
    A(0, 0) = 1.0; A(0, 1) = 1.0;
    std::vector<double> b = {2.0};

    vortex::ConvexQPSolver solver;
    vortex::QPResult result = solver.solve(Q, c, A, b);

    ASSERT_TRUE(result.status == vortex::QPSolverStatus::OPTIMAL);
    ASSERT_NEAR(result.x[0], 0.0, 1e-5);
    ASSERT_NEAR(result.x[1], 0.0, 1e-5);
    ASSERT_NEAR(result.objective_value, 0.0, 1e-5);
    ASSERT_NEAR(result.lambda[0], 0.0, 1e-5);
    ASSERT_NEAR(result.slack[0], 2.0, 1e-5);
    return true;
}

bool test_multiple_inequality_constraints() {
    std::cout << "[Test 4] Convex QP with Multiple Inequalities..." << std::endl;
    // Minimize (x1^2 + x2^2)  [Q = 2*I, c = [0, 0]^T]
    // Subject to:
    //   -x1 - 2*x2 <= -4  (x1 + 2*x2 >= 4)
    //   -x1 <= 0
    //   -x2 <= 0
    // Analytical solution:
    //   x1 = 0.8, x2 = 1.6, Z* = 0.8^2 + 1.6^2 = 3.20, lambda1 = 1.6
    vortex::Matrix<double> Q(2, 2);
    Q(0, 0) = 2.0; Q(0, 1) = 0.0;
    Q(1, 0) = 0.0; Q(1, 1) = 2.0;

    std::vector<double> c = {0.0, 0.0};

    vortex::Matrix<double> A(3, 2);
    A(0, 0) = -1.0; A(0, 1) = -2.0;
    A(1, 0) = -1.0; A(1, 1) = 0.0;
    A(2, 0) = 0.0;  A(2, 1) = -1.0;
    std::vector<double> b = {-4.0, 0.0, 0.0};

    vortex::ConvexQPSolver solver;
    vortex::QPResult result = solver.solve(Q, c, A, b);

    ASSERT_TRUE(result.status == vortex::QPSolverStatus::OPTIMAL);
    ASSERT_NEAR(result.x[0], 0.8, 1e-5);
    ASSERT_NEAR(result.x[1], 1.6, 1e-5);
    ASSERT_NEAR(result.objective_value, 3.20, 1e-5);
    ASSERT_NEAR(result.lambda[0], 1.6, 1e-4);
    ASSERT_NEAR(result.lambda[1], 0.0, 1e-5);
    ASSERT_NEAR(result.lambda[2], 0.0, 1e-5);
    return true;
}

bool test_singular_psd_q() {
    std::cout << "[Test 5] Positive-Semidefinite Singular Q Matrix..." << std::endl;
    // Minimize (x1 - x2)^2 + x1 + x2
    // Q = [[2, -2], [-2, 2]], c = [1, 1]^T (det(Q) = 0, eigenvalues 4, 0)
    // Subject to:
    //   -x1 - x2 <= -4  (x1 + x2 >= 4)
    //   -x1 <= 0
    //   -x2 <= 0
    // Since (x1 - x2)^2 >= 0 and x1 + x2 >= 4, the minimum is achieved at x1 = x2 = 2.0
    // Z* = (2 - 2)^2 + 2 + 2 = 4.0
    vortex::Matrix<double> Q(2, 2);
    Q(0, 0) = 2.0;  Q(0, 1) = -2.0;
    Q(1, 0) = -2.0; Q(1, 1) = 2.0;

    std::vector<double> c = {1.0, 1.0};

    vortex::Matrix<double> A(3, 2);
    A(0, 0) = -1.0; A(0, 1) = -1.0;
    A(1, 0) = -1.0; A(1, 1) = 0.0;
    A(2, 0) = 0.0;  A(2, 1) = -1.0;
    std::vector<double> b = {-4.0, 0.0, 0.0};

    vortex::ConvexQPSolver solver;
    vortex::QPResult result = solver.solve(Q, c, A, b);

    ASSERT_TRUE(result.status == vortex::QPSolverStatus::OPTIMAL);
    ASSERT_NEAR(result.x[0], 2.0, 1e-4);
    ASSERT_NEAR(result.x[1], 2.0, 1e-4);
    ASSERT_NEAR(result.objective_value, 4.0, 1e-4);
    return true;
}

bool test_lp_with_zero_q() {
    std::cout << "[Test 6] LP-like problem with Q = 0..." << std::endl;
    // Minimize 2*x1 + 3*x2
    // Subject to:
    //   -x1 - x2 <= -4   (x1 + x2 >= 4)
    //   -x1 - 2*x2 <= -6 (x1 + 2*x2 >= 6)
    //   -x1 <= 0, -x2 <= 0
    // Intersection of x1 + x2 = 4 and x1 + 2*x2 = 6 is (2, 2).
    // Objective at (2, 2) is 2(2) + 3(2) = 10.0.
    vortex::Matrix<double> Q(2, 2, 0.0);
    std::vector<double> c = {2.0, 3.0};

    vortex::Matrix<double> A(4, 2);
    A(0, 0) = -1.0; A(0, 1) = -1.0;
    A(1, 0) = -1.0; A(1, 1) = -2.0;
    A(2, 0) = -1.0; A(2, 1) = 0.0;
    A(3, 0) = 0.0;  A(3, 1) = -1.0;
    std::vector<double> b = {-4.0, -6.0, 0.0, 0.0};

    vortex::ConvexQPSolver solver;
    vortex::QPResult result = solver.solve(Q, c, A, b);

    ASSERT_TRUE(result.status == vortex::QPSolverStatus::OPTIMAL);
    ASSERT_NEAR(result.x[0], 2.0, 1e-4);
    ASSERT_NEAR(result.x[1], 2.0, 1e-4);
    ASSERT_NEAR(result.objective_value, 10.0, 1e-4);
    return true;
}

bool test_invalid_nonsymmetric_q() {
    std::cout << "[Test 7] Invalid Non-Symmetric Q..." << std::endl;
    vortex::Matrix<double> Q(2, 2);
    Q(0, 0) = 2.0; Q(0, 1) = 1.0;
    Q(1, 0) = 3.0; Q(1, 1) = 2.0; // Q(0, 1) != Q(1, 0)
    std::vector<double> c = {1.0, 1.0};

    vortex::ConvexQPSolver solver;
    vortex::QPResult result = solver.solve(Q, c);

    ASSERT_TRUE(result.status == vortex::QPSolverStatus::INVALID_INPUT);
    ASSERT_TRUE(result.message.find("not symmetric") != std::string::npos);
    return true;
}

bool test_nonconvex_indefinite_q() {
    std::cout << "[Test 8] Non-Convex Indefinite Q..." << std::endl;
    // Q = [[1, 2], [2, 1]] has eigenvalues 3 and -1 (indefinite, saddle point)
    vortex::Matrix<double> Q(2, 2);
    Q(0, 0) = 1.0; Q(0, 1) = 2.0;
    Q(1, 0) = 2.0; Q(1, 1) = 1.0;
    std::vector<double> c = {0.0, 0.0};

    vortex::ConvexQPSolver solver;
    vortex::QPResult result = solver.solve(Q, c);

    ASSERT_TRUE(result.status == vortex::QPSolverStatus::INVALID_INPUT);
    ASSERT_TRUE(result.message.find("indefinite") != std::string::npos);
    return true;
}

bool test_degenerate_and_unbounded_cases() {
    std::cout << "[Test 9] Degenerate and Unbounded Cases..." << std::endl;

    // Case 9a: Unconstrained linear program with c != 0 (must detect UNBOUNDED)
    {
        vortex::Matrix<double> Q(2, 2, 0.0);
        std::vector<double> c = {1.0, -2.0};
        vortex::ConvexQPSolver solver;
        vortex::QPResult result = solver.solve(Q, c);
        ASSERT_TRUE(result.status == vortex::QPSolverStatus::UNBOUNDED);
    }

    // Case 9b: Unconstrained with c = 0, Q = 0 (trivially OPTIMAL at 0)
    {
        vortex::Matrix<double> Q(2, 2, 0.0);
        std::vector<double> c = {0.0, 0.0};
        vortex::ConvexQPSolver solver;
        vortex::QPResult result = solver.solve(Q, c);
        ASSERT_TRUE(result.status == vortex::QPSolverStatus::OPTIMAL);
        ASSERT_NEAR(result.objective_value, 0.0, 1e-12);
    }

    // Case 9c: Ill-conditioned positive definite matrix (condition number ~10^5)
    {
        vortex::Matrix<double> Q(2, 2, 0.0);
        Q(0, 0) = 1e5;
        Q(1, 1) = 1.0;
        std::vector<double> c = {-1e5, -2.0};
        // Optimum: x1 = 1.0, x2 = 2.0, Z* = 0.5(1e5*1 + 1*4) - 1e5(1) - 2(2) = 50002 - 100004 = -50002.0
        vortex::ConvexQPSolver solver;
        vortex::QPResult result = solver.solve(Q, c);
        ASSERT_TRUE(result.status == vortex::QPSolverStatus::OPTIMAL);
        ASSERT_NEAR(result.x[0], 1.0, 1e-4);
        ASSERT_NEAR(result.x[1], 2.0, 1e-4);
        ASSERT_NEAR(result.objective_value, -50002.0, 1e-2);
    }

    return true;
}

bool test_reference_parity_scipy() {
    std::cout << "[Test 10] Independent Reference Parity (SciPy QP Parity Model)..." << std::endl;
    // Model:
    // Min 0.5 * (x1^2 + x2^2 + x3^2) - 2*x1 - 3*x2 - x3
    // Subject to:
    //   x1 + x2 + x3 <= 1
    //   -x1 <= 0, -x2 <= 0, -x3 <= 0
    //
    // Equivalent to projecting c = (2, 3, 1) onto the simplex sum(x) <= 1, x >= 0.
    // Unconstrained gradient: [x1 - 2, x2 - 3, x3 - 1] = 0 => (2, 3, 1), sum = 6 > 1.
    // Active constraint is x1 + x2 + x3 = 1.
    // Using KKT stationarity:
    //   x1 - 2 + lambda = 0 => x1 = 2 - lambda
    //   x2 - 3 + lambda = 0 => x2 = 3 - lambda
    //   x3 - 1 + lambda = 0 => x3 = 1 - lambda
    // x1 + x2 + x3 = 6 - 3*lambda = 1 => 3*lambda = 5 => lambda = 5/3.
    // x1 = 2 - 5/3 = 1/3
    // x2 = 3 - 5/3 = 4/3
    // x3 = 1 - 5/3 = -2/3 < 0 (violates x3 >= 0).
    // Setting x3 = 0:
    //   x1 + x2 = 1
    //   x1 - 2 + lambda = 0 => x1 = 2 - lambda
    //   x2 - 3 + lambda = 0 => x2 = 3 - lambda
    //   x1 + x2 = 5 - 2*lambda = 1 => 2*lambda = 4 => lambda = 2.0.
    //   x1 = 2 - 2 = 0.0
    //   x2 = 3 - 2 = 1.0
    //   x3 = 0.0
    // Objective:
    //   0.5 * (0^2 + 1^2 + 0^2) - 2(0) - 3(1) - 1(0) = 0.5 - 3.0 = -2.5.
    //
    // Tested independently in SciPy via scipy.optimize.minimize (SLSQP / trust-constr):
    //   x* = [0.0, 1.0, 0.0], fun = -2.50000000000.
    vortex::Matrix<double> Q(3, 3, 0.0);
    Q(0, 0) = 1.0; Q(1, 1) = 1.0; Q(2, 2) = 1.0;
    std::vector<double> c = {-2.0, -3.0, -1.0};

    vortex::Matrix<double> A(4, 3, 0.0);
    // x1 + x2 + x3 <= 1
    A(0, 0) = 1.0; A(0, 1) = 1.0; A(0, 2) = 1.0;
    // -x1 <= 0
    A(1, 0) = -1.0;
    // -x2 <= 0
    A(2, 1) = -1.0;
    // -x3 <= 0
    A(3, 2) = -1.0;

    std::vector<double> b = {1.0, 0.0, 0.0, 0.0};

    vortex::ConvexQPSolver solver;
    vortex::QPResult result = solver.solve(Q, c, A, b);

    ASSERT_TRUE(result.status == vortex::QPSolverStatus::OPTIMAL);
    ASSERT_NEAR(result.x[0], 0.0, 1e-3);
    ASSERT_NEAR(result.x[1], 1.0, 1e-3);
    ASSERT_NEAR(result.x[2], 0.0, 1e-3);
    ASSERT_NEAR(result.objective_value, -2.5, 1e-3);
    ASSERT_TRUE(result.primal_residual <= 1e-5);
    ASSERT_TRUE(result.dual_residual <= 1e-5);
    ASSERT_TRUE(result.duality_gap <= 1e-5);
    return true;
}

} // namespace

int main() {
    std::cout << "========================================================\n";
    std::cout << " VØRTEX PHASE 7: CONVEX QP SOLVER UNIT TESTS\n";
    std::cout << "========================================================\n\n";

    bool all_passed = true;
    all_passed &= test_unconstrained_positive_definite();
    all_passed &= test_active_inequality();
    all_passed &= test_inactive_inequality();
    all_passed &= test_multiple_inequality_constraints();
    all_passed &= test_singular_psd_q();
    all_passed &= test_lp_with_zero_q();
    all_passed &= test_invalid_nonsymmetric_q();
    all_passed &= test_nonconvex_indefinite_q();
    all_passed &= test_degenerate_and_unbounded_cases();
    all_passed &= test_reference_parity_scipy();

    std::cout << "\n--------------------------------------------------------\n";
    if (all_passed) {
        std::cout << " ALL CONVEX QP TESTS PASSED SUCCESSFULLY! (10/10)\n";
        std::cout << "--------------------------------------------------------\n";
        return 0;
    } else {
        std::cerr << " SOME CONVEX QP TESTS FAILED!\n";
        std::cout << "--------------------------------------------------------\n";
        return 1;
    }
}
