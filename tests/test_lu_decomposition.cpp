#include "vortex/matrix.hpp"
#include "vortex/lu_decomposition.hpp"
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
// 1. Basic LU Factorization and Reconstruction (P * A == L * U)
// ==============================================================================
void test_lu_reconstruction() {
    std::cout << "\n--- Running Suite: LU Factorization & P*A == L*U Reconstruction ---\n";

    // 3x3 matrix requiring row swapping
    vortex::Matrix<double> A = {
        {0.0, 2.0, 1.0},
        {1.0, 1.0, 2.0},
        {2.0, 1.0, 1.0}
    };

    vortex::LUDecomposition<double> lu(A);

    record_result("LU decomposition computes without singularity", !lu.is_singular());
    record_result("Reconstruction verifies P * A == L * U", lu.verify_factorization(A, 1e-9));

    auto L = lu.L();
    auto U = lu.U();
    auto P = lu.P();

    // Check L is unit lower triangular
    bool l_unit_lower = true;
    for (std::size_t i = 0; i < L.rows(); ++i) {
        if (!approx_eq(L(i, i), 1.0)) l_unit_lower = false;
        for (std::size_t j = i + 1; j < L.cols(); ++j) {
            if (!approx_eq(L(i, j), 0.0)) l_unit_lower = false;
        }
    }
    record_result("L matrix is unit lower-triangular", l_unit_lower);

    // Check U is upper triangular
    bool u_upper = true;
    for (std::size_t i = 0; i < U.rows(); ++i) {
        for (std::size_t j = 0; j < i; ++j) {
            if (!approx_eq(U(i, j), 0.0)) u_upper = false;
        }
    }
    record_result("U matrix is upper-triangular", u_upper);

    // Check P is a valid permutation matrix
    auto PA = P * A;
    auto LU = L * U;
    record_result("P * A matches L * U product directly", PA.is_approx(LU, 1e-9));
}

// ==============================================================================
// 2. Forward and Backward Substitution (A * x = b)
// ==============================================================================
void test_forward_backward_solve() {
    std::cout << "\n--- Running Suite: Forward and Backward Substitution (A * x = b) ---\n";

    // System:
    //  2x +  y -  z =  8
    // -3x -  y + 2z = -11
    // -2x +  y + 2z = -3
    // Exact solution: x = 2, y = 3, z = -1
    vortex::Matrix<double> A = {
        { 2.0,  1.0, -1.0},
        {-3.0, -1.0,  2.0},
        {-2.0,  1.0,  2.0}
    };
    std::vector<double> b = {8.0, -11.0, -3.0};

    vortex::LUDecomposition<double> lu(A);
    std::vector<double> x = lu.solve(b);

    record_result("Solution length matches dimension", x.size() == 3);
    record_result("x[0] == 2.0", approx_eq(x[0], 2.0), "Got: " + std::to_string(x[0]));
    record_result("x[1] == 3.0", approx_eq(x[1], 3.0), "Got: " + std::to_string(x[1]));
    record_result("x[2] == -1.0", approx_eq(x[2], -1.0), "Got: " + std::to_string(x[2]));

    double res = lu.residual_norm(A, x, b);
    record_result("Residual ||A*x - b|| is near machine epsilon", res < 1e-12,
                  "Residual: " + std::to_string(res));
}

// ==============================================================================
// 3. Transpose Solve (A^T * pi = c)
// ==============================================================================
void test_transpose_solve() {
    std::cout << "\n--- Running Suite: Transpose Solve (A^T * pi = c) ---\n";

    vortex::Matrix<double> A = {
        {1.0, 2.0, 4.0},
        {3.0, 8.0, 14.0},
        {2.0, 6.0, 13.0}
    };
    std::vector<double> c = {3.0, 5.0, 2.0};

    vortex::LUDecomposition<double> lu(A);
    std::vector<double> pi = lu.solve_transpose(c);

    double res_t = lu.residual_norm_transpose(A, pi, c);
    record_result("Transpose residual ||A^T * pi - c|| is near zero", res_t < 1e-12,
                  "Residual: " + std::to_string(res_t));

    // Verify directly with A.transpose() * pi
    auto AT = A.transpose();
    vortex::LUDecomposition<double> lu_at(AT);
    auto pi_direct = lu_at.solve(c);

    bool pi_matches = true;
    for (std::size_t i = 0; i < pi.size(); ++i) {
        if (!approx_eq(pi[i], pi_direct[i], 1e-9)) pi_matches = false;
    }
    record_result("solve_transpose matches direct solve on A^T", pi_matches);
}

// ==============================================================================
// 4. Matrix Inversion & Determinant
// ==============================================================================
void test_inverse_and_determinant() {
    std::cout << "\n--- Running Suite: Inversion & Determinant ---\n";

    vortex::Matrix<double> A = {
        {4.0, 7.0},
        {2.0, 6.0}
    };
    // Det = 4*6 - 7*2 = 24 - 14 = 10
    // Inv = 1/10 * [6, -7; -2, 4] = [0.6, -0.7; -0.2, 0.4]

    vortex::LUDecomposition<double> lu(A);
    record_result("Determinant equals 10.0", approx_eq(lu.determinant(), 10.0),
                  "Got: " + std::to_string(lu.determinant()));

    auto inv = lu.inverse();
    record_result("Inverse entry (0, 0) == 0.6", approx_eq(inv(0, 0), 0.6));
    record_result("Inverse entry (0, 1) == -0.7", approx_eq(inv(0, 1), -0.7));
    record_result("Inverse entry (1, 0) == -0.2", approx_eq(inv(1, 0), -0.2));
    record_result("Inverse entry (1, 1) == 0.4", approx_eq(inv(1, 1), 0.4));

    auto I = A * inv;
    record_result("A * A^-1 == Identity", I.is_approx(vortex::Matrix<double>::identity(2), 1e-9));
}

// ==============================================================================
// 5. Singularity Detection
// ==============================================================================
void test_singularity_detection() {
    std::cout << "\n--- Running Suite: Singularity & Rank Deficiency Detection ---\n";

    // Linearly dependent rows (Row 2 = 2 * Row 1)
    vortex::Matrix<double> singular = {
        {1.0, 2.0, 3.0},
        {2.0, 4.0, 6.0},
        {5.0, 7.0, 8.0}
    };

    vortex::LUDecomposition<double> lu(singular);
    record_result("Correctly detects singular matrix", lu.is_singular());
    record_result("Determinant of singular matrix is zero", approx_eq(lu.determinant(), 0.0));

    bool caught_exception = false;
    try {
        std::vector<double> b = {1.0, 2.0, 3.0};
        [[maybe_unused]] auto dummy = lu.solve(b);
    } catch (const std::runtime_error&) {
        caught_exception = true;
    }
    record_result("Calling solve on singular matrix throws runtime_error", caught_exception);
}

// ==============================================================================
// 6. Ill-Conditioned & Hilbert Matrix Stability Checks
// ==============================================================================
void test_ill_conditioned_stability() {
    std::cout << "\n--- Running Suite: Numerical Stability on Ill-Conditioned Matrix ---\n";

    // 4x4 Hilbert Matrix: H_ij = 1 / (i + j + 1)
    vortex::Matrix<double> H(4, 4);
    for (std::size_t i = 0; i < 4; ++i) {
        for (std::size_t j = 0; j < 4; ++j) {
            H(i, j) = 1.0 / static_cast<double>(i + j + 1);
        }
    }

    // Exact b for x = [1, 1, 1, 1]^T
    std::vector<double> x_exact = {1.0, 1.0, 1.0, 1.0};
    std::vector<double> b(4, 0.0);
    for (std::size_t i = 0; i < 4; ++i) {
        for (std::size_t j = 0; j < 4; ++j) {
            b[i] += H(i, j) * x_exact[j];
        }
    }

    vortex::LUDecomposition<double> lu(H);
    record_result("Hilbert matrix decomposed without failure", !lu.is_singular());

    std::vector<double> x_computed = lu.solve(b);
    double res = lu.residual_norm(H, x_computed, b);
    record_result("Hilbert solve residual ||H*x - b|| < 1e-10", res < 1e-10,
                  "Residual: " + std::to_string(res));

    record_result("Condition ratio is positive and small",
                  lu.condition_ratio() > 0.0 && lu.condition_ratio() < 1e-2,
                  "Condition ratio: " + std::to_string(lu.condition_ratio()));
}

int main() {
    std::cout << "=========================================================\n";
    std::cout << "     VORTEX LU Decomposition Unit Test Suite (Phase 4)   \n";
    std::cout << "=========================================================\n";

    test_lu_reconstruction();
    test_forward_backward_solve();
    test_transpose_solve();
    test_inverse_and_determinant();
    test_singularity_detection();
    test_ill_conditioned_stability();

    std::cout << "\n=========================================================\n";
    std::cout << "  SUMMARY: " << passed_tests << " / " << total_tests << " tests passed.\n";
    std::cout << "=========================================================\n";

    return (passed_tests == total_tests) ? 0 : 1;
}
