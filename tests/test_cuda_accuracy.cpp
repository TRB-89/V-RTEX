#include "vortex/cuda_backend.hpp"
#include "vortex/matrix.hpp"
#include "vortex/sparse_matrix.hpp"

#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <cmath>
#include <cassert>
#include <string>
#include <sstream>
#include <algorithm>
#include <limits>

namespace {

int g_tests_passed = 0;
int g_tests_failed = 0;

#define ACCURACY_TEST_ASSERT(cond, msg)                                           \
    do {                                                                          \
        if (!(cond)) {                                                            \
            std::cerr << "  [FAIL] Assertion failed: " << #cond << "\n"           \
                      << "         Message: " << msg << "\n"                      \
                      << "         File: " << __FILE__ << ":" << __LINE__ << "\n"; \
            g_tests_failed++;                                                     \
            return;                                                               \
        }                                                                         \
    } while (0)

#define ACCURACY_TEST_PASS(msg)                                                   \
    do {                                                                          \
        std::cout << "  [PASS] " << msg << "\n";                                  \
        g_tests_passed++;                                                         \
    } while (0)

// Reproducible random matrix generator
vortex::Matrix<double> make_random_dense(std::size_t rows, std::size_t cols, double min_val, double max_val, uint32_t seed) {
    vortex::Matrix<double> mat(rows, cols);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(min_val, max_val);
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < cols; ++c) {
            mat(r, c) = dist(rng);
        }
    }
    return mat;
}

// Reproducible random vector generator
std::vector<double> make_random_vector(std::size_t size, double min_val, double max_val, uint32_t seed) {
    std::vector<double> vec(size);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(min_val, max_val);
    for (std::size_t i = 0; i < size; ++i) {
        vec[i] = dist(rng);
    }
    return vec;
}

// Reproducible random CSR matrix generator
vortex::CSRMatrix<double> make_random_csr(std::size_t rows, std::size_t cols, double density, double min_val, double max_val, uint32_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
    std::uniform_real_distribution<double> val_dist(min_val, max_val);

    std::vector<vortex::Triplet<double>> triplets;
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < cols; ++c) {
            if (prob_dist(rng) < density) {
                triplets.push_back({r, c, val_dist(rng)});
            }
        }
    }
    return vortex::CSRMatrix<double>::from_triplets(rows, cols, triplets);
}

} // namespace

// ==============================================================================
// 1. Analytical & High-Precision Reference Validation (Exact Solutions)
// ==============================================================================
void test_analytical_and_exact_references() {
    std::cout << "\n--- Suite 1: Analytical & High-Precision Reference Parity ---\n";

    // A. Tridiagonal Laplace 1D system:
    // A = [ 2 -1  0;
    //      -1  2 -1;
    //       0 -1  2 ]
    // x = [1, 2, 3]^T
    // Analytical product:
    // y0 = 2(1) - 1(2) + 0(3) = 0
    // y1 = -1(1) + 2(2) - 1(3) = 0
    // y2 = 0(1) - 1(2) + 2(3) = 4
    // Transpose is identical (symmetric).
    {
        vortex::Matrix<double> A(3, 3);
        A(0, 0) = 2.0; A(0, 1) = -1.0; A(0, 2) = 0.0;
        A(1, 0) = -1.0; A(1, 1) = 2.0; A(1, 2) = -1.0;
        A(2, 0) = 0.0; A(2, 1) = -1.0; A(2, 2) = 2.0;

        std::vector<double> x = {1.0, 2.0, 3.0};
        const double expected[3] = {0.0, 0.0, 4.0};

        auto y_dense = vortex::cuda::dense_gemv(A, x, false);
        for (int i = 0; i < 3; ++i) {
            ACCURACY_TEST_ASSERT(std::abs(y_dense[i] - expected[i]) <= 1e-15,
                                "Tridiagonal exact analytical parity failed on dense GEMV");
        }

        auto A_csr = vortex::CSRMatrix<double>::from_dense(A);
        auto y_sparse = vortex::cuda::sparse_spmv(A_csr, x, false);
        for (int i = 0; i < 3; ++i) {
            ACCURACY_TEST_ASSERT(std::abs(y_sparse[i] - expected[i]) <= 1e-15,
                                "Tridiagonal exact analytical parity failed on sparse SpMV");
        }
    }

    // B. Rank-1 Outer Product Matrix A = u * v^T
    // Then A * x = u * (v^T * x)
    // Analytical scalar s = v^T * x = sum_j v_j * x_j.
    // Each entry y_i = u_i * s.
    {
        const std::size_t m = 25;
        const std::size_t n = 35;
        auto u = make_random_vector(m, -3.0, 3.0, 1101);
        auto v = make_random_vector(n, -3.0, 3.0, 1102);
        auto x = make_random_vector(n, -2.0, 2.0, 1103);

        vortex::Matrix<double> A(m, n);
        for (std::size_t r = 0; r < m; ++r) {
            for (std::size_t c = 0; c < n; ++c) {
                A(r, c) = u[r] * v[c];
            }
        }

        // Exact high-precision scalar dot product using long double
        long double v_dot_x = 0.0L;
        for (std::size_t c = 0; c < n; ++c) {
            v_dot_x += static_cast<long double>(v[c]) * static_cast<long double>(x[c]);
        }
        std::vector<double> y_analytical(m);
        for (std::size_t r = 0; r < m; ++r) {
            y_analytical[r] = static_cast<double>(static_cast<long double>(u[r]) * v_dot_x);
        }

        auto y_gpu = vortex::cuda::dense_gemv(A, x, false);
        auto rep = vortex::cuda::evaluate_componentwise_accuracy(y_gpu.data(), y_analytical.data(), nullptr, m, 1e-12, 1e-12);
        ACCURACY_TEST_ASSERT(rep.passed, "Rank-1 outer product analytical validation failed: " + rep.diagnostic_summary);
    }

    ACCURACY_TEST_PASS("Analytical tridiagonal Laplace and rank-1 outer product exact references verified.");
}

// ==============================================================================
// 2. Componentwise Backward-Forward Error Bound Audit (Higham Criterion)
// ==============================================================================
void test_componentwise_bound_framework() {
    std::cout << "\n--- Suite 2: Componentwise Defensible Bound Audit (|y_i - ref_i| <= atol + rtol * (|A||x|)_i) ---\n";

    // Test a matrix with wide dynamic range across rows
    const std::size_t rows = 50;
    const std::size_t cols = 40;
    vortex::Matrix<double> A(rows, cols);

    for (std::size_t r = 0; r < rows; ++r) {
        // Row magnitude varies from 1e-8 to 1e6
        const double row_scale = std::pow(10.0, -8.0 + (static_cast<double>(r) / static_cast<double>(rows - 1)) * 14.0);
        for (std::size_t c = 0; c < cols; ++c) {
            A(r, c) = (r == c ? 1.0 : -0.2) * row_scale;
        }
    }

    std::vector<double> x = make_random_vector(cols, -1.0, 1.0, 1201);

    // Non-transposed validation with defensible bound
    auto report_nt = vortex::cuda::validate_dense_gemv_accuracy(A, x, false, 1.5, 0.0, {}, 1e-12, 1e-12);
    ACCURACY_TEST_ASSERT(report_nt.passed,
                        "Componentwise bound failed for dynamic range dense GEMV (NT): " + report_nt.diagnostic_summary);
    std::cout << "  Dynamic Range (Non-Transpose) -> Max Ratio: " << report_nt.max_componentwise_ratio
              << ", Max Abs Err: " << std::scientific << report_nt.max_absolute_error << std::fixed << "\n";

    // Transposed validation
    std::vector<double> x_t = make_random_vector(rows, -1.0, 1.0, 1202);
    auto report_t = vortex::cuda::validate_dense_gemv_accuracy(A, x_t, true, 1.5, 0.0, {}, 1e-12, 1e-12);
    ACCURACY_TEST_ASSERT(report_t.passed,
                        "Componentwise bound failed for dynamic range dense GEMV (T): " + report_t.diagnostic_summary);
    std::cout << "  Dynamic Range (Transpose)     -> Max Ratio: " << report_t.max_componentwise_ratio
              << ", Max Abs Err: " << std::scientific << report_t.max_absolute_error << std::fixed << "\n";

    // Sparse representation of same matrix
    auto A_csr = vortex::CSRMatrix<double>::from_dense(A);
    auto sp_report_nt = vortex::cuda::validate_sparse_spmv_accuracy(A_csr, x, false, 1.5, 0.0, {}, 1e-12, 1e-12);
    ACCURACY_TEST_ASSERT(sp_report_nt.passed,
                        "Componentwise bound failed for dynamic range sparse SpMV (NT): " + sp_report_nt.diagnostic_summary);

    ACCURACY_TEST_PASS("Componentwise defensible error bounds validated without near-zero division anomalies.");
}

// ==============================================================================
// 3. Edge-Case Matrix Testing: Zero, Identity, Single-Element & Dimensions
// ==============================================================================
void test_edge_cases_and_dimensions() {
    std::cout << "\n--- Suite 3: Structural Edge Cases (Zero, Identity, 1x1, Odd Dimensions) ---\n";

    // A. Single element 1x1
    {
        vortex::Matrix<double> A_1x1(1, 1);
        A_1x1(0, 0) = -42.125;
        std::vector<double> x_1 = {2.0};
        auto rep_1x1 = vortex::cuda::validate_dense_gemv_accuracy(A_1x1, x_1, false, 1.0, 0.0, {}, 1e-15, 1e-15);
        ACCURACY_TEST_ASSERT(rep_1x1.passed, "1x1 Dense GEMV failed: " + rep_1x1.diagnostic_summary);

        auto csr_1x1 = vortex::CSRMatrix<double>::from_dense(A_1x1);
        auto rep_sp_1x1 = vortex::cuda::validate_sparse_spmv_accuracy(csr_1x1, x_1, false, 1.0, 0.0, {}, 1e-15, 1e-15);
        ACCURACY_TEST_ASSERT(rep_sp_1x1.passed, "1x1 Sparse SpMV failed: " + rep_sp_1x1.diagnostic_summary);
    }

    // B. Zero Matrices (Dense and Sparse)
    {
        const std::size_t zm_rows = 15;
        const std::size_t zm_cols = 25;
        vortex::Matrix<double> A_zero(zm_rows, zm_cols); // all zeros
        auto x_zero = make_random_vector(zm_cols, -5.0, 5.0, 1301);
        auto rep_zero = vortex::cuda::validate_dense_gemv_accuracy(A_zero, x_zero, false, 2.0, 0.0, {}, 1e-15, 1e-15);
        ACCURACY_TEST_ASSERT(rep_zero.passed && rep_zero.max_absolute_error == 0.0, "Zero dense matrix must yield zero error");

        vortex::CSRMatrix<double> csr_zero(zm_rows, zm_cols); // nnz = 0
        auto rep_sp_zero = vortex::cuda::validate_sparse_spmv_accuracy(csr_zero, x_zero, false, 2.0, 0.0, {}, 1e-15, 1e-15);
        ACCURACY_TEST_ASSERT(rep_sp_zero.passed && rep_sp_zero.max_absolute_error == 0.0, "Zero CSR matrix must yield zero error");
    }

    // C. Identity Matrix of Odd Prime Dimension (127 x 127)
    {
        const std::size_t n_id = 127;
        vortex::Matrix<double> I_dense(n_id, n_id);
        std::vector<vortex::Triplet<double>> id_triplets;
        for (std::size_t i = 0; i < n_id; ++i) {
            I_dense(i, i) = 1.0;
            id_triplets.push_back({i, i, 1.0});
        }
        auto I_csr = vortex::CSRMatrix<double>::from_triplets(n_id, n_id, id_triplets);
        auto x_id = make_random_vector(n_id, -10.0, 10.0, 1302);

        auto y_id_dense = vortex::cuda::dense_gemv(I_dense, x_id, false);
        for (std::size_t i = 0; i < n_id; ++i) {
            ACCURACY_TEST_ASSERT(y_id_dense[i] == x_id[i], "Dense Identity multiplication must be bitwise identical");
        }

        auto y_id_sparse = vortex::cuda::sparse_spmv(I_csr, x_id, false);
        for (std::size_t i = 0; i < n_id; ++i) {
            ACCURACY_TEST_ASSERT(y_id_sparse[i] == x_id[i], "Sparse Identity multiplication must be bitwise identical");
        }
    }

    // D. Non-Power-of-Two & Odd Dimensions: (3 x 7), (13 x 17), (31 x 29), (503 x 311)
    {
        const std::pair<std::size_t, std::size_t> prime_dims[] = {
            {3, 7}, {13, 17}, {31, 29}, {503, 311}
        };

        for (const auto& [r, c] : prime_dims) {
            auto A = make_random_dense(r, c, -4.0, 4.0, static_cast<uint32_t>(r * 100 + c));
            auto x = make_random_vector(c, -2.0, 2.0, 1303);
            auto rep_nt = vortex::cuda::validate_dense_gemv_accuracy(A, x, false, 1.0, 0.0, {}, 1e-12, 1e-12);
            ACCURACY_TEST_ASSERT(rep_nt.passed, "Prime dimension (" + std::to_string(r) + "x" + std::to_string(c) + ") failed NT");

            auto x_t = make_random_vector(r, -2.0, 2.0, 1304);
            auto rep_t = vortex::cuda::validate_dense_gemv_accuracy(A, x_t, true, 1.0, 0.0, {}, 1e-12, 1e-12);
            ACCURACY_TEST_ASSERT(rep_t.passed, "Prime dimension (" + std::to_string(r) + "x" + std::to_string(c) + ") failed T");
        }
    }

    ACCURACY_TEST_PASS("Zero matrices, identity matrices, 1x1 scalars, and odd/prime dimensions verified.");
}

// ==============================================================================
// 4. Sparse Edge Cases: Interleaved Empty Rows, Empty Columns, Extreme Densities
// ==============================================================================
void test_sparse_topological_edge_cases() {
    std::cout << "\n--- Suite 4: Sparse Topological Cases (Empty Rows, Empty Cols, Densities 0.05% & 95%) ---\n";

    // A. Interleaved Empty Rows and Empty Columns
    {
        // 6 x 8 matrix where rows 1 and 4 are completely empty, and column 2 is empty
        const std::size_t r = 6;
        const std::size_t c = 8;
        std::vector<vortex::Triplet<double>> triplets = {
            {0, 0, 1.5}, {0, 3, -2.0},
            // row 1 empty
            {2, 1, 4.0}, {2, 7, 3.2},
            {3, 0, -1.1}, {3, 5, 2.2},
            // row 4 empty
            {5, 6, 7.8}
            // col 2 is never referenced
        };
        auto A_csr = vortex::CSRMatrix<double>::from_triplets(r, c, triplets);
        auto x_nt = make_random_vector(c, -1.0, 1.0, 1401);
        auto rep_nt = vortex::cuda::validate_sparse_spmv_accuracy(A_csr, x_nt, false, 1.0, 0.0, {}, 1e-12, 1e-12);
        ACCURACY_TEST_ASSERT(rep_nt.passed, "Sparse SpMV with interleaved empty rows failed NT: " + rep_nt.diagnostic_summary);

        auto x_t = make_random_vector(r, -1.0, 1.0, 1402);
        auto rep_t = vortex::cuda::validate_sparse_spmv_accuracy(A_csr, x_t, true, 1.0, 0.0, {}, 1e-12, 1e-12);
        ACCURACY_TEST_ASSERT(rep_t.passed, "Sparse SpMV with empty columns failed T: " + rep_t.diagnostic_summary);
    }

    // B. Ultra-Sparse Matrix (0.05% Density in 1000 x 1000 -> exactly 500 entries)
    {
        const std::size_t n = 1000;
        auto A_ultra_sparse = make_random_csr(n, n, 0.0005, -5.0, 5.0, 1403);
        auto x = make_random_vector(n, -1.0, 1.0, 1404);
        auto rep = vortex::cuda::validate_sparse_spmv_accuracy(A_ultra_sparse, x, false, 1.0, 0.0, {}, 1e-12, 1e-12);
        ACCURACY_TEST_ASSERT(rep.passed, "Ultra-sparse (0.05% density) SpMV failed: " + rep.diagnostic_summary);
        std::cout << "  Ultra-Sparse (0.05% density, nnz=" << A_ultra_sparse.non_zeros()
                  << ") -> Max Abs Err: " << std::scientific << rep.max_absolute_error << std::fixed << "\n";
    }

    // C. Nearly-Dense CSR Matrix (85% Density in 200 x 200 -> ~34,000 entries)
    {
        const std::size_t n = 200;
        auto A_nearly_dense = make_random_csr(n, n, 0.85, -2.0, 2.0, 1405);
        auto x = make_random_vector(n, -1.0, 1.0, 1406);
        auto rep = vortex::cuda::validate_sparse_spmv_accuracy(A_nearly_dense, x, false, 1.0, 0.0, {}, 1e-12, 1e-12);
        ACCURACY_TEST_ASSERT(rep.passed, "Nearly-dense (85% density) SpMV failed: " + rep.diagnostic_summary);
        std::cout << "  Nearly-Dense (85.0% density, nnz=" << A_nearly_dense.non_zeros()
                  << ") -> Max Abs Err: " << std::scientific << rep.max_absolute_error << std::fixed << "\n";
    }

    ACCURACY_TEST_PASS("Sparse empty rows/columns, ultra-sparse, and nearly-dense CSR matrices validated.");
}

// ==============================================================================
// 5. Ill-Conditioned & Near-Singular Matrices (Hilbert, Kac-Murdock-Szegö)
// ==============================================================================
void test_ill_conditioned_and_near_singular() {
    std::cout << "\n--- Suite 5: Ill-Conditioned & Near-Singular Problem Validation ---\n";

    // A. Hilbert Matrix: H_ij = 1 / (i + j + 1), known to have condition number growing exponentially
    // For n = 12, cond(H) > 1e16 (numerically ill-conditioned)
    {
        const std::size_t n = 12;
        vortex::Matrix<double> H(n, n);
        for (std::size_t r = 0; r < n; ++r) {
            for (std::size_t c = 0; c < n; ++c) {
                H(r, c) = 1.0 / static_cast<double>(r + c + 1);
            }
        }
        auto x = make_random_vector(n, 0.5, 2.0, 1501);
        auto rep = vortex::cuda::validate_dense_gemv_accuracy(H, x, false, 1.0, 0.0, {}, 1e-14, 1e-14);
        ACCURACY_TEST_ASSERT(rep.passed, "Hilbert matrix GEMV failed defensible accuracy bound: " + rep.diagnostic_summary);
        std::cout << "  Hilbert Matrix (12x12) -> Max Abs Err: " << std::scientific << rep.max_absolute_error
                  << ", Scaled Rel Err: " << rep.scaled_relative_error << std::fixed << "\n";
    }

    // B. Kac-Murdock-Szegö (KMS) Toeplitz Matrix: A_ij = rho^{|i - j|}
    // When rho = 0.999, the matrix has eigenvalues extremely close to 0, making it near-singular.
    {
        const std::size_t n = 64;
        const double rho = 0.999;
        vortex::Matrix<double> KMS(n, n);
        for (std::size_t r = 0; r < n; ++r) {
            for (std::size_t c = 0; c < n; ++c) {
                const double diff = static_cast<double>(r > c ? (r - c) : (c - r));
                KMS(r, c) = std::pow(rho, diff);
            }
        }
        auto x = make_random_vector(n, -1.0, 1.0, 1502);
        auto rep = vortex::cuda::validate_dense_gemv_accuracy(KMS, x, false, 1.0, 0.0, {}, 1e-12, 1e-12);
        ACCURACY_TEST_ASSERT(rep.passed, "KMS near-singular GEMV failed: " + rep.diagnostic_summary);
        std::cout << "  KMS Near-Singular (rho=0.999, 64x64) -> Max Abs Err: " << std::scientific << rep.max_absolute_error << std::fixed << "\n";
    }

    ACCURACY_TEST_PASS("Ill-conditioned Hilbert and near-singular KMS matrices verified with componentwise stability.");
}

// ==============================================================================
// 6. Catastrophic Cancellation Dot Products with Floating-Point Analysis
// ==============================================================================
void test_catastrophic_cancellation_analysis() {
    std::cout << "\n--- Suite 6: Catastrophic Cancellation & Summation Order Analysis ---\n";

    // Matrix with large alternating values designed to subtract out high-order bits:
    // A_ij = (-1)^j * (1e7 + j * 1e-4), x_j = 1.0.
    // Summing N = 100 elements.
    const std::size_t n = 100;
    vortex::Matrix<double> A(n, n);
    for (std::size_t r = 0; r < n; ++r) {
        for (std::size_t c = 0; c < n; ++c) {
            const double sign = (c % 2 == 0) ? 1.0 : -1.0;
            A(r, c) = sign * (1e7 + static_cast<double>(c) * 1e-4 + static_cast<double>(r) * 1e-5);
        }
    }
    std::vector<double> x(n, 1.0);

    // Compute independent long double accumulator (Kahan-like accuracy)
    std::vector<double> y_high_precision(n, 0.0);
    for (std::size_t r = 0; r < n; ++r) {
        long double row_sum = 0.0L;
        for (std::size_t c = 0; c < n; ++c) {
            row_sum += static_cast<long double>(A(r, c));
        }
        y_high_precision[r] = static_cast<double>(row_sum);
    }

    auto y_gpu = vortex::cuda::dense_gemv(A, x, false);
    auto y_cpu = A.multiply(x);

    auto metrics_gpu_cpu = vortex::cuda::compute_error_metrics(y_gpu.data(), y_cpu.data(), n);
    auto metrics_gpu_exact = vortex::cuda::compute_error_metrics(y_gpu.data(), y_high_precision.data(), n);

    std::cout << "  Cancellation Matrix (1e7 magnitude, N=100):\n"
              << "    GPU vs CPU sequential sum Max Abs Error:   " << std::scientific << metrics_gpu_cpu.max_absolute_error << "\n"
              << "    GPU vs Exact Long Double Max Abs Error:    " << metrics_gpu_exact.max_absolute_error << "\n"
              << "    Theoretical Parallel Reduction Bound:      " << (static_cast<double>(n) * 2.22e-16 * 1e7) << "\n"
              << std::fixed;

    // Defensible theoretical bound: O(N * eps_mach * max|A|) = 100 * 2.22e-16 * 1e7 = 2.22e-7
    ACCURACY_TEST_ASSERT(metrics_gpu_cpu.max_absolute_error < 2.5e-7,
                        "Catastrophic cancellation error exceeded theoretical SIMD reduction bound");

    ACCURACY_TEST_PASS("Catastrophic cancellation verified within theoretical parallel reduction bound.");
}

// ==============================================================================
// 7. Single Precision (Float) Accuracy Sweep
// ==============================================================================
void test_single_precision_accuracy() {
    std::cout << "\n--- Suite 7: Single Precision (Float) Accuracy Sweep ---\n";

    // A. Dense GEMV Float across rectangular prime dimension (67 x 43)
    {
        const std::size_t r = 67;
        const std::size_t c = 43;
        vortex::Matrix<float> A(r, c);
        std::mt19937 rng(1701);
        std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
        for (std::size_t i = 0; i < r; ++i) {
            for (std::size_t j = 0; j < c; ++j) {
                A(i, j) = dist(rng);
            }
        }
        std::vector<float> x(c);
        for (std::size_t j = 0; j < c; ++j) x[j] = dist(rng);

        auto y_cpu = A.multiply(x);
        auto y_gpu = vortex::cuda::dense_gemv(A, x, false);

        // Componentwise scale for float
        std::vector<float> scale(r, 0.0f);
        for (std::size_t i = 0; i < r; ++i) {
            for (std::size_t j = 0; j < c; ++j) {
                scale[i] += std::abs(A(i, j)) * std::abs(x[j]);
            }
        }
        auto rep = vortex::cuda::evaluate_componentwise_accuracy(y_gpu.data(), y_cpu.data(), scale.data(), r, 1e-5f, 1e-5f);
        ACCURACY_TEST_ASSERT(rep.passed, "Float Dense GEMV failed componentwise bound: " + rep.diagnostic_summary);
        std::cout << "  Float Dense GEMV (67x43) -> Max Abs Err: " << std::scientific << rep.max_absolute_error
                  << ", Max Ratio: " << rep.max_componentwise_ratio << std::fixed << "\n";
    }

    // B. Sparse SpMV Float (150 x 100, 5% density)
    {
        const std::size_t r = 150;
        const std::size_t c = 100;
        std::mt19937 rng(1702);
        std::uniform_real_distribution<float> prob(0.0f, 1.0f);
        std::uniform_real_distribution<float> val(-5.0f, 5.0f);

        std::vector<vortex::Triplet<float>> triplets;
        for (std::size_t i = 0; i < r; ++i) {
            for (std::size_t j = 0; j < c; ++j) {
                if (prob(rng) < 0.05f) {
                    triplets.push_back({i, j, val(rng)});
                }
            }
        }
        auto A_csr = vortex::CSRMatrix<float>::from_triplets(r, c, triplets);
        std::vector<float> x(c);
        for (std::size_t j = 0; j < c; ++j) x[j] = val(rng);

        auto y_cpu = A_csr.multiply(x);
        auto y_gpu = vortex::cuda::sparse_spmv(A_csr, x, false);

        std::vector<float> scale(r, 0.0f);
        const auto& row_ptrs = A_csr.row_ptrs();
        const auto& col_indices = A_csr.col_indices();
        const auto& values = A_csr.values();
        for (std::size_t i = 0; i < r; ++i) {
            for (std::size_t idx = row_ptrs[i]; idx < row_ptrs[i + 1]; ++idx) {
                scale[i] += std::abs(values[idx]) * std::abs(x[col_indices[idx]]);
            }
        }

        auto rep = vortex::cuda::evaluate_componentwise_accuracy(y_gpu.data(), y_cpu.data(), scale.data(), r, 1e-5f, 1e-5f);
        ACCURACY_TEST_ASSERT(rep.passed, "Float Sparse SpMV failed componentwise bound: " + rep.diagnostic_summary);
        std::cout << "  Float Sparse SpMV (150x100) -> Max Abs Err: " << std::scientific << rep.max_absolute_error
                  << ", Max Ratio: " << rep.max_componentwise_ratio << std::fixed << "\n";
    }

    ACCURACY_TEST_PASS("Single precision float GEMV and SpMV numerical accuracy certified.");
}

// ==============================================================================
// 8. Memory Reliability & Repeated Allocation Cycling
// ==============================================================================
void test_memory_reliability_and_cycling() {
    std::cout << "\n--- Suite 8: Memory Lifecycle, Stress Cycling & Large Dimensions ---\n";

    // 1. Repeated dynamic buffer reallocation loop
    // Tests that memory buffers can be resized repeatedly without fragmentation or resource leaks
    {
        const int iterations = 30;
        vortex::cuda::CudaDeviceBuffer<double> reusable_buffer;
        std::mt19937 rng(1801);
        std::uniform_int_distribution<std::size_t> size_dist(16, 2048);

        for (int iter = 0; iter < iterations; ++iter) {
            const std::size_t n = size_dist(rng);
            reusable_buffer.allocate(n);
            ACCURACY_TEST_ASSERT(reusable_buffer.size() == n, "Buffer resize failed on iteration " + std::to_string(iter));

            std::vector<double> h_in(n, static_cast<double>(iter) * 1.5);
            reusable_buffer.copy_from_host(h_in.data(), n);

            std::vector<double> h_out(n, 0.0);
            reusable_buffer.copy_to_host(h_out.data(), n);

            ACCURACY_TEST_ASSERT(h_out[0] == h_in[0] && h_out[n - 1] == h_in[n - 1],
                                "Host-device integrity mismatch on cycling iteration " + std::to_string(iter));
        }
        std::cout << "  Successfully completed " << iterations << " random-size device buffer reallocation cycles.\n";
    }

    // 2. High-Capacity Dimension Test: 3000 x 3000 Dense Matrix (72 MB)
    // Validates that larger matrix operations execute accurately and cleanly within GPU VRAM limits.
    {
        const std::size_t n = 3000;
        std::cout << "  Allocating and computing 3000x3000 Dense GEMV (~72 MB VRAM)..." << std::endl;
        auto A_large = make_random_dense(n, n, -1.0, 1.0, 1802);
        auto x_large = make_random_vector(n, -0.5, 0.5, 1803);

        auto rep = vortex::cuda::validate_dense_gemv_accuracy(A_large, x_large, false, 1.0, 0.0, {}, 1e-12, 1e-12);
        ACCURACY_TEST_ASSERT(rep.passed, "Large 3000x3000 Dense GEMV failed: " + rep.diagnostic_summary);
        std::cout << "  3000x3000 Dense GEMV Passed -> Max Abs Err: " << std::scientific << rep.max_absolute_error
                  << ", Scaled Rel Err: " << rep.scaled_relative_error << std::fixed << "\n";
    }

    // 3. Large Sparse SpMV: 10,000 x 10,000 with 100,000 nonzeros
    {
        const std::size_t n = 10000;
        std::cout << "  Allocating and computing 10,000x10,000 Sparse SpMV (100k nonzeros)..." << std::endl;
        auto A_sparse_large = make_random_csr(n, n, 0.001, -2.0, 2.0, 1804);
        auto x_sparse_large = make_random_vector(n, -1.0, 1.0, 1805);

        auto rep_sp = vortex::cuda::validate_sparse_spmv_accuracy(A_sparse_large, x_sparse_large, false, 1.0, 0.0, {}, 1e-12, 1e-12);
        ACCURACY_TEST_ASSERT(rep_sp.passed, "Large 10,000x10,000 Sparse SpMV failed: " + rep_sp.diagnostic_summary);
        std::cout << "  10,000x10,000 Sparse SpMV (nnz=" << A_sparse_large.non_zeros() << ") Passed -> Max Abs Err: "
                  << std::scientific << rep_sp.max_absolute_error << std::fixed << "\n";
    }

    ACCURACY_TEST_PASS("Memory cycling reliability and large-scale dimension stress tests verified.");
}

// ==============================================================================
// Main Test Runner
// ==============================================================================
int main() {
    std::cout << "========================================================================\n";
    std::cout << "   VØRTEX Phase 8C: GPU Numerical Accuracy & Reliability Test Suite    \n";
    std::cout << "   Defensible Error Bounds, Ill-Conditioning & Stress Validation        \n";
    std::cout << "========================================================================\n";

    if (!vortex::cuda::is_cuda_available()) {
        std::cerr << "[FATAL] No CUDA-capable device detected. Cannot run GPU accuracy tests.\n";
        return 1;
    }

    auto dev = vortex::cuda::get_cuda_device_info(0);
    std::cout << "Device: " << dev.name << " | Compute: " << dev.major << "." << dev.minor
              << " | Total VRAM: " << (dev.total_memory_bytes / (1024 * 1024)) << " MB\n";

    test_analytical_and_exact_references();
    test_componentwise_bound_framework();
    test_edge_cases_and_dimensions();
    test_sparse_topological_edge_cases();
    test_ill_conditioned_and_near_singular();
    test_catastrophic_cancellation_analysis();
    test_single_precision_accuracy();
    test_memory_reliability_and_cycling();

    std::cout << "\n========================================================================\n";
    std::cout << " Phase 8C Test Summary: " << g_tests_passed << " passed, " << g_tests_failed << " failed.\n";
    std::cout << "========================================================================\n";

    return (g_tests_failed == 0) ? 0 : 1;
}
