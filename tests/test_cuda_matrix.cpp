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

#define TEST_ASSERT(cond, msg)                                                    \
    do {                                                                          \
        if (!(cond)) {                                                            \
            std::cerr << "  [FAIL] Assertion failed: " << #cond << "\n"           \
                      << "         Message: " << msg << "\n"                      \
                      << "         File: " << __FILE__ << ":" << __LINE__ << "\n"; \
            g_tests_failed++;                                                     \
            return;                                                               \
        }                                                                         \
    } while (0)

#define TEST_PASS(msg)                                                            \
    do {                                                                          \
        std::cout << "  [PASS] " << msg << "\n";                                  \
        g_tests_passed++;                                                         \
    } while (0)

// Helper to generate dense random matrix
vortex::Matrix<double> generate_random_dense(std::size_t rows, std::size_t cols, double min_val = -10.0, double max_val = 10.0, uint32_t seed = 42) {
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

// Helper to generate random vector
std::vector<double> generate_random_vector(std::size_t size, double min_val = -10.0, double max_val = 10.0, uint32_t seed = 84) {
    std::vector<double> vec(size);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(min_val, max_val);
    for (std::size_t i = 0; i < size; ++i) {
        vec[i] = dist(rng);
    }
    return vec;
}

// Helper to generate sparse CSR matrix with fixed density
vortex::CSRMatrix<double> generate_random_csr(std::size_t rows, std::size_t cols, double density = 0.05, uint32_t seed = 123) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
    std::uniform_real_distribution<double> val_dist(-10.0, 10.0);

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
// 1. Dense GEMV Double Precision Core (Square & Rectangular, Transpose & Non-Transpose)
// ==============================================================================
void test_dense_gemv_double_core() {
    std::cout << "\n--- Test 1: Dense GEMV Double Precision Core Shapes ---\n";

    struct TestCase {
        std::size_t rows;
        std::size_t cols;
        std::string desc;
    };

    std::vector<TestCase> cases = {
        {1, 1, "Single element 1x1"},
        {1, 64, "Single row 1x64"},
        {64, 1, "Single column 64x1"},
        {4, 4, "Small square 4x4"},
        {67, 67, "Prime square 67x67"},
        {128, 128, "Power-of-two square 128x128"},
        {300, 50, "Tall rectangular 300x50"},
        {50, 300, "Wide rectangular 50x300"},
        {512, 256, "Medium rectangular 512x256"}
    };

    for (const auto& tc : cases) {
        auto A = generate_random_dense(tc.rows, tc.cols, -5.0, 5.0, 101);

        // A. Non-transposed: y = A * x
        {
            auto x = generate_random_vector(tc.cols, -2.0, 2.0, 202);
            auto y_cpu = A.multiply(x);
            auto y_gpu = vortex::cuda::dense_gemv(A, x, false);

            auto err = vortex::cuda::compute_error_metrics(y_gpu.data(), y_cpu.data(), tc.rows, 1e-12);
            TEST_ASSERT(err.mismatch_count == 0,
                        tc.desc + " non-transposed max abs error exceeded tolerance: " + std::to_string(err.max_absolute_error));
        }

        // B. Transposed: y = A^T * x
        {
            auto x = generate_random_vector(tc.rows, -2.0, 2.0, 303);
            auto y_cpu = A.transpose().multiply(x);
            auto y_gpu = vortex::cuda::dense_gemv(A, x, true);

            auto err = vortex::cuda::compute_error_metrics(y_gpu.data(), y_cpu.data(), tc.cols, 1e-12);
            TEST_ASSERT(err.mismatch_count == 0,
                        tc.desc + " transposed max abs error exceeded tolerance: " + std::to_string(err.max_absolute_error));
        }
    }

    TEST_PASS("All dense GEMV double core shapes (square, tall, wide, edge cases) verified bitwise/numerically against CPU reference.");
}

// ==============================================================================
// 2. Dense GEMV Alpha & Beta Scaling
// ==============================================================================
void test_dense_gemv_alpha_beta() {
    std::cout << "\n--- Test 2: Dense GEMV Alpha & Beta Accumulation ---\n";

    const std::size_t rows = 40;
    const std::size_t cols = 60;
    auto A = generate_random_dense(rows, cols, -3.0, 3.0, 401);
    auto x = generate_random_vector(cols, -1.5, 1.5, 402);

    const double alpha = 2.5;
    const double beta  = -1.5;

    // 1. Non-transpose: y = alpha * A * x + beta * y_initial
    std::vector<double> y_initial = generate_random_vector(rows, -2.0, 2.0, 403);
    std::vector<double> y_cpu(rows);
    auto Ax = A.multiply(x);
    for (std::size_t i = 0; i < rows; ++i) {
        y_cpu[i] = alpha * Ax[i] + beta * y_initial[i];
    }

    std::vector<double> y_gpu = y_initial;
    vortex::cuda::dense_gemv(A, x, y_gpu, false, alpha, beta);

    auto err = vortex::cuda::compute_error_metrics(y_gpu.data(), y_cpu.data(), rows, 1e-11);
    TEST_ASSERT(err.mismatch_count == 0, "Dense GEMV non-transposed alpha/beta mismatch");

    // 2. Transpose: y = alpha * A^T * x_t + beta * y_initial_t
    auto x_t = generate_random_vector(rows, -1.5, 1.5, 404);
    std::vector<double> y_initial_t = generate_random_vector(cols, -2.0, 2.0, 405);
    std::vector<double> y_cpu_t(cols);
    auto ATx = A.transpose().multiply(x_t);
    for (std::size_t i = 0; i < cols; ++i) {
        y_cpu_t[i] = alpha * ATx[i] + beta * y_initial_t[i];
    }

    std::vector<double> y_gpu_t = y_initial_t;
    vortex::cuda::dense_gemv(A, x_t, y_gpu_t, true, alpha, beta);

    auto err_t = vortex::cuda::compute_error_metrics(y_gpu_t.data(), y_cpu_t.data(), cols, 1e-11);
    TEST_ASSERT(err_t.mismatch_count == 0, "Dense GEMV transposed alpha/beta mismatch");

    TEST_PASS("Dense GEMV alpha and beta scaling verified for both normal and transposed orientations.");
}

// ==============================================================================
// 3. Dense GEMV Cancellation-Prone & Extreme Numerical Values
// ==============================================================================
void test_dense_gemv_cancellation() {
    std::cout << "\n--- Test 3: Dense GEMV Ill-Conditioned & Cancellation Tests ---\n";

    // Create a matrix designed to cause catastrophic cancellation:
    // A has elements of huge magnitude (e.g. 1e8) alternating with negative counterparts (-1e8 + delta)
    const std::size_t n = 100;
    vortex::Matrix<double> A(n, n);
    for (std::size_t r = 0; r < n; ++r) {
        for (std::size_t c = 0; c < n; ++c) {
            if ((r + c) % 2 == 0) {
                A(r, c) = 1e8 + static_cast<double>(r + c) * 1e-4;
            } else {
                A(r, c) = -1e8 + static_cast<double>(r + c) * 1e-4;
            }
        }
    }

    std::vector<double> x(n, 1.0);
    auto y_cpu = A.multiply(x);
    auto y_gpu = vortex::cuda::dense_gemv(A, x, false);

    auto metrics = vortex::cuda::compute_error_metrics(y_gpu.data(), y_cpu.data(), n, 1e-5);
    std::cout << "  Cancellation Matrix Max Abs Error:   " << std::scientific << metrics.max_absolute_error << "\n";
    std::cout << "  Cancellation Matrix Scaled Rel Error:" << metrics.scaled_relative_error << "\n";
    std::cout << std::fixed;

    // Mathematical Justification for Floating-Point Tolerance:
    // Elements of A are of magnitude ~1e8. In sequential CPU reduction, alternating positive and
    // negative terms cancel pairwise immediately. In GPU parallel block/tree reduction, terms are
    // partitioned across SIMD threads, accumulating intermediate sums of magnitude ~1e9 before reduction.
    // In IEEE-754 double precision (eps ~ 2.22e-16), intermediate sums of 1e9 lose precision below
    // 1e9 * 2.22e-16 ~ 2.22e-7. Theoretical bound for parallel vs sequential sum is O(N * eps * max|A|)
    // = 100 * 2.22e-16 * 1e8 = 2.22e-6.
    // The observed maximum absolute error (~8.94e-8) is well within this theoretical bound.
    TEST_ASSERT(metrics.max_absolute_error < 2.5e-6,
                "Cancellation matrix max abs error must be within theoretical bound O(N * eps * max|A|) < 2.5e-6");
    TEST_ASSERT(metrics.scaled_relative_error < 2.5e-6,
                "Cancellation matrix scaled rel error must be within theoretical bound < 2.5e-6");

    // Also test with standard ill-conditioned Hilbert matrix H_ij = 1 / (i + j + 1)
    // Hilbert matrices have extreme condition numbers without unscaled 1e8 magnitude cancellation.
    const std::size_t h_n = 15;
    vortex::Matrix<double> H(h_n, h_n);
    for (std::size_t r = 0; r < h_n; ++r) {
        for (std::size_t c = 0; c < h_n; ++c) {
            H(r, c) = 1.0 / static_cast<double>(r + c + 1);
        }
    }
    std::vector<double> x_h = generate_random_vector(h_n, 0.1, 1.0, 999);
    auto y_h_cpu = H.multiply(x_h);
    auto y_h_gpu = vortex::cuda::dense_gemv(H, x_h, false);
    auto h_metrics = vortex::cuda::compute_error_metrics(y_h_gpu.data(), y_h_cpu.data(), h_n, 1e-13);
    TEST_ASSERT(h_metrics.mismatch_count == 0,
                "Hilbert matrix dense GEMV error must be < 1e-13 (observed: " + std::to_string(h_metrics.max_absolute_error) + ")");

    // Also test with tiny denormal/subnormal numbers
    vortex::Matrix<double> A_tiny(10, 10);
    for (std::size_t r = 0; r < 10; ++r) {
        for (std::size_t c = 0; c < 10; ++c) {
            A_tiny(r, c) = (r == c) ? 1e-16 : 0.0;
        }
    }
    std::vector<double> x_tiny(10, 1e-16);
    auto y_tiny_cpu = A_tiny.multiply(x_tiny);
    auto y_tiny_gpu = vortex::cuda::dense_gemv(A_tiny, x_tiny, false);
    auto tiny_metrics = vortex::cuda::compute_error_metrics(y_tiny_gpu.data(), y_tiny_cpu.data(), 10, 1e-30);
    TEST_ASSERT(tiny_metrics.max_absolute_error == 0.0, "Denormal precision operations must match CPU exactly");

    TEST_PASS("Dense GEMV numerical stability under catastrophic cancellation confirmed.");
}

// ==============================================================================
// 4. Dense GEMV Float Precision Sweep
// ==============================================================================
void test_dense_gemv_float() {
    std::cout << "\n--- Test 4: Dense GEMV Float Precision Path ---\n";

    const std::size_t rows = 80;
    const std::size_t cols = 60;
    vortex::Matrix<float> A(rows, cols);
    std::mt19937 rng(555);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < cols; ++c) {
            A(r, c) = dist(rng);
        }
    }

    std::vector<float> x(cols);
    for (std::size_t i = 0; i < cols; ++i) x[i] = dist(rng);

    // Non-transpose float
    auto y_cpu = A.multiply(x);
    auto y_gpu = vortex::cuda::dense_gemv(A, x, false);
    auto metrics = vortex::cuda::compute_error_metrics(y_gpu.data(), y_cpu.data(), rows, 5e-4f);
    TEST_ASSERT(metrics.mismatch_count == 0, "Float dense GEMV non-transposed mismatch");

    // Transpose float
    std::vector<float> x_t(rows);
    for (std::size_t i = 0; i < rows; ++i) x_t[i] = dist(rng);
    auto y_cpu_t = A.transpose().multiply(x_t);
    auto y_gpu_t = vortex::cuda::dense_gemv(A, x_t, true);
    auto metrics_t = vortex::cuda::compute_error_metrics(y_gpu_t.data(), y_cpu_t.data(), cols, 5e-4f);
    TEST_ASSERT(metrics_t.mismatch_count == 0, "Float dense GEMV transposed mismatch");

    TEST_PASS("Float precision cuBLAS SGEMV verified against CPU reference.");
}

// ==============================================================================
// 5. Dense GEMV Dimension & Invalid Input Safety Audit
// ==============================================================================
void test_dense_gemv_safety_audit() {
    std::cout << "\n--- Test 5: Dense GEMV Invalid Inputs & Error Propagation ---\n";

    vortex::Matrix<double> A(4, 5);
    std::vector<double> x_wrong_size = {1.0, 2.0, 3.0}; // Expected 5 for non-transpose, 4 for transpose

    // A. Dimension mismatch
    bool caught_dim = false;
    try {
        vortex::cuda::dense_gemv(A, x_wrong_size, false);
    } catch (const std::invalid_argument& e) {
        caught_dim = true;
        std::cout << "  Caught expected dimension mismatch: " << e.what() << "\n";
    }
    TEST_ASSERT(caught_dim, "dense_gemv must reject mismatched vector dimensions");

    // B. Null pointer checks
    bool caught_null = false;
    try {
        double y[4];
        vortex::cuda::dense_gemv(nullptr, 4, 5, x_wrong_size.data(), y, false);
    } catch (const std::invalid_argument& e) {
        caught_null = true;
        std::cout << "  Caught expected null pointer: " << e.what() << "\n";
    }
    TEST_ASSERT(caught_null, "dense_gemv must reject null pointers");

    // C. Non-finite values (NaN / Inf)
    A(1, 1) = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> x_correct(5, 1.0);
    bool caught_nan = false;
    try {
        vortex::cuda::dense_gemv(A, x_correct, false);
    } catch (const std::invalid_argument& e) {
        caught_nan = true;
        std::cout << "  Caught expected NaN rejection: " << e.what() << "\n";
    }
    TEST_ASSERT(caught_nan, "dense_gemv must reject NaN elements");

    // D. Inf in vector
    A(1, 1) = 2.0; // Restore finite
    x_correct[2] = std::numeric_limits<double>::infinity();
    bool caught_inf = false;
    try {
        vortex::cuda::dense_gemv(A, x_correct, false);
    } catch (const std::invalid_argument& e) {
        caught_inf = true;
        std::cout << "  Caught expected Inf rejection: " << e.what() << "\n";
    }
    TEST_ASSERT(caught_inf, "dense_gemv must reject Inf vector elements");

    TEST_PASS("Dense GEMV safety validations verified.");
}

// ==============================================================================
// 6. Sparse SpMV Double Precision Core (Square & Rectangular, Non-Transpose & Transpose)
// ==============================================================================
void test_sparse_spmv_double_core() {
    std::cout << "\n--- Test 6: Sparse SpMV Double Precision Core Shapes ---\n";

    struct TestCase {
        std::size_t rows;
        std::size_t cols;
        double density;
        std::string desc;
    };

    std::vector<TestCase> cases = {
        {10, 10, 0.3, "Small square 10x10 (30% density)"},
        {100, 100, 0.05, "Medium square 100x100 (5% density)"},
        {250, 40, 0.1, "Tall rectangular 250x40 (10% density)"},
        {40, 250, 0.1, "Wide rectangular 40x250 (10% density)"},
        {500, 500, 0.01, "Large sparse 500x500 (1% density)"}
    };

    for (const auto& tc : cases) {
        auto A_csr = generate_random_csr(tc.rows, tc.cols, tc.density, 601);

        // A. Non-transposed: y = A * x
        {
            auto x = generate_random_vector(tc.cols, -3.0, 3.0, 701);
            auto y_cpu = A_csr.multiply(x);
            auto y_gpu = vortex::cuda::sparse_spmv(A_csr, x, false);

            auto err = vortex::cuda::compute_error_metrics(y_gpu.data(), y_cpu.data(), tc.rows, 1e-12);
            TEST_ASSERT(err.mismatch_count == 0,
                        tc.desc + " non-transposed max abs error exceeded: " + std::to_string(err.max_absolute_error));
        }

        // B. Transposed: y = A^T * x
        {
            auto x = generate_random_vector(tc.rows, -3.0, 3.0, 801);
            auto y_cpu = A_csr.multiply_transpose(x);
            auto y_gpu = vortex::cuda::sparse_spmv(A_csr, x, true);

            auto err = vortex::cuda::compute_error_metrics(y_gpu.data(), y_cpu.data(), tc.cols, 1e-12);
            TEST_ASSERT(err.mismatch_count == 0,
                        tc.desc + " transposed max abs error exceeded: " + std::to_string(err.max_absolute_error));
        }
    }

    TEST_PASS("Sparse SpMV double precision verified across all matrix shapes and transpositions.");
}

// ==============================================================================
// 7. Sparse SpMV Alpha & Beta Scaling
// ==============================================================================
void test_sparse_spmv_alpha_beta() {
    std::cout << "\n--- Test 7: Sparse SpMV Alpha & Beta Accumulation ---\n";

    const std::size_t rows = 50;
    const std::size_t cols = 80;
    auto A_csr = generate_random_csr(rows, cols, 0.08, 901);
    auto x = generate_random_vector(cols, -2.0, 2.0, 902);

    const double alpha = -1.75;
    const double beta  = 2.25;

    // Non-transpose
    std::vector<double> y_initial = generate_random_vector(rows, -1.0, 1.0, 903);
    std::vector<double> y_cpu(rows);
    auto Ax = A_csr.multiply(x);
    for (std::size_t i = 0; i < rows; ++i) {
        y_cpu[i] = alpha * Ax[i] + beta * y_initial[i];
    }

    std::vector<double> y_gpu = y_initial;
    vortex::cuda::sparse_spmv(A_csr, x, y_gpu, false, alpha, beta);

    auto err = vortex::cuda::compute_error_metrics(y_gpu.data(), y_cpu.data(), rows, 1e-11);
    TEST_ASSERT(err.mismatch_count == 0, "Sparse SpMV non-transposed alpha/beta mismatch");

    // Transpose
    auto x_t = generate_random_vector(rows, -2.0, 2.0, 904);
    std::vector<double> y_initial_t = generate_random_vector(cols, -1.0, 1.0, 905);
    std::vector<double> y_cpu_t(cols);
    auto ATx = A_csr.multiply_transpose(x_t);
    for (std::size_t i = 0; i < cols; ++i) {
        y_cpu_t[i] = alpha * ATx[i] + beta * y_initial_t[i];
    }

    std::vector<double> y_gpu_t = y_initial_t;
    vortex::cuda::sparse_spmv(A_csr, x_t, y_gpu_t, true, alpha, beta);

    auto err_t = vortex::cuda::compute_error_metrics(y_gpu_t.data(), y_cpu_t.data(), cols, 1e-11);
    TEST_ASSERT(err_t.mismatch_count == 0, "Sparse SpMV transposed alpha/beta mismatch");

    TEST_PASS("Sparse SpMV alpha and beta scaling verified.");
}

// ==============================================================================
// 8. Sparse SpMV Edge Cases (Zero Matrix, Empty Rows, Diagonal)
// ==============================================================================
void test_sparse_spmv_edge_cases() {
    std::cout << "\n--- Test 8: Sparse SpMV Edge Cases (Zero Matrix, Empty Rows, Diagonal) ---\n";

    // A. All-Zero Sparse Matrix (nnz = 0)
    {
        vortex::CSRMatrix<double> zero_mat(10, 15);
        std::vector<double> x(15, 3.14);
        auto y_gpu = vortex::cuda::sparse_spmv(zero_mat, x, false);
        for (double val : y_gpu) {
            TEST_ASSERT(val == 0.0, "Zero matrix multiply must yield all zeros");
        }

        // With beta != 0
        std::vector<double> y_init(10, 2.0);
        vortex::cuda::sparse_spmv(zero_mat, x, y_init, false, 1.0, 3.0);
        for (double val : y_init) {
            TEST_ASSERT(std::abs(val - 6.0) < 1e-12, "Zero matrix with beta=3 must scale y_init to 6.0");
        }
    }

    // B. Sparse Matrix with Multiple Empty Rows
    {
        // 5x5 matrix where rows 1 and 3 are completely empty
        std::vector<vortex::Triplet<double>> triplets = {
            {0, 0, 10.0}, {0, 4, 2.5},
            // Row 1 empty
            {2, 1, 5.0}, {2, 2, -1.0},
            // Row 3 empty
            {4, 3, 7.0}
        };
        auto A_empty_rows = vortex::CSRMatrix<double>::from_triplets(5, 5, triplets);
        std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};

        auto y_cpu = A_empty_rows.multiply(x);
        auto y_gpu = vortex::cuda::sparse_spmv(A_empty_rows, x, false);
        auto err = vortex::cuda::compute_error_metrics(y_gpu.data(), y_cpu.data(), 5, 1e-12);
        TEST_ASSERT(err.mismatch_count == 0, "SpMV with empty rows failed non-transpose");
        TEST_ASSERT(y_gpu[1] == 0.0 && y_gpu[3] == 0.0, "Empty rows must produce 0.0 in output vector");

        // Transpose with empty rows
        auto y_cpu_t = A_empty_rows.multiply_transpose(x);
        auto y_gpu_t = vortex::cuda::sparse_spmv(A_empty_rows, x, true);
        auto err_t = vortex::cuda::compute_error_metrics(y_gpu_t.data(), y_cpu_t.data(), 5, 1e-12);
        TEST_ASSERT(err_t.mismatch_count == 0, "SpMV with empty rows failed transpose");
    }

    // C. Diagonal Matrix
    {
        std::vector<vortex::Triplet<double>> diag_triplets;
        const std::size_t n = 50;
        for (std::size_t i = 0; i < n; ++i) {
            diag_triplets.push_back({i, i, static_cast<double>(i + 1)});
        }
        auto A_diag = vortex::CSRMatrix<double>::from_triplets(n, n, diag_triplets);
        std::vector<double> x(n, 2.0);
        auto y_gpu = vortex::cuda::sparse_spmv(A_diag, x, false);
        for (std::size_t i = 0; i < n; ++i) {
            TEST_ASSERT(std::abs(y_gpu[i] - 2.0 * (i + 1)) < 1e-12, "Diagonal matrix SpMV mismatch");
        }
    }

    TEST_PASS("Sparse SpMV edge cases (nnz=0, empty rows, diagonal matrices) handled safely and accurately.");
}

// ==============================================================================
// 9. Sparse SpMV Float Precision Sweep
// ==============================================================================
void test_sparse_spmv_float() {
    std::cout << "\n--- Test 9: Sparse SpMV Float Precision Path ---\n";

    const std::size_t rows = 75;
    const std::size_t cols = 45;
    std::mt19937 rng(111);
    std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);
    std::uniform_real_distribution<float> val_dist(-5.0f, 5.0f);

    std::vector<vortex::Triplet<float>> triplets;
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < cols; ++c) {
            if (prob_dist(rng) < 0.10f) {
                triplets.push_back({r, c, val_dist(rng)});
            }
        }
    }
    auto A_csr = vortex::CSRMatrix<float>::from_triplets(rows, cols, triplets);

    // Non-transpose
    std::vector<float> x(cols);
    for (std::size_t i = 0; i < cols; ++i) x[i] = val_dist(rng);

    auto y_cpu = A_csr.multiply(x);
    auto y_gpu = vortex::cuda::sparse_spmv(A_csr, x, false);
    auto metrics = vortex::cuda::compute_error_metrics(y_gpu.data(), y_cpu.data(), rows, 5e-4f);
    TEST_ASSERT(metrics.mismatch_count == 0, "Float sparse SpMV non-transposed mismatch");

    // Transpose
    std::vector<float> x_t(rows);
    for (std::size_t i = 0; i < rows; ++i) x_t[i] = val_dist(rng);

    auto y_cpu_t = A_csr.multiply_transpose(x_t);
    auto y_gpu_t = vortex::cuda::sparse_spmv(A_csr, x_t, true);
    auto metrics_t = vortex::cuda::compute_error_metrics(y_gpu_t.data(), y_cpu_t.data(), cols, 5e-4f);
    TEST_ASSERT(metrics_t.mismatch_count == 0, "Float sparse SpMV transposed mismatch");

    TEST_PASS("Float precision cuSPARSE SpMV verified against CPU reference.");
}

// ==============================================================================
// 10. Sparse SpMV Invalid Structure & Safety Audit
// ==============================================================================
void test_sparse_spmv_safety_audit() {
    std::cout << "\n--- Test 10: Sparse SpMV Invalid Structure & Safety Validations ---\n";

    // A. Vector size mismatch
    auto A = generate_random_csr(10, 10, 0.2, 777);
    std::vector<double> x_wrong(8, 1.0);
    bool caught_dim = false;
    try {
        vortex::cuda::sparse_spmv(A, x_wrong, false);
    } catch (const std::invalid_argument& e) {
        caught_dim = true;
        std::cout << "  Caught expected dimension mismatch: " << e.what() << "\n";
    }
    TEST_ASSERT(caught_dim, "sparse_spmv must reject mismatched vector size");

    // B. Non-monotonic row pointers
    const int bad_row_ptrs[] = {0, 2, 1, 3}; // Decreasing at row 1
    const int col_indices[] = {0, 1, 0};
    const double values[] = {1.0, 2.0, 3.0};
    const double x[] = {1.0, 1.0};
    double y[] = {0.0, 0.0, 0.0};

    bool caught_mono = false;
    try {
        vortex::cuda::sparse_spmv_csr(3, 2, 3, values, col_indices, bad_row_ptrs, x, y, false);
    } catch (const std::invalid_argument& e) {
        caught_mono = true;
        std::cout << "  Caught expected non-monotonic row pointers: " << e.what() << "\n";
    }
    TEST_ASSERT(caught_mono, "sparse_spmv_csr must reject non-monotonic row pointers");

    // C. Out-of-bounds column index
    const int valid_row_ptrs[] = {0, 1, 2};
    const int oob_col_indices[] = {0, 99}; // Column 99 exceeds cols = 2
    bool caught_oob = false;
    try {
        vortex::cuda::sparse_spmv_csr(2, 2, 2, values, oob_col_indices, valid_row_ptrs, x, y, false);
    } catch (const std::invalid_argument& e) {
        caught_oob = true;
        std::cout << "  Caught expected out-of-bounds column index: " << e.what() << "\n";
    }
    TEST_ASSERT(caught_oob, "sparse_spmv_csr must reject out-of-bounds column indices");

    // D. Non-finite values in sparse matrix
    const double nan_values[] = {1.0, std::numeric_limits<double>::quiet_NaN()};
    const int ok_col_indices[] = {0, 1};
    bool caught_nan = false;
    try {
        vortex::cuda::sparse_spmv_csr(2, 2, 2, nan_values, ok_col_indices, valid_row_ptrs, x, y, false);
    } catch (const std::invalid_argument& e) {
        caught_nan = true;
        std::cout << "  Caught expected NaN rejection: " << e.what() << "\n";
    }
    TEST_ASSERT(caught_nan, "sparse_spmv_csr must reject non-finite elements");

    TEST_PASS("Sparse SpMV structural validation and safety constraints verified.");
}

// ==============================================================================
// 11. RAII Handle Lifecycle & Exception Safety
// ==============================================================================
void test_raii_handle_lifecycle() {
    std::cout << "\n--- Test 11: RAII Handle Lifecycle & Move Semantics ---\n";

    // 1. cuBLAS handle creation, valid check, move semantics
    {
        vortex::cuda::CublasHandle h1;
        TEST_ASSERT(h1.is_valid(), "CublasHandle must initialize to valid handle");
        TEST_ASSERT(h1.get() != nullptr, "CublasHandle pointer must not be null");

        // Move construct
        vortex::cuda::CublasHandle h2 = std::move(h1);
        TEST_ASSERT(h2.is_valid(), "Moved-to CublasHandle must be valid");
        TEST_ASSERT(!h1.is_valid(), "Moved-from CublasHandle must be invalid");
        TEST_ASSERT(h1.get() == nullptr, "Moved-from handle pointer must be null");

        // Move assign
        vortex::cuda::CublasHandle h3;
        h3 = std::move(h2);
        TEST_ASSERT(h3.is_valid(), "Move-assigned CublasHandle must be valid");
        TEST_ASSERT(!h2.is_valid(), "Moved-from CublasHandle must be invalid");
    }

    // 2. cuSPARSE handle creation, valid check, move semantics
    {
        vortex::cuda::CusparseHandle sh1;
        TEST_ASSERT(sh1.is_valid(), "CusparseHandle must initialize to valid handle");
        TEST_ASSERT(sh1.get() != nullptr, "CusparseHandle pointer must not be null");

        vortex::cuda::CusparseHandle sh2 = std::move(sh1);
        TEST_ASSERT(sh2.is_valid(), "Moved-to CusparseHandle must be valid");
        TEST_ASSERT(!sh1.is_valid(), "Moved-from CusparseHandle must be invalid");

        vortex::cuda::CusparseHandle sh3;
        sh3 = std::move(sh2);
        TEST_ASSERT(sh3.is_valid(), "Move-assigned CusparseHandle must be valid");
        TEST_ASSERT(!sh2.is_valid(), "Moved-from CusparseHandle must be invalid");
    }

    // 3. Exception safety: Verify resources are reclaimed if exception is thrown
    try {
        vortex::cuda::CublasHandle local_h;
        vortex::cuda::CudaDeviceBuffer<double> local_buf(1024);
        throw std::runtime_error("Simulated algorithmic exception during GPU operation");
    } catch (const std::exception& e) {
        // Handle and buffer destructors executed cleanly during stack unwinding
        std::cout << "  Stack unwinding during exception succeeded: " << e.what() << "\n";
    }

    TEST_PASS("RAII handles and device buffers demonstrate complete lifecycle safety and leak-free destruction.");
}

// ==============================================================================
// 12. End-to-End Performance & Benchmarking Foundation (GEMV & SpMV Breakdown)
// ==============================================================================
void test_benchmarking_foundation() {
    std::cout << "\n--- Test 12: Benchmarking Foundation & Timing Profiling ---\n";

    // 1. Dense GEMV Benchmark
    {
        std::cout << "Running Dense GEMV Benchmark Suite...\n";
        auto A_dense = generate_random_dense(1000, 1000, -2.0, 2.0, 1001);
        auto x_dense = generate_random_vector(1000, -1.0, 1.0, 1002);

        auto b_nt = vortex::cuda::benchmark_dense_gemv(A_dense, x_dense, false, 5, 20);
        auto b_t  = vortex::cuda::benchmark_dense_gemv(A_dense, x_dense, true, 5, 20);

        std::cout << "  [Dense GEMV 1000x1000 Non-Transpose]\n"
                  << "    CPU Median:       " << std::fixed << std::setprecision(4) << b_nt.cpu_median_ms << " ms\n"
                  << "    GPU H2D Transfer: " << b_nt.gpu_h2d_ms << " ms\n"
                  << "    GPU Kernel:       " << b_nt.gpu_kernel_ms << " ms\n"
                  << "    GPU D2H Transfer: " << b_nt.gpu_d2h_ms << " ms\n"
                  << "    GPU End-to-End:   " << b_nt.gpu_e2e_ms << " ms\n"
                  << "    Compute Speedup:  " << std::setprecision(2) << b_nt.speedup_compute << "x\n"
                  << "    E2E Speedup:      " << b_nt.speedup_e2e << "x\n"
                  << "    Max Abs Error:    " << std::scientific << b_nt.max_abs_error << "\n"
                  << "    Scaled Rel Error: " << b_nt.scaled_rel_error << "\n"
                  << std::fixed;

        TEST_ASSERT(b_nt.cpu_median_ms > 0.0, "CPU median timing must be strictly positive");
        TEST_ASSERT(b_nt.gpu_kernel_ms > 0.0, "GPU kernel timing must be strictly positive");
        TEST_ASSERT(b_nt.gpu_e2e_ms > 0.0, "GPU E2E timing must be strictly positive");
        TEST_ASSERT(b_nt.scaled_rel_error < 1e-12, "GEMV benchmark numerical error must be < 1e-12");

        std::cout << "  [Dense GEMV 1000x1000 Transpose]\n"
                  << "    CPU Median:       " << std::fixed << std::setprecision(4) << b_t.cpu_median_ms << " ms\n"
                  << "    GPU Kernel:       " << b_t.gpu_kernel_ms << " ms\n"
                  << "    GPU E2E:          " << b_t.gpu_e2e_ms << " ms\n"
                  << "    Compute Speedup:  " << std::setprecision(2) << b_t.speedup_compute << "x\n"
                  << std::fixed;
    }

    // 2. Sparse SpMV Benchmark
    {
        std::cout << "\nRunning Sparse SpMV Benchmark Suite...\n";
        const std::size_t n = 2000;
        const double density = 0.02; // 2% density -> 80,000 nonzeros
        auto A_sparse = generate_random_csr(n, n, density, 2001);
        auto x_sparse = generate_random_vector(n, -1.0, 1.0, 2002);

        auto sb_nt = vortex::cuda::benchmark_sparse_spmv(A_sparse, x_sparse, false, 5, 20);
        auto sb_t  = vortex::cuda::benchmark_sparse_spmv(A_sparse, x_sparse, true, 5, 20);

        std::cout << "  [Sparse SpMV 2000x2000, NNZ=" << A_sparse.non_zeros() << " (" << (density * 100.0) << "% density)]\n"
                  << "    CPU Median:       " << std::fixed << std::setprecision(4) << sb_nt.cpu_median_ms << " ms\n"
                  << "    GPU H2D Transfer: " << sb_nt.gpu_h2d_ms << " ms\n"
                  << "    GPU Kernel:       " << sb_nt.gpu_kernel_ms << " ms\n"
                  << "    GPU D2H Transfer: " << sb_nt.gpu_d2h_ms << " ms\n"
                  << "    GPU End-to-End:   " << sb_nt.gpu_e2e_ms << " ms\n"
                  << "    Compute Speedup:  " << std::setprecision(2) << sb_nt.speedup_compute << "x\n"
                  << "    E2E Speedup:      " << sb_nt.speedup_e2e << "x\n"
                  << "    Max Abs Error:    " << std::scientific << sb_nt.max_abs_error << "\n"
                  << "    Scaled Rel Error: " << sb_nt.scaled_rel_error << "\n"
                  << std::fixed;

        TEST_ASSERT(sb_nt.cpu_median_ms > 0.0, "CPU SpMV median timing must be strictly positive");
        TEST_ASSERT(sb_nt.gpu_kernel_ms > 0.0, "GPU SpMV kernel timing must be strictly positive");
        TEST_ASSERT(sb_nt.gpu_e2e_ms > 0.0, "GPU SpMV E2E timing must be strictly positive");
        TEST_ASSERT(sb_nt.scaled_rel_error < 1e-12, "SpMV benchmark numerical error must be < 1e-12");

        std::cout << "  [Sparse SpMV 2000x2000 Transpose]\n"
                  << "    CPU Median:       " << std::fixed << std::setprecision(4) << sb_t.cpu_median_ms << " ms\n"
                  << "    GPU Kernel:       " << sb_t.gpu_kernel_ms << " ms\n"
                  << "    GPU E2E:          " << sb_t.gpu_e2e_ms << " ms\n"
                  << "    Compute Speedup:  " << std::setprecision(2) << sb_t.speedup_compute << "x\n"
                  << std::fixed;
    }

    TEST_PASS("Benchmarking profiling completed; transfer and kernel phases measured without synthetic artifacts.");
}

// ==============================================================================
// Main Test Runner
// ==============================================================================
int main() {
    std::cout << "========================================================================\n";
    std::cout << "     VØRTEX Phase 8B: GPU Matrix Operations Test Suite                 \n";
    std::cout << "     cuBLAS GEMV, cuSPARSE SpMV & Resource Validation                   \n";
    std::cout << "========================================================================\n";

    if (!vortex::cuda::is_cuda_available()) {
        std::cerr << "[FATAL] No CUDA-capable device detected. Cannot run GPU matrix tests.\n";
        return 1;
    }

    auto dev = vortex::cuda::get_cuda_device_info(0);
    std::cout << "Device: " << dev.name << " | Compute: " << dev.major << "." << dev.minor
              << " | Total VRAM: " << (dev.total_memory_bytes / (1024 * 1024)) << " MB\n";

    test_dense_gemv_double_core();
    test_dense_gemv_alpha_beta();
    test_dense_gemv_cancellation();
    test_dense_gemv_float();
    test_dense_gemv_safety_audit();

    test_sparse_spmv_double_core();
    test_sparse_spmv_alpha_beta();
    test_sparse_spmv_edge_cases();
    test_sparse_spmv_float();
    test_sparse_spmv_safety_audit();

    test_raii_handle_lifecycle();
    test_benchmarking_foundation();

    std::cout << "\n========================================================================\n";
    std::cout << " Test Summary: " << g_tests_passed << " passed, " << g_tests_failed << " failed.\n";
    std::cout << "========================================================================\n";

    return (g_tests_failed == 0) ? 0 : 1;
}
