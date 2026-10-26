#include "vortex/matrix.hpp"
#include "vortex/sparse_matrix.hpp"
#include "vortex/simplex.hpp"
#include "vortex/revised_simplex.hpp"
#include "vortex/sparse_revised_simplex.hpp"

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <random>

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

    double max_abs_error(const std::vector<double>& a, const std::vector<double>& b) {
        if (a.size() != b.size()) return std::numeric_limits<double>::infinity();
        double max_err = 0.0;
        for (std::size_t i = 0; i < a.size(); ++i) {
            max_err = std::max(max_err, std::abs(a[i] - b[i]));
        }
        return max_err;
    }

    double scaled_rel_error(const std::vector<double>& actual, const std::vector<double>& ref) {
        if (actual.size() != ref.size()) return std::numeric_limits<double>::infinity();
        double max_scaled = 0.0;
        for (std::size_t i = 0; i < actual.size(); ++i) {
            double denom = std::max(1.0, std::abs(ref[i]));
            double err = std::abs(actual[i] - ref[i]) / denom;
            max_scaled = std::max(max_scaled, err);
        }
        return max_scaled;
    }
}

// ==============================================================================
// 1. SpMV Parity: Dense vs CSR vs CSC with Reproducible Data & Error Metrics
// ==============================================================================
void test_spmv_numerical_parity() {
    std::cout << "\n--- 1. SpMV Numerical Parity (Dense vs CSR vs CSC) ---\n";

    const std::size_t rows = 40;
    const std::size_t cols = 30;
    const double density = 0.15; // 15% nonzeros

    std::mt19937_64 rng(1337);
    std::uniform_real_distribution<double> dist_val(-50.0, 50.0);
    std::uniform_real_distribution<double> dist_prob(0.0, 1.0);

    vortex::Matrix<double> dense(rows, cols, 0.0);
    std::vector<vortex::Triplet<double>> triplets;

    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            if (dist_prob(rng) < density) {
                double val = dist_val(rng);
                dense(i, j) = val;
                triplets.push_back({i, j, val});
            }
        }
    }

    auto csr = vortex::CSRMatrix<double>::from_triplets(rows, cols, triplets);
    auto csc = vortex::CSCMatrix<double>::from_triplets(rows, cols, triplets);

    std::vector<double> x(cols);
    for (std::size_t j = 0; j < cols; ++j) {
        x[j] = dist_val(rng);
    }

    auto y_dense = dense * x;
    auto y_csr = csr * x;
    auto y_csc = csc * x;

    double err_abs_csr = max_abs_error(y_csr, y_dense);
    double err_rel_csr = scaled_rel_error(y_csr, y_dense);
    double err_abs_csc = max_abs_error(y_csc, y_dense);
    double err_rel_csc = scaled_rel_error(y_csc, y_dense);

    std::cout << "  SpMV 40x30 Error Metrics:\n"
              << "    CSR max_abs_err: " << err_abs_csr << ", scaled_rel_err: " << err_rel_csr << "\n"
              << "    CSC max_abs_err: " << err_abs_csc << ", scaled_rel_err: " << err_rel_csc << "\n";

    record_result("CSR SpMV matches Dense reference within 1e-12", err_abs_csr < 1e-12 && err_rel_csr < 1e-12);
    record_result("CSC SpMV matches Dense reference within 1e-12", err_abs_csc < 1e-12 && err_rel_csc < 1e-12);
}

// ==============================================================================
// 2. Edge Cases: Zero Matrix, Identity, Rectangular, Empty Rows/Cols, Duplicates, Cancellation
// ==============================================================================
void test_matrix_edge_cases() {
    std::cout << "\n--- 2. Matrix Storage & Operation Edge Cases ---\n";

    // 2.1 Zero matrix
    {
        vortex::Matrix<double> zero_dense(10, 10, 0.0);
        auto zero_csr = vortex::CSRMatrix<double>::from_dense(zero_dense);
        auto zero_csc = vortex::CSCMatrix<double>::from_dense(zero_dense);
        std::vector<double> x(10, 5.0);

        auto y_csr = zero_csr * x;
        auto y_csc = zero_csc * x;

        bool all_zeros_csr = std::all_of(y_csr.begin(), y_csr.end(), [](double v){ return v == 0.0; });
        bool all_zeros_csc = std::all_of(y_csc.begin(), y_csc.end(), [](double v){ return v == 0.0; });

        record_result("Zero matrix SpMV produces all-zero output (CSR & CSC)",
                      all_zeros_csr && all_zeros_csc && zero_csr.non_zeros() == 0);
    }

    // 2.2 Identity matrix
    {
        const std::size_t N = 15;
        std::vector<vortex::Triplet<double>> id_triplets;
        for (std::size_t i = 0; i < N; ++i) id_triplets.push_back({i, i, 1.0});

        auto id_csr = vortex::CSRMatrix<double>::from_triplets(N, N, id_triplets);
        auto id_csc = vortex::CSCMatrix<double>::from_triplets(N, N, id_triplets);

        std::vector<double> x(N);
        for (std::size_t i = 0; i < N; ++i) x[i] = static_cast<double>(i + 1) * 2.5;

        auto y_csr = id_csr * x;
        auto y_csc = id_csc * x;

        record_result("Identity matrix SpMV returns identical vector (CSR)", max_abs_error(y_csr, x) == 0.0);
        record_result("Identity matrix SpMV returns identical vector (CSC)", max_abs_error(y_csc, x) == 0.0);
    }

    // 2.3 Rectangular shapes: Tall (50x5) and Wide (5x50)
    {
        // Tall 50x5
        std::vector<vortex::Triplet<double>> tall_triplets = {
            {0, 1, 2.0}, {10, 4, 3.5}, {25, 0, -1.0}, {49, 3, 7.2}
        };
        auto tall_csr = vortex::CSRMatrix<double>::from_triplets(50, 5, tall_triplets);
        auto tall_csc = vortex::CSCMatrix<double>::from_triplets(50, 5, tall_triplets);
        std::vector<double> x_tall = {1.0, 2.0, 3.0, 4.0, 5.0};
        auto y_tall_csr = tall_csr * x_tall;
        auto y_tall_csc = tall_csc * x_tall;

        record_result("Tall rectangular 50x5 SpMV dimension is 50", y_tall_csr.size() == 50);
        record_result("Tall rectangular 50x5 CSR and CSC outputs match", max_abs_error(y_tall_csr, y_tall_csc) == 0.0);

        // Wide 5x50
        std::vector<vortex::Triplet<double>> wide_triplets = {
            {1, 5, 4.0}, {3, 22, -8.0}, {4, 49, 1.5}
        };
        auto wide_csr = vortex::CSRMatrix<double>::from_triplets(5, 50, wide_triplets);
        auto wide_csc = vortex::CSCMatrix<double>::from_triplets(5, 50, wide_triplets);
        std::vector<double> x_wide(50, 1.0);
        auto y_wide_csr = wide_csr * x_wide;
        auto y_wide_csc = wide_csc * x_wide;

        record_result("Wide rectangular 5x50 SpMV dimension is 5", y_wide_csr.size() == 5);
        record_result("Wide rectangular 5x50 CSR and CSC outputs match", max_abs_error(y_wide_csr, y_wide_csc) == 0.0);
    }

    // 2.4 Empty rows and columns
    {
        // 4x4 matrix with row 1 completely empty and col 2 completely empty
        std::vector<vortex::Triplet<double>> triplets = {
            {0, 0, 5.0}, {0, 3, 2.0},
            {2, 1, 4.0},
            {3, 0, 1.0}, {3, 3, 9.0}
        };
        auto csr = vortex::CSRMatrix<double>::from_triplets(4, 4, triplets);
        auto csc = vortex::CSCMatrix<double>::from_triplets(4, 4, triplets);
        std::vector<double> x = {2.0, 3.0, 10.0, 4.0};

        auto y_csr = csr * x;
        auto y_csc = csc * x;

        record_result("SpMV with empty row: row 1 output is exactly 0.0", y_csr[1] == 0.0 && y_csc[1] == 0.0);
        record_result("CSR col_ptrs and row_ptrs handle empty lines correctly",
                      csr.row_ptrs()[1] == csr.row_ptrs()[2] && csc.col_ptrs()[2] == csc.col_ptrs()[3]);
    }

    // 2.5 Duplicate triplets & Cancellation
    {
        std::vector<vortex::Triplet<double>> triplets = {
            {1, 2, 4.0},
            {1, 2, 6.0},     // duplicate: sums to 10.0
            {3, 3, 5.0},
            {3, 3, -5.0}     // duplicate cancellation: sums to 0.0!
        };
        auto csr = vortex::CSRMatrix<double>::from_triplets(4, 4, triplets);
        auto csc = vortex::CSCMatrix<double>::from_triplets(4, 4, triplets);

        record_result("Duplicate triplets are accumulated at (1, 2) to 10.0", approx_eq(csr(1, 2), 10.0));
        record_result("Cancellation triplets at (3, 3) sum to 0.0", csr(3, 3) == 0.0);

        // Stored zero disclosure:
        record_result("Cancellation results in stored zero entry", csr.has_stored_zeros());
        record_result("count_stored_zeros accurately detects 1 stored zero", csr.count_stored_zeros() == 1);
        record_result("true_sparsity is higher than stored_sparsity", csr.true_sparsity() > csr.stored_sparsity());

        csr.drop_zeros();
        csc.drop_zeros();
        record_result("drop_zeros removes stored zero without altering matrix values",
                      csr.non_zeros() == 1 && csr(1, 2) == 10.0 && !csr.has_stored_zeros());
    }

    // 2.6 Widely differing magnitudes (dynamic range 1e-8 to 1e+6)
    {
        std::vector<vortex::Triplet<double>> triplets = {
            {0, 0, 1e-8},
            {0, 1, 1e6},
            {1, 0, 1e6},
            {1, 1, 1e-8}
        };
        auto csr = vortex::CSRMatrix<double>::from_triplets(2, 2, triplets);
        auto dense = csr.to_dense();
        std::vector<double> x = {1e4, 1e-4};

        auto y_dense = dense * x;
        auto y_csr = csr * x;

        double err_abs = max_abs_error(y_csr, y_dense);
        double err_rel = scaled_rel_error(y_csr, y_dense);

        record_result("SpMV with 1e-8 to 1e+6 dynamic range has max_abs_err < 1e-12", err_abs < 1e-12);
        record_result("SpMV with 1e-8 to 1e+6 dynamic range has scaled_rel_err < 1e-12", err_rel < 1e-12);
    }
}

// ==============================================================================
// 3. Finite Values, NaN and Inf Rejection
// ==============================================================================
void test_finite_values_and_nan_inf_rejection() {
    std::cout << "\n--- 3. Finite Values, NaN and Inf Rejection ---\n";

    // 3.1 Dense matrix is_finite()
    vortex::Matrix<double> dense_ok(2, 2, 1.0);
    record_result("Normal dense matrix is_finite() == true", dense_ok.is_finite());

    dense_ok(0, 1) = std::numeric_limits<double>::quiet_NaN();
    record_result("Dense matrix with NaN is_finite() == false", !dense_ok.is_finite());

    dense_ok(0, 1) = std::numeric_limits<double>::infinity();
    record_result("Dense matrix with Inf is_finite() == false", !dense_ok.is_finite());

    // 3.2 Sparse matrix is_finite()
    std::vector<vortex::Triplet<double>> triplets_nan = {{0, 0, std::numeric_limits<double>::quiet_NaN()}};
    auto csr_nan = vortex::CSRMatrix<double>::from_triplets(2, 2, triplets_nan);
    auto csc_nan = vortex::CSCMatrix<double>::from_triplets(2, 2, triplets_nan);
    record_result("CSR with NaN is_finite() == false", !csr_nan.is_finite());
    record_result("CSC with NaN is_finite() == false", !csc_nan.is_finite());

    // 3.3 Solver rejection of NaN / Inf in c, A, b
    vortex::SimplexSolver two_phase_solver;
    vortex::RevisedSimplexSolver revised_solver;
    vortex::SparseRevisedSimplexSolver sparse_revised_solver;

    std::vector<double> c_nan = {1.0, std::numeric_limits<double>::quiet_NaN()};
    vortex::Matrix<double> A_valid(2, 2, 1.0);
    std::vector<double> b_valid = {10.0, 10.0};
    std::vector<vortex::ConstraintType> types = {vortex::ConstraintType::LEQ, vortex::ConstraintType::LEQ};

    auto res1 = two_phase_solver.solve(c_nan, A_valid, b_valid, types);
    record_result("Two-Phase solver rejects NaN in objective c with INVALID_INPUT",
                  res1.status == vortex::SolverStatus::INVALID_INPUT);

    auto res2 = revised_solver.solve(c_nan, A_valid, b_valid, types);
    record_result("Revised Simplex rejects NaN in objective c with INVALID_INPUT",
                  res2.status == vortex::SolverStatus::INVALID_INPUT);

    auto res3 = sparse_revised_solver.solve(c_nan, A_valid, b_valid, types);
    record_result("Sparse Revised Simplex rejects NaN in objective c with INVALID_INPUT",
                  res3.status == vortex::SolverStatus::INVALID_INPUT);

    std::vector<double> c_valid = {1.0, 2.0};
    std::vector<double> b_inf = {10.0, std::numeric_limits<double>::infinity()};
    auto res4 = sparse_revised_solver.solve(c_valid, A_valid, b_inf, types);
    record_result("Sparse Revised Simplex rejects Inf in RHS b with INVALID_INPUT",
                  res4.status == vortex::SolverStatus::INVALID_INPUT);

    auto res5 = sparse_revised_solver.solve(c_valid, csc_nan, b_valid, types);
    record_result("Sparse Revised Simplex rejects NaN in sparse matrix A with INVALID_INPUT",
                  res5.status == vortex::SolverStatus::INVALID_INPUT);
}

// ==============================================================================
// 4. Memory Audit Calculations & Container Capacity vs Logical Size
// ==============================================================================
void test_memory_audit_calculations() {
    std::cout << "\n--- 4. Memory Audit Formulas & Capacity vs Logical Size ---\n";

    // 4.1 Dense logical storage: R * C * sizeof(double)
    vortex::Matrix<double> mat_500(500, 500, 0.0);
    std::size_t dense_logical = mat_500.logical_memory_bytes();
    std::size_t expected_dense = 500 * 500 * sizeof(double); // 2,000,000 bytes

    record_result("Dense 500x500 logical memory is exactly 2,000,000 bytes",
                  dense_logical == expected_dense && dense_logical == 2000000);
    double mib = static_cast<double>(dense_logical) / (1024.0 * 1024.0);
    double kib = static_cast<double>(dense_logical) / 1024.0;
    std::cout << "    Dense 500x500: " << dense_logical << " bytes = "
              << kib << " KiB = " << mib << " MiB (Previous '1 MB' was integer truncation of 1.907 MiB)\n";

    // 4.2 Sparse logical storage vs allocated capacity
    // 2000x2000 with 20,000 nonzeros (99.5% sparsity)
    const std::size_t N = 2000;
    const std::size_t nnz = 20000;
    std::vector<vortex::Triplet<double>> triplets;
    triplets.reserve(nnz);
    const std::size_t entries_per_row = nnz / N; // 10
    for (std::size_t r = 0; r < N; ++r) {
        for (std::size_t k = 0; k < entries_per_row; ++k) {
            triplets.push_back({r, (r + k * 197) % N, 1.0});
        }
    }

    auto csr = vortex::CSRMatrix<double>::from_triplets(N, N, triplets);
    auto csc = vortex::CSCMatrix<double>::from_triplets(N, N, triplets);

    std::size_t csr_logical = csr.logical_memory_bytes();
    std::size_t csr_capacity = csr.allocated_memory_bytes();
    std::size_t csc_logical = csc.logical_memory_bytes();

    // Theoretical CSR logical: nnz * 8 + nnz * 8 + (N + 1) * 8 = 20000*16 + 2001*8 = 320000 + 16008 = 336,008 bytes
    record_result("CSR 2000x2000 with 20k nnz logical storage matches theoretical 336,008 bytes",
                  csr_logical == (nnz * sizeof(double) + nnz * sizeof(std::size_t) + (N + 1) * sizeof(std::size_t)));
    record_result("CSR allocated capacity is >= logical size", csr_capacity >= csr_logical);

    std::size_t combined_logical = csr_logical + csc_logical;
    std::cout << "    CSR Logical: " << csr_logical << " bytes (" << (csr_logical / 1024.0) << " KiB)\n"
              << "    CSR Allocated Capacity: " << csr_capacity << " bytes (" << (csr_capacity / 1024.0) << " KiB)\n"
              << "    Combined CSR + CSC: " << combined_logical << " bytes (" << (combined_logical / 1024.0) << " KiB)\n";

    record_result("Combined CSR+CSC logical size is exactly 2 * nnz * 16 + 2 * (N+1) * 8",
                  combined_logical == 2 * csr_logical);
}

// ==============================================================================
// 5. LP Solver Validation Against Two-Phase & Independent Reference (SciPy/HiGHS)
// ==============================================================================
void test_lp_solver_validation_against_references() {
    std::cout << "\n--- 5. LP Solver Validation Against Independent References (SciPy/HiGHS) ---\n";

    vortex::SimplexSolver two_phase_solver;
    vortex::RevisedSimplexSolver revised_solver;
    vortex::SparseRevisedSimplexSolver sparse_solver;

    // --------------------------------------------------------------------------
    // Test 5.1: Standard Product-Mix LP (SciPy / HiGHS reference: Z* = 180.0)
    // max 3x1 + 2x2 s.t. 2x1 + x2 <= 100, x1 + x2 <= 80, x1 <= 40, x1, x2 >= 0
    // Optimal: x = [20, 60], Z = 180.0
    // --------------------------------------------------------------------------
    {
        std::vector<double> c = {3.0, 2.0};
        vortex::Matrix<double> A(3, 2, 0.0);
        A(0, 0) = 2.0; A(0, 1) = 1.0;
        A(1, 0) = 1.0; A(1, 1) = 1.0;
        A(2, 0) = 1.0; A(2, 1) = 0.0;
        std::vector<double> b = {100.0, 80.0, 40.0};
        std::vector<vortex::ConstraintType> types = {
            vortex::ConstraintType::LEQ, vortex::ConstraintType::LEQ, vortex::ConstraintType::LEQ
        };

        auto res_tp = two_phase_solver.solve(c, A, b, types, vortex::ObjectiveSense::MAXIMIZE);
        auto res_rev = revised_solver.solve(c, A, b, types, vortex::ObjectiveSense::MAXIMIZE);
        auto res_sp = sparse_solver.solve(c, A, b, types, vortex::ObjectiveSense::MAXIMIZE);

        record_result("Product-Mix LP: Two-Phase Z* == 180.0", approx_eq(res_tp.objective_value, 180.0));
        record_result("Product-Mix LP: Revised Simplex Z* == 180.0", approx_eq(res_rev.objective_value, 180.0));
        record_result("Product-Mix LP: Sparse Revised Simplex Z* == 180.0", approx_eq(res_sp.objective_value, 180.0));
        record_result("Product-Mix LP: Optimal primal point x* == [20, 60]",
                      approx_eq(res_sp.variable_values[0], 20.0) && approx_eq(res_sp.variable_values[1], 60.0));

        // Duality gap check
        double primal_obj = res_sp.objective_value;
        double dual_obj = 0.0;
        for (std::size_t i = 0; i < b.size(); ++i) dual_obj += b[i] * res_sp.dual_values[i];
        record_result("Product-Mix LP: Strong Duality Gap |c^T x - b^T pi| <= 1e-8",
                      approx_eq(primal_obj, dual_obj, 1e-8));
    }

    // --------------------------------------------------------------------------
    // Test 5.2: Mixed GEQ Minimization (SciPy / HiGHS reference: Z* = 10.0)
    // min 2x1 + 3x2 s.t. x1 + x2 >= 4, x1 + 2x2 >= 6, x1, x2 >= 0
    // Optimal: x = [2, 2], Z = 10.0
    // --------------------------------------------------------------------------
    {
        std::vector<double> c = {2.0, 3.0};
        vortex::Matrix<double> A(2, 2, 0.0);
        A(0, 0) = 1.0; A(0, 1) = 1.0;
        A(1, 0) = 1.0; A(1, 1) = 2.0;
        std::vector<double> b = {4.0, 6.0};
        std::vector<vortex::ConstraintType> types = {
            vortex::ConstraintType::GEQ, vortex::ConstraintType::GEQ
        };

        auto res_tp = two_phase_solver.solve(c, A, b, types, vortex::ObjectiveSense::MINIMIZE);
        auto res_rev = revised_solver.solve(c, A, b, types, vortex::ObjectiveSense::MINIMIZE);
        auto res_sp = sparse_solver.solve(c, A, b, types, vortex::ObjectiveSense::MINIMIZE);

        record_result("Mixed GEQ Min: Two-Phase Z* == 10.0", approx_eq(res_tp.objective_value, 10.0));
        record_result("Mixed GEQ Min: Revised Simplex Z* == 10.0", approx_eq(res_rev.objective_value, 10.0));
        record_result("Mixed GEQ Min: Sparse Revised Simplex Z* == 10.0", approx_eq(res_sp.objective_value, 10.0));
        record_result("Mixed GEQ Min: Optimal primal point x* == [2, 2]",
                      approx_eq(res_sp.variable_values[0], 2.0) && approx_eq(res_sp.variable_values[1], 2.0));
    }

    // --------------------------------------------------------------------------
    // Test 5.3: Equality Constraint Problem (SciPy reference: Z* = 36.0)
    // max 3x1 + 5x2 s.t. x1 <= 4, 2x2 <= 12, 3x1 + 2x2 = 18, x1, x2 >= 0
    // Optimal: x = [2, 6], Z = 36.0
    // --------------------------------------------------------------------------
    {
        std::vector<double> c = {3.0, 5.0};
        vortex::Matrix<double> A(3, 2, 0.0);
        A(0, 0) = 1.0; A(0, 1) = 0.0;
        A(1, 0) = 0.0; A(1, 1) = 2.0;
        A(2, 0) = 3.0; A(2, 1) = 2.0;
        std::vector<double> b = {4.0, 12.0, 18.0};
        std::vector<vortex::ConstraintType> types = {
            vortex::ConstraintType::LEQ, vortex::ConstraintType::LEQ, vortex::ConstraintType::EQ
        };

        auto res_tp = two_phase_solver.solve(c, A, b, types, vortex::ObjectiveSense::MAXIMIZE);
        auto res_sp = sparse_solver.solve(c, A, b, types, vortex::ObjectiveSense::MAXIMIZE);

        record_result("Equality Constraint LP: Two-Phase Z* == 36.0", approx_eq(res_tp.objective_value, 36.0));
        record_result("Equality Constraint LP: Sparse Revised Simplex Z* == 36.0", approx_eq(res_sp.objective_value, 36.0));
        record_result("Equality Constraint LP: Optimal point x* == [2, 6]",
                      approx_eq(res_sp.variable_values[0], 2.0) && approx_eq(res_sp.variable_values[1], 6.0));
    }

    // --------------------------------------------------------------------------
    // Test 5.4: Beale's Cycling / Degenerate LP (Bland's Rule Verification)
    // max 10x1 - 57x2 - 9x3 - 24x4
    // s.t. 0.5x1 - 5.5x2 - 2.5x3 + 9x4 <= 0
    //      0.5x1 - 1.5x2 - 0.5x3 + x4 <= 0
    //      x1 <= 1
    // Optimal: x = [1, 0, 1, 0], Z* = 1.0
    // --------------------------------------------------------------------------
    {
        std::vector<double> c = {10.0, -57.0, -9.0, -24.0};
        vortex::Matrix<double> A(3, 4, 0.0);
        A(0, 0) = 0.5; A(0, 1) = -5.5; A(0, 2) = -2.5; A(0, 3) = 9.0;
        A(1, 0) = 0.5; A(1, 1) = -1.5; A(1, 2) = -0.5; A(1, 3) = 1.0;
        A(2, 0) = 1.0; A(2, 1) =  0.0; A(2, 2) =  0.0; A(2, 3) = 0.0;
        std::vector<double> b = {0.0, 0.0, 1.0};
        std::vector<vortex::ConstraintType> types = {
            vortex::ConstraintType::LEQ, vortex::ConstraintType::LEQ, vortex::ConstraintType::LEQ
        };

        vortex::SimplexOptions tp_options;
        tp_options.pivot_rule = vortex::PivotRule::BLAND;
        vortex::SimplexSolver tp_bland_solver(tp_options);

        vortex::RevisedSimplexOptions sp_options;
        sp_options.pivot_rule = vortex::PivotRule::BLAND;
        vortex::SparseRevisedSimplexSolver sp_bland_solver(sp_options);

        auto res_tp = tp_bland_solver.solve(c, A, b, types, vortex::ObjectiveSense::MAXIMIZE);
        auto res_sp = sp_bland_solver.solve(c, A, b, types, vortex::ObjectiveSense::MAXIMIZE);

        std::cout << "    [Diagnostic] Beale Two-Phase status: " << static_cast<int>(res_tp.status)
                  << ", Z: " << res_tp.objective_value << ", iters: " << res_tp.iterations
                  << ", msg: " << res_tp.message << "\n";
        std::cout << "    [Diagnostic] Beale Sparse status: " << static_cast<int>(res_sp.status)
                  << ", Z: " << res_sp.objective_value << ", iters: " << res_sp.iterations
                  << ", msg: " << res_sp.message << "\n";

        record_result("Beale's Cycling LP: Two-Phase terminates at Z* == 1.0 via Bland's Rule",
                      res_tp.status == vortex::SolverStatus::OPTIMAL && approx_eq(res_tp.objective_value, 1.0));
        record_result("Beale's Cycling LP: Sparse Revised Simplex terminates at Z* == 1.0",
                      res_sp.status == vortex::SolverStatus::OPTIMAL && approx_eq(res_sp.objective_value, 1.0));
    }

    // --------------------------------------------------------------------------
    // Test 5.5: Infeasible LP Detection
    // max x1 + x2 s.t. x1 + x2 <= 2, x1 + x2 >= 5, x1, x2 >= 0
    // --------------------------------------------------------------------------
    {
        std::vector<double> c = {1.0, 1.0};
        vortex::Matrix<double> A(2, 2, 0.0);
        A(0, 0) = 1.0; A(0, 1) = 1.0;
        A(1, 0) = 1.0; A(1, 1) = 1.0;
        std::vector<double> b = {2.0, 5.0};
        std::vector<vortex::ConstraintType> types = {
            vortex::ConstraintType::LEQ, vortex::ConstraintType::GEQ
        };

        auto res_tp = two_phase_solver.solve(c, A, b, types, vortex::ObjectiveSense::MAXIMIZE);
        auto res_sp = sparse_solver.solve(c, A, b, types, vortex::ObjectiveSense::MAXIMIZE);

        record_result("Infeasible LP: Two-Phase status == INFEASIBLE", res_tp.status == vortex::SolverStatus::INFEASIBLE);
        record_result("Infeasible LP: Sparse Revised Simplex status == INFEASIBLE", res_sp.status == vortex::SolverStatus::INFEASIBLE);
    }

    // --------------------------------------------------------------------------
    // Test 5.6: Unbounded LP Detection
    // max 2x1 + x2 s.t. x1 - x2 <= 10, x1, x2 >= 0
    // --------------------------------------------------------------------------
    {
        std::vector<double> c = {2.0, 1.0};
        vortex::Matrix<double> A(1, 2, 0.0);
        A(0, 0) = 1.0; A(0, 1) = -1.0;
        std::vector<double> b = {10.0};
        std::vector<vortex::ConstraintType> types = {vortex::ConstraintType::LEQ};

        auto res_tp = two_phase_solver.solve(c, A, b, types, vortex::ObjectiveSense::MAXIMIZE);
        auto res_sp = sparse_solver.solve(c, A, b, types, vortex::ObjectiveSense::MAXIMIZE);

        record_result("Unbounded LP: Two-Phase status == UNBOUNDED", res_tp.status == vortex::SolverStatus::UNBOUNDED);
        record_result("Unbounded LP: Sparse Revised Simplex status == UNBOUNDED", res_sp.status == vortex::SolverStatus::UNBOUNDED);
    }

    // --------------------------------------------------------------------------
    // Test 5.7: Primal & Dual Residual Feasibility Checks
    // Verify ||Ax - b||_inf <= tol and reduced cost dual feasibility
    // --------------------------------------------------------------------------
    {
        // 5-var, 3-constraint sparse LP
        std::vector<double> c = {10.0, 14.0, 21.0, 8.0, 12.0};
        vortex::Matrix<double> A(3, 5, 0.0);
        A(0, 0) = 2.0; A(0, 1) = 3.0; A(0, 2) = 0.0; A(0, 3) = 1.0; A(0, 4) = 0.0;
        A(1, 0) = 0.0; A(1, 1) = 1.0; A(1, 2) = 4.0; A(1, 3) = 0.0; A(1, 4) = 2.0;
        A(2, 0) = 1.0; A(2, 1) = 0.0; A(2, 2) = 2.0; A(2, 3) = 3.0; A(2, 4) = 1.0;
        std::vector<double> b = {45.0, 60.0, 50.0};
        std::vector<vortex::ConstraintType> types = {
            vortex::ConstraintType::LEQ, vortex::ConstraintType::LEQ, vortex::ConstraintType::LEQ
        };

        auto res = sparse_solver.solve(c, A, b, types, vortex::ObjectiveSense::MAXIMIZE);
        record_result("Multi-var Sparse LP optimal status", res.status == vortex::SolverStatus::OPTIMAL);

        // Primal residual: Ax + s = b
        std::vector<double> Ax = A * res.variable_values;
        double max_primal_res = 0.0;
        for (std::size_t i = 0; i < 3; ++i) {
            double res_val = std::abs(Ax[i] + res.slack_values[i] - b[i]);
            max_primal_res = std::max(max_primal_res, res_val);
        }
        record_result("Primal residual ||Ax + s - b||_inf <= 1e-8", max_primal_res <= 1e-8);

        // Dual feasibility: reduced costs for non-basic <= 0 (for maximization)
        bool dual_feasible = true;
        for (double rc : res.reduced_costs) {
            if (rc > 1e-6) dual_feasible = false;
        }
        record_result("Dual feasibility: reduced costs satisfy optimality conditions", dual_feasible);
    }
}

// ==============================================================================
// Main Test Runner
// ==============================================================================
int main() {
    std::cout << "==============================================================================\n";
    std::cout << "     VORTEX Phase 5.1: Numerical Correctness and Memory Audit Suite           \n";
    std::cout << "==============================================================================\n";

    test_spmv_numerical_parity();
    test_matrix_edge_cases();
    test_finite_values_and_nan_inf_rejection();
    test_memory_audit_calculations();
    test_lp_solver_validation_against_references();

    std::cout << "\n==============================================================================\n";
    std::cout << "  SUMMARY: " << passed_tests << " / " << total_tests << " numerical audit tests passed.\n";
    std::cout << "==============================================================================\n";

    return (passed_tests == total_tests) ? 0 : 1;
}
