#include "vortex/matrix.hpp"
#include "vortex/sparse_matrix.hpp"
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
// 1. CSR Matrix Construction, Indexing & Sparsity
// ==============================================================================
void test_csr_basics() {
    std::cout << "\n--- Running Suite: CSR Matrix Basics & Indexing ---\n";

    // 4x5 sparse matrix with 4 non-zeros
    std::vector<vortex::Triplet<double>> triplets = {
        {0, 1, 10.0},
        {1, 3, 20.0},
        {3, 0, 30.0},
        {3, 4, 40.0}
    };

    auto csr = vortex::CSRMatrix<double>::from_triplets(4, 5, triplets);

    record_result("CSR dimensions are 4x5", csr.rows() == 4 && csr.cols() == 5);
    record_result("CSR non_zeros == 4", csr.non_zeros() == 4);
    record_result("CSR sparsity == 80% (16 zeros out of 20)", approx_eq(csr.sparsity(), 0.80));

    // Test elements
    record_result("CSR (0, 1) == 10.0", approx_eq(csr(0, 1), 10.0));
    record_result("CSR (1, 3) == 20.0", approx_eq(csr(1, 3), 20.0));
    record_result("CSR (3, 0) == 30.0", approx_eq(csr(3, 0), 30.0));
    record_result("CSR (3, 4) == 40.0", approx_eq(csr(3, 4), 40.0));
    record_result("CSR zero entry (2, 2) == 0.0", approx_eq(csr(2, 2), 0.0));

    // Row pointers
    const auto& row_ptrs = csr.row_ptrs();
    record_result("row_ptrs has size rows + 1 == 5", row_ptrs.size() == 5);
    record_result("row_ptrs values correct",
                  row_ptrs[0] == 0 && row_ptrs[1] == 1 && row_ptrs[2] == 2 &&
                  row_ptrs[3] == 2 && row_ptrs[4] == 4);

    // Duplicate summing
    std::vector<vortex::Triplet<double>> duplicates = {
        {0, 0, 5.0},
        {0, 0, 7.0}
    };
    auto csr_dup = vortex::CSRMatrix<double>::from_triplets(2, 2, duplicates, true);
    record_result("Duplicate triplets summed to 12.0", approx_eq(csr_dup(0, 0), 12.0));
}

// ==============================================================================
// 2. CSC Matrix Construction, Indexing & Column Access
// ==============================================================================
void test_csc_basics() {
    std::cout << "\n--- Running Suite: CSC Matrix Basics & Column Access ---\n";

    std::vector<vortex::Triplet<double>> triplets = {
        {0, 1, 10.0},
        {1, 3, 20.0},
        {3, 0, 30.0},
        {3, 4, 40.0}
    };

    auto csc = vortex::CSCMatrix<double>::from_triplets(4, 5, triplets);

    record_result("CSC dimensions are 4x5", csc.rows() == 4 && csc.cols() == 5);
    record_result("CSC non_zeros == 4", csc.non_zeros() == 4);
    record_result("CSC sparsity == 80%", approx_eq(csc.sparsity(), 0.80));

    record_result("CSC (0, 1) == 10.0", approx_eq(csc(0, 1), 10.0));
    record_result("CSC (1, 3) == 20.0", approx_eq(csc(1, 3), 20.0));
    record_result("CSC (3, 0) == 30.0", approx_eq(csc(3, 0), 30.0));
    record_result("CSC (3, 4) == 40.0", approx_eq(csc(3, 4), 40.0));
    record_result("CSC zero entry (0, 0) == 0.0", approx_eq(csc(0, 0), 0.0));

    const auto& col_ptrs = csc.col_ptrs();
    record_result("col_ptrs has size cols + 1 == 6", col_ptrs.size() == 6);
    record_result("col_ptrs values correct",
                  col_ptrs[0] == 0 && col_ptrs[1] == 1 && col_ptrs[2] == 2 &&
                  col_ptrs[3] == 2 && col_ptrs[4] == 3 && col_ptrs[5] == 4);

    // Column extraction
    auto col1 = csc.get_column_dense(1);
    record_result("get_column_dense(1) has length 4", col1.size() == 4);
    record_result("col1[0] == 10.0 and others zero",
                  approx_eq(col1[0], 10.0) && approx_eq(col1[1], 0.0) &&
                  approx_eq(col1[2], 0.0) && approx_eq(col1[3], 0.0));
}

// ==============================================================================
// 3. Dense <-> Sparse Roundtrips
// ==============================================================================
void test_dense_sparse_conversions() {
    std::cout << "\n--- Running Suite: Dense <-> Sparse Roundtrips ---\n";

    vortex::Matrix<double> dense = {
        {1.0, 0.0, 0.0, 4.0},
        {0.0, 2.0, 0.0, 0.0},
        {0.0, 0.0, 3.0, 0.0}
    };

    // Dense -> CSR -> Dense
    auto csr = vortex::CSRMatrix<double>::from_dense(dense);
    record_result("from_dense CSR has 4 non-zeros", csr.non_zeros() == 4);
    auto dense_from_csr = csr.to_dense();
    record_result("CSR -> to_dense matches original dense matrix", dense.is_approx(dense_from_csr));

    // Dense -> CSC -> Dense
    auto csc = vortex::CSCMatrix<double>::from_dense(dense);
    record_result("from_dense CSC has 4 non-zeros", csc.non_zeros() == 4);
    auto dense_from_csc = csc.to_dense();
    record_result("CSC -> to_dense matches original dense matrix", dense.is_approx(dense_from_csc));
}

// ==============================================================================
// 4. CSR <-> CSC Interconversion and Transpose
// ==============================================================================
void test_csr_csc_interconversions() {
    std::cout << "\n--- Running Suite: CSR <-> CSC Interconversion & Transposition ---\n";

    vortex::Matrix<double> dense = {
        {5.0, 0.0, 2.0},
        {0.0, 0.0, 0.0},
        {1.0, 4.0, 0.0},
        {0.0, 3.0, 6.0}
    };

    auto csr = vortex::CSRMatrix<double>::from_dense(dense);
    auto csc_from_csr = csr.to_csc();

    record_result("CSR -> to_csc preserves non-zeros count", csc_from_csr.non_zeros() == csr.non_zeros());
    record_result("CSR -> to_csc preserves exact elements",
                  dense.is_approx(csc_from_csr.to_dense()));

    auto csr_from_csc = csc_from_csr.to_csr();
    record_result("CSC -> to_csr roundtrip matches original",
                  dense.is_approx(csr_from_csc.to_dense()));

    // Transpose
    auto csr_t = csr.transpose();
    auto dense_t = dense.transpose();
    record_result("CSR transpose matches dense transpose",
                  dense_t.is_approx(csr_t.to_dense()));

    auto csc_t = csc_from_csr.transpose();
    record_result("CSC transpose matches dense transpose",
                  dense_t.is_approx(csc_t.to_dense()));
}

// ==============================================================================
// 5. Sparse Matrix-Vector Multiplication (SpMV)
// ==============================================================================
void test_spmv_multiplication() {
    std::cout << "\n--- Running Suite: Sparse Matrix-Vector Multiplication (SpMV) ---\n";

    vortex::Matrix<double> dense = {
        {2.0, 0.0, 1.0, 0.0},
        {0.0, 3.0, 0.0, 4.0},
        {5.0, 0.0, 6.0, 0.0}
    };
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0};

    // Expected y = dense * x:
    // [2*1 + 1*3, 3*2 + 4*4, 5*1 + 6*3] = [5.0, 22.0, 23.0]
    std::vector<double> y_expected = {5.0, 22.0, 23.0};

    // CSR SpMV
    auto csr = vortex::CSRMatrix<double>::from_dense(dense);
    auto y_csr = csr.multiply(x);
    record_result("CSR SpMV y[0] == 5.0", approx_eq(y_csr[0], y_expected[0]));
    record_result("CSR SpMV y[1] == 22.0", approx_eq(y_csr[1], y_expected[1]));
    record_result("CSR SpMV y[2] == 23.0", approx_eq(y_csr[2], y_expected[2]));

    // CSC SpMV
    auto csc = vortex::CSCMatrix<double>::from_dense(dense);
    auto y_csc = csc.multiply(x);
    record_result("CSC SpMV matches expected output",
                  approx_eq(y_csc[0], y_expected[0]) &&
                  approx_eq(y_csc[1], y_expected[1]) &&
                  approx_eq(y_csc[2], y_expected[2]));

    // Transpose SpMV: y = A^T * v, where v = [2, 1, 3]^T
    // A^T has columns [2, 0, 5], [0, 3, 0], [1, 0, 6], [0, 4, 0]
    // y0 = 2*2 + 0*1 + 5*3 = 19
    // y1 = 0*2 + 3*1 + 0*3 = 3
    // y2 = 1*2 + 0*1 + 6*3 = 20
    // y3 = 0*2 + 4*1 + 0*3 = 4
    std::vector<double> v = {2.0, 1.0, 3.0};
    std::vector<double> yt_expected = {19.0, 3.0, 20.0, 4.0};

    auto yt_csr = csr.multiply_transpose(v);
    auto yt_csc = csc.multiply_transpose(v);

    record_result("CSR multiply_transpose matches expected",
                  approx_eq(yt_csr[0], yt_expected[0]) && approx_eq(yt_csr[1], yt_expected[1]) &&
                  approx_eq(yt_csr[2], yt_expected[2]) && approx_eq(yt_csr[3], yt_expected[3]));

    record_result("CSC multiply_transpose matches expected",
                  approx_eq(yt_csc[0], yt_expected[0]) && approx_eq(yt_csc[1], yt_expected[1]) &&
                  approx_eq(yt_csc[2], yt_expected[2]) && approx_eq(yt_csc[3], yt_expected[3]));

    // Column dot product in CSC (pricing step)
    record_result("CSC column_dot(0, v) == 19.0", approx_eq(csc.column_dot(0, v), 19.0));
    record_result("CSC column_dot(2, v) == 20.0", approx_eq(csc.column_dot(2, v), 20.0));
}

// ==============================================================================
// 6. Edge Cases & Memory Footprint
// ==============================================================================
void test_edge_cases_and_memory() {
    std::cout << "\n--- Running Suite: Sparse Edge Cases & Memory Footprint ---\n";

    // 1. Empty matrix
    vortex::CSRMatrix<double> empty_csr(0, 0);
    record_result("Empty CSR has 0 non-zeros", empty_csr.non_zeros() == 0 && empty_csr.empty());
    auto empty_csc = empty_csr.to_csc();
    record_result("Empty CSC has 0 non-zeros", empty_csc.non_zeros() == 0 && empty_csc.empty());

    // 2. Pure zero matrix
    vortex::CSRMatrix<double> zero_csr(10, 10);
    record_result("Zero CSR has 100% sparsity", approx_eq(zero_csr.sparsity(), 1.0));
    std::vector<double> x(10, 5.0);
    auto y_zero = zero_csr.multiply(x);
    bool all_zero = true;
    for (double val : y_zero) if (val != 0.0) all_zero = false;
    record_result("Zero CSR SpMV yields all zeros", all_zero);

    // 3. Memory savings calculation
    // 1000x1000 matrix with 1% density (99% sparsity)
    std::size_t N = 1000;
    std::size_t nnz = 10000;
    std::size_t dense_mem = N * N * sizeof(double); // 8 MB
    std::size_t csr_mem = nnz * sizeof(double) + nnz * sizeof(std::size_t) + (N + 1) * sizeof(std::size_t); // ~168 KB

    record_result("Sparse memory is significantly smaller than dense for 99% sparsity",
                  csr_mem < (dense_mem / 40));
}

int main() {
    std::cout << "=========================================================\n";
    std::cout << "      VORTEX Sparse Matrix Engine Unit Tests (Phase 5)   \n";
    std::cout << "=========================================================\n";

    test_csr_basics();
    test_csc_basics();
    test_dense_sparse_conversions();
    test_csr_csc_interconversions();
    test_spmv_multiplication();
    test_edge_cases_and_memory();

    std::cout << "\n=========================================================\n";
    std::cout << "  SUMMARY: " << passed_tests << " / " << total_tests << " tests passed.\n";
    std::cout << "=========================================================\n";

    return (passed_tests == total_tests) ? 0 : 1;
}
