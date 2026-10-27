#pragma once

#include "vortex/matrix.hpp"
#include "vortex/sparse_matrix.hpp"
#include "vortex/lu_decomposition.hpp"
#include "vortex/revised_simplex.hpp"
#include "simplex.hpp"
#include <vector>
#include <string>
#include <optional>
#include <iostream>

namespace vortex {

/**
 * @brief Extended result structure returned by the Sparse-Aware Revised Simplex solver.
 */
struct SparseRevisedSimplexResult : public RevisedSimplexResult {
    std::size_t non_zeros{0};             ///< Total non-zeros in standard-form constraint matrix
    double matrix_sparsity{0.0};          ///< Percentage of zero entries (0.0 to 1.0)
    std::size_t dense_memory_bytes{0};    ///< Memory required if stored densely (m * n * 8)
    std::size_t sparse_memory_bytes{0};   ///< Memory consumed by CSC storage
    double memory_savings_ratio{1.0};     ///< dense_memory / sparse_memory
    double total_pricing_time_ms{0.0};    ///< Cumulative time spent in reduced-cost evaluations
};

/**
 * @brief Sparse-Aware Revised Simplex Solver using Compressed Sparse Column (CSC) storage.
 *
 * Designed for large-scale and sparse linear programming problems where the constraint
 * matrix is typically 90% - 99.9% sparse.
 *
 * Core algorithmic advantages:
 *  1. Sparse Pricing (Reduced Costs):
 *     c_bar_j = c_j - dot(A[:, j], pi)
 *     Computed in O(nnz_j) operations instead of O(m), skipping all zero elements.
 *  2. Sparse Basis Extraction:
 *     Only non-zero entries of the active m basic columns are extracted into the Basis matrix.
 *  3. Compressed Memory Footprint:
 *     Stores only values, row indices, and column pointers, reducing memory from O(m * n) to O(nnz).
 *  4. Fast Sparse Residual Checking:
 *     Calculates ||A * x - b|| via SpMV in O(nnz) time.
 */
class SparseRevisedSimplexSolver {
public:
    explicit SparseRevisedSimplexSolver(RevisedSimplexOptions options = RevisedSimplexOptions{});

    // ==========================================
    // Standard Phase 2 Problem Solvers (<=, b >= 0, Maximize)
    // ==========================================

    [[nodiscard]] SparseRevisedSimplexResult solve(
        const std::vector<double>& c,
        const CSCMatrix<double>& A_csc,
        const std::vector<double>& b);

    [[nodiscard]] SparseRevisedSimplexResult solve(
        const std::vector<double>& c,
        const Matrix<double>& A_dense,
        const std::vector<double>& b);

    // ==========================================
    // General Two-Phase Problem Solvers (<=, >=, ==, b in R, Min/Max)
    // ==========================================

    [[nodiscard]] SparseRevisedSimplexResult solve(
        const std::vector<double>& c,
        const CSCMatrix<double>& A_csc,
        const std::vector<double>& b,
        const std::vector<ConstraintType>& constraint_types,
        ObjectiveSense sense = ObjectiveSense::MAXIMIZE);

    [[nodiscard]] SparseRevisedSimplexResult solve(
        const std::vector<double>& c,
        const Matrix<double>& A_dense,
        const std::vector<double>& b,
        const std::vector<ConstraintType>& constraint_types,
        ObjectiveSense sense = ObjectiveSense::MAXIMIZE);

    [[nodiscard]] SparseRevisedSimplexResult solve(const LinearProgram& lp);

    // ==========================================
    // Core Sparse Operations (Exposed for modularity & unit testing)
    // ==========================================

    /**
     * @brief Evaluates reduced costs using fast CSC column dot products:
     *   c_bar_j = c_std[j] - dot(A_csc[:, j], pi)
     * Takes O(nnz) total operations across all candidate columns.
     */
    [[nodiscard]] std::vector<double> compute_sparse_reduced_costs(
        const std::vector<double>& c_std,
        const CSCMatrix<double>& A_csc,
        const std::vector<double>& pi) const;

    /**
     * @brief Extracts m x m Basis matrix from CSC constraint matrix.
     */
    [[nodiscard]] Matrix<double> extract_basis_matrix(
        const CSCMatrix<double>& A_csc,
        const std::vector<std::size_t>& basis) const;

    /**
     * @brief Extracts step direction: B * d = A_csc[:, entering_col].
     */
    [[nodiscard]] std::vector<double> compute_step_direction(
        const LUDecomposition<double>& lu,
        const CSCMatrix<double>& A_csc,
        std::size_t entering_col) const;

    [[nodiscard]] const RevisedSimplexOptions& options() const noexcept { return options_; }
    void set_options(const RevisedSimplexOptions& options) noexcept { options_ = options; }

private:
    RevisedSimplexOptions options_;

    /**
     * @brief Core sparse revised simplex optimization loop.
     */
    [[nodiscard]] SparseRevisedSimplexResult run_sparse_simplex_loop(
        const std::vector<double>& c_std,
        const CSCMatrix<double>& A_std_csc,
        const std::vector<double>& b,
        std::vector<std::size_t>& basis,
        const std::vector<bool>& allowed_variables,
        std::size_t max_iters);

    [[nodiscard]] std::optional<std::string> validate_input_standard(
        const std::vector<double>& c,
        std::size_t a_rows,
        std::size_t a_cols,
        const std::vector<double>& b) const;

    [[nodiscard]] std::optional<std::string> validate_input_general(
        const std::vector<double>& c,
        std::size_t a_rows,
        std::size_t a_cols,
        const std::vector<double>& b,
        const std::vector<ConstraintType>& constraint_types) const;
};

} // namespace vortex
