#include "vortex/sparse_revised_simplex.hpp"
#include "vortex/cuda_backend.hpp"
#include <limits>
#include <cmath>
#include <chrono>
#include <memory>

namespace vortex {

SparseRevisedSimplexSolver::SparseRevisedSimplexSolver(RevisedSimplexOptions options)
    : options_(options) {}

std::optional<std::string> SparseRevisedSimplexSolver::validate_input_standard(
    const std::vector<double>& c,
    std::size_t a_rows,
    std::size_t a_cols,
    const std::vector<double>& b) const {

    if (c.empty()) {
        return "Invalid input: objective vector c cannot be empty.";
    }
    if (a_rows == 0 || a_cols == 0) {
        return "Invalid input: constraint matrix A cannot have 0 rows or 0 columns.";
    }
    if (b.empty()) {
        return "Invalid input: right-hand side vector b cannot be empty.";
    }
    if (a_cols != c.size()) {
        return "Dimension mismatch: A columns (" + std::to_string(a_cols) +
               ") must equal c size (" + std::to_string(c.size()) + ").";
    }
    if (a_rows != b.size()) {
        return "Dimension mismatch: A rows (" + std::to_string(a_rows) +
               ") must equal b size (" + std::to_string(b.size()) + ").";
    }

    for (double val : c) {
        if (!std::isfinite(val)) return "Invalid input: objective vector c contains non-finite values (NaN or Inf).";
    }
    for (double val : b) {
        if (!std::isfinite(val)) return "Invalid input: right-hand side vector b contains non-finite values (NaN or Inf).";
    }

    for (std::size_t i = 0; i < b.size(); ++i) {
        if (b[i] < -options_.tolerance) {
            return "Unsupported input: negative RHS detected (b[" + std::to_string(i) +
                   "] = " + std::to_string(b[i]) + " < 0). Standard form requires b >= 0.";
        }
    }

    return std::nullopt;
}

std::optional<std::string> SparseRevisedSimplexSolver::validate_input_general(
    const std::vector<double>& c,
    std::size_t a_rows,
    std::size_t a_cols,
    const std::vector<double>& b,
    const std::vector<ConstraintType>& constraint_types) const {

    if (c.empty()) {
        return "Invalid input: objective vector c cannot be empty.";
    }
    if (a_rows == 0 || a_cols == 0) {
        return "Invalid input: constraint matrix A cannot have 0 rows or 0 columns.";
    }
    if (b.empty()) {
        return "Invalid input: right-hand side vector b cannot be empty.";
    }
    if (a_cols != c.size()) {
        return "Dimension mismatch: A columns (" + std::to_string(a_cols) +
               ") must equal c size (" + std::to_string(c.size()) + ").";
    }
    if (a_rows != b.size()) {
        return "Dimension mismatch: A rows (" + std::to_string(a_rows) +
               ") must equal b size (" + std::to_string(b.size()) + ").";
    }
    if (constraint_types.size() != a_rows) {
        return "Dimension mismatch: constraint_types size (" + std::to_string(constraint_types.size()) +
               ") must equal A rows (" + std::to_string(a_rows) + ").";
    }

    for (double val : c) {
        if (!std::isfinite(val)) return "Invalid input: objective vector c contains non-finite values (NaN or Inf).";
    }
    for (double val : b) {
        if (!std::isfinite(val)) return "Invalid input: right-hand side vector b contains non-finite values (NaN or Inf).";
    }

    return std::nullopt;
}

std::vector<double> SparseRevisedSimplexSolver::compute_sparse_reduced_costs(
    const std::vector<double>& c_std,
    const CSCMatrix<double>& A_csc,
    const std::vector<double>& pi) const {

    const std::size_t num_vars = c_std.size();
    std::vector<double> reduced_costs(num_vars, 0.0);

    // Exploits CSC column dot: only touches non-zero rows in column j
    for (std::size_t j = 0; j < num_vars; ++j) {
        reduced_costs[j] = c_std[j] - A_csc.column_dot(j, pi);
    }

    return reduced_costs;
}

Matrix<double> SparseRevisedSimplexSolver::extract_basis_matrix(
    const CSCMatrix<double>& A_csc,
    const std::vector<std::size_t>& basis) const {

    const std::size_t m = A_csc.rows();
    Matrix<double> B(m, m, 0.0);

    const auto& values = A_csc.values();
    const auto& row_indices = A_csc.row_indices();
    const auto& col_ptrs = A_csc.col_ptrs();

    for (std::size_t col = 0; col < m; ++col) {
        const std::size_t var_idx = basis[col];
        const std::size_t start = col_ptrs[var_idx];
        const std::size_t end = col_ptrs[var_idx + 1];

        for (std::size_t idx = start; idx < end; ++idx) {
            B(row_indices[idx], col) = values[idx];
        }
    }

    return B;
}

std::vector<double> SparseRevisedSimplexSolver::compute_step_direction(
    const LUDecomposition<double>& lu,
    const CSCMatrix<double>& A_csc,
    std::size_t entering_col) const {

    std::vector<double> col_dense = A_csc.get_column_dense(entering_col);
    return lu.solve(col_dense);
}

SparseRevisedSimplexResult SparseRevisedSimplexSolver::run_sparse_simplex_loop(
    const std::vector<double>& c_std,
    const CSCMatrix<double>& A_std_csc,
    const std::vector<double>& b,
    std::vector<std::size_t>& basis,
    const std::vector<bool>& allowed_variables,
    std::size_t max_iters) {

    const std::size_t m = A_std_csc.rows();
    [[maybe_unused]] const std::size_t n = A_std_csc.cols();
    const std::size_t nnz = A_std_csc.non_zeros();
    SparseRevisedSimplexResult result;
    result.iterations = 0;
    result.basis_indices = basis;
    result.non_zeros = nnz;
    result.matrix_sparsity = A_std_csc.sparsity();
    result.device_diagnostics.requested_backend = options_.backend;

    [[maybe_unused]] bool use_gpu = false;
    if (options_.backend == SolverBackend::GPU) {
#if defined(VORTEX_ENABLE_CUDA)
        if (!cuda::is_cuda_available()) {
            result.status = SolverStatus::INVALID_INPUT;
            result.device_diagnostics.executed_backend = ExecutedBackend::CPU;
            result.device_diagnostics.fallback_reason = "GPU backend requested but no CUDA device is available.";
            result.message = result.device_diagnostics.fallback_reason;
            return result;
        }
        use_gpu = true;
#else
        result.status = SolverStatus::INVALID_INPUT;
        result.device_diagnostics.executed_backend = ExecutedBackend::CPU;
        result.device_diagnostics.fallback_reason = "GPU backend requested but VORTEX was compiled without CUDA.";
        result.message = result.device_diagnostics.fallback_reason;
        return result;
#endif
    } else if (options_.backend == SolverBackend::AUTO) {
#if defined(VORTEX_ENABLE_CUDA)
        if (cuda::is_cuda_available() && m >= 32 && nnz >= 2000) {
            use_gpu = true;
        } else {
            use_gpu = false;
            result.device_diagnostics.executed_backend = ExecutedBackend::CPU;
            if (!cuda::is_cuda_available()) {
                result.device_diagnostics.fallback_reason = "CUDA not available on system; AUTO selected CPU.";
            } else {
                result.device_diagnostics.fallback_reason = "Problem non-zero count (nnz=" + std::to_string(nnz) +
                    ", m=" + std::to_string(m) + ") below GPU sparse threshold (m >= 32, nnz >= 2000); AUTO selected CPU.";
            }
        }
#else
        use_gpu = false;
        result.device_diagnostics.executed_backend = ExecutedBackend::CPU;
        result.device_diagnostics.fallback_reason = "VORTEX compiled without CUDA; AUTO selected CPU.";
#endif
    } else {
        result.device_diagnostics.executed_backend = ExecutedBackend::CPU;
    }

#if defined(VORTEX_ENABLE_CUDA)
    std::unique_ptr<cuda::GpuSparsePricingContext> gpu_ctx;
    if (use_gpu) {
        try {
            auto A_csr = A_std_csc.to_csr();
            gpu_ctx = std::make_unique<cuda::GpuSparsePricingContext>(A_csr);
            result.device_diagnostics.executed_backend = ExecutedBackend::HYBRID;
        } catch (const std::exception& e) {
            if (options_.backend == SolverBackend::GPU) {
                result.status = SolverStatus::INVALID_INPUT;
                result.device_diagnostics.executed_backend = ExecutedBackend::CPU;
                result.device_diagnostics.fallback_reason = "GPU sparse initialization failed: " + std::string(e.what());
                result.message = result.device_diagnostics.fallback_reason;
                return result;
            } else {
                use_gpu = false;
                result.device_diagnostics.executed_backend = ExecutedBackend::CPU;
                result.device_diagnostics.fallback_reason = "GPU sparse initialization failed (" + std::string(e.what()) + "); AUTO falling back to CPU.";
            }
        }
    }
#endif

    auto finalize_diagnostics = [&]() {
#if defined(VORTEX_ENABLE_CUDA)
        if (use_gpu && gpu_ctx) {
            result.device_diagnostics.gpu_spmv_calls = gpu_ctx->spmv_calls();
            result.device_diagnostics.h2d_transfers = gpu_ctx->h2d_transfers();
            result.device_diagnostics.d2h_transfers = gpu_ctx->d2h_transfers();
        }
#endif
    };

    LUDecomposition<double> lu;

    while (result.iterations < max_iters) {
        // 1. Sparse Basis Extraction: extract active basic columns into m x m matrix
        Matrix<double> B_mat = extract_basis_matrix(A_std_csc, basis);
        bool non_singular = lu.compute(B_mat, options_.pivot_tolerance);

        if (!non_singular || lu.is_singular()) {
            result.status = SolverStatus::INVALID_INPUT;
            result.message = "Sparse Revised Simplex error: Basis matrix is singular or ill-conditioned.";
            finalize_diagnostics();
            return result;
        }

        // 2. Compute primal basic solution x_B = B^-1 * b
        std::vector<double> x_B = lu.solve(b);
        for (std::size_t i = 0; i < m; ++i) {
            if (x_B[i] < 0.0 && x_B[i] >= -options_.tolerance) {
                x_B[i] = 0.0;
            }
        }

        // 3. Compute dual multipliers: B^T * pi = c_B
        std::vector<double> c_B(m);
        for (std::size_t i = 0; i < m; ++i) {
            c_B[i] = c_std[basis[i]];
        }
        std::vector<double> pi = lu.solve_transpose(c_B);

        // 4. Numerical stability check
        double primal_res = lu.residual_norm(B_mat, x_B, b);
        double dual_res = lu.residual_norm_transpose(B_mat, pi, c_B);
        result.max_primal_residual = std::max(result.max_primal_residual, primal_res);
        result.max_dual_residual = std::max(result.max_dual_residual, dual_res);
        result.basis_condition = lu.condition_ratio();

        // 5. Sparse Pricing: compute reduced costs in O(nnz) time
        auto t_start = std::chrono::high_resolution_clock::now();
        std::vector<double> reduced_costs;
#if defined(VORTEX_ENABLE_CUDA)
        if (use_gpu && gpu_ctx) {
            gpu_ctx->compute_reduced_costs(c_std, pi, reduced_costs);
            result.device_diagnostics.gpu_iterations++;
        } else {
            reduced_costs = compute_sparse_reduced_costs(c_std, A_std_csc, pi);
        }
#else
        reduced_costs = compute_sparse_reduced_costs(c_std, A_std_csc, pi);
#endif
        auto t_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> pricing_duration = t_end - t_start;
        result.total_pricing_time_ms += pricing_duration.count();

        // 6. Select entering variable
        const std::size_t num_vars = reduced_costs.size();
        std::vector<bool> is_basic(num_vars, false);
        for (std::size_t idx : basis) {
            if (idx < num_vars) is_basic[idx] = true;
        }

        int entering_var = -1;
        if (options_.pivot_rule == PivotRule::BLAND) {
            for (std::size_t j = 0; j < num_vars; ++j) {
                if (is_basic[j]) continue;
                if (j < allowed_variables.size() && !allowed_variables[j]) continue;
                if (reduced_costs[j] > options_.tolerance) {
                    entering_var = static_cast<int>(j);
                    break;
                }
            }
        } else {
            double max_reduced_cost = options_.tolerance;
            for (std::size_t j = 0; j < num_vars; ++j) {
                if (is_basic[j]) continue;
                if (j < allowed_variables.size() && !allowed_variables[j]) continue;
                if (reduced_costs[j] > max_reduced_cost) {
                    max_reduced_cost = reduced_costs[j];
                    entering_var = static_cast<int>(j);
                }
            }
        }

        if (entering_var == -1) {
            // Optimality reached
            result.status = SolverStatus::OPTIMAL;
            result.basis_indices = basis;
            result.dual_values = pi;
            result.reduced_costs = reduced_costs;

            double z = 0.0;
            for (std::size_t i = 0; i < m; ++i) {
                z += c_B[i] * x_B[i];
            }
            result.objective_value = z;
            finalize_diagnostics();
            return result;
        }

        // 7. Compute step direction d = B^-1 * A_enter
        std::vector<double> d = compute_step_direction(lu, A_std_csc, static_cast<std::size_t>(entering_var));

        // 8. Minimum ratio test: select leaving variable
        int leaving_row = -1;
        double min_ratio = std::numeric_limits<double>::infinity();

        for (std::size_t i = 0; i < m; ++i) {
            if (d[i] > options_.tolerance) {
                const double rhs = (x_B[i] < 0.0) ? 0.0 : x_B[i];
                const double ratio = rhs / d[i];

                if (ratio < min_ratio - options_.tolerance) {
                    min_ratio = ratio;
                    leaving_row = static_cast<int>(i);
                } else if (std::abs(ratio - min_ratio) <= options_.tolerance) {
                    if (leaving_row != -1 && basis[i] < basis[static_cast<std::size_t>(leaving_row)]) {
                        leaving_row = static_cast<int>(i);
                    }
                }
            }
        }

        if (leaving_row == -1) {
            result.status = SolverStatus::UNBOUNDED;
            result.basis_indices = basis;
            result.message = "Objective is unbounded (no positive entries in pivot column direction).";
            finalize_diagnostics();
            return result;
        }

        // 9. Update basis
        basis[static_cast<std::size_t>(leaving_row)] = static_cast<std::size_t>(entering_var);
        result.iterations++;
    }

    result.status = SolverStatus::MAX_ITERATIONS_REACHED;
    result.basis_indices = basis;
    result.message = "Maximum iteration limit (" + std::to_string(max_iters) + ") reached.";
    finalize_diagnostics();
    return result;
}

SparseRevisedSimplexResult SparseRevisedSimplexSolver::solve(
    const std::vector<double>& c,
    const CSCMatrix<double>& A_csc,
    const std::vector<double>& b) {

    auto validation_err = validate_input_standard(c, A_csc.rows(), A_csc.cols(), b);
    if (validation_err) {
        SparseRevisedSimplexResult res;
        res.status = SolverStatus::INVALID_INPUT;
        res.message = *validation_err;
        return res;
    }
    if (!A_csc.is_finite()) {
        SparseRevisedSimplexResult res;
        res.status = SolverStatus::INVALID_INPUT;
        res.message = "Invalid input: constraint matrix A contains non-finite values (NaN or Inf).";
        return res;
    }

    const std::size_t m = A_csc.rows();
    const std::size_t n = c.size();
    const std::size_t total_cols = n + m;

    // Construct standard-form CSC matrix [A | I_m]
    // Columns 0..n-1 are from A_csc; columns n..n+m-1 are from Identity
    std::vector<double> values;
    std::vector<std::size_t> row_indices;
    std::vector<std::size_t> col_ptrs(total_cols + 1, 0);

    values.reserve(A_csc.non_zeros() + m);
    row_indices.reserve(A_csc.non_zeros() + m);

    // Copy original A_csc columns
    for (std::size_t col = 0; col < n; ++col) {
        col_ptrs[col] = values.size();
        for (std::size_t idx = A_csc.col_ptrs()[col]; idx < A_csc.col_ptrs()[col + 1]; ++idx) {
            values.push_back(A_csc.values()[idx]);
            row_indices.push_back(A_csc.row_indices()[idx]);
        }
    }

    // Append slack identity columns
    for (std::size_t i = 0; i < m; ++i) {
        col_ptrs[n + i] = values.size();
        values.push_back(1.0);
        row_indices.push_back(i);
    }
    col_ptrs[total_cols] = values.size();

    CSCMatrix<double> A_std_csc(m, total_cols, values, row_indices, col_ptrs);

    std::vector<double> c_std(total_cols, 0.0);
    for (std::size_t j = 0; j < n; ++j) {
        c_std[j] = c[j];
    }

    // Initial basis: slack variables
    std::vector<std::size_t> basis(m);
    for (std::size_t i = 0; i < m; ++i) {
        basis[i] = n + i;
    }

    std::vector<bool> allowed(total_cols, true);

    SparseRevisedSimplexResult res = run_sparse_simplex_loop(
        c_std, A_std_csc, b, basis, allowed, options_.max_iterations);

    res.dense_memory_bytes = m * total_cols * sizeof(double);
    res.sparse_memory_bytes = A_std_csc.memory_bytes();
    res.memory_savings_ratio = (res.sparse_memory_bytes > 0) ?
        (static_cast<double>(res.dense_memory_bytes) / static_cast<double>(res.sparse_memory_bytes)) : 1.0;

    if (res.status == SolverStatus::OPTIMAL) {
        res.message = "Optimal solution found successfully via Sparse Revised Simplex.";

        Matrix<double> B_mat = extract_basis_matrix(A_std_csc, basis);
        LUDecomposition<double> lu(B_mat, options_.pivot_tolerance);
        std::vector<double> x_B = lu.solve(b);

        res.variable_values.assign(n, 0.0);
        res.slack_values.assign(m, 0.0);

        for (std::size_t i = 0; i < m; ++i) {
            const std::size_t var_idx = basis[i];
            const double val = (x_B[i] < 0.0) ? 0.0 : x_B[i];
            if (var_idx < n) {
                res.variable_values[var_idx] = val;
            } else if (var_idx < total_cols) {
                res.slack_values[var_idx - n] = val;
            }
        }

        double z = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
            z += c[j] * res.variable_values[j];
        }
        res.objective_value = z;
    }

    return res;
}

SparseRevisedSimplexResult SparseRevisedSimplexSolver::solve(
    const std::vector<double>& c,
    const Matrix<double>& A_dense,
    const std::vector<double>& b) {
    return solve(c, CSCMatrix<double>::from_dense(A_dense, options_.tolerance), b);
}

SparseRevisedSimplexResult SparseRevisedSimplexSolver::solve(
    const std::vector<double>& c,
    const CSCMatrix<double>& A_csc,
    const std::vector<double>& b,
    const std::vector<ConstraintType>& constraint_types,
    ObjectiveSense sense) {

    auto validation_err = validate_input_general(c, A_csc.rows(), A_csc.cols(), b, constraint_types);
    if (validation_err) {
        SparseRevisedSimplexResult res;
        res.status = SolverStatus::INVALID_INPUT;
        res.message = *validation_err;
        return res;
    }
    if (!A_csc.is_finite()) {
        SparseRevisedSimplexResult res;
        res.status = SolverStatus::INVALID_INPUT;
        res.message = "Invalid input: constraint matrix A contains non-finite values (NaN or Inf).";
        return res;
    }

    const std::size_t m = A_csc.rows();
    const std::size_t n = c.size();

    std::vector<double> c_norm = c;
    if (sense == ObjectiveSense::MINIMIZE) {
        for (auto& val : c_norm) val = -val;
    }

    // Row sign adjustments for negative RHS
    std::vector<double> row_scale(m, 1.0);
    std::vector<double> b_norm = b;
    std::vector<ConstraintType> types_norm = constraint_types;

    for (std::size_t i = 0; i < m; ++i) {
        if (b_norm[i] < -options_.tolerance) {
            b_norm[i] = -b_norm[i];
            row_scale[i] = -1.0;
            if (types_norm[i] == ConstraintType::LEQ) {
                types_norm[i] = ConstraintType::GEQ;
            } else if (types_norm[i] == ConstraintType::GEQ) {
                types_norm[i] = ConstraintType::LEQ;
            }
        }
        if (std::abs(b_norm[i]) <= options_.tolerance) {
            b_norm[i] = 0.0;
        }
    }

    std::size_t num_slacks = 0;
    std::size_t num_surplus = 0;
    std::size_t num_artificial = 0;

    for (std::size_t i = 0; i < m; ++i) {
        if (types_norm[i] == ConstraintType::LEQ) {
            num_slacks++;
        } else if (types_norm[i] == ConstraintType::GEQ) {
            num_surplus++;
            num_artificial++;
        } else if (types_norm[i] == ConstraintType::EQ) {
            num_artificial++;
        }
    }

    const std::size_t slack_offset = n;
    const std::size_t surplus_offset = slack_offset + num_slacks;
    const std::size_t artificial_offset = surplus_offset + num_surplus;
    const std::size_t total_cols = artificial_offset + num_artificial;

    // Assemble standard-form CSC matrix
    std::vector<double> values;
    std::vector<std::size_t> row_indices;
    std::vector<std::size_t> col_ptrs(total_cols + 1, 0);

    values.reserve(A_csc.non_zeros() + num_slacks + num_surplus + num_artificial);
    row_indices.reserve(A_csc.non_zeros() + num_slacks + num_surplus + num_artificial);

    // 1. Original columns with row_scale
    for (std::size_t col = 0; col < n; ++col) {
        col_ptrs[col] = values.size();
        for (std::size_t idx = A_csc.col_ptrs()[col]; idx < A_csc.col_ptrs()[col + 1]; ++idx) {
            const std::size_t r = A_csc.row_indices()[idx];
            values.push_back(A_csc.values()[idx] * row_scale[r]);
            row_indices.push_back(r);
        }
    }

    std::vector<std::size_t> basis(m);
    std::size_t cur_slack = 0;
    std::size_t cur_surplus = 0;
    std::size_t cur_art = 0;

    for (std::size_t i = 0; i < m; ++i) {
        if (types_norm[i] == ConstraintType::LEQ) {
            const std::size_t col = slack_offset + cur_slack++;
            col_ptrs[col] = values.size();
            values.push_back(1.0);
            row_indices.push_back(i);
            basis[i] = col;
        } else if (types_norm[i] == ConstraintType::GEQ) {
            const std::size_t col_surplus = surplus_offset + cur_surplus++;
            const std::size_t col_art = artificial_offset + cur_art++;
            col_ptrs[col_surplus] = values.size();
            values.push_back(-1.0);
            row_indices.push_back(i);

            col_ptrs[col_art] = values.size();
            values.push_back(1.0);
            row_indices.push_back(i);
            basis[i] = col_art;
        } else if (types_norm[i] == ConstraintType::EQ) {
            const std::size_t col_art = artificial_offset + cur_art++;
            col_ptrs[col_art] = values.size();
            values.push_back(1.0);
            row_indices.push_back(i);
            basis[i] = col_art;
        }
    }
    col_ptrs[total_cols] = values.size();

    CSCMatrix<double> A_std_csc(m, total_cols, values, row_indices, col_ptrs);

    std::size_t phase1_iters = 0;
    std::size_t phase2_iters = 0;
    SparseRevisedSimplexResult p1_res;

    // Phase I if artificials exist
    if (num_artificial > 0) {
        std::vector<double> c_phase1(total_cols, 0.0);
        for (std::size_t j = artificial_offset; j < total_cols; ++j) {
            c_phase1[j] = -1.0;
        }
        std::vector<bool> allowed_p1(total_cols, true);

        p1_res = run_sparse_simplex_loop(
            c_phase1, A_std_csc, b_norm, basis, allowed_p1, options_.max_iterations);

        phase1_iters = p1_res.iterations;

        if (p1_res.status != SolverStatus::OPTIMAL) {
            SparseRevisedSimplexResult res;
            res.status = p1_res.status;
            res.phase1_iterations = phase1_iters;
            res.iterations = phase1_iters;
            res.message = "Phase I failed: " + p1_res.message;
            res.device_diagnostics = p1_res.device_diagnostics;
            return res;
        }

        Matrix<double> B_mat = extract_basis_matrix(A_std_csc, basis);
        LUDecomposition<double> lu(B_mat, options_.pivot_tolerance);
        std::vector<double> x_B = lu.solve(b_norm);

        double artificial_sum = 0.0;
        for (std::size_t i = 0; i < m; ++i) {
            if (basis[i] >= artificial_offset) {
                artificial_sum += (x_B[i] > 0.0 ? x_B[i] : 0.0);
            }
        }

        if (artificial_sum > options_.tolerance) {
            SparseRevisedSimplexResult res;
            res.status = SolverStatus::INFEASIBLE;
            res.phase1_iterations = phase1_iters;
            res.iterations = phase1_iters;
            res.message = "Problem is infeasible: artificial variable sum W = " +
                          std::to_string(artificial_sum) + " > 0 at Phase I optimality.";
            res.device_diagnostics = p1_res.device_diagnostics;
            return res;
        }

        // Drive out zero artificials
        for (std::size_t i = 0; i < m; ++i) {
            if (basis[i] >= artificial_offset) {
                for (std::size_t j = 0; j < artificial_offset; ++j) {
                    bool already_basic = false;
                    for (std::size_t b_idx : basis) {
                        if (b_idx == j) { already_basic = true; break; }
                    }
                    if (already_basic) continue;

                    std::vector<double> d = compute_step_direction(lu, A_std_csc, j);
                    if (std::abs(d[i]) > options_.tolerance) {
                        basis[i] = j;
                        B_mat = extract_basis_matrix(A_std_csc, basis);
                        lu.compute(B_mat, options_.pivot_tolerance);
                        break;
                    }
                }
            }
        }
    }

    // Phase II
    std::vector<double> c_phase2(total_cols, 0.0);
    for (std::size_t j = 0; j < n; ++j) {
        c_phase2[j] = c_norm[j];
    }

    std::vector<bool> allowed_p2(total_cols, true);
    for (std::size_t j = artificial_offset; j < total_cols; ++j) {
        allowed_p2[j] = false;
    }

    std::size_t remaining_iters = (options_.max_iterations > phase1_iters) ?
                                  (options_.max_iterations - phase1_iters) : 0;

    SparseRevisedSimplexResult p2_res = run_sparse_simplex_loop(
        c_phase2, A_std_csc, b_norm, basis, allowed_p2, remaining_iters);

    phase2_iters = p2_res.iterations;

    SparseRevisedSimplexResult res;
    res.status = p2_res.status;
    res.phase1_iterations = phase1_iters;
    res.phase2_iterations = phase2_iters;
    res.iterations = phase1_iters + phase2_iters;
    res.basis_indices = basis;
    res.dual_values = p2_res.dual_values;
    res.reduced_costs = p2_res.reduced_costs;
    res.max_primal_residual = std::max(p2_res.max_primal_residual, 0.0);
    res.max_dual_residual = std::max(p2_res.max_dual_residual, 0.0);
    res.basis_condition = p2_res.basis_condition;
    res.non_zeros = A_std_csc.non_zeros();
    res.matrix_sparsity = A_std_csc.sparsity();
    res.dense_memory_bytes = m * total_cols * sizeof(double);
    res.sparse_memory_bytes = A_std_csc.memory_bytes();
    res.memory_savings_ratio = (res.sparse_memory_bytes > 0) ?
        (static_cast<double>(res.dense_memory_bytes) / static_cast<double>(res.sparse_memory_bytes)) : 1.0;
    res.total_pricing_time_ms = p2_res.total_pricing_time_ms;

    res.device_diagnostics.requested_backend = options_.backend;
    if (p1_res.device_diagnostics.executed_backend == ExecutedBackend::HYBRID ||
        p2_res.device_diagnostics.executed_backend == ExecutedBackend::HYBRID) {
        res.device_diagnostics.executed_backend = ExecutedBackend::HYBRID;
    } else {
        res.device_diagnostics.executed_backend = ExecutedBackend::CPU;
    }
    res.device_diagnostics.gpu_iterations = p1_res.device_diagnostics.gpu_iterations + p2_res.device_diagnostics.gpu_iterations;
    res.device_diagnostics.gpu_spmv_calls = p1_res.device_diagnostics.gpu_spmv_calls + p2_res.device_diagnostics.gpu_spmv_calls;
    res.device_diagnostics.h2d_transfers = p1_res.device_diagnostics.h2d_transfers + p2_res.device_diagnostics.h2d_transfers;
    res.device_diagnostics.d2h_transfers = p1_res.device_diagnostics.d2h_transfers + p2_res.device_diagnostics.d2h_transfers;
    res.device_diagnostics.fallback_reason = p2_res.device_diagnostics.fallback_reason.empty() ? p1_res.device_diagnostics.fallback_reason : p2_res.device_diagnostics.fallback_reason;

    if (res.status == SolverStatus::OPTIMAL) {
        res.message = "Optimal solution found successfully via Two-Phase Sparse Revised Simplex.";

        Matrix<double> B_mat = extract_basis_matrix(A_std_csc, basis);
        LUDecomposition<double> lu(B_mat, options_.pivot_tolerance);
        std::vector<double> x_B = lu.solve(b_norm);

        res.variable_values.assign(n, 0.0);
        res.slack_values.assign(num_slacks, 0.0);
        res.surplus_values.assign(num_surplus, 0.0);

        for (std::size_t i = 0; i < m; ++i) {
            const std::size_t var_idx = basis[i];
            const double val = (x_B[i] < 0.0) ? 0.0 : x_B[i];

            if (var_idx < n) {
                res.variable_values[var_idx] = val;
            } else if (var_idx >= slack_offset && var_idx < surplus_offset) {
                res.slack_values[var_idx - slack_offset] = val;
            } else if (var_idx >= surplus_offset && var_idx < artificial_offset) {
                res.surplus_values[var_idx - surplus_offset] = val;
            }
        }

        double z = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
            z += c[j] * res.variable_values[j];
        }
        res.objective_value = z;

        if (sense == ObjectiveSense::MINIMIZE) {
            for (auto& pi_val : res.dual_values) {
                pi_val = -pi_val;
            }
        }
    } else {
        res.message = p2_res.message;
    }

    return res;
}

SparseRevisedSimplexResult SparseRevisedSimplexSolver::solve(
    const std::vector<double>& c,
    const Matrix<double>& A_dense,
    const std::vector<double>& b,
    const std::vector<ConstraintType>& constraint_types,
    ObjectiveSense sense) {
    return solve(c, CSCMatrix<double>::from_dense(A_dense, options_.tolerance), b, constraint_types, sense);
}

SparseRevisedSimplexResult SparseRevisedSimplexSolver::solve(const LinearProgram& lp) {
    return solve(lp.c, lp.A, lp.b, lp.constraint_types, lp.sense);
}

} // namespace vortex
