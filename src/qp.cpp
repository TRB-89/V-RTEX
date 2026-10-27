#include "vortex/qp.hpp"
#include "vortex/lu_decomposition.hpp"
#include "vortex/cuda_backend.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <memory>

namespace vortex {

ConvexQPSolver::ConvexQPSolver(QPOptions options)
    : options_(options) {}

QPResult ConvexQPSolver::solve(const QuadraticProgram& qp) {
    bool is_strictly_pd = false;
    bool is_zero_q = false;

    // Check GPU backend support if explicitly requested
    if (options_.backend == SolverBackend::GPU) {
#if defined(VORTEX_ENABLE_CUDA)
        if (!cuda::is_cuda_available()) {
            QPResult res;
            res.status = QPSolverStatus::UNSUPPORTED;
            res.device_diagnostics.requested_backend = SolverBackend::GPU;
            res.device_diagnostics.executed_backend = ExecutedBackend::CPU;
            res.device_diagnostics.fallback_reason = "GPU backend requested but no CUDA device is available.";
            res.message = res.device_diagnostics.fallback_reason;
            return res;
        }
#else
        QPResult res;
        res.status = QPSolverStatus::UNSUPPORTED;
        res.device_diagnostics.requested_backend = SolverBackend::GPU;
        res.device_diagnostics.executed_backend = ExecutedBackend::CPU;
        res.device_diagnostics.fallback_reason = "GPU backend requested but VORTEX was compiled without CUDA.";
        res.message = res.device_diagnostics.fallback_reason;
        return res;
#endif
    }

    // 1. Input validation
    auto validation_error = validate_input(qp, is_strictly_pd, is_zero_q);
    if (validation_error.has_value()) {
        QPResult res;
        res.status = QPSolverStatus::INVALID_INPUT;
        res.device_diagnostics.requested_backend = options_.backend;
        res.device_diagnostics.executed_backend = ExecutedBackend::CPU;
        res.message = validation_error.value();
        return res;
    }

    const std::size_t m = qp.A.rows();

    // 2. Unconstrained fast-path (m = 0)
    if (m == 0) {
        return solve_unconstrained(qp, is_strictly_pd, is_zero_q);
    }

    // 3. Inequality-constrained Primal-Dual Interior-Point Method (m > 0)
    return solve_interior_point(qp);
}

QPResult ConvexQPSolver::solve(
    const Matrix<double>& Q,
    const std::vector<double>& c,
    const Matrix<double>& A,
    const std::vector<double>& b) {
    QuadraticProgram qp;
    qp.Q = Q;
    qp.c = c;
    qp.A = A;
    qp.b = b;
    return solve(qp);
}

std::optional<std::string> ConvexQPSolver::validate_input(
    const QuadraticProgram& qp,
    bool& out_is_strictly_pd,
    bool& out_is_zero_q) const {
    const std::size_t n = qp.Q.rows();
    out_is_strictly_pd = false;
    out_is_zero_q = false;

    // Dimensions
    if (n == 0) {
        return "Dimension error: Hessian matrix Q has dimension 0x0.";
    }
    if (qp.Q.cols() != n) {
        return "Dimension error: Hessian matrix Q must be square (rows != cols).";
    }
    if (qp.c.size() != n) {
        return "Dimension error: Linear objective vector c size (" +
               std::to_string(qp.c.size()) + ") does not match Q dimension (" +
               std::to_string(n) + ").";
    }

    const std::size_t m = qp.A.rows();
    if (m > 0) {
        if (qp.A.cols() != n) {
            return "Dimension error: Constraint matrix A columns (" +
                   std::to_string(qp.A.cols()) + ") do not match variable count (" +
                   std::to_string(n) + ").";
        }
        if (qp.b.size() != m) {
            return "Dimension error: Right-hand side vector b size (" +
                   std::to_string(qp.b.size()) + ") does not match constraint count (" +
                   std::to_string(m) + ").";
        }
    } else if (!qp.b.empty()) {
        return "Dimension error: Constraint matrix A has 0 rows but vector b is non-empty.";
    }

    // Check finiteness
    for (std::size_t i = 0; i < n; ++i) {
        if (!std::isfinite(qp.c[i])) {
            return "Non-finite coefficient in linear objective vector c at index " + std::to_string(i);
        }
        for (std::size_t j = 0; j < n; ++j) {
            if (!std::isfinite(qp.Q(i, j))) {
                return "Non-finite coefficient in Hessian matrix Q at (" +
                       std::to_string(i) + ", " + std::to_string(j) + ").";
            }
        }
    }

    for (std::size_t i = 0; i < m; ++i) {
        if (!std::isfinite(qp.b[i])) {
            return "Non-finite coefficient in right-hand side b at index " + std::to_string(i);
        }
        for (std::size_t j = 0; j < n; ++j) {
            if (!std::isfinite(qp.A(i, j))) {
                return "Non-finite coefficient in constraint matrix A at (" +
                       std::to_string(i) + ", " + std::to_string(j) + ").";
            }
        }
    }

    // Check symmetry of Q: |Q_ij - Q_ji| <= tol * max(1, |Q_ij|, |Q_ji|)
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            double diff = std::abs(qp.Q(i, j) - qp.Q(j, i));
            double scale = std::max({1.0, std::abs(qp.Q(i, j)), std::abs(qp.Q(j, i))});
            if (diff > options_.symmetry_tolerance * scale) {
                return "Matrix Q is not symmetric: |Q(" + std::to_string(i) + "," + std::to_string(j) +
                       ") - Q(" + std::to_string(j) + "," + std::to_string(i) + ")| = " +
                       std::to_string(diff) + " > tolerance.";
            }
        }
    }

    // Check if Q == 0
    double q_max_abs = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            q_max_abs = std::max(q_max_abs, std::abs(qp.Q(i, j)));
        }
    }
    if (q_max_abs <= 1e-15) {
        out_is_zero_q = true;
        out_is_strictly_pd = false;
        return std::nullopt; // Q = 0 is convex (linear objective)
    }

    // Eigenvalue check for positive semidefiniteness (PSD)
    std::vector<double> eigenvalues = compute_eigenvalues_symmetric(qp.Q);
    double min_eig = *std::min_element(eigenvalues.begin(), eigenvalues.end());

    if (min_eig < -options_.psd_tolerance) {
        std::ostringstream oss;
        oss << "Matrix Q is indefinite (minimum eigenvalue = " << std::setprecision(6) << min_eig
            << " < -" << options_.psd_tolerance << "); non-convex QP is not supported.";
        return oss.str();
    }

    if (min_eig > options_.psd_tolerance) {
        out_is_strictly_pd = true;
    } else {
        out_is_strictly_pd = false; // singular PSD
    }

    return std::nullopt;
}

std::vector<double> ConvexQPSolver::compute_eigenvalues_symmetric(const Matrix<double>& M) const {
    const std::size_t n = M.rows();
    if (n == 0) return {};
    if (n == 1) return {M(0, 0)};

    // Jacobi eigenvalue algorithm on symmetric matrix
    Matrix<double> A_mat = M;
    // Symmetrize explicitly to remove tiny floating point imbalances
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            double avg = 0.5 * (A_mat(i, j) + A_mat(j, i));
            A_mat(i, j) = avg;
            A_mat(j, i) = avg;
        }
    }

    const std::size_t max_sweeps = 100;
    for (std::size_t sweep = 0; sweep < max_sweeps; ++sweep) {
        double off_diag_sum = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = i + 1; j < n; ++j) {
                off_diag_sum += std::abs(A_mat(i, j));
            }
        }

        if (off_diag_sum <= 1e-14) {
            break;
        }

        for (std::size_t p = 0; p < n; ++p) {
            for (std::size_t q = p + 1; q < n; ++q) {
                double apq = A_mat(p, q);
                if (std::abs(apq) <= 1e-16) continue;

                double app = A_mat(p, p);
                double aqq = A_mat(q, q);
                double tau = (aqq - app) / (2.0 * apq);
                double t = 0.0;
                if (tau >= 0.0) {
                    t = 1.0 / (tau + std::sqrt(1.0 + tau * tau));
                } else {
                    t = -1.0 / (-tau + std::sqrt(1.0 + tau * tau));
                }
                double c = 1.0 / std::sqrt(1.0 + t * t);
                double s = t * c;
                double tau_rot = s / (1.0 + c);

                A_mat(p, p) = app - t * apq;
                A_mat(q, q) = aqq + t * apq;
                A_mat(p, q) = 0.0;
                A_mat(q, p) = 0.0;

                for (std::size_t k = 0; k < n; ++k) {
                    if (k != p && k != q) {
                        double akp = A_mat(k, p);
                        double akq = A_mat(k, q);
                        A_mat(k, p) = akp - s * (akq + tau_rot * akp);
                        A_mat(p, k) = A_mat(k, p);
                        A_mat(k, q) = akq + s * (akp - tau_rot * akq);
                        A_mat(q, k) = A_mat(k, q);
                    }
                }
            }
        }
    }

    std::vector<double> eigenvalues(n);
    for (std::size_t i = 0; i < n; ++i) {
        eigenvalues[i] = A_mat(i, i);
    }
    return eigenvalues;
}

QPResult ConvexQPSolver::solve_unconstrained(
    const QuadraticProgram& qp,
    bool is_strictly_pd,
    bool is_zero_q) const {
    const std::size_t n = qp.Q.rows();
    QPResult res;
    res.iterations = 1;
    res.device_diagnostics.requested_backend = options_.backend;
    res.device_diagnostics.executed_backend = ExecutedBackend::CPU;
    res.device_diagnostics.fallback_reason = "Unconstrained QP is solved directly on CPU.";

    // Linear objective with zero Hessian: min c^T x
    if (is_zero_q) {
        double c_norm = 0.0;
        for (double val : qp.c) c_norm = std::max(c_norm, std::abs(val));

        if (c_norm <= options_.dual_tolerance) {
            res.status = QPSolverStatus::OPTIMAL;
            res.x.assign(n, 0.0);
            res.objective_value = 0.0;
            res.dual_residual = c_norm;
            res.primal_residual = 0.0;
            res.duality_gap = 0.0;
            res.message = "Trivially optimal unconstrained linear objective (c = 0).";
            return res;
        } else {
            res.status = QPSolverStatus::UNBOUNDED;
            res.x.assign(n, 0.0);
            res.objective_value = -std::numeric_limits<double>::infinity();
            res.dual_residual = c_norm;
            res.message = "Unconstrained linear program with non-zero gradient is unbounded below.";
            return res;
        }
    }

    // Strictly PD or singular PSD: solve Q x = -c
    std::vector<double> neg_c(n);
    for (std::size_t i = 0; i < n; ++i) {
        neg_c[i] = -qp.c[i];
    }

    Matrix<double> Q_reg = qp.Q;
    double reg = is_strictly_pd ? 0.0 : options_.regularization;
    if (reg > 0.0) {
        for (std::size_t i = 0; i < n; ++i) {
            Q_reg(i, i) += reg;
        }
    }

    LUDecomposition<double> lu;
    bool success = lu.compute(Q_reg, 1e-12);
    if (!success) {
        // Try increasing regularization for singular PSD
        for (std::size_t i = 0; i < n; ++i) {
            Q_reg(i, i) += 1e-6;
        }
        success = lu.compute(Q_reg, 1e-12);
    }

    if (!success) {
        res.status = QPSolverStatus::NUMERICAL_ERROR;
        res.message = "LU factorization of unconstrained Hessian failed.";
        return res;
    }

    std::vector<double> x_sol = lu.solve(neg_c);

    // Compute residual r_d = Q * x + c
    double dual_res = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double qx_i = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
            qx_i += qp.Q(i, j) * x_sol[j];
        }
        dual_res = std::max(dual_res, std::abs(qx_i + qp.c[i]));
    }

    if (dual_res <= options_.dual_tolerance || (is_strictly_pd && dual_res <= 1e-4)) {
        res.status = QPSolverStatus::OPTIMAL;
        res.x = x_sol;
        res.dual_residual = dual_res;
        res.primal_residual = 0.0;
        res.duality_gap = 0.0;

        // Objective value: 0.5 * x^T * Q * x + c^T * x
        double quad_part = 0.0;
        double lin_part = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            lin_part += qp.c[i] * x_sol[i];
            double row_dot = 0.0;
            for (std::size_t j = 0; j < n; ++j) {
                row_dot += qp.Q(i, j) * x_sol[j];
            }
            quad_part += x_sol[i] * row_dot;
        }
        res.objective_value = 0.5 * quad_part + lin_part;
        res.message = "Unconstrained convex QP solved to stationarity.";
        return res;
    }

    // If residual is large on singular PSD, c has a non-zero projection onto the null space of Q
    res.status = QPSolverStatus::UNBOUNDED;
    res.x = x_sol;
    res.objective_value = -std::numeric_limits<double>::infinity();
    res.dual_residual = dual_res;
    res.message = "Singular positive semidefinite Hessian with non-zero gradient in null space is unbounded below.";
    return res;
}

QPResult ConvexQPSolver::solve_interior_point(const QuadraticProgram& qp) const {
    const std::size_t n = qp.Q.rows();
    const std::size_t m = qp.A.rows();

    QPResult res;
    res.x.assign(n, 0.0);
    res.slack.assign(m, 1.0);
    res.lambda.assign(m, 1.0);
    res.device_diagnostics.requested_backend = options_.backend;

    [[maybe_unused]] bool use_gpu = false;
    if (options_.backend == SolverBackend::GPU) {
        use_gpu = true;
    } else if (options_.backend == SolverBackend::AUTO) {
#if defined(VORTEX_ENABLE_CUDA)
        if (cuda::is_cuda_available() && n >= 32 && m >= 16) {
            use_gpu = true;
        } else {
            use_gpu = false;
            res.device_diagnostics.executed_backend = ExecutedBackend::CPU;
            if (!cuda::is_cuda_available()) {
                res.device_diagnostics.fallback_reason = "CUDA not available on system; AUTO selected CPU.";
            } else {
                res.device_diagnostics.fallback_reason = "QP dimensions (n=" + std::to_string(n) + ", m=" + std::to_string(m) +
                    ") below GPU threshold (n >= 32, m >= 16); AUTO selected CPU.";
            }
        }
#else
        use_gpu = false;
        res.device_diagnostics.executed_backend = ExecutedBackend::CPU;
        res.device_diagnostics.fallback_reason = "VORTEX compiled without CUDA; AUTO selected CPU.";
#endif
    } else {
        res.device_diagnostics.executed_backend = ExecutedBackend::CPU;
    }

#if defined(VORTEX_ENABLE_CUDA)
    std::unique_ptr<cuda::GpuQPIterationContext> gpu_ctx;
    if (use_gpu) {
        try {
            gpu_ctx = std::make_unique<cuda::GpuQPIterationContext>(qp.Q, qp.A);
            res.device_diagnostics.executed_backend = ExecutedBackend::HYBRID;
        } catch (const std::exception& e) {
            if (options_.backend == SolverBackend::GPU) {
                res.status = QPSolverStatus::NUMERICAL_ERROR;
                res.device_diagnostics.executed_backend = ExecutedBackend::CPU;
                res.device_diagnostics.fallback_reason = "GPU initialization failed: " + std::string(e.what());
                res.message = res.device_diagnostics.fallback_reason;
                return res;
            } else {
                use_gpu = false;
                res.device_diagnostics.executed_backend = ExecutedBackend::CPU;
                res.device_diagnostics.fallback_reason = "GPU initialization failed (" + std::string(e.what()) + "); AUTO falling back to CPU.";
            }
        }
    }
#endif

    auto finalize_diagnostics = [&]() {
#if defined(VORTEX_ENABLE_CUDA)
        if (use_gpu && gpu_ctx) {
            res.device_diagnostics.gpu_iterations = res.iterations;
            res.device_diagnostics.gpu_gemv_calls = gpu_ctx->gemv_calls();
            res.device_diagnostics.h2d_transfers = gpu_ctx->h2d_transfers();
            res.device_diagnostics.d2h_transfers = gpu_ctx->d2h_transfers();
        }
#endif
    };

    // Initial point selection:
    // Solve (Q + A^T A + I) x0 = -c + A^T b
    Matrix<double> M_init(n, n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            double at_a = 0.0;
            for (std::size_t k = 0; k < m; ++k) {
                at_a += qp.A(k, i) * qp.A(k, j);
            }
            M_init(i, j) = qp.Q(i, j) + at_a;
        }
        M_init(i, i) += 1.0; // Identity regularization
    }

    std::vector<double> rhs_init(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        double at_b = 0.0;
        for (std::size_t k = 0; k < m; ++k) {
            at_b += qp.A(k, i) * qp.b[k];
        }
        rhs_init[i] = -qp.c[i] + at_b;
    }

    LUDecomposition<double> lu_init(M_init);
    std::vector<double> x(n, 0.0);
    if (!lu_init.is_singular()) {
        x = lu_init.solve(rhs_init);
    }

    // Set initial slack s > 0 and dual multipliers lambda > 0
    std::vector<double> s(m, 1.0);
    std::vector<double> lambda(m, 1.0);

    for (std::size_t i = 0; i < m; ++i) {
        double ax_i = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
            ax_i += qp.A(i, j) * x[j];
        }
        double slack_val = qp.b[i] - ax_i;
        if (slack_val >= 1e-2) {
            s[i] = slack_val;
        } else {
            s[i] = std::max(1.0, 1.0 - slack_val);
        }
        lambda[i] = 1.0;
    }

    double delta = options_.regularization;

    // Primal-Dual Interior-Point Iterations
    for (std::size_t iter = 0; iter < options_.max_iterations; ++iter) {
        res.iterations = iter + 1;

        // 1. Compute residuals:
        //    r_p = A * x + s - b  (primal residual, m x 1)
        //    r_d = Q * x + c + A^T * lambda  (dual residual, n x 1)
        std::vector<double> r_p(m, 0.0);
        std::vector<double> ax(m, 0.0);
#if defined(VORTEX_ENABLE_CUDA)
        if (use_gpu && gpu_ctx) {
            gpu_ctx->compute_Ax(x, ax);
        } else
#endif
        {
            for (std::size_t i = 0; i < m; ++i) {
                for (std::size_t j = 0; j < n; ++j) {
                    ax[i] += qp.A(i, j) * x[j];
                }
            }
        }
        double norm_rp = 0.0;
        for (std::size_t i = 0; i < m; ++i) {
            r_p[i] = ax[i] + s[i] - qp.b[i];
            norm_rp = std::max(norm_rp, std::abs(r_p[i]));
        }

        std::vector<double> r_d(n, 0.0);
        std::vector<double> qx(n, 0.0);
        std::vector<double> atl(n, 0.0);
#if defined(VORTEX_ENABLE_CUDA)
        if (use_gpu && gpu_ctx) {
            gpu_ctx->compute_Qx(x, qx);
            gpu_ctx->compute_AT_vec(lambda, atl);
        } else
#endif
        {
            for (std::size_t j = 0; j < n; ++j) {
                for (std::size_t k = 0; k < n; ++k) {
                    qx[j] += qp.Q(j, k) * x[k];
                }
                for (std::size_t i = 0; i < m; ++i) {
                    atl[j] += qp.A(i, j) * lambda[i];
                }
            }
        }
        double norm_rd = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
            r_d[j] = qx[j] + qp.c[j] + atl[j];
            norm_rd = std::max(norm_rd, std::abs(r_d[j]));
        }

        // Duality gap: mu = (s^T * lambda) / m
        double comp_sum = 0.0;
        for (std::size_t i = 0; i < m; ++i) {
            comp_sum += s[i] * lambda[i];
        }
        double mu = comp_sum / static_cast<double>(m);

        res.primal_residual = norm_rp;
        res.dual_residual = norm_rd;
        res.duality_gap = mu;

        // Record genuine step-level iteration telemetry (bounded to max_iterations)
        double iter_quad = 0.0, iter_lin = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
            iter_lin += qp.c[j] * x[j];
            iter_quad += x[j] * qx[j];
        }
        res.iteration_history.push_back({iter, 0.5 * iter_quad + iter_lin, norm_rp, norm_rd, mu});

        if (options_.verbose) {
            std::cout << "[VØRTEX QP IPM] Iter " << std::setw(3) << iter
                      << " | ||r_p||=" << std::scientific << std::setprecision(3) << norm_rp
                      << " | ||r_d||=" << norm_rd
                      << " | mu=" << mu << std::fixed << std::endl;
        }

        // 2. Check convergence
        if (norm_rp <= options_.primal_tolerance &&
            norm_rd <= options_.dual_tolerance &&
            mu <= options_.duality_gap_tolerance) {
            res.status = QPSolverStatus::OPTIMAL;
            res.x = x;
            res.slack = s;
            res.lambda = lambda;

            // Objective value
            double quad_part = 0.0;
            double lin_part = 0.0;
            std::vector<double> qx_final(n, 0.0);
#if defined(VORTEX_ENABLE_CUDA)
            if (use_gpu && gpu_ctx) {
                gpu_ctx->compute_Qx(x, qx_final);
            } else
#endif
            {
                for (std::size_t i = 0; i < n; ++i) {
                    for (std::size_t j = 0; j < n; ++j) {
                        qx_final[i] += qp.Q(i, j) * x[j];
                    }
                }
            }
            for (std::size_t i = 0; i < n; ++i) {
                lin_part += qp.c[i] * x[i];
                quad_part += x[i] * qx_final[i];
            }
            res.objective_value = 0.5 * quad_part + lin_part;
            res.message = "Optimality conditions (primal feasibility, dual feasibility, complementarity) satisfied.";
            finalize_diagnostics();
            return res;
        }

        // 3. Check divergence / unboundedness
        double x_max = 0.0;
        for (double val : x) x_max = std::max(x_max, std::abs(val));
        double s_max = 0.0;
        for (double val : s) s_max = std::max(s_max, std::abs(val));
        double l_max = 0.0;
        for (double val : lambda) l_max = std::max(l_max, std::abs(val));

        if (x_max > 1e14 || s_max > 1e14 || l_max > 1e14) {
            if (norm_rp > 1.0 && l_max > 1e8) {
                res.status = QPSolverStatus::INFEASIBLE;
                res.message = "Dual multipliers diverged while primal constraints remain violated (infeasible model).";
            } else {
                res.status = QPSolverStatus::UNBOUNDED;
                res.message = "Iterates diverged to infinity along unconstrained recession direction.";
            }
            res.x = x;
            res.slack = s;
            res.lambda = lambda;
            finalize_diagnostics();
            return res;
        }

        // 4. Form reduced KKT matrix:
        //    H = Q + A^T * D * A + delta * I
        //    where D = diag(lambda_i / s_i) > 0
        std::vector<double> d(m);
        for (std::size_t i = 0; i < m; ++i) {
            d[i] = lambda[i] / s[i];
        }

        Matrix<double> H(n, n, 0.0);
        for (std::size_t j = 0; j < n; ++j) {
            for (std::size_t k = j; k < n; ++k) {
                double at_d_a = 0.0;
                for (std::size_t i = 0; i < m; ++i) {
                    at_d_a += qp.A(i, j) * d[i] * qp.A(i, k);
                }
                double val = qp.Q(j, k) + at_d_a;
                if (j == k) val += delta;
                H(j, k) = val;
                H(k, j) = val;
            }
        }

        LUDecomposition<double> lu(H);
        if (lu.is_singular()) {
            delta = std::max(1e-8, delta * 100.0);
            if (delta > 1.0) {
                res.status = QPSolverStatus::NUMERICAL_ERROR;
                res.message = "KKT reduced system is singular and could not be factorized.";
                res.x = x;
                res.slack = s;
                res.lambda = lambda;
                finalize_diagnostics();
                return res;
            }
            // Retry with larger delta
            for (std::size_t j = 0; j < n; ++j) H(j, j) += delta;
            lu.compute(H);
            if (lu.is_singular()) {
                res.status = QPSolverStatus::NUMERICAL_ERROR;
                res.message = "KKT matrix factorization failed after regularization.";
                res.x = x;
                res.slack = s;
                res.lambda = lambda;
                finalize_diagnostics();
                return res;
            }
        }

        // 5. Predictor step (affine scaling, sigma = 0):
        //    rhs_aff = -r_d - A^T * (D * r_p)
        std::vector<double> d_rp(m);
        for (std::size_t i = 0; i < m; ++i) {
            d_rp[i] = d[i] * r_p[i];
        }

        std::vector<double> at_drp(n, 0.0);
#if defined(VORTEX_ENABLE_CUDA)
        if (use_gpu && gpu_ctx) {
            gpu_ctx->compute_AT_vec(d_rp, at_drp);
        } else
#endif
        {
            for (std::size_t j = 0; j < n; ++j) {
                for (std::size_t i = 0; i < m; ++i) {
                    at_drp[j] += qp.A(i, j) * d_rp[i];
                }
            }
        }

        std::vector<double> rhs_aff(n);
        for (std::size_t j = 0; j < n; ++j) {
            rhs_aff[j] = -r_d[j] - at_drp[j];
        }

        std::vector<double> dx_aff = lu.solve(rhs_aff);

        // ds_aff = -r_p - A * dx_aff
        std::vector<double> a_dx_aff(m, 0.0);
#if defined(VORTEX_ENABLE_CUDA)
        if (use_gpu && gpu_ctx) {
            gpu_ctx->compute_Ax(dx_aff, a_dx_aff);
        } else
#endif
        {
            for (std::size_t i = 0; i < m; ++i) {
                for (std::size_t j = 0; j < n; ++j) {
                    a_dx_aff[i] += qp.A(i, j) * dx_aff[j];
                }
            }
        }

        std::vector<double> ds_aff(m);
        for (std::size_t i = 0; i < m; ++i) {
            ds_aff[i] = -r_p[i] - a_dx_aff[i];
        }

        // dlambda_aff = -lambda - D * ds_aff
        std::vector<double> dl_aff(m);
        for (std::size_t i = 0; i < m; ++i) {
            dl_aff[i] = -lambda[i] - d[i] * ds_aff[i];
        }

        // Maximum affine step size
        double alpha_s_aff = 1.0;
        double alpha_l_aff = 1.0;
        for (std::size_t i = 0; i < m; ++i) {
            if (ds_aff[i] < 0.0) {
                alpha_s_aff = std::min(alpha_s_aff, -s[i] / ds_aff[i]);
            }
            if (dl_aff[i] < 0.0) {
                alpha_l_aff = std::min(alpha_l_aff, -lambda[i] / dl_aff[i]);
            }
        }
        double alpha_aff = std::min(alpha_s_aff, alpha_l_aff);

        // Affine gap
        double mu_aff_sum = 0.0;
        for (std::size_t i = 0; i < m; ++i) {
            double s_next = s[i] + alpha_aff * ds_aff[i];
            double l_next = lambda[i] + alpha_aff * dl_aff[i];
            mu_aff_sum += s_next * l_next;
        }
        double mu_aff = mu_aff_sum / static_cast<double>(m);

        // Centering parameter sigma
        double sigma = (mu > 1e-15) ? std::pow(mu_aff / mu, 3.0) : 0.0;
        sigma = std::clamp(sigma, 1e-4, 0.999);

        // 6. Corrector / Centering step:
        //    r_comp = -s_i * lambda_i - ds_aff_i * dl_aff_i + sigma * mu
        //    w_i = r_comp_i / s_i
        //    rhs_corr = -r_d - A^T * (w + D * r_p)
        std::vector<double> w_plus_drp(m);
        std::vector<double> r_comp(m);
        for (std::size_t i = 0; i < m; ++i) {
            r_comp[i] = -s[i] * lambda[i] - ds_aff[i] * dl_aff[i] + sigma * mu;
            double w_i = r_comp[i] / s[i];
            w_plus_drp[i] = w_i + d_rp[i];
        }

        std::vector<double> at_w(n, 0.0);
#if defined(VORTEX_ENABLE_CUDA)
        if (use_gpu && gpu_ctx) {
            gpu_ctx->compute_AT_vec(w_plus_drp, at_w);
        } else
#endif
        {
            for (std::size_t j = 0; j < n; ++j) {
                for (std::size_t i = 0; i < m; ++i) {
                    at_w[j] += qp.A(i, j) * w_plus_drp[i];
                }
            }
        }

        std::vector<double> rhs_corr(n);
        for (std::size_t j = 0; j < n; ++j) {
            rhs_corr[j] = -r_d[j] - at_w[j];
        }

        std::vector<double> dx = lu.solve(rhs_corr);

        // ds = -r_p - A * dx
        std::vector<double> a_dx(m, 0.0);
#if defined(VORTEX_ENABLE_CUDA)
        if (use_gpu && gpu_ctx) {
            gpu_ctx->compute_Ax(dx, a_dx);
        } else
#endif
        {
            for (std::size_t i = 0; i < m; ++i) {
                for (std::size_t j = 0; j < n; ++j) {
                    a_dx[i] += qp.A(i, j) * dx[j];
                }
            }
        }

        std::vector<double> ds(m);
        for (std::size_t i = 0; i < m; ++i) {
            ds[i] = -r_p[i] - a_dx[i];
        }

        // dlambda = (r_comp - lambda * ds) / s
        std::vector<double> dl(m);
        for (std::size_t i = 0; i < m; ++i) {
            dl[i] = (r_comp[i] - lambda[i] * ds[i]) / s[i];
        }

        // 7. Fraction-to-the-boundary step length
        double tau = options_.step_damping; // e.g. 0.995
        double alpha_s = 1.0;
        double alpha_l = 1.0;
        for (std::size_t i = 0; i < m; ++i) {
            if (ds[i] < 0.0) {
                alpha_s = std::min(alpha_s, -tau * s[i] / ds[i]);
            }
            if (dl[i] < 0.0) {
                alpha_l = std::min(alpha_l, -tau * lambda[i] / dl[i]);
            }
        }

        // Update iterates
        for (std::size_t j = 0; j < n; ++j) {
            x[j] += alpha_s * dx[j];
        }
        for (std::size_t i = 0; i < m; ++i) {
            s[i] += alpha_s * ds[i];
            lambda[i] += alpha_l * dl[i];
            if (s[i] <= 1e-15) s[i] = 1e-15;
            if (lambda[i] <= 1e-15) lambda[i] = 1e-15;
        }
    }

    // Maximum iterations reached
    res.status = QPSolverStatus::MAX_ITERATIONS;
    res.x = x;
    res.slack = s;
    res.lambda = lambda;
    double quad_part = 0.0;
    double lin_part = 0.0;
    std::vector<double> qx_final(n, 0.0);
#if defined(VORTEX_ENABLE_CUDA)
    if (use_gpu && gpu_ctx) {
        gpu_ctx->compute_Qx(x, qx_final);
    } else
#endif
    {
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                qx_final[i] += qp.Q(i, j) * x[j];
            }
        }
    }
    for (std::size_t i = 0; i < n; ++i) {
        lin_part += qp.c[i] * x[i];
        quad_part += x[i] * qx_final[i];
    }
    res.objective_value = 0.5 * quad_part + lin_part;
    res.message = "Maximum interior-point iterations reached without satisfying convergence criteria.";
    finalize_diagnostics();
    return res;
}

} // namespace vortex
