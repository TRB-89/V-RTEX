#include "vortex/revised_simplex.hpp"
#include "vortex/sparse_revised_simplex.hpp"
#include "vortex/qp.hpp"
#include "vortex/cuda_backend.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>
#include <random>
#include <cassert>

namespace {

int total_tests = 0;
int passed_tests = 0;

void record_result(const std::string& test_name, bool passed, const std::string& details = "") {
    total_tests++;
    if (passed) {
        passed_tests++;
        std::cout << "  [PASS] " << test_name << "\n";
    } else {
        std::cout << "  [FAIL] " << test_name;
        if (!details.empty()) {
            std::cout << " - " << details;
        }
        std::cout << "\n";
    }
}

bool approx_equal(double a, double b, double tol = 1e-6) {
    if (std::isnan(a) || std::isnan(b)) return false;
    if (std::isinf(a) && std::isinf(b)) return (a > 0) == (b > 0);
    return std::abs(a - b) <= tol * (1.0 + std::max(std::abs(a), std::abs(b)));
}

} // anonymous namespace

// ==============================================================================
// Test 1: Backend Selection & Heuristics Safety Verification
// ==============================================================================
void test_backend_selection_and_safety() {
    std::cout << "\n=== Test 1: Backend Selection & Safety Verification ===\n";

    // 1. Explicit CPU backend on Revised Simplex
    {
        vortex::RevisedSimplexOptions opts;
        opts.backend = vortex::SolverBackend::CPU;
        vortex::RevisedSimplexSolver solver(opts);

        std::vector<double> c = {3.0, 2.0};
        vortex::Matrix<double> A(2, 2);
        A(0, 0) = 2.0; A(0, 1) = 1.0;
        A(1, 0) = 1.0; A(1, 1) = 2.0;
        std::vector<double> b = {100.0, 80.0};

        auto res = solver.solve(c, A, b);
        record_result("Explicit CPU backend returns OPTIMAL", res.status == vortex::SolverStatus::OPTIMAL);
        record_result("Explicit CPU backend executed_backend is CPU", res.device_diagnostics.executed_backend == vortex::ExecutedBackend::CPU);
        record_result("Explicit CPU backend has 0 GPU iterations", res.device_diagnostics.gpu_iterations == 0);
        record_result("Explicit CPU backend has 0 GPU GEMV calls", res.device_diagnostics.gpu_gemv_calls == 0);
    }

    // 2. AUTO backend threshold check on small LP
    {
        vortex::RevisedSimplexOptions opts;
        opts.backend = vortex::SolverBackend::AUTO;
        vortex::RevisedSimplexSolver solver(opts);

        std::vector<double> c = {1.0, 1.0};
        vortex::Matrix<double> A(2, 2);
        A(0, 0) = 1.0; A(0, 1) = 0.0;
        A(1, 0) = 0.0; A(1, 1) = 1.0;
        std::vector<double> b = {5.0, 5.0};

        auto res = solver.solve(c, A, b);
        record_result("AUTO backend on small LP (2x4) falls back to CPU", res.device_diagnostics.executed_backend == vortex::ExecutedBackend::CPU);
        record_result("AUTO backend on small LP documents fallback reason", !res.device_diagnostics.fallback_reason.empty());
    }

    // 3. Explicit GPU backend on system
    {
        vortex::RevisedSimplexOptions opts;
        opts.backend = vortex::SolverBackend::GPU;
        vortex::RevisedSimplexSolver solver(opts);

        std::vector<double> c = {3.0, 2.0};
        vortex::Matrix<double> A(2, 2);
        A(0, 0) = 2.0; A(0, 1) = 1.0;
        A(1, 0) = 1.0; A(1, 1) = 2.0;
        std::vector<double> b = {100.0, 80.0};

        auto res = solver.solve(c, A, b);
        if (vortex::cuda::is_cuda_available()) {
            record_result("Explicit GPU backend executes HYBRID when CUDA device available",
                          res.device_diagnostics.executed_backend == vortex::ExecutedBackend::HYBRID);
            record_result("Explicit GPU backend tracks GPU iterations > 0", res.device_diagnostics.gpu_iterations > 0);
            record_result("Explicit GPU backend tracks GEMV calls > 0", res.device_diagnostics.gpu_gemv_calls > 0);
            record_result("Explicit GPU backend tracks H2D transfers > 0", res.device_diagnostics.h2d_transfers > 0);
            record_result("Explicit GPU backend tracks D2H transfers > 0", res.device_diagnostics.d2h_transfers > 0);
        } else {
            record_result("Explicit GPU backend reports INVALID_INPUT when CUDA unavailable",
                          res.status == vortex::SolverStatus::INVALID_INPUT);
            record_result("Explicit GPU backend documents unavailable reason without silent CPU mislabeling",
                          !res.device_diagnostics.fallback_reason.empty());
        }
    }
}

// ==============================================================================
// Test 2: Dense Revised Simplex LP Hybrid Correctness & Cross-Validation
// ==============================================================================
void test_dense_revised_simplex_hybrid_cross_validation() {
    std::cout << "\n=== Test 2: Dense Revised Simplex LP Hybrid Cross-Validation ===\n";

    // Create a medium LP instance: 35 constraints, 40 decision variables
    const std::size_t m = 35;
    const std::size_t n = 40;

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist_A(0.1, 5.0);
    std::uniform_real_distribution<double> dist_c(1.0, 20.0);
    std::uniform_real_distribution<double> dist_b(50.0, 200.0);

    std::vector<double> c(n);
    for (std::size_t j = 0; j < n; ++j) c[j] = dist_c(rng);

    vortex::Matrix<double> A(m, n);
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            A(i, j) = dist_A(rng);
        }
    }

    std::vector<double> b(m);
    for (std::size_t i = 0; i < m; ++i) b[i] = dist_b(rng);

    // Solve with CPU backend
    vortex::RevisedSimplexOptions cpu_opts;
    cpu_opts.backend = vortex::SolverBackend::CPU;
    vortex::RevisedSimplexSolver cpu_solver(cpu_opts);
    auto cpu_res = cpu_solver.solve(c, A, b);

    record_result("CPU solver reaches OPTIMAL", cpu_res.status == vortex::SolverStatus::OPTIMAL);

    if (vortex::cuda::is_cuda_available()) {
        // Solve with GPU backend
        vortex::RevisedSimplexOptions gpu_opts;
        gpu_opts.backend = vortex::SolverBackend::GPU;
        vortex::RevisedSimplexSolver gpu_solver(gpu_opts);
        auto gpu_res = gpu_solver.solve(c, A, b);

        record_result("GPU solver reaches OPTIMAL", gpu_res.status == vortex::SolverStatus::OPTIMAL);
        record_result("Objective value matches CPU within 1e-6",
                      approx_equal(gpu_res.objective_value, cpu_res.objective_value, 1e-6),
                      "CPU obj: " + std::to_string(cpu_res.objective_value) + ", GPU obj: " + std::to_string(gpu_res.objective_value));

        bool vars_match = true;
        for (std::size_t j = 0; j < n; ++j) {
            if (!approx_equal(gpu_res.variable_values[j], cpu_res.variable_values[j], 1e-5)) {
                vars_match = false;
                break;
            }
        }
        record_result("Solution variables match CPU within 1e-5", vars_match);

        record_result("GPU iterations match pricing evaluations (iters + 1)",
                      gpu_res.device_diagnostics.gpu_iterations == gpu_res.iterations + 1,
                      "gpu_iters=" + std::to_string(gpu_res.device_diagnostics.gpu_iterations) + ", iters=" + std::to_string(gpu_res.iterations));
        record_result("cuBLAS GEMV calls equal pricing evaluations (iters + 1)",
                      gpu_res.device_diagnostics.gpu_gemv_calls == gpu_res.iterations + 1);
        record_result("H2D transfers = 1 (A_std initial) + 2 * pricing_calls",
                      gpu_res.device_diagnostics.h2d_transfers == (1 + 2 * (gpu_res.iterations + 1)));
        record_result("D2H transfers = 1 * pricing_calls",
                      gpu_res.device_diagnostics.d2h_transfers == gpu_res.iterations + 1);
        record_result("GPU residual ||Ax - b|| within tolerance",
                      gpu_res.max_primal_residual <= 1e-5);

        // Also test AUTO backend on this problem
        vortex::RevisedSimplexOptions auto_opts;
        auto_opts.backend = vortex::SolverBackend::AUTO;
        vortex::RevisedSimplexSolver auto_solver(auto_opts);
        auto auto_res = auto_solver.solve(c, A, b);
        record_result("AUTO backend solves successfully", auto_res.status == vortex::SolverStatus::OPTIMAL);
    }
}

// ==============================================================================
// Test 3: Two-Phase Dense Revised Simplex LP with Mixed Constraints
// ==============================================================================
void test_two_phase_dense_simplex_hybrid() {
    std::cout << "\n=== Test 3: Two-Phase Dense Revised Simplex Hybrid ===\n";

    // 20 variables, 15 constraints with mixed LEQ, GEQ, and EQ constraints (guaranteed feasible)
    const std::size_t n = 20;
    const std::size_t m = 15;

    std::vector<double> c(n, 1.0);
    for (std::size_t j = 0; j < n; ++j) c[j] = static_cast<double>(j + 1);

    vortex::Matrix<double> A(m, n, 0.0);
    // Identity block ensures simple feasibility
    for (std::size_t i = 0; i < m; ++i) {
        A(i, i) = 1.0;
        if (i + 1 < n) A(i, i + 1) = 0.5;
    }

    std::vector<double> b(m, 10.0);
    std::vector<vortex::ConstraintType> types(m, vortex::ConstraintType::LEQ);
    // Add a couple of GEQ and EQ constraints with feasible RHS
    types[0] = vortex::ConstraintType::GEQ; b[0] = 2.0;
    types[1] = vortex::ConstraintType::EQ;  b[1] = 5.0;

    // CPU Reference
    vortex::RevisedSimplexOptions cpu_opts;
    cpu_opts.backend = vortex::SolverBackend::CPU;
    vortex::RevisedSimplexSolver cpu_solver(cpu_opts);
    auto cpu_res = cpu_solver.solve(c, A, b, types, vortex::ObjectiveSense::MINIMIZE);

    std::cout << "  CPU Two-Phase status: " << to_string(cpu_res.status) << ", obj: " << cpu_res.objective_value
              << ", iters: " << cpu_res.iterations << " (P1: " << cpu_res.phase1_iterations << ", P2: " << cpu_res.phase2_iterations << ")\n";

    if (vortex::cuda::is_cuda_available()) {
        vortex::RevisedSimplexOptions gpu_opts;
        gpu_opts.backend = vortex::SolverBackend::GPU;
        vortex::RevisedSimplexSolver gpu_solver(gpu_opts);
        auto gpu_res = gpu_solver.solve(c, A, b, types, vortex::ObjectiveSense::MINIMIZE);

        std::cout << "  GPU Two-Phase status: " << to_string(gpu_res.status) << ", obj: " << gpu_res.objective_value
                  << ", iters: " << gpu_res.iterations << ", gpu_iters: " << gpu_res.device_diagnostics.gpu_iterations << "\n";

        record_result("Two-Phase solver status matches CPU", gpu_res.status == cpu_res.status);
        record_result("Two-Phase objective matches CPU within 1e-5",
                      approx_equal(gpu_res.objective_value, cpu_res.objective_value, 1e-5));
        record_result("Two-Phase tracks Phase I and Phase II iterations",
                      gpu_res.iterations == (gpu_res.phase1_iterations + gpu_res.phase2_iterations));
        record_result("Two-Phase cumulative GPU iterations tracked",
                      gpu_res.device_diagnostics.gpu_iterations > 0);
        record_result("Two-Phase cumulative GEMV calls tracked",
                      gpu_res.device_diagnostics.gpu_gemv_calls > 0);
    }
}

// ==============================================================================
// Test 4: Sparse Revised Simplex LP Hybrid Correctness
// ==============================================================================
void test_sparse_revised_simplex_hybrid() {
    std::cout << "\n=== Test 4: Sparse Revised Simplex LP Hybrid ===\n";

    // Bounded sparse LP: 40 constraints, 35 variables
    const std::size_t m = 40;
    const std::size_t n = 35;

    std::vector<vortex::Triplet<double>> triplets;
    // Bounding diagonal entries for each variable
    for (std::size_t j = 0; j < n; ++j) {
        triplets.push_back({j, j, 2.0});
        if (j + 1 < n) triplets.push_back({j, j + 1, 0.5});
    }
    // Additional coupling rows
    for (std::size_t i = n; i < m; ++i) {
        for (std::size_t j = 0; j < n; j += 4) {
            triplets.push_back({i, j, 0.25});
        }
    }

    auto A_csc = vortex::CSCMatrix<double>::from_triplets(m, n, triplets);
    std::vector<double> c(n);
    for (std::size_t j = 0; j < n; ++j) c[j] = static_cast<double>(j + 1);

    std::vector<double> b(m, 50.0);

    // CPU reference
    vortex::RevisedSimplexOptions cpu_opts;
    cpu_opts.backend = vortex::SolverBackend::CPU;
    vortex::SparseRevisedSimplexSolver cpu_solver(cpu_opts);
    auto cpu_res = cpu_solver.solve(c, A_csc, b);

    record_result("Sparse CPU solver finds OPTIMAL", cpu_res.status == vortex::SolverStatus::OPTIMAL);

    if (vortex::cuda::is_cuda_available()) {
        vortex::RevisedSimplexOptions gpu_opts;
        gpu_opts.backend = vortex::SolverBackend::GPU;
        vortex::SparseRevisedSimplexSolver gpu_solver(gpu_opts);
        auto gpu_res = gpu_solver.solve(c, A_csc, b);

        record_result("Sparse GPU solver finds OPTIMAL", gpu_res.status == vortex::SolverStatus::OPTIMAL);
        record_result("Sparse GPU objective matches CPU within 1e-6",
                      approx_equal(gpu_res.objective_value, cpu_res.objective_value, 1e-6));
        record_result("Sparse GPU iterations match pricing evaluations (iters + 1)",
                      gpu_res.device_diagnostics.gpu_iterations == gpu_res.iterations + 1);
        record_result("Sparse cuSPARSE SpMV calls tracked",
                      gpu_res.device_diagnostics.gpu_spmv_calls == gpu_res.iterations + 1);
        record_result("Sparse H2D and D2H transfers tracked",
                      gpu_res.device_diagnostics.h2d_transfers > 0 && gpu_res.device_diagnostics.d2h_transfers > 0);
    }
}

// ==============================================================================
// Test 5: Convex QP Primal-Dual Interior-Point Hybrid Correctness
// ==============================================================================
void test_convex_qp_hybrid_integration() {
    std::cout << "\n=== Test 5: Convex QP Hybrid Interior-Point Integration ===\n";

    // Constrained Portfolio / Convex QP Problem:
    // Minimize (1/2) * x^T * Q * x + c^T * x
    // Subject to A * x <= b
    // n = 35 variables, m = 20 constraints
    const std::size_t n = 35;
    const std::size_t m = 20;

    std::mt19937 rng(9876);
    std::uniform_real_distribution<double> dist_rand(0.1, 2.0);

    // Build symmetric positive definite matrix Q = R^T * R + 0.5 * I
    vortex::Matrix<double> R(n, n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            R(i, j) = (i == j) ? 1.0 : (dist_rand(rng) * 0.1);
        }
    }
    vortex::Matrix<double> Q(n, n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            double dot = 0.0;
            for (std::size_t k = 0; k < n; ++k) {
                dot += R(k, i) * R(k, j);
            }
            Q(i, j) = dot + (i == j ? 0.5 : 0.0);
        }
    }

    std::vector<double> c(n);
    for (std::size_t j = 0; j < n; ++j) c[j] = -dist_rand(rng);

    vortex::Matrix<double> A(m, n);
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            A(i, j) = dist_rand(rng) * 0.5;
        }
    }

    std::vector<double> b(m);
    for (std::size_t i = 0; i < m; ++i) b[i] = dist_rand(rng) * 10.0 + 5.0;

    vortex::QuadraticProgram qp{Q, c, A, b};

    // 1. CPU Reference solve
    vortex::QPOptions cpu_opts;
    cpu_opts.backend = vortex::SolverBackend::CPU;
    vortex::ConvexQPSolver cpu_solver(cpu_opts);
    auto cpu_res = cpu_solver.solve(qp);

    record_result("CPU QP solver converges to OPTIMAL", cpu_res.status == vortex::QPSolverStatus::OPTIMAL);

    if (vortex::cuda::is_cuda_available()) {
        // 2. GPU solve
        vortex::QPOptions gpu_opts;
        gpu_opts.backend = vortex::SolverBackend::GPU;
        vortex::ConvexQPSolver gpu_solver(gpu_opts);
        auto gpu_res = gpu_solver.solve(qp);

        record_result("GPU QP solver converges to OPTIMAL", gpu_res.status == vortex::QPSolverStatus::OPTIMAL);
        record_result("GPU QP objective matches CPU within 1e-5",
                      approx_equal(gpu_res.objective_value, cpu_res.objective_value, 1e-5),
                      "CPU: " + std::to_string(cpu_res.objective_value) + ", GPU: " + std::to_string(gpu_res.objective_value));

        bool x_matches = true;
        for (std::size_t j = 0; j < n; ++j) {
            if (!approx_equal(gpu_res.x[j], cpu_res.x[j], 1e-4)) {
                x_matches = false;
                break;
            }
        }
        record_result("GPU QP solution vector x matches CPU within 1e-4", x_matches);

        record_result("GPU QP primal residual ||Ax + s - b|| <= 1e-6", gpu_res.primal_residual <= 1e-6);
        record_result("GPU QP dual residual ||Qx + c + A^T lambda|| <= 1e-6", gpu_res.dual_residual <= 1e-6);
        record_result("GPU QP duality gap <= 1e-6", gpu_res.duality_gap <= 1e-6);

        record_result("GPU QP executed backend is HYBRID",
                      gpu_res.device_diagnostics.executed_backend == vortex::ExecutedBackend::HYBRID);
        record_result("GPU QP iterations equal total solver iterations",
                      gpu_res.device_diagnostics.gpu_iterations == gpu_res.iterations);
        record_result("cuBLAS GEMV operations called in every interior iteration (>= 6 * iters)",
                      gpu_res.device_diagnostics.gpu_gemv_calls >= 6 * gpu_res.iterations);
        record_result("H2D and D2H transfers tracked accurately",
                      gpu_res.device_diagnostics.h2d_transfers > 0 && gpu_res.device_diagnostics.d2h_transfers > 0);

        // 3. AUTO backend heuristic validation on this problem (n=35 >= 32, m=20 >= 16)
        vortex::QPOptions auto_opts;
        auto_opts.backend = vortex::SolverBackend::AUTO;
        vortex::ConvexQPSolver auto_solver(auto_opts);
        auto auto_res = auto_solver.solve(qp);
        record_result("AUTO backend selects HYBRID for n=35, m=20 problem",
                      auto_res.device_diagnostics.executed_backend == vortex::ExecutedBackend::HYBRID);
    }
}

// ==============================================================================
// Test 6: Infeasible, Unbounded & Degenerate Cases
// ==============================================================================
void test_edge_cases_and_robustness() {
    std::cout << "\n=== Test 6: Edge Cases, Degeneracy & Robustness ===\n";

    const auto edge_backend = vortex::cuda::is_cuda_available() ?
        vortex::SolverBackend::GPU : vortex::SolverBackend::CPU;

    // 1. Unbounded LP
    {
        vortex::RevisedSimplexOptions opts;
        opts.backend = edge_backend;
        vortex::RevisedSimplexSolver solver(opts);

        std::vector<double> c = {2.0, 1.0};
        vortex::Matrix<double> A(1, 2);
        A(0, 0) = -1.0; A(0, 1) = -1.0; // Negative row direction -> unbounded
        std::vector<double> b = {10.0};

        auto res = solver.solve(c, A, b);
        record_result("Solver correctly reports UNBOUNDED", res.status == vortex::SolverStatus::UNBOUNDED);
    }

    // 2. Infeasible LP
    {
        vortex::RevisedSimplexOptions opts;
        opts.backend = edge_backend;
        vortex::RevisedSimplexSolver solver(opts);

        // Contradictory: x1 <= 2 and x1 >= 5
        std::vector<double> c = {1.0};
        vortex::Matrix<double> A(2, 1);
        A(0, 0) = 1.0;
        A(1, 0) = 1.0;
        std::vector<double> b = {2.0, 5.0};
        std::vector<vortex::ConstraintType> types = {vortex::ConstraintType::LEQ, vortex::ConstraintType::GEQ};

        auto res = solver.solve(c, A, b, types, vortex::ObjectiveSense::MAXIMIZE);
        record_result("Solver correctly reports INFEASIBLE", res.status == vortex::SolverStatus::INFEASIBLE);
    }

    // 3. Degenerate problem: cycling protection (Bland's rule tie-break)
    {
        vortex::RevisedSimplexOptions opts;
        opts.backend = edge_backend;
        opts.pivot_rule = vortex::PivotRule::BLAND;
        vortex::RevisedSimplexSolver solver(opts);

        // Beale's cycling problem
        std::vector<double> c = {10.0, -57.0, -9.0, -24.0};
        vortex::Matrix<double> A(3, 4);
        A(0, 0) = 0.5;  A(0, 1) = -5.5; A(0, 2) = -2.5; A(0, 3) = 9.0;
        A(1, 0) = 0.5;  A(1, 1) = -1.5; A(1, 2) = -0.5; A(1, 3) = 1.0;
        A(2, 0) = 1.0;  A(2, 1) = 0.0;  A(2, 2) = 0.0;  A(2, 3) = 0.0;
        std::vector<double> b = {0.0, 0.0, 1.0};

        auto res = solver.solve(c, A, b);
        record_result("Degenerate Beale problem terminates with OPTIMAL under Bland rule",
                      res.status == vortex::SolverStatus::OPTIMAL);
    }

    // 4. Invalid inputs propagation (Dimension mismatch)
    {
        vortex::RevisedSimplexOptions opts;
        opts.backend = edge_backend;
        vortex::RevisedSimplexSolver solver(opts);

        std::vector<double> c = {1.0, 2.0};
        vortex::Matrix<double> A(3, 3); // 3 columns vs c with 2 elements
        std::vector<double> b = {1.0, 2.0, 3.0};

        auto res = solver.solve(c, A, b);
        record_result("Mismatched dimensions rejected with INVALID_INPUT",
                      res.status == vortex::SolverStatus::INVALID_INPUT);
    }
}

// ==============================================================================
// Main Runner
// ==============================================================================
int main() {
    std::cout << "==============================================================================\n";
    std::cout << "  VØRTEX PHASE 8D: HYBRID CPU/GPU OPTIMIZATION SOLVER INTEGRATION TESTS\n";
    std::cout << "==============================================================================\n";

    if (vortex::cuda::is_cuda_available()) {
        auto dev = vortex::cuda::get_cuda_device_info();
        std::cout << "Detected Active CUDA Device: " << dev.name << " (CC " << dev.major << "." << dev.minor << ")\n";
        std::cout << "Global Device Memory: " << (dev.total_memory_bytes / (1024 * 1024)) << " MB\n";
    } else {
        std::cout << "CUDA device not detected; testing CPU execution and graceful GPU rejection paths.\n";
    }

    test_backend_selection_and_safety();
    test_dense_revised_simplex_hybrid_cross_validation();
    test_two_phase_dense_simplex_hybrid();
    test_sparse_revised_simplex_hybrid();
    test_convex_qp_hybrid_integration();
    test_edge_cases_and_robustness();

    std::cout << "\n==============================================================================\n";
    std::cout << "  SUMMARY: " << passed_tests << " / " << total_tests << " tests passed.\n";
    std::cout << "==============================================================================\n";

    return (passed_tests == total_tests) ? 0 : 1;
}
