#include "vortex/matrix.hpp"
#include "vortex/sparse_matrix.hpp"
#include "vortex/sparse_revised_simplex.hpp"
#include "vortex/revised_simplex.hpp"
#include <chrono>
#include <iostream>
#include <random>
#include <iomanip>
#include <vector>

void benchmark_memory_footprint() {
    std::cout << "=========================================================\n";
    std::cout << "  BENCHMARK 1: Memory Footprint (Dense vs CSR vs CSC)    \n";
    std::cout << "=========================================================\n\n";

    struct TestCase {
        std::size_t rows;
        std::size_t cols;
        double target_sparsity;
    };

    std::vector<TestCase> cases = {
        {100, 100, 0.90},
        {500, 500, 0.95},
        {1000, 1000, 0.98},
        {2000, 2000, 0.995}
    };

    std::cout << std::left << std::setw(14) << "Dimensions"
              << std::setw(12) << "Sparsity"
              << std::setw(12) << "Non-Zeros"
              << std::setw(14) << "Dense Mem"
              << std::setw(14) << "Sparse Mem"
              << std::setw(12) << "Savings"
              << "\n";
    std::cout << std::string(78, '-') << "\n";

    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
    std::uniform_real_distribution<double> val_dist(-10.0, 10.0);

    for (const auto& tc : cases) {
        std::vector<vortex::Triplet<double>> triplets;
        const double density = 1.0 - tc.target_sparsity;

        for (std::size_t r = 0; r < tc.rows; ++r) {
            for (std::size_t c = 0; c < tc.cols; ++c) {
                if (prob_dist(rng) < density) {
                    triplets.emplace_back(r, c, val_dist(rng));
                }
            }
        }

        auto csr = vortex::CSRMatrix<double>::from_triplets(tc.rows, tc.cols, triplets);
        const std::size_t dense_bytes = tc.rows * tc.cols * sizeof(double);
        const std::size_t sparse_bytes = csr.memory_bytes();
        const double savings = static_cast<double>(dense_bytes) / static_cast<double>(sparse_bytes);

        std::string dim_str = std::to_string(tc.rows) + "x" + std::to_string(tc.cols);
        std::string dens_mem_str = (dense_bytes >= 1024 * 1024) ?
            (std::to_string(dense_bytes / (1024 * 1024)) + " MB") :
            (std::to_string(dense_bytes / 1024) + " KB");
        std::string sp_mem_str = (sparse_bytes >= 1024 * 1024) ?
            (std::to_string(sparse_bytes / (1024 * 1024)) + " MB") :
            (std::to_string(sparse_bytes / 1024) + " KB");

        std::cout << std::left << std::setw(14) << dim_str
                  << std::setw(12) << (std::to_string(static_cast<int>(tc.target_sparsity * 100)) + "%")
                  << std::setw(12) << csr.non_zeros()
                  << std::setw(14) << dens_mem_str
                  << std::setw(14) << sp_mem_str
                  << std::fixed << std::setprecision(1) << savings << "x\n";
    }
}

void benchmark_spmv_execution_time() {
    std::cout << "\n=========================================================\n";
    std::cout << "  BENCHMARK 2: Matrix-Vector Multiplication (SpMV vs Dense)\n";
    std::cout << "=========================================================\n\n";

    constexpr std::size_t N = 800;
    constexpr double sparsity = 0.98;
    constexpr std::size_t iterations = 200;

    std::cout << "Matrix size: " << N << "x" << N << " (Sparsity: " << (sparsity * 100.0) << "%)\n";
    std::cout << "Averaged over " << iterations << " iterations:\n\n";

    std::mt19937_64 rng(12345);
    std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
    std::uniform_real_distribution<double> val_dist(-5.0, 5.0);

    vortex::Matrix<double> dense(N, N, 0.0);
    std::vector<vortex::Triplet<double>> triplets;
    const double density = 1.0 - sparsity;

    for (std::size_t r = 0; r < N; ++r) {
        for (std::size_t c = 0; c < N; ++c) {
            if (prob_dist(rng) < density) {
                double val = val_dist(rng);
                dense(r, c) = val;
                triplets.emplace_back(r, c, val);
            }
        }
    }

    auto csr = vortex::CSRMatrix<double>::from_triplets(N, N, triplets);
    auto csc = csr.to_csc();

    std::vector<double> x(N, 1.0);

    // 1. Dense Matrix-Vector
    double total_dense_ms = 0.0;
    std::vector<double> y_dense(N, 0.0);
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        for (std::size_t it = 0; it < iterations; ++it) {
            for (std::size_t r = 0; r < N; ++r) {
                double sum = 0.0;
                for (std::size_t c = 0; c < N; ++c) {
                    sum += dense(r, c) * x[c];
                }
                y_dense[r] = sum;
            }
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> dur = t1 - t0;
        total_dense_ms = dur.count();
    }
    double avg_dense_us = (total_dense_ms / iterations) * 1000.0;

    // 2. CSR SpMV
    double total_csr_ms = 0.0;
    std::vector<double> y_csr;
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        for (std::size_t it = 0; it < iterations; ++it) {
            y_csr = csr.multiply(x);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> dur = t1 - t0;
        total_csr_ms = dur.count();
    }
    double avg_csr_us = (total_csr_ms / iterations) * 1000.0;

    // 3. CSC SpMV
    double total_csc_ms = 0.0;
    std::vector<double> y_csc;
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        for (std::size_t it = 0; it < iterations; ++it) {
            y_csc = csc.multiply(x);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> dur = t1 - t0;
        total_csc_ms = dur.count();
    }
    double avg_csc_us = (total_csc_ms / iterations) * 1000.0;

    std::cout << "-> Dense GEMV:  " << std::fixed << std::setprecision(2) << avg_dense_us << " us / mult\n";
    std::cout << "-> CSR SpMV:    " << std::fixed << std::setprecision(2) << avg_csr_us << " us / mult ("
              << (avg_dense_us / avg_csr_us) << "x speedup)\n";
    std::cout << "-> CSC SpMV:    " << std::fixed << std::setprecision(2) << avg_csc_us << " us / mult ("
              << (avg_dense_us / avg_csc_us) << "x speedup)\n";
}

void benchmark_sparse_pricing() {
    std::cout << "\n=========================================================\n";
    std::cout << "  BENCHMARK 3: Simplex Pricing (Dense vs CSC Column Dot) \n";
    std::cout << "=========================================================\n\n";

    constexpr std::size_t M = 400;   // 400 constraints
    constexpr std::size_t N = 1500;  // 1500 variables
    constexpr double sparsity = 0.97;
    constexpr std::size_t iterations = 500;

    std::cout << "Constraint matrix: " << M << " rows x " << N << " cols (" << (sparsity * 100.0) << "% sparse)\n";
    std::cout << "Evaluating reduced costs over " << iterations << " iterations:\n\n";

    std::mt19937_64 rng(999);
    std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
    std::uniform_real_distribution<double> val_dist(-2.0, 2.0);

    vortex::Matrix<double> A_dense(M, N, 0.0);
    std::vector<vortex::Triplet<double>> triplets;
    const double density = 1.0 - sparsity;

    for (std::size_t r = 0; r < M; ++r) {
        for (std::size_t c = 0; c < N; ++c) {
            if (prob_dist(rng) < density) {
                double val = val_dist(rng);
                A_dense(r, c) = val;
                triplets.emplace_back(r, c, val);
            }
        }
    }

    auto A_csc = vortex::CSCMatrix<double>::from_triplets(M, N, triplets);
    std::vector<double> c_std(N, 1.0);
    std::vector<double> pi(M, 0.5);

    // Dense pricing: c_bar_j = c_j - sum_i pi_i * A_ij
    double dense_pricing_ms = 0.0;
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        for (std::size_t it = 0; it < iterations; ++it) {
            std::vector<double> red_costs(N, 0.0);
            for (std::size_t j = 0; j < N; ++j) {
                double dot = 0.0;
                for (std::size_t i = 0; i < M; ++i) {
                    dot += pi[i] * A_dense(i, j);
                }
                red_costs[j] = c_std[j] - dot;
            }
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> dur = t1 - t0;
        dense_pricing_ms = dur.count();
    }

    // Sparse pricing using CSC column_dot
    vortex::SparseRevisedSimplexSolver solver;
    double sparse_pricing_ms = 0.0;
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        for (std::size_t it = 0; it < iterations; ++it) {
            auto red_costs = solver.compute_sparse_reduced_costs(c_std, A_csc, pi);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> dur = t1 - t0;
        sparse_pricing_ms = dur.count();
    }

    double dense_us_per_iter = (dense_pricing_ms / iterations) * 1000.0;
    double sparse_us_per_iter = (sparse_pricing_ms / iterations) * 1000.0;

    std::cout << "-> Dense Pricing:   " << std::fixed << std::setprecision(2) << dense_us_per_iter << " us / iteration\n";
    std::cout << "-> Sparse Pricing:  " << std::fixed << std::setprecision(2) << sparse_us_per_iter << " us / iteration ("
              << (dense_us_per_iter / sparse_us_per_iter) << "x speedup)\n";
}

int main() {
    std::cout << "=========================================================\n";
    std::cout << "       VORTEX Sparse Engine Performance Benchmark        \n";
    std::cout << "=========================================================\n\n";

    benchmark_memory_footprint();
    benchmark_spmv_execution_time();
    benchmark_sparse_pricing();

    std::cout << "\n=========================================================\n";
    std::cout << "   Sparse Engine Benchmarks Completed Successfully!      \n";
    std::cout << "=========================================================\n";

    return 0;
}
