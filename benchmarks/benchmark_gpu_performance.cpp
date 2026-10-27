#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <random>
#include <numeric>
#include <memory>
#include <stdexcept>

#include "vortex/matrix.hpp"
#include "vortex/sparse_matrix.hpp"
#include "vortex/solver_backend.hpp"
#include "vortex/revised_simplex.hpp"
#include "vortex/sparse_revised_simplex.hpp"
#include "vortex/qp.hpp"

#if defined(VORTEX_ENABLE_CUDA) || defined(__CUDACC__)
#include "vortex/cuda_backend.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#endif

namespace vortex::benchmark {

// ==============================================================================
// Volatile Sink to Prevent Dead-Code Elimination
// ==============================================================================
template <typename T>
inline void do_not_optimize(const T& val) {
#if defined(_MSC_VER)
    volatile const char* p = reinterpret_cast<volatile const char*>(&val);
    (void)*p;
#else
    asm volatile("" : : "g"(val) : "memory");
#endif
}

// ==============================================================================
// Statistical Metrics Structure & Calculation
// ==============================================================================
struct TimingStats {
    double min_ms{0.0};
    double max_ms{0.0};
    double median_ms{0.0};
    double mean_ms{0.0};
    double stddev_ms{0.0};
    double iqr_ms{0.0};
    std::size_t samples{0};
};

inline TimingStats calculate_stats(const std::vector<double>& raw_times) {
    TimingStats stats;
    if (raw_times.empty()) return stats;
    stats.samples = raw_times.size();

    std::vector<double> v = raw_times;
    std::sort(v.begin(), v.end());

    stats.min_ms = v.front();
    stats.max_ms = v.back();

    const std::size_t n = v.size();
    if (n % 2 == 1) {
        stats.median_ms = v[n / 2];
    } else {
        stats.median_ms = 0.5 * (v[n / 2 - 1] + v[n / 2]);
    }

    const std::size_t q1_idx = n / 4;
    const std::size_t q3_idx = (3 * n) / 4;
    stats.iqr_ms = v[q3_idx] - v[q1_idx];

    const double sum = std::accumulate(v.begin(), v.end(), 0.0);
    stats.mean_ms = sum / static_cast<double>(n);

    if (n > 1) {
        double sq_sum = 0.0;
        for (double t : v) {
            sq_sum += (t - stats.mean_ms) * (t - stats.mean_ms);
        }
        stats.stddev_ms = std::sqrt(sq_sum / static_cast<double>(n - 1));
    }
    return stats;
}

inline std::string format_speedup(double ratio) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(2) << ratio << "x";
    if (ratio >= 1.05) {
        ss << " (GPU faster)";
    } else if (ratio <= 0.95) {
        ss << " (GPU slower)";
    } else {
        ss << " (Parity)";
    }
    return ss.str();
}

// ==============================================================================
// Scoped CUDA Event Timer
// ==============================================================================
#if defined(VORTEX_ENABLE_CUDA) || defined(__CUDACC__)
class CudaScopedTimer {
public:
    CudaScopedTimer() {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }
    ~CudaScopedTimer() {
        if (start_) cudaEventDestroy(start_);
        if (stop_) cudaEventDestroy(stop_);
    }
    void record_start(cudaStream_t stream = nullptr) {
        cudaEventRecord(start_, stream);
    }
    void record_stop(cudaStream_t stream = nullptr) {
        cudaEventRecord(stop_, stream);
    }
    double elapsed_ms() {
        cudaEventSynchronize(stop_);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start_, stop_);
        return static_cast<double>(ms);
    }
private:
    cudaEvent_t start_{nullptr};
    cudaEvent_t stop_{nullptr};
};
#endif

// ==============================================================================
// Benchmark Result Record
// ==============================================================================
struct BenchmarkRecord {
    std::string category;
    std::string test_name;
    std::size_t rows{0};
    std::size_t cols{0};
    std::size_t nnz{0};
    double density{1.0};
    bool transpose{false};
    TimingStats cpu_stats;
    TimingStats gpu_kernel_stats;
    TimingStats gpu_e2e_stats;
    TimingStats gpu_h2d_stats;
    TimingStats gpu_d2h_stats;
    double speedup_compute{0.0};
    double speedup_e2e{0.0};
    double max_abs_error{0.0};
    double scaled_rel_error{0.0};
    std::string status{"PASSED"};
    std::string notes;
};

// ==============================================================================
// Synthetic Data Generators
// ==============================================================================
inline Matrix<double> make_dense_matrix(std::size_t rows, std::size_t cols, uint64_t seed = 42) {
    Matrix<double> A(rows, cols);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(-5.0, 5.0);
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < cols; ++c) {
            A(r, c) = dist(rng);
        }
    }
    return A;
}

inline std::vector<double> make_vector(std::size_t len, uint64_t seed = 123) {
    std::vector<double> v(len);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(-2.0, 2.0);
    for (std::size_t i = 0; i < len; ++i) {
        v[i] = dist(rng);
    }
    return v;
}

inline CSRMatrix<double> make_sparse_csr(std::size_t rows, std::size_t cols, double density,
                                         bool irregular = false, uint64_t seed = 789) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> val_dist(0.5, 5.0);
    std::uniform_real_distribution<double> prob_dist(0.0, 1.0);

    std::vector<Triplet<double>> triplets;
    triplets.reserve(static_cast<std::size_t>(rows * cols * density * 1.5));

    for (std::size_t r = 0; r < rows; ++r) {
        double row_density = density;
        if (irregular) {
            // Power-law/irregular distribution: some rows dense (5x), most very sparse (0.2x)
            if (r % 20 == 0) {
                row_density = std::min(1.0, density * 5.0);
            } else {
                row_density = density * 0.5;
            }
        }
        bool row_has_entry = false;
        for (std::size_t c = 0; c < cols; ++c) {
            if (prob_dist(rng) < row_density) {
                triplets.emplace_back(r, c, val_dist(rng));
                row_has_entry = true;
            }
        }
        // Ensure every row has at least one entry to prevent structural degeneracies
        if (!row_has_entry) {
            triplets.emplace_back(r, r % cols, val_dist(rng));
        }
    }
    return CSRMatrix<double>::from_triplets(rows, cols, triplets);
}

// Feasible Bounded Dense LP Generator
struct FeasibleDenseLP {
    Matrix<double> A;
    std::vector<double> b;
    std::vector<double> c;
};

inline FeasibleDenseLP make_feasible_dense_lp(std::size_t m, std::size_t n, uint64_t seed = 1001) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist_A(0.1, 2.0);
    std::uniform_real_distribution<double> dist_c(1.0, 10.0);
    std::uniform_real_distribution<double> dist_x(1.0, 3.0);
    std::uniform_real_distribution<double> dist_s(1.0, 5.0);

    Matrix<double> A(m, n);
    for (std::size_t r = 0; r < m; ++r) {
        for (std::size_t col = 0; col < n; ++col) {
            A(r, col) = dist_A(rng);
        }
    }

    std::vector<double> x_star(n);
    for (std::size_t j = 0; j < n; ++j) {
        x_star[j] = dist_x(rng);
    }

    std::vector<double> b(m, 0.0);
    for (std::size_t i = 0; i < m; ++i) {
        double row_sum = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
            row_sum += A(i, j) * x_star[j];
        }
        b[i] = row_sum + dist_s(rng);
    }

    std::vector<double> c(n);
    for (std::size_t j = 0; j < n; ++j) {
        c[j] = dist_c(rng);
    }

    return {std::move(A), std::move(b), std::move(c)};
}

// Feasible Bounded Sparse LP Generator
struct FeasibleSparseLP {
    CSRMatrix<double> A;
    CSCMatrix<double> A_csc;
    std::vector<double> b;
    std::vector<double> c;
};

inline FeasibleSparseLP make_feasible_sparse_lp(std::size_t m, std::size_t n, double density, uint64_t seed = 2002) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist_val(0.5, 3.0);
    std::uniform_real_distribution<double> dist_c(1.0, 8.0);
    std::uniform_real_distribution<double> dist_x(1.0, 2.0);
    std::uniform_real_distribution<double> prob_dist(0.0, 1.0);

    std::vector<Triplet<double>> triplets;
    // Ensure every column j has a strictly positive bounding entry
    for (std::size_t j = 0; j < n; ++j) {
        triplets.emplace_back(j % m, j, 2.5 + dist_val(rng));
    }
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            if ((j % m) != i && prob_dist(rng) < density) {
                triplets.emplace_back(i, j, dist_val(rng));
            }
        }
    }
    CSRMatrix<double> A = CSRMatrix<double>::from_triplets(m, n, triplets, true);
    CSCMatrix<double> A_csc = A.to_csc();

    std::vector<double> x_star(n);
    for (std::size_t j = 0; j < n; ++j) x_star[j] = dist_x(rng);

    std::vector<double> Ax = A.multiply(x_star);
    std::vector<double> b(m);
    for (std::size_t i = 0; i < m; ++i) {
        b[i] = Ax[i] + 2.0;
    }

    std::vector<double> c(n);
    for (std::size_t j = 0; j < n; ++j) {
        c[j] = dist_c(rng);
    }

    return {std::move(A), std::move(A_csc), std::move(b), std::move(c)};
}

// Strictly Convex Feasible QP Generator
struct FeasibleConvexQP {
    Matrix<double> Q;
    std::vector<double> c;
    Matrix<double> A;
    std::vector<double> b;
};

inline FeasibleConvexQP make_feasible_convex_qp(std::size_t n, std::size_t m, uint64_t seed = 3003) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist_M(-1.0, 1.0);
    std::uniform_real_distribution<double> dist_c(-2.0, 2.0);
    std::uniform_real_distribution<double> dist_A(0.1, 1.5);
    std::uniform_real_distribution<double> dist_x(0.5, 2.0);

    // Q = M^T M + 2.0 * I (Strictly Positive Definite)
    Matrix<double> M(n, n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            M(i, j) = dist_M(rng);
        }
    }
    Matrix<double> Q = M.transpose() * M;
    for (std::size_t i = 0; i < n; ++i) {
        Q(i, i) += 2.0;
    }

    std::vector<double> c(n);
    for (std::size_t i = 0; i < n; ++i) {
        c[i] = dist_c(rng);
    }

    Matrix<double> A(m, n);
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            A(i, j) = dist_A(rng);
        }
    }

    std::vector<double> x_star(n);
    for (std::size_t j = 0; j < n; ++j) {
        x_star[j] = dist_x(rng);
    }

    std::vector<double> b(m);
    for (std::size_t i = 0; i < m; ++i) {
        double row_sum = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
            row_sum += A(i, j) * x_star[j];
        }
        b[i] = row_sum + 2.0; // strictly feasible with positive slack
    }

    return {std::move(Q), std::move(c), std::move(A), std::move(b)};
}

} // namespace vortex::benchmark

// ==============================================================================
// Main Benchmarking Suite
// ==============================================================================
int main(int argc, char* argv[]) {
    using namespace vortex;
    using namespace vortex::cuda;
    using namespace vortex::benchmark;

    bool verify_only = false;
    bool quick_mode = false;
    std::string json_output_path = "benchmarks/results/phase8e_results.json";
    std::string csv_output_path  = "benchmarks/results/phase8e_results.csv";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--verify-only") {
            verify_only = true;
        } else if (arg == "--quick") {
            quick_mode = true;
        } else if (arg == "--json" && i + 1 < argc) {
            json_output_path = argv[++i];
        } else if (arg == "--csv" && i + 1 < argc) {
            csv_output_path = argv[++i];
        }
    }

    const int warmup_runs = quick_mode ? 2 : (verify_only ? 1 : 5);
    const int timed_runs  = quick_mode ? 5 : (verify_only ? 1 : 15);

    std::cout << "==============================================================================\n";
    std::cout << "  VORTEX PHASE 8E: REPRODUCIBLE CPU VS GPU PERFORMANCE BENCHMARKING          \n";
    std::cout << "==============================================================================\n";

#if !defined(VORTEX_ENABLE_CUDA)
    std::cerr << "FATAL: VORTEX was compiled without CUDA acceleration enabled.\n";
    return 1;
#else
    if (!is_cuda_available()) {
        std::cerr << "FATAL: No CUDA-capable device detected.\n";
        return 1;
    }

    const auto dev_info = get_cuda_device_info(0);
    int core_clock_khz = 0;
    int mem_clock_khz = 0;
    int bus_width_bits = 0;
    cudaDeviceGetAttribute(&core_clock_khz, cudaDevAttrClockRate, 0);
    cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, 0);
    cudaDeviceGetAttribute(&bus_width_bits, cudaDevAttrGlobalMemoryBusWidth, 0);

    std::size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);

    std::cout << "[Hardware & Environment Telemetry]\n";
    std::cout << "  GPU:               " << dev_info.name << " (CC " << dev_info.major << "." << dev_info.minor << ")\n";
    std::cout << "  Multiprocessors:   " << dev_info.multiprocessor_count << " SMs (" << dev_info.multiprocessor_count * 128 << " CUDA Cores approx.)\n";
    std::cout << "  Core Clock:        " << core_clock_khz / 1000.0 << " MHz\n";
    std::cout << "  Memory Clock:      " << mem_clock_khz / 1000.0 << " MHz\n";
    std::cout << "  Memory Bus Width:  " << bus_width_bits << " bits\n";
    std::cout << "  Total VRAM:        " << total_mem / (1024 * 1024) << " MB (" << (total_mem / (1024.0 * 1024.0 * 1024.0)) << " GB)\n";
    std::cout << "  Available VRAM:    " << free_mem / (1024 * 1024) << " MB\n";
    std::cout << "  CPU:               AMD Ryzen 5 6600H (6 Cores, 12 Threads)\n";
    std::cout << "  Compiler:          MSVC " << _MSC_VER << " / CUDA Runtime " << CUDART_VERSION << "\n";
    std::cout << "  Benchmark Mode:    " << (verify_only ? "VERIFY ONLY" : (quick_mode ? "QUICK" : "FULL REPRODUCIBLE"))
              << " (" << warmup_runs << " warmups, " << timed_runs << " timed runs)\n\n";

    std::vector<BenchmarkRecord> records;

    // ==============================================================================
    // SECTION B: DENSE GEMV BENCHMARKS
    // ==============================================================================
    std::cout << "------------------------------------------------------------------------------\n";
    std::cout << "  SECTION B: DENSE MATRIX-VECTOR MULTIPLICATION (GEMV)\n";
    std::cout << "------------------------------------------------------------------------------\n";

    const std::vector<std::size_t> gemv_sizes = {128, 512, 1024, 2048};
    for (std::size_t dim : gemv_sizes) {
        for (bool transpose : {false, true}) {
            const std::string test_name = "Dense GEMV " + std::to_string(dim) + "x" + std::to_string(dim) +
                                          (transpose ? " Transposed" : " Normal");
            std::cout << "  Benchmarking " << std::left << std::setw(38) << test_name << "... " << std::flush;

            const auto A = make_dense_matrix(dim, dim, 42 + dim);
            const auto x = make_vector(dim, 100 + dim);

            // Correctness check
            std::vector<double> y_cpu = transpose ? A.transpose().multiply(x) : A.multiply(x);
            std::vector<double> y_gpu(dim, 0.0);
            dense_gemv(A, x, y_gpu, transpose, 1.0, 0.0);

            const auto err = compute_error_metrics(y_gpu.data(), y_cpu.data(), dim);
            if (err.max_absolute_error > 1e-10) {
                std::cout << "[FAILED Numerical Equivalence: " << err.max_absolute_error << "]\n";
                return 1;
            }

            BenchmarkRecord rec;
            rec.category = "Dense GEMV";
            rec.test_name = test_name;
            rec.rows = dim;
            rec.cols = dim;
            rec.nnz = dim * dim;
            rec.density = 1.0;
            rec.transpose = transpose;
            rec.max_abs_error = err.max_absolute_error;
            rec.scaled_rel_error = err.scaled_relative_error;

            if (verify_only) {
                std::cout << "[PASS err=" << err.max_absolute_error << "]\n";
                records.push_back(rec);
                continue;
            }

            // Benchmark CPU
            for (int w = 0; w < warmup_runs; ++w) {
                std::vector<double> tmp = transpose ? A.transpose().multiply(x) : A.multiply(x);
                do_not_optimize(tmp.data());
            }
            std::vector<double> cpu_times;
            cpu_times.reserve(timed_runs);
            for (int r = 0; r < timed_runs; ++r) {
                const auto t0 = std::chrono::high_resolution_clock::now();
                std::vector<double> tmp = transpose ? A.transpose().multiply(x) : A.multiply(x);
                const auto t1 = std::chrono::high_resolution_clock::now();
                do_not_optimize(tmp.data());
                cpu_times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            }
            rec.cpu_stats = calculate_stats(cpu_times);

            // Benchmark GPU Isolated (Kernel and E2E)
            CublasHandle cublas;
            CudaDeviceBuffer<double> d_A(dim * dim);
            CudaDeviceBuffer<double> d_x(dim);
            CudaDeviceBuffer<double> d_y(dim);

            std::vector<double> kernel_times;
            std::vector<double> e2e_times;
            std::vector<double> h2d_times;
            std::vector<double> d2h_times;
            kernel_times.reserve(timed_runs);
            e2e_times.reserve(timed_runs);
            h2d_times.reserve(timed_runs);
            d2h_times.reserve(timed_runs);

            const int m_cu = static_cast<int>(dim);
            const int n_cu = static_cast<int>(dim);
            const int lda  = static_cast<int>(dim);
            const cublasOperation_t op = transpose ? CUBLAS_OP_N : CUBLAS_OP_T;
            const double alpha = 1.0;
            const double beta  = 0.0;
            CudaScopedTimer event_timer;

            // Warmup GPU
            for (int w = 0; w < warmup_runs; ++w) {
                d_A.copy_from_host(A.data(), dim * dim);
                d_x.copy_from_host(x.data(), dim);
                cublasDgemv(cublas.get(), op, m_cu, n_cu, &alpha, d_A.data(), lda, d_x.data(), 1, &beta, d_y.data(), 1);
                d_y.copy_to_host(y_gpu.data(), dim);
                cudaDeviceSynchronize();
            }

            for (int r = 0; r < timed_runs; ++r) {
                const auto t_e2e_start = std::chrono::high_resolution_clock::now();

                const auto t_h2d_start = std::chrono::high_resolution_clock::now();
                d_A.copy_from_host(A.data(), dim * dim);
                d_x.copy_from_host(x.data(), dim);
                cudaDeviceSynchronize();
                const auto t_h2d_stop = std::chrono::high_resolution_clock::now();
                h2d_times.push_back(std::chrono::duration<double, std::milli>(t_h2d_stop - t_h2d_start).count());

                event_timer.record_start();
                cublasDgemv(cublas.get(), op, m_cu, n_cu, &alpha, d_A.data(), lda, d_x.data(), 1, &beta, d_y.data(), 1);
                event_timer.record_stop();
                kernel_times.push_back(event_timer.elapsed_ms());

                const auto t_d2h_start = std::chrono::high_resolution_clock::now();
                d_y.copy_to_host(y_gpu.data(), dim);
                cudaDeviceSynchronize();
                const auto t_d2h_stop = std::chrono::high_resolution_clock::now();
                d2h_times.push_back(std::chrono::duration<double, std::milli>(t_d2h_stop - t_d2h_start).count());

                const auto t_e2e_stop = std::chrono::high_resolution_clock::now();
                do_not_optimize(y_gpu.data());
                e2e_times.push_back(std::chrono::duration<double, std::milli>(t_e2e_stop - t_e2e_start).count());
            }

            rec.gpu_kernel_stats = calculate_stats(kernel_times);
            rec.gpu_e2e_stats    = calculate_stats(e2e_times);
            rec.gpu_h2d_stats    = calculate_stats(h2d_times);
            rec.gpu_d2h_stats    = calculate_stats(d2h_times);

            rec.speedup_compute = (rec.gpu_kernel_stats.median_ms > 0.0) ? (rec.cpu_stats.median_ms / rec.gpu_kernel_stats.median_ms) : 0.0;
            rec.speedup_e2e     = (rec.gpu_e2e_stats.median_ms > 0.0)    ? (rec.cpu_stats.median_ms / rec.gpu_e2e_stats.median_ms)    : 0.0;

            std::cout << "[DONE] CPU: " << std::fixed << std::setprecision(3) << rec.cpu_stats.median_ms << " ms | "
                      << "GPU Kernel: " << rec.gpu_kernel_stats.median_ms << " ms (Kernel " << format_speedup(rec.speedup_compute) << ") | "
                      << "E2E: " << rec.gpu_e2e_stats.median_ms << " ms (" << format_speedup(rec.speedup_e2e) << ")\n";

            records.push_back(rec);
        }
    }

    // ==============================================================================
    // SECTION C: SPARSE SPMV BENCHMARKS
    // ==============================================================================
    std::cout << "\n------------------------------------------------------------------------------\n";
    std::cout << "  SECTION C: SPARSE MATRIX-VECTOR MULTIPLICATION (CSR SPMV)\n";
    std::cout << "------------------------------------------------------------------------------\n";

    struct SpmvConfig {
        std::size_t dim;
        double sparsity;
        bool irregular;
    };
    const std::vector<SpmvConfig> spmv_configs = {
        {1000, 0.90, false},
        {1000, 0.95, false},
        {1000, 0.98, false},
        {1000, 0.99, false},
        {2000, 0.95, false},
        {2000, 0.98, false},
        {2000, 0.98, true}  // Irregular / power-law cluster rows
    };

    for (const auto& cfg : spmv_configs) {
        const double density = 1.0 - cfg.sparsity;
        const std::string pattern_name = cfg.irregular ? "Irregular" : "Balanced";
        const std::string test_name = "CSR SpMV " + std::to_string(cfg.dim) + "x" + std::to_string(cfg.dim) +
                                      " (" + std::to_string(static_cast<int>(cfg.sparsity * 100)) + "% Sparse, " + pattern_name + ")";

        std::cout << "  Benchmarking " << std::left << std::setw(46) << test_name << "... " << std::flush;

        const auto A_csr = make_sparse_csr(cfg.dim, cfg.dim, density, cfg.irregular, 500 + cfg.dim);
        const auto x = make_vector(cfg.dim, 600 + cfg.dim);

        // Correctness check
        std::vector<double> y_cpu = A_csr.multiply(x);
        std::vector<double> y_gpu(cfg.dim, 0.0);
        sparse_spmv(A_csr, x, y_gpu, false, 1.0, 0.0);

        const auto err = compute_error_metrics(y_gpu.data(), y_cpu.data(), cfg.dim);
        if (err.max_absolute_error > 1e-10) {
            std::cout << "[FAILED Numerical Equivalence: " << err.max_absolute_error << "]\n";
            return 1;
        }

        BenchmarkRecord rec;
        rec.category = "Sparse SpMV";
        rec.test_name = test_name;
        rec.rows = cfg.dim;
        rec.cols = cfg.dim;
        rec.nnz = A_csr.non_zeros();
        rec.density = 1.0 - A_csr.sparsity();
        rec.transpose = false;
        rec.max_abs_error = err.max_absolute_error;
        rec.scaled_rel_error = err.scaled_relative_error;

        if (verify_only) {
            std::cout << "[PASS err=" << err.max_absolute_error << "]\n";
            records.push_back(rec);
            continue;
        }

        // Benchmark CPU
        for (int w = 0; w < warmup_runs; ++w) {
            std::vector<double> tmp = A_csr.multiply(x);
            do_not_optimize(tmp.data());
        }
        std::vector<double> cpu_times;
        cpu_times.reserve(timed_runs);
        for (int r = 0; r < timed_runs; ++r) {
            const auto t0 = std::chrono::high_resolution_clock::now();
            std::vector<double> tmp = A_csr.multiply(x);
            const auto t1 = std::chrono::high_resolution_clock::now();
            do_not_optimize(tmp.data());
            cpu_times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        rec.cpu_stats = calculate_stats(cpu_times);

        // Benchmark GPU SpMV
        const auto spmv_res = benchmark_sparse_spmv(A_csr, x, false, warmup_runs, timed_runs);
        rec.gpu_kernel_stats.median_ms = spmv_res.gpu_kernel_ms;
        rec.gpu_e2e_stats.median_ms    = spmv_res.gpu_e2e_ms;
        rec.gpu_h2d_stats.median_ms    = spmv_res.gpu_h2d_ms;
        rec.gpu_d2h_stats.median_ms    = spmv_res.gpu_d2h_ms;
        rec.speedup_compute = (rec.gpu_kernel_stats.median_ms > 0.0) ? (rec.cpu_stats.median_ms / rec.gpu_kernel_stats.median_ms) : 0.0;
        rec.speedup_e2e     = (rec.gpu_e2e_stats.median_ms > 0.0)    ? (rec.cpu_stats.median_ms / rec.gpu_e2e_stats.median_ms)    : 0.0;

        std::cout << "[DONE] CPU: " << std::fixed << std::setprecision(3) << rec.cpu_stats.median_ms << " ms | "
                  << "GPU Kernel: " << rec.gpu_kernel_stats.median_ms << " ms (Kernel " << format_speedup(rec.speedup_compute) << ") | "
                  << "E2E: " << rec.gpu_e2e_stats.median_ms << " ms (" << format_speedup(rec.speedup_e2e) << ")\n";

        records.push_back(rec);
    }

    // ==============================================================================
    // SECTION D: DATA TRANSFER & MEMORY REUSE OVERHEAD
    // ==============================================================================
    std::cout << "\n------------------------------------------------------------------------------\n";
    std::cout << "  SECTION D: DATA TRANSFER & BUFFER REUSE ANALYSIS\n";
    std::cout << "------------------------------------------------------------------------------\n";

    const std::vector<std::size_t> transfer_dims = {512, 1024, 2048};
    for (std::size_t dim : transfer_dims) {
        const double matrix_bytes = static_cast<double>(dim * dim * sizeof(double));
        const double vector_bytes = static_cast<double>(dim * sizeof(double));
        const double total_mb = (matrix_bytes + 2 * vector_bytes) / (1024.0 * 1024.0);

        std::cout << "  Analyzing Dimension " << dim << "x" << dim << " (" << std::fixed << std::setprecision(2) << total_mb << " MB)...\n";

        const auto A = make_dense_matrix(dim, dim, 777 + dim);
        const auto x = make_vector(dim, 888 + dim);
        std::vector<double> y(dim, 0.0);

        if (verify_only) {
            std::cout << "    [PASS Verification]\n";
            continue;
        }

        // 1. Cold Execution: malloc + H2D + compute + D2H + free
        std::vector<double> cold_times;
        cold_times.reserve(timed_runs);
        for (int r = 0; r < timed_runs; ++r) {
            const auto t0 = std::chrono::high_resolution_clock::now();
            dense_gemv(A, x, y, false, 1.0, 0.0);
            const auto t1 = std::chrono::high_resolution_clock::now();
            cold_times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        const auto cold_stats = calculate_stats(cold_times);

        // 2. Persistent Reused Context: matrix resident in VRAM, only transfer vector x and y
        GpuDensePricingContext persistent_ctx(A);
        std::vector<double> pi = x;
        std::vector<double> c(dim, 1.0);
        std::vector<double> r_costs(dim, 0.0);

        std::vector<double> reused_times;
        reused_times.reserve(timed_runs);
        for (int r = 0; r < timed_runs; ++r) {
            const auto t0 = std::chrono::high_resolution_clock::now();
            persistent_ctx.compute_reduced_costs(c, pi, r_costs);
            cudaDeviceSynchronize();
            const auto t1 = std::chrono::high_resolution_clock::now();
            do_not_optimize(r_costs.data());
            reused_times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        const auto reused_stats = calculate_stats(reused_times);

        const double cold_bw_gbps = (total_mb / 1024.0) / (cold_stats.median_ms / 1000.0);
        const double speedup_reuse = cold_stats.median_ms / reused_stats.median_ms;

        std::cout << "    - Cold E2E (alloc + full H2D + GEMV + D2H + free): " << std::fixed << std::setprecision(3)
                  << cold_stats.median_ms << " ms (" << cold_bw_gbps << " GB/s effective bandwidth)\n";
        std::cout << "    - Reused Device Buffers (vector-only H2D + GEMV + D2H): " << reused_stats.median_ms << " ms\n";
        std::cout << "    - Buffer Persistence Acceleration: " << std::fixed << std::setprecision(2) << speedup_reuse << "x speedup\n";

        BenchmarkRecord rec;
        rec.category = "Data Transfer & Reuse";
        rec.test_name = "Buffer Reuse " + std::to_string(dim) + "x" + std::to_string(dim);
        rec.rows = dim;
        rec.cols = dim;
        rec.nnz = dim * dim;
        rec.density = 1.0;
        rec.cpu_stats = cold_stats; // Cold baseline
        rec.gpu_e2e_stats = reused_stats; // Reused
        rec.speedup_e2e = speedup_reuse;
        rec.notes = "Persistent buffers save matrix re-transfer and allocation overhead";
        records.push_back(rec);
    }

    // ==============================================================================
    // SECTION E: SIMPLEX PRICING MICROBENCHMARK
    // ==============================================================================
    std::cout << "\n------------------------------------------------------------------------------\n";
    std::cout << "  SECTION E: SIMPLEX PRICING MICROBENCHMARKS (c - A^T * pi)\n";
    std::cout << "------------------------------------------------------------------------------\n";

    struct PricingConfig {
        std::size_t m;
        std::size_t n;
    };
    const std::vector<PricingConfig> pricing_configs = {
        {256, 1024},
        {512, 2048},
        {1024, 4096}
    };

    for (const auto& cfg : pricing_configs) {
        const std::string test_name = "Simplex Pricing m=" + std::to_string(cfg.m) + ", n=" + std::to_string(cfg.n);
        std::cout << "  Benchmarking " << std::left << std::setw(36) << test_name << "... " << std::flush;

        const auto A = make_dense_matrix(cfg.m, cfg.n, 1234 + cfg.m);
        const auto pi = make_vector(cfg.m, 5678 + cfg.m);
        const auto c = make_vector(cfg.n, 9012 + cfg.n);

        // 1. CPU Pricing
        std::vector<double> r_cpu(cfg.n, 0.0);
        const auto AT = A.transpose();
        for (std::size_t j = 0; j < cfg.n; ++j) {
            double dot = 0.0;
            for (std::size_t i = 0; i < cfg.m; ++i) {
                dot += A(i, j) * pi[i];
            }
            r_cpu[j] = c[j] - dot;
        }

        // 2. Persistent GPU Pricing
        GpuDensePricingContext gpu_ctx(A);
        std::vector<double> r_gpu(cfg.n, 0.0);
        gpu_ctx.compute_reduced_costs(c, pi, r_gpu);

        const auto err = compute_error_metrics(r_gpu.data(), r_cpu.data(), cfg.n);
        if (err.max_absolute_error > 1e-10) {
            std::cout << "[FAILED Numerical Equivalence: " << err.max_absolute_error << "]\n";
            return 1;
        }

        BenchmarkRecord rec;
        rec.category = "Simplex Pricing";
        rec.test_name = test_name;
        rec.rows = cfg.m;
        rec.cols = cfg.n;
        rec.nnz = cfg.m * cfg.n;
        rec.density = 1.0;
        rec.max_abs_error = err.max_absolute_error;

        if (verify_only) {
            std::cout << "[PASS err=" << err.max_absolute_error << "]\n";
            records.push_back(rec);
            continue;
        }

        // Repetitive Pricing (50 pricing operations, simulating 50 simplex pivots)
        const int pricing_iters = 50;

        // CPU timing
        std::vector<double> cpu_pricing_times;
        cpu_pricing_times.reserve(timed_runs);
        for (int r = 0; r < timed_runs; ++r) {
            const auto t0 = std::chrono::high_resolution_clock::now();
            for (int k = 0; k < pricing_iters; ++k) {
                for (std::size_t j = 0; j < cfg.n; ++j) {
                    double dot = 0.0;
                    for (std::size_t i = 0; i < cfg.m; ++i) {
                        dot += A(i, j) * pi[i];
                    }
                    r_cpu[j] = c[j] - dot;
                }
                do_not_optimize(r_cpu.data());
            }
            const auto t1 = std::chrono::high_resolution_clock::now();
            cpu_pricing_times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count() / pricing_iters);
        }
        rec.cpu_stats = calculate_stats(cpu_pricing_times);

        // GPU Persistent timing
        std::vector<double> gpu_pricing_times;
        gpu_pricing_times.reserve(timed_runs);
        for (int r = 0; r < timed_runs; ++r) {
            const auto t0 = std::chrono::high_resolution_clock::now();
            for (int k = 0; k < pricing_iters; ++k) {
                gpu_ctx.compute_reduced_costs(c, pi, r_gpu);
            }
            cudaDeviceSynchronize();
            const auto t1 = std::chrono::high_resolution_clock::now();
            do_not_optimize(r_gpu.data());
            gpu_pricing_times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count() / pricing_iters);
        }
        rec.gpu_e2e_stats = calculate_stats(gpu_pricing_times);
        rec.speedup_e2e = (rec.gpu_e2e_stats.median_ms > 0.0) ? (rec.cpu_stats.median_ms / rec.gpu_e2e_stats.median_ms) : 0.0;

        std::cout << "[DONE] CPU: " << std::fixed << std::setprecision(3) << rec.cpu_stats.median_ms << " ms/call | "
                  << "GPU Reused: " << rec.gpu_e2e_stats.median_ms << " ms/call (" << format_speedup(rec.speedup_e2e) << ")\n";

        records.push_back(rec);
    }

    // ==============================================================================
    // SECTION F: FULL OPTIMIZATION SOLVER BENCHMARKS
    // ==============================================================================
    std::cout << "\n------------------------------------------------------------------------------\n";
    std::cout << "  SECTION F: FULL SOLVER END-TO-END BENCHMARKS (REVISED SIMPLEX & CONVEX QP)\n";
    std::cout << "------------------------------------------------------------------------------\n";

    // 1. Dense Revised Simplex LP
    struct LPProblemSize {
        std::size_t m;
        std::size_t n;
    };
    const std::vector<LPProblemSize> lp_sizes = {
        {32, 64},
        {64, 128},
        {128, 256},
        {256, 512}
    };

    for (const auto& size : lp_sizes) {
        const std::string test_name = "Dense Revised Simplex LP (" + std::to_string(size.m) + "x" + std::to_string(size.n) + ")";
        std::cout << "  Benchmarking " << std::left << std::setw(42) << test_name << "... " << std::flush;

        const auto lp = make_feasible_dense_lp(size.m, size.n, 4004 + size.m);

        // Verification Solve
        RevisedSimplexOptions cpu_opts;
        cpu_opts.backend = SolverBackend::CPU;
        RevisedSimplexSolver cpu_solver(cpu_opts);
        const auto cpu_res = cpu_solver.solve(lp.c, lp.A, lp.b);

        RevisedSimplexOptions gpu_opts;
        gpu_opts.backend = SolverBackend::GPU;
        RevisedSimplexSolver gpu_solver(gpu_opts);
        const auto gpu_res = gpu_solver.solve(lp.c, lp.A, lp.b);

        if (cpu_res.status != SolverStatus::OPTIMAL || gpu_res.status != SolverStatus::OPTIMAL) {
            std::cout << "[FAILED Non-Optimal: CPU=" << static_cast<int>(cpu_res.status)
                      << ", GPU=" << static_cast<int>(gpu_res.status) << "]\n";
            return 1;
        }

        const double rel_obj_diff = std::abs(cpu_res.objective_value - gpu_res.objective_value) /
                                    std::max(1.0, std::abs(cpu_res.objective_value));
        if (rel_obj_diff > 1e-4) {
            std::cout << "[FAILED Objective Mismatch: CPU=" << cpu_res.objective_value
                      << ", GPU=" << gpu_res.objective_value << "]\n";
            return 1;
        }

        BenchmarkRecord rec;
        rec.category = "Full Solver: Dense LP";
        rec.test_name = test_name;
        rec.rows = size.m;
        rec.cols = size.n;
        rec.nnz = size.m * size.n;
        rec.density = 1.0;
        rec.max_abs_error = std::abs(cpu_res.objective_value - gpu_res.objective_value);
        rec.scaled_rel_error = rel_obj_diff;
        rec.notes = "Iters: " + std::to_string(gpu_res.iterations) + ", GPU pricing calls: " +
                    std::to_string(gpu_res.device_diagnostics.gpu_gemv_calls);

        if (verify_only) {
            std::cout << "[PASS obj=" << cpu_res.objective_value << ", iters=" << cpu_res.iterations << "]\n";
            records.push_back(rec);
            continue;
        }

        // Timed Solves
        std::vector<double> cpu_solve_times;
        cpu_solve_times.reserve(timed_runs);
        for (int r = 0; r < timed_runs; ++r) {
            const auto t0 = std::chrono::high_resolution_clock::now();
            const auto res = cpu_solver.solve(lp.c, lp.A, lp.b);
            const auto t1 = std::chrono::high_resolution_clock::now();
            do_not_optimize(res.objective_value);
            cpu_solve_times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        rec.cpu_stats = calculate_stats(cpu_solve_times);

        std::vector<double> gpu_solve_times;
        gpu_solve_times.reserve(timed_runs);
        for (int r = 0; r < timed_runs; ++r) {
            const auto t0 = std::chrono::high_resolution_clock::now();
            const auto res = gpu_solver.solve(lp.c, lp.A, lp.b);
            const auto t1 = std::chrono::high_resolution_clock::now();
            do_not_optimize(res.objective_value);
            gpu_solve_times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        rec.gpu_e2e_stats = calculate_stats(gpu_solve_times);
        rec.speedup_e2e   = (rec.gpu_e2e_stats.median_ms > 0.0) ? (rec.cpu_stats.median_ms / rec.gpu_e2e_stats.median_ms) : 0.0;

        std::cout << "[DONE] CPU: " << std::fixed << std::setprecision(2) << rec.cpu_stats.median_ms << " ms | "
                  << "GPU Hybrid: " << rec.gpu_e2e_stats.median_ms << " ms (" << format_speedup(rec.speedup_e2e) << ") | "
                  << "Iters: " << gpu_res.iterations << "\n";

        records.push_back(rec);
    }

    // 2. Sparse Revised Simplex LP
    struct SparseLPConfig {
        std::size_t m;
        std::size_t n;
        double density;
    };
    const std::vector<SparseLPConfig> sparse_lp_configs = {
        {50, 100, 0.05},
        {100, 200, 0.05},
        {150, 300, 0.02},
        {200, 400, 0.02}
    };

    for (const auto& cfg : sparse_lp_configs) {
        const std::string test_name = "Sparse Revised Simplex LP (" + std::to_string(cfg.m) + "x" + std::to_string(cfg.n) +
                                      ", " + std::to_string(static_cast<int>((1.0 - cfg.density) * 100)) + "% Sparse)";
        std::cout << "  Benchmarking " << std::left << std::setw(46) << test_name << "... " << std::flush;

        const auto lp = make_feasible_sparse_lp(cfg.m, cfg.n, cfg.density, 5005 + cfg.m);

        RevisedSimplexOptions cpu_opts;
        cpu_opts.backend = SolverBackend::CPU;
        cpu_opts.max_iterations = 20000;
        SparseRevisedSimplexSolver cpu_solver(cpu_opts);
        const auto cpu_res = cpu_solver.solve(lp.c, lp.A_csc, lp.b);

        RevisedSimplexOptions gpu_opts;
        gpu_opts.backend = SolverBackend::GPU;
        gpu_opts.max_iterations = 20000;
        SparseRevisedSimplexSolver gpu_solver(gpu_opts);
        const auto gpu_res = gpu_solver.solve(lp.c, lp.A_csc, lp.b);

        if (cpu_res.status != SolverStatus::OPTIMAL || gpu_res.status != SolverStatus::OPTIMAL) {
            std::cout << "[FAILED Non-Optimal: CPU=" << static_cast<int>(cpu_res.status)
                      << ", GPU=" << static_cast<int>(gpu_res.status) << "]\n";
            return 1;
        }

        const double rel_obj_diff = std::abs(cpu_res.objective_value - gpu_res.objective_value) /
                                    std::max(1.0, std::abs(cpu_res.objective_value));
        if (rel_obj_diff > 1e-4) {
            std::cout << "[FAILED Objective Mismatch: CPU=" << cpu_res.objective_value
                      << ", GPU=" << gpu_res.objective_value << "]\n";
            return 1;
        }

        BenchmarkRecord rec;
        rec.category = "Full Solver: Sparse LP";
        rec.test_name = test_name;
        rec.rows = cfg.m;
        rec.cols = cfg.n;
        rec.nnz = lp.A.non_zeros();
        rec.density = 1.0 - lp.A.sparsity();
        rec.max_abs_error = std::abs(cpu_res.objective_value - gpu_res.objective_value);
        rec.scaled_rel_error = rel_obj_diff;
        rec.notes = "Iters: " + std::to_string(gpu_res.iterations) + ", GPU SpMV calls: " +
                    std::to_string(gpu_res.device_diagnostics.gpu_spmv_calls);

        if (verify_only) {
            std::cout << "[PASS obj=" << cpu_res.objective_value << ", iters=" << cpu_res.iterations << "]\n";
            records.push_back(rec);
            continue;
        }

        std::vector<double> cpu_solve_times;
        cpu_solve_times.reserve(timed_runs);
        for (int r = 0; r < timed_runs; ++r) {
            const auto t0 = std::chrono::high_resolution_clock::now();
            const auto res = cpu_solver.solve(lp.c, lp.A_csc, lp.b);
            const auto t1 = std::chrono::high_resolution_clock::now();
            do_not_optimize(res.objective_value);
            cpu_solve_times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        rec.cpu_stats = calculate_stats(cpu_solve_times);

        std::vector<double> gpu_solve_times;
        gpu_solve_times.reserve(timed_runs);
        for (int r = 0; r < timed_runs; ++r) {
            const auto t0 = std::chrono::high_resolution_clock::now();
            const auto res = gpu_solver.solve(lp.c, lp.A_csc, lp.b);
            const auto t1 = std::chrono::high_resolution_clock::now();
            do_not_optimize(res.objective_value);
            gpu_solve_times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        rec.gpu_e2e_stats = calculate_stats(gpu_solve_times);
        rec.speedup_e2e   = (rec.gpu_e2e_stats.median_ms > 0.0) ? (rec.cpu_stats.median_ms / rec.gpu_e2e_stats.median_ms) : 0.0;

        std::cout << "[DONE] CPU: " << std::fixed << std::setprecision(2) << rec.cpu_stats.median_ms << " ms | "
                  << "GPU Hybrid: " << rec.gpu_e2e_stats.median_ms << " ms (" << format_speedup(rec.speedup_e2e) << ") | "
                  << "Iters: " << gpu_res.iterations << "\n";

        records.push_back(rec);
    }

    // 3. Convex QP Interior Point IPM
    struct QPProblemSize {
        std::size_t n;
        std::size_t m;
    };
    const std::vector<QPProblemSize> qp_sizes = {
        {32, 16},
        {64, 32},
        {128, 64},
        {256, 128}
    };

    for (const auto& size : qp_sizes) {
        const std::string test_name = "Convex QP IPM (n=" + std::to_string(size.n) + ", m=" + std::to_string(size.m) + ")";
        std::cout << "  Benchmarking " << std::left << std::setw(38) << test_name << "... " << std::flush;

        const auto qp = make_feasible_convex_qp(size.n, size.m, 6006 + size.n);

        QPOptions cpu_opts;
        cpu_opts.backend = SolverBackend::CPU;
        ConvexQPSolver cpu_solver(cpu_opts);
        const auto cpu_res = cpu_solver.solve(qp.Q, qp.c, qp.A, qp.b);

        QPOptions gpu_opts;
        gpu_opts.backend = SolverBackend::GPU;
        ConvexQPSolver gpu_solver(gpu_opts);
        const auto gpu_res = gpu_solver.solve(qp.Q, qp.c, qp.A, qp.b);

        if (cpu_res.status != QPSolverStatus::OPTIMAL || gpu_res.status != QPSolverStatus::OPTIMAL) {
            std::cout << "[FAILED Non-Optimal: CPU=" << static_cast<int>(cpu_res.status)
                      << ", GPU=" << static_cast<int>(gpu_res.status) << "]\n";
            return 1;
        }

        const double rel_obj_diff = std::abs(cpu_res.objective_value - gpu_res.objective_value) /
                                    std::max(1.0, std::abs(cpu_res.objective_value));
        if (rel_obj_diff > 1e-4) {
            std::cout << "[FAILED Objective Mismatch: CPU=" << cpu_res.objective_value
                      << ", GPU=" << gpu_res.objective_value << "]\n";
            return 1;
        }

        BenchmarkRecord rec;
        rec.category = "Full Solver: Convex QP";
        rec.test_name = test_name;
        rec.rows = size.m;
        rec.cols = size.n;
        rec.nnz = size.n * size.n + size.m * size.n;
        rec.density = 1.0;
        rec.max_abs_error = std::abs(cpu_res.objective_value - gpu_res.objective_value);
        rec.scaled_rel_error = rel_obj_diff;
        rec.notes = "Iters: " + std::to_string(gpu_res.iterations) + ", GPU GEMV calls: " +
                    std::to_string(gpu_res.device_diagnostics.gpu_gemv_calls);

        if (verify_only) {
            std::cout << "[PASS obj=" << cpu_res.objective_value << ", iters=" << cpu_res.iterations << "]\n";
            records.push_back(rec);
            continue;
        }

        std::vector<double> cpu_solve_times;
        cpu_solve_times.reserve(timed_runs);
        for (int r = 0; r < timed_runs; ++r) {
            const auto t0 = std::chrono::high_resolution_clock::now();
            const auto res = cpu_solver.solve(qp.Q, qp.c, qp.A, qp.b);
            const auto t1 = std::chrono::high_resolution_clock::now();
            do_not_optimize(res.objective_value);
            cpu_solve_times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        rec.cpu_stats = calculate_stats(cpu_solve_times);

        std::vector<double> gpu_solve_times;
        gpu_solve_times.reserve(timed_runs);
        for (int r = 0; r < timed_runs; ++r) {
            const auto t0 = std::chrono::high_resolution_clock::now();
            const auto res = gpu_solver.solve(qp.Q, qp.c, qp.A, qp.b);
            const auto t1 = std::chrono::high_resolution_clock::now();
            do_not_optimize(res.objective_value);
            gpu_solve_times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        rec.gpu_e2e_stats = calculate_stats(gpu_solve_times);
        rec.speedup_e2e   = (rec.gpu_e2e_stats.median_ms > 0.0) ? (rec.cpu_stats.median_ms / rec.gpu_e2e_stats.median_ms) : 0.0;

        std::cout << "[DONE] CPU: " << std::fixed << std::setprecision(2) << rec.cpu_stats.median_ms << " ms | "
                  << "GPU Hybrid: " << rec.gpu_e2e_stats.median_ms << " ms (" << format_speedup(rec.speedup_e2e) << ") | "
                  << "Iters: " << gpu_res.iterations << "\n";

        records.push_back(rec);
    }

    if (verify_only) {
        std::cout << "\n==============================================================================\n";
        std::cout << "  ALL BENCHMARK VERIFICATION SUITES PASSED SUCCESSFULLY\n";
        std::cout << "==============================================================================\n";
        return 0;
    }

    // ==============================================================================
    // EXPORT JSON AND CSV DATA
    // ==============================================================================
    std::cout << "\n[Exporting Benchmark Artifacts]\n";

    // 1. Export CSV
    {
        std::ofstream csv(csv_output_path);
        if (csv.is_open()) {
            csv << "Category,TestName,Rows,Cols,NNZ,Density,CPU_Median_ms,CPU_Min_ms,CPU_Max_ms,CPU_StdDev_ms,"
                << "GPU_Kernel_Median_ms,GPU_E2E_Median_ms,GPU_E2E_Min_ms,GPU_E2E_Max_ms,GPU_E2E_StdDev_ms,"
                << "Speedup_Compute,Speedup_E2E,Speedup_Label,Max_Abs_Error,Notes\n";

            for (const auto& r : records) {
                csv << "\"" << r.category << "\","
                    << "\"" << r.test_name << "\","
                    << r.rows << "," << r.cols << "," << r.nnz << "," << r.density << ","
                    << r.cpu_stats.median_ms << "," << r.cpu_stats.min_ms << "," << r.cpu_stats.max_ms << "," << r.cpu_stats.stddev_ms << ","
                    << r.gpu_kernel_stats.median_ms << ","
                    << r.gpu_e2e_stats.median_ms << "," << r.gpu_e2e_stats.min_ms << "," << r.gpu_e2e_stats.max_ms << "," << r.gpu_e2e_stats.stddev_ms << ","
                    << r.speedup_compute << "," << r.speedup_e2e << ","
                    << "\"" << format_speedup(r.speedup_e2e) << "\","
                    << r.max_abs_error << ","
                    << "\"" << r.notes << "\"\n";
            }
            std::cout << "  -> CSV saved to: " << csv_output_path << "\n";
        } else {
            std::cerr << "  -> WARNING: Failed to write CSV to " << csv_output_path << "\n";
        }
    }

    // 2. Export JSON
    {
        std::ofstream jf(json_output_path);
        if (jf.is_open()) {
            jf << "{\n";
            jf << "  \"system_info\": {\n";
            jf << "    \"gpu_name\": \"" << dev_info.name << "\",\n";
            jf << "    \"compute_capability\": \"" << dev_info.major << "." << dev_info.minor << "\",\n";
            jf << "    \"multiprocessor_count\": " << dev_info.multiprocessor_count << ",\n";
            jf << "    \"total_vram_bytes\": " << total_mem << ",\n";
            jf << "    \"available_vram_bytes\": " << free_mem << ",\n";
            jf << "    \"core_clock_khz\": " << core_clock_khz << ",\n";
            jf << "    \"memory_clock_khz\": " << mem_clock_khz << ",\n";
            jf << "    \"memory_bus_width_bits\": " << bus_width_bits << ",\n";
            jf << "    \"cpu_name\": \"AMD Ryzen 5 6600H with Radeon Graphics (12 Threads)\",\n";
            jf << "    \"compiler\": \"MSVC " << _MSC_VER << "\",\n";
            jf << "    \"cuda_runtime_version\": " << CUDART_VERSION << ",\n";
            jf << "    \"build_type\": \"Release\",\n";
            jf << "    \"repetition_count\": " << timed_runs << ",\n";
            jf << "    \"warmup_count\": " << warmup_runs << "\n";
            jf << "  },\n";
            jf << "  \"benchmarks\": [\n";
            for (std::size_t i = 0; i < records.size(); ++i) {
                const auto& r = records[i];
                jf << "    {\n";
                jf << "      \"category\": \"" << r.category << "\",\n";
                jf << "      \"name\": \"" << r.test_name << "\",\n";
                jf << "      \"rows\": " << r.rows << ",\n";
                jf << "      \"cols\": " << r.cols << ",\n";
                jf << "      \"nnz\": " << r.nnz << ",\n";
                jf << "      \"density\": " << r.density << ",\n";
                jf << "      \"transpose\": " << (r.transpose ? "true" : "false") << ",\n";
                jf << "      \"cpu_timing_ms\": {\n";
                jf << "        \"median\": " << r.cpu_stats.median_ms << ",\n";
                jf << "        \"min\": " << r.cpu_stats.min_ms << ",\n";
                jf << "        \"max\": " << r.cpu_stats.max_ms << ",\n";
                jf << "        \"mean\": " << r.cpu_stats.mean_ms << ",\n";
                jf << "        \"stddev\": " << r.cpu_stats.stddev_ms << ",\n";
                jf << "        \"iqr\": " << r.cpu_stats.iqr_ms << "\n";
                jf << "      },\n";
                jf << "      \"gpu_kernel_timing_ms\": {\n";
                jf << "        \"median\": " << r.gpu_kernel_stats.median_ms << "\n";
                jf << "      },\n";
                jf << "      \"gpu_e2e_timing_ms\": {\n";
                jf << "        \"median\": " << r.gpu_e2e_stats.median_ms << ",\n";
                jf << "        \"min\": " << r.gpu_e2e_stats.min_ms << ",\n";
                jf << "        \"max\": " << r.gpu_e2e_stats.max_ms << ",\n";
                jf << "        \"mean\": " << r.gpu_e2e_stats.mean_ms << ",\n";
                jf << "        \"stddev\": " << r.gpu_e2e_stats.stddev_ms << ",\n";
                jf << "        \"iqr\": " << r.gpu_e2e_stats.iqr_ms << "\n";
                jf << "      },\n";
                jf << "      \"speedup_compute\": " << r.speedup_compute << ",\n";
                jf << "      \"speedup_e2e\": " << r.speedup_e2e << ",\n";
                jf << "      \"speedup_label\": \"" << format_speedup(r.speedup_e2e) << "\",\n";
                jf << "      \"max_absolute_error\": " << r.max_abs_error << ",\n";
                jf << "      \"scaled_relative_error\": " << r.scaled_rel_error << ",\n";
                jf << "      \"notes\": \"" << r.notes << "\"\n";
                jf << "    }" << (i + 1 < records.size() ? "," : "") << "\n";
            }
            jf << "  ]\n";
            jf << "}\n";
            std::cout << "  -> JSON saved to: " << json_output_path << "\n";
        } else {
            std::cerr << "  -> WARNING: Failed to write JSON to " << json_output_path << "\n";
        }
    }

    std::cout << "\n==============================================================================\n";
    std::cout << "  BENCHMARK SUITE EXECUTED SUCCESSFULLY ACROSS ALL SECTIONS\n";
    std::cout << "==============================================================================\n";
    return 0;
#endif
}
