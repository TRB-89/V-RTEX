#include "vortex/matrix.hpp"
#include <chrono>
#include <iostream>
#include <random>
#include <iomanip>

int main() {
    std::cout << "=========================================================\n";
    std::cout << "           VORTEX Matrix Performance Benchmark          \n";
    std::cout << "=========================================================\n";

    constexpr std::size_t N = 300; // 300 x 300 = 90,000 elements, 27,000,000 operations
    std::cout << "Benchmarking with " << N << "x" << N << " matrices...\n\n";

    vortex::Matrix<double> A(N, N);
    vortex::Matrix<double> B(N, N);

    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> dist(-10.0, 10.0);

    for (std::size_t r = 0; r < N; ++r) {
        for (std::size_t c = 0; c < N; ++c) {
            A(r, c) = dist(rng);
            B(r, c) = dist(rng);
        }
    }

    // Benchmark Addition
    {
        auto start = std::chrono::high_resolution_clock::now();
        auto C = A + B;
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
        std::cout << "-> Matrix Addition (" << N << "x" << N << "): "
                  << std::fixed << std::setprecision(3) << duration.count() << " ms\n";
    }

    // Benchmark Transpose
    {
        auto start = std::chrono::high_resolution_clock::now();
        auto A_T = A.transpose();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
        std::cout << "-> Matrix Transpose (" << N << "x" << N << "): "
                  << std::fixed << std::setprecision(3) << duration.count() << " ms\n";
    }

    // Benchmark Multiplication
    {
        auto start = std::chrono::high_resolution_clock::now();
        auto C = A * B;
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
        std::cout << "-> Matrix Multiplication (" << N << "x" << N << "): "
                  << std::fixed << std::setprecision(3) << duration.count() << " ms\n";
        
        // Print verification trace
        std::cout << "   (Check C(0,0) = " << C(0, 0) << ")\n";
    }

    std::cout << "\nBenchmark complete successfully.\n";
    return 0;
}
