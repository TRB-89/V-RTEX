#include "vortex/cuda_backend.hpp"

#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <cmath>
#include <cassert>
#include <string>

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

} // namespace

// ==============================================================================
// 1. Device Detection and Capability Audit
// ==============================================================================
void test_device_detection() {
    std::cout << "\n--- Test 1: CUDA Device Detection & Hardware Audit ---\n";

    TEST_ASSERT(vortex::cuda::is_cuda_available(), "CUDA device must be detected on system");

    vortex::cuda::CudaDeviceInfo dev;
    try {
        dev = vortex::cuda::get_cuda_device_info(0);
    } catch (const std::exception& e) {
        TEST_ASSERT(false, std::string("get_cuda_device_info failed: ") + e.what());
    }

    std::cout << "  GPU Name:               " << dev.name << "\n";
    std::cout << "  Compute Capability:     " << dev.major << "." << dev.minor << "\n";
    std::cout << "  Total VRAM:             " << (dev.total_memory_bytes / (1024 * 1024)) << " MB\n";
    std::cout << "  Free VRAM:              " << (dev.free_memory_bytes / (1024 * 1024)) << " MB\n";
    std::cout << "  Multiprocessors (SMs):  " << dev.multiprocessor_count << "\n";
    std::cout << "  Max Threads Per Block:  " << dev.max_threads_per_block << "\n";
    std::cout << "  Warp Size:              " << dev.warp_size << "\n";

    TEST_ASSERT(!dev.name.empty(), "Device name cannot be empty");
    TEST_ASSERT(dev.major >= 8, "Expected compute capability >= 8 (e.g. Ampere sm_86)");
    TEST_ASSERT(dev.total_memory_bytes > 0, "Total memory must be > 0");

    TEST_PASS("Device detection and hardware properties successfully queried.");
}

// ==============================================================================
// 2. RAII Device Buffer Lifecycle and Memory Safety
// ==============================================================================
void test_raii_device_buffer() {
    std::cout << "\n--- Test 2: RAII Device Buffer Safety & Lifecycle ---\n";

    // A. Default construction
    vortex::cuda::CudaDeviceBuffer<double> buf_empty;
    TEST_ASSERT(buf_empty.empty(), "Default buffer must be empty");
    TEST_ASSERT(buf_empty.size() == 0, "Default buffer size must be 0");
    TEST_ASSERT(buf_empty.data() == nullptr, "Default buffer pointer must be nullptr");

    // B. Allocation and Host-Device Round-Trip
    const std::size_t n = 512;
    vortex::cuda::CudaDeviceBuffer<double> buf(n);
    TEST_ASSERT(!buf.empty(), "Allocated buffer must not be empty");
    TEST_ASSERT(buf.size() == n, "Allocated buffer size must match requested count");
    TEST_ASSERT(buf.data() != nullptr, "Allocated buffer data must be non-null");

    std::vector<double> h_src(n);
    for (std::size_t i = 0; i < n; ++i) {
        h_src[i] = static_cast<double>(i) * 1.5 - 30.0;
    }

    buf.copy_from_host(h_src.data(), n);

    std::vector<double> h_dst(n, 0.0);
    buf.copy_to_host(h_dst.data(), n);

    for (std::size_t i = 0; i < n; ++i) {
        TEST_ASSERT(std::abs(h_src[i] - h_dst[i]) == 0.0, "Host-device-host roundtrip must be bitwise identical");
    }

    // C. Move semantics
    vortex::cuda::CudaDeviceBuffer<double> buf_moved = std::move(buf);
    TEST_ASSERT(buf.data() == nullptr && buf.size() == 0, "Moved-from buffer must be reset");
    TEST_ASSERT(buf_moved.data() != nullptr && buf_moved.size() == n, "Moved-to buffer must own resources");

    // D. Explicit reset / free
    buf_moved.reset();
    TEST_ASSERT(buf_moved.empty() && buf_moved.data() == nullptr, "Reset buffer must release memory");

    TEST_PASS("RAII Device Buffer allocation, transfer, and move semantics verified.");
}

// ==============================================================================
// 3. Vector Addition Numerical Verification Across Various Sizes
// ==============================================================================
void test_vector_addition_sizes() {
    std::cout << "\n--- Test 3: Vector Addition Numerical Correctness Sweep ---\n";

    // Test a wide variety of sizes, explicitly including non-multiples of block_size (256)
    const std::vector<std::size_t> test_sizes = {
        1,       // Single element
        2,       // 2 elements
        31,      // Less than warp size (32)
        32,      // Exact warp size
        33,      // Warp size + 1
        127,     // Prime below block size
        255,     // block_size - 1
        256,     // Exact block size
        257,     // block_size + 1
        500,     // Non-multiple
        1000,    // Non-multiple
        1024,    // Exact 4 blocks
        5003,    // Prime size
        65536,   // Exact 256 blocks
        100007   // Large non-multiple prime
    };

    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> dist(-1000.0, 1000.0);

    for (std::size_t n : test_sizes) {
        std::vector<double> a(n);
        std::vector<double> b(n);
        for (std::size_t i = 0; i < n; ++i) {
            a[i] = dist(rng);
            b[i] = dist(rng);
        }

        // Test with default block size 256
        const auto report = vortex::cuda::execute_vector_add_verified(a, b, 256);

        TEST_ASSERT(report.success, "Vector addition execution failed for size " + std::to_string(n));
        TEST_ASSERT(report.max_numerical_error <= 1e-12,
                    "Max numerical error exceeds tolerance: " + std::to_string(report.max_numerical_error));

        std::cout << "  Size: " << std::setw(7) << n
                  << " | Blocks: " << std::setw(4) << report.grid_size
                  << " (Threads: " << std::setw(3) << report.block_size << ")"
                  << " | GPU: " << report.device_name
                  << " | Max Error: " << std::scientific << std::setprecision(2) << report.max_numerical_error
                  << " [PASS]\n";
    }

    // Also test single-precision (float)
    {
        const std::size_t n_float = 7777;
        std::vector<float> a_f(n_float), b_f(n_float);
        for (std::size_t i = 0; i < n_float; ++i) {
            a_f[i] = static_cast<float>(dist(rng));
            b_f[i] = static_cast<float>(dist(rng));
        }

        std::vector<float> c_f = vortex::cuda::vector_add(a_f, b_f, 128);
        float max_err_f = 0.0f;
        for (std::size_t i = 0; i < n_float; ++i) {
            const float diff = std::abs(c_f[i] - (a_f[i] + b_f[i]));
            if (diff > max_err_f) max_err_f = diff;
        }

        TEST_ASSERT(max_err_f <= 1e-5f, "Float vector add error exceeds tolerance: " + std::to_string(max_err_f));
        std::cout << "  Float Size: " << n_float
                  << " | Max Error: " << std::scientific << std::setprecision(2) << max_err_f
                  << " [PASS]\n";
    }

    TEST_PASS("All vector sizes (including primes & non-multiples) mathematically verified against CPU.");
}

// ==============================================================================
// 4. Invalid Input Handling and Exception Robustness
// ==============================================================================
void test_invalid_inputs() {
    std::cout << "\n--- Test 4: Invalid Inputs & Error Propagation Audit ---\n";

    double dummy = 1.0;

    // A. Null pointer inputs to raw vector_add
    bool caught = false;
    try {
        vortex::cuda::vector_add(nullptr, &dummy, &dummy, 1);
    } catch (const std::invalid_argument& e) {
        caught = true;
        std::cout << "  Caught expected exception (null a): " << e.what() << "\n";
    }
    TEST_ASSERT(caught, "Null pointer a must throw std::invalid_argument");

    caught = false;
    try {
        vortex::cuda::vector_add(&dummy, nullptr, &dummy, 1);
    } catch (const std::invalid_argument& e) {
        caught = true;
        std::cout << "  Caught expected exception (null b): " << e.what() << "\n";
    }
    TEST_ASSERT(caught, "Null pointer b must throw std::invalid_argument");

    caught = false;
    try {
        vortex::cuda::vector_add(&dummy, &dummy, nullptr, 1);
    } catch (const std::invalid_argument& e) {
        caught = true;
        std::cout << "  Caught expected exception (null c): " << e.what() << "\n";
    }
    TEST_ASSERT(caught, "Null pointer c must throw std::invalid_argument");

    // B. Zero size vector
    caught = false;
    try {
        vortex::cuda::vector_add(&dummy, &dummy, &dummy, 0);
    } catch (const std::invalid_argument& e) {
        caught = true;
        std::cout << "  Caught expected exception (size 0): " << e.what() << "\n";
    }
    TEST_ASSERT(caught, "Vector size n=0 must throw std::invalid_argument");

    // C. Invalid block sizes
    caught = false;
    try {
        vortex::cuda::vector_add(&dummy, &dummy, &dummy, 1, 0);
    } catch (const std::invalid_argument& e) {
        caught = true;
        std::cout << "  Caught expected exception (block_size 0): " << e.what() << "\n";
    }
    TEST_ASSERT(caught, "block_size=0 must throw std::invalid_argument");

    caught = false;
    try {
        vortex::cuda::vector_add(&dummy, &dummy, &dummy, 1, 2048);
    } catch (const std::invalid_argument& e) {
        caught = true;
        std::cout << "  Caught expected exception (block_size > 1024): " << e.what() << "\n";
    }
    TEST_ASSERT(caught, "block_size > 1024 must throw std::invalid_argument");

    // D. Dimension mismatch in std::vector overload
    std::vector<double> v1(10, 1.0);
    std::vector<double> v2(20, 2.0);
    caught = false;
    try {
        vortex::cuda::vector_add(v1, v2);
    } catch (const std::invalid_argument& e) {
        caught = true;
        std::cout << "  Caught expected exception (size mismatch): " << e.what() << "\n";
    }
    TEST_ASSERT(caught, "Mismatched dimensions must throw std::invalid_argument");

    // E. Invalid Device ID
    caught = false;
    try {
        vortex::cuda::get_cuda_device_info(9999);
    } catch (const std::invalid_argument& e) {
        caught = true;
        std::cout << "  Caught expected exception (invalid device ID): " << e.what() << "\n";
    }
    TEST_ASSERT(caught, "Invalid device ID must throw std::invalid_argument");

    TEST_PASS("All invalid input edge cases cleanly handled with descriptive exceptions.");
}

// ==============================================================================
// 5. Execution Summary and Status Reporting
// ==============================================================================
void test_diagnostic_report() {
    std::cout << "\n--- Test 5: Diagnostic Verified Report ---\n";

    const std::size_t n = 10000;
    std::vector<double> a(n, 3.1415926535);
    std::vector<double> b(n, 2.7182818284);

    const auto report = vortex::cuda::execute_vector_add_verified(a, b, 256);

    std::cout << "  Status:                 " << (report.success ? "SUCCESS" : "FAILED") << "\n";
    std::cout << "  Device Name:            " << report.device_name << "\n";
    std::cout << "  Elements Processed:     " << report.vector_size << "\n";
    std::cout << "  Block Size:             " << report.block_size << "\n";
    std::cout << "  Grid Size:              " << report.grid_size << "\n";
    std::cout << "  Max Numerical Error:    " << std::scientific << report.max_numerical_error << "\n";
    std::cout << "  Status Message:         " << report.status_message << "\n";

    TEST_ASSERT(report.success, "Verified diagnostic report must indicate success");
    TEST_ASSERT(report.max_numerical_error == 0.0, "Identical floating point additions must have zero error");

    TEST_PASS("Diagnostic report generated and verified.");
}

int main() {
    std::cout << "========================================================================\n";
    std::cout << "     VORTEX Phase 8A: CUDA GPU Backend Foundation Test Suite            \n";
    std::cout << "========================================================================\n";

    test_device_detection();
    test_raii_device_buffer();
    test_vector_addition_sizes();
    test_invalid_inputs();
    test_diagnostic_report();

    std::cout << "\n========================================================================\n";
    std::cout << " Test Summary: " << g_tests_passed << " passed, " << g_tests_failed << " failed.\n";
    std::cout << "========================================================================\n";

    return (g_tests_failed == 0) ? 0 : 1;
}
