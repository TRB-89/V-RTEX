#include "vortex/matrix.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <cassert>

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
}

void test_constructors_and_access() {
    std::cout << "\n--- Running Suite: Constructors & Element Access ---\n";

    // 1. Default constructor
    vortex::Matrix<double> empty_mat;
    record_result("Default constructor creates 0x0 matrix",
                  empty_mat.rows() == 0 && empty_mat.cols() == 0 && empty_mat.empty());

    // 2. Sized constructor with default values
    vortex::Matrix<double> zeros(3, 4);
    bool all_zero = true;
    for (std::size_t r = 0; r < 3; ++r) {
        for (std::size_t c = 0; c < 4; ++c) {
            if (zeros(r, c) != 0.0) all_zero = false;
        }
    }
    record_result("Sized constructor initializes elements to zero",
                  zeros.rows() == 3 && zeros.cols() == 4 && all_zero);

    // 3. Sized constructor with custom initial value
    vortex::Matrix<int> fives(2, 3, 5);
    bool all_five = true;
    for (std::size_t r = 0; r < 2; ++r) {
        for (std::size_t c = 0; c < 3; ++c) {
            if (fives(r, c) != 5) all_five = false;
        }
    }
    record_result("Constructor with custom initial value", all_five);

    // 4. Initializer list constructor
    vortex::Matrix<double> init_mat = {
        {1.0, 2.0},
        {3.0, 4.0},
        {5.0, 6.0}
    };
    record_result("Initializer list sets proper dimensions",
                  init_mat.rows() == 3 && init_mat.cols() == 2 &&
                  init_mat(0, 0) == 1.0 && init_mat(2, 1) == 6.0);

    // 5. Ragged initializer list throws exception
    bool caught_ragged = false;
    try {
        vortex::Matrix<double> ragged = {
            {1.0, 2.0},
            {3.0, 4.0, 5.0} // Inconsistent column count
        };
    } catch (const std::invalid_argument&) {
        caught_ragged = true;
    }
    record_result("Ragged initializer list throws std::invalid_argument", caught_ragged);

    // 6. Bounds checking with at()
    bool caught_oob = false;
    try {
        init_mat.at(10, 0);
    } catch (const std::out_of_range&) {
        caught_oob = true;
    }
    record_result("Matrix::at() throws std::out_of_range for invalid row", caught_oob);
}

void test_matrix_addition_and_subtraction() {
    std::cout << "\n--- Running Suite: Addition & Subtraction ---\n";

    vortex::Matrix<double> A = {
        {1.0, 2.0},
        {3.0, 4.0}
    };
    vortex::Matrix<double> B = {
        {5.0, 6.0},
        {7.0, 8.0}
    };

    vortex::Matrix<double> sum = A + B;
    vortex::Matrix<double> expected_sum = {
        {6.0, 8.0},
        {10.0, 12.0}
    };
    record_result("Matrix addition (A + B)", sum == expected_sum);

    vortex::Matrix<double> diff = B - A;
    vortex::Matrix<double> expected_diff = {
        {4.0, 4.0},
        {4.0, 4.0}
    };
    record_result("Matrix subtraction (B - A)", diff == expected_diff);

    // Dimension mismatch
    vortex::Matrix<double> incompatible(3, 2);
    bool caught_dim_mismatch = false;
    try {
        auto fail = A + incompatible;
    } catch (const std::invalid_argument&) {
        caught_dim_mismatch = true;
    }
    record_result("Matrix addition dimension mismatch throws std::invalid_argument", caught_dim_mismatch);
}

void test_matrix_multiplication() {
    std::cout << "\n--- Running Suite: Matrix Multiplication ---\n";

    // 1. Rectangular multiplication (2x3) * (3x2) -> (2x2)
    // [1, 2, 3]   [7,  8]   [ (1*7 + 2*9 + 3*2), (1*8 + 2*1 + 3*3) ]   [ 31, 19 ]
    // [4, 5, 6] * [9,  1] = [ (4*7 + 5*9 + 6*2), (4*8 + 5*1 + 6*3) ] = [ 85, 55 ]
    //             [2,  3]
    vortex::Matrix<double> A = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0}
    };

    vortex::Matrix<double> B = {
        {7.0, 8.0},
        {9.0, 1.0},
        {2.0, 3.0}
    };

    vortex::Matrix<double> C = A * B;
    vortex::Matrix<double> expected_C = {
        {31.0, 19.0},
        {85.0, 55.0}
    };
    record_result("Rectangular matrix multiplication (2x3 * 3x2)", C == expected_C);

    // 2. Identity matrix multiplication: A * I == A and I * A == A
    vortex::Matrix<double> sq = {
        {2.0, -1.0},
        {5.0,  3.0}
    };
    auto I = vortex::Matrix<double>::identity(2);
    record_result("Square matrix * Identity equals original (A * I == A)", (sq * I) == sq);
    record_result("Identity * Square matrix equals original (I * A == A)", (I * sq) == sq);

    // 3. Multiplication dimension mismatch: (2x3) * (2x2) should fail
    vortex::Matrix<double> incompatible(2, 2);
    bool caught_mul_dim = false;
    try {
        auto fail = A * incompatible;
    } catch (const std::invalid_argument&) {
        caught_mul_dim = true;
    }
    record_result("Matrix multiplication dimension mismatch throws std::invalid_argument", caught_mul_dim);
}

void test_matrix_transpose() {
    std::cout << "\n--- Running Suite: Matrix Transpose ---\n";

    // 1. Rectangular transpose (2x3 -> 3x2)
    vortex::Matrix<double> mat = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0}
    };

    vortex::Matrix<double> mat_T = mat.transpose();
    vortex::Matrix<double> expected_T = {
        {1.0, 4.0},
        {2.0, 5.0},
        {3.0, 6.0}
    };
    record_result("Rectangular transpose dimensions and values (2x3 -> 3x2)",
                  mat_T.rows() == 3 && mat_T.cols() == 2 && mat_T == expected_T);

    // 2. Double transpose property: (A^T)^T == A
    record_result("Double transpose property ((A^T)^T == A)", mat_T.transpose() == mat);

    // 3. Symmetric matrix transpose
    vortex::Matrix<double> sym = {
        {1.0, 7.0, 3.0},
        {7.0, 4.0, -5.0},
        {3.0, -5.0, 6.0}
    };
    record_result("Symmetric matrix equals its transpose (S^T == S)", sym.transpose() == sym);
}

void test_scalar_operations() {
    std::cout << "\n--- Running Suite: Scalar Multiplication ---\n";

    vortex::Matrix<double> A = {
        {2.0, -3.0},
        {0.5,  4.0}
    };

    vortex::Matrix<double> scaled_right = A * 3.0;
    vortex::Matrix<double> scaled_left = 3.0 * A;
    vortex::Matrix<double> expected = {
        {6.0, -9.0},
        {1.5, 12.0}
    };

    record_result("Right scalar multiplication (A * 3.0)", scaled_right == expected);
    record_result("Left scalar multiplication (3.0 * A)", scaled_left == expected);
    record_result("Commutative property: A * s == s * A", scaled_right == scaled_left);
}

int main() {
    std::cout << "=========================================================\n";
    std::cout << "          VORTEX Automated Test Suite (SIH26119)         \n";
    std::cout << "=========================================================\n";

    test_constructors_and_access();
    test_matrix_addition_and_subtraction();
    test_matrix_multiplication();
    test_matrix_transpose();
    test_scalar_operations();

    std::cout << "\n=========================================================\n";
    std::cout << "  SUMMARY: " << passed_tests << " / " << total_tests << " tests passed.\n";
    std::cout << "=========================================================\n";

    return (passed_tests == total_tests) ? 0 : 1;
}
