#pragma once

#include <vector>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <concepts>
#include <cmath>
#include <initializer_list>
#include <string>

namespace vortex {

/**
 * @brief Concept constraining Matrix numeric types to floating point or integral types.
 */
template <typename T>
concept Arithmetic = std::is_arithmetic_v<T>;

/**
 * @brief High-performance 2D Matrix class implemented for VORTEX (SIH26119).
 *
 * Uses a flat 1D std::vector in row-major layout for optimal CPU cache utilization
 * and spatial locality.
 *
 * @tparam T Arithmetic data type (default: double)
 */
template <Arithmetic T = double>
class Matrix {
private:
    std::size_t rows_{0};
    std::size_t cols_{0};
    std::vector<T> data_;

    [[nodiscard]] inline std::size_t index(std::size_t r, std::size_t c) const noexcept {
        return r * cols_ + c;
    }

public:
    // ==========================================
    // Constructors & Destructors
    // ==========================================

    /// Default constructor: creates an empty 0x0 matrix
    Matrix() = default;

    /// Constructor: dimensions with optional default initial value (zeros by default)
    Matrix(std::size_t rows, std::size_t cols, T initial_value = T{0})
        : rows_(rows), cols_(cols), data_(rows * cols, initial_value) {}

    /// Constructor from initializer list of rows: {{1, 2}, {3, 4}}
    Matrix(std::initializer_list<std::initializer_list<T>> list) {
        rows_ = list.size();
        cols_ = (rows_ > 0) ? list.begin()->size() : 0;
        data_.reserve(rows_ * cols_);

        for (const auto& row : list) {
            if (row.size() != cols_) {
                throw std::invalid_argument("Matrix: All rows must have the same number of columns.");
            }
            data_.insert(data_.end(), row.begin(), row.end());
        }
    }

    /// Factory method: Identity matrix of size n x n
    [[nodiscard]] static Matrix identity(std::size_t n) {
        Matrix result(n, n, T{0});
        for (std::size_t i = 0; i < n; ++i) {
            result(i, i) = T{1};
        }
        return result;
    }

    // ==========================================
    // Dimensions & Memory Queries
    // ==========================================

    [[nodiscard]] std::size_t rows() const noexcept { return rows_; }
    [[nodiscard]] std::size_t cols() const noexcept { return cols_; }
    [[nodiscard]] std::size_t size() const noexcept { return data_.size(); }
    [[nodiscard]] bool empty() const noexcept { return data_.empty(); }

    [[nodiscard]] const T* data() const noexcept { return data_.data(); }
    [[nodiscard]] T* data() noexcept { return data_.data(); }

    [[nodiscard]] std::size_t logical_memory_bytes() const noexcept {
        return rows_ * cols_ * sizeof(T);
    }

    [[nodiscard]] std::size_t allocated_memory_bytes() const noexcept {
        return data_.capacity() * sizeof(T) + sizeof(*this);
    }

    [[nodiscard]] bool is_finite() const noexcept {
        for (const auto& val : data_) {
            if (!std::isfinite(static_cast<double>(val))) return false;
        }
        return true;
    }

    // ==========================================
    // Element Access
    // ==========================================

    /// Fast unchecked access via operator()
    inline T& operator()(std::size_t r, std::size_t c) noexcept {
        return data_[index(r, c)];
    }

    /// Fast unchecked const access via operator()
    inline const T& operator()(std::size_t r, std::size_t c) const noexcept {
        return data_[index(r, c)];
    }

    /// Bounds-checked access via at()
    T& at(std::size_t r, std::size_t c) {
        if (r >= rows_ || c >= cols_) {
            throw std::out_of_range("Matrix::at: Indices (" + std::to_string(r) + ", " +
                                    std::to_string(c) + ") out of bounds for matrix of size (" +
                                    std::to_string(rows_) + ", " + std::to_string(cols_) + ")");
        }
        return data_[index(r, c)];
    }

    /// Bounds-checked const access via at()
    const T& at(std::size_t r, std::size_t c) const {
        if (r >= rows_ || c >= cols_) {
            throw std::out_of_range("Matrix::at: Indices (" + std::to_string(r) + ", " +
                                    std::to_string(c) + ") out of bounds for matrix of size (" +
                                    std::to_string(rows_) + ", " + std::to_string(cols_) + ")");
        }
        return data_[index(r, c)];
    }

    // ==========================================
    // Row and Column Operations
    // ==========================================

    /**
     * @brief Swaps two rows in the matrix.
     */
    void swap_rows(std::size_t r1, std::size_t r2) {
        if (r1 >= rows_ || r2 >= rows_) {
            throw std::out_of_range("Matrix::swap_rows: Row index out of bounds.");
        }
        if (r1 == r2) return;
        for (std::size_t c = 0; c < cols_; ++c) {
            std::swap((*this)(r1, c), (*this)(r2, c));
        }
    }

    /**
     * @brief Extracts a column as a std::vector.
     */
    [[nodiscard]] std::vector<T> get_column(std::size_t col) const {
        if (col >= cols_) {
            throw std::out_of_range("Matrix::get_column: Column index out of bounds.");
        }
        std::vector<T> res(rows_);
        for (std::size_t r = 0; r < rows_; ++r) {
            res[r] = (*this)(r, col);
        }
        return res;
    }

    /**
     * @brief Sets an entire column with values from a std::vector.
     */
    void set_column(std::size_t col, const std::vector<T>& values) {
        if (col >= cols_) {
            throw std::out_of_range("Matrix::set_column: Column index out of bounds.");
        }
        if (values.size() != rows_) {
            throw std::invalid_argument("Matrix::set_column: Value size mismatch with row count.");
        }
        for (std::size_t r = 0; r < rows_; ++r) {
            (*this)(r, col) = values[r];
        }
    }

    /**
     * @brief Extracts a row as a std::vector.
     */
    [[nodiscard]] std::vector<T> get_row(std::size_t row) const {
        if (row >= rows_) {
            throw std::out_of_range("Matrix::get_row: Row index out of bounds.");
        }
        std::vector<T> res(cols_);
        for (std::size_t c = 0; c < cols_; ++c) {
            res[c] = (*this)(row, c);
        }
        return res;
    }

    /**
     * @brief Sets an entire row with values from a std::vector.
     */
    void set_row(std::size_t row, const std::vector<T>& values) {
        if (row >= rows_) {
            throw std::out_of_range("Matrix::set_row: Row index out of bounds.");
        }
        if (values.size() != cols_) {
            throw std::invalid_argument("Matrix::set_row: Value size mismatch with column count.");
        }
        for (std::size_t c = 0; c < cols_; ++c) {
            (*this)(row, c) = values[c];
        }
    }

    // ==========================================
    // Core Operations: Transpose
    // ==========================================

    /**
     * @brief Computes the transpose of the matrix: A^T.
     * Dimensions change from (rows x cols) to (cols x rows).
     */
    [[nodiscard]] Matrix transpose() const {
        Matrix result(cols_, rows_);
        for (std::size_t r = 0; r < rows_; ++r) {
            for (std::size_t c = 0; c < cols_; ++c) {
                result(c, r) = (*this)(r, c);
            }
        }
        return result;
    }

    // ==========================================
    // Arithmetic Operations: Addition & Subtraction
    // ==========================================

    Matrix& operator+=(const Matrix& other) {
        if (rows_ != other.rows_ || cols_ != other.cols_) {
            throw std::invalid_argument("Matrix addition requires identical dimensions: (" +
                                        std::to_string(rows_) + "x" + std::to_string(cols_) + ") vs (" +
                                        std::to_string(other.rows_) + "x" + std::to_string(other.cols_) + ")");
        }
        for (std::size_t i = 0; i < data_.size(); ++i) {
            data_[i] += other.data_[i];
        }
        return *this;
    }

    [[nodiscard]] friend Matrix operator+(Matrix lhs, const Matrix& rhs) {
        lhs += rhs;
        return lhs;
    }

    Matrix& operator-=(const Matrix& other) {
        if (rows_ != other.rows_ || cols_ != other.cols_) {
            throw std::invalid_argument("Matrix subtraction requires identical dimensions: (" +
                                        std::to_string(rows_) + "x" + std::to_string(cols_) + ") vs (" +
                                        std::to_string(other.rows_) + "x" + std::to_string(other.cols_) + ")");
        }
        for (std::size_t i = 0; i < data_.size(); ++i) {
            data_[i] -= other.data_[i];
        }
        return *this;
    }

    [[nodiscard]] friend Matrix operator-(Matrix lhs, const Matrix& rhs) {
        lhs -= rhs;
        return lhs;
    }

    // ==========================================
    // Arithmetic Operations: Scalar Multiplication
    // ==========================================

    Matrix& operator*=(const T& scalar) noexcept {
        for (auto& val : data_) {
            val *= scalar;
        }
        return *this;
    }

    [[nodiscard]] friend Matrix operator*(Matrix lhs, const T& scalar) noexcept {
        lhs *= scalar;
        return lhs;
    }

    [[nodiscard]] friend Matrix operator*(const T& scalar, Matrix rhs) noexcept {
        rhs *= scalar;
        return rhs;
    }

    // ==========================================
    // Core Operations: Matrix Multiplication
    // ==========================================

    /**
     * @brief Computes Matrix-Matrix multiplication: C = A * B.
     *
     * If A is (M x K) and B is (K x N), C will be (M x N).
     * Employs loop order (i, k, j) to ensure row-major sequential memory access
     * and maximize L1/L2 cache line hits.
     */
    [[nodiscard]] friend Matrix operator*(const Matrix& A, const Matrix& B) {
        if (A.cols_ != B.rows_) {
            throw std::invalid_argument("Matrix multiplication dimension mismatch: left cols (" +
                                        std::to_string(A.cols_) + ") must match right rows (" +
                                        std::to_string(B.rows_) + ")");
        }

        const std::size_t M = A.rows_;
        const std::size_t K = A.cols_;
        const std::size_t N = B.cols_;

        Matrix result(M, N, T{0});

        // Loop order i-k-j: Cache-friendly because B(k, j) and result(i, j)
        // are accessed sequentially in memory along row j.
        for (std::size_t i = 0; i < M; ++i) {
            for (std::size_t k = 0; k < K; ++k) {
                const T a_ik = A(i, k);
                for (std::size_t j = 0; j < N; ++j) {
                    result(i, j) += a_ik * B(k, j);
                }
            }
        }

        return result;
    }

    /**
     * @brief Computes Matrix-Vector multiplication: y = A * x.
     */
    [[nodiscard]] std::vector<T> multiply(const std::vector<T>& x) const {
        if (x.size() != cols_) {
            throw std::invalid_argument("Matrix-vector multiplication dimension mismatch: cols (" +
                                        std::to_string(cols_) + ") must match vector size (" +
                                        std::to_string(x.size()) + ")");
        }
        std::vector<T> y(rows_, T{0});
        for (std::size_t i = 0; i < rows_; ++i) {
            T sum = T{0};
            const std::size_t row_offset = i * cols_;
            for (std::size_t j = 0; j < cols_; ++j) {
                sum += data_[row_offset + j] * x[j];
            }
            y[i] = sum;
        }
        return y;
    }

    [[nodiscard]] friend std::vector<T> operator*(const Matrix& A, const std::vector<T>& x) {
        return A.multiply(x);
    }

    // ==========================================
    // Comparison & Tolerance Checking
    // ==========================================

    /**
     * @brief Checks if two matrices are approximately equal within a numerical tolerance.
     */
    [[nodiscard]] bool is_approx(const Matrix& other, double tolerance = 1e-6) const noexcept {
        if (rows_ != other.rows_ || cols_ != other.cols_) {
            return false;
        }
        for (std::size_t i = 0; i < data_.size(); ++i) {
            if (std::abs(static_cast<double>(data_[i] - other.data_[i])) > tolerance) {
                return false;
            }
        }
        return true;
    }

    [[nodiscard]] bool operator==(const Matrix& other) const noexcept {
        return is_approx(other, 1e-9);
    }

    [[nodiscard]] bool operator!=(const Matrix& other) const noexcept {
        return !(*this == other);
    }

    // ==========================================
    // Printing & Formatting
    // ==========================================

    void print(std::ostream& os = std::cout, int precision = 4) const {
        os << "[" << rows_ << "x" << cols_ << " Matrix]\n";
        for (std::size_t r = 0; r < rows_; ++r) {
            os << "  [ ";
            for (std::size_t c = 0; c < cols_; ++c) {
                os << std::fixed << std::setprecision(precision) << (*this)(r, c);
                if (c + 1 < cols_) os << ", ";
            }
            os << " ]\n";
        }
    }

    friend std::ostream& operator<<(std::ostream& os, const Matrix& m) {
        m.print(os);
        return os;
    }
};

} // namespace vortex
