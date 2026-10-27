#pragma once

#include "vortex/matrix.hpp"
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <string>

namespace vortex {

// Forward declarations
template <Arithmetic T> class CSRMatrix;
template <Arithmetic T> class CSCMatrix;

/**
 * @brief Coordinate triplet (row, col, value) for sparse matrix construction.
 */
template <Arithmetic T = double>
struct Triplet {
    std::size_t row{0};
    std::size_t col{0};
    T value{T{0}};

    Triplet() = default;
    Triplet(std::size_t r, std::size_t c, T val)
        : row(r), col(c), value(val) {}
};

/**
 * @brief Compressed Sparse Row (CSR) Matrix implementation.
 *
 * Stores non-zero values row-by-row:
 *  - values_: non-zero values of length nnz
 *  - col_indices_: column index for each non-zero value, length nnz
 *  - row_ptrs_: starting index of each row in values_, length rows + 1
 *
 * Optimized for:
 *  - Row-wise slicing and constraint querying
 *  - Sparse Matrix-Vector multiplication (y = A * x)
 *  - Compact memory footprint for large sparse systems
 */
template <Arithmetic T = double>
class CSRMatrix {
    template <Arithmetic U> friend class CSCMatrix;
private:
    std::size_t rows_{0};
    std::size_t cols_{0};
    std::vector<T> values_;
    std::vector<std::size_t> col_indices_;
    std::vector<std::size_t> row_ptrs_;

public:
    CSRMatrix() = default;

    CSRMatrix(std::size_t rows, std::size_t cols)
        : rows_(rows), cols_(cols), row_ptrs_(rows + 1, 0) {}

    CSRMatrix(std::size_t rows, std::size_t cols,
              std::vector<T> values,
              std::vector<std::size_t> col_indices,
              std::vector<std::size_t> row_ptrs)
        : rows_(rows), cols_(cols),
          values_(std::move(values)),
          col_indices_(std::move(col_indices)),
          row_ptrs_(std::move(row_ptrs)) {

        if (row_ptrs_.size() != rows_ + 1) {
            throw std::invalid_argument("CSRMatrix: row_ptrs size must be rows + 1.");
        }
        if (values_.size() != col_indices_.size()) {
            throw std::invalid_argument("CSRMatrix: values and col_indices sizes must match.");
        }
    }

    // ==========================================
    // Dimensions & Memory Queries
    // ==========================================

    [[nodiscard]] std::size_t rows() const noexcept { return rows_; }
    [[nodiscard]] std::size_t cols() const noexcept { return cols_; }
    [[nodiscard]] std::size_t non_zeros() const noexcept { return values_.size(); }
    [[nodiscard]] bool empty() const noexcept { return values_.empty() && rows_ == 0 && cols_ == 0; }

    [[nodiscard]] double sparsity() const noexcept {
        return stored_sparsity();
    }

    [[nodiscard]] double stored_sparsity() const noexcept {
        if (rows_ == 0 || cols_ == 0) return 1.0;
        return 1.0 - (static_cast<double>(non_zeros()) / static_cast<double>(rows_ * cols_));
    }

    [[nodiscard]] double true_sparsity(double tol = 0.0) const noexcept {
        if (rows_ == 0 || cols_ == 0) return 1.0;
        return 1.0 - (static_cast<double>(count_true_nonzeros(tol)) / static_cast<double>(rows_ * cols_));
    }

    [[nodiscard]] std::size_t count_true_nonzeros(double tol = 0.0) const noexcept {
        std::size_t count = 0;
        for (const auto& v : values_) {
            if (std::abs(static_cast<double>(v)) > tol) ++count;
        }
        return count;
    }

    [[nodiscard]] std::size_t count_stored_zeros(double tol = 0.0) const noexcept {
        return values_.size() - count_true_nonzeros(tol);
    }

    [[nodiscard]] bool has_stored_zeros(double tol = 0.0) const noexcept {
        return count_stored_zeros(tol) > 0;
    }

    [[nodiscard]] std::size_t logical_memory_bytes() const noexcept {
        return (values_.size() * sizeof(T)) +
               (col_indices_.size() * sizeof(std::size_t)) +
               (row_ptrs_.size() * sizeof(std::size_t));
    }

    [[nodiscard]] std::size_t allocated_memory_bytes() const noexcept {
        return (values_.capacity() * sizeof(T)) +
               (col_indices_.capacity() * sizeof(std::size_t)) +
               (row_ptrs_.capacity() * sizeof(std::size_t)) +
               sizeof(*this);
    }

    [[nodiscard]] std::size_t memory_bytes() const noexcept {
        return allocated_memory_bytes();
    }

    void shrink_to_fit() {
        values_.shrink_to_fit();
        col_indices_.shrink_to_fit();
        row_ptrs_.shrink_to_fit();
    }

    void drop_zeros(double tol = 0.0) {
        std::vector<T> new_vals;
        std::vector<std::size_t> new_cols;
        std::vector<std::size_t> new_ptrs(rows_ + 1, 0);
        new_vals.reserve(values_.size());
        new_cols.reserve(col_indices_.size());

        for (std::size_t r = 0; r < rows_; ++r) {
            new_ptrs[r] = new_vals.size();
            for (std::size_t idx = row_ptrs_[r]; idx < row_ptrs_[r + 1]; ++idx) {
                if (std::abs(static_cast<double>(values_[idx])) > tol) {
                    new_vals.push_back(values_[idx]);
                    new_cols.push_back(col_indices_[idx]);
                }
            }
        }
        new_ptrs[rows_] = new_vals.size();
        values_ = std::move(new_vals);
        col_indices_ = std::move(new_cols);
        row_ptrs_ = std::move(new_ptrs);
    }

    [[nodiscard]] bool is_finite() const noexcept {
        for (const auto& v : values_) {
            if (!std::isfinite(static_cast<double>(v))) return false;
        }
        return true;
    }

    [[nodiscard]] const std::vector<T>& values() const noexcept { return values_; }
    [[nodiscard]] const std::vector<std::size_t>& col_indices() const noexcept { return col_indices_; }
    [[nodiscard]] const std::vector<std::size_t>& row_ptrs() const noexcept { return row_ptrs_; }

    // ==========================================
    // Element Access
    // ==========================================

    /**
     * @brief Retrieves element (r, c) via binary search in row r.
     * Complexity: O(log(nnz_in_row)).
     */
    [[nodiscard]] T operator()(std::size_t r, std::size_t c) const noexcept {
        if (r >= rows_ || c >= cols_) return T{0};

        const std::size_t start = row_ptrs_[r];
        const std::size_t end = row_ptrs_[r + 1];

        auto it = std::lower_bound(col_indices_.begin() + start,
                                   col_indices_.begin() + end,
                                   c);

        if (it != (col_indices_.begin() + end) && *it == c) {
            std::size_t idx = std::distance(col_indices_.begin(), it);
            return values_[idx];
        }
        return T{0};
    }

    [[nodiscard]] T at(std::size_t r, std::size_t c) const {
        if (r >= rows_ || c >= cols_) {
            throw std::out_of_range("CSRMatrix::at: Index (" + std::to_string(r) +
                                    ", " + std::to_string(c) + ") out of range (" +
                                    std::to_string(rows_) + ", " + std::to_string(cols_) + ")");
        }
        return (*this)(r, c);
    }

    // ==========================================
    // Conversions
    // ==========================================

    /**
     * @brief Constructs a CSRMatrix from a dense Matrix.
     */
    [[nodiscard]] static CSRMatrix from_dense(const Matrix<T>& dense, double tol = 1e-12) {
        CSRMatrix csr(dense.rows(), dense.cols());
        csr.row_ptrs_.resize(dense.rows() + 1, 0);

        for (std::size_t r = 0; r < dense.rows(); ++r) {
            csr.row_ptrs_[r] = csr.values_.size();
            for (std::size_t c = 0; c < dense.cols(); ++c) {
                const T val = dense(r, c);
                if (std::abs(static_cast<double>(val)) > tol) {
                    csr.values_.push_back(val);
                    csr.col_indices_.push_back(c);
                }
            }
        }
        csr.row_ptrs_[dense.rows()] = csr.values_.size();
        return csr;
    }

    /**
     * @brief Constructs a CSRMatrix from a list of (row, col, value) triplets.
     */
    [[nodiscard]] static CSRMatrix from_triplets(std::size_t rows, std::size_t cols,
                                                 std::vector<Triplet<T>> triplets,
                                                 bool sum_duplicates = true) {
        CSRMatrix csr(rows, cols);
        if (triplets.empty()) return csr;

        // Sort triplets row-major: by row first, then col
        std::sort(triplets.begin(), triplets.end(), [](const Triplet<T>& a, const Triplet<T>& b) {
            if (a.row != b.row) return a.row < b.row;
            return a.col < b.col;
        });

        csr.row_ptrs_.assign(rows + 1, 0);

        // Sum duplicates if requested and pack into CSR arrays
        std::size_t current_row = 0;
        for (std::size_t i = 0; i < triplets.size(); ++i) {
            if (triplets[i].row >= rows || triplets[i].col >= cols) {
                throw std::out_of_range("CSRMatrix::from_triplets: Triplet coordinates exceed dimensions.");
            }

            T val = triplets[i].value;
            if (sum_duplicates) {
                while (i + 1 < triplets.size() &&
                       triplets[i + 1].row == triplets[i].row &&
                       triplets[i + 1].col == triplets[i].col) {
                    val += triplets[++i].value;
                }
            }

            while (current_row < triplets[i].row) {
                csr.row_ptrs_[++current_row] = csr.values_.size();
            }
            csr.values_.push_back(val);
            csr.col_indices_.push_back(triplets[i].col);
        }

        while (current_row < rows) {
            csr.row_ptrs_[++current_row] = csr.values_.size();
        }

        return csr;
    }

    /**
     * @brief Converts CSRMatrix to a dense Matrix.
     */
    [[nodiscard]] Matrix<T> to_dense() const {
        Matrix<T> dense(rows_, cols_, T{0});
        for (std::size_t r = 0; r < rows_; ++r) {
            for (std::size_t idx = row_ptrs_[r]; idx < row_ptrs_[r + 1]; ++idx) {
                dense(r, col_indices_[idx]) = values_[idx];
            }
        }
        return dense;
    }

    /**
     * @brief Converts CSRMatrix to CSCMatrix in O(nnz + rows + cols) time.
     */
    [[nodiscard]] CSCMatrix<T> to_csc() const;

    // ==========================================
    // Operations & SpMV
    // ==========================================

    /**
     * @brief Sparse Matrix-Vector multiplication: y = A * x.
     * Complexity: O(nnz).
     */
    [[nodiscard]] std::vector<T> multiply(const std::vector<T>& x) const {
        if (x.size() != cols_) {
            throw std::invalid_argument("CSRMatrix::multiply: Dimension mismatch: x size (" +
                                        std::to_string(x.size()) + ") != cols (" + std::to_string(cols_) + ")");
        }

        std::vector<T> y(rows_, T{0});
        for (std::size_t r = 0; r < rows_; ++r) {
            T sum = T{0};
            for (std::size_t idx = row_ptrs_[r]; idx < row_ptrs_[r + 1]; ++idx) {
                sum += values_[idx] * x[col_indices_[idx]];
            }
            y[r] = sum;
        }
        return y;
    }

    [[nodiscard]] friend std::vector<T> operator*(const CSRMatrix<T>& A, const std::vector<T>& x) {
        return A.multiply(x);
    }

    /**
     * @brief Sparse Matrix-Vector transpose multiplication: y = A^T * x.
     * Complexity: O(nnz).
     */
    [[nodiscard]] std::vector<T> multiply_transpose(const std::vector<T>& x) const {
        if (x.size() != rows_) {
            throw std::invalid_argument("CSRMatrix::multiply_transpose: Dimension mismatch.");
        }

        std::vector<T> y(cols_, T{0});
        for (std::size_t r = 0; r < rows_; ++r) {
            const T xr = x[r];
            if (std::abs(static_cast<double>(xr)) <= 1e-15) continue;
            for (std::size_t idx = row_ptrs_[r]; idx < row_ptrs_[r + 1]; ++idx) {
                y[col_indices_[idx]] += values_[idx] * xr;
            }
        }
        return y;
    }

    /**
     * @brief Transposes CSR matrix into a new CSR matrix: A^T.
     */
    [[nodiscard]] CSRMatrix<T> transpose() const;

    // ==========================================
    // Slicing & Extraction
    // ==========================================

    /**
     * @brief Extracts row r as a dense std::vector.
     */
    [[nodiscard]] std::vector<T> get_row_dense(std::size_t r) const {
        if (r >= rows_) throw std::out_of_range("CSRMatrix::get_row_dense: Out of bounds.");
        std::vector<T> row(cols_, T{0});
        for (std::size_t idx = row_ptrs_[r]; idx < row_ptrs_[r + 1]; ++idx) {
            row[col_indices_[idx]] = values_[idx];
        }
        return row;
    }

    void print(std::ostream& os = std::cout, int precision = 4) const {
        os << "[CSRMatrix " << rows_ << "x" << cols_ << ", " << non_zeros()
           << " non-zeros (" << std::fixed << std::setprecision(1) << (sparsity() * 100.0)
           << "% sparse)]\n";
        to_dense().print(os, precision);
    }

    friend std::ostream& operator<<(std::ostream& os, const CSRMatrix<T>& m) {
        m.print(os);
        return os;
    }
};

/**
 * @brief Compressed Sparse Column (CSC) Matrix implementation.
 *
 * Stores non-zero values column-by-column:
 *  - values_: non-zero values of length nnz
 *  - row_indices_: row index for each non-zero value, length nnz
 *  - col_ptrs_: starting index of each column in values_, length cols + 1
 *
 * Optimized for:
 *  - Column-wise operations in Revised Simplex:
 *      * Fast dual pricing: (pi^T * A_j) in O(nnz_j) operations
 *      * Fast entering column extraction: B * d = A_j
 *      * Fast basis matrix column assembly
 *  - Transpose operations (CSC transpose is CSR with zero copy)
 */
template <Arithmetic T = double>
class CSCMatrix {
    template <Arithmetic U> friend class CSRMatrix;
private:
    std::size_t rows_{0};
    std::size_t cols_{0};
    std::vector<T> values_;
    std::vector<std::size_t> row_indices_;
    std::vector<std::size_t> col_ptrs_;

public:
    CSCMatrix() = default;

    CSCMatrix(std::size_t rows, std::size_t cols)
        : rows_(rows), cols_(cols), col_ptrs_(cols + 1, 0) {}

    CSCMatrix(std::size_t rows, std::size_t cols,
              std::vector<T> values,
              std::vector<std::size_t> row_indices,
              std::vector<std::size_t> col_ptrs)
        : rows_(rows), cols_(cols),
          values_(std::move(values)),
          row_indices_(std::move(row_indices)),
          col_ptrs_(std::move(col_ptrs)) {

        if (col_ptrs_.size() != cols_ + 1) {
            throw std::invalid_argument("CSCMatrix: col_ptrs size must be cols + 1.");
        }
        if (values_.size() != row_indices_.size()) {
            throw std::invalid_argument("CSCMatrix: values and row_indices sizes must match.");
        }
    }

    // ==========================================
    // Dimensions & Memory Queries
    // ==========================================

    [[nodiscard]] std::size_t rows() const noexcept { return rows_; }
    [[nodiscard]] std::size_t cols() const noexcept { return cols_; }
    [[nodiscard]] std::size_t non_zeros() const noexcept { return values_.size(); }
    [[nodiscard]] bool empty() const noexcept { return values_.empty() && rows_ == 0 && cols_ == 0; }

    [[nodiscard]] double sparsity() const noexcept {
        return stored_sparsity();
    }

    [[nodiscard]] double stored_sparsity() const noexcept {
        if (rows_ == 0 || cols_ == 0) return 1.0;
        return 1.0 - (static_cast<double>(non_zeros()) / static_cast<double>(rows_ * cols_));
    }

    [[nodiscard]] double true_sparsity(double tol = 0.0) const noexcept {
        if (rows_ == 0 || cols_ == 0) return 1.0;
        return 1.0 - (static_cast<double>(count_true_nonzeros(tol)) / static_cast<double>(rows_ * cols_));
    }

    [[nodiscard]] std::size_t count_true_nonzeros(double tol = 0.0) const noexcept {
        std::size_t count = 0;
        for (const auto& v : values_) {
            if (std::abs(static_cast<double>(v)) > tol) ++count;
        }
        return count;
    }

    [[nodiscard]] std::size_t count_stored_zeros(double tol = 0.0) const noexcept {
        return values_.size() - count_true_nonzeros(tol);
    }

    [[nodiscard]] bool has_stored_zeros(double tol = 0.0) const noexcept {
        return count_stored_zeros(tol) > 0;
    }

    [[nodiscard]] std::size_t logical_memory_bytes() const noexcept {
        return (values_.size() * sizeof(T)) +
               (row_indices_.size() * sizeof(std::size_t)) +
               (col_ptrs_.size() * sizeof(std::size_t));
    }

    [[nodiscard]] std::size_t allocated_memory_bytes() const noexcept {
        return (values_.capacity() * sizeof(T)) +
               (row_indices_.capacity() * sizeof(std::size_t)) +
               (col_ptrs_.capacity() * sizeof(std::size_t)) +
               sizeof(*this);
    }

    [[nodiscard]] std::size_t memory_bytes() const noexcept {
        return allocated_memory_bytes();
    }

    void shrink_to_fit() {
        values_.shrink_to_fit();
        row_indices_.shrink_to_fit();
        col_ptrs_.shrink_to_fit();
    }

    void drop_zeros(double tol = 0.0) {
        std::vector<T> new_vals;
        std::vector<std::size_t> new_rows;
        std::vector<std::size_t> new_ptrs(cols_ + 1, 0);
        new_vals.reserve(values_.size());
        new_rows.reserve(row_indices_.size());

        for (std::size_t c = 0; c < cols_; ++c) {
            new_ptrs[c] = new_vals.size();
            for (std::size_t idx = col_ptrs_[c]; idx < col_ptrs_[c + 1]; ++idx) {
                if (std::abs(static_cast<double>(values_[idx])) > tol) {
                    new_vals.push_back(values_[idx]);
                    new_rows.push_back(row_indices_[idx]);
                }
            }
        }
        new_ptrs[cols_] = new_vals.size();
        values_ = std::move(new_vals);
        row_indices_ = std::move(new_rows);
        col_ptrs_ = std::move(new_ptrs);
    }

    [[nodiscard]] bool is_finite() const noexcept {
        for (const auto& v : values_) {
            if (!std::isfinite(static_cast<double>(v))) return false;
        }
        return true;
    }

    [[nodiscard]] const std::vector<T>& values() const noexcept { return values_; }
    [[nodiscard]] const std::vector<std::size_t>& row_indices() const noexcept { return row_indices_; }
    [[nodiscard]] const std::vector<std::size_t>& col_ptrs() const noexcept { return col_ptrs_; }

    // ==========================================
    // Element Access
    // ==========================================

    [[nodiscard]] T operator()(std::size_t r, std::size_t c) const noexcept {
        if (r >= rows_ || c >= cols_) return T{0};

        const std::size_t start = col_ptrs_[c];
        const std::size_t end = col_ptrs_[c + 1];

        auto it = std::lower_bound(row_indices_.begin() + start,
                                   row_indices_.begin() + end,
                                   r);

        if (it != (row_indices_.begin() + end) && *it == r) {
            std::size_t idx = std::distance(row_indices_.begin(), it);
            return values_[idx];
        }
        return T{0};
    }

    [[nodiscard]] T at(std::size_t r, std::size_t c) const {
        if (r >= rows_ || c >= cols_) {
            throw std::out_of_range("CSCMatrix::at: Index (" + std::to_string(r) +
                                    ", " + std::to_string(c) + ") out of range.");
        }
        return (*this)(r, c);
    }

    // ==========================================
    // Conversions
    // ==========================================

    /**
     * @brief Constructs a CSCMatrix from a dense Matrix.
     */
    [[nodiscard]] static CSCMatrix from_dense(const Matrix<T>& dense, double tol = 1e-12) {
        CSCMatrix csc(dense.rows(), dense.cols());
        csc.col_ptrs_.resize(dense.cols() + 1, 0);

        for (std::size_t c = 0; c < dense.cols(); ++c) {
            csc.col_ptrs_[c] = csc.values_.size();
            for (std::size_t r = 0; r < dense.rows(); ++r) {
                const T val = dense(r, c);
                if (std::abs(static_cast<double>(val)) > tol) {
                    csc.values_.push_back(val);
                    csc.row_indices_.push_back(r);
                }
            }
        }
        csc.col_ptrs_[dense.cols()] = csc.values_.size();
        return csc;
    }

    /**
     * @brief Constructs a CSCMatrix from triplets.
     */
    [[nodiscard]] static CSCMatrix from_triplets(std::size_t rows, std::size_t cols,
                                                 std::vector<Triplet<T>> triplets,
                                                 bool sum_duplicates = true) {
        CSCMatrix csc(rows, cols);
        if (triplets.empty()) return csc;

        // Sort column-major: by col first, then row
        std::sort(triplets.begin(), triplets.end(), [](const Triplet<T>& a, const Triplet<T>& b) {
            if (a.col != b.col) return a.col < b.col;
            return a.row < b.row;
        });

        csc.col_ptrs_.assign(cols + 1, 0);

        std::size_t current_col = 0;
        for (std::size_t i = 0; i < triplets.size(); ++i) {
            if (triplets[i].row >= rows || triplets[i].col >= cols) {
                throw std::out_of_range("CSCMatrix::from_triplets: Out of bounds coordinates.");
            }

            T val = triplets[i].value;
            if (sum_duplicates) {
                while (i + 1 < triplets.size() &&
                       triplets[i + 1].col == triplets[i].col &&
                       triplets[i + 1].row == triplets[i].row) {
                    val += triplets[++i].value;
                }
            }

            while (current_col < triplets[i].col) {
                csc.col_ptrs_[++current_col] = csc.values_.size();
            }
            csc.values_.push_back(val);
            csc.row_indices_.push_back(triplets[i].row);
        }

        while (current_col < cols) {
            csc.col_ptrs_[++current_col] = csc.values_.size();
        }

        return csc;
    }

    /**
     * @brief Converts CSCMatrix to a dense Matrix.
     */
    [[nodiscard]] Matrix<T> to_dense() const {
        Matrix<T> dense(rows_, cols_, T{0});
        for (std::size_t c = 0; c < cols_; ++c) {
            for (std::size_t idx = col_ptrs_[c]; idx < col_ptrs_[c + 1]; ++idx) {
                dense(row_indices_[idx], c) = values_[idx];
            }
        }
        return dense;
    }

    /**
     * @brief Converts CSCMatrix to CSRMatrix in O(nnz + rows + cols) time.
     */
    [[nodiscard]] CSRMatrix<T> to_csr() const {
        CSRMatrix<T> csr(rows_, cols_);
        if (values_.empty()) return csr;

        csr.row_ptrs_.assign(rows_ + 1, 0);

        // 1. Count non-zeros in each row
        for (std::size_t idx = 0; idx < values_.size(); ++idx) {
            csr.row_ptrs_[row_indices_[idx] + 1]++;
        }

        // 2. Prefix sum to compute row_ptrs
        for (std::size_t r = 0; r < rows_; ++r) {
            csr.row_ptrs_[r + 1] += csr.row_ptrs_[r];
        }

        // 3. Scatter into CSR arrays
        csr.values_.resize(values_.size());
        csr.col_indices_.resize(values_.size());
        std::vector<std::size_t> next_pos = csr.row_ptrs_;

        for (std::size_t c = 0; c < cols_; ++c) {
            for (std::size_t idx = col_ptrs_[c]; idx < col_ptrs_[c + 1]; ++idx) {
                const std::size_t r = row_indices_[idx];
                const std::size_t dest = next_pos[r]++;
                csr.values_[dest] = values_[idx];
                csr.col_indices_[dest] = c;
            }
        }

        return csr;
    }

    // ==========================================
    // Operations & SpMV
    // ==========================================

    /**
     * @brief Sparse Matrix-Vector multiplication: y = A * x.
     * Complexity: O(nnz).
     */
    [[nodiscard]] std::vector<T> multiply(const std::vector<T>& x) const {
        if (x.size() != cols_) {
            throw std::invalid_argument("CSCMatrix::multiply: Dimension mismatch.");
        }

        std::vector<T> y(rows_, T{0});
        for (std::size_t c = 0; c < cols_; ++c) {
            const T xc = x[c];
            if (std::abs(static_cast<double>(xc)) <= 1e-15) continue;
            for (std::size_t idx = col_ptrs_[c]; idx < col_ptrs_[c + 1]; ++idx) {
                y[row_indices_[idx]] += values_[idx] * xc;
            }
        }
        return y;
    }

    [[nodiscard]] friend std::vector<T> operator*(const CSCMatrix<T>& A, const std::vector<T>& x) {
        return A.multiply(x);
    }

    /**
     * @brief Sparse Matrix-Vector transpose multiplication: y = A^T * x.
     * Evaluates y_c = dot(A[:, c], x) touching only non-zeros in column c!
     * Complexity: O(nnz).
     */
    [[nodiscard]] std::vector<T> multiply_transpose(const std::vector<T>& x) const {
        if (x.size() != rows_) {
            throw std::invalid_argument("CSCMatrix::multiply_transpose: Dimension mismatch.");
        }

        std::vector<T> y(cols_, T{0});
        for (std::size_t c = 0; c < cols_; ++c) {
            T sum = T{0};
            for (std::size_t idx = col_ptrs_[c]; idx < col_ptrs_[c + 1]; ++idx) {
                sum += values_[idx] * x[row_indices_[idx]];
            }
            y[c] = sum;
        }
        return y;
    }

    /**
     * @brief Computes dot product of column c with vector x: dot(A[:, c], x).
     * Highly optimized for Simplex pricing / reduced-cost computation.
     */
    [[nodiscard]] T column_dot(std::size_t c, const std::vector<T>& x) const {
        if (c >= cols_) throw std::out_of_range("CSCMatrix::column_dot: Invalid col.");
        T sum = T{0};
        for (std::size_t idx = col_ptrs_[c]; idx < col_ptrs_[c + 1]; ++idx) {
            sum += values_[idx] * x[row_indices_[idx]];
        }
        return sum;
    }

    /**
     * @brief Transposes CSC matrix into a new CSC matrix: A^T.
     */
    [[nodiscard]] CSCMatrix<T> transpose() const {
        // A CSC matrix's transpose data is structurally identical to CSR
        CSRMatrix<T> as_csr(cols_, rows_, values_, row_indices_, col_ptrs_);
        return as_csr.to_csc();
    }

    // ==========================================
    // Slicing & Extraction
    // ==========================================

    /**
     * @brief Extracts column c as a dense std::vector.
     */
    [[nodiscard]] std::vector<T> get_column_dense(std::size_t c) const {
        if (c >= cols_) throw std::out_of_range("CSCMatrix::get_column_dense: Out of bounds.");
        std::vector<T> col(rows_, T{0});
        for (std::size_t idx = col_ptrs_[c]; idx < col_ptrs_[c + 1]; ++idx) {
            col[row_indices_[idx]] = values_[idx];
        }
        return col;
    }

    void print(std::ostream& os = std::cout, int precision = 4) const {
        os << "[CSCMatrix " << rows_ << "x" << cols_ << ", " << non_zeros()
           << " non-zeros (" << std::fixed << std::setprecision(1) << (sparsity() * 100.0)
           << "% sparse)]\n";
        to_dense().print(os, precision);
    }

    friend std::ostream& operator<<(std::ostream& os, const CSCMatrix<T>& m) {
        m.print(os);
        return os;
    }
};

// ==========================================
// Inline Implementations of CSR methods needing CSC
// ==========================================

template <Arithmetic T>
CSCMatrix<T> CSRMatrix<T>::to_csc() const {
    CSCMatrix<T> csc(rows_, cols_);
    if (values_.empty()) return csc;

    csc.col_ptrs_.assign(cols_ + 1, 0);

    // 1. Count non-zeros per column
    for (std::size_t idx = 0; idx < values_.size(); ++idx) {
        csc.col_ptrs_[col_indices_[idx] + 1]++;
    }

    // 2. Prefix sum to compute col_ptrs
    for (std::size_t c = 0; c < cols_; ++c) {
        csc.col_ptrs_[c + 1] += csc.col_ptrs_[c];
    }

    // 3. Scatter into CSC arrays
    csc.values_.resize(values_.size());
    csc.row_indices_.resize(values_.size());
    std::vector<std::size_t> next_pos = csc.col_ptrs_;

    for (std::size_t r = 0; r < rows_; ++r) {
        for (std::size_t idx = row_ptrs_[r]; idx < row_ptrs_[r + 1]; ++idx) {
            const std::size_t c = col_indices_[idx];
            const std::size_t dest = next_pos[c]++;
            csc.values_[dest] = values_[idx];
            csc.row_indices_[dest] = r;
        }
    }

    return csc;
}

template <Arithmetic T>
CSRMatrix<T> CSRMatrix<T>::transpose() const {
    // CSR transpose data is structurally identical to CSC
    CSCMatrix<T> as_csc(cols_, rows_, values_, col_indices_, row_ptrs_);
    return as_csc.to_csr();
}

} // namespace vortex
