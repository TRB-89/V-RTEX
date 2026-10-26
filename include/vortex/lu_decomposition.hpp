#pragma once

#include "vortex/matrix.hpp"
#include <vector>
#include <cmath>
#include <stdexcept>
#include <string>
#include <numeric>
#include <algorithm>
#include <limits>
#include <iostream>

namespace vortex {

/**
 * @brief LU Factorization with Partial Pivoting: P * A = L * U.
 *
 * Implements high-performance Gaussian elimination with row pivoting for numerical stability.
 * Provides:
 *  - LU decomposition storing unit lower-triangular L and upper-triangular U
 *  - Forward substitution for L y = P b
 *  - Backward substitution for U x = y
 *  - Transpose solve for A^T pi = c (essential for Simplex dual multipliers / pricing)
 *  - Determinant, matrix inversion, and condition/stability metrics
 *  - Residual norm validation ||A x - b|| and ||A^T pi - c||
 *
 * @tparam T Arithmetic type (default: double)
 */
template <Arithmetic T = double>
class LUDecomposition {
private:
    std::size_t n_{0};
    Matrix<T> lu_;
    std::vector<std::size_t> perm_;
    std::size_t num_swaps_{0};
    bool is_singular_{false};
    double tolerance_{1e-12};

    // Numerical stability diagnostics
    double min_pivot_{0.0};
    double max_pivot_{0.0};
    double condition_ratio_{0.0};
    double pivot_growth_{1.0};
    T det_{T{0}};

public:
    LUDecomposition() = default;

    /**
     * @brief Construct and factorize square matrix A.
     */
    explicit LUDecomposition(const Matrix<T>& A, double tolerance = 1e-12) {
        compute(A, tolerance);
    }

    /**
     * @brief Computes the LU factorization of matrix A with partial pivoting: P * A = L * U.
     *
     * @param A Square matrix of size n x n
     * @param tolerance Numerical zero threshold for pivot detection
     * @return true if matrix is non-singular, false if singular
     */
    bool compute(const Matrix<T>& A, double tolerance = 1e-12) {
        if (A.rows() != A.cols()) {
            throw std::invalid_argument("LUDecomposition: Matrix must be square. Given (" +
                                        std::to_string(A.rows()) + "x" + std::to_string(A.cols()) + ")");
        }

        n_ = A.rows();
        tolerance_ = tolerance;
        is_singular_ = false;
        num_swaps_ = 0;

        if (n_ == 0) {
            lu_ = Matrix<T>(0, 0);
            perm_.clear();
            det_ = T{1};
            return true;
        }

        lu_ = A;
        perm_.resize(n_);
        std::iota(perm_.begin(), perm_.end(), 0);

        // Track initial max element for growth factor
        double a_max = 0.0;
        for (std::size_t i = 0; i < n_; ++i) {
            for (std::size_t j = 0; j < n_; ++j) {
                a_max = std::max(a_max, std::abs(static_cast<double>(A(i, j))));
            }
        }

        min_pivot_ = std::numeric_limits<double>::infinity();
        max_pivot_ = 0.0;

        // Doolittle algorithm with partial pivoting
        for (std::size_t k = 0; k < n_; ++k) {
            // Find pivot: row p >= k with maximum absolute value in column k
            std::size_t pivot_row = k;
            double max_val = std::abs(static_cast<double>(lu_(k, k)));

            for (std::size_t i = k + 1; i < n_; ++i) {
                double val = std::abs(static_cast<double>(lu_(i, k)));
                if (val > max_val) {
                    max_val = val;
                    pivot_row = i;
                }
            }

            // Check for numerical singularity
            if (max_val < tolerance_) {
                is_singular_ = true;
                min_pivot_ = 0.0;
                condition_ratio_ = 0.0;
                det_ = T{0};
                return false;
            }

            // Swap rows in working matrix and update permutation vector
            if (pivot_row != k) {
                lu_.swap_rows(k, pivot_row);
                std::swap(perm_[k], perm_[pivot_row]);
                num_swaps_++;
            }

            const T pivot = lu_(k, k);
            const double abs_pivot = std::abs(static_cast<double>(pivot));
            min_pivot_ = std::min(min_pivot_, abs_pivot);
            max_pivot_ = std::max(max_pivot_, abs_pivot);

            // Compute multipliers and eliminate row entries
            for (std::size_t i = k + 1; i < n_; ++i) {
                const T mult = lu_(i, k) / pivot;
                lu_(i, k) = mult; // Store multiplier in L's position

                for (std::size_t j = k + 1; j < n_; ++j) {
                    lu_(i, j) -= mult * lu_(k, j);
                }
            }
        }

        // Compute determinant: product of diagonal of U * (-1)^swaps
        det_ = (num_swaps_ % 2 == 1) ? T{-1} : T{1};
        for (std::size_t i = 0; i < n_; ++i) {
            det_ *= lu_(i, i);
        }

        // Stability diagnostics
        condition_ratio_ = (max_pivot_ > 0.0) ? (min_pivot_ / max_pivot_) : 0.0;

        double u_max = 0.0;
        for (std::size_t i = 0; i < n_; ++i) {
            for (std::size_t j = 0; j < n_; ++j) {
                u_max = std::max(u_max, std::abs(static_cast<double>(lu_(i, j))));
            }
        }
        pivot_growth_ = (a_max > 0.0) ? (u_max / a_max) : 1.0;

        return !is_singular_;
    }

    /**
     * @brief Solves the linear system A * x = b via forward and backward substitution.
     *
     * Since P * A = L * U, this solves L * (U * x) = P * b.
     *
     * @param b Right-hand side vector of length n
     * @return Solution vector x of length n
     */
    [[nodiscard]] std::vector<T> solve(const std::vector<T>& b) const {
        if (is_singular_) {
            throw std::runtime_error("LUDecomposition::solve: Cannot solve system with singular matrix.");
        }
        if (b.size() != n_) {
            throw std::invalid_argument("LUDecomposition::solve: Dimension mismatch: b size (" +
                                        std::to_string(b.size()) + ") != n (" + std::to_string(n_) + ")");
        }

        if (n_ == 0) return {};

        // 1. Permute b: b_perm[i] = b[perm_[i]]
        std::vector<T> y(n_);
        for (std::size_t i = 0; i < n_; ++i) {
            y[i] = b[perm_[i]];
        }

        // 2. Forward substitution: L * y = P * b (L has unit diagonal)
        for (std::size_t i = 0; i < n_; ++i) {
            for (std::size_t j = 0; j < i; ++j) {
                y[i] -= lu_(i, j) * y[j];
            }
        }

        // 3. Backward substitution: U * x = y
        std::vector<T> x = y;
        for (std::size_t i = n_; i-- > 0;) {
            for (std::size_t j = i + 1; j < n_; ++j) {
                x[i] -= lu_(i, j) * x[j];
            }
            x[i] /= lu_(i, i);
        }

        return x;
    }

    /**
     * @brief Solves the transposed linear system A^T * pi = c.
     *
     * Since P * A = L * U, A = P^T * L * U => A^T = U^T * L^T * P.
     * Hence, U^T * (L^T * P * pi) = c.
     *  1. Solve U^T * w = c for w (forward substitution with lower-triangular U^T)
     *  2. Solve L^T * v = w for v (backward substitution with upper-triangular L^T)
     *  3. Unpermute: pi[perm_[i]] = v[i]
     *
     * @param c Vector of costs / dual right-hand side of length n
     * @return Solution vector pi (pricing vector / dual multipliers)
     */
    [[nodiscard]] std::vector<T> solve_transpose(const std::vector<T>& c) const {
        if (is_singular_) {
            throw std::runtime_error("LUDecomposition::solve_transpose: Cannot solve with singular matrix.");
        }
        if (c.size() != n_) {
            throw std::invalid_argument("LUDecomposition::solve_transpose: Dimension mismatch: c size (" +
                                        std::to_string(c.size()) + ") != n (" + std::to_string(n_) + ")");
        }

        if (n_ == 0) return {};

        // 1. Solve U^T * w = c (Forward substitution on lower-triangular U^T)
        std::vector<T> w(n_);
        for (std::size_t i = 0; i < n_; ++i) {
            T sum = c[i];
            for (std::size_t j = 0; j < i; ++j) {
                sum -= lu_(j, i) * w[j]; // U^T(i, j) = U(j, i) = lu_(j, i)
            }
            w[i] = sum / lu_(i, i);
        }

        // 2. Solve L^T * v = w (Backward substitution on upper-triangular L^T, unit diagonal)
        std::vector<T> v(n_);
        for (std::size_t i = n_; i-- > 0;) {
            T sum = w[i];
            for (std::size_t j = i + 1; j < n_; ++j) {
                sum -= lu_(j, i) * v[j]; // L^T(i, j) = L(j, i) = lu_(j, i)
            }
            v[i] = sum; // Diagonal of L^T is 1
        }

        // 3. Unpermute v: v = P * pi => pi[perm_[i]] = v[i]
        std::vector<T> pi(n_);
        for (std::size_t i = 0; i < n_; ++i) {
            pi[perm_[i]] = v[i];
        }

        return pi;
    }

    /**
     * @brief Computes the explicit matrix inverse A^-1 by column-wise solve.
     */
    [[nodiscard]] Matrix<T> inverse() const {
        if (is_singular_) {
            throw std::runtime_error("LUDecomposition::inverse: Cannot invert singular matrix.");
        }

        Matrix<T> inv(n_, n_);
        std::vector<T> e(n_, T{0});

        for (std::size_t j = 0; j < n_; ++j) {
            e[j] = T{1};
            std::vector<T> col = solve(e);
            inv.set_column(j, col);
            e[j] = T{0};
        }

        return inv;
    }

    // ==========================================
    // Query & Extraction Methods
    // ==========================================

    [[nodiscard]] bool is_singular() const noexcept { return is_singular_; }
    [[nodiscard]] std::size_t dimension() const noexcept { return n_; }
    [[nodiscard]] const std::vector<std::size_t>& permutation() const noexcept { return perm_; }
    [[nodiscard]] std::size_t num_swaps() const noexcept { return num_swaps_; }
    [[nodiscard]] T determinant() const noexcept { return det_; }
    [[nodiscard]] double condition_ratio() const noexcept { return condition_ratio_; }
    [[nodiscard]] double pivot_growth_factor() const noexcept { return pivot_growth_; }
    [[nodiscard]] double min_pivot() const noexcept { return min_pivot_; }
    [[nodiscard]] double max_pivot() const noexcept { return max_pivot_; }
    [[nodiscard]] const Matrix<T>& LU() const noexcept { return lu_; }

    /**
     * @brief Extracts the unit lower-triangular matrix L.
     */
    [[nodiscard]] Matrix<T> L() const {
        Matrix<T> result(n_, n_, T{0});
        for (std::size_t i = 0; i < n_; ++i) {
            result(i, i) = T{1};
            for (std::size_t j = 0; j < i; ++j) {
                result(i, j) = lu_(i, j);
            }
        }
        return result;
    }

    /**
     * @brief Extracts the upper-triangular matrix U.
     */
    [[nodiscard]] Matrix<T> U() const {
        Matrix<T> result(n_, n_, T{0});
        for (std::size_t i = 0; i < n_; ++i) {
            for (std::size_t j = i; j < n_; ++j) {
                result(i, j) = lu_(i, j);
            }
        }
        return result;
    }

    /**
     * @brief Extracts the permutation matrix P where (P * A)(i, :) = A(perm[i], :).
     */
    [[nodiscard]] Matrix<T> P() const {
        Matrix<T> result(n_, n_, T{0});
        for (std::size_t i = 0; i < n_; ++i) {
            result(i, perm_[i]) = T{1};
        }
        return result;
    }

    // ==========================================
    // Stability & Verification Utilities
    // ==========================================

    /**
     * @brief Computes infinity-norm residual ||A * x - b||_inf = max_i |(A * x)_i - b_i|.
     */
    [[nodiscard]] double residual_norm(const Matrix<T>& A,
                                       const std::vector<T>& x,
                                       const std::vector<T>& b) const {
        if (A.rows() != b.size() || A.cols() != x.size()) {
            throw std::invalid_argument("LUDecomposition::residual_norm: Dimension mismatch.");
        }
        double max_res = 0.0;
        for (std::size_t i = 0; i < A.rows(); ++i) {
            double ax_i = 0.0;
            for (std::size_t j = 0; j < A.cols(); ++j) {
                ax_i += static_cast<double>(A(i, j) * x[j]);
            }
            max_res = std::max(max_res, std::abs(ax_i - static_cast<double>(b[i])));
        }
        return max_res;
    }

    /**
     * @brief Computes infinity-norm transpose residual ||A^T * pi - c||_inf.
     */
    [[nodiscard]] double residual_norm_transpose(const Matrix<T>& A,
                                                 const std::vector<T>& pi,
                                                 const std::vector<T>& c) const {
        if (A.rows() != pi.size() || A.cols() != c.size()) {
            throw std::invalid_argument("LUDecomposition::residual_norm_transpose: Dimension mismatch.");
        }
        double max_res = 0.0;
        for (std::size_t j = 0; j < A.cols(); ++j) {
            double at_pi_j = 0.0;
            for (std::size_t i = 0; i < A.rows(); ++i) {
                at_pi_j += static_cast<double>(A(i, j) * pi[i]);
            }
            max_res = std::max(max_res, std::abs(at_pi_j - static_cast<double>(c[j])));
        }
        return max_res;
    }

    /**
     * @brief Verifies that P * A == L * U within tolerance.
     */
    [[nodiscard]] bool verify_factorization(const Matrix<T>& A, double tolerance = 1e-9) const {
        if (is_singular_) return false;
        Matrix<T> PA = P() * A;
        Matrix<T> LU_prod = L() * U();
        return PA.is_approx(LU_prod, tolerance);
    }
};

} // namespace vortex
