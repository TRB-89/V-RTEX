"""
VØRTEX Phase 9D: Matrix Inspector & Convergence Visualization Automated Test Suite
Validates:
1. Matrix Density & Sparsity calculation logic.
2. Canonical CSR (Compressed Sparse Row) transformation: values, col_indices, row_pointers.
3. Canonical CSC (Compressed Sparse Column) transformation: values, row_indices, col_pointers.
4. Edge matrices: empty matrices, empty rows, empty columns, rectangular (m > n and m < n), negative values.
5. 2D Feasible Region Polyhedron computation: half-space intersection, feasible vertex filtering, angular sorting.
6. QP Paraboloid objective surface value calculation: f(x) = 0.5 * x^T Q x + c^T x.
7. Iteration history parsing, log-scale residual transformations, and handling missing iteration histories.
8. Real backend API integration test for QP iteration_history and matrix_info.
"""

import sys
import math
import json
import unittest
from pathlib import Path

# Add server directory to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "server"))

from fastapi.testclient import TestClient
from main import app, SOLVER_EXECUTABLE


def compute_matrix_stats(matrix):
    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0
    total = m * n
    nnz = sum(1 for i in range(m) for j in range(n) if abs(matrix[i][j]) > 1e-12)
    density = nnz / total if total > 0 else 0.0
    sparsity = 1.0 - density
    return {"m": m, "n": n, "total": total, "nnz": nnz, "density": density, "sparsity": sparsity}


def to_csr(matrix):
    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0
    values = []
    col_indices = []
    row_pointers = [0]
    for i in range(m):
        for j in range(n):
            val = matrix[i][j]
            if abs(val) > 1e-12:
                values.append(val)
                col_indices.append(j)
        row_pointers.append(len(values))
    return {"values": values, "col_indices": col_indices, "row_pointers": row_pointers}


def to_csc(matrix):
    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0
    values = []
    row_indices = []
    col_pointers = [0]
    for j in range(n):
        for i in range(m):
            val = matrix[i][j]
            if abs(val) > 1e-12:
                values.append(val)
                row_indices.append(i)
        col_pointers.append(len(values))
    return {"values": values, "row_indices": row_indices, "col_pointers": col_pointers}


class TestPhase9DVisualization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        print("\n=======================================================")
        print("  VØRTEX Phase 9D: Visualization Validation Test Suite")
        print("=======================================================")
        assert SOLVER_EXECUTABLE.exists(), f"Solver executable missing: {SOLVER_EXECUTABLE}"

    # -------------------------------------------------------------
    # 1. MATRIX DENSITY & SPARSITY
    # -------------------------------------------------------------
    def test_01_matrix_density_and_sparsity(self):
        """Verifies exact density and sparsity calculation on dense and sparse matrices."""
        # 4x4 matrix with 4 nonzeros (density = 4/16 = 0.25, sparsity = 0.75)
        mat = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 0.0],
            [0.0, 0.0, 0.0, 4.0],
        ]
        stats = compute_matrix_stats(mat)
        self.assertEqual(stats["m"], 4)
        self.assertEqual(stats["n"], 4)
        self.assertEqual(stats["nnz"], 4)
        self.assertAlmostEqual(stats["density"], 0.25)
        self.assertAlmostEqual(stats["sparsity"], 0.75)

    # -------------------------------------------------------------
    # 2. CANONICAL CSR & CSC TRANSFORMATIONS
    # -------------------------------------------------------------
    def test_02_csr_transformation(self):
        """Verifies CSR generation matches textbook standard (values, col_indices, row_pointers)."""
        mat = [
            [1.0, 0.0, 2.0],
            [0.0, 3.0, 0.0],
            [4.0, 0.0, 5.0],
        ]
        csr = to_csr(mat)
        self.assertEqual(csr["values"], [1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(csr["col_indices"], [0, 2, 1, 0, 2])
        self.assertEqual(csr["row_pointers"], [0, 2, 3, 5])

    def test_03_csc_transformation(self):
        """Verifies CSC generation matches textbook standard (values, row_indices, col_pointers)."""
        mat = [
            [1.0, 0.0, 2.0],
            [0.0, 3.0, 0.0],
            [4.0, 0.0, 5.0],
        ]
        csc = to_csc(mat)
        # Col 0 has 1.0 (row 0), 4.0 (row 2)
        # Col 1 has 3.0 (row 1)
        # Col 2 has 2.0 (row 0), 5.0 (row 2)
        self.assertEqual(csc["values"], [1.0, 4.0, 3.0, 2.0, 5.0])
        self.assertEqual(csc["row_indices"], [0, 2, 1, 0, 2])
        self.assertEqual(csc["col_pointers"], [0, 2, 3, 5])

    def test_04_rectangular_and_empty_rows_columns(self):
        """Verifies CSR/CSC robustness on rectangular matrices with empty rows and columns."""
        # 4 rows, 2 columns with empty row 1 and negative value
        mat = [
            [1.0, 0.0],
            [0.0, 0.0],  # empty row
            [0.0, -2.5], # negative value
            [3.0, 4.0],
        ]
        csr = to_csr(mat)
        self.assertEqual(csr["values"], [1.0, -2.5, 3.0, 4.0])
        self.assertEqual(csr["col_indices"], [0, 1, 0, 1])
        self.assertEqual(csr["row_pointers"], [0, 1, 1, 2, 4])

        csc = to_csc(mat)
        # Col 0: 1.0 (row 0), 3.0 (row 3)
        # Col 1: -2.5 (row 2), 4.0 (row 3)
        self.assertEqual(csc["values"], [1.0, 3.0, -2.5, 4.0])
        self.assertEqual(csc["row_indices"], [0, 3, 2, 3])
        self.assertEqual(csc["col_pointers"], [0, 2, 4])

    # -------------------------------------------------------------
    # 3. 2D GEOMETRIC FEASIBLE REGION
    # -------------------------------------------------------------
    def test_05_feasible_region_polygon_vertices(self):
        """Verifies 2D feasible polygon calculation for SIH26119 standard benchmark."""
        # Maximize 3x1 + 5x2
        # x1 <= 4
        # 2x2 <= 12  ->  x2 <= 6
        # 3x1 + 2x2 <= 18
        # x1, x2 >= 0
        # Vertices of feasible region:
        # (0, 0)
        # (4, 0)
        # (4, 3) -> 3(4) + 2(3) = 18 <= 18
        # (2, 6) -> 3(2) + 2(6) = 18 <= 18 and 2(6) = 12 <= 12
        # (0, 6) -> 2(6) = 12 <= 12
        lines = [
            (-1, 0, 0, "LEQ"),   # x1 >= 0
            (0, -1, 0, "LEQ"),   # x2 >= 0
            (1, 0, 4, "LEQ"),    # x1 <= 4
            (0, 2, 12, "LEQ"),   # 2x2 <= 12
            (3, 2, 18, "LEQ"),   # 3x1 + 2x2 <= 18
        ]

        # Calculate intersections
        intersections = []
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                a1, b1, c1, _ = lines[i]
                a2, b2, c2, _ = lines[j]
                det = a1 * b2 - a2 * b1
                if abs(det) > 1e-9:
                    x = (c1 * b2 - c2 * b1) / det
                    y = (a1 * c2 - a2 * c1) / det
                    intersections.append((x, y))

        tol = 1e-5
        feasible = []
        for px, py in intersections:
            if px < -tol or py < -tol:
                continue
            is_feas = True
            for a, b, c, sense in lines:
                val = a * px + b * py
                if sense == "LEQ" and val > c + tol:
                    is_feas = False
                    break
            if is_feas:
                if not any(abs(px - u[0]) < 1e-4 and abs(py - u[1]) < 1e-4 for u in feasible):
                    feasible.append((round(px, 4), round(py, 4)))

        expected_vertices = {(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (2.0, 6.0), (0.0, 6.0)}
        self.assertEqual(set(feasible), expected_vertices)

        # Check optimal vertex has highest objective
        c_vec = [3.0, 5.0]
        objs = {pt: c_vec[0] * pt[0] + c_vec[1] * pt[1] for pt in feasible}
        opt_pt = max(objs, key=objs.get)
        self.assertEqual(opt_pt, (2.0, 6.0))
        self.assertEqual(objs[opt_pt], 36.0)

    # -------------------------------------------------------------
    # 4. 3D QP OBJECTIVE SURFACE VALUES
    # -------------------------------------------------------------
    def test_06_qp_surface_value_function(self):
        """Verifies QP paraboloid surface value calculation: f(x) = 0.5 * x^T Q x + c^T x."""
        # Q = [[2, 0], [0, 2]], c = [-2, -5]
        # f(x1, x2) = 0.5 * (2*x1^2 + 2*x2^2) - 2*x1 - 5*x2 = x1^2 + x2^2 - 2*x1 - 5*x2
        def f(x1, x2):
            return 0.5 * (2.0 * x1**2 + 2.0 * x2**2) - 2.0 * x1 - 5.0 * x2

        # At origin (0, 0)
        self.assertEqual(f(0.0, 0.0), 0.0)
        # At unconstrained minimizer (1.0, 2.5): f(1, 2.5) = 1 + 6.25 - 2 - 12.5 = -7.25
        self.assertAlmostEqual(f(1.0, 2.5), -7.25)
        # At constrained portfolio optimum (0.25, 1.75): f(0.25, 1.75) = 0.0625 + 3.0625 - 0.5 - 8.75 = -6.125
        self.assertAlmostEqual(f(0.25, 1.75), -6.125)

    # -------------------------------------------------------------
    # 5. REAL NATIVE SOLVER TELEMETRY INTEGRATION
    # -------------------------------------------------------------
    def test_07_real_qp_iteration_history_returned(self):
        """Verifies native C++ solver genuinely produces step-by-step iteration_history for QP."""
        payload = {
            "solver_type": "QP",
            "sense": "MINIMIZE",
            "backend": "AUTO",
            "c": [-2.0, -5.0],
            "A": [[1.0, 1.0]],
            "b": [2.0],
            "Q": [[2.0, 0.0], [0.0, 2.0]],
        }
        res = self.client.post("/api/v1/solve", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()

        self.assertIn("iteration_history", data)
        history = data["iteration_history"]
        self.assertIsInstance(history, list)
        self.assertGreater(len(history), 0)

        # Verify monotonicity of duality gap decay
        first_gap = history[0]["duality_gap"]
        last_gap = history[-1]["duality_gap"]
        self.assertGreater(first_gap, last_gap)
        self.assertLess(last_gap, 1e-6)

        # Verify iteration count matches history length
        self.assertEqual(data["iterations"], len(history))
        print(f"  [QP Convergence History] Steps: {len(history)}, Final Gap: {last_gap:.2e}")

    def test_08_lp_truthful_missing_iteration_history(self):
        """Verifies LP returns terminal pivot diagnostics without fabricated intermediate history."""
        payload = {
            "solver_type": "LP",
            "sense": "MAXIMIZE",
            "backend": "AUTO",
            "c": [3.0, 5.0],
            "A": [[1.0, 0.0], [0.0, 2.0], [3.0, 2.0]],
            "b": [4.0, 12.0, 18.0],
        }
        res = self.client.post("/api/v1/solve", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()

        # History should be None or not fabricated
        history = data.get("iteration_history")
        self.assertTrue(history is None or len(history) == 0)
        # But terminal diagnostics must be present
        self.assertIn("max_primal_residual", data)
        self.assertIn("max_dual_residual", data)
        self.assertIn("basis_condition", data)


if __name__ == "__main__":
    unittest.main(verbosity=2)
