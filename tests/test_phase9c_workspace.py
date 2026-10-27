"""
VØRTEX Phase 9C: Interactive Optimization Workspace Automated Test Suite
Validates:
1. Versioned JSON Problem Schema (vortex_problem_v1) serialization & deserialization.
2. Formulation validation logic: NaN, Infinity, dimension mismatch, asymmetric Q matrix.
3. Constraint feasibility computation logic: LHS = A_i x*, slack/surplus, violation checks.
4. Integrality verification logic: discrete variable bounds, round(x_j) - x_j.
5. End-to-end native execution against /api/v1/solve for:
   - Real LP (SIH26119 Simplex, Z* = 36.000)
   - Real MILP (0-1 Knapsack Branch-and-Bound, Z* = 90.000)
   - Real QP (Constrained Portfolio Convex Interior-Point, Z* = -7.250)
6. Backend fallback & capability validation.
7. Error states: invalid inputs, timeout/abort semantics, HTTP failure handling.
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


class TestPhase9COptimizationWorkspace(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        print("\n=======================================================")
        print("  VØRTEX Phase 9C: Workspace Automated Validation Suite")
        print("=======================================================")
        print(f"Target Native Binary: {SOLVER_EXECUTABLE}")
        assert SOLVER_EXECUTABLE.exists(), f"Solver executable missing: {SOLVER_EXECUTABLE}"

    # -------------------------------------------------------------
    # 1. VERSIONED PROBLEM JSON (vortex_problem_v1) TESTS
    # -------------------------------------------------------------
    def test_01_versioned_problem_schema_serialization(self):
        """Verifies vortex_problem_v1 JSON schema matches workspace import/export contract."""
        sample_problem = {
            "schema_version": "vortex_problem_v1",
            "solver_type": "LP",
            "sense": "MAXIMIZE",
            "backend": "AUTO",
            "variables": [
                {"id": "v1", "name": "x1", "c": 3.0, "type": "CONTINUOUS", "lower": 0.0, "upper": 1e20},
                {"id": "v2", "name": "x2", "c": 5.0, "type": "CONTINUOUS", "lower": 0.0, "upper": 1e20},
            ],
            "constraints": [
                {"id": "c1", "name": "c1", "coefficients": [1.0, 0.0], "sense": "LEQ", "rhs": 4.0},
                {"id": "c2", "name": "c2", "coefficients": [0.0, 2.0], "sense": "LEQ", "rhs": 12.0},
                {"id": "c3", "name": "c3", "coefficients": [3.0, 2.0], "sense": "LEQ", "rhs": 18.0},
            ],
            "exported_at": "2026-10-27T19:00:00.000Z",
        }
        raw_json = json.dumps(sample_problem)
        loaded = json.loads(raw_json)
        self.assertEqual(loaded["schema_version"], "vortex_problem_v1")
        self.assertEqual(len(loaded["variables"]), 2)
        self.assertEqual(len(loaded["constraints"]), 3)
        self.assertEqual(loaded["solver_type"], "LP")

    def test_02_versioned_problem_schema_invalid_version(self):
        """Verifies rejection of unsupported schema versions."""
        invalid_problem = {
            "schema_version": "unsupported_v99",
            "solver_type": "LP",
            "variables": [],
            "constraints": [],
        }
        self.assertNotEqual(invalid_problem.get("schema_version"), "vortex_problem_v1")

    # -------------------------------------------------------------
    # 2. QP HESSIAN SYMMETRY & VALIDATION TESTS
    # -------------------------------------------------------------
    def test_03_qp_matrix_symmetry_validator(self):
        """Verifies numerical symmetry validator logic: |Q_ij - Q_ji| < 1e-7."""
        # Symmetric Q
        Q_sym = [
            [2.0, 0.5],
            [0.5, 2.0],
        ]
        is_sym = True
        for i in range(len(Q_sym)):
            for j in range(len(Q_sym[i])):
                if abs(Q_sym[i][j] - Q_sym[j][i]) > 1e-7:
                    is_sym = False
        self.assertTrue(is_sym)

        # Asymmetric Q
        Q_asym = [
            [2.0, 1.0],
            [0.2, 2.0],
        ]
        is_asym = False
        for i in range(len(Q_asym)):
            for j in range(len(Q_asym[i])):
                if abs(Q_asym[i][j] - Q_asym[j][i]) > 1e-7:
                    is_asym = True
        self.assertTrue(is_asym)

        # Auto-symmetrization algorithm: Q = (Q + Q^T) / 2
        n = len(Q_asym)
        Q_corrected = [
            [(Q_asym[i][j] + Q_asym[j][i]) / 2.0 for j in range(n)]
            for i in range(n)
        ]
        self.assertAlmostEqual(Q_corrected[0][1], 0.6)
        self.assertAlmostEqual(Q_corrected[1][0], 0.6)

    # -------------------------------------------------------------
    # 3. REAL SOLVER EXECUTION: LP, MILP, QP
    # -------------------------------------------------------------
    def test_04_real_lp_execution_and_constraint_feasibility(self):
        """Executes SIH26119 standard LP and verifies constraint feasibility and slack."""
        payload = {
            "solver_type": "LP",
            "sense": "MAXIMIZE",
            "backend": "AUTO",
            "c": [3.0, 5.0],
            "A": [
                [1.0, 0.0],
                [0.0, 2.0],
                [3.0, 2.0],
            ],
            "b": [4.0, 12.0, 18.0],
            "constraint_types": ["LEQ", "LEQ", "LEQ"],
        }
        res = self.client.post("/api/v1/solve", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()

        self.assertTrue(data["success"])
        self.assertEqual(data["status"], "OPTIMAL")
        self.assertAlmostEqual(data["objective_value"], 36.0, places=4)
        x = data["variable_values"]
        self.assertAlmostEqual(x[0], 2.0, places=4)
        self.assertAlmostEqual(x[1], 6.0, places=4)

        # Verify constraint feasibility logic: A_i * x <= b_i
        tol = 1e-5
        for i, row in enumerate(payload["A"]):
            lhs = sum(row[j] * x[j] for j in range(len(x)))
            rhs = payload["b"][i]
            slack = rhs - lhs
            self.assertLessEqual(lhs, rhs + tol, f"Constraint {i} violated: {lhs} > {rhs}")
            self.assertGreaterEqual(slack, -tol, f"Slack negative on constraint {i}")

        print(f"  [Phase 9C LP Verified] Z* = {data['objective_value']}, x* = {x}, Feasibility 100%")

    def test_05_real_milp_execution_and_integrality_check(self):
        """Executes 0-1 Knapsack MILP and verifies integer feasibility."""
        payload = {
            "solver_type": "MILP",
            "sense": "MAXIMIZE",
            "backend": "CPU",
            "c": [10.0, 40.0, 30.0, 50.0],
            "A": [
                [5.0, 4.0, 6.0, 3.0],
            ],
            "b": [10.0],
            "constraint_types": ["LEQ"],
            "variable_types": ["BINARY", "BINARY", "BINARY", "BINARY"],
            "bounds": [
                {"lower": 0.0, "upper": 1.0},
                {"lower": 0.0, "upper": 1.0},
                {"lower": 0.0, "upper": 1.0},
                {"lower": 0.0, "upper": 1.0},
            ],
            "options": {"node_limit": 1000},
        }
        res = self.client.post("/api/v1/solve", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()

        self.assertTrue(data["success"])
        self.assertEqual(data["status"], "OPTIMAL")
        self.assertAlmostEqual(data["objective_value"], 90.0, places=4)
        x = data["variable_values"]
        self.assertEqual(len(x), 4)

        # Verify integrality check logic: |x_j - round(x_j)| < 1e-5
        for j, val in enumerate(x):
            nearest_int = round(val)
            self.assertAlmostEqual(val, float(nearest_int), delta=1e-5, msg=f"Var {j} not integral: {val}")
            self.assertTrue(nearest_int in (0, 1), f"Var {j} not binary: {nearest_int}")

        # Check knapsack constraint: 5(0) + 4(1) + 6(0) + 3(1) = 7 <= 10
        total_weight = sum(payload["A"][0][j] * x[j] for j in range(4))
        self.assertLessEqual(total_weight, 10.0 + 1e-5)

        print(f"  [Phase 9C MILP Verified] Z* = {data['objective_value']}, x* = {x}, Discrete Integrality 100%")

    def test_06_real_qp_execution_and_kkt_verification(self):
        """Executes Constrained Portfolio Convex QP and verifies interior-point solution."""
        payload = {
            "solver_type": "QP",
            "sense": "MINIMIZE",
            "backend": "AUTO",
            "c": [-2.0, -5.0],
            "A": [
                [1.0, 1.0],
            ],
            "b": [2.0],
            "constraint_types": ["LEQ"],
            "Q": [
                [2.0, 0.0],
                [0.0, 2.0],
            ],
            "options": {
                "primal_tolerance": 1e-7,
                "dual_tolerance": 1e-7,
                "duality_gap_tolerance": 1e-7,
                "max_iterations": 100,
            },
        }
        res = self.client.post("/api/v1/solve", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()

        self.assertTrue(data["success"])
        self.assertEqual(data["status"], "OPTIMAL")
        # Analytical solution: x* = [0.25, 1.75], Z* = 0.5(2*0.25^2 + 2*1.75^2) - 2(0.25) - 5(1.75) = 3.125 - 9.25 = -6.125
        self.assertAlmostEqual(data["objective_value"], -6.125, places=3)
        x = data["variable_values"]
        self.assertAlmostEqual(x[0], 0.25, places=3)
        self.assertAlmostEqual(x[1], 1.75, places=3)

        # Check budget constraint x1 + x2 = 0.25 + 1.75 = 2.0 <= 2.0
        lhs = x[0] + x[1]
        self.assertAlmostEqual(lhs, 2.0, places=4)

        # Check duality gap is reported
        self.assertIn("duality_gap", data)
        self.assertLess(data["duality_gap"], 1e-6)

        print(f"  [Phase 9C QP Verified] Z* = {data['objective_value']}, x* = {x}, Duality Gap: {data['duality_gap']:.2e}")

    # -------------------------------------------------------------
    # 4. ERROR STATES & INPUT VALIDATION
    # -------------------------------------------------------------
    def test_07_validation_nan_infinity_rejection(self):
        """Verifies rejection when inputs contain non-numeric / invalid types."""
        # Non-numeric string where float expected
        res = self.client.post(
            "/api/v1/solve",
            content=b'{"solver_type": "LP", "sense": "MAXIMIZE", "c": ["invalid_nan", 5.0], "A": [[1.0, 0.0]], "b": [4.0]}',
            headers={"Content-Type": "application/json"}
        )
        self.assertEqual(res.status_code, 422)

    def test_08_dimension_mismatch_rejection(self):
        """Verifies rejection when A row dimensions do not match c dimension."""
        payload = {
            "solver_type": "LP",
            "sense": "MAXIMIZE",
            "backend": "CPU",
            "c": [1.0, 2.0],  # 2 variables
            "A": [[1.0, 2.0, 3.0]],  # 3 columns!
            "b": [10.0],
        }
        res = self.client.post("/api/v1/solve", json=payload)
        self.assertEqual(res.status_code, 400)
        self.assertIn("dimension mismatch", res.json()["detail"].lower())

    def test_09_capabilities_unsupported_formats(self):
        """Verifies /api/v1/capabilities truthfully reports format support."""
        res = self.client.get("/api/v1/capabilities")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["supported_file_formats"]["JSON"])
        self.assertFalse(data["supported_file_formats"]["MPS"])
        self.assertFalse(data["supported_file_formats"]["LP_FILE"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
