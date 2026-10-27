"""
VØRTEX Phase 9B: Production API v1 Automated Unit and Integration Test Suite
Validates Native C++20 / CUDA Solver Integration across LP, MILP, and QP
"""

import sys
import unittest
from pathlib import Path

# Add server directory to Python path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "server"))

from fastapi.testclient import TestClient
from main import app, SOLVER_EXECUTABLE


class TestVortexApiV1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        print("\n=======================================================")
        print("  VØRTEX Phase 9B: API v1 Integration Test Suite       ")
        print("=======================================================")
        print(f"Target Binary: {SOLVER_EXECUTABLE}")
        assert SOLVER_EXECUTABLE.exists(), f"Native executable missing at {SOLVER_EXECUTABLE}"

    def test_01_health_v1(self):
        """Verifies GET /api/v1/health endpoint contract."""
        res = self.client.get("/api/v1/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "online")
        self.assertEqual(data["version"], "3.0.0")
        self.assertTrue(data["executable_exists"])

    def test_02_capabilities_v1(self):
        """Verifies GET /api/v1/capabilities native solver capabilities contract."""
        res = self.client.get("/api/v1/capabilities")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["engine"], "VØRTEX Optimization Engine")
        self.assertIn("LP", data["solvers"])
        self.assertIn("MILP", data["solvers"])
        self.assertIn("QP", data["solvers"])
        self.assertTrue(data["supported_file_formats"]["JSON"])
        self.assertFalse(data["supported_file_formats"]["MPS"])  # Truthful reporting
        self.assertIn("cuda", data)
        print(f"  [Capabilities] CUDA Available: {data['cuda']['available']}, Formats: JSON only")

    def test_03_solve_lp_standard(self):
        """Verifies POST /api/v1/solve for standard SIH26119 LP (Z* = 36.0000)."""
        payload = {
            "solver_type": "LP",
            "sense": "MAXIMIZE",
            "backend": "AUTO",
            "c": [3.0, 5.0],
            "A": [
                [1.0, 0.0],
                [0.0, 2.0],
                [3.0, 2.0]
            ],
            "b": [4.0, 12.0, 18.0],
            "constraint_types": ["LEQ", "LEQ", "LEQ"]
        }
        res = self.client.post("/api/v1/solve", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["status"], "OPTIMAL")
        self.assertAlmostEqual(data["objective_value"], 36.0, places=4)
        self.assertAlmostEqual(data["variable_values"][0], 2.0, places=4)
        self.assertAlmostEqual(data["variable_values"][1], 6.0, places=4)
        self.assertIn("device_diagnostics", data)
        print(f"  [LP v1] Status: {data['status']}, Obj: {data['objective_value']}, Time: {data['wall_time_ms']:.3f} ms")

    def test_04_solve_lp_two_phase(self):
        """Verifies POST /api/v1/solve for Two-Phase Simplex LP (Z* = 24.0000)."""
        payload = {
            "solver_type": "LP",
            "sense": "MAXIMIZE",
            "backend": "AUTO",
            "c": [3.0, 2.0],
            "A": [
                [2.0, 1.0],
                [2.0, 3.0],
                [1.0, 1.0]
            ],
            "b": [18.0, 12.0, 8.0],
            "constraint_types": ["LEQ", "GEQ", "EQ"]
        }
        res = self.client.post("/api/v1/solve", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["status"], "OPTIMAL")
        self.assertAlmostEqual(data["objective_value"], 24.0, places=4)
        self.assertAlmostEqual(data["variable_values"][0], 8.0, places=4)
        self.assertAlmostEqual(data["variable_values"][1], 0.0, places=4)

    def test_05_solve_milp_knapsack(self):
        """Verifies POST /api/v1/solve for Branch-and-Bound MILP (Z* = 90.0000)."""
        payload = {
            "solver_type": "MILP",
            "sense": "MAXIMIZE",
            "backend": "CPU",
            "c": [10.0, 40.0, 30.0, 50.0],
            "A": [[5.0, 4.0, 6.0, 3.0]],
            "b": [10.0],
            "constraint_types": ["LEQ"],
            "variable_types": ["BINARY", "BINARY", "BINARY", "BINARY"],
            "bounds": [
                {"lower": 0.0, "upper": 1.0},
                {"lower": 0.0, "upper": 1.0},
                {"lower": 0.0, "upper": 1.0},
                {"lower": 0.0, "upper": 1.0}
            ]
        }
        res = self.client.post("/api/v1/solve", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["status"], "OPTIMAL")
        self.assertAlmostEqual(data["objective_value"], 90.0, places=4)
        self.assertEqual(round(data["variable_values"][1]), 1)
        self.assertEqual(round(data["variable_values"][3]), 1)
        print(f"  [MILP v1] Status: {data['status']}, Obj: {data['objective_value']}, Explored: {data['nodes_explored']}")

    def test_06_solve_qp_portfolio(self):
        """Verifies POST /api/v1/solve for Convex QP Interior-Point Solver (Z* = -7.2500)."""
        payload = {
            "solver_type": "QP",
            "backend": "AUTO",
            "Q": [
                [2.0, 0.0],
                [0.0, 2.0]
            ],
            "c": [-2.0, -5.0],
            "A": [
                [1.0, -1.0],
                [1.0, 2.0],
                [-1.0, 0.0],
                [0.0, -1.0]
            ],
            "b": [2.0, 6.0, 0.0, 0.0]
        }
        res = self.client.post("/api/v1/solve", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["status"], "OPTIMAL")
        self.assertAlmostEqual(data["objective_value"], -7.25, places=3)
        self.assertLess(data["duality_gap"], 1e-6)
        print(f"  [QP v1] Status: {data['status']}, Obj: {data['objective_value']:.4f}, Duality Gap: {data['duality_gap']:.2e}")

    def test_07_invalid_dimension_mismatch(self):
        """Verifies dimension mismatch is rejected with HTTP 400."""
        payload = {
            "solver_type": "LP",
            "c": [1.0, 2.0],
            "A": [[1.0, 2.0, 3.0]],  # 3 columns, expected 2
            "b": [1.0]
        }
        res = self.client.post("/api/v1/solve", json=payload)
        self.assertEqual(res.status_code, 400)
        self.assertIn("Dimension mismatch", res.json()["detail"])

    def test_08_invalid_solver_type(self):
        """Verifies unsupported solver formulation is rejected with HTTP 400."""
        payload = {
            "solver_type": "SDP",  # Semidefinite programming not supported
            "c": [1.0, 2.0],
            "A": [[1.0, 2.0]],
            "b": [1.0]
        }
        res = self.client.post("/api/v1/solve", json=payload)
        self.assertEqual(res.status_code, 400)
        self.assertIn("Unsupported solver_type", res.json()["detail"])

    def test_09_qp_missing_matrix_q(self):
        """Verifies QP request missing Hessian Q is rejected with HTTP 400."""
        payload = {
            "solver_type": "QP",
            "c": [1.0, 2.0],
            "A": [[1.0, 2.0]],
            "b": [1.0]
        }
        res = self.client.post("/api/v1/solve", json=payload)
        self.assertEqual(res.status_code, 400)
        self.assertIn("requires symmetric matrix Q", res.json()["detail"])

    def test_10_cors_headers(self):
        """Verifies CORS headers for allowed origins."""
        headers = {
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
        }
        res = self.client.options("/api/v1/solve", headers=headers)
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.headers.get("access-control-allow-origin"), "http://localhost:5173")


if __name__ == "__main__":
    unittest.main(verbosity=2)
