"""
VØRTEX Optimization Engine - Full-Stack & Native Adapter Integration Test Suite
Phase 9: Comprehensive Automated End-to-End Validation
"""

import sys
import os
import unittest
from pathlib import Path

# Add server directory to Python path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "server"))

from fastapi.testclient import TestClient
from main import app, SOLVER_EXECUTABLE


class TestVortexFullStackIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        print("\n=======================================================")
        print("  VØRTEX Phase 9 Full-Stack Integration Test Suite")
        print("=======================================================")
        print(f"Native Solver Executable: {SOLVER_EXECUTABLE}")
        assert SOLVER_EXECUTABLE.exists(), f"Solver executable missing at {SOLVER_EXECUTABLE}"

    def test_01_health_check(self):
        """Verifies API health and executable discovery."""
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "online")
        self.assertTrue(data["executable_exists"])

    def test_02_device_telemetry(self):
        """Verifies real native CUDA hardware query."""
        res = self.client.get("/api/device")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("cuda_available", data)
        self.assertIn("system_info", data)
        self.assertEqual(data["system_info"]["solver_name"], "VØRTEX Optimization Engine")
        if data["cuda_available"]:
            self.assertIsNotNone(data["gpu"])
            self.assertIn("NVIDIA", data["gpu"]["name"])
            self.assertGreater(data["gpu"]["multiprocessor_count"], 0)
            print(f"  [CUDA Active] Device: {data['gpu']['name']} ({data['gpu']['multiprocessor_count']} SMs)")

    def test_03_presets_availability(self):
        """Verifies reference presets across LP, MILP, and QP."""
        res = self.client.get("/api/presets")
        self.assertEqual(res.status_code, 200)
        presets = res.json()
        self.assertGreaterEqual(len(presets), 5)
        preset_types = {p["solver_type"] for p in presets}
        self.assertIn("LP", preset_types)
        self.assertIn("MILP", preset_types)
        self.assertIn("QP", preset_types)

    def test_04_phase8e_benchmarks(self):
        """Verifies Phase 8E empirical benchmark datasets are served intact."""
        res = self.client.get("/api/benchmarks")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("system_info", data)
        self.assertIn("benchmarks", data)
        self.assertGreaterEqual(len(data["benchmarks"]), 30)

    def test_05_lp_standard_solution(self):
        """Solves SIH26119 standard LP through native Revised Simplex."""
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
            "constraint_types": ["LEQ", "LEQ", "LEQ"],
            "options": {"tolerance": 1e-9}
        }
        res = self.client.post("/api/solve", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["status"], "OPTIMAL")
        self.assertAlmostEqual(data["objective_value"], 36.0, places=4)
        self.assertAlmostEqual(data["variable_values"][0], 2.0, places=4)
        self.assertAlmostEqual(data["variable_values"][1], 6.0, places=4)
        self.assertIn("device_diagnostics", data)
        print(f"  [LP Solve] Status: {data['status']}, Z*: {data['objective_value']}, Time: {data['wall_time_ms']:.3f} ms")

    def test_06_lp_two_phase_artificials(self):
        """Solves Two-Phase LP with mixed constraints (LEQ, GEQ, EQ)."""
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
            "constraint_types": ["LEQ", "GEQ", "EQ"],
            "options": {"tolerance": 1e-9}
        }
        res = self.client.post("/api/solve", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["status"], "OPTIMAL")
        self.assertAlmostEqual(data["objective_value"], 24.0, places=4)
        self.assertAlmostEqual(data["variable_values"][0], 8.0, places=4)
        self.assertAlmostEqual(data["variable_values"][1], 0.0, places=4)

    def test_07_milp_knapsack_solution(self):
        """Solves 0-1 Binary Knapsack through Branch-and-Bound MILP engine."""
        payload = {
            "solver_type": "MILP",
            "sense": "MAXIMIZE",
            "backend": "CPU",
            "c": [10.0, 40.0, 30.0, 50.0],
            "A": [
                [5.0, 4.0, 6.0, 3.0]
            ],
            "b": [10.0],
            "constraint_types": ["LEQ"],
            "variable_types": ["BINARY", "BINARY", "BINARY", "BINARY"],
            "bounds": [
                {"lower": 0.0, "upper": 1.0},
                {"lower": 0.0, "upper": 1.0},
                {"lower": 0.0, "upper": 1.0},
                {"lower": 0.0, "upper": 1.0}
            ],
            "options": {"node_limit": 5000}
        }
        res = self.client.post("/api/solve", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["status"], "OPTIMAL")
        self.assertAlmostEqual(data["objective_value"], 90.0, places=4)
        self.assertEqual(round(data["variable_values"][1]), 1)
        self.assertEqual(round(data["variable_values"][3]), 1)
        self.assertGreater(data["nodes_explored"], 0)
        print(f"  [MILP Solve] Status: {data['status']}, Z*: {data['objective_value']}, Nodes: {data['nodes_explored']}")

    def test_08_convex_qp_portfolio_solution(self):
        """Solves Constrained Convex QP using Primal-Dual Interior-Point method."""
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
            "b": [2.0, 6.0, 0.0, 0.0],
            "options": {
                "primal_tolerance": 1e-7,
                "dual_tolerance": 1e-7,
                "duality_gap_tolerance": 1e-7
            }
        }
        res = self.client.post("/api/solve", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["status"], "OPTIMAL")
        self.assertAlmostEqual(data["objective_value"], -7.25, places=3)
        self.assertAlmostEqual(data["variable_values"][0], 1.0, places=3)
        self.assertAlmostEqual(data["variable_values"][1], 2.5, places=3)
        self.assertLess(data["duality_gap"], 1e-6)
        print(f"  [QP Solve] Status: {data['status']}, Z*: {data['objective_value']:.4f}, Duality Gap: {data['duality_gap']:.2e}")

    def test_09_dimension_mismatch_validation(self):
        """Rejects mismatched matrix rows and vector dimensions with HTTP 400."""
        payload = {
            "solver_type": "LP",
            "c": [1.0, 2.0],
            "A": [
                [1.0, 2.0, 3.0]  # 3 columns, expected 2
            ],
            "b": [5.0]
        }
        res = self.client.post("/api/solve", json=payload)
        self.assertEqual(res.status_code, 400)
        self.assertIn("Dimension mismatch", res.json()["detail"])

    def test_10_qp_missing_matrix_q_validation(self):
        """Rejects QP requests missing Hessian matrix Q with HTTP 400."""
        payload = {
            "solver_type": "QP",
            "c": [1.0, 2.0],
            "A": [[1.0, 1.0]],
            "b": [1.0]
        }
        res = self.client.post("/api/solve", json=payload)
        self.assertEqual(res.status_code, 400)
        self.assertIn("requires symmetric matrix Q", res.json()["detail"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
