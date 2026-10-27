"""
VØRTEX Phase 9B: Direct Native C++20 / CUDA Executable (vortex_solver_runner) CLI Test Suite
Tests structured JSON input/output, capabilities, device info, stdout piping, and error handling.
"""

import subprocess
import json
import tempfile
import os
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXECUTABLE = REPO_ROOT / "build-cuda" / "vortex_solver_runner.exe"
if not EXECUTABLE.exists():
    EXECUTABLE = REPO_ROOT / "build-cuda" / "vortex_solver_runner"


class TestNativeSolverRunnerCLI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert EXECUTABLE.exists(), f"Native executable not found at {EXECUTABLE}"
        print(f"\n[CLI Test] Target Executable: {EXECUTABLE}")

    def test_01_capabilities_flag(self):
        """Verifies --capabilities outputs structured JSON with actual solver algorithms."""
        res = subprocess.run([str(EXECUTABLE), "--capabilities"], capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(res.returncode, 0)
        data = json.loads(res.stdout)
        self.assertEqual(data["engine"], "VØRTEX Optimization Engine")
        self.assertIn("LP", data["solvers"])
        self.assertIn("MILP", data["solvers"])
        self.assertIn("QP", data["solvers"])
        self.assertTrue(data["supported_file_formats"]["JSON"])
        self.assertFalse(data["supported_file_formats"]["MPS"])

    def test_02_device_info_flag(self):
        """Verifies --device-info outputs valid hardware telemetry."""
        res = subprocess.run([str(EXECUTABLE), "--device-info"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        data = json.loads(res.stdout)
        self.assertIn("cuda_available", data)

    def test_03_lp_json_file(self):
        """Solves LP through --input and --output file arguments."""
        problem = {
            "solver_type": "LP",
            "sense": "MAXIMIZE",
            "backend": "AUTO",
            "c": [3.0, 5.0],
            "A": [[1.0, 0.0], [0.0, 2.0], [3.0, 2.0]],
            "b": [4.0, 12.0, 18.0],
            "constraint_types": ["LEQ", "LEQ", "LEQ"]
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f_in:
            json.dump(problem, f_in)
            in_path = f_in.name

        out_path = in_path + ".out.json"

        try:
            res = subprocess.run([str(EXECUTABLE), "--input", in_path, "--output", out_path], capture_output=True, text=True)
            self.assertEqual(res.returncode, 0)
            with open(out_path, "r") as f_out:
                result = json.load(f_out)
            self.assertTrue(result["success"])
            self.assertEqual(result["status"], "OPTIMAL")
            self.assertAlmostEqual(result["objective_value"], 36.0, places=4)
        finally:
            if os.path.exists(in_path): os.remove(in_path)
            if os.path.exists(out_path): os.remove(out_path)

    def test_04_qp_stdin_stdout_piping(self):
        """Solves QP through stdin/stdout stream piping (--stdin --stdout)."""
        problem = {
            "solver_type": "QP",
            "backend": "AUTO",
            "Q": [[2.0, 0.0], [0.0, 2.0]],
            "c": [-2.0, -5.0],
            "A": [[1.0, -1.0], [1.0, 2.0], [-1.0, 0.0], [0.0, -1.0]],
            "b": [2.0, 6.0, 0.0, 0.0]
        }
        input_data = json.dumps(problem)
        res = subprocess.run(
            [str(EXECUTABLE), "--stdin", "--stdout"],
            input=input_data,
            capture_output=True,
            text=True
        )
        self.assertEqual(res.returncode, 0)
        result = json.loads(res.stdout)
        self.assertTrue(result["success"])
        self.assertEqual(result["status"], "OPTIMAL")
        self.assertAlmostEqual(result["objective_value"], -7.25, places=3)
        self.assertLess(result["duality_gap"], 1e-6)

    def test_05_milp_stdin_stdout_piping(self):
        """Solves MILP through stdin/stdout stream piping."""
        problem = {
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
        input_data = json.dumps(problem)
        res = subprocess.run(
            [str(EXECUTABLE), "--stdin", "--stdout"],
            input=input_data,
            capture_output=True,
            text=True
        )
        self.assertEqual(res.returncode, 0)
        result = json.loads(res.stdout)
        self.assertTrue(result["success"])
        self.assertEqual(result["status"], "OPTIMAL")
        self.assertAlmostEqual(result["objective_value"], 90.0, places=4)
        self.assertGreater(result["nodes_explored"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
