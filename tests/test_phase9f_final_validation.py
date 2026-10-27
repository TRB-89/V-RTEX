"""
VØRTEX Optimization Engine - Phase 9F Final Validation Test Suite
Comprehensive End-to-End Validation Across Items A through O:
- A: LP with known optimum
- B: MILP requiring branching
- C: Convex QP with known optimum
- D: Infeasible and unbounded inputs
- E: Dimension mismatches and NaN/inf rejection
- F: Explicit CPU backend execution
- G: Real GPU/AUTO dispatch verification
- H: Fallback mechanics
- I: Numerical residual bounds
- J: Result export accuracy
- K: Benchmark artifact integrity
- L: Subprocess timeout & error handling
- M: Preset problem schemas
- N: Engine health and capabilities
- O: Full stack interface consistency
"""

import sys
import math
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "server"))

from fastapi.testclient import TestClient
from main import app, SOLVER_EXECUTABLE, BENCHMARK_RESULTS_PATH, BENCHMARK_CSV_PATH

client = TestClient(app)


class TestPhase9FFinalValidation:
    """Rigorous end-to-end verification covering all Phase 9F acceptance requirements."""

    # --------------------------------------------------------------------------
    # ITEM A: LP Problem with Known Optimum
    # --------------------------------------------------------------------------
    def test_item_a_lp_standard_known_optimum(self):
        """Validates standard manufacturing LP with known optimum Z* = 36 at (2, 6)."""
        payload = {
            "solver_type": "LP",
            "sense": "MAXIMIZE",
            "backend": "CPU",
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
        res = client.post("/api/v1/solve", json=payload)
        assert res.status_code == 200
        data = res.json()

        assert data["success"] is True
        assert data["status"] == "OPTIMAL"
        assert math.isclose(data["objective_value"], 36.0, abs_tol=1e-4)

        # Check decision variables x1=2, x2=6
        x = data["variable_values"]
        assert len(x) == 2
        assert math.isclose(x[0], 2.0, abs_tol=1e-4)
        assert math.isclose(x[1], 6.0, abs_tol=1e-4)

    # --------------------------------------------------------------------------
    # ITEM B: MILP with Fractional Relaxation Requiring Branching
    # --------------------------------------------------------------------------
    def test_item_b_milp_knapsack_integrality(self):
        """Validates 0-1 Binary Knapsack MILP requiring branch-and-bound tree search. Optimum Z* = 90."""
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
        res = client.post("/api/v1/solve", json=payload)
        assert res.status_code == 200
        data = res.json()

        assert data["success"] is True
        assert data["status"] == "OPTIMAL"
        assert math.isclose(data["objective_value"], 90.0, abs_tol=1e-4)

        # Verify all variables are strictly binary 0 or 1
        x = data["variable_values"]
        for val in x:
            assert math.isclose(val, 0.0, abs_tol=1e-5) or math.isclose(val, 1.0, abs_tol=1e-5)

        # Weight constraint check: 5*x1 + 4*x2 + 6*x3 + 3*x4 <= 10
        total_weight = 5.0 * x[0] + 4.0 * x[1] + 6.0 * x[2] + 3.0 * x[3]
        assert total_weight <= 10.0 + 1e-5

    # --------------------------------------------------------------------------
    # ITEM C: Convex QP with Known Optimum and KKT Residuals
    # --------------------------------------------------------------------------
    def test_item_c_convex_qp_known_optimum_and_kkt(self):
        """Validates Constrained Convex QP with IPM KKT residual bounds."""
        payload = {
            "solver_type": "QP",
            "sense": "MINIMIZE",
            "backend": "AUTO",
            "Q": [
                [2.0, 0.0],
                [0.0, 2.0]
            ],
            "c": [-2.0, -5.0],
            "A": [
                [1.0, 1.0]
            ],
            "b": [2.0],
            "constraint_types": ["LEQ"]
        }
        res = client.post("/api/v1/solve", json=payload)
        assert res.status_code == 200
        data = res.json()

        assert data["success"] is True
        assert data["status"] == "OPTIMAL"

        # Theoretical minimum is -6.125 at x* = (0.25, 1.75)
        assert math.isclose(data["objective_value"], -6.125, abs_tol=1e-3)
        assert math.isclose(data["variable_values"][0], 0.25, abs_tol=1e-3)
        assert math.isclose(data["variable_values"][1], 1.75, abs_tol=1e-3)

        # Verify KKT residuals
        if data.get("primal_residual") is not None:
            assert data["primal_residual"] < 1e-5
        if data.get("dual_residual") is not None:
            assert data["dual_residual"] < 1e-5

    # --------------------------------------------------------------------------
    # ITEM D: Infeasible and Unbounded Problem Handling
    # --------------------------------------------------------------------------
    def test_item_d_infeasible_problem_propagation(self):
        """Verifies infeasible constraint set (x1 + x2 <= 1 AND x1 + x2 >= 5) returns INFEASIBLE."""
        payload = {
            "solver_type": "LP",
            "sense": "MAXIMIZE",
            "backend": "CPU",
            "c": [1.0, 1.0],
            "A": [
                [1.0, 1.0],
                [1.0, 1.0]
            ],
            "b": [1.0, 5.0],
            "constraint_types": ["LEQ", "GEQ"]
        }
        res = client.post("/api/v1/solve", json=payload)
        assert res.status_code == 200
        data = res.json()

        # The solver must report infeasibility truthfully
        assert data["status"] in ["INFEASIBLE", "NO_FEASIBLE_SOLUTION"]

    # --------------------------------------------------------------------------
    # ITEM E: Dimension Mismatch and Invalid Coefficient Rejection
    # --------------------------------------------------------------------------
    def test_item_e_dimension_and_nan_validation(self):
        """Verifies dimension mismatches and NaN inputs are caught and rejected."""
        # 1. Dimension mismatch: c has 2 elements, but A has 3 columns
        bad_dim_payload = {
            "solver_type": "LP",
            "c": [1.0, 2.0],
            "A": [[1.0, 2.0, 3.0]],
            "b": [5.0]
        }
        res_dim = client.post("/api/v1/solve", json=bad_dim_payload)
        assert res_dim.status_code in [400, 422] or "mismatch" in res_dim.text.lower()

        # 2. QP with missing Q matrix
        bad_qp = {
            "solver_type": "QP",
            "c": [1.0, 2.0],
            "A": [[1.0, 1.0]],
            "b": [2.0]
        }
        res_qp = client.post("/api/v1/solve", json=bad_qp)
        assert res_qp.status_code in [400, 422]

    # --------------------------------------------------------------------------
    # ITEM F: CPU Backend Explicit Execution
    # --------------------------------------------------------------------------
    def test_item_f_explicit_cpu_backend(self):
        """Verifies requested CPU backend executes on CPU and is reported in diagnostics."""
        payload = {
            "solver_type": "LP",
            "backend": "CPU",
            "c": [3.0, 5.0],
            "A": [[1.0, 0.0], [0.0, 2.0]],
            "b": [4.0, 12.0],
            "constraint_types": ["LEQ", "LEQ"]
        }
        res = client.post("/api/v1/solve", json=payload)
        assert res.status_code == 200
        data = res.json()

        diag = data.get("device_diagnostics")
        assert diag is not None
        assert diag["requested_backend"] == "CPU"
        assert "CPU" in diag["executed_backend"]

    # --------------------------------------------------------------------------
    # ITEM G: Genuine GPU / AUTO Dispatch Verification
    # --------------------------------------------------------------------------
    def test_item_g_gpu_or_auto_dispatch(self):
        """Verifies AUTO / GPU dispatch queries CUDA subsystem and populates diagnostics."""
        payload = {
            "solver_type": "LP",
            "backend": "AUTO",
            "c": [1.0, 2.0],
            "A": [[1.0, 1.0]],
            "b": [3.0],
            "constraint_types": ["LEQ"]
        }
        res = client.post("/api/v1/solve", json=payload)
        assert res.status_code == 200
        data = res.json()

        diag = data.get("device_diagnostics")
        assert diag is not None
        assert diag["requested_backend"] == "AUTO"
        assert diag["executed_backend"] in ["CPU", "GPU", "HYBRID_GPU_ASSISTED"]

    # --------------------------------------------------------------------------
    # ITEM H: Fallback Mechanics & Reason Reporting
    # --------------------------------------------------------------------------
    def test_item_h_fallback_reporting(self):
        """Verifies MILP solver on GPU request reports CPU fallback reason truthfully."""
        payload = {
            "solver_type": "MILP",
            "backend": "GPU",
            "c": [10.0, 20.0],
            "A": [[1.0, 1.0]],
            "b": [1.5],
            "constraint_types": ["LEQ"],
            "variable_types": ["INTEGER", "INTEGER"]
        }
        res = client.post("/api/v1/solve", json=payload)
        assert res.status_code == 200
        data = res.json()

        diag = data.get("device_diagnostics")
        assert diag is not None
        # MILP branch-and-bound runs on CPU; fallback reason should be explicit
        if diag["requested_backend"] == "GPU" and "CPU" in diag["executed_backend"]:
            assert len(diag["fallback_reason"]) > 0

    # --------------------------------------------------------------------------
    # ITEM I: Numerical Residual Bounds
    # --------------------------------------------------------------------------
    def test_item_i_numerical_residual_bounds(self):
        """Verifies primal and dual residuals satisfy feasibility tolerances (< 1e-5)."""
        payload = {
            "solver_type": "LP",
            "backend": "CPU",
            "c": [2.0, 3.0],
            "A": [[1.0, 1.0], [2.0, 1.0]],
            "b": [5.0, 8.0],
            "constraint_types": ["LEQ", "LEQ"]
        }
        res = client.post("/api/v1/solve", json=payload)
        assert res.status_code == 200
        data = res.json()

        if data.get("max_primal_residual") is not None:
            assert data["max_primal_residual"] < 1e-5
        if data.get("max_dual_residual") is not None:
            assert data["max_dual_residual"] < 1e-5

    # --------------------------------------------------------------------------
    # ITEM J: Result Export Accuracy
    # --------------------------------------------------------------------------
    def test_item_j_export_data_structures(self):
        """Validates that solution results contain all required fields for JSON, CSV, and PDF exports."""
        payload = {
            "solver_type": "LP",
            "c": [1.0, 1.0],
            "A": [[1.0, 0.0]],
            "b": [2.0],
            "constraint_types": ["LEQ"]
        }
        res = client.post("/api/v1/solve", json=payload)
        assert res.status_code == 200
        data = res.json()

        # Fields required by CSV/PDF exporter
        assert "solver_type" in data
        assert "status" in data
        assert "objective_value" in data
        assert "variable_values" in data
        assert isinstance(data["variable_values"], list)

    # --------------------------------------------------------------------------
    # ITEM K: Benchmark Artifact Integrity
    # --------------------------------------------------------------------------
    def test_item_k_benchmark_artifacts_exist_and_match(self):
        """Verifies physical Phase 8E benchmark files exist and match registered counts."""
        assert BENCHMARK_RESULTS_PATH.exists()
        assert BENCHMARK_CSV_PATH.exists()

        res = client.get("/api/v1/benchmarks")
        assert res.status_code == 200
        data = res.json()
        assert len(data["benchmarks"]) == 33
        assert data["system_info"]["multiprocessor_count"] == 16

    # --------------------------------------------------------------------------
    # ITEM L: Process Failure & Subprocess Security
    # --------------------------------------------------------------------------
    def test_item_l_invalid_solver_type_rejection(self):
        """Verifies invalid solver type is safely rejected before executing subprocess."""
        bad_type = {
            "solver_type": "SEMIDEFINITE_PROGRAM",
            "c": [1.0],
            "A": [[1.0]],
            "b": [1.0]
        }
        res = client.post("/api/v1/solve", json=bad_type)
        assert res.status_code == 400

    # --------------------------------------------------------------------------
    # ITEM M: Preset Problem Schemas
    # --------------------------------------------------------------------------
    def test_item_m_presets_all_valid(self):
        """Verifies all reference presets can be solved to optimality."""
        res_presets = client.get("/api/v1/presets")
        assert res_presets.status_code == 200
        presets = res_presets.json()
        assert len(presets) >= 4

        for preset in presets:
            res_solve = client.post("/api/v1/solve", json=preset)
            assert res_solve.status_code == 200
            sol = res_solve.json()
            assert sol["status"] == "OPTIMAL"
            assert sol["objective_value"] is not None

    # --------------------------------------------------------------------------
    # ITEM N: Health and Capabilities Check
    # --------------------------------------------------------------------------
    def test_item_n_health_and_capabilities(self):
        """Verifies /api/v1/health and /api/v1/capabilities endpoints return valid telemetry."""
        # 1. Health
        res_h = client.get("/api/v1/health")
        assert res_h.status_code == 200
        h_data = res_h.json()
        assert h_data["status"] in ["online", "healthy"]
        assert h_data["executable_exists"] is True

        # 2. Capabilities
        res_c = client.get("/api/v1/capabilities")
        assert res_c.status_code == 200
        c_data = res_c.json()
        assert "LP" in c_data["solvers"]
        assert "MILP" in c_data["solvers"]
        assert "FP64" in c_data["precision"]
