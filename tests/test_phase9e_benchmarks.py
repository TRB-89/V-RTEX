import sys
import math
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "server"))

from fastapi.testclient import TestClient
from main import app, load_and_validate_benchmarks, compute_benchmark_memory

client = TestClient(app)


class TestPhase9EBenchmarks:
    """Automated test suite validating Phase 9E empirical benchmark ingestion, formulas, and security."""

    def test_01_benchmark_dataset_provenance_and_schema(self):
        """Verifies GET /api/v1/benchmarks returns genuine Phase 8E metadata and schema."""
        response = client.get("/api/v1/benchmarks")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        data = response.json()

        # Provenance verification
        provenance = data.get("provenance")
        assert provenance is not None, "Provenance missing"
        assert provenance["source_file"] == "benchmarks/results/phase8e_results.json"
        assert provenance["record_count"] == 33
        assert len(provenance["checksum_sha256"]) == 64
        assert "cudaEventSynchronize" in provenance["timing_methodology"]

        # Hardware telemetry verification
        sys_info = data.get("system_info")
        assert sys_info is not None, "System info missing"
        assert "RTX 3050" in sys_info["gpu_name"]
        assert sys_info["compute_capability"] == "8.6"
        assert sys_info["multiprocessor_count"] == 16
        assert sys_info["total_vram_bytes"] == 4294443008
        assert "Ryzen 5 6600H" in sys_info["cpu_name"]
        assert sys_info["repetition_count"] == 15
        assert sys_info["warmup_count"] == 5

        # Benchmark records count
        benchmarks = data.get("benchmarks", [])
        assert len(benchmarks) == 33, f"Expected 33 benchmarks, got {len(benchmarks)}"

    def test_02_speedup_formula_and_slowdown_labeling(self):
        """Validates that speedup_e2e = CPU_median / GPU_E2E_median and slowdowns (< 1.0x) are flagged."""
        response = client.get("/api/v1/benchmarks")
        assert response.status_code == 200
        benchmarks = response.json()["benchmarks"]

        slowdown_count = 0
        speedup_count = 0

        for r in benchmarks:
            cpu_med = r["cpu_timing_ms"]["median"]
            gpu_e2e_med = r["gpu_e2e_timing_ms"]["median"]
            expected_speedup = round(cpu_med / gpu_e2e_med, 6)

            assert math.isclose(r["speedup_e2e"], expected_speedup, rel_tol=1e-4), \
                f"Speedup mismatch for {r['name']}: got {r['speedup_e2e']}, expected {expected_speedup}"

            if r["speedup_e2e"] < 1.0:
                assert r["is_slowdown"] is True, f"Record {r['name']} with speedup {r['speedup_e2e']} should be labeled as slowdown"
                slowdown_count += 1
            else:
                assert r["is_slowdown"] is False
                speedup_count += 1

        # Truthful distribution check: small problem sizes dominate dataset with slowdowns
        assert slowdown_count > 15, f"Expected substantial slowdown count on small problems, got {slowdown_count}"
        assert speedup_count > 5, f"Expected large problems to achieve speedup, got {speedup_count}"

        # Verify specific known anchors:
        # 1. Dense GEMV 128x128 Normal: slowdown ~0.20x
        gemv_small = next(r for r in benchmarks if r["name"] == "Dense GEMV 128x128 Normal")
        assert gemv_small["speedup_e2e"] < 0.25
        assert gemv_small["is_slowdown"] is True

        # 2. Simplex Pricing 1024x4096: extreme speedup > 900x
        pricing_large = next(r for r in benchmarks if r["name"] == "Simplex Pricing m=1024, n=4096")
        assert pricing_large["speedup_e2e"] > 900.0
        assert pricing_large["is_slowdown"] is False

    def test_03_time_and_memory_unit_consistency(self):
        """Validates analytical memory conversions (bytes, KiB, MiB) and timing units."""
        response = client.get("/api/v1/benchmarks")
        assert response.status_code == 200
        benchmarks = response.json()["benchmarks"]

        for r in benchmarks:
            mem = r["memory"]
            assert mem["host_bytes"] > 0
            assert mem["gpu_bytes"] > 0

            # KiB and MiB mathematical consistency
            expected_kib = round(mem["host_bytes"] / 1024.0, 3)
            expected_mib = round(mem["host_bytes"] / (1024.0 * 1024.0), 4)

            assert math.isclose(mem["host_kib"], expected_kib, rel_tol=1e-3)
            assert math.isclose(mem["host_mib"], expected_mib, rel_tol=1e-3)

            # Timing consistency: min <= median <= max
            cpu = r["cpu_timing_ms"]
            if cpu.get("min") and cpu.get("max"):
                assert cpu["min"] <= cpu["median"] + 1e-6
                assert cpu["median"] <= cpu["max"] + 1e-6

            gpu = r["gpu_e2e_timing_ms"]
            if gpu.get("min") and gpu.get("max") and gpu["min"] > 0:
                assert gpu["min"] <= gpu["median"] + 1e-6
                assert gpu["median"] <= gpu["max"] + 1e-6

    def test_04_single_benchmark_retrieval_and_traversal_prevention(self):
        """Tests GET /api/v1/benchmarks/{id} with valid slugs, indices, and directory traversal rejection."""
        # 1. Lookup by slug
        res_slug = client.get("/api/v1/benchmarks/dense_gemv_128x128_normal")
        assert res_slug.status_code == 200
        assert res_slug.json()["benchmark"]["name"] == "Dense GEMV 128x128 Normal"

        # 2. Lookup by index
        res_idx = client.get("/api/v1/benchmarks/0")
        assert res_idx.status_code == 200
        assert res_idx.json()["benchmark"]["name"] == "Dense GEMV 128x128 Normal"

        # 3. Directory traversal rejection
        res_trav1 = client.get("/api/v1/benchmarks/..%2f..%2fetc%2fpasswd")
        assert res_trav1.status_code in [400, 404]

        # 4. Non-existent identifier
        res_missing = client.get("/api/v1/benchmarks/non_existent_solver_benchmark_xyz")
        assert res_missing.status_code == 404

    def test_05_artifact_download_and_security(self):
        """Tests safe artifact downloading and whitelist enforcement."""
        # 1. Valid JSON artifact
        res_json = client.get("/api/v1/benchmarks/artifacts/phase8e_results.json")
        assert res_json.status_code == 200
        assert "application/json" in res_json.headers.get("content-type", "")

        # 2. Valid CSV artifact
        res_csv = client.get("/api/v1/benchmarks/artifacts/phase8e_results.csv")
        assert res_csv.status_code == 200
        assert "text/csv" in res_csv.headers.get("content-type", "")
        assert "Category,TestName,Rows,Cols" in res_csv.text

        # 3. Unauthorized file access rejected with 403
        res_unauth = client.get("/api/v1/benchmarks/artifacts/main.py")
        assert res_unauth.status_code == 403

        # 4. Traversal in artifact endpoint rejected
        res_art_trav = client.get("/api/v1/benchmarks/artifacts/..%2fmain.py")
        assert res_art_trav.status_code in [403, 404]

    def test_06_numerical_accuracy_and_missing_values(self):
        """Verifies accuracy metrics are non-negative, and missing values are marked properly."""
        response = client.get("/api/v1/benchmarks")
        assert response.status_code == 200
        benchmarks = response.json()["benchmarks"]

        for r in benchmarks:
            if r.get("max_absolute_error") is not None:
                assert r["max_absolute_error"] >= 0.0
                assert not math.isnan(r["max_absolute_error"])
                # FP64 elementwise GEMV errors should be strictly bounded by 1e-11
                if "Dense GEMV" in r["category"]:
                    assert r["max_absolute_error"] < 1e-11

            if r.get("scaled_relative_error") is not None:
                assert r["scaled_relative_error"] >= 0.0

    def test_07_comparison_compatibility_filtering(self):
        """Validates workload categorizations and microbenchmark vs full-solver distinctions."""
        response = client.get("/api/v1/benchmarks")
        assert response.status_code == 200
        benchmarks = response.json()["benchmarks"]

        categories = {r["category"] for r in benchmarks}
        assert "Dense GEMV" in categories
        assert "Sparse SpMV" in categories
        assert "Simplex Pricing" in categories
        assert "Full Solver: Dense LP" in categories
        assert "Full Solver: Sparse LP" in categories
        assert "Full Solver: Convex QP" in categories
        assert "Data Transfer & Reuse" in categories

        for r in benchmarks:
            if "Full Solver" in r["category"]:
                assert r["is_microbenchmark"] is False, f"Expected {r['name']} to be a full-solver workload"
            else:
                assert r["is_microbenchmark"] is True, f"Expected {r['name']} to be a microbenchmark workload"

    def test_08_memory_calculation_helper(self):
        """Directly tests compute_benchmark_memory analytical models."""
        # 1. Dense 1000x1000 FP64: (1,000,000 + 1,000 + 1,000) * 8 = 8,016,000 bytes
        mem_dense = compute_benchmark_memory("Dense GEMV", 1000, 1000, 1000000)
        assert mem_dense.host_bytes == 8016000
        assert "Dense" in mem_dense.storage_format

        # 2. Sparse 1000x1000 (10,000 nnz): 10,000*12 + 1001*4 + 2000*8 = 120,000 + 4,004 + 16,000 = 140,004 bytes
        mem_sparse = compute_benchmark_memory("Sparse SpMV", 1000, 1000, 10000)
        assert mem_sparse.host_bytes == 140004
        assert "CSR" in mem_sparse.storage_format
        # Verify 98% memory reduction compared to dense
        assert mem_sparse.host_bytes < mem_dense.host_bytes * 0.05
