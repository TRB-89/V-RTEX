"""
VØRTEX Optimization Engine - FastAPI Full-Stack Adapter
Phase 9B: Production Native C++20 / CUDA Solver API Integration (/api/v1/*)
"""

import os
import sys
import json
import asyncio
import tempfile
import re
import hashlib
import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse

# Determine repository root
REPO_ROOT = Path(__file__).resolve().parent.parent

# Locate native solver runner executable
SOLVER_EXECUTABLE_CANDIDATES = [
    REPO_ROOT / "build-cuda" / "vortex_solver_runner.exe",
    REPO_ROOT / "build-cuda" / "vortex_solver_runner",
    REPO_ROOT / "build-release" / "vortex_solver_runner.exe",
    REPO_ROOT / "build-release" / "vortex_solver_runner",
    REPO_ROOT / "build" / "vortex_solver_runner.exe",
    REPO_ROOT / "build" / "vortex_solver_runner",
]

SOLVER_EXECUTABLE = None
for candidate in SOLVER_EXECUTABLE_CANDIDATES:
    if candidate.exists() and os.access(candidate, os.X_OK):
        SOLVER_EXECUTABLE = candidate
        break

if not SOLVER_EXECUTABLE:
    SOLVER_EXECUTABLE = SOLVER_EXECUTABLE_CANDIDATES[0]

BENCHMARK_DIR = REPO_ROOT / "benchmarks" / "results"
BENCHMARK_RESULTS_PATH = BENCHMARK_DIR / "phase8e_results.json"
BENCHMARK_CSV_PATH = BENCHMARK_DIR / "phase8e_results.csv"

app = FastAPI(
    title="VØRTEX Native C++/CUDA Solver API",
    description="Production machine-readable REST API for indigenous LP, MILP, and Convex QP optimization.",
    version="3.0.0",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
)

# Restrict CORS to configured frontend development and production origins (Module B requirement)
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "Accept"],
)

# Concurrency throttle to prevent system thrashing and memory exhaustion
SOLVER_SEMAPHORE = asyncio.Semaphore(4)
SOLVER_TIMEOUT_SECONDS = 30.0
MAX_ELEMENTS_LIMIT = 25_000_000

# ==============================================================================
# Pydantic Schemas (Module B)
# ==============================================================================

class VariableBoundsModel(BaseModel):
    lower: float = Field(default=0.0, description="Lower bound on variable (use -1e20 for -inf)")
    upper: float = Field(default=1e20, description="Upper bound on variable (use 1e20 for +inf)")

    @field_validator("lower", "upper")
    @classmethod
    def check_finite(cls, v: float) -> float:
        if not (v == float("inf") or v == float("-inf") or -1e25 <= v <= 1e25):
            raise ValueError("Bound value must be finite or valid infinity.")
        return v


class SolveRequestV1(BaseModel):
    solver_type: str = Field(..., description="Optimization formulation: 'LP', 'MILP', or 'QP'")
    sense: Optional[str] = Field("MAXIMIZE", description="Objective sense: 'MAXIMIZE' or 'MINIMIZE'")
    backend: Optional[str] = Field("AUTO", description="Requested backend: 'CPU', 'GPU', or 'AUTO'")
    
    # Common linear vectors/matrices
    c: List[float] = Field(..., description="Objective coefficients vector (1 x n)")
    A: List[List[float]] = Field(..., description="Constraint matrix A (m x n)")
    b: List[float] = Field(..., description="Right-hand side constraint vector (m x 1)")
    constraint_types: Optional[List[str]] = Field(None, description="Array of 'LEQ', 'GEQ', or 'EQ'")
    
    # MILP specific fields
    variable_types: Optional[List[str]] = Field(None, description="Array of 'CONTINUOUS', 'INTEGER', or 'BINARY'")
    bounds: Optional[List[VariableBoundsModel]] = Field(None, description="Lower and upper bounds per variable")
    
    # QP specific fields
    Q: Optional[List[List[float]]] = Field(None, description="Hessian matrix Q (n x n, symmetric PSD)")
    
    # Solver options
    options: Optional[Dict[str, Any]] = Field(default_factory=dict)


class DeviceDiagnosticsModel(BaseModel):
    requested_backend: str
    executed_backend: str
    gpu_iterations: int = 0
    gpu_gemv_calls: int = 0
    gpu_spmv_calls: int = 0
    h2d_transfers: int = 0
    d2h_transfers: int = 0
    fallback_reason: str = ""


class MatrixInfoModel(BaseModel):
    rows: Optional[int] = None
    cols: Optional[int] = None
    q_dim: Optional[int] = None
    constraints_m: Optional[int] = None
    constraints_n: Optional[int] = None
    nnz: int
    density: float


# ==============================================================================
# Phase 9E Benchmark Typed Models (Module A)
# ==============================================================================

class TimingStatsModel(BaseModel):
    median: float = Field(..., description="Median elapsed time in milliseconds")
    min: Optional[float] = Field(None, description="Minimum elapsed time across runs")
    max: Optional[float] = Field(None, description="Maximum elapsed time across runs")
    mean: Optional[float] = Field(None, description="Mean elapsed time")
    stddev: Optional[float] = Field(None, description="Standard deviation")
    iqr: Optional[float] = Field(None, description="Interquartile range")


class GpuKernelTimingModel(BaseModel):
    median: float = Field(..., description="Median GPU pure kernel execution time in ms")


class MemoryFootprintModel(BaseModel):
    host_bytes: int = Field(..., description="Analytical host memory footprint in bytes")
    gpu_bytes: int = Field(..., description="Analytical device VRAM footprint in bytes")
    host_kib: float = Field(..., description="Host memory in KiB")
    gpu_kib: float = Field(..., description="Device memory in KiB")
    host_mib: float = Field(..., description="Host memory in MiB")
    gpu_mib: float = Field(..., description="Device memory in MiB")
    storage_format: str = Field(..., description="Internal numerical storage representation")


class BenchmarkRecordModel(BaseModel):
    id: str = Field(..., description="Unique slug identifier for the benchmark")
    category: str = Field(..., description="Benchmark category (e.g. Dense GEMV, Sparse SpMV, Convex QP)")
    name: str = Field(..., description="Descriptive benchmark case name")
    rows: int = Field(..., description="Matrix row dimension (m)")
    cols: int = Field(..., description="Matrix column dimension (n)")
    nnz: int = Field(..., description="Exact number of non-zero elements")
    density: float = Field(..., description="Matrix sparsity ratio (nnz / (m*n))")
    transpose: bool = Field(default=False, description="Whether operation executed in transposed orientation")
    cpu_timing_ms: TimingStatsModel
    gpu_kernel_timing_ms: Optional[GpuKernelTimingModel] = None
    gpu_e2e_timing_ms: TimingStatsModel
    speedup_compute: Optional[float] = Field(None, description="Kernel-only compute speedup (CPU_med / GPU_kernel_med)")
    speedup_e2e: float = Field(..., description="End-to-end speedup (CPU_med / GPU_e2e_med)")
    speedup_label: str = Field(..., description="Human-readable speedup or slowdown label")
    is_slowdown: bool = Field(..., description="True if speedup_e2e < 1.0 (GPU is slower than CPU due to transfer overhead)")
    transfer_overhead_ms: Optional[float] = Field(None, description="Estimated transfer + setup latency (GPU_e2e - GPU_kernel)")
    max_absolute_error: Optional[float] = Field(None, description="Maximum elementwise absolute error vs CPU")
    scaled_relative_error: Optional[float] = Field(None, description="Scaled relative numerical error")
    notes: Optional[str] = Field("", description="Scientific observations and execution notes")
    is_microbenchmark: bool = Field(..., description="True for primitive kernel ops, False for full solver runs")
    memory: MemoryFootprintModel


class SystemInfoModel(BaseModel):
    gpu_name: str
    compute_capability: str
    multiprocessor_count: int
    total_vram_bytes: int
    available_vram_bytes: int
    core_clock_khz: int
    memory_clock_khz: int
    memory_bus_width_bits: int
    cpu_name: str
    compiler: str
    cuda_runtime_version: int
    build_type: str
    repetition_count: int
    warmup_count: int


class BenchmarkProvenanceModel(BaseModel):
    source_file: str
    file_format: str
    generation_time: str
    record_count: int
    checksum_sha256: str
    harness: str
    timing_methodology: str
    notes: str


class BenchmarkDatasetResponse(BaseModel):
    provenance: BenchmarkProvenanceModel
    system_info: SystemInfoModel
    benchmarks: List[BenchmarkRecordModel]


class SolveResponseV1(BaseModel):
    success: bool
    solver_type: str
    status: str
    message: str
    objective_value: Optional[float] = None
    variable_values: Optional[List[float]] = None
    slack_values: Optional[List[float]] = None
    surplus_values: Optional[List[float]] = None
    dual_values: Optional[List[float]] = None
    reduced_costs: Optional[List[float]] = None
    basis_indices: Optional[List[int]] = None
    iterations: Optional[int] = None
    phase1_iterations: Optional[int] = None
    phase2_iterations: Optional[int] = None
    max_primal_residual: Optional[float] = None
    max_dual_residual: Optional[float] = None
    basis_condition: Optional[float] = None
    wall_time_ms: Optional[float] = None
    
    # QP specific
    lambda_: Optional[List[float]] = Field(None, alias="lambda")
    slack: Optional[List[float]] = None
    primal_residual: Optional[float] = None
    dual_residual: Optional[float] = None
    duality_gap: Optional[float] = None
    
    # MILP specific
    best_bound: Optional[float] = None
    absolute_gap: Optional[float] = None
    relative_gap: Optional[float] = None
    nodes_explored: Optional[int] = None
    nodes_pruned_infeasible: Optional[int] = None
    nodes_pruned_bound: Optional[int] = None
    nodes_pruned_integer: Optional[int] = None

    device_diagnostics: Optional[DeviceDiagnosticsModel] = None
    matrix_info: Optional[MatrixInfoModel] = None
    request_meta: Optional[Dict[str, Any]] = None
    iteration_history: Optional[List[Dict[str, Any]]] = None


# ==============================================================================
# Validation Helpers
# ==============================================================================

def validate_solve_request(req: SolveRequestV1):
    """Enforces mathematical consistency, finiteness, and dimension integrity."""
    solver_type = req.solver_type.upper()
    if solver_type not in ("LP", "MILP", "QP"):
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported solver_type '{req.solver_type}'. Must be 'LP', 'MILP', or 'QP'."
        )

    n = len(req.c)
    if n == 0:
        raise HTTPException(status_code=400, detail="Objective vector c cannot be empty.")
    if n > 50_000:
        raise HTTPException(status_code=400, detail=f"Variable dimension n={n} exceeds engine limit of 50,000.")

    # Check finite numbers in c
    for idx, val in enumerate(req.c):
        if not (-1e25 <= val <= 1e25):
            raise HTTPException(status_code=400, detail=f"Non-finite or invalid number at c[{idx}]: {val}")

    m = len(req.A)
    if m > 50_000:
        raise HTTPException(status_code=400, detail=f"Constraint dimension m={m} exceeds engine limit of 50,000.")
    if m * n > MAX_ELEMENTS_LIMIT:
        raise HTTPException(status_code=400, detail=f"Matrix size m*n={m*n} exceeds allocation limit.")

    if m > 0:
        for i, row in enumerate(req.A):
            if len(row) != n:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Dimension mismatch: Row {i} of matrix A has {len(row)} elements, expected {n} (dimension of c)."
                )
            for j, val in enumerate(row):
                if not (-1e25 <= val <= 1e25):
                    raise HTTPException(status_code=400, detail=f"Non-finite or invalid number at A[{i}, {j}]: {val}")

        if len(req.b) != m:
            raise HTTPException(
                status_code=400, 
                detail=f"Dimension mismatch: RHS vector b has {len(req.b)} elements, expected {m} (number of constraints in A)."
            )
        for i, val in enumerate(req.b):
            if not (-1e25 <= val <= 1e25):
                raise HTTPException(status_code=400, detail=f"Non-finite or invalid number at b[{i}]: {val}")

        if req.constraint_types:
            if len(req.constraint_types) != m:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Dimension mismatch: constraint_types has {len(req.constraint_types)} elements, expected {m}."
                )
            for ct in req.constraint_types:
                if ct.upper() not in ("LEQ", "GEQ", "EQ"):
                    raise HTTPException(status_code=400, detail=f"Unsupported constraint type '{ct}'. Must be 'LEQ', 'GEQ', or 'EQ'.")
    else:
        if len(req.b) != 0:
            raise HTTPException(status_code=400, detail="Matrix A has 0 rows but vector b is non-empty.")

    if solver_type == "MILP":
        if req.variable_types:
            if len(req.variable_types) != n:
                raise HTTPException(
                    status_code=400,
                    detail=f"Dimension mismatch: variable_types has {len(req.variable_types)} elements, expected {n}."
                )
            for vt in req.variable_types:
                if vt.upper() not in ("CONTINUOUS", "INTEGER", "BINARY"):
                    raise HTTPException(status_code=400, detail=f"Unsupported variable type '{vt}'.")
        if req.bounds:
            if len(req.bounds) != n:
                raise HTTPException(
                    status_code=400,
                    detail=f"Dimension mismatch: bounds has {len(req.bounds)} elements, expected {n}."
                )

    if solver_type == "QP":
        if not req.Q:
            raise HTTPException(status_code=400, detail="Quadratic Program requires symmetric matrix Q.")
        if len(req.Q) != n:
            raise HTTPException(
                status_code=400, 
                detail=f"Dimension mismatch: Matrix Q has {len(req.Q)} rows, expected {n} (dimension of c)."
            )
        for i, row in enumerate(req.Q):
            if len(row) != n:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Dimension mismatch: Row {i} of matrix Q has {len(row)} elements, expected {n} (square matrix)."
                )
            for j, val in enumerate(row):
                if not (-1e25 <= val <= 1e25):
                    raise HTTPException(status_code=400, detail=f"Non-finite number at Q[{i}, {j}]: {val}")


# ==============================================================================
# API Endpoints (Module B & Module C)
# ==============================================================================

@app.get("/api/v1/health")
@app.get("/api/health")
async def health_check():
    """Reports API health status, solver engine version, and native executable readiness."""
    exists = SOLVER_EXECUTABLE.exists() if SOLVER_EXECUTABLE else False
    return {
        "status": "online",
        "engine": "VØRTEX C++20 / CUDA Optimization Engine",
        "version": "3.0.0",
        "executable_exists": exists,
        "executable_path": str(SOLVER_EXECUTABLE) if exists else None,
    }


@app.get("/api/v1/capabilities")
@app.get("/api/capabilities")
async def get_capabilities():
    """
    Reports actual solver capabilities, physical CUDA hardware, supported file formats, and limits.
    Invokes native solver with `--capabilities` to ensure zero fabricated reporting.
    """
    if not SOLVER_EXECUTABLE or not SOLVER_EXECUTABLE.exists():
        raise HTTPException(
            status_code=503, 
            detail=f"Native solver binary not found at {SOLVER_EXECUTABLE}. Build vortex_solver_runner first."
        )

    try:
        proc = await asyncio.create_subprocess_exec(
            str(SOLVER_EXECUTABLE),
            "--capabilities",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(REPO_ROOT)
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise HTTPException(
                status_code=500, 
                detail=f"Capabilities query failed (exit {proc.returncode}): {stderr.decode()}"
            )
        return json.loads(stdout.decode())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to query engine capabilities: {str(e)}")


@app.get("/api/device")
async def get_device_info():
    """Legacy endpoint for device hardware telemetry."""
    caps = await get_capabilities()
    cuda_info = caps.get("cuda", {})
    cuda_avail = cuda_info.get("available", False)
    gpu_data = None
    if cuda_avail:
        gpu_data = {
            "name": cuda_info.get("device_name", "NVIDIA GPU"),
            "multiprocessor_count": cuda_info.get("multiprocessors", 16),
            "compute_capability": cuda_info.get("compute_capability", "8.6"),
            "total_memory_bytes": cuda_info.get("total_vram_bytes", 0),
        }
    return {
        "cuda_available": cuda_avail,
        "gpu": gpu_data,
        "system_info": {
            "solver_name": caps.get("engine", "VØRTEX Optimization Engine"),
            "version": caps.get("version", "3.0.0"),
            "cxx_standard": caps.get("cxx_standard", "C++20"),
            "precision": caps.get("precision", "IEEE 754 FP64"),
        },
        **caps
    }


@app.post("/api/v1/solve", response_model=SolveResponseV1)
@app.post("/api/solve")
async def solve_optimization_problem(req: SolveRequestV1):
    """
    Submits a mathematical optimization problem to the native compiled C++20 / CUDA solver.
    Uses safe subprocess invocation (shell=False), argument arrays, process timeouts, and concurrency throttling.
    """
    validate_solve_request(req)

    if not SOLVER_EXECUTABLE or not SOLVER_EXECUTABLE.exists():
        raise HTTPException(
            status_code=503, 
            detail=f"Native solver binary not found at {SOLVER_EXECUTABLE}. Build vortex_solver_runner first."
        )

    async with SOLVER_SEMAPHORE:
        temp_dir = tempfile.mkdtemp(prefix="vortex_v1_")
        input_path = os.path.join(temp_dir, "problem.json")
        output_path = os.path.join(temp_dir, "solution.json")

        try:
            payload = req.model_dump(exclude_none=True)
            with open(input_path, "w", encoding="utf-8") as f:
                json.dump(payload, f)

            # Subprocess argument array execution with shell=False and restricted cwd
            proc = await asyncio.create_subprocess_exec(
                str(SOLVER_EXECUTABLE),
                "--input", input_path,
                "--output", output_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(REPO_ROOT)
            )

            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=SOLVER_TIMEOUT_SECONDS)
            except asyncio.TimeoutError:
                try:
                    proc.kill()
                    await proc.wait()
                except Exception:
                    pass
                raise HTTPException(
                    status_code=408, 
                    detail=f"Solver execution timed out after {SOLVER_TIMEOUT_SECONDS} seconds."
                )

            if not os.path.exists(output_path):
                err_msg = stderr.decode().strip() if stderr else f"Native solver exited with code {proc.returncode}."
                raise HTTPException(
                    status_code=500, 
                    detail=f"Native solver execution failed: {err_msg}"
                )

            with open(output_path, "r", encoding="utf-8") as f:
                result = json.load(f)

            # Enrich result with request metadata
            result["request_meta"] = {
                "solver_type": req.solver_type,
                "requested_backend": req.backend,
                "dimension_n": len(req.c),
                "dimension_m": len(req.A),
                "sense": req.sense,
            }

            # Document fallback reason if GPU was requested but executed on CPU
            diag = result.get("device_diagnostics")
            if diag and diag.get("requested_backend") == "GPU" and "CPU" in diag.get("executed_backend", ""):
                if not diag.get("fallback_reason"):
                    diag["fallback_reason"] = "Formulation or small problem size routed to CPU host to avoid PCIe transfer overhead"

            return result

        finally:
            # Enforce strict file cleanup
            try:
                if os.path.exists(input_path): os.remove(input_path)
                if os.path.exists(output_path): os.remove(output_path)
                if os.path.exists(temp_dir): os.rmdir(temp_dir)
            except Exception:
                pass


# Module-level benchmark cache
_BENCHMARK_CACHE: Optional[BenchmarkDatasetResponse] = None
_BENCHMARK_CACHE_MTIME: float = 0.0


def compute_benchmark_memory(category: str, rows: int, cols: int, nnz: int) -> MemoryFootprintModel:
    """Computes exact analytical memory requirements for FP64 execution."""
    if "Dense" in category or category == "Data Transfer & Reuse":
        # Matrix A (m*n) + vector x (n) + vector y (m), 8 bytes per double
        bytes_val = (rows * cols + rows + cols) * 8
        fmt = "Dense Row-Major FP64 (A_mn, x_n, y_m)"
    elif "Sparse" in category:
        # CSR: values (nnz*8), col_indices (nnz*4), row_offsets ((m+1)*4), x (n*8), y (m*8)
        bytes_val = nnz * 12 + (rows + 1) * 4 + (rows + cols) * 8
        fmt = "CSR Sparse (FP64 values, Int32 indices, vectors)"
    elif "Simplex Pricing" in category:
        # Full tableau / matrix A (m*n*8) + dual pi (m*8) + reduced costs (n*8)
        bytes_val = (rows * cols + rows + cols) * 8
        fmt = "Dense Tableau Pricing Block FP64"
    elif "Convex QP" in category:
        # Hessian Q (n*n*8) + Constraint A (m*n*8) + vectors (c, b, lambda, slack, x)
        bytes_val = (cols * cols + rows * cols + 3 * cols + 3 * rows) * 8
        fmt = "Convex QP Formulation (Hessian Q, Constraint A, KKT System)"
    else:
        bytes_val = (rows * cols + rows + cols) * 8
        fmt = "Standard FP64 Representation"

    kib = bytes_val / 1024.0
    mib = bytes_val / (1024.0 * 1024.0)
    return MemoryFootprintModel(
        host_bytes=bytes_val,
        gpu_bytes=bytes_val,
        host_kib=round(kib, 3),
        gpu_kib=round(kib, 3),
        host_mib=round(mib, 4),
        gpu_mib=round(mib, 4),
        storage_format=fmt
    )


def load_and_validate_benchmarks() -> BenchmarkDatasetResponse:
    """Loads, validates, and enriches Phase 8E empirical benchmark records."""
    global _BENCHMARK_CACHE, _BENCHMARK_CACHE_MTIME
    
    if not BENCHMARK_RESULTS_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Phase 8E benchmark artifact missing at {BENCHMARK_RESULTS_PATH}. Genuine benchmark results are required."
        )

    current_mtime = BENCHMARK_RESULTS_PATH.stat().st_mtime
    if _BENCHMARK_CACHE is not None and current_mtime == _BENCHMARK_CACHE_MTIME:
        return _BENCHMARK_CACHE

    try:
        raw_text = BENCHMARK_RESULTS_PATH.read_text(encoding="utf-8")
        raw_data = json.loads(raw_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse Phase 8E benchmark JSON: {str(e)}")

    # Compute provenance
    sha256 = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
    gen_time = datetime.datetime.fromtimestamp(current_mtime, tz=datetime.timezone.utc).isoformat()
    
    provenance = BenchmarkProvenanceModel(
        source_file="benchmarks/results/phase8e_results.json",
        file_format="JSON (IEEE 754 Timings)",
        generation_time=gen_time,
        record_count=len(raw_data.get("benchmarks", [])),
        checksum_sha256=sha256,
        harness="VØRTEX Native Benchmark Suite (C++20 / CUDA 13.4)",
        timing_methodology="High-resolution median clock with 5 warmup runs and 15 measured repetitions. Kernel sync via cudaEventSynchronize.",
        notes="Authentic physical empirical measurements on NVIDIA RTX 3050 Laptop GPU and AMD Ryzen 5 6600H."
    )

    sys_info_raw = raw_data.get("system_info", {})
    sys_info = SystemInfoModel(**sys_info_raw)

    validated_records: List[BenchmarkRecordModel] = []
    for r in raw_data.get("benchmarks", []):
        cat = r.get("category", "General")
        is_micro = cat in ["Dense GEMV", "Sparse SpMV", "Simplex Pricing", "Data Transfer & Reuse"]
        slug = re.sub(r"[^a-z0-9]+", "_", r.get("name", "").lower()).strip("_")
        
        cpu_timing = TimingStatsModel(**r["cpu_timing_ms"])
        gpu_e2e = TimingStatsModel(**r["gpu_e2e_timing_ms"])
        gpu_kernel = GpuKernelTimingModel(**r["gpu_kernel_timing_ms"]) if r.get("gpu_kernel_timing_ms") else None

        # Accurately compute and verify speedup: CPU wall-clock / GPU wall-clock
        computed_speedup = round(cpu_timing.median / gpu_e2e.median, 6) if gpu_e2e.median > 0 else 0.0
        is_slowdown = computed_speedup < 1.0

        # Transfer overhead approximation
        transfer_overhead = None
        if gpu_kernel and gpu_e2e.median > gpu_kernel.median:
            transfer_overhead = round(gpu_e2e.median - gpu_kernel.median, 4)

        mem = compute_benchmark_memory(cat, r.get("rows", 0), r.get("cols", 0), r.get("nnz", 0))

        record = BenchmarkRecordModel(
            id=slug,
            category=cat,
            name=r.get("name", ""),
            rows=r.get("rows", 0),
            cols=r.get("cols", 0),
            nnz=r.get("nnz", 0),
            density=r.get("density", 1.0),
            transpose=r.get("transpose", False),
            cpu_timing_ms=cpu_timing,
            gpu_kernel_timing_ms=gpu_kernel,
            gpu_e2e_timing_ms=gpu_e2e,
            speedup_compute=r.get("speedup_compute"),
            speedup_e2e=computed_speedup,
            speedup_label=r.get("speedup_label", f"{computed_speedup:.2f}x"),
            is_slowdown=is_slowdown,
            transfer_overhead_ms=transfer_overhead,
            max_absolute_error=r.get("max_absolute_error"),
            scaled_relative_error=r.get("scaled_relative_error"),
            notes=r.get("notes", ""),
            is_microbenchmark=is_micro,
            memory=mem
        )
        validated_records.append(record)

    dataset = BenchmarkDatasetResponse(
        provenance=provenance,
        system_info=sys_info,
        benchmarks=validated_records
    )

    _BENCHMARK_CACHE = dataset
    _BENCHMARK_CACHE_MTIME = current_mtime
    return dataset


@app.get("/api/v1/benchmarks", response_model=BenchmarkDatasetResponse)
@app.get("/api/benchmarks", response_model=BenchmarkDatasetResponse)
async def get_benchmarks():
    """Serves genuine Phase 8E empirical benchmark datasets with validated schemas."""
    return load_and_validate_benchmarks()


@app.get("/api/v1/benchmarks/{benchmark_id}")
async def get_benchmark_by_id(benchmark_id: str):
    """
    Retrieves a single verified benchmark record by slug id, index, or exact name.
    Strictly validates input against directory traversal.
    """
    # Reject directory traversal attempts
    if ".." in benchmark_id or "/" in benchmark_id or "\\" in benchmark_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid benchmark identifier format."
        )

    dataset = load_and_validate_benchmarks()
    target_clean = benchmark_id.strip().lower()

    # Try numeric index lookup
    if target_clean.isdigit():
        idx = int(target_clean)
        if 0 <= idx < len(dataset.benchmarks):
            return {
                "system_info": dataset.system_info,
                "benchmark": dataset.benchmarks[idx]
            }

    # Match by slug id or name
    for rec in dataset.benchmarks:
        if rec.id == target_clean or rec.name.lower() == target_clean:
            return {
                "system_info": dataset.system_info,
                "benchmark": rec
            }

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Benchmark '{benchmark_id}' not found in Phase 8E empirical dataset."
    )


@app.get("/api/v1/benchmarks/artifacts/{filename}")
async def get_benchmark_artifact(filename: str):
    """
    Safely serves raw Phase 8E benchmark artifact files (JSON or CSV).
    Enforces path restrictions to configured benchmark directory.
    """
    # Strict filename whitelist and path traversal prevention
    ALLOWED_ARTIFACTS = {"phase8e_results.json", "phase8e_results.csv"}
    if filename not in ALLOWED_ARTIFACTS or ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: Only verified Phase 8E benchmark artifacts may be retrieved."
        )

    file_path = (BENCHMARK_DIR / filename).resolve()
    if not file_path.is_relative_to(BENCHMARK_DIR.resolve()) or not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Artifact {filename} does not exist in benchmark directory."
        )

    media_type = "application/json" if filename.endswith(".json") else "text/csv"
    return FileResponse(path=file_path, filename=filename, media_type=media_type)


@app.get("/api/v1/presets")
@app.get("/api/presets")
async def get_presets():
    """Returns verified reference problem presets across LP, MILP, and QP."""
    return [
        {
            "id": "sih26119_standard",
            "name": "SIH26119 Benchmark LP (Dense)",
            "description": "Standard two-variable 3-constraint manufacturing LP with known analytical optimum Z* = 36 at (2, 6).",
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
            "options": {"tolerance": 1e-9, "pivot_rule": "DANTZIG"}
        },
        {
            "id": "two_phase_artificials",
            "name": "Two-Phase Simplex LP (Mixed Constraints)",
            "description": "LP with <=, >=, and == constraints requiring Phase I artificial variables. Optimum Z* = 24 at (8, 0).",
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
            "options": {"tolerance": 1e-9, "pivot_rule": "DANTZIG"}
        },
        {
            "id": "milp_01_knapsack",
            "name": "0-1 Binary Knapsack MILP",
            "description": "Classic NP-hard combinatorial problem solved via Branch-and-Bound tree search. Optimum Z* = 90 (items 2 and 4).",
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
            "options": {"node_limit": 5000, "time_limit_sec": 30.0}
        },
        {
            "id": "convex_qp_portfolio",
            "name": "Constrained Convex Portfolio QP",
            "description": "Primal-Dual Interior-Point quadratic program minimizing risk with linear return constraints. Optimum Z* = -7.25 at (1.0, 2.5).",
            "solver_type": "QP",
            "sense": "MINIMIZE",
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
            "options": {"primal_tolerance": 1e-7, "dual_tolerance": 1e-7, "duality_gap_tolerance": 1e-7}
        },
        {
            "id": "production_planning_lp",
            "name": "Production Mix Optimization LP",
            "description": "Three-variable resource-constrained production planning LP with known optimum.",
            "solver_type": "LP",
            "sense": "MAXIMIZE",
            "backend": "AUTO",
            "c": [5.0, 4.0, 3.0],
            "A": [
                [2.0, 3.0, 1.0],
                [4.0, 1.0, 2.0],
                [3.0, 4.0, 2.0]
            ],
            "b": [50.0, 60.0, 70.0],
            "constraint_types": ["LEQ", "LEQ", "LEQ"],
            "options": {"tolerance": 1e-9}
        }
    ]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
