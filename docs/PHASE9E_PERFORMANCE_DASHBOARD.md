# VØRTEX — PHASE 9E: GPU PERFORMANCE & BENCHMARK DASHBOARD

## 1. Executive Summary

Phase 9E integrates the empirical, physical benchmark datasets from Phase 8E into a scientific performance lab dashboard within the VØRTEX React/TypeScript user interface and FastAPI backend.

The dashboard accurately distinguishes:
1. **GPU Pure Kernel Compute Time ($T_{\text{kernel}}$)**: Isolated using `cudaEventSynchronize` on the physical NVIDIA GeForce RTX 3050 Laptop GPU.
2. **GPU End-to-End Wall-Clock Time ($T_{\text{gpu\_e2e}}$)**: Inclusive of PCIe Host-to-Device transfer latency, driver context initialization, synchronization, and Device-to-Host readout.
3. **CPU Reference Runtime ($T_{\text{cpu}}$)**: High-resolution median clock measurements on the 12-thread AMD Ryzen 5 6600H host CPU.
4. **End-to-End Speedup**: Formally computed as:
   $$\text{Speedup}_{\text{E2E}} = \frac{T_{\text{CPU, median}}}{T_{\text{GPU E2E, median}}}$$
   Workloads where $\text{Speedup}_{\text{E2E}} < 1.0\times$ are explicitly labeled and color-coded as **GPU Slowdown** rather than silently masked.

---

## 2. Benchmark Source Files and Schemas

### 2.1 Artifact Provenance
- **Source JSON Artifact**: [`benchmarks/results/phase8e_results.json`](file:///c:/Users/TRB/modified%20mindspace%20sdg/CrediAI/vortex/benchmarks/results/phase8e_results.json) (29.9 KB, SHA-256: `18b86c92cc958d9c...`)
- **Source CSV Artifact**: [`benchmarks/results/phase8e_results.csv`](file:///c:/Users/TRB/modified%20mindspace%20sdg/CrediAI/vortex/benchmarks/results/phase8e_results.csv) (6.9 KB)
- **Record Count**: 33 empirical experiment records across 7 mathematical categories.
- **Physical Hardware**:
  - GPU: NVIDIA GeForce RTX 3050 Laptop GPU (16 SMs, Compute Capability 8.6, 4,294,443,008 bytes GDDR6, 128-bit memory bus @ 6001 MHz).
  - CPU: AMD Ryzen 5 6600H with Radeon Graphics (6 Cores / 12 Threads, AVX2).
  - Toolchain: MSVC 19.51, C++20, CUDA Runtime 13040 (13.4), CMake 4.3.

### 2.2 Typed Schema Definitions

#### Pydantic Schema (`server/main.py`)
```python
class TimingStatsModel(BaseModel):
    median: float
    min: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None
    stddev: Optional[float] = None
    iqr: Optional[float] = None

class GpuKernelTimingModel(BaseModel):
    median: float

class MemoryFootprintModel(BaseModel):
    host_bytes: int
    gpu_bytes: int
    host_kib: float
    gpu_kib: float
    host_mib: float
    gpu_mib: float
    storage_format: str

class BenchmarkRecordModel(BaseModel):
    id: str
    category: str
    name: str
    rows: int
    cols: int
    nnz: int
    density: float
    transpose: bool = False
    cpu_timing_ms: TimingStatsModel
    gpu_kernel_timing_ms: Optional[GpuKernelTimingModel] = None
    gpu_e2e_timing_ms: TimingStatsModel
    speedup_compute: Optional[float] = None
    speedup_e2e: float
    speedup_label: str
    is_slowdown: bool
    transfer_overhead_ms: Optional[float] = None
    max_absolute_error: Optional[float] = None
    scaled_relative_error: Optional[float] = None
    notes: Optional[str] = ""
    is_microbenchmark: bool
    memory: MemoryFootprintModel
```

---

## 3. Timing Methodology and Limitations

### 3.1 Kernel Isolation vs End-to-End
1. **Pure Kernel Latency**: Measured using CUDA event recording:
   ```cpp
   cudaEventRecord(start_event, stream);
   kernel<<<grid, block, 0, stream>>>(...);
   cudaEventRecord(stop_event, stream);
   cudaEventSynchronize(stop_event);
   cudaEventElapsedTime(&kernel_ms, start_event, stop_event);
   ```
2. **End-to-End Latency**: Measured using `std::chrono::high_resolution_clock` surrounding device memory allocation (`cudaMalloc`), host-to-device transfer (`cudaMemcpyHostToDevice`), kernel launch, synchronization (`cudaStreamSynchronize`), device-to-host readout (`cudaMemcpyDeviceToHost`), and buffer deallocation.
3. **Transfer Latency Approximation**:
   $$T_{\text{transfer+setup}} \approx T_{\text{gpu\_e2e}} - T_{\text{gpu\_kernel}}$$
   *Note: Because asynchronous stream overlap and driver command batching can occur, separately measured timings are not guaranteed to sum with arithmetic exactness to wall-clock time.*

### 3.2 Cold-Start vs Warmup Runs
- **Warmup Protocol**: 5 warmup iterations are executed before timing collection. This primes PCIe root complexes, loads CUDA driver runtime structures, and clocks the GPU core from idle (210 MHz) to sustained boost (1500 MHz).
- **Repetitions**: 15 timed repetitions are recorded for each benchmark to compute the median, minimum, maximum, mean, sample standard deviation, and interquartile range (IQR).

---

## 4. Supported Performance Comparisons

| Benchmark Category | Cases | Problem Sizes | Peak Observed Speedup | PCIe Slowdown Observed |
| :--- | :---: | :--- | :---: | :---: |
| **Dense GEMV** | 8 | 128x128 to 2048x2048 | **22.67x** (2048x2048 Transposed) | **0.20x** (128x128 Normal) |
| **Sparse SpMV** | 7 | 1000x1000 to 2000x2000 (90%–99% sparse) | **0.23x** (Single SpMV) | **0.04x** (99% Sparse SpMV) |
| **Simplex Pricing** | 3 | 256x1024 to 1024x4096 | **904.88x** (1024x4096) | None (All $> 26\times$) |
| **Full Solver: Dense LP** | 4 | 32x64 to 256x512 | **0.95x** (Near-Parity) | **0.20x** (32x64) |
| **Full Solver: Sparse LP**| 4 | 50x100 to 200x400 (95%–98% sparse) | **0.94x** (Near-Parity) | **0.26x** (50x100) |
| **Full Solver: Convex QP**| 4 | n=32, m=16 to n=256, m=128 | **0.97x** (Near-Parity) | **0.06x** (n=32, m=16) |
| **Data Transfer & Reuse** | 3 | 512x512 to 2048x2048 | **31.19x** (Buffer Reuse) | None (Eliminates PCIe copies) |

### Key Scientific Findings
1. **PCIe Bottleneck on Small Dimension**: On problems smaller than $512 \times 512$, transferring vectors and matrices over the PCIe bus dominates total execution time, making single GPU operations slower than CPU cache-resident operations.
2. **Buffer Reuse Imperative**: Keeping problem data resident in GPU VRAM across iterations yields a **31.2x speedup**, proving that optimization solvers must operate with persistent memory structures.
3. **Simplex Pricing Dominance**: Column pricing in large linear programs achieves extreme scaling (**up to 904.88x**) on the GPU due to embarrassingly parallel inner product reductions across warp lanes.

---

## 5. Security & Ingestion Architecture

### 5.1 Endpoints
- `GET /api/v1/benchmarks`: Returns `BenchmarkDatasetResponse` including provenance checksum, system hardware telemetry, and 33 validated records.
- `GET /api/v1/benchmarks/{id}`: Retrieves single record by slug (`dense_gemv_128x128_normal`) or numeric index (`0..32`). Directory traversal (`..`, `/`, `\`) is rejected.
- `GET /api/v1/benchmarks/artifacts/{filename}`: Safe download endpoint strictly restricted to `phase8e_results.json` and `phase8e_results.csv`. All other filenames return `403 Forbidden`.

---

## 6. Reproducibility Instructions

To reproduce the benchmark suite using the indigenous VØRTEX C++20 / CUDA build tree:

```powershell
# 1. Initialize MSVC 64-bit environment
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

# 2. Configure native CMake build with CUDA acceleration enabled
cmake -B build-cuda -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DVORTEX_ENABLE_CUDA=ON

# 3. Build native benchmark harness and test executables
cmake --build build-cuda --config Release

# 4. Execute Phase 8E empirical benchmark runner
.\build-cuda\vortex_benchmarks.exe --repetitions=15 --warmup=5 --output=benchmarks/results/phase8e_results.json
```

---

## 7. Automated Test Results

### Backend & Ingestion Tests (`pytest tests/test_phase9e_benchmarks.py -v`)
- `test_01_benchmark_dataset_provenance_and_schema`: **PASSED**
- `test_02_speedup_formula_and_slowdown_labeling`: **PASSED**
- `test_03_time_and_memory_unit_consistency`: **PASSED**
- `test_04_single_benchmark_retrieval_and_traversal_prevention`: **PASSED**
- `test_05_artifact_download_and_security`: **PASSED**
- `test_06_numerical_accuracy_and_missing_values`: **PASSED**
- `test_07_comparison_compatibility_filtering`: **PASSED**
- `test_08_memory_calculation_helper`: **PASSED**

**Overall Python Suite**: 50/50 tests passing.
**Native CTest Suite**: 14/14 unit test suites passing (3.69s).
**Frontend TypeScript Build**: Clean production build (`npm run build`), 2550 modules transformed, 0 errors.

---

## 8. Known Limitations

1. **Single-SpMV PCIe Penalty**: Isolated single Sparse Matrix-Vector multiplications exhibit end-to-end GPU slowdowns because the kernel completes in under $30\ \mu\text{s}$, whereas PCIe dispatch and synchronous readback require $\sim 150\ \mu\text{s}$. SpMV acceleration is advantageous only when executed iteratively within persistent GPU memory.
2. **Host RAM Tracking**: The host CPU memory footprint currently reflects exact analytical FP64 structure requirements ($nnz \times 12 + m \times 8$). Dynamic OS virtual memory page allocations are not polled in real-time during benchmark execution.
