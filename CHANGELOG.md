# Changelog — VØRTEX Optimization Engine

All notable changes to the VØRTEX project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0-research-prototype] — 2026-10-27

### Added
- **Phase 1 (Linear Algebra Core)**:
  - Cache-friendly 1D contiguous FP64 `Matrix` class with C++20 concept constraints.
  - In-place LU Decomposition with partial pivoting and forward/backward substitution.
  - Cholesky decomposition for positive-definite systems.
- **Phase 2 (Standard Simplex)**:
  - Standard form Primal Simplex LP solver for $\le$ constraints.
  - Dantzig and Bland pivot selection rules for cycling prevention.
- **Phase 3 (Two-Phase Simplex)**:
  - Phase I artificial variable drive-out mechanics for $\le, \ge, =$ mixed constraints.
  - Automated RHS normalization and infeasibility/unboundedness detection.
- **Phase 4 (Revised Simplex)**:
  - Basis matrix operations using $LU$ factorization updates.
  - Numerical stability monitoring via basis condition estimation.
- **Phase 5 (Sparse Matrix Engine)**:
  - Compressed Sparse Row (CSR) and Compressed Sparse Column (CSC) matrix structures.
  - Sparse-aware Revised Simplex pricing with $O(nnz)$ column selection.
- **Phase 6 (Mixed-Integer Linear Programming)**:
  - Branch-and-Bound tree search supporting Continuous, Integer, and Binary variables.
  - Most-fractional variable selection and best-bound node exploration.
- **Phase 7 (Convex Quadratic Programming)**:
  - Mehrotra Predictor-Corrector Primal-Dual Interior-Point Method (IPM).
  - Jacobi eigenvalue rotation for positive semi-definiteness verification ($Q \succeq 0$).
  - Dynamic KKT regularization ($10^{-12} \le \delta \le 10^{-6}$) and unconstrained stationarity fast-path.
- **Phase 8 (Native CUDA Acceleration Engine)**:
  - Native double-precision cuBLAS GEMV and cuSPARSE CSR SpMV pipelines.
  - Custom warp-level parallel reduction kernels for high-speed Simplex pricing.
  - Hybrid CPU/GPU scheduler with dynamic small-dimension fallback.
- **Phase 9A (Cinematic Web Interface)**:
  - Three.js metallic vortex hero sculpture with dynamic lighting and GSAP scroll scrubbing.
  - Dark futurism editorial identity (Obsidian `#0C0D13`, Graphite `#191923`, Chrome `#E7E8EF`, Electric Blue `#2514ED`).
- **Phase 9B (Production REST API Adapter)**:
  - Native machine-readable JSON CLI executable `vortex_solver_runner.exe`.
  - Secure FastAPI adapter with concurrency throttling, timeouts, and subprocess argument isolation.
- **Phase 9C (Optimization Workspace)**:
  - Dynamic variable, constraint, and Hessian $Q$ matrix editors.
  - Real-time mathematical formulation previews and reference problem presets.
- **Phase 9D (Visual Mathematical Diagnostics)**:
  - Interactive sparse matrix sparsity heatmaps with live CSR/CSC format inspectors.
  - 2D convex feasible region polytope geometry visualizer with active constraint boundary highlights.
  - 3D Three.js quadratic objective paraboloid surface viewer with real IPM iteration descent trajectories.
- **Phase 9E (GPU Performance & Benchmark Dashboard)**:
  - Empirical Phase 8E benchmark dataset ingestion (33 physical hardware records on RTX 3050 Laptop GPU).
  - CPU vs GPU comparison studio with kernel-only vs end-to-end breakdown and truthful slowdown labeling.
  - GPU stage breakdown timeline, memory footprint analysis, and side-by-side run comparison.
- **Phase 9F (Final Verification & Reports)**:
  - Downloadable executive PDF solution report generator using `jsPDF`.
  - Comprehensive 64-test automated Python suite and 14 native CTest suites.

### Security
- Applied strict subprocess execution via argument arrays (`shell=False`).
- Implemented file path sanitization and directory traversal prevention on all benchmark endpoints.
- Enforced 30-second execution timeouts and maximum matrix allocation limit of 25,000,000 elements.
- Restricted CORS headers to configured origins.

### Fixed
- Fixed quadratic objective value analytical minimum in KKT test suite to $-6.125$.
- Fixed missing `iteration_history` serialization in Pydantic `SolveResponseV1` model.
- Fixed TypeScript `verbatimModuleSyntax` import issues across all React workspace components.
