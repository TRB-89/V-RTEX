# VØRTEX Phase 9D: Matrix Inspector & Convergence Visualization Documentation

## 1. Executive Summary & Design System

**Phase 9D** expands VØRTEX from an interactive optimization workspace into a scientific visualization and linear algebra audit laboratory. It provides real-time, interactive graphical inspection of active optimization models and solver diagnostics without fabricating numerical telemetry.

The visual direction strictly adheres to the dark futuristic VØRTEX design language:
- **Obsidian**: `#0C0D13` (Core background & viewport canvas)
- **Graphite**: `#191923` (Panels, architectural grids, borders)
- **Chrome Silver**: `#E7E8EF` (High-contrast typography & active vector states)
- **Muted Violet**: `#77768F` (Secondary indices, inactive cells, dimensions)
- **Electric Blue**: `#2514ED` (Active selections, computation highlights, positive nonzeros)

---

## 2. Visualization Architecture

Phase 9D delivers three interconnected modules across the React/TypeScript frontend:

```
web/src/components/
├── MatrixInspector.tsx         # Module A: Canvas spy plot, dense matrix inspector & CSR/CSC
├── ConvergenceLab.tsx          # Module B: Convergence telemetry, KKT audit & sub-tab container
├── FeasibleRegionPlot2D.tsx    # Module C: 2D linear constraint geometry & objective contours
├── ConvexQPSurface3D.tsx       # Module C: 3D Three.js objective paraboloid surface
└── OptimizationWorkspace.tsx   # Phase 9C: Formulation generator & solver dispatcher
```

### Module Responsibilities:

1. **`MatrixInspector.tsx` (Module A)**:
   - **Canvas-based Matrix Spy Plot**: High-performance HTML5 `<canvas>` rendering capable of handling thousands of elements without DOM thrashing. Nonzero elements are color-coded: electric blue (`#2514ED`) for positive nonzeros, amber (`#F59E0B`) for negative nonzeros.
   - **Interactive Zoom & Pan**: Smooth mouse drag panning, zoom scaling (0.2× to 10×), and reset controls.
   - **Dense Value Inspector**: Full numerical table displaying exact cell values $A_{ij}$ with row and column crosshair highlights.
   - **Canonical CSR / CSC Storage**: Dynamically derives Compressed Sparse Row and Compressed Sparse Column representations (`values`, `col_indices`, `row_offsets` / `row_indices`, `col_offsets`).
   - **Matrix Switching**: Supports switching between Constraint Matrix $A$ ($m \times n$) and Hessian Matrix $Q$ ($n \times n$).
   - **Data Export**: One-click export to PNG spy plot images and CSV matrix files.

2. **`ConvergenceLab.tsx` (Module B)**:
   - **Zero-Fabrication Convergence Telemetry**: Displays genuine iteration history recorded directly by the native solver. When a solver reports terminal basis metrics (such as standard Simplex LP or Branch-and-Bound MILP), it displays an honest scientific disclaimer rather than fabricating intermediate points.
   - **Log-Scale Residual Trajectory**: Tabular and graphical tracking of primal residual $||Ax + s - b||_\infty$, dual residual $||Qx + c + A^T \lambda||_\infty$, and normalized duality gap $\mu = \frac{s^T \lambda}{\max(1, m)}$.
   - **Telemetry Export**: Export convergence trajectory records to CSV and versioned JSON.

3. **`FeasibleRegionPlot2D.tsx` (Module C)**:
   - **2D Feasible Polyhedron**: Active for 2-variable LP models ($n = 2$). Computes half-space pairwise intersections, filters feasible vertices, sorts them polar-angularly around the centroid, and renders a shaded SVG polygon.
   - **Constraint Boundary Lines**: Displays exact boundary lines $A_{i,1} x_1 + A_{i,2} x_2 = b_i$ across the visible domain.
   - **Objective Contours & Gradient**: Displays dashed contour lines $c_1 x_1 + c_2 x_2 = k$ and the gradient direction vector $\nabla f = [c_1, c_2]^T$.
   - **Optimal Vertex Marker**: Real optimal solution $x^* = (x_1^*, x_2^*)$ marked with a glowing target indicator.
   - **High-Dimension Notice**: Clearly explains when geometric visualization is unavailable ($n \ne 2$).

4. **`ConvexQPSurface3D.tsx` (Module C)**:
   - **3D Paraboloid Mesh**: Active for 2-variable QP models ($n = 2$). Implements Three.js parametric mesh rendering of $f(x_1, x_2) = \frac{1}{2}(Q_{11} x_1^2 + 2 Q_{12} x_1 x_2 + Q_{22} x_2^2) + c_1 x_1 + c_2 x_2$.
   - **Interactive Orbit**: Mouse-driven 3D rotation, auto-rotation toggle, and grid floor helper.
   - **Resource Management**: Strictly disposes of Three.js geometries, materials, renderers, animation frames, and window event listeners on unmount.
   - **Reduced Motion Support**: Respects the user's reduced-motion preference by disabling automatic camera rotation.

---

## 3. Mathematical Formulas & Data Transformations

### 1. Matrix Density & Sparsity
For an $m \times n$ matrix $M$:
$$\text{nnz} = \sum_{i=1}^m \sum_{j=1}^n \mathbb{I}(|M_{ij}| > 10^{-12})$$
$$\text{Density} = \frac{\text{nnz}}{m \cdot n}, \quad \text{Sparsity} = 1 - \text{Density}$$

### 2. Compressed Sparse Row (CSR) Representation
- `values`: Array of nonzero elements scanned row by row.
- `col_indices`: Array containing the column index for each entry in `values`.
- `row_pointers` (size $m+1$): `row_pointers[i]` stores the starting index of row $i$ in `values`.

### 3. Compressed Sparse Column (CSC) Representation
- `values`: Array of nonzero elements scanned column by column.
- `row_indices`: Array containing the row index for each entry in `values`.
- `col_pointers` (size $n+1$): `col_pointers[j]` stores the starting index of column $j$ in `values`.

### 4. 2D Half-Space Feasible Region Computation
Given constraints $A_i x \le b_i$ and bounds $x_1 \ge 0, x_2 \ge 0$:
1. Find all pairwise intersection points $(x_1, x_2)$ of lines $L_i: a_{i1} x_1 + a_{i2} x_2 = b_i$ and $L_j: a_{j1} x_1 + a_{j2} x_2 = b_j$:
   $$\det = a_{i1} a_{j2} - a_{i2} a_{j1} \ne 0 \implies x_1 = \frac{b_i a_{j2} - a_{i2} b_j}{\det}, \quad x_2 = \frac{a_{i1} b_j - b_i a_{j1}}{\det}$$
2. Filter points satisfying $A_k x \le b_k + \epsilon$ and $x \ge -\epsilon$ ($\epsilon = 10^{-5}$).
3. Compute centroid $(\bar{x}_1, \bar{x}_2) = \frac{1}{|V|} \sum_{v \in V} v$.
4. Sort vertices by polar angle $\theta_v = \text{atan2}(v_2 - \bar{x}_2, v_1 - \bar{x}_1)$.

### 5. Quadratic Objective Surface
For convex QP with $n = 2$:
$$f(x_1, x_2) = \frac{1}{2} \begin{bmatrix} x_1 & x_2 \end{bmatrix} \begin{bmatrix} Q_{11} & Q_{12} \\ Q_{21} & Q_{22} \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} + \begin{bmatrix} c_1 & c_2 \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}$$
$$= \frac{1}{2} (Q_{11} x_1^2 + 2 Q_{12} x_1 x_2 + Q_{22} x_2^2) + c_1 x_1 + c_2 x_2$$

---

## 4. API & Native Telemetry Dependencies

To support genuine convergence history without altering numerical solver behavior, bounded step telemetry recording was added to the native C++ interior-point solver:

1. **`include/vortex/qp.hpp`**:
   ```cpp
   struct QPIterationRecord {
       std::size_t iteration{0};
       double objective_value{0.0};
       double primal_residual{0.0};
       double dual_residual{0.0};
       double duality_gap{0.0};
   };
   ```
2. **`src/qp.cpp`**:
   Appends genuine step metrics at each interior-point iteration:
   ```cpp
   res.iteration_history.push_back({iter, 0.5 * cur_quad + cur_lin, norm_rp, norm_rd, mu});
   ```
3. **`server/main.py` (`SolveResponseV1`)**:
   Exposes `iteration_history: Optional[List[Dict[str, Any]]] = None` in the REST API response.

---

## 5. Testing & Verification Summary

### Automated Test Suites:
```powershell
# Phase 9D Visualization Test Suite
python tests/test_phase9d_visualization.py

# Complete Integration Test Suites
python tests/test_phase9c_workspace.py
python tests/test_api_v1.py
python tests/test_native_json_interface.py
python tests/test_full_stack_integration.py

# Native CTest Unit Test Suite
ctest --test-dir build-cuda -E GpuBenchmarkCorrectnessTests --output-on-failure
```

### Results:
- **`test_phase9d_visualization.py`**: **8/8 tests passed (100%)**.
  - Density and sparsity calculation on dense and sparse matrices.
  - Canonical CSR and CSC generation and pointer indexing.
  - Rectangular matrices ($m > n$, $m < n$), empty rows, and negative coefficients.
  - 2D feasible polygon calculation for SIH26119 benchmark (vertices: $(0,0), (4,0), (4,3), (2,6), (0,6)$).
  - Paraboloid surface value calculation at $(0,0)$, unconstrained minimizer $(1, 2.5)$, and constrained portfolio optimum $(0.25, 1.75)$.
  - Genuine native QP `iteration_history` step monotonic duality gap decay verification ($1.2 \to 5.62 \times 10^{-12}$).
  - Truthful missing iteration history handling for LP Simplex.
- **Native CTest Suite**: **14/14 unit tests passed (100%)** in 3.66s.
- **Frontend Production Build**: **`npm run build` compiled 2550 modules with zero errors**.

---

## 6. Known Limitations

1. **2D Geometry Limitation ($n = 2$)**:
   - Linear constraint boundary plots and 3D objective surfaces are mathematically defined only for two-variable models. When $n \ne 2$, the UI displays an informative notice explaining why higher-dimensional models cannot be projected without geometric distortion.
2. **Simplex Per-Pivot History**:
   - The compiled native Simplex and Branch-and-Bound engines report terminal basis diagnostics and summary node metrics. The UI honestly labels iteration history as unavailable rather than inventing intermediate points.
