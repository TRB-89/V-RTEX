# VØRTEX Optimization Engine — Render Free Tier Deployment Guide

This guide details how to build, deploy, and verify the **VØRTEX Native Optimization Engine** on **Render Free Web Services** using the multi-stage Docker configuration.

---

## 1. Architectural Overview & Render Free Constraints

Render's Free Web Service tier provides:
- **Memory**: 512 MB RAM limit.
- **Compute**: Shared vCPU (0.1–0.5 vCPU burstable).
- **Inactivity Behavior**: Containers spin down after 15 minutes of inactivity; incoming requests trigger a cold boot (~30–50 seconds).
- **Hardware Acceleration**: No NVIDIA GPU / CUDA hardware is present on the free tier.

To run reliably within these tight constraints, VØRTEX employs:
1. **Multi-Stage Compilation**: Stage 1 builds the pure C++20 Linux ELF binary (`vortex_solver_runner`) using Debian Bookworm and GCC 12 with `VORTEX_ENABLE_CUDA=OFF`. The binary is stripped of debug symbols, reducing memory overhead and image footprint.
2. **Lean Runtime Container**: Stage 2 uses `python:3.11-slim-bookworm` with only standard runtime libraries (`libstdc++6`, `curl`), running as a dedicated non-root user (`vortex`).
3. **Hardware Safeguards & Memory Throttle**:
   - `MAX_CONCURRENT_SOLVES=1`: Submits solver tasks through an `asyncio.Semaphore(1)` queue to prevent multiple simultaneous solves from breaching 512 MB RAM.
   - `MAX_ELEMENTS_LIMIT=1000000`: Caps problem matrix dimension to 1,000,000 elements (~8 MB raw FP64 data, ~40 MB parsed in Python).
   - `MAX_SOLVER_MEMORY_MB=300`: On Linux, limits the subprocess virtual address space (`RLIMIT_AS`) so a rogue problem fails cleanly without killing the FastAPI server.
   - `SOLVER_TIMEOUT_SECONDS=15`: Aborts solves taking longer than 15 seconds, preventing Render HTTP gateway 502/504 timeouts.

All indigenous CPU LP (dense & sparse revised simplex), MILP (branch-and-bound), and Convex QP (primal-dual interior-point) solvers are **fully preserved**.

---

## 2. Environment Variables Reference

| Variable | Default | Purpose / Description |
| :--- | :--- | :--- |
| `PORT` | `10000` | Injected dynamically by Render. Uvicorn binds to `0.0.0.0:${PORT}`. |
| `HOST` | `0.0.0.0` | IP interface binding for container networking. |
| `VORTEX_SOLVER_BIN` | `/app/build/vortex_solver_runner` | Explicit path to compiled Linux ELF native solver runner. |
| `MAX_CONCURRENT_SOLVES` | `1` | Concurrency limit (semaphore) to prevent OOM in 512 MB RAM. |
| `SOLVER_TIMEOUT_SECONDS` | `15` | Wall-clock execution timeout in seconds per solve request. |
| `MAX_ELEMENTS_LIMIT` | `1000000` | Maximum constraint matrix elements ($m \times n$). |
| `MAX_VARIABLES_LIMIT` | `10000` | Maximum variable dimension ($n$). |
| `MAX_CONSTRAINTS_LIMIT` | `10000` | Maximum constraint dimension ($m$). |
| `MAX_SOLVER_MEMORY_MB` | `300` | Subprocess address space ceiling in megabytes (Linux `RLIMIT_AS`). |
| `ALLOWED_ORIGINS` | `*` | CORS allowed origins (comma-separated URLs or `*`). |

---

## 3. Deployment Options

### Method A: One-Click Render Blueprint (Recommended)

The repository includes a ready-to-use [`render.yaml`](../render.yaml) specification:

1. Push your branch to GitHub.
2. In the [Render Dashboard](https://dashboard.render.com/), click **New +** and select **Blueprint**.
3. Connect your GitHub repository.
4. Render automatically parses `render.yaml`, detects the Docker service `vortex-solver-api`, sets the free tier plan, and provisions the environment.
5. Click **Apply**.

---

### Method B: Manual Web Service Setup

If configuring manually in the Render dashboard:

1. In Render Dashboard, click **New +** > **Web Service**.
2. Connect your Git repository and select branch `phase-9-ui` (or `main`).
3. Set the following settings:
   - **Name**: `vortex-solver-api`
   - **Region**: Choose the region closest to your users (e.g., Oregon, Ohio, Frankfurt).
   - **Language / Runtime**: **Docker**
   - **Dockerfile Path**: `./Dockerfile`
   - **Instance Type**: **Free** (512 MB RAM, 0.5 CPU)
4. Under **Health Check Path**, enter:
   ```
   /healthz
   ```
5. Under **Environment Variables**, verify or add:
   ```env
   PORT=10000
   MAX_CONCURRENT_SOLVES=1
   SOLVER_TIMEOUT_SECONDS=15
   MAX_ELEMENTS_LIMIT=1000000
   MAX_SOLVER_MEMORY_MB=300
   ALLOWED_ORIGINS=*
   ```
6. Click **Create Web Service**.

---

## 4. Verification and Health Checks

Once deployment completes and the status is **Live**, verify your instance using the following endpoints (replace `https://vortex-solver-api.onrender.com` with your actual Render URL).

### 1. Basic Health Check

```bash
curl -f https://vortex-solver-api.onrender.com/healthz
```

**Expected JSON Response:**
```json
{
  "status": "online",
  "engine": "VØRTEX C++20 / CUDA Optimization Engine",
  "version": "3.0.0",
  "executable_exists": true,
  "executable_runnable": true,
  "executable_path": "/app/build/vortex_solver_runner",
  "platform": "linux"
}
```

Notice `executable_path` points to the Linux binary (`vortex_solver_runner`), and `platform` is `"linux"`.

---

### 2. Query Engine Capabilities

Verify that the C++20 engine is reporting accurate non-CUDA capabilities:

```bash
curl -s https://vortex-solver-api.onrender.com/api/v1/capabilities
```

**Expected response indicates:**
- `cxx_standard`: `"C++20"`
- `cuda.available`: `false`
- `solvers.LP.backends`: `["CPU", "GPU", "AUTO"]`
- `solvers.MILP.backends`: `["CPU"]`
- `solvers.QP.backends`: `["CPU", "GPU", "AUTO"]`

---

### 3. Test LP Problem Execution (Standard Simplex)

Solve a 2-variable, 3-constraint LP ($Z^* = 36$ at $(2, 6)$):

```bash
curl -X POST https://vortex-solver-api.onrender.com/api/v1/solve \
  -H "Content-Type: application/json" \
  -d '{
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
    "constraint_types": ["LEQ", "LEQ", "LEQ"]
  }'
```

**Expected Result:**
```json
{
  "success": true,
  "solver_type": "LP",
  "status": "OPTIMAL",
  "objective_value": 36.0,
  "variable_values": [2.0, 6.0],
  "iterations": 2
}
```

---

### 4. Test MILP Problem Execution (Branch-and-Bound Knapsack)

Solve a 0-1 binary knapsack problem ($Z^* = 90$ with variables $x_1 = 1, x_3 = 1$):

```bash
curl -X POST https://vortex-solver-api.onrender.com/api/v1/solve \
  -H "Content-Type: application/json" \
  -d '{
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
    ]
  }'
```

**Expected Result:**
```json
{
  "success": true,
  "solver_type": "MILP",
  "status": "OPTIMAL",
  "objective_value": 90.0,
  "variable_values": [0.0, 1.0, 0.0, 1.0],
  "best_bound": 90.0,
  "absolute_gap": 0.0
}
```

---

### 5. Test QP Problem Execution (Primal-Dual Interior-Point)

Solve a constrained convex quadratic program ($Z^* = -7.25$ at $(1.0, 2.5)$):

```bash
curl -X POST https://vortex-solver-api.onrender.com/api/v1/solve \
  -H "Content-Type: application/json" \
  -d '{
    "solver_type": "QP",
    "sense": "MINIMIZE",
    "backend": "CPU",
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
  }'
```

**Expected Result:**
```json
{
  "success": true,
  "solver_type": "QP",
  "status": "OPTIMAL",
  "objective_value": -7.25,
  "variable_values": [1.0, 2.5]
}
```

---

## 5. Troubleshooting & Cold Start Management

### Cold Starts
- On Render Free, services go to sleep after 15 minutes of inactivity.
- The first request after sleep may take **30–50 seconds** while the container starts. Subsequent requests respond in **5–50 ms**.
- If a persistent warm instance is required, set up a free uptime monitor (e.g., UptimeRobot) pinging `https://<your-service>.onrender.com/healthz` every 10 minutes.

### Out of Memory (OOM) Errors
- If the service receives an unexpected massive matrix exceeding 512 MB, the request will be rejected by `MAX_ELEMENTS_LIMIT` with an HTTP 400 error rather than crashing the container.
- If a problem's internal memory exceeds 300 MB, `MAX_SOLVER_MEMORY_MB` terminates the subprocess safely, returning an HTTP 500 error while keeping the FastAPI server operational.
