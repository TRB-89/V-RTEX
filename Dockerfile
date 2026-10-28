# ==============================================================================
# VØRTEX Optimization Engine - Multi-Stage Production Dockerfile for Render Free Tier
# Phase 9: Headless Native C++20 Linux Solver + FastAPI / Uvicorn Service
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Native C++20 Compilation Builder (CUDA Disabled, Linux ELF Binary)
# ------------------------------------------------------------------------------
FROM debian:bookworm-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive

# Install modern C++20 build toolchain (GCC 12.2, CMake, make)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy C++ engine sources and CMake configuration
COPY CMakeLists.txt ./
COPY include/ ./include/
COPY src/ ./src/

# Compile the native C++20 solver runner with optimizations and CUDA disabled
RUN cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_STANDARD=20 \
    -DVORTEX_ENABLE_CUDA=OFF \
    && cmake --build build --target vortex_solver_runner -j$(nproc) \
    && strip build/vortex_solver_runner \
    && chmod +x build/vortex_solver_runner

# ------------------------------------------------------------------------------
# Stage 2: Production Python Runtime Environment (Target: Render Free 512MB RAM)
# ------------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS runner

# Optimize Python execution for containerized environments
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=10000 \
    HOST=0.0.0.0 \
    VORTEX_SOLVER_BIN=/app/build/vortex_solver_runner \
    MAX_CONCURRENT_SOLVES=1 \
    SOLVER_TIMEOUT_SECONDS=15 \
    MAX_ELEMENTS_LIMIT=1000000 \
    MAX_SOLVER_MEMORY_MB=300 \
    ALLOWED_ORIGINS="*"

# Install minimal runtime dependencies: libstdc++6 and curl for health check
RUN apt-get update && apt-get install -y --no-install-recommends \
    libstdc++6 \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create dedicated non-root application user for container security
RUN groupadd -r vortex && useradd -r -g vortex -d /app -s /sbin/nologin vortex

WORKDIR /app

# Install Python production dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy compiled native Linux C++20 binary from builder stage
COPY --from=builder /workspace/build/vortex_solver_runner /app/build/vortex_solver_runner
RUN chmod +x /app/build/vortex_solver_runner

# Copy FastAPI backend application and static benchmark datasets
COPY server/ ./server/
COPY benchmarks/results/ ./benchmarks/results/

# Grant ownership to non-root user
RUN chown -R vortex:vortex /app

USER vortex

# Expose Render PORT (Render injects $PORT at runtime, default 10000)
EXPOSE 10000

# Container Healthcheck verifying FastAPI and native solver readiness
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://127.0.0.1:${PORT}/healthz || exit 1

# Launch FastAPI application through uvicorn bound to 0.0.0.0 and PORT
CMD ["sh", "-c", "exec uvicorn server.main:app --host 0.0.0.0 --port ${PORT}"]
