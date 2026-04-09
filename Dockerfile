# Multi-stage Dockerfile for Auto-ART
# Stage 1: Builder (compile dependencies)
# Stage 2: Runtime (minimal image, non-root user)
#
# Features:
# - Multi-stage build for smaller runtime image
# - Non-root user (UID 1001) for security
# - Health check endpoint
# - GPU support via nvidia/cuda base (use Dockerfile.gpu)
# - Read-only filesystem compatible

# ============================================================
# Stage 1: Builder
# ============================================================
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /build

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Install UV for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency files first (Docker layer caching)
COPY pyproject.toml uv.lock* ./

# Install dependencies into a virtual environment
RUN uv sync --frozen --no-dev --no-install-project

# Copy source code and README (required by hatchling build)
COPY README.md ./
COPY auto_art ./auto_art

# Install the project itself (non-editable so .pth points to site-packages)
RUN uv sync --frozen --no-dev --no-editable && \
    uv pip install "gunicorn>=22.0.0"

# ============================================================
# Stage 2: Runtime
# ============================================================
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Security: non-root user
RUN groupadd --gid 1001 autoart && \
    useradd --uid 1001 --gid autoart --shell /bin/bash --create-home autoart

WORKDIR /app

# Copy only the virtual environment and source from builder
COPY --from=builder /build/.venv /app/.venv
COPY --from=builder /build/auto_art /app/auto_art
COPY --from=builder /build/pyproject.toml /app/pyproject.toml

# Fix venv shebangs (builder path → runtime path)
RUN find /app/.venv/bin -type f -exec \
    sed -i 's|#!/build/.venv/bin/python|#!/app/.venv/bin/python|g' {} + 2>/dev/null || true

# Set up path to use venv
ENV PATH="/app/.venv/bin:$PATH"
ENV VIRTUAL_ENV="/app/.venv"

# Create directories for runtime data
RUN mkdir -p /app/data /app/audit /tmp/auto_art && \
    chown -R autoart:autoart /app /tmp/auto_art

# Flask configuration
ENV FLASK_APP=auto_art.api.app:app
ENV AUTO_ART_AUDIT_DIR=/app/audit
ENV AUTO_ART_AUTH_MODE=required

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')" || exit 1

# Expose API port
EXPOSE 5000

# Switch to non-root user
USER autoart

# Run with gunicorn for production
CMD ["gunicorn", \
     "--bind", "0.0.0.0:5000", \
     "--workers", "4", \
     "--timeout", "120", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "auto_art.api.app:app"]
