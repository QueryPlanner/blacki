# syntax=docker/dockerfile:1

# ============================================================================
# MCP Bridge Stage: Install the exact locked Zepto transport dependency
# ============================================================================
FROM node:22-bookworm-slim AS mcp-bridge

WORKDIR /opt/blacki-mcp-bridge

COPY mcp-bridge/package.json mcp-bridge/package-lock.json ./

RUN --mount=type=cache,target=/root/.npm \
    npm ci --omit=dev --ignore-scripts

# ============================================================================
# Builder Stage: Install dependencies with optimal caching
# ============================================================================
FROM python:3.13-slim-bookworm AS builder

# Install uv
RUN pip install uv==0.9.26

# Set working directory
WORKDIR /app

# Environment variables for optimal uv behavior
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never

# Copy dependency files - explicit cache invalidation when either file changes
COPY pyproject.toml uv.lock ./

# Install dependencies (cache mount provides the performance optimization)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-dev

# Copy only source code
COPY src ./src

# Install project (create empty README to satisfy package metadata requirements)
RUN --mount=type=cache,target=/root/.cache/uv \
    touch README.md && \
    uv sync --locked --no-editable --no-dev

# ============================================================================
# Runtime Stage: Minimal production image
# ============================================================================
FROM python:3.13-slim-bookworm AS runtime

# Node is copied from the matching Debian base so the production runtime never
# downloads npm packages. libatomic1 is the only extra shared library required
# by the official Node binary on the slim image.
RUN apt-get update && \
    apt-get install -y --no-install-recommends libatomic1 && \
    rm -rf /var/lib/apt/lists/*

COPY --from=mcp-bridge /usr/local/bin/node /usr/local/bin/node
COPY --from=mcp-bridge /opt/blacki-mcp-bridge /opt/blacki-mcp-bridge
RUN ln -s /opt/blacki-mcp-bridge/node_modules/.bin/mcp-remote \
        /usr/local/bin/mcp-remote && \
    node --version && \
    node -e "const p=require('/opt/blacki-mcp-bridge/node_modules/mcp-remote/package.json'); if(p.version!=='0.1.38') process.exit(1)"

# Create non-root user for security (matching common host UID 1000)
RUN groupadd -g 1000 app && \
    useradd -u 1000 -g app -s /bin/sh -m app

# Set working directory
WORKDIR /app

# Pre-create persistent runtime directories and set ownership
RUN mkdir -p /app/src/.adk/artifacts /app/data /app/logs && \
    chown -R app:app /app

# Copy application and virtual environment from builder
COPY --from=builder --chown=app:app /app .

# Copy entrypoint script and set ownership/permissions
COPY --chown=app:app entrypoint.sh .
RUN chmod +x entrypoint.sh

# Set environment to use virtual environment
ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    AGENT_DIR=/app/src \
    HOST=0.0.0.0 \
    PORT=8080

# Expose port (default 8080)
EXPOSE 8080

# Set the entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]

# Run the FastAPI server
CMD ["python", "-m", "blacki.server"]
