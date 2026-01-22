FROM python:3.13-slim

WORKDIR /app

# Install uv
RUN pip install uv

# Copy project files (pyproject.toml + lock for reproducible builds)
COPY pyproject.toml uv.lock ./
COPY terrorblade ./terrorblade

# Install dependencies from lock file with CPU-only PyTorch
# --extra cpu triggers the pytorch-cpu index defined in pyproject.toml
# --extra mcp installs MCP server dependencies
RUN uv sync --frozen --extra cpu --extra mcp && \
    uv pip install mcp-proxy && \
    uv run python -c "import duckdb; duckdb.connect(':memory:').execute('INSTALL vss')"

# Create directories for logs and data
RUN mkdir -p /app/logs /data

# Copy entrypoint script
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Default environment variables
ENV DB_PATH=/data/telegram_data.db
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["/entrypoint.sh"]
CMD ["mcp"]
