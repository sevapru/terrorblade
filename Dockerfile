# Base image for both development and production
FROM ubuntu:22.04 AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:${PATH}"

# Set the working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml ./

# Install Python dependencies using uv
RUN uv pip install --system --no-cache --requirement pyproject.toml --extra dev

# --- Development Stage ---
FROM base AS dev

# Development dependencies are already installed in the base stage with --extra dev
# Add tools for debugging with Cursor (assuming ptvsd or debugpy)
# You might need to adjust this depending on the specific tools Cursor uses
RUN uv pip install --system --no-cache debugpy

# Copy the rest of the application code
COPY . .

# Default command for development (e.g., run with debugger)
# This will be overridden by docker-compose or specific run commands
CMD ["python3", "-m", "debugpy", "--listen", "0.0.0.0:5678", "-m", "terrorblade.main"]

# --- Production Stage for Terrorblade ---
FROM base AS terrorblade_prod

# Remove development dependencies for production
# Create a new virtual environment or reinstall without dev dependencies
# For simplicity, we'll assume the base image is lean enough or rebuild.
# A more optimized approach would be to have a separate production dependency install in base.
# Re-copying files and reinstalling without dev dependencies:
COPY pyproject.toml ./
RUN uv pip install --system --no-cache --requirement pyproject.toml
COPY . .

# Command to run Terrorblade
CMD ["python3", "-m", "terrorblade.main"]

# --- Production Stage for Thoth ---
FROM base AS thoth_prod

# Remove development dependencies for production
COPY pyproject.toml ./
RUN uv pip install --system --no-cache --requirement pyproject.toml
COPY . .

# Placeholder for Thoth - assuming it's a web server or a script
# You'll need to replace this with the actual command to run Thoth
# For example, if Thoth is a Flask app in thoth_app.py:
# CMD ["gunicorn", "--bind", "0.0.0.0:8000", "thoth_app:app"]
# For now, a simple placeholder:
CMD ["echo", "Thoth container started. Implement Thoth entrypoint."]

# --- Loki Logging (Example) ---
# This section is illustrative. Actual integration depends on your setup.
# You might use a sidecar container or configure your app to send logs to Loki.

# Example: Install promtail or another Loki log shipper (if needed in the container)
# RUN apt-get update && apt-get install -y promtail && rm -rf /var/lib/apt/lists/*

# Configure your application's logging to output in a format Loki can parse (e.g., JSON)
# See terrorblade/utils/logger.py for potential integration points. 