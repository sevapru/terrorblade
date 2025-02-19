.PHONY: install install-dev install-cuda dev test lint clean check-env check-duckdb check-uv setup-uv

# Check if .env file exists
check-env:
	@if [ ! -f .env ]; then \
		echo "Error: .env file not found. Please create one based on .env.example"; \
		exit 1; \
	fi

# Check if duckdb is installed
check-duckdb:
	@which duckdb > /dev/null || { echo "Error: duckdb is not installed. Please install it first."; exit 1; }

# Check if uv is installed
check-uv:
	@which uv > /dev/null || { echo "Error: uv is not installed. Installing..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; }

setup-uv: check-uv
	@echo "Setting up uv..."
	@uv venv

install: setup-uv check-env check-duckdb
	uv pip install -e .

install-dev: setup-uv check-env check-duckdb
	uv pip install -e ".[dev]"

install-cuda: setup-uv check-env check-duckdb
	uv pip install -e ".[dev,cuda]" --extra-index-url=https://pypi.nvidia.com

dev: install-dev
	@echo "Development environment is ready!"

test: check-env
	pytest tests/

lint:
	black .
	isort .
	mypy .

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf .venv/ 