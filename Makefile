.PHONY: install install-dev install-cuda install-full dev test lint clean check-env check-duckdb check-uv setup-uv show-info

# Colors
YELLOW := \033[1;33m
GREEN := \033[1;32m
BLUE := \033[1;34m
RED := \033[1;31m
NC := \033[0m

# Check if .env file exists
check-env:
	@if [ ! -f .env ]; then \
		echo "$(YELLOW)Warning: .env file not found. You may want to create one based on .env.example$(NC)"; \
	fi

# Python version check
check-python:
	@echo "$(BLUE)Checking Python version...$(NC)"
	@python3 -c '\
	import sys, re; \
	required = re.search(r"requires-python = \"(.+?)\"", open("pyproject.toml").read()).group(1); \
	current = f"{sys.version_info.major}.{sys.version_info.minor}"; \
	from packaging import specifiers; \
	spec = specifiers.SpecifierSet(required); \
	is_compatible = spec.contains(current); \
	print(f"Required: {required}"); \
	print(f"Current:  {current}"); \
	print(f"Status:   {"Compatible" if is_compatible else "Incompatible"}"); \
	exit(0 if is_compatible else 1);' \
	|| (echo "$(RED)Error: Python version is not compatible with this project$(NC)" && exit 1)

# Show package information
show-info: check-python
	@echo "$(GREEN)Terrorblade$(NC) v$$(grep -m1 'version = ' pyproject.toml | cut -d'"' -f2)"
	@echo "$(BLUE)Description:$(NC) $$(grep -m1 'description = ' pyproject.toml | cut -d'"' -f2)"
	@echo "$(BLUE)Author:$(NC) $$(python3 -c 'print([line.split("name =")[1].split(",")[0].strip().strip("\"") + " <" + line.split("email =")[1].split("}")[0].strip().strip("\"") + ">" for line in open("pyproject.toml") if "name =" in line and "email =" in line][0])')"
	@echo "$(BLUE)License:$(NC) $$(head -n1 LICENSE)"
	@echo "$(BLUE)Build time:$(NC) $$(date '+%Y-%m-%d %H:%M:%S')"

# Check if duckdb is installed
check-duckdb:
	@which duckdb > /dev/null || { echo "$(YELLOW)Error: duckdb is not installed. Please install it first.$(NC)"; exit 1; }

# Check if uv is installed
check-uv:
	@which uv > /dev/null || { echo "$(YELLOW)Error: uv is not installed. Installing...$(NC)"; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; }

setup-uv: check-uv
	@echo "$(GREEN)Setting up uv...$(NC)"
	@if [ -z "$$VIRTUAL_ENV" ]; then \
		uv venv; \
	else \
		echo "$(GREEN)Using existing virtual environment at $$VIRTUAL_ENV$(NC)"; \
	fi

install: setup-uv check-env check-duckdb
	@uv pip install -e .
	@make show-info

install-dev: setup-uv check-env check-duckdb
	@uv pip install -e ".[dev]"
	@make show-info

install-cuda: setup-uv check-env check-duckdb
	@uv pip install -e ".[dev,cuda]" --extra-index-url=https://pypi.nvidia.com
	@make show-info

install-full: setup-uv check-env check-duckdb
	@uv pip install -e ".[dev,cuda]" --extra-index-url=https://pypi.nvidia.com
	@make show-info
	@echo "$(GREEN)Full installation completed with all dependencies!$(NC)"

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