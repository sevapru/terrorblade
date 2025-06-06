.PHONY: install install-cuda test lint clean check-env check-duckdb check-uv setup-uv show-info install-duckdb

# Colors
YELLOW := \033[1;33m
GREEN := \033[1;32m
BLUE := \033[1;34m
RED := \033[1;31m
NC := \033[0m

# Check if .env file exists
check-env:
	@if [ ! -f .env ]; then \
		echo -e "$(YELLOW)Warning: .env file not found. You may want to create one based on .env.example$(NC)"; \
	fi

# Python version check
check-python:
	@echo -e "$(BLUE)Checking Python version...$(NC)"
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
	|| (echo -e "$(RED)Error: Python version is not compatible with this project$(NC)" && exit 1)
	@echo "";	

# Show package information
show-info: check-python
	@echo -e "$(GREEN)Terrorblade$(NC) v$$(grep -m1 'version = ' pyproject.toml | cut -d'"' -f2)"
	@echo -e "$(BLUE)Description:$(NC) $$(grep -m1 'description = ' pyproject.toml | cut -d'"' -f2)"
	@echo -e "$(BLUE)Author:$(NC) $$(python3 -c 'print([line.split("name =")[1].split(",")[0].strip().strip("\"") + " <" + line.split("email =")[1].split("}")[0].strip().strip("\"") + ">" for line in open("pyproject.toml") if "name =" in line and "email =" in line][0])')"
	@echo -e "$(BLUE)License:$(NC) $$(head -n1 LICENSE)"
	@echo -e "$(BLUE)Build time:$(NC) $$(date '+%Y-%m-%d %H:%M:%S')"
	@echo "";

# Install DuckDB if not present
install-duckdb:
	@echo -e "$(BLUE)Installing DuckDB...$(NC)"
	@curl -fsSL https://install.duckdb.org | sh
	@echo -e "$(GREEN)DuckDB installation completed!$(NC)"

# Check if duckdb is installed
check-duckdb:
	@which duckdb > /dev/null || { echo -e "$(YELLOW)DuckDB not found. Installing...$(NC)"; make install-duckdb; }

# Check if uv is installed
check-uv:
	@which uv > /dev/null || { echo -e "$(YELLOW)Error: uv is not installed. Installing...$(NC)"; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; }

setup-uv: check-uv
	@echo -e "$(GREEN)Setting up uv...$(NC)"
	@if [ -z "$$VIRTUAL_ENV" ]; then \
		if [ ! -d ".venv" ]; then \
			uv venv; \
			echo -e "$(YELLOW)Virtual environment created at .venv$(NC)"; \
		fi; \
		echo -e "$(BLUE)To activate the virtual environment, run:$(NC)"; \
		echo -e "$(GREEN)source .venv/bin/activate$(NC)"; \
	else \
		echo -e "$(GREEN)Using existing virtual environment at $$VIRTUAL_ENV$(NC)"; \
	fi
	@echo "";

install: setup-uv check-env check-duckdb
	@uv pip install -e ".[dev]"
	@make show-info
	@echo -e "Development environment is ready!$(NC)"
	@echo "";
	@if [ -z "$$VIRTUAL_ENV" ]; then \
		echo -e "Remember to activate your virtual environment:$(NC)"; \
		echo -e "$(GREEN)source .venv/bin/activate$(NC)"; \
	else \
		echo -e "$(GREEN)Using virtual environment at $$VIRTUAL_ENV$(NC)"; \
	fi

install-cuda: setup-uv check-env check-duckdb
	@uv pip install -e ".[dev,cuda]" --extra-index-url=https://pypi.nvidia.com
	@make show-info
	@echo -e "CUDA development environment is ready!$(NC)"
	@if [ -z "$$VIRTUAL_ENV" ]; then \
		echo -e "$(YELLOW)Remember to activate your virtual environment:$(NC)"; \
		echo -e "$(GREEN)source .venv/bin/activate$(NC)"; \
	else \
		echo -e "$(GREEN)Using virtual environment at $$VIRTUAL_ENV$(NC)"; \
	fi

test: check-env
	pytest tests/

lint:
	@echo -e "$(BLUE)Running formatters...$(NC)"
	black .
	isort .
	@echo -e "$(BLUE)Running linters...$(NC)"
	ruff check .
#	@echo -e "$(BLUE)Running type checker...$(NC)"
#	mypy .
	@echo -e "$(GREEN)Linting completed successfully!$(NC)"

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