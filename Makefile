# Include modular makefiles
include make/build.mk
include make/deploy.mk
include make/development.mk
include make/modules.mk
include make/test.mk

.PHONY: help install install-all install-subprojects install-terrorblade-only install-thoth-only test lint clean check-env check-duckdb check-uv setup-uv show-info install-duckdb

# Default target
.DEFAULT_GOAL := help

# Help target
help:
	@echo -e "$(GREEN)Terrorblade Makefile Help$(NC)"
	@echo -e "$(BLUE)========================$(NC)"
	@echo ""
	@echo -e "$(YELLOW)Main targets:$(NC)"
	@echo "  help              Show this help message"
	@echo "  install           Install development environment (main project)"
	@echo "  install-all       Install all subprojects (terrorblade + thoth)"
	@echo "  install-subprojects Install only subprojects"  
	@echo "  install-terrorblade-only Install only terrorblade"
	@echo "  install-thoth-only Install only thoth"
	@echo "  show-info         Display project information"
	@echo ""
	@echo -e "$(YELLOW)Development:$(NC)"
	@echo "  dev               Start development environment"
	@echo "  dev-watch         Start development with file watching"
	@echo "  format            Format code"
	@echo "  lint              Run linters and formatters"
	@echo "  check             Run all checks (lint + test)"
	@echo ""
	@echo -e "$(YELLOW)Testing:$(NC)"
	@echo "  test              Run all tests"
	@echo "  test-unit         Run unit tests"
	@echo "  test-integration  Run integration tests" 
	@echo "  test-coverage     Run tests with coverage"
	@echo ""
	@echo -e "$(YELLOW)Build & Deploy:$(NC)"
	@echo "  build             Build the project"
	@echo "  clean             Clean build artifacts"
	@echo "  package           Create deployment package"
	@echo "  deploy            Deploy to staging"
	@echo "  deploy-staging    Deploy to staging environment"
	@echo "  deploy-prod       Deploy to production environment"
	@echo ""
	@echo -e "$(YELLOW)Modules:$(NC)"
	@echo "  list-modules      List available modules"
	@echo "  install-modules   Install all modules"
	@echo "  install-terrorblade Install Terrorblade module"
	@echo "  install-thoth     Install Thoth module"


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
			echo -e "$(BLUE)Creating virtual environment...$(NC)"; \
			uv venv --python 3.12; \
			echo -e "$(YELLOW)Virtual environment created at .venv$(NC)"; \
		else \
			echo -e "$(GREEN)Virtual environment already exists at .venv$(NC)"; \
		fi; \
		echo -e "$(BLUE)To activate the virtual environment, run:$(NC)"; \
		echo -e "$(GREEN)source .venv/bin/activate$(NC)"; \
	else \
		echo -e "$(GREEN)Using existing virtual environment at $$VIRTUAL_ENV$(NC)"; \
	fi
	@echo "";

install: setup-uv check-env check-duckdb
	@echo -e "$(BLUE)Installing Terrorblade with development dependencies...$(NC)"
	@uv pip install -e ".[dev]"
	@echo -e "$(BLUE)Verifying installation...$(NC)"
	@python3 -c "import black, isort, ruff; print('✓ Code quality tools installed')" 2>/dev/null || echo "⚠ Some tools may not be available"
	@make show-info
	@echo -e "$(GREEN)Development environment is ready!$(NC)"
	@echo "";
	@if [ -z "$$VIRTUAL_ENV" ]; then \
		echo -e "$(YELLOW)Remember to activate your virtual environment:$(NC)"; \
		echo -e "$(GREEN)source .venv/bin/activate$(NC)"; \
		echo -e "$(BLUE)Then test with: make lint$(NC)"; \
	else \
		echo -e "$(GREEN)Using virtual environment at $$VIRTUAL_ENV$(NC)"; \
		echo -e "$(BLUE)Test your setup with: make lint$(NC)"; \
	fi

# Subproject installation system
install-subprojects: setup-uv
	@echo -e "$(BLUE)Installing all subprojects...$(NC)"
	@for subproject in terrorblade thoth; do \
		if [ -d "$$subproject" ]; then \
			echo -e "$(YELLOW)Installing $$subproject...$(NC)"; \
			if [ -f "$$subproject/pyproject.toml" ]; then \
				uv pip install -e "$$subproject[dev]"; \
			else \
				echo -e "$(RED)No pyproject.toml found in $$subproject$(NC)"; \
			fi; \
		else \
			echo -e "$(YELLOW)Subproject $$subproject not found, skipping...$(NC)"; \
		fi; \
	done
	@echo -e "$(GREEN)All available subprojects installed!$(NC)"

# Enhanced installation options
install-terrorblade-only: setup-uv check-env
	@echo -e "$(BLUE)Installing Terrorblade only...$(NC)"
	@uv pip install -e ".[dev]"
	@echo -e "$(GREEN)Terrorblade installed!$(NC)"

install-thoth-only: setup-uv check-env
	@echo -e "$(BLUE)Installing Thoth only...$(NC)"
	@if [ -d "thoth" ]; then \
		uv pip install -e "thoth[dev]"; \
		echo -e "$(GREEN)Thoth installed!$(NC)"; \
	else \
		echo -e "$(RED)Thoth directory not found$(NC)"; \
		exit 1; \
	fi

install-all: install install-subprojects
	@echo -e "$(GREEN)Complete installation finished!$(NC)"

# Override specific targets from included files with project-specific implementations

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