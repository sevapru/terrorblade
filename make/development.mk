# Development processes
.PHONY: dev dev-watch format dev-lint check

dev: check-env
	@echo "Starting development environment..."
	@echo "Activating virtual environment and starting Python REPL..."
	@if [ -f ".venv/bin/activate" ]; then \
		echo "source .venv/bin/activate && python3 -c \"import terrorblade; print('Terrorblade development environment ready')\""; \
		bash -c "source .venv/bin/activate && python3"; \
	else \
		echo "Virtual environment not found. Run 'make install' first."; \
		exit 1; \
	fi

dev-watch: check-env
	@echo "Starting development environment with file watching..."
	@if command -v watchmedo >/dev/null 2>&1; then \
		echo "Starting file watcher..."; \
		watchmedo auto-restart --directory=. --pattern="*.py" --recursive -- python3 -m terrorblade; \
	elif command -v entr >/dev/null 2>&1; then \
		echo "Using entr for file watching..."; \
		find . -name "*.py" | entr -r python3 -m terrorblade; \
	else \
		echo "Installing watchdog for file watching..."; \
		uv pip install watchdog; \
		watchmedo auto-restart --directory=. --pattern="*.py" --recursive -- python3 -m terrorblade; \
	fi

format:
	@echo "Formatting code..."
	@echo "Running black formatter..."
	@black . --exclude="thoth|.venv|build|dist" || echo "Black formatting completed with warnings"
	@echo "Running isort import sorter..."
	@isort . --skip-glob="thoth/*" --skip-glob=".venv/*" --skip-glob="build/*" --skip-glob="dist/*" || echo "isort completed with warnings"
	@echo "Code formatting completed!"

dev-lint:
	@echo "Running development linter..."
	@echo "Running ruff for fast linting and auto-fixing..."
	@ruff check . --fix || echo "Ruff linting completed with some issues"
	@echo "Development linting completed!"

check: dev-lint test
	@echo "Running all development checks..."
	@echo "Running comprehensive code quality checks..."
	@echo "1. Checking with ruff..."
	@ruff check . || echo "⚠ Ruff found some issues"
	@echo "2. Verifying import sorting..."
	@isort . --check-only --diff --skip-glob="thoth/*" --skip-glob=".venv/*" || echo "⚠ Import sorting issues found"
	@echo "3. Checking code formatting..."
	@black . --check --exclude="thoth|.venv|build|dist" || echo "⚠ Code formatting issues found"
	@echo "All checks completed!"