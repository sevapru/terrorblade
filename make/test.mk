# Testing and code quality
include make/common.mk

.PHONY: test check clean format

# Run all tests
test: check-python
	$(call log_section,🧪 Running Tests)
	@test_dirs=""; \
	if [ -d "tests" ]; then \
		test_dirs="$$test_dirs tests"; \
	fi; \
	if [ -d "terrorblade/tests" ]; then \
		test_dirs="$$test_dirs terrorblade/tests"; \
	fi; \
	if [ -d "thoth" ] && [ -d "thoth/tests" ]; then \
		test_dirs="$$test_dirs thoth/tests"; \
	fi; \
	if [ -n "$$test_dirs" ]; then \
		echo -e "$(BLUE)[INFO]$(NC) Running pytest on:$$test_dirs"; \
		if [ -n "$$VIRTUAL_ENV" ]; then \
			python -m pytest $$test_dirs -v --tb=short || { echo -e "$(RED)[✗]$(NC) Tests failed"; exit 1; }; \
		elif [ -d ".venv" ]; then \
			.venv/bin/python -m pytest $$test_dirs -v --tb=short || { echo -e "$(RED)[✗]$(NC) Tests failed"; exit 1; }; \
		else \
			python -m pytest $$test_dirs -v --tb=short || { echo -e "$(RED)[✗]$(NC) Tests failed"; exit 1; }; \
		fi; \
		echo -e "$(GREEN)[✓]$(NC) All tests passed!"; \
	else \
		echo -e "$(YELLOW)[⚠]$(NC) No test directories found. Skipping tests."; \
	fi

# Run linting and formatting checks
check: check-python
	$(call log_section,✨ Code Quality Check)
	
	@# Determine Python command to use
	@PYTHON_CMD="python"; \
	if [ -z "$$VIRTUAL_ENV" ] && [ -d ".venv" ]; then \
		PYTHON_CMD=".venv/bin/python"; \
	fi; \
	\
	echo -e "$(BLUE)[INFO]$(NC) Checking code formatting with black..."; \
	$$PYTHON_CMD -m black --check --diff . || { echo -e "$(RED)[✗]$(NC) Code formatting issues found. Run: make format"; exit 1; }; \
	\
	echo -e "$(BLUE)[INFO]$(NC) Checking import sorting with isort..."; \
	$$PYTHON_CMD -m isort --check-only --diff . || { echo -e "$(RED)[✗]$(NC) Import sorting issues found. Run: make format"; exit 1; }; \
	\
	echo -e "$(BLUE)[INFO]$(NC) Running ruff linter..."; \
	$$PYTHON_CMD -m ruff check . || { echo -e "$(RED)[✗]$(NC) Linting issues found. Run: $$PYTHON_CMD -m ruff check --fix ."; exit 1; }; \
	\
	echo -e "$(BLUE)[INFO]$(NC) Running mypy type checker..."; \
	if [ -d "thoth" ]; then \
		$$PYTHON_CMD -m mypy terrorblade/ thoth/ || { echo -e "$(YELLOW)[⚠]$(NC) Type checking issues found"; true; }; \
	else \
		$$PYTHON_CMD -m mypy terrorblade/ || { echo -e "$(YELLOW)[⚠]$(NC) Type checking issues found"; true; }; \
	fi; \
	\
	echo -e "$(GREEN)[✓]$(NC) All code quality checks passed!"

# Format code (fix issues found by check)
format: check-python
	$(call log_section,🎨 Formatting Code)
	@# Determine Python command to use
	@PYTHON_CMD="python"; \
	if [ -z "$$VIRTUAL_ENV" ] && [ -d ".venv" ]; then \
		PYTHON_CMD=".venv/bin/python"; \
	fi; \
	\
	echo -e "$(BLUE)[INFO]$(NC) Formatting code with black..."; \
	$$PYTHON_CMD -m black .; \
	echo -e "$(BLUE)[INFO]$(NC) Sorting imports with isort..."; \
	$$PYTHON_CMD -m isort .; \
	echo -e "$(BLUE)[INFO]$(NC) Auto-fixing ruff issues..."; \
	$$PYTHON_CMD -m ruff check --fix .; \
	echo -e "$(GREEN)[✓]$(NC) Code formatted successfully!"

# Clean test artifacts
clean:
	$(call log_info,Cleaning test artifacts...)
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@find . -name "*.pyc" -delete 2>/dev/null || true
	@find . -name ".coverage" -delete 2>/dev/null || true
	@rm -rf htmlcov/ 2>/dev/null || true
	@rm -rf build/ dist/ 2>/dev/null || true
	$(call log_success,Cleaned test artifacts)