# Testing processes
.PHONY: test test-unit test-integration test-coverage test-watch test-verbose

test: test-unit test-integration
	@echo "All tests completed successfully!"

test-unit:
	@echo "Running unit tests..."
	@test_dirs=""; \
	if [ -d "terrorblade/tests" ]; then \
		test_dirs="$$test_dirs terrorblade/tests"; \
	fi; \
	if [ -d "thoth/tests" ]; then \
		test_dirs="$$test_dirs thoth/tests"; \
	fi; \
	if [ -n "$$test_dirs" ]; then \
		echo "Found test directories:$$test_dirs"; \
		pytest $$test_dirs -k "not integration" -v; \
	else \
		echo "No test directories found. Checked terrorblade/tests and thoth/tests"; \
		echo "Creating terrorblade/tests directory..."; \
		mkdir -p terrorblade/tests; \
		echo "Please add your unit tests to terrorblade/tests/"; \
	fi

test-integration:
	@echo "Running integration tests..."
	@test_dirs=""; \
	if [ -d "terrorblade/tests" ]; then \
		test_dirs="$$test_dirs terrorblade/tests"; \
	fi; \
	if [ -d "thoth/tests" ]; then \
		test_dirs="$$test_dirs thoth/tests"; \
	fi; \
	if [ -n "$$test_dirs" ]; then \
		echo "Found test directories:$$test_dirs"; \
		pytest $$test_dirs -k "integration" -v; \
	else \
		echo "No integration test directories found"; \
	fi

test-coverage:
	@echo "Running tests with coverage..."
	@echo "Installing coverage if not present..."
	@uv pip install coverage pytest-cov 2>/dev/null || true
	@test_dirs=""; \
	if [ -d "terrorblade/tests" ]; then \
		test_dirs="$$test_dirs terrorblade/tests"; \
	fi; \
	if [ -d "thoth/tests" ]; then \
		test_dirs="$$test_dirs thoth/tests"; \
	fi; \
	if [ -n "$$test_dirs" ]; then \
		echo "Running coverage on test directories:$$test_dirs"; \
		pytest $$test_dirs --cov=terrorblade --cov=thoth --cov-report=html --cov-report=term-missing; \
		echo "Coverage report generated in htmlcov/"; \
	else \
		echo "No test directories found"; \
	fi

test-watch:
	@echo "Running tests in watch mode..."
	@test_dirs=""; \
	if [ -d "terrorblade/tests" ]; then \
		test_dirs="$$test_dirs terrorblade/tests"; \
	fi; \
	if [ -d "thoth/tests" ]; then \
		test_dirs="$$test_dirs thoth/tests"; \
	fi; \
	if [ -n "$$test_dirs" ]; then \
		if command -v pytest-watch >/dev/null 2>&1; then \
			ptw $$test_dirs; \
		elif command -v entr >/dev/null 2>&1; then \
			find . -name "*.py" | entr -c pytest $$test_dirs; \
		else \
			echo "Installing pytest-watch..."; \
			uv pip install pytest-watch; \
			ptw $$test_dirs; \
		fi; \
	else \
		echo "No test directories found"; \
	fi

test-verbose:
	@echo "Running tests in verbose mode..."
	@test_dirs=""; \
	if [ -d "terrorblade/tests" ]; then \
		test_dirs="$$test_dirs terrorblade/tests"; \
	fi; \
	if [ -d "thoth/tests" ]; then \
		test_dirs="$$test_dirs thoth/tests"; \
	fi; \
	if [ -n "$$test_dirs" ]; then \
		pytest $$test_dirs -v -s --tb=short; \
	else \
		echo "No test directories found"; \
	fi