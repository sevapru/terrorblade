# Requirements management using pyproject.toml as source of truth
include make/common.mk

.PHONY: requirements requirements-install requirements-sync requirements-update

# Main requirements target - install dependencies from pyproject.toml
requirements: requirements-install
	$(call log_success,Requirements ready from pyproject.toml)

# Install dependencies from pyproject.toml
requirements-install: check-uv
	$(call log_info,Installing dependencies from pyproject.toml...)
	@uv pip install -e ".[dev,security]"
	$(call log_success,Dependencies installed from pyproject.toml)

# Install with thoth dependencies
requirements-thoth: check-uv
	$(call log_info,Installing with thoth dependencies...)
	@uv pip install -e ".[dev,security,thoth]"
	$(call log_success,Dependencies with thoth installed from pyproject.toml)

# Sync environment with pyproject.toml
requirements-sync: check-uv
	$(call log_info,Syncing environment with pyproject.toml...)
	@uv pip install -e ".[dev,security]"
	$(call log_success,Environment synced with pyproject.toml)

# Legacy: Update requirements files (for backwards compatibility)
requirements-update: check-uv
	$(call log_warning,Legacy requirements files are deprecated. Dependencies are now managed in pyproject.toml)
	$(call log_info,To update dependencies, edit pyproject.toml directly)

# Show requirements status
requirements-status:
	$(call log_section,Requirements Status)
	@echo -e "$(BLUE)Dependencies source:$(NC)"
	@if [ -f pyproject.toml ]; then \
		echo "  [OK] pyproject.toml (source of truth)"; \
		echo "  Dependencies sections:"; \
		echo "    - dependencies (core)"; \
		echo "    - optional-dependencies.dev (development)"; \
		echo "    - optional-dependencies.thoth (ML/analysis)"; \
		echo "    - optional-dependencies.security (security tools)"; \
		echo "    - optional-dependencies.viz (visualization)"; \
	else \
		echo "  [ERROR] pyproject.toml missing"; \
	fi
	@echo -e "$(BLUE)Legacy files:$(NC)"
	@if [ -f scripts/requirements.in ]; then \
		echo "  [DEPRECATED] scripts/requirements.in (use pyproject.toml instead)"; \
	fi
	@if [ -f scripts/requirements-dev.in ]; then \
		echo "  [DEPRECATED] scripts/requirements-dev.in (use pyproject.toml instead)"; \
	fi 