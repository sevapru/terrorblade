# Requirements management using uv.lock as source of truth
include make/common.mk

.PHONY: requirements requirements-install requirements-sync requirements-update requirements-lock

# Main requirements target - install dependencies from uv.lock
requirements: requirements-install
	$(call log_success,Requirements ready from uv.lock)

# Install dependencies from uv.lock
requirements-install: check-uv
	$(call log_info,Installing dependencies from uv.lock...)
	@uv sync --frozen --extra dev --extra security
	$(call log_success,Dependencies installed from lockfile)

# Install with thoth dependencies
requirements-thoth: check-uv
	$(call log_info,Installing with thoth dependencies...)
	@uv sync --frozen --extra dev --extra security --extra thoth
	$(call log_success,Dependencies with thoth installed from lockfile)

# Sync environment with uv.lock
requirements-sync: check-uv
	$(call log_info,Syncing environment with uv.lock...)
	@uv sync --frozen --extra dev --extra security
	$(call log_success,Environment synced with lockfile)

# Update lockfile from pyproject.toml
requirements-lock: check-uv
	$(call log_info,Updating uv.lock from pyproject.toml...)
	@uv lock
	$(call log_success,Lockfile updated)

# Legacy: Update requirements files (for backwards compatibility)
requirements-update: requirements-lock
	$(call log_info,Use 'make requirements-lock' to update the lockfile)

# Show requirements status
requirements-status:
	$(call log_section,Requirements Status)
	@echo -e "$(BLUE)Dependencies source:$(NC)"
	@if [ -f uv.lock ]; then \
		echo "  [OK] uv.lock (lockfile - source of truth for installs)"; \
	else \
		echo "  [WARNING] uv.lock missing - run 'make requirements-lock'"; \
	fi
	@if [ -f pyproject.toml ]; then \
		echo "  [OK] pyproject.toml (dependency definitions)"; \
		echo "  Optional dependency groups:"; \
		echo "    - dev (development tools)"; \
		echo "    - thoth (ML/analysis)"; \
		echo "    - security (security tools)"; \
		echo "    - viz (visualization)"; \
	else \
		echo "  [ERROR] pyproject.toml missing"; \
	fi