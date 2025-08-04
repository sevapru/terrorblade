# Requirements management with uv
include make/common.mk

.PHONY: requirements requirements-compile requirements-sync requirements-update

# Main requirements target - compile and install for current platform
requirements: requirements-compile
	$(call log_success,Requirements ready for current platform)

# Compile requirements for current platform only
requirements-compile: check-uv
	$(call log_info,Compiling requirements for current platform...)
	@uv pip compile scripts/requirements.in --output-file requirements.txt --upgrade --quiet
	@uv pip compile scripts/requirements-dev.in --output-file requirements-dev.txt --upgrade --quiet
	$(call log_success,Requirements compiled successfully)

# Sync environment with compiled requirements
requirements-sync: check-uv
	$(call log_info,Syncing environment with requirements...)
	@if [ -f requirements-dev.txt ]; then \
		uv pip sync requirements-dev.txt; \
		$(call log_success,Environment synced with development requirements); \
	else \
		$(call log_warning,requirements-dev.txt not found. Run make requirements-compile first); \
	fi

# Update all dependencies to latest versions
requirements-update: check-uv
	$(call log_info,Updating all dependencies to latest versions...)
	@uv pip compile scripts/requirements.in --output-file requirements.txt --upgrade
	@uv pip compile scripts/requirements-dev.in --output-file requirements-dev.txt --upgrade
	@if [ -f scripts/requirements-cuda.in ]; then \
		uv pip compile scripts/requirements-cuda.in --output-file requirements-cuda.txt --upgrade || true; \
	fi
	$(call log_success,All requirements updated to latest versions)

# Show requirements status
requirements-status:
	$(call log_section,Requirements Status)
	@echo -e "$(BLUE)Core requirements:$(NC)"
	@if [ -f requirements.txt ]; then \
		echo "  ✓ requirements.txt ($(shell wc -l < requirements.txt) packages)"; \
	else \
		echo "  ✗ requirements.txt missing"; \
	fi
	@echo -e "$(BLUE)Development requirements:$(NC)"
	@if [ -f requirements-dev.txt ]; then \
		echo "  ✓ requirements-dev.txt ($(shell wc -l < requirements-dev.txt) packages)"; \
	else \
		echo "  ✗ requirements-dev.txt missing"; \
	fi 