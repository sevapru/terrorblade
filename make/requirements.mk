# Requirements management with uv
.PHONY: requirements requirements-compile requirements-sync requirements-update requirements-dev requirements-cuda requirements-lock requirements-check requirements-export

# Colors for requirements output
REQ_YELLOW := \033[1;33m
REQ_GREEN := \033[1;32m
REQ_BLUE := \033[1;34m
REQ_RED := \033[1;31m
REQ_NC := \033[0m

# Default requirements target
requirements: requirements-compile
	@echo -e "$(REQ_GREEN)Requirements management completed!$(REQ_NC)"

# Compile all requirements files using uv
requirements-compile: check-uv
	@echo -e "$(REQ_BLUE)Compiling requirements files with uv...$(REQ_NC)"
	@echo -e "$(REQ_BLUE)Compiling production requirements...$(REQ_NC)"
	@uv pip compile requirements.in --output-file requirements.txt --upgrade
	@echo -e "$(REQ_BLUE)Compiling development requirements...$(REQ_NC)"
	@uv pip compile requirements-dev.in --output-file requirements-dev.txt --upgrade
	@echo -e "$(REQ_BLUE)Compiling CUDA requirements...$(REQ_NC)"
	@uv pip compile requirements-cuda.in --output-file requirements-cuda.txt --upgrade
	@echo -e "$(REQ_GREEN)All requirements files compiled successfully!$(REQ_NC)"

# Sync current environment with requirements
requirements-sync: check-uv
	@echo -e "$(REQ_BLUE)Syncing environment with requirements...$(REQ_NC)"
	@if [ -f requirements-dev.txt ]; then \
		uv pip sync requirements-dev.txt; \
	else \
		echo -e "$(REQ_YELLOW)requirements-dev.txt not found. Run 'make requirements-compile' first.$(REQ_NC)"; \
		exit 1; \
	fi
	@echo -e "$(REQ_GREEN)Environment synced successfully!$(REQ_NC)"

# Update requirements to latest versions
requirements-update: check-uv
	@echo -e "$(REQ_BLUE)Updating all requirements to latest versions...$(REQ_NC)"
	@uv pip compile requirements.in --output-file requirements.txt --upgrade
	@uv pip compile requirements-dev.in --output-file requirements-dev.txt --upgrade  
	@uv pip compile requirements-cuda.in --output-file requirements-cuda.txt --upgrade
	@echo -e "$(REQ_GREEN)All requirements updated to latest versions!$(REQ_NC)"

# Install development requirements
requirements-dev: requirements-compile
	@echo -e "$(REQ_BLUE)Installing development requirements...$(REQ_NC)"
	@uv pip install -r requirements-dev.txt
	@echo -e "$(REQ_GREEN)Development requirements installed!$(REQ_NC)"

# Install CUDA requirements
requirements-cuda: requirements-compile
	@echo -e "$(REQ_BLUE)Installing CUDA requirements...$(REQ_NC)"
	@uv pip install -r requirements-cuda.txt
	@echo -e "$(REQ_GREEN)CUDA requirements installed!$(REQ_NC)"

# Lock requirements (create exact version lockfile)
requirements-lock: check-uv
	@echo -e "$(REQ_BLUE)Creating requirements lockfile...$(REQ_NC)"
	@uv pip compile requirements.in --output-file requirements-lock.txt --generate-hashes
	@uv pip compile requirements-dev.in --output-file requirements-dev-lock.txt --generate-hashes
	@echo -e "$(REQ_GREEN)Lockfiles created with hashes for security!$(REQ_NC)"

# Check requirements for inconsistencies
requirements-check: check-uv
	@echo -e "$(REQ_BLUE)Checking requirements for inconsistencies...$(REQ_NC)"
	@if [ -f requirements.txt ]; then \
		uv pip check; \
		echo -e "$(REQ_GREEN)Requirements check passed!$(REQ_NC)"; \
	else \
		echo -e "$(REQ_RED)requirements.txt not found. Run 'make requirements-compile' first.$(REQ_NC)"; \
		exit 1; \
	fi

# Export current environment to requirements format
requirements-export: check-uv
	@echo -e "$(REQ_BLUE)Exporting current environment...$(REQ_NC)"
	@mkdir -p exports
	@uv pip freeze > exports/current-environment.txt
	@echo -e "$(REQ_GREEN)Current environment exported to exports/current-environment.txt$(REQ_NC)"

# Show requirements info
requirements-info:
	@echo -e "$(REQ_GREEN)Requirements Management System$(REQ_NC)"
	@echo -e "$(REQ_BLUE)================================$(REQ_NC)"
	@echo ""
	@echo -e "$(REQ_YELLOW)Available requirement files:$(REQ_NC)"
	@echo "  requirements.in          - Core production dependencies"
	@echo "  requirements-dev.in      - Development dependencies"  
	@echo "  requirements-cuda.in     - CUDA/GPU dependencies"
	@echo ""
	@echo -e "$(REQ_YELLOW)Generated files:$(REQ_NC)"
	@if [ -f requirements.txt ]; then echo "  ✓ requirements.txt"; else echo "  ✗ requirements.txt"; fi
	@if [ -f requirements-dev.txt ]; then echo "  ✓ requirements-dev.txt"; else echo "  ✗ requirements-dev.txt"; fi
	@if [ -f requirements-cuda.txt ]; then echo "  ✓ requirements-cuda.txt"; else echo "  ✗ requirements-cuda.txt"; fi
	@echo ""
	@echo -e "$(REQ_YELLOW)Common commands:$(REQ_NC)"
	@echo "  make requirements-compile    - Compile all .in files to .txt"
	@echo "  make requirements-sync       - Sync environment with requirements"
	@echo "  make requirements-update     - Update to latest versions"
	@echo "  make requirements-lock       - Create secure lockfiles with hashes" 