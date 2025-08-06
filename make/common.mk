ifndef COMMON_MK_INCLUDED
COMMON_MK_INCLUDED := 1

# Color definitions (shared across all makefiles)
YELLOW := \033[1;33m
GREEN := \033[1;32m
BLUE := \033[1;34m
RED := \033[1;31m
CYAN := \033[0;36m
PURPLE := \033[0;35m
NC := \033[0m

# Common echo functions
define log_info
	@echo -e "$(BLUE)[INFO]$(NC) $(1)"
endef

define log_success
	@echo -e "$(GREEN)[OK]$(NC) $(1)"
endef

define log_warning
	@echo -e "$(YELLOW)[⚠]$(NC) $(1)"
endef

define log_error
	@echo -e "$(RED)[ERROR]$(NC) $(1)"
endef

define log_section
	@echo ""
	@echo -e "$(CYAN)$(1)$(NC)"
endef

# Common utility functions
define check_command
	@command -v $(1) >/dev/null || { $(call log_error,$(1) not found. Please install $(1) first); exit 1; }
endef

# Environment checks
check-env:
	@if [ ! -f .env ]; then \
		$(call log_warning,.env file not found. You may want to create one based on .env.example); \
	fi

check-uv:
	$(call check_command,uv)

check-python:
	$(call check_command,python3)
	$(call log_info,Checking Python version compatibility...)
	@python3 -c '\
	import sys, re; \
	required = re.search(r"requires-python = \"(.+?)\"", open("pyproject.toml").read()).group(1); \
	current = f"{sys.version_info.major}.{sys.version_info.minor}"; \
	from packaging import specifiers; \
	spec = specifiers.SpecifierSet(required); \
	is_compatible = spec.contains(current); \
	print(f"  Required: {required}"); \
	print(f"  Current:  {current}"); \
	print(f"  Status:   {"[OK] Compatible" if is_compatible else "[ERROR] Incompatible"}"); \
	exit(0 if is_compatible else 1);' \
	|| { $(call log_error,Python version incompatible with project requirements); exit 1; }

# Smart virtual environment setup
setup-venv: check-uv check-python
	@if [ -n "$$VIRTUAL_ENV" ]; then \
		echo -e "$(GREEN)[OK]$(NC) Using active virtual environment: $$VIRTUAL_ENV"; \
	elif [ -d ".venv" ]; then \
		echo -e "$(BLUE)[INFO]$(NC) Found existing .venv directory"; \
		echo -e "$(GREEN)[OK]$(NC) Virtual environment ready at .venv"; \
	else \
		echo -e "$(BLUE)[INFO]$(NC) Creating new virtual environment..."; \
		uv venv --python python3; \
		echo -e "$(GREEN)[OK]$(NC) Virtual environment created at .venv"; \
	fi

# Validate and sync virtual environment
validate-venv:
	@if [ -n "$$VIRTUAL_ENV" ]; then \
		echo -e "$(GREEN)[OK]$(NC) Virtual environment is active"; \
	elif [ -d ".venv" ]; then \
		echo -e "$(YELLOW)[⚠]$(NC) Virtual environment exists but not activated"; \
		echo -e "$(BLUE)[INFO]$(NC) To activate: source .venv/bin/activate"; \
	else \
		echo -e "$(RED)[✗]$(NC) No virtual environment found"; \
		exit 1; \
	fi

# Show project info
show-info: check-python
	@echo ""
	@echo -e "$(PURPLE)🗡️  Terrorblade$(NC) v$$(grep -m1 'version = ' pyproject.toml | cut -d'"' -f2)"
	@echo -e "$(BLUE)Description:$(NC) $$(grep -m1 'description = ' pyproject.toml | cut -d'"' -f2)"
	@echo -e "$(BLUE)Python:$(NC) $$(python3 --version | cut -d' ' -f2)"
	@echo -e "$(BLUE)Environment:$(NC) $${VIRTUAL_ENV:-None}"
	@echo ""

endif # COMMON_MK_INCLUDED 