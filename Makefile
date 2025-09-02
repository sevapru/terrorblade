include make/common.mk
include make/requirements.mk
include make/security.mk
include make/test.mk
include make/docs.mk

.PHONY: help install test check format requirements security clean show-info setup-mcp docs-build docs-deploy docs-serve docs-setup docs-clean docs-check docs-help

.DEFAULT_GOAL := help

help:
	@echo -e "$(PURPLE)🗡️  Terrorblade Makefile$(NC)"
	@echo -e "$(BLUE)=======================$(NC)"
	@echo ""
	@echo -e "$(YELLOW)Main Commands:$(NC)"
	@echo "  install        Set up complete development environment"
	@echo "  test           Run all tests"
	@echo "  check          Run code quality checks (linting, formatting)"
	@echo "  format         Auto-fix code formatting and linting issues"
	@echo "  requirements   Compile and manage dependencies"
	@echo "  security       Run security scans"
	@echo "  setup-mcp      Set up MCP server for Claude integration"
	@echo "  clean          Clean build artifacts and cache"
	@echo "  show-info      Display project information"
	@echo ""
	@echo -e "$(YELLOW)Documentation:$(NC)"
	@echo "  docs-setup     Set up documentation environment"
	@echo "  docs-build     Build documentation locally"
	@echo "  docs-serve     Serve documentation locally"
	@echo "  docs-deploy    Deploy documentation to VPS"
	@echo "  docs-clean     Clean documentation build"
	@echo "  docs-check     Check documentation status"
	@echo "  docs-help      Show documentation help"
	@echo ""
	@echo -e "$(YELLOW)Quick Start:$(NC)"
	@echo "  make install   # Set up everything"
	@echo "  make test      # Run tests"
	@echo "  make check     # Check code quality"
	@echo "  make docs-setup # Set up documentation"
	@echo ""
	@echo -e "$(BLUE)For more details: $(CYAN)https://github.com/sevapru/terrorblade$(NC)"

# Main installation target - unified dev environment
install: setup-venv check-env requirements-compile
	$(call log_section,🚀 Installing Terrorblade)
	$(call log_info,Installing development dependencies...)
	@if [ -n "$$VIRTUAL_ENV" ]; then \
		echo -e "$(GREEN)[✓]$(NC) Using active environment: $$VIRTUAL_ENV"; \
		uv pip install -r requirements-dev.txt; \
	elif [ -d ".venv" ]; then \
		echo -e "$(BLUE)[INFO]$(NC) Using .venv environment"; \
		uv pip install -r requirements-dev.txt --python .venv/bin/python; \
	else \
		echo -e "$(BLUE)[INFO]$(NC) Installing in current Python environment"; \
		uv pip install -r requirements-dev.txt; \
	fi
	
	$(call log_info,Installing terrorblade in editable mode...)
	@if [ -n "$$VIRTUAL_ENV" ]; then \
		uv pip install -e .; \
	elif [ -d ".venv" ]; then \
		uv pip install -e . --python .venv/bin/python; \
	else \
		uv pip install -e .; \
	fi
	
	@if [ -d "thoth" ]; then \
		echo -e "$(BLUE)[INFO]$(NC) Installing thoth in editable mode..."; \
		if [ -n "$$VIRTUAL_ENV" ]; then \
			uv pip install -e thoth || echo -e "$(YELLOW)[⚠]$(NC) Thoth installation failed - continuing without it"; \
		elif [ -d ".venv" ]; then \
			uv pip install -e thoth --python .venv/bin/python || echo -e "$(YELLOW)[⚠]$(NC) Thoth installation failed - continuing without it"; \
		else \
			uv pip install -e thoth || echo -e "$(YELLOW)[⚠]$(NC) Thoth installation failed - continuing without it"; \
		fi; \
	fi
	
	$(call log_info,Verifying installation...)
	@if [ -n "$$VIRTUAL_ENV" ]; then \
		python -c "import terrorblade; print('✓ Terrorblade imported successfully')" || { echo -e "$(RED)[✗]$(NC) Terrorblade import failed"; exit 1; }; \
	elif [ -d ".venv" ]; then \
		.venv/bin/python -c "import terrorblade; print('✓ Terrorblade imported successfully')" || { echo -e "$(RED)[✗]$(NC) Terrorblade import failed"; exit 1; }; \
	else \
		python -c "import terrorblade; print('✓ Terrorblade imported successfully')" || { echo -e "$(RED)[✗]$(NC) Terrorblade import failed"; exit 1; }; \
	fi
	
	@# Validate essential tools are available  
	@if [ -n "$$VIRTUAL_ENV" ]; then \
		python -c "import black, isort, ruff; print('✓ Code quality tools available')" 2>/dev/null || echo -e "$(YELLOW)[⚠]$(NC) Some code quality tools may not be available"; \
	elif [ -d ".venv" ]; then \
		.venv/bin/python -c "import black, isort, ruff; print('✓ Code quality tools available')" 2>/dev/null || echo -e "$(YELLOW)[⚠]$(NC) Some code quality tools may not be available"; \
	else \
		python -c "import black, isort, ruff; print('✓ Code quality tools available')" 2>/dev/null || echo -e "$(YELLOW)[⚠]$(NC) Some code quality tools may not be available"; \
	fi
	
	$(call log_success,Installation completed!)
	$(call log_section,🎯 Next Steps)
	@echo -e "  $(BLUE)make test$(NC)      # Run tests"
	@echo -e "  $(BLUE)make check$(NC)     # Check code quality" 
	@echo -e "  $(BLUE)make security$(NC)  # Run security scans"
	@echo -e "  $(BLUE)make docs-setup$(NC) # Set up documentation"
	@echo ""
	@if [ -z "$$VIRTUAL_ENV" ] && [ -d ".venv" ]; then \
		echo -e "$(YELLOW)💡 To activate the environment:$(NC) $(GREEN)source .venv/bin/activate$(NC)"; \
		echo ""; \
	fi

# MCP server setup for Claude integration
setup-mcp:
	$(call log_section,🤖 Setting up MCP Server for Claude)
	@if [ ! -f "scripts/setup-mcp-service.sh" ]; then \
		echo -e "$(RED)[✗]$(NC) setup-mcp-service.sh not found in scripts/"; \
		exit 1; \
	fi
	@chmod +x scripts/setup-mcp-service.sh
	@bash scripts/setup-mcp-service.sh
	$(call log_success,MCP setup completed!)
	@echo ""
	@echo -e "$(YELLOW)📚 For more details, see:$(NC) $(CYAN)terrorblade/mcp/README.md$(NC)" 