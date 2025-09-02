# make/docs.mk
ifndef DOCS_MK_INCLUDED
DOCS_MK_INCLUDED := 1

# Documentation build and deployment targets
.PHONY: docs-build docs-serve docs-deploy docs-clean docs-setup docs-check docs-help

# Setup documentation environment
docs-setup:
	@bash scripts/setup-docs.sh

# Build documentation locally
docs-build:
	@bash scripts/build-docs.sh

# Serve documentation locally
docs-serve:
	@bash scripts/serve-docs.sh

# Clean documentation build artifacts
docs-clean:
	$(call log_section,🧹 Cleaning Documentation Build)
	$(call log_info,Removing build artifacts...)
	@rm -rf docs-mkdocs/site/
	@rm -rf docs-mkdocs/.cache/
	$(call log_success,Documentation build cleaned!)

# Check documentation status
docs-check:
	@bash scripts/check-docs.sh

# Help for documentation commands
docs-help:
	$(call log_section,📚 Documentation Commands Help)
	@echo -e "$(YELLOW)Available commands:$(NC)"
	@echo -e "  $(BLUE)docs-setup$(NC)     - Set up documentation environment"
	@echo -e "  $(BLUE)docs-build$(NC)     - Build documentation locally"
	@echo -e "  $(BLUE)docs-serve$(NC)     - Serve documentation locally (http://127.0.0.1:8000)"
	@echo -e "  $(BLUE)docs-clean$(NC)     - Clean build artifacts"
	@echo -e "  $(BLUE)docs-check$(NC)     - Check documentation status"
	@echo -e "  $(BLUE)docs-help$(NC)      - Show this help message"
	@echo ""
	@echo -e "$(YELLOW)Quick start:$(NC)"
	@echo -e "  1. $(BLUE)make docs-setup$(NC)    # Initial setup"
	@echo -e "  2. $(BLUE)make docs-build$(NC)    # Build documentation"
	@echo -e "  3. $(BLUE)make docs-serve$(NC)    # Preview locally"

endif # DOCS_MK_INCLUDED