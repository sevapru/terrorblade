ifndef CLI_MK_INCLUDED
CLI_MK_INCLUDED := 1

# CLI commands for Terrorblade tools

# Function to run CLI with proper environment
define run_cli
	@if [ -n "$$VIRTUAL_ENV" ]; then \
		python -m terrorblade.examples.analyze_dialogues $(1); \
	elif [ -d ".venv" ]; then \
		.venv/bin/python -m terrorblade.examples.analyze_dialogues $(1); \
	else \
		python -m terrorblade.examples.analyze_dialogues $(1); \
	fi
endef

# Function to run installed CLI command
define run_installed_cli
	@if command -v terrorblade-cli >/dev/null 2>&1; then \
		terrorblade-cli $(1); \
	else \
		$(call log_warning,terrorblade-cli not found. Installing in editable mode...); \
		$(MAKE) install-cli; \
		terrorblade-cli $(1); \
	fi
endef

# Install CLI entry point
install-cli:
	$(call log_section,📱 Installing Terrorblade CLI)
	@if [ -n "$$VIRTUAL_ENV" ]; then \
		pip install -e .; \
	elif [ -d ".venv" ]; then \
		.venv/bin/pip install -e .; \
	else \
		pip install -e .; \
	fi
	$(call log_success,CLI installed! Use 'terrorblade-cli' or 'make cli')

# Run CLI in interactive mode
cli:
	$(call log_section,🔍 Launching Terrorblade CLI)
	$(call log_info,Starting interactive message analysis...)
	$(call run_installed_cli,--interactive)

# Run CLI with specific phone number
cli-phone:
	@if [ -z "$(PHONE)" ]; then \
		$(call log_error,Please provide PHONE number: make cli-phone PHONE=+1234567890); \
		exit 1; \
	fi
	$(call log_section,🔍 Launching Terrorblade CLI for $(PHONE))
	$(call run_installed_cli,--phone $(PHONE) --interactive)

# Run CLI in non-interactive mode
cli-run:
	$(call log_section,🔍 Running Terrorblade CLI Analysis)
	$(call run_installed_cli,)

# Show CLI help
cli-help:
	$(call log_section,📚 Terrorblade CLI Help)
	$(call run_installed_cli,--help)

endif # CLI_MK_INCLUDED