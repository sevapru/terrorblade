# Module installation and management
.PHONY: install-terrorblade install-thoth install-modules list-modules module-status

# Terrorblade module installation
install-terrorblade:
	@echo "Installing Terrorblade module..."
	@echo "Available Terrorblade installation options:"
	@echo "  1. Basic installation      - make install-terrorblade-basic"
	@echo "  2. Full installation       - make install-terrorblade-full"
	@echo "  3. Development installation - make install-terrorblade-dev"

install-terrorblade-basic:
	@echo "Installing Terrorblade (basic)..."
	@echo "Installing core Terrorblade dependencies..."
	@if [ -d "terrorblade/" ]; then \
		uv pip install -e terrorblade/; \
		echo "Terrorblade basic installation completed!"; \
	else \
		echo "Terrorblade directory not found. Installing from current directory..."; \
		uv pip install -e .; \
		echo "Basic installation completed!"; \
	fi

install-terrorblade-full:
	@echo "Installing Terrorblade (full)..."
	@echo "Installing Terrorblade with all dependencies..."
	@if [ -d "terrorblade/" ]; then \
		uv pip install -e "terrorblade/[all]"; \
	else \
		uv pip install -e ".[all]"; \
	fi
	@echo "Terrorblade full installation completed!"

install-terrorblade-dev:
	@echo "Installing Terrorblade (development)..."
	@echo "Installing Terrorblade with development dependencies..."
	@if [ -d "terrorblade/" ]; then \
		uv pip install -e "terrorblade/[dev]"; \
	else \
		uv pip install -e ".[dev]"; \
	fi
	@echo "Terrorblade development installation completed!"

# Thoth module installation
install-thoth:
	@echo "Installing Thoth module..."
	@echo "Available Thoth installation options:"
	@echo "  1. Basic installation      - make install-thoth-basic"
	@echo "  2. Full installation       - make install-thoth-full"
	@echo "  3. Development installation - make install-thoth-dev"

install-thoth-basic:
	@echo "Installing Thoth (basic)..."
	@echo "Installing core Thoth dependencies..."
	@if [ -d "thoth/" ]; then \
		uv pip install -e thoth/; \
		echo "Thoth basic installation completed!"; \
	else \
		echo "Thoth directory not found. Please check project structure."; \
		exit 1; \
	fi

install-thoth-full:
	@echo "Installing Thoth (full)..."
	@echo "Installing Thoth with all dependencies..."
	@if [ -d "thoth/" ]; then \
		uv pip install -e "thoth/[all]"; \
		echo "Thoth full installation completed!"; \
	else \
		echo "Thoth directory not found. Please check project structure."; \
		exit 1; \
	fi

install-thoth-dev:
	@echo "Installing Thoth (development)..."
	@echo "Installing Thoth with development dependencies..."
	@if [ -d "thoth/" ]; then \
		uv pip install -e "thoth/[dev]"; \
		echo "Thoth development installation completed!"; \
	else \
		echo "Thoth directory not found. Please check project structure."; \
		exit 1; \
	fi

# Module management
install-modules: install-terrorblade-basic install-thoth-basic
	@echo "Installing all modules (basic)..."
	@echo "All modules installed successfully!"

list-modules:
	@echo "Available modules:"
	@echo "  - Terrorblade: Security analysis and threat detection"
	@echo "  - Thoth: Knowledge management and data processing"
	@echo ""
	@echo "Installation commands:"
	@echo "  make install-terrorblade-[basic|full|dev]"
	@echo "  make install-thoth-[basic|full|dev]"
	@echo "  make install-modules  (installs basic versions)"

module-status:
	@echo "Checking module installation status..."
	@echo ""
	@echo "Terrorblade status:"
	@python3 -c "try: import terrorblade; print('  ✓ Installed'); print('  Version:', getattr(terrorblade, '__version__', 'Unknown'))" 2>/dev/null || echo "  ✗ Not installed"
	@echo ""
	@echo "Thoth status:"
	@python3 -c "try: import thoth; print('  ✓ Installed'); print('  Version:', getattr(thoth, '__version__', 'Unknown'))" 2>/dev/null || echo "  ✗ Not installed"