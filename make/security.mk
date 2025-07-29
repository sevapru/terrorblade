# Security scanning and vulnerability management
.PHONY: security security-scan security-audit security-bandit security-safety security-semgrep security-pip-audit security-update security-report security-ci-local security-full

# Colors for output
SECURITY_YELLOW := \033[1;33m
SECURITY_GREEN := \033[1;32m
SECURITY_BLUE := \033[1;34m
SECURITY_RED := \033[1;31m
SECURITY_NC := \033[0m

# Main security target - runs all security checks
security: security-scan
	@echo -e "$(SECURITY_GREEN)All security checks completed!$(SECURITY_NC)"

# Comprehensive security scan (equivalent to CI security job)
security-scan: security-bandit security-safety security-pip-audit security-semgrep
	@echo -e "$(SECURITY_GREEN)Security scan completed successfully!$(SECURITY_NC)"

# Code security analysis with bandit
security-bandit:
	@echo -e "$(SECURITY_BLUE)Running Bandit security linter...$(SECURITY_NC)"
	@mkdir -p reports
	@uv pip install --quiet "bandit[toml]" 2>/dev/null || true
	@if [ -d .venv ]; then \
		source .venv/bin/activate && bandit -r terrorblade/ thoth/ -f json -o reports/bandit-report.json --quiet || true; \
		source .venv/bin/activate && bandit -r terrorblade/ thoth/ --format txt || echo -e "$(SECURITY_YELLOW)Bandit found potential security issues$(SECURITY_NC)"; \
	else \
		bandit -r terrorblade/ thoth/ -f json -o reports/bandit-report.json --quiet || true; \
		bandit -r terrorblade/ thoth/ --format txt || echo -e "$(SECURITY_YELLOW)Bandit found potential security issues$(SECURITY_NC)"; \
	fi

# Dependency vulnerability scanning with safety
security-safety:
	@echo -e "$(SECURITY_BLUE)Running Safety dependency check...$(SECURITY_NC)"
	@mkdir -p reports
	@uv pip install --quiet safety 2>/dev/null || true
	@echo -e "$(SECURITY_BLUE)Generating requirements for safety check...$(SECURITY_NC)"
	@uv pip freeze > reports/requirements-freeze.txt
	@if [ -d .venv ]; then \
		source .venv/bin/activate && safety scan --target reports/requirements-freeze.txt --format json > reports/safety-report.json || true; \
		source .venv/bin/activate && safety scan --target reports/requirements-freeze.txt || echo -e "$(SECURITY_YELLOW)Safety found vulnerable dependencies$(SECURITY_NC)"; \
	else \
		safety scan --target reports/requirements-freeze.txt --format json > reports/safety-report.json || true; \
		safety scan --target reports/requirements-freeze.txt || echo -e "$(SECURITY_YELLOW)Safety found vulnerable dependencies$(SECURITY_NC)"; \
	fi

# Advanced dependency audit with pip-audit
security-pip-audit:
	@echo -e "$(SECURITY_BLUE)Running pip-audit vulnerability scan...$(SECURITY_NC)"
	@mkdir -p reports
	@uv pip install --quiet pip-audit 2>/dev/null || true
	@if [ -d .venv ]; then \
		source .venv/bin/activate && pip-audit --format=json --output=reports/pip-audit-report.json --progress-spinner=off || true; \
		source .venv/bin/activate && pip-audit --format=cyclonedx-json --output=reports/sbom.json --progress-spinner=off || true; \
		source .venv/bin/activate && pip-audit || echo -e "$(SECURITY_YELLOW)pip-audit found vulnerable dependencies$(SECURITY_NC)"; \
	else \
		pip-audit --format=json --output=reports/pip-audit-report.json --progress-spinner=off || true; \
		pip-audit --format=cyclonedx-json --output=reports/sbom.json --progress-spinner=off || true; \
		pip-audit || echo -e "$(SECURITY_YELLOW)pip-audit found vulnerable dependencies$(SECURITY_NC)"; \
	fi

# Static analysis with semgrep
security-semgrep:
	@echo -e "$(SECURITY_BLUE)Running Semgrep static analysis...$(SECURITY_NC)"
	@mkdir -p reports
	@uv pip install --quiet semgrep 2>/dev/null || true
	@if [ -d .venv ]; then \
		source .venv/bin/activate && semgrep --config=auto --json --output=reports/semgrep-report.json terrorblade/ thoth/ --quiet || true; \
		source .venv/bin/activate && semgrep --config=auto terrorblade/ thoth/ || echo -e "$(SECURITY_YELLOW)Semgrep found potential issues$(SECURITY_NC)"; \
	else \
		semgrep --config=auto --json --output=reports/semgrep-report.json terrorblade/ thoth/ --quiet || true; \
		semgrep --config=auto terrorblade/ thoth/ || echo -e "$(SECURITY_YELLOW)Semgrep found potential issues$(SECURITY_NC)"; \
	fi

# Update all security tools
security-update:
	@echo -e "$(SECURITY_BLUE)Updating security tools...$(SECURITY_NC)"
	@uv pip install --upgrade "bandit[toml]" safety pip-audit semgrep
	@echo -e "$(SECURITY_GREEN)Security tools updated!$(SECURITY_NC)"

# Generate comprehensive security report
security-report: security-scan
	@echo -e "$(SECURITY_BLUE)Generating security report...$(SECURITY_NC)"
	@mkdir -p reports
	@echo "# Security Scan Report" > reports/security-summary.md
	@echo "Generated on: $$(date)" >> reports/security-summary.md
	@echo "" >> reports/security-summary.md
	@echo "## Bandit Results" >> reports/security-summary.md
	@if [ -f reports/bandit-report.json ]; then \
		echo "- Report: reports/bandit-report.json" >> reports/security-summary.md; \
	fi
	@echo "" >> reports/security-summary.md
	@echo "## Safety Results" >> reports/security-summary.md
	@if [ -f reports/safety-report.json ]; then \
		echo "- Report: reports/safety-report.json" >> reports/security-summary.md; \
	fi
	@echo "" >> reports/security-summary.md
	@echo "## pip-audit Results" >> reports/security-summary.md
	@if [ -f reports/pip-audit-report.json ]; then \
		echo "- Vulnerability Report: reports/pip-audit-report.json" >> reports/security-summary.md; \
		echo "- SBOM: reports/sbom.json" >> reports/security-summary.md; \
	fi
	@echo "" >> reports/security-summary.md
	@echo "## Semgrep Results" >> reports/security-summary.md
	@if [ -f reports/semgrep-report.json ]; then \
		echo "- Report: reports/semgrep-report.json" >> reports/security-summary.md; \
	fi
	@echo -e "$(SECURITY_GREEN)Security report generated at reports/security-summary.md$(SECURITY_NC)"

# Run equivalent of GitHub Actions security workflow locally
security-ci-local: 
	@echo -e "$(SECURITY_BLUE)Running local equivalent of GitHub Actions security workflow...$(SECURITY_NC)"
	@mkdir -p reports
	@echo -e "$(SECURITY_BLUE)Step 1: Bandit Security Linter$(SECURITY_NC)"
	@make security-bandit
	@echo -e "$(SECURITY_BLUE)Step 2: Safety Security Check$(SECURITY_NC)" 
	@make security-safety
	@echo -e "$(SECURITY_BLUE)Step 3: pip-audit Vulnerability Scan$(SECURITY_NC)"
	@make security-pip-audit
	@echo -e "$(SECURITY_BLUE)Step 4: Semgrep Static Analysis$(SECURITY_NC)"
	@make security-semgrep
	@echo -e "$(SECURITY_GREEN)Local CI security workflow completed!$(SECURITY_NC)"
	@echo -e "$(SECURITY_BLUE)Reports available in: reports/$(SECURITY_NC)"

# Full security audit with detailed reporting
security-full: security-update security-ci-local security-report
	@echo -e "$(SECURITY_GREEN)Full security audit completed!$(SECURITY_NC)"
	@echo -e "$(SECURITY_BLUE)Review reports in the reports/ directory$(SECURITY_NC)"

# Check if reports directory exists and create it
.PHONY: ensure-reports-dir
ensure-reports-dir:
	@mkdir -p reports 