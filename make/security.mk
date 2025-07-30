# Security scanning and vulnerability management
include make/common.mk

.PHONY: security security-scan security-report

# Main security target - runs all security checks
security: security-scan
	$(call log_success,All security checks completed!)

# Comprehensive security scan
security-scan:
	$(call log_section,🛡️  Security Scan)
	@mkdir -p reports
	$(call log_info,Installing security tools...)
	@uv pip install --quiet "bandit[toml]" safety pip-audit semgrep 2>/dev/null || true
	
	$(call log_info,Running Bandit code analysis...)
	@bandit -r terrorblade/ thoth/ -f json -o reports/bandit-report.json --quiet || true
	@bandit -r terrorblade/ thoth/ --format txt > reports/bandit-summary.txt || true
	
	$(call log_info,Running Safety dependency check...)
	@uv pip freeze > reports/requirements-freeze.txt
	@safety scan --file reports/requirements-freeze.txt --json > reports/safety-report.json || true
	@safety scan --file reports/requirements-freeze.txt > reports/safety-summary.txt || true
	
	$(call log_info,Running pip-audit vulnerability scan...)
	@pip-audit --format=json --output=reports/pip-audit-report.json --progress-spinner=off || true
	@pip-audit > reports/pip-audit-summary.txt || true
	
	$(call log_info,Running Semgrep static analysis...)
	@semgrep --config=auto --json --output=reports/semgrep-report.json terrorblade/ thoth/ --quiet || true
	@semgrep --config=auto terrorblade/ thoth/ > reports/semgrep-summary.txt || true
	
	$(call log_success,Security scan completed - check reports/ for details)

# Generate comprehensive security report
security-report: security-scan
	$(call log_info,Generating security report...)
	@echo "# Security Scan Report" > reports/security-report.md
	@echo "Generated: $(shell date)" >> reports/security-report.md
	@echo "" >> reports/security-report.md
	
	@echo "## Summary" >> reports/security-report.md
	@echo "- **Bandit**: $(shell if [ -f reports/bandit-report.json ]; then echo 'Completed'; else echo 'Failed'; fi)" >> reports/security-report.md
	@echo "- **Safety**: $(shell if [ -f reports/safety-report.json ]; then echo 'Completed'; else echo 'Failed'; fi)" >> reports/security-report.md
	@echo "- **pip-audit**: $(shell if [ -f reports/pip-audit-report.json ]; then echo 'Completed'; else echo 'Failed'; fi)" >> reports/security-report.md
	@echo "- **Semgrep**: $(shell if [ -f reports/semgrep-report.json ]; then echo 'Completed'; else echo 'Failed'; fi)" >> reports/security-report.md
	
	@echo "" >> reports/security-report.md
	@echo "## Files Generated" >> reports/security-report.md
	@echo "- JSON reports: \`reports/*-report.json\`" >> reports/security-report.md
	@echo "- Text summaries: \`reports/*-summary.txt\`" >> reports/security-report.md
	
	$(call log_success,Security report generated: reports/security-report.md)

# Clean security reports
security-clean:
	$(call log_info,Cleaning security reports...)
	@rm -rf reports/
	$(call log_success,Security reports cleaned) 