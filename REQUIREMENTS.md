# Requirements Management System

This project uses a unified, platform-independent requirements management system powered by `uv` for both **terrorblade** and **thoth** projects.

## System Overview

The requirements are organized into multiple files:

### Source Files (`.in` files)
- `requirements.in` - Core production dependencies for both projects
- `requirements-dev.in` - Development and testing dependencies  
- `requirements-cuda.in` - GPU/CUDA acceleration dependencies (optional)

### Generated Files (`.txt` files)
- `requirements.txt` - Compiled production dependencies
- `requirements-dev.txt` - Compiled development dependencies  
- `requirements-cuda.txt` - Compiled CUDA dependencies
- `requirements-lock.txt` - Locked production dependencies with hashes
- `requirements-dev-lock.txt` - Locked development dependencies with hashes

**Note**: The `.txt` files are auto-generated and should not be edited manually.

## Quick Start

### Setup Development Environment
```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Setup and install development environment
make install
```

### Requirements Management Commands

```bash
# Compile all requirements files
make requirements-compile

# Install development dependencies
make requirements-dev

# Sync environment with requirements (exact versions)
make requirements-sync

# Update all dependencies to latest versions
make requirements-update

# Create secure lockfiles with hashes
make requirements-lock

# Check for dependency conflicts
make requirements-check

# Show requirements system info
make requirements-info
```

## Security Scanning

The project includes comprehensive security vulnerability scanning:

### Security Commands
```bash
# Run all security scans
make security

# Run GitHub Actions security workflow locally
make security-ci-local

# Generate comprehensive security report
make security-report

# Update security tools
make security-update
```

### Security Tools Included
- **Bandit** - Python code security analysis
- **Safety** - Known vulnerability database checking
- **pip-audit** - Advanced dependency vulnerability scanning with SBOM generation
- **Semgrep** - Static application security testing (SAST)

## Local CI Execution

You can run the complete CI pipeline locally:

```bash
# Run complete CI pipeline (lint, test, security)
make ci-local

# Run individual workflows
make security-ci-local

# Use the provided script for more control
./scripts/run-ci-locally.sh ci
./scripts/run-ci-locally.sh security
./scripts/run-ci-locally.sh all
```

### Using act (Optional)
If you have [act](https://github.com/nektos/act) installed, you can run actual GitHub Actions locally:

```bash
./scripts/run-ci-locally.sh ci --with-act
./scripts/run-ci-locally.sh security --with-act
```

## Platform Support

### Standard Dependencies
The core requirements work on:
- ✅ Linux (x86_64, aarch64)
- ✅ macOS (x86_64, Apple Silicon)
- ✅ Windows (x86_64)

### CUDA Dependencies
CUDA packages are platform-specific and only work on:
- ✅ Linux x86_64 with NVIDIA GPU
- ❌ Other platforms (will fail gracefully)

## Workflow Integration

### GitHub Actions
The project uses unified requirements in CI/CD:

- **CI Workflow** (`.github/workflows/ci.yml`) - Code quality, testing
- **Security Workflow** (`.github/workflows/security.yml`) - Comprehensive security scanning

### Automatic Dependency Updates
- Daily security scans via GitHub Actions
- Dependency review on pull requests
- Supply chain security monitoring

## Best Practices

### Adding Dependencies

1. **Production dependencies**: Add to `requirements.in`
2. **Development dependencies**: Add to `requirements-dev.in`  
3. **CUDA dependencies**: Add to `requirements-cuda.in`
4. Always recompile after changes: `make requirements-compile`

### Version Pinning
- Use `>=` for minimum versions in `.in` files
- Let `uv` resolve exact versions in `.txt` files
- Use lockfiles for production deployments

### Security
- Run security scans before releases: `make security`
- Review security reports in `reports/` directory
- Update dependencies regularly: `make requirements-update`

## Troubleshooting

### Common Issues

**CUDA compilation fails:**
```bash
# Skip CUDA requirements for development
make requirements-compile  # This will fail on CUDA but succeed on others
# Or edit requirements-cuda.in to remove problematic packages
```

**Dependency conflicts:**
```bash
# Check for conflicts
make requirements-check

# Clear cache and retry
rm -rf ~/.cache/uv
make requirements-compile
```

**Security scan failures:**
```bash
# Update security tools
make security-update

# Run individual scans to isolate issues
make security-bandit
make security-safety
```

### Getting Help

1. Check `make help` for all available commands
2. Run `make requirements-info` for system status
3. Check the `reports/` directory for detailed scan results
4. Review GitHub Actions logs for CI failures

## Migration Guide

If migrating from the old pyproject.toml-only system:

1. Dependencies are now centralized in `requirements*.in` files
2. Use `make requirements-compile` instead of `pip install -e .[dev]`
3. Security scanning is now mandatory and comprehensive
4. Use `make ci-local` to test changes before pushing

This system provides better dependency management, enhanced security, and unified workflow execution across both local development and CI/CD environments. 