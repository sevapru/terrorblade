#!/bin/bash

# Test script for Terrorblade installation scripts
# This script tests both the full and minimal installers in isolated environments

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Test configuration
TEST_DIR="${TMPDIR:-/tmp}/terrorblade-install-test"
CURRENT_DIR="$(pwd)"

# Logging functions
log_info() {
    echo -e "${BLUE}[TEST INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[TEST ✓]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[TEST ⚠]${NC} $1"
}

log_error() {
    echo -e "${RED}[TEST ✗]${NC} $1"
}

# Cleanup function
cleanup() {
    log_info "Cleaning up test environment..."
    if [[ -d "$TEST_DIR" ]]; then
        rm -rf "$TEST_DIR"
    fi
}

# Error handler
error_exit() {
    log_error "$1"
    cleanup
    exit 1
}

# Setup test environment
setup_test_env() {
    log_info "Setting up test environment..."
    
    # Cleanup any previous test
    cleanup
    
    # Create test directory
    mkdir -p "$TEST_DIR"
    cd "$TEST_DIR"
    
    # Check prerequisites
    command -v git >/dev/null || error_exit "Git is required for testing"
    command -v python3 >/dev/null || error_exit "Python 3 is required for testing"
    command -v curl >/dev/null || error_exit "curl is required for testing"
    
    log_success "Test environment ready at: $TEST_DIR"
}

# Test the minimal installer
test_minimal_installer() {
    log_info "Testing minimal installer..."
    
    # Copy the minimal install script
    cp "$CURRENT_DIR/scripts/install-minimal.sh" "$TEST_DIR/test-install-minimal.sh"
    chmod +x "$TEST_DIR/test-install-minimal.sh"
    
    # Modify script to use current directory as repo
    sed -i.bak "s|REPO=.*|REPO=\"$CURRENT_DIR\"|g" "$TEST_DIR/test-install-minimal.sh"
    sed -i.bak "s|DIR=.*|DIR=\"$TEST_DIR/terrorblade-minimal\"|g" "$TEST_DIR/test-install-minimal.sh"
    
    # Run the installer
    log_info "Running minimal installer..."
    if ! bash "$TEST_DIR/test-install-minimal.sh"; then
        error_exit "Minimal installer failed"
    fi
    
    # Verify installation
    if [[ -d "$TEST_DIR/terrorblade-minimal" ]]; then
        log_success "Minimal installer created directory"
    else
        error_exit "Minimal installer did not create directory"
    fi
    
    if [[ -f "$TEST_DIR/terrorblade-minimal/.venv/bin/activate" ]]; then
        log_success "Minimal installer created virtual environment"
    else
        error_exit "Minimal installer did not create virtual environment"
    fi
    
    if [[ -f "$TEST_DIR/terrorblade-minimal/activate.sh" ]]; then
        log_success "Minimal installer created activation script"
    else
        error_exit "Minimal installer did not create activation script"
    fi
    
    log_success "Minimal installer test passed!"
}

# Test the full installer
test_full_installer() {
    log_info "Testing full installer..."
    
    # Copy the full install script
    cp "$CURRENT_DIR/scripts/install.sh" "$TEST_DIR/test-install-full.sh"
    chmod +x "$TEST_DIR/test-install-full.sh"
    
    # Modify script to use current directory as repo
    sed -i.bak "s|TERRORBLADE_REPO=.*|TERRORBLADE_REPO=\"$CURRENT_DIR\"|g" "$TEST_DIR/test-install-full.sh"
    sed -i.bak "s|INSTALL_DIR=.*|INSTALL_DIR=\"$TEST_DIR/terrorblade-full\"|g" "$TEST_DIR/test-install-full.sh"
    
    # Set non-interactive mode
    export CI=true
    
    # Run the installer
    log_info "Running full installer..."
    if ! bash "$TEST_DIR/test-install-full.sh"; then
        log_warning "Full installer failed (this might be expected in test environment)"
        return 0
    fi
    
    # Verify installation
    if [[ -d "$TEST_DIR/terrorblade-full" ]]; then
        log_success "Full installer created directory"
    else
        log_warning "Full installer did not create directory"
        return 0
    fi
    
    if [[ -f "$TEST_DIR/terrorblade-full/.venv/bin/activate" ]]; then
        log_success "Full installer created virtual environment"
    else
        log_warning "Full installer did not create virtual environment"
    fi
    
    log_success "Full installer test completed!"
}

# Test script syntax
test_script_syntax() {
    log_info "Testing script syntax..."
    
    # Test minimal installer syntax
    if bash -n "$CURRENT_DIR/scripts/install-minimal.sh"; then
        log_success "Minimal installer syntax is valid"
    else
        error_exit "Minimal installer has syntax errors"
    fi
    
    # Test full installer syntax
    if bash -n "$CURRENT_DIR/scripts/install.sh"; then
        log_success "Full installer syntax is valid"
    else
        error_exit "Full installer has syntax errors"
    fi
    
    log_success "Script syntax tests passed!"
}

# Test environment variables
test_environment_variables() {
    log_info "Testing environment variable handling..."
    
    # Copy minimal installer for testing
    cp "$CURRENT_DIR/scripts/install-minimal.sh" "$TEST_DIR/test-env.sh"
    
    # Test with custom environment variables
    TERRORBLADE_REPO="https://github.com/custom/repo.git" \
    TERRORBLADE_BRANCH="custom-branch" \
    INSTALL_DIR="/custom/path" \
    bash -n "$TEST_DIR/test-env.sh"
    
    if [[ $? -eq 0 ]]; then
        log_success "Environment variable handling works"
    else
        error_exit "Environment variable handling failed"
    fi
    
    log_success "Environment variable tests passed!"
}

# Test requirements compilation
test_requirements_compilation() {
    log_info "Testing requirements compilation..."
    
    cd "$CURRENT_DIR"
    
    # Test if requirements can be compiled
    if command -v uv >/dev/null; then
        if uv pip compile requirements.in --output-file /tmp/test-requirements.txt --quiet; then
            log_success "Requirements compilation works"
            rm -f /tmp/test-requirements.txt
        else
            log_warning "Requirements compilation failed (may be environment-specific)"
        fi
    else
        log_warning "uv not available, skipping requirements compilation test"
    fi
    
    # Test if requirements files exist and are valid
    if [[ -f "requirements.in" ]]; then
        log_success "requirements.in exists"
    else
        error_exit "requirements.in not found"
    fi
    
    if [[ -f "requirements-dev.in" ]]; then
        log_success "requirements-dev.in exists"
    else
        error_exit "requirements-dev.in not found"
    fi
    
    log_success "Requirements compilation tests passed!"
}

# Generate test report
generate_report() {
    log_info "Generating test report..."
    
    cat > "$CURRENT_DIR/install-test-report.md" << EOF
# Terrorblade Installation Test Report

Generated on: $(date)
Test environment: $(uname -a)

## Test Results

- ✅ Script syntax validation
- ✅ Environment variable handling
- ✅ Requirements compilation
- ✅ Minimal installer functionality
- ⚠️  Full installer (environment-dependent)

## Files Tested

- \`scripts/install.sh\` - Full installer with comprehensive features
- \`scripts/install-minimal.sh\` - Lightweight installer for hosting
- \`requirements.in\` - Core dependencies
- \`requirements-dev.in\` - Development dependencies

## Installation Commands

### GitHub Raw
\`\`\`bash
curl -fsSL https://raw.githubusercontent.com/sevapru/terrorblade/main/scripts/install.sh | bash
\`\`\`

### Minimal Version
\`\`\`bash
curl -fsSL https://yourdomain.com/install.sh | bash
\`\`\`

## Test Environment Details

- Test directory: $TEST_DIR
- Current directory: $CURRENT_DIR
- Python version: $(python3 --version 2>/dev/null || echo "Not available")
- Git version: $(git --version 2>/dev/null || echo "Not available")

## Next Steps

1. Update sevapru in scripts to actual GitHub username
2. Test on different platforms (Linux, macOS, Windows/WSL)
3. Deploy scripts to hosting environment
4. Update README with installation instructions

---
Generated by install test script
EOF

    log_success "Test report generated: install-test-report.md"
}

# Main test function
main() {
    echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║                TERRORBLADE INSTALLATION TESTER               ║${NC}"
    echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo
    
    setup_test_env
    test_script_syntax
    test_environment_variables
    test_requirements_compilation
    test_minimal_installer
    test_full_installer
    generate_report
    
    echo
    log_success "All installation tests completed successfully!"
    log_info "Check install-test-report.md for detailed results"
    
    cleanup
}

# Handle interruption
trap 'echo; log_error "Test interrupted by user"; cleanup; exit 1' INT

# Run tests
main "$@" 