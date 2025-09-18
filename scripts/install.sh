#!/bin/bash

# Terrorblade One-Liner Installation Script
# Usage: curl -fsSL https://raw.githubusercontent.com/sevapru/terrorblade/main/scripts/install.sh | bash
# Or: curl -fsSL https://sevap.ru/install.sh | bash

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
TERRORBLADE_REPO="${TERRORBLADE_REPO:-git@github.com:sevapru/terrorblade.git}"
TERRORBLADE_BRANCH="${TERRORBLADE_BRANCH:-main}"
INSTALL_DIR="${INSTALL_DIR:-$(pwd)/terrorblade}"
PYTHON_VERSION="${PYTHON_VERSION:-3.13}"

# Script info
SCRIPT_VERSION="1.0.0"
SCRIPT_NAME="Terrorblade Installer"

# Print banner
print_banner() {
    echo -e "${PURPLE}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║                    🗡️  TERRORBLADE INSTALLER                 ║"
    echo "║                                                              ║"
    echo "║       Unified AI Platform for Data Analysis & ML            ║"
    echo "║                                                              ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo -e "${CYAN}Version: ${SCRIPT_VERSION}${NC}"
    echo
}

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[⚠]${NC} $1"
}

log_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Error handler
error_exit() {
    log_error "$1"
    echo
    log_error "Installation failed. Please check the error above and try again."
    log_info "For help, visit: https://github.com/sevapru/terrorblade/issues"
    exit 1
}

# Detect OS and architecture
detect_platform() {
    OS=$(uname -s | tr '[:upper:]' '[:lower:]')
    ARCH=$(uname -m)
    
    case $OS in
        linux*)
            PLATFORM="linux"
            ;;
        darwin*)
            PLATFORM="macos"
            ;;
        mingw*|msys*|cygwin*)
            PLATFORM="windows"
            ;;
        *)
            error_exit "Unsupported operating system: $OS"
            ;;
    esac
    
    case $ARCH in
        x86_64|amd64)
            ARCHITECTURE="x64"
            ;;
        arm64|aarch64)
            ARCHITECTURE="arm64"
            ;;
        *)
            log_warning "Architecture $ARCH may not be fully supported"
            ARCHITECTURE="x64"
            ;;
    esac
    
    log_info "Detected platform: $PLATFORM ($ARCHITECTURE)"
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    if [[ $EUID -eq 0 ]]; then
        log_warning "Running as root is not recommended. Consider using a regular user account."
        read -p "Continue anyway? (y/N): " -r
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    # Check Git
    if ! command -v git &> /dev/null; then
        error_exit "Git is required but not installed. Please install Git first."
    fi
    log_success "Git found: $(git --version)"
    
    # Check Python
    PYTHON_CMD=""
    for cmd in python3 python; do
        if command -v $cmd &> /dev/null; then
            PYTHON_VERSION_FOUND=$($cmd --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
            MAJOR=$(echo "$PYTHON_VERSION_FOUND" | cut -d. -f1)
            MINOR=$(echo "$PYTHON_VERSION_FOUND" | cut -d. -f2)
            VERSION_NUM=$((MAJOR * 100 + MINOR))
            if [[ $VERSION_NUM -ge 309 ]]; then
                PYTHON_CMD=$cmd
                break
            fi
        fi
    done
    
    if [[ -z "$PYTHON_CMD" ]]; then
        error_exit "Python 3.9+ is required but not found. Please install Python first."
    fi
    log_success "Python found: $($PYTHON_CMD --version)"
    
    # Check curl
    if ! command -v curl &> /dev/null; then
        error_exit "curl is required but not installed."
    fi
    
    if command -v df &> /dev/null; then
        AVAILABLE_SPACE=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
        if [[ $AVAILABLE_SPACE -lt 8 ]]; then
            log_warning "Low disk space detected. At least 8GB recommended."
        fi
    fi
}

install_uv() {
    log_info "Installing uv (Python package manager)..."
    
    if command -v uv &> /dev/null; then
        log_success "uv already installed: $(uv --version)"
        return
    fi
    
    # Install uv using the official installer
    if ! curl -LsSf https://astral.sh/uv/install.sh | sh; then
        error_exit "Failed to install uv"
    fi
    
    # Add uv to PATH for current session
    export PATH="$HOME/.cargo/bin:$PATH"
    
    if ! command -v uv &> /dev/null; then
        error_exit "uv installation failed - command not found after installation"
    fi
    
    log_success "uv installed successfully: $(uv --version)"
}

# Clone or update repository
setup_repository() {
    log_info "Setting up Terrorblade repository..."
    
    if [[ -d "$INSTALL_DIR" ]]; then
        log_info "Directory $INSTALL_DIR already exists."
        read -p "Update existing installation? (Y/n): " -r
        if [[ $REPLY =~ ^[Nn]$ ]]; then
            log_info "Using existing installation."
            return
        fi
        
        log_info "Updating repository..."
        cd "$INSTALL_DIR"
        if ! git pull origin "$TERRORBLADE_BRANCH"; then
            log_warning "Git pull failed, trying fresh clone..."
            cd ..
            rm -rf "$INSTALL_DIR"
            git clone -b "$TERRORBLADE_BRANCH" "$TERRORBLADE_REPO" "$INSTALL_DIR"
        fi
    else
        log_info "Cloning repository..."
        if ! git clone -b "$TERRORBLADE_BRANCH" "$TERRORBLADE_REPO" "$INSTALL_DIR"; then
            # If SSH fails, try HTTPS as fallback
            if [[ "$TERRORBLADE_REPO" == *"git@"* ]]; then
                log_warning "SSH clone failed, trying HTTPS fallback..."
                HTTPS_REPO="https://github.com/sevapru/terrorblade.git"
                if ! git clone -b "$TERRORBLADE_BRANCH" "$HTTPS_REPO" "$INSTALL_DIR"; then
                    error_exit "Failed to clone repository (both SSH and HTTPS failed)"
                fi
                log_success "Repository cloned using HTTPS fallback"
            else
                error_exit "Failed to clone repository"
            fi
        fi
    fi
    
    cd "$INSTALL_DIR"
    log_success "Repository ready at: $INSTALL_DIR"
}

# Setup virtual environment and install dependencies
setup_environment() {
    log_info "Setting up Python environment..."
    
    cd "$INSTALL_DIR"
    
    # Create virtual environment with uv
    log_info "Creating virtual environment..."
    if ! uv venv --python "$PYTHON_CMD"; then
        error_exit "Failed to create virtual environment"
    fi
    
    # Activate virtual environment
    source .venv/bin/activate
    log_success "Virtual environment created and activated"
    
    # Install dependencies using our unified requirements system
    log_info "Installing dependencies (this may take a few minutes)..."
    
    # Use make install which now includes requirements compilation
    if command -v make &> /dev/null; then
        log_info "Using make for installation..."
        if ! make install; then
            log_warning "Make install failed, trying direct uv installation..."
            # Fallback to direct installation
            install_dependencies_direct
        fi
    else
        log_info "Make not available, using direct installation..."
        install_dependencies_direct
    fi
    
    log_success "Dependencies installed successfully"
}

# Direct dependency installation (fallback)
install_dependencies_direct() {
    log_info "Installing dependencies from pyproject.toml..."

    # Install terrorblade in editable mode with dev dependencies
    uv pip install -e ".[dev,security]"

    # Try to install thoth dependencies if available
    if [[ -d "thoth" ]]; then
        uv pip install -e ".[thoth]" || log_warning "Thoth dependencies installation failed, continuing without them"
    fi
}

# Verify installation
verify_installation() {
    log_info "Verifying installation..."
    
    # Test imports
    if $PYTHON_CMD -c "import terrorblade; print('✓ Terrorblade imported successfully')" 2>/dev/null; then
        log_success "Terrorblade package verification passed"
    else
        error_exit "Terrorblade package verification failed"
    fi
    
    # Test key dependencies
    if $PYTHON_CMD -c "import polars, torch, duckdb; print('✓ Core dependencies available')" 2>/dev/null; then
        log_success "Core dependencies verification passed"
    else
        log_warning "Some core dependencies may not be available"
    fi
    
    # Test CLI (if available)
    if command -v terrorblade &> /dev/null; then
        log_success "Terrorblade CLI available"
    fi
}

# Create convenience scripts
create_shortcuts() {
    log_info "Creating convenience scripts..."
    
    # Create activation script
    cat > activate.sh << 'EOF'
#!/bin/bash
# Terrorblade Environment Activation
cd "$(dirname "$0")"
source .venv/bin/activate
echo "🗡️  Terrorblade environment activated!"
echo "Available commands:"
echo "  make help          - Show all available commands"
echo "  make test          - Run tests"
echo "  make security      - Run security scans"
echo "  make lint          - Run code quality checks"
echo ""
echo "To deactivate: deactivate"
EOF
    chmod +x activate.sh
    
    # Create desktop shortcut (Linux only)
    if [[ "$PLATFORM" == "linux" ]] && command -v xdg-desktop-menu &> /dev/null; then
        DESKTOP_FILE="$HOME/.local/share/applications/terrorblade.desktop"
        mkdir -p "$(dirname "$DESKTOP_FILE")"
        cat > "$DESKTOP_FILE" << EOF
[Desktop Entry]
Name=Terrorblade
Comment=AI Platform for Data Analysis & ML
Exec=gnome-terminal --working-directory="$INSTALL_DIR" -- bash -c "source .venv/bin/activate; bash"
Icon=application-x-executable
Terminal=true
Type=Application
Categories=Development;
EOF
        log_success "Desktop shortcut created"
    fi
}

# Print completion message
print_completion() {
    echo
    echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                   🎉 INSTALLATION COMPLETE! 🎉               ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo
    log_success "Terrorblade has been successfully installed!"
    echo
    echo -e "${CYAN}📍 Installation Location:${NC} $INSTALL_DIR"
    echo -e "${CYAN}🐍 Python Environment:${NC} $INSTALL_DIR/.venv"
    echo
    echo -e "${YELLOW}🚀 Getting Started:${NC}"
    echo -e "   ${BLUE}cd $INSTALL_DIR${NC}"
    echo -e "   ${BLUE}source .venv/bin/activate${NC}  (or: ${BLUE}./activate.sh${NC})"
    echo -e "   ${BLUE}make help${NC}                   # See all available commands"
    echo -e "   ${BLUE}make test${NC}                   # Run tests to verify setup"
    echo
    echo -e "${YELLOW}📚 Next Steps:${NC}"
    echo -e "   • Check out the examples in ${BLUE}examples/${NC}"
    echo -e "   • Read the documentation: ${BLUE}README.md${NC}"
    echo -e "   • Run ${BLUE}make security${NC} to check for vulnerabilities"
    echo -e "   • Join the community: ${BLUE}https://github.com/sevapru/terrorblade${NC}"
    echo
    echo -e "${GREEN}Hope you have a good day! (c) Seva${NC}"
}

# Main installation flow
main() {
    print_banner
    
    log_info "Starting Terrorblade installation..."
    echo
    
    detect_platform
    check_prerequisites
    install_uv
    setup_repository
    setup_environment
    verify_installation
    create_shortcuts
    
    print_completion
}

# Handle interruption
trap 'echo; log_error "Installation interrupted by user"; exit 1' INT

# Run main function
main "$@" 