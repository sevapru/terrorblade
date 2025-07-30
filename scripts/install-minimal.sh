#!/bin/bash
# Terrorblade Minimal Installer
# Usage: curl -fsSL https://yourdomain.com/install.sh | bash

set -e

# Colors
G='\033[0;32m'; Y='\033[1;33m'; B='\033[0;34m'; R='\033[0;31m'; NC='\033[0m'

# Config
REPO="${TERRORBLADE_REPO:-https://github.com/sevapru/terrorblade.git}"
DIR="${INSTALL_DIR:-$HOME/terrorblade}"
BRANCH="${TERRORBLADE_BRANCH:-main}"

# Functions
log() { echo -e "${B}[INFO]${NC} $1"; }
success() { echo -e "${G}[✓]${NC} $1"; }
error() { echo -e "${R}[✗]${NC} $1"; exit 1; }

# Banner
echo -e "${G}🗡️  Terrorblade Installer${NC}"
echo

# Check prerequisites
log "Checking prerequisites..."
command -v git >/dev/null || error "Git required"
command -v python3 >/dev/null || error "Python 3.9+ required"
command -v curl >/dev/null || error "curl required"
success "Prerequisites OK"

# Install uv
log "Installing uv..."
if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi
success "uv ready: $(uv --version 2>/dev/null || echo 'installed')"

# Setup repository
log "Setting up repository..."
if [[ -d "$DIR" ]]; then
    log "Updating existing installation..."
    cd "$DIR" && git pull origin "$BRANCH" 2>/dev/null || {
        cd .. && rm -rf "$DIR" && git clone -b "$BRANCH" "$REPO" "$DIR"
    }
else
    git clone -b "$BRANCH" "$REPO" "$DIR"
fi
cd "$DIR"
success "Repository ready"

# Install dependencies
log "Installing dependencies..."
uv venv --python python3.13
source .venv/bin/activate

# Use make if available, otherwise direct installation
if command -v make >/dev/null && [[ -f Makefile ]]; then
    make install 2>/dev/null || {
        log "Make failed, using direct install..."
        uv pip compile requirements-dev.in --output-file requirements-dev.txt 2>/dev/null || true
        uv pip install -r requirements-dev.txt 2>/dev/null || uv pip install -e .
    }
else
    uv pip compile requirements-dev.in --output-file requirements-dev.txt 2>/dev/null || true
    uv pip install -r requirements-dev.txt 2>/dev/null || uv pip install -e .
fi

# Create activation script
cat > activate.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
source .venv/bin/activate
echo "🗡️  Terrorblade environment activated!"
EOF
chmod +x activate.sh

success "Installation complete!"
echo
echo -e "${Y}🚀 Quick Start:${NC}"
echo -e "   cd $DIR"
echo -e "   source .venv/bin/activate"
echo -e "   make help"