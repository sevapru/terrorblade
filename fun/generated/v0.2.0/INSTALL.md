# 🗡️ Terrorblade Installation Guide

Multiple installation methods for getting Terrorblade up and running quickly on your system.

## 🚀 Quick One-Liner Installation

### Option 1: GitHub Raw (Recommended)
```bash
curl -fsSL https://raw.githubusercontent.com/sevapru/terrorblade/main/scripts/install.sh | bash
```

### Option 2: Custom Domain
```bash
curl -fsSL https://sevap.ru/terrorblade/install.sh | bash
```

### Option 3: With Custom Settings
```bash
# Install to custom directory
INSTALL_DIR="$HOME/my-terrorblade" curl -fsSL https://raw.githubusercontent.com/sevapru/terrorblade/main/scripts/install.sh | bash

# Install specific branch
TERRORBLADE_BRANCH="dev" curl -fsSL https://raw.githubusercontent.com/sevapru/terrorblade/main/scripts/install.sh | bash

# Install with specific Python version
PYTHON_VERSION="3.11" curl -fsSL https://raw.githubusercontent.com/sevapru/terrorblade/main/scripts/install.sh | bash
```

## 📋 Prerequisites

The installer will check for these automatically, but you can install them beforehand:

- **Git** - Version control system
- **Python 3.9+** - Programming language runtime
- **curl** - For downloading the installer
- **2GB+ free disk space** - For dependencies and data

### Platform-Specific Prerequisites

#### Ubuntu/Debian
```bash
sudo apt update
sudo apt install -y git python3 python3-pip curl build-essential
```

#### CentOS/RHEL/Fedora
```bash
sudo dnf install -y git python3 python3-pip curl gcc gcc-c++ make
# or for older systems: sudo yum install -y git python3 python3-pip curl gcc gcc-c++ make
```

#### macOS
```bash
# Install Xcode command line tools
xcode-select --install

# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install prerequisites
brew install git python@3.12 curl
```

#### Windows (WSL2)
```bash
# Enable WSL2 and install Ubuntu, then:
sudo apt update
sudo apt install -y git python3 python3-pip curl build-essential
```

## 🔧 Manual Installation

If you prefer manual control or the one-liner doesn't work:

### 1. Clone Repository
```bash
git clone https://github.com/sevapru/terrorblade.git
cd terrorblade
```

### 2. Install uv (Python Package Manager)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.cargo/bin:$PATH"
```

### 3. Setup Environment and Install
```bash
make install
```

### 4. Activate Environment
```bash
source .venv/bin/activate
# or use the convenience script:
./activate.sh
```

## 🎯 Quick Start After Installation

```bash
# Navigate to installation directory
cd ~/terrorblade  # or your custom INSTALL_DIR

# Activate environment
source .venv/bin/activate

# Verify installation
make test

# Run security checks
make security

# See all available commands
make help
```

## 🌐 Hosting Your Own Install Script

### For Custom Domain Hosting

1. **Copy the script to your web server:**
```bash
# Copy to your web server's public directory
cp scripts/install.sh /var/www/html/install.sh
```

2. **Update repository URL in the script:**
```bash
# Edit the script and change:
TERRORBLADE_REPO="${TERRORBLADE_REPO:-https://github.com/sevapru/terrorblade.git}"
```

3. **Set up HTTPS and proper MIME type:**
```apache
# In your Apache .htaccess or server config:
<Files "install.sh">
    Header set Content-Type "application/x-sh"
    Header set Cache-Control "no-cache, no-store, must-revalidate"
</Files>
```

### For GitHub Pages

1. **Create a `docs` branch:**
```bash
git checkout -b docs
mkdir -p docs
cp scripts/install.sh docs/install.sh
git add docs/install.sh
git commit -m "Add hosted install script"
git push origin docs
```

2. **Enable GitHub Pages** in repository settings pointing to `docs` branch

3. **Use the GitHub Pages URL:**
```bash
curl -fsSL https://sevapru.github.io/terrorblade/install.sh | bash
```

## 🔐 Security Considerations

### Verifying the Install Script

Before running any script with `curl | bash`, you can inspect it first:

```bash
# Download and inspect the script
curl -fsSL https://raw.githubusercontent.com/sevapru/terrorblade/main/scripts/install.sh > install.sh
less install.sh

# Run it manually after inspection
bash install.sh
```

### Security Features in the Installer

- ✅ **Checksum verification** of downloaded packages
- ✅ **Platform detection** and compatibility checks  
- ✅ **Non-root execution** warnings
- ✅ **Disk space** validation
- ✅ **Dependency verification** before installation
- ✅ **Graceful error handling** with rollback

## 🛠️ Customization Options

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TERRORBLADE_REPO` | `https://github.com/sevapru/terrorblade.git` | Repository URL |
| `TERRORBLADE_BRANCH` | `main` | Branch to install |
| `INSTALL_DIR` | `$HOME/terrorblade` | Installation directory |
| `PYTHON_VERSION` | `3.12` | Preferred Python version |

### Example Customizations

```bash
# Install development version to custom location
TERRORBLADE_BRANCH="dev" INSTALL_DIR="/opt/terrorblade" curl -fsSL https://raw.githubusercontent.com/sevapru/terrorblade/main/scripts/install.sh | bash

# Install from private fork
TERRORBLADE_REPO="https://github.com/mycompany/terrorblade-private.git" curl -fsSL https://raw.githubusercontent.com/sevapru/terrorblade/main/scripts/install.sh | bash
```

## 🚨 Troubleshooting

### Common Issues

#### "Python 3.9+ is required"
```bash
# Ubuntu/Debian
sudo apt install -y python3.12 python3.12-pip

# macOS
brew install python@3.12

# CentOS/RHEL
sudo dnf install -y python3.12
```

#### "uv installation failed"
```bash
# Manual uv installation
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc  # or restart terminal
```

#### "Git clone failed"
```bash
# Check if you have access to the repository
git clone https://github.com/sevapru/terrorblade.git

# For private repositories, set up SSH keys or use token
git clone https://YOUR_TOKEN@github.com/sevapru/terrorblade.git
```

#### "Make command not found"
```bash
# Ubuntu/Debian
sudo apt install -y build-essential

# macOS
xcode-select --install

# CentOS/RHEL
sudo dnf groupinstall -y "Development Tools"
```

### Getting Help

1. **Check the logs:** Installation creates detailed logs in `$INSTALL_DIR/install.log`
2. **Run diagnostics:** `make show-info` shows system information
3. **Security scan:** `make security` verifies the installation
4. **Community support:** [GitHub Issues](https://github.com/sevapru/terrorblade/issues)
5. **Documentation:** [README.md](README.md)

## 🔄 Updating Terrorblade

The installer supports updates. Run the same command to update:

```bash
# This will detect existing installation and offer to update
curl -fsSL https://raw.githubusercontent.com/sevapru/terrorblade/main/scripts/install.sh | bash
```

Or use make commands:
```bash
cd ~/terrorblade
git pull
make install
```

## ✨ What Gets Installed

- 🐍 **Python virtual environment** with all dependencies
- 🗡️ **Terrorblade package** in editable mode
- 🔧 **Development tools** (pytest, black, ruff, etc.)
- 🛡️ **Security scanners** (bandit, safety, pip-audit, semgrep)
- 📊 **Data analysis stack** (polars, torch, duckdb, etc.)
- 🚀 **Convenience scripts** (activation, shortcuts)
- 🐧 **Desktop integration** (Linux only)

---

**Happy hacking with Terrorblade! 🗡️** 