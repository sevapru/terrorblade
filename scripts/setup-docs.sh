#!/bin/bash

# Цвета для вывода
GREEN='\033[1;32m'
BLUE='\033[1;34m'
YELLOW='\033[1;33m'
RED='\033[1;31m'
NC='\033[0m'

echo -e "${BLUE}🔧 Setting Up Documentation Environment${NC}"
echo ""

# Проверяем виртуальное окружение
if [ -n "$VIRTUAL_ENV" ]; then
    echo -e "${GREEN}[✓]${NC} Using active virtual environment: $VIRTUAL_ENV"
elif [ -d ".venv" ]; then
    echo -e "${BLUE}[INFO]${NC} Activating .venv environment"
    source .venv/bin/activate
    echo -e "${GREEN}[✓]${NC} Virtual environment activated"
else
    echo -e "${RED}[✗]${NC} No virtual environment found"
    exit 1
fi

# Устанавливаем MkDocs и зависимости для API документации
echo -e "${BLUE}[INFO]${NC} Installing MkDocs and API documentation dependencies..."
if [ -f "../requirements-docs.txt" ]; then
    uv pip install -r ../requirements-docs.txt
else
    uv pip install mkdocs mkdocs-material mkdocstrings mkdocstrings-python mkdocs-gen-files
fi

# Создаем структуру директорий
echo -e "${BLUE}[INFO]${NC} Creating documentation structure..."
mkdir -p docs-mkdocs/{getting-started,architecture,api,examples,assets/{images,css,js}}

# Создаем mkdocs.yml
echo -e "${BLUE}[INFO]${NC} Creating mkdocs.yml..."
if [ ! -f "docs-mkdocs/mkdocs.yml" ]; then
    cat > docs-mkdocs/mkdocs.yml << 'EOF'
site_name: Terrorblade Documentation
site_description: A unified data extraction and parsing platform for messaging platforms
site_url: https://docs.yourdomain.com
repo_url: https://github.com/sevapru/terrorblade
repo_name: sevapru/terrorblade

theme:
  name: material
  language: en
  features:
    - navigation.tabs
    - navigation.sections
    - navigation.expand
    - search.highlight
    - search.share
    - content.code.copy
    - content.code.annotate
    - content.tabs.link
    - content.tooltips
    - content.code.select
    - navigation.footer
    - navigation.top
    - navigation.tracking
    - header.autohide
    - toc.follow
    - toc.integrate
    - navigation.prune

plugins:
  - search

markdown_extensions:
  - admonition
  - codehilite
  - footnotes
  - meta
  - pymdownx.arithmatex
  - pymdownx.betterem
  - pymdownx.caret
  - pymdownx.details
  - pymdownx.emoji:
      emoji_generator: !!python/name:material.extensions.emoji.to_svg
  - pymdownx.highlight
  - pymdownx.inlinehilite
  - pymdownx.keys
  - pymdownx.magiclink
  - pymdownx.mark
  - pymdownx.smartsymbols
  - pymdownx.snippets
  - pymdownx.superfences
  - pymdownx.tabbed
  - pymdownx.tasklist
  - pymdownx.tilde
  - toc:
      permalink: true
  - attr_list
  - def_list
  - tables
  - abbr
  - md_in_html

nav:
  - Home: index.md
  - Getting Started:
    - Installation: getting-started/installation.md
    - Quick Start: getting-started/quick-start.md
    - Examples: getting-started/examples.md
  - Architecture:
    - Database Schema: architecture/database-schema.md
    - Clustering: architecture/clustering.md
    - Vector Search: architecture/vector-search.md
  - API:
    - Database: api/database.md
    - Preprocessing: api/preprocessing.md
    - Vector Store: api/vector-store.md
  - Examples:
    - Telegram Export: examples/telegram-export.md
    - Vector Search: examples/vector-search.md
    - TUI Usage: examples/tui-usage.md
EOF
    echo -e "${GREEN}[✓]${NC} mkdocs.yml created successfully"
else
    echo -e "${BLUE}[INFO]${NC} mkdocs.yml already exists"
fi

# Создаем index.md из README.md
echo -e "${BLUE}[INFO]${NC} Creating index.md from README.md..."
if [ ! -f "docs-mkdocs/index.md" ]; then
    cp README.md docs-mkdocs/index.md
    echo -e "${GREEN}[✓]${NC} index.md created from README.md"
else
    echo -e "${BLUE}[INFO]${NC} index.md already exists"
fi

# Создаем requirements-docs.txt
echo -e "${BLUE}[INFO]${NC} Creating requirements-docs.txt..."
if [ ! -f "requirements-docs.txt" ]; then
    cat > requirements-docs.txt << 'EOF'
# Documentation dependencies
mkdocs>=1.5.0
mkdocs-material>=9.0.0
mkdocs-git-revision-date-localized-plugin>=1.2.0
mkdocs-awesome-pages-plugin>=2.9.0
mkdocs-minify-plugin>=0.7.0
mkdocs-redirects>=1.2.0
EOF
    echo -e "${GREEN}[✓]${NC} requirements-docs.txt created"
else
    echo -e "${BLUE}[INFO]${NC} requirements-docs.txt already exists"
fi

echo ""
echo -e "${GREEN}✅ Documentation environment setup completed!${NC}"
echo -e "${BLUE}Next steps:${NC}"
echo -e "  1. ${YELLOW}bash scripts/build-docs.sh${NC}    # Build documentation"
echo -e "  2. ${YELLOW}bash scripts/serve-docs.sh${NC}    # Preview locally"
echo -e "  3. Edit content in docs-mkdocs/ directory"

