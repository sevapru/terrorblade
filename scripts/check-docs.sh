#!/bin/bash

# Цвета для вывода
GREEN='\033[1;32m'
BLUE='\033[1;34m'
YELLOW='\033[1;33m'
RED='\033[1;31m'
NC='\033[0m'

echo -e "${BLUE}🔍 Checking Documentation Status${NC}"
echo ""

# Проверяем директорию документации
if [ -d "docs-mkdocs" ]; then
    echo -e "${GREEN}[✓]${NC} Documentation directory exists"
    
    if [ -f "docs-mkdocs/mkdocs.yml" ]; then
        echo -e "${GREEN}[✓]${NC} mkdocs.yml found"
    else
        echo -e "${YELLOW}[⚠]${NC} mkdocs.yml not found"
    fi
    
    if [ -d "docs-mkdocs/site" ]; then
        echo -e "${GREEN}[✓]${NC} Build directory exists"
        site_files=$(find docs-mkdocs/site -name "*.html" | wc -l)
        echo -e "${BLUE}[INFO]${NC} Found $site_files HTML files in build"
    else
        echo -e "${BLUE}[INFO]${NC} No build directory found - run 'make docs-build'"
    fi
    
    if [ -f "docs-mkdocs/docs/index.md" ]; then
        echo -e "${GREEN}[✓]${NC} Main index.md found"
    else
        echo -e "${YELLOW}[⚠]${NC} Main index.md not found"
    fi
else
    echo -e "${YELLOW}[⚠]${NC} Documentation directory not found - run 'make docs-setup'"
fi

# Проверяем requirements-docs.txt
if [ -f "requirements-docs.txt" ]; then
    echo -e "${GREEN}[✓]${NC} requirements-docs.txt found"
else
    echo -e "${YELLOW}[⚠]${NC} requirements-docs.txt not found"
fi

# Проверяем MkDocs установку
if command -v mkdocs >/dev/null; then
    echo -e "${GREEN}[✓]${NC} MkDocs is installed"
    mkdocs_version=$(mkdocs --version | cut -d' ' -f3)
    echo -e "${BLUE}[INFO]${NC} MkDocs version: $mkdocs_version"
else
    echo -e "${YELLOW}[⚠]${NC} MkDocs not found"
fi

echo ""
echo -e "${BLUE}Available commands:${NC}"
echo -e "  ${GREEN}make docs-setup${NC}  - Set up documentation environment"
echo -e "  ${GREEN}make docs-build${NC}  - Build documentation locally"
echo -e "  ${GREEN}make docs-serve${NC}  - Serve documentation locally"
echo -e "  ${GREEN}make docs-clean${NC}  - Clean build artifacts"

