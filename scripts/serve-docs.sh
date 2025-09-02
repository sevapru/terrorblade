#!/bin/bash

# Цвета для вывода
GREEN='\033[1;32m'
BLUE='\033[1;34m'
YELLOW='\033[1;33m'
RED='\033[1;31m'
NC='\033[0m'

echo -e "${BLUE}🌐 Serving Documentation Locally${NC}"
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

# Проверяем что директория документации существует
if [ ! -d "docs-mkdocs" ]; then
    echo -e "${RED}[✗]${NC} Documentation directory not found"
    echo -e "${BLUE}[INFO]${NC} Run: bash scripts/setup-docs.sh"
    exit 1
fi

# Устанавливаем зависимости если нужно
echo -e "${BLUE}[INFO]${NC} Checking MkDocs installation..."
if ! command -v mkdocs >/dev/null; then
    echo -e "${BLUE}[INFO]${NC} Installing MkDocs..."
    if [ -f "requirements-docs.txt" ]; then
        uv pip install -r requirements-docs.txt
    else
        uv pip install mkdocs mkdocs-material
    fi
fi

echo -e "${BLUE}[INFO]${NC} Starting MkDocs server..."
echo -e "${GREEN}Documentation will be available at: http://127.0.0.1:8000${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop the server${NC}"
echo ""

cd docs-mkdocs && mkdocs serve --dev-addr=127.0.0.1:8000

