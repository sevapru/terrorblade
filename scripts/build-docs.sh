#!/bin/bash

# Цвета для вывода
GREEN='\033[1;32m'
BLUE='\033[1;34m'
YELLOW='\033[1;33m'
RED='\033[1;31m'
NC='\033[0m'

echo -e "${BLUE}📚 Building Documentation${NC}"
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

# Устанавливаем зависимости
echo -e "${BLUE}[INFO]${NC} Installing MkDocs dependencies..."
if [ -f "requirements-docs.txt" ]; then
    uv pip install -r requirements-docs.txt
else
    uv pip install mkdocs mkdocs-material
fi

# Делаем скрипт генерации API исполняемым (он находится в родительской директории)
chmod +x ../scripts/generate-api-docs.py 2>/dev/null || echo "Note: generate-api-docs.py script will be called by mkdocs-gen-files plugin"

# Собираем документацию (API docs будут сгенерированы автоматически)
echo -e "${BLUE}[INFO]${NC} Building documentation with API reference..."
cd docs-mkdocs && mkdocs build --clean

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ Documentation built successfully!${NC}"
    echo -e "${BLUE}Output directory:${NC} docs-mkdocs/site/"
    echo -e "${BLUE}To serve locally:${NC} bash scripts/serve-docs.sh"
else
    echo -e "${RED}[✗]${NC} Documentation build failed"
    exit 1
fi

