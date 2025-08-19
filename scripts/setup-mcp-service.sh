#!/bin/bash

# Setup script for Terrorblade MCP service and Claude integration

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE_FILE="$PROJECT_ROOT/terrorblade/mcp/terrorblade-mcp.service"
SYSTEM_SERVICE_DIR="/etc/systemd/system"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🤖 Setting up Terrorblade MCP service and Claude integration...${NC}"
echo ""

# Check if service file exists
if [ ! -f "$SERVICE_FILE" ]; then
    echo -e "${RED}❌ Service file not found: $SERVICE_FILE${NC}"
    exit 1
fi

# Check if virtual environment exists
if [ ! -f "$PROJECT_ROOT/.venv/bin/terrorblade-mcp" ]; then
    echo -e "${RED}❌ Virtual environment or terrorblade-mcp not found at: $PROJECT_ROOT/.venv/bin/terrorblade-mcp${NC}"
    echo -e "${YELLOW}💡 Please run 'make install' first${NC}"
    exit 1
fi

echo -e "${YELLOW}📋 Setting up systemd service...${NC}"

# Copy service file to systemd directory
sudo cp "$SERVICE_FILE" "$SYSTEM_SERVICE_DIR/terrorblade-mcp.service"

# Reload systemd daemon
sudo systemctl daemon-reload

# Enable the service (start on boot)
sudo systemctl enable terrorblade-mcp.service

# Start the service
sudo systemctl start terrorblade-mcp.service

# Show status
echo ""
echo -e "${GREEN}✅ Service status:${NC}"
sudo systemctl status terrorblade-mcp.service --no-pager -l

echo ""
echo -e "${YELLOW}🔧 Claude Integration Setup:${NC}"
echo "Run this command to add Terrorblade MCP to Claude:"
echo ""
echo -e "${GREEN}claude mcp add-json terrorblade '{
  \"command\": \"$PROJECT_ROOT/.venv/bin/terrorblade-mcp\",
  \"args\": [],
  \"env\": {
    \"DUCKDB_PATH\": \"$PROJECT_ROOT/telegram_data.db\"
  }
}'${NC}"

echo ""
echo -e "${YELLOW}⚠️  Note: Update the DUCKDB_PATH to point to your actual database file!${NC}"

echo ""
echo -e "${BLUE}📚 Service commands:${NC}"
echo "  Start:   sudo systemctl start terrorblade-mcp"
echo "  Stop:    sudo systemctl stop terrorblade-mcp"  
echo "  Status:  sudo systemctl status terrorblade-mcp"
echo "  Logs:    sudo journalctl -u terrorblade-mcp -f"

echo ""
echo -e "${BLUE}📖 Documentation:${NC}"
echo "  MCP docs: $PROJECT_ROOT/terrorblade/mcp/README.md"

echo ""
echo -e "${GREEN}🎉 MCP server setup completed!${NC}"