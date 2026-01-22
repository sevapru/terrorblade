# Terrorblade Docker Deployment

Docker configuration for easy deployment of Terrorblade MCP server.

## Prerequisites

- Docker and Docker Compose
- A DuckDB database file created by Terrorblade (`telegram_data.db`) OR a remote DuckDB connection

## Quick Start

```bash
cd docker

# Build the image
docker compose build

# Run data check to verify database
docker compose --profile tools run --rm check

# Run MCP server
docker compose up -d mcp
```

## Services

| Service | Description | Profile |
|---------|-------------|---------|
| `mcp` | MCP server with local DB (stdio) | default |
| `mcp-remote` | MCP server with remote DB only | `remote` |
| `cli` | Interactive CLI tool | `cli` |
| `check` | Data validation | `tools` |

## Configuration

### Local Database

Set your database path via environment variable:

```bash
# Option 1: Export variable
export DB_PATH=/path/to/telegram_data.db
docker compose up -d mcp

# Option 2: Inline
DB_PATH=/path/to/telegram_data.db docker compose up -d mcp

# Option 3: Create .env file
echo "DB_PATH=/path/to/telegram_data.db" > .env
docker compose up -d mcp
```

### Remote Database (MotherDuck, S3, HTTP)

Terrorblade supports remote DuckDB connections for distributed deployments:

```bash
# MotherDuck
export DUCKDB_REMOTE_URL="md:my_database"
export MOTHERDUCK_TOKEN="your_token_here"
docker compose --profile remote up -d mcp-remote

# S3
export DUCKDB_REMOTE_URL="s3://bucket/path/telegram_data.db"
export AWS_ACCESS_KEY_ID="your_key"
export AWS_SECRET_ACCESS_KEY="your_secret"
docker compose --profile remote up -d mcp-remote

# HTTP/HTTPS
export DUCKDB_REMOTE_URL="https://example.com/telegram_data.db"
docker compose --profile remote up -d mcp-remote
```

### Separate Deployment (MCP + Remote DuckDB)

You can deploy the MCP server and DuckDB database separately:

1. **Deploy DuckDB** - Host your database on MotherDuck, S3, or any HTTP server
2. **Deploy MCP** - Run only the MCP service pointing to remote DB

```bash
# On DB server: Upload database to S3/MotherDuck/HTTP

# On MCP server: Start service with remote connection
DUCKDB_REMOTE_URL="md:my_database" \
MOTHERDUCK_TOKEN="token" \
docker compose --profile remote up -d mcp-remote
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DB_PATH` | Path to local DuckDB database | `./data/telegram_data.db` |
| `DUCKDB_REMOTE_URL` | Remote database URL (MotherDuck, S3, HTTP) | - |
| `MOTHERDUCK_TOKEN` | MotherDuck authentication token | - |
| `AWS_ACCESS_KEY_ID` | AWS credentials for S3 | - |
| `AWS_SECRET_ACCESS_KEY` | AWS credentials for S3 | - |
| `AWS_REGION` | AWS region for S3 | `us-east-1` |
| `PHONE` | Phone number for CLI | `+79992004210` |
| `OPENAI_API_KEY` | OpenAI API key for AI features | - |

## Usage Examples

```bash
# Check database validity
docker compose --profile tools run --rm check

# Run CLI interactively
docker compose --profile cli run --rm cli

# View MCP server logs
docker compose logs -f mcp

# Stop all services
docker compose down
```

## CI/CD Deployment

A GitHub Actions workflow is provided for VPS deployment:

```yaml
# .github/workflows/deploy-mcp.yml
# Supports:
# - Separate deployment of MCP and DuckDB
# - MotherDuck, S3, and HTTP remote connections
# - Automatic container management
```

Required secrets:
- `SSH_PRIVATE_KEY` - SSH key for VPS access
- `VPS_HOST`, `VPS_PORT`, `VPS_USER`, `VPS_PATH` - VPS connection details
- `VPS_DB_PATH` - Path to database on VPS
- `OPENAI_API_KEY` - OpenAI API key
- `DUCKDB_REMOTE_URL` - (Optional) Remote database URL
- `MOTHERDUCK_TOKEN` - (Optional) MotherDuck token

## Using with Cursor / Claude Desktop

### Local Database

```json
{
  "mcpServers": {
    "terrorblade": {
      "command": "docker",
      "args": [
        "run", "--rm", "-i",
        "-v", "/path/to/telegram_data.db:/data/telegram_data.db:ro",
        "terrorblade-mcp",
        "mcp"
      ]
    }
  }
}
```

### Remote Database (MotherDuck)

```json
{
  "mcpServers": {
    "terrorblade": {
      "command": "docker",
      "args": [
        "run", "--rm", "-i",
        "-e", "DUCKDB_REMOTE_URL=md:my_database",
        "-e", "MOTHERDUCK_TOKEN=your_token",
        "terrorblade-mcp",
        "mcp"
      ]
    }
  }
}
```
