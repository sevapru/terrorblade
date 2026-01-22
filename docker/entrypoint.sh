#!/bin/bash
set -e

# Activate the virtual environment created by uv sync
source /app/.venv/bin/activate

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_header() {
    echo -e "\n${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

show_banner() {
    echo -e "${GREEN}"
    echo "  ╔════════════════════════════════════════════════════════╗"
    echo "  ║                                                        ║"
    echo "  ║   ████████╗███████╗██████╗ ██████╗  ██████╗ ██████╗    ║"
    echo "  ║      ██║   ██╔════╝██╔══██╗██╔══██╗██╔═══██╗██╔══██╗   ║"
    echo "  ║      ██║   █████╗  ██████╔╝██████╔╝██║   ██║██████╔╝   ║"
    echo "  ║      ██║   ██╔══╝  ██╔══██╗██╔══██╗██║   ██║██╔══██╗   ║"
    echo "  ║      ██║   ███████╗██║  ██║██║  ██║╚██████╔╝██║  ██║   ║"
    echo "  ║      ╚═╝   ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝   ║"
    echo "  ║                BLADE                                   ║"
    echo "  ║          Telegram Message Analyzer                     ║"
    echo "  ║                                                        ║"
    echo "  ╚════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

show_environment() {
    log_header "Environment Information"
    
    log_info "Python version: $(python --version 2>&1)"
    log_info "Working directory: $(pwd)"
    log_info "User: $(whoami)"
    log_info "Date: $(date)"
    
    echo ""
    log_info "Key packages:"
    python -c "
import sys
packages = ['polars', 'duckdb', 'fastmcp', 'sentence_transformers', 'openai']
for pkg in packages:
    try:
        mod = __import__(pkg.replace('-', '_'))
        version = getattr(mod, '__version__', 'unknown')
        print(f'  - {pkg}: {version}')
    except ImportError:
        print(f'  - {pkg}: NOT INSTALLED')
"
    
    echo ""
    log_info "Environment variables:"
    echo "  - DB_PATH: ${DB_PATH:-not set}"
    echo "  - DUCKDB_REMOTE_URL: ${DUCKDB_REMOTE_URL:-not set}"
    echo "  - MOTHERDUCK_TOKEN: ${MOTHERDUCK_TOKEN:+[SET]}"
    echo "  - AWS_ACCESS_KEY_ID: ${AWS_ACCESS_KEY_ID:+[SET]}"
    echo "  - OPENAI_API_KEY: ${OPENAI_API_KEY:+[SET]}"
    echo "  - PHONE: ${PHONE:-not set}"
}

# Check if using remote database
is_remote_db() {
    local path="$1"
    case "$path" in
        md:*|http://*|https://*|s3://*|gcs://*|az://*)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

# Check if database file exists and is valid
check_data() {
    log_header "Database Validation"
    
    # Check for remote database URL first
    if [ -n "$DUCKDB_REMOTE_URL" ]; then
        log_info "Using remote DuckDB connection: $DUCKDB_REMOTE_URL"
        
        if is_remote_db "$DUCKDB_REMOTE_URL"; then
            log_success "Remote database URL detected"
            
            # Validate remote connection
            python << PYTHON_SCRIPT
import sys
import os

remote_url = os.environ.get('DUCKDB_REMOTE_URL', '')
motherduck_token = os.environ.get('MOTHERDUCK_TOKEN', '')

try:
    import duckdb
    
    # For MotherDuck, we need to set the token
    if remote_url.startswith('md:'):
        if motherduck_token:
            conn = duckdb.connect(remote_url, config={'motherduck_token': motherduck_token})
        else:
            conn = duckdb.connect(remote_url)
    else:
        # For HTTP/S3 connections, load httpfs extension
        conn = duckdb.connect(':memory:')
        conn.execute("INSTALL httpfs; LOAD httpfs;")
        # Try to attach the remote database
        conn.execute(f"ATTACH '{remote_url}' AS remote_db (READ_ONLY);")
    
    print("  ✓ Remote database connection successful")
    conn.close()
    sys.exit(0)
except Exception as e:
    print(f"  ⚠ Remote connection test: {e}")
    print("  → Will attempt connection at runtime")
    sys.exit(0)  # Don't fail - connection might work at runtime
PYTHON_SCRIPT
            return 0
        fi
    fi
    
    if [ -z "$DB_PATH" ]; then
        DB_PATH="/data/telegram_data.db"
        log_warn "DB_PATH not set, using default: $DB_PATH"
    fi
    
    # Check if DB_PATH is a remote URL
    if is_remote_db "$DB_PATH"; then
        log_info "Using remote database: $DB_PATH"
        log_success "Remote database URL detected in DB_PATH"
        return 0
    fi
    
    log_info "Checking local database at: $DB_PATH"
    
    if [ ! -f "$DB_PATH" ]; then
        log_error "Database file not found!"
        log_info "Please mount your database file:"
        echo "    docker run -v /path/to/telegram_data.db:/data/telegram_data.db ..."
        echo ""
        log_info "Or use a remote database by setting DUCKDB_REMOTE_URL:"
        echo "    - MotherDuck: DUCKDB_REMOTE_URL=md:my_database"
        echo "    - S3: DUCKDB_REMOTE_URL=s3://bucket/path/database.db"
        echo "    - HTTP: DUCKDB_REMOTE_URL=https://example.com/database.db"
        return 1
    fi
    
    # Get file info
    FILE_SIZE=$(du -h "$DB_PATH" | cut -f1)
    FILE_PERMS=$(stat -c "%a" "$DB_PATH" 2>/dev/null || stat -f "%Lp" "$DB_PATH" 2>/dev/null || echo "unknown")
    log_success "Database file found"
    echo "  - Size: $FILE_SIZE"
    echo "  - Permissions: $FILE_PERMS"
    
    # Validate DuckDB and get detailed info
    log_info "Validating DuckDB structure..."
    
    python << PYTHON_SCRIPT
import sys
import duckdb
import os

db_path = os.environ.get('DB_PATH', '/data/telegram_data.db')

try:
    conn = duckdb.connect(db_path, read_only=True)
    
    # Get tables
    tables = conn.execute("SHOW TABLES").fetchall()
    table_names = [t[0] for t in tables]
    
    print(f"  ✓ Valid DuckDB database")
    print(f"  - Tables: {len(table_names)}")
    
    # Categorize tables
    message_tables = [t for t in table_names if t.startswith('messages_')]
    chat_tables = [t for t in table_names if t.startswith('chat_names_')]
    cluster_tables = [t for t in table_names if t.startswith('message_clusters_')]
    embedding_tables = [t for t in table_names if t.startswith('message_embeddings_')]
    
    if message_tables:
        print(f"  - Message tables: {len(message_tables)}")
        for mt in message_tables[:3]:  # Show first 3
            try:
                count = conn.execute(f"SELECT COUNT(*) FROM {mt}").fetchone()[0]
                print(f"      • {mt}: {count:,} messages")
            except:
                print(f"      • {mt}: (count unavailable)")
    
    if chat_tables:
        print(f"  - Chat tables: {len(chat_tables)}")
    
    if cluster_tables:
        print(f"  - Cluster tables: {len(cluster_tables)}")
        
    if embedding_tables:
        print(f"  - Embedding tables: {len(embedding_tables)}")
        for et in embedding_tables[:2]:
            try:
                count = conn.execute(f"SELECT COUNT(*) FROM {et}").fetchone()[0]
                print(f"      • {et}: {count:,} embeddings")
            except:
                pass
    
    # Check for VSS extension
    try:
        conn.execute("SELECT * FROM duckdb_extensions() WHERE extension_name = 'vss'")
        print(f"  - VSS extension: available")
    except:
        print(f"  - VSS extension: not loaded (will load on demand)")
    
    conn.close()
    sys.exit(0)
    
except Exception as e:
    print(f"  ✗ Database validation failed: {e}")
    sys.exit(1)
PYTHON_SCRIPT

    return $?
}

# Run health check
health_check() {
    log_header "Health Check"
    
    if check_data; then
        log_success "All checks passed!"
        return 0
    else
        log_error "Health check failed"
        return 1
    fi
}

# Show help
show_help() {
    log_header "Help"
    echo ""
    echo "Usage: docker run [options] terrorblade <command>"
    echo ""
    echo "Commands:"
    echo "  mcp       Start the MCP server (stdio mode)"
    echo "  mcp-sse   Start the MCP server (SSE/HTTP mode on port 8787)"
    echo "  cli       Start the CLI analysis tool"
    echo "  check     Check data availability and exit"
    echo "  shell     Start an interactive shell"
    echo "  help      Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  DB_PATH           Path to DuckDB database (default: /data/telegram_data.db)"
    echo "  OPENAI_API_KEY    OpenAI API key for AI-powered analysis"
    echo "  PHONE             Phone number for CLI (e.g., +1234567890)"
    echo ""
    echo "Examples:"
    echo "  # Run MCP server with mounted database"
    echo "  docker run -v /path/to/db:/data/telegram_data.db terrorblade mcp"
    echo ""
    echo "  # Run CLI analysis tool"
    echo "  docker run -it -v /path/to/db:/data/telegram_data.db -e PHONE=+123 terrorblade cli"
}

# Test MCP server functionality
test_mcp() {
    log_header "MCP Server Test"
    
    log_info "Testing MCP server initialization..."
    
    python << 'PYTHON_SCRIPT'
import sys
try:
    from terrorblade.mcp.server import mcp
    print("  ✓ MCP server module loaded successfully")
    print(f"  - Server name: {mcp.name}")
    
    # List available tools
    tools = list(mcp._tool_manager._tools.keys()) if hasattr(mcp, '_tool_manager') else []
    if tools:
        print(f"  - Available tools: {len(tools)}")
        for tool in tools:
            print(f"      • {tool}")
    
    sys.exit(0)
except Exception as e:
    print(f"  ✗ MCP server test failed: {e}")
    sys.exit(1)
PYTHON_SCRIPT

    return $?
}

# Main entrypoint logic
main() {
    show_banner
    show_environment
    
    case "${1:-mcp}" in
        mcp)
            log_header "Starting MCP Server"
            
            if ! check_data; then
                log_error "Cannot start MCP server without valid database"
                exit 1
            fi
            
            if ! test_mcp; then
                log_error "MCP server test failed"
                exit 1
            fi
            
            log_success "All checks passed, launching MCP server..."
            echo ""
            log_info "MCP server is running on stdio"
            log_info "Waiting for client connections..."
            echo ""
            
            exec python -m terrorblade.mcp.server
            ;;
        mcp-sse)
            log_header "Starting MCP Server (SSE mode)"
            
            if ! check_data; then
                log_error "Cannot start MCP server without valid database"
                exit 1
            fi
            
            if ! test_mcp; then
                log_error "MCP server test failed"
                exit 1
            fi
            
            SSE_PORT="${MCP_SSE_PORT:-8787}"
            
            log_success "All checks passed, launching MCP server with SSE transport..."
            echo ""
            log_info "MCP server is running on http://0.0.0.0:${SSE_PORT}/sse"
            log_info "Connect using: http://localhost:${SSE_PORT}/sse"
            echo ""
            
            # Use FastMCP native SSE transport instead of mcp-proxy for better stability
            exec python -c "
from terrorblade.mcp.server import mcp
mcp.run(transport='sse', host='0.0.0.0', port=$SSE_PORT)
"
            ;;
        cli)
            log_header "Starting CLI Tool"
            
            if ! check_data; then
                log_error "Cannot start CLI without valid database"
                exit 1
            fi
            
            PHONE_ARG=""
            if [ -n "$PHONE" ]; then
                PHONE_ARG="--phone $PHONE"
                log_info "Using phone: $PHONE"
            fi
            
            log_success "Launching CLI..."
            echo ""
            
            exec python -m terrorblade.examples.analyze_dialogues $PHONE_ARG "${@:2}"
            ;;
        check)
            health_check
            exit $?
            ;;
        shell)
            log_header "Interactive Shell"
            log_info "Starting bash shell..."
            exec /bin/bash
            ;;
        help|--help|-h)
            show_help
            exit 0
            ;;
        *)
            log_error "Unknown command: $1"
            show_help
            exit 1
            ;;
    esac
}

# Run main
main "$@"
