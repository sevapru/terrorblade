# Terrorblade MCP Server

Exposes semantic vector search and cluster tools over the Model Context Protocol (MCP) for Telegram message analysis stored in DuckDB.

## Available Tools

- **`vector_search`** — Semantic search over messages with optional cluster snippets
  - Parameters: `phone`, `query`, `top_k=10`, `chat_id=None`, `similarity_threshold=0.0`, `include_cluster_messages=True`, `db_path="auto"`

- **`cluster_search`** — Find conversation clusters related to your query
  - Parameters: `phone`, `query`, `top_k=50`, `max_clusters=10`, `similarity_threshold=0.0`, `db_path="auto"`

- **`get_cluster`** — Retrieve all messages from a specific conversation cluster
  - Parameters: `phone`, `chat_id`, `group_id`, `db_path="auto"`

- **`random_large_cluster`** — Return a random large conversation cluster for exploration
  - Parameters: `phone`, `min_size=10`, `db_path="auto"`


## Quick Setup

### Add to Claude

```bash
claude mcp add-json terrorblade '{
  "command": "~/terrorblade/.venv/bin/terrorblade-mcp",
  "args": [],
  "env": {
    "DUCKDB_PATH": "~/terrorblade/telegram_data.db"
  }
}'
```

### Alternative Commands

```bash
# Using Python module directly
claude mcp add-json terrorblade '{
  "command": "~/terrorblade/.venv/bin/python",
  "args": ["-m", "terrorblade.mcp.server"],
  "env": {
    "DUCKDB_PATH": "~/terrorblade/telegram_data.db"
  }
}'  

# Using uv (if available)
claude mcp add-json terrorblade '{
  "command": "uv",
  "args": ["run", "terrorblade-mcp"],
  "env": {
    "DUCKDB_PATH": "~/terrorblade/telegram_data.db"
  }
}'
```

### Run Locally (stdio)

```bash
# Using the installed entrypoint
terrorblade-mcp

# Or directly via Python
python -m terrorblade.mcp.server

# Or with uv
uv run terrorblade-mcp
```

## Configuration

### Database Path Resolution

Priority order:

1. Explicit `db_path` parameter (unless set to `"auto"` or `"default"`)
2. Environment variable `DUCKDB_PATH`
3. Default path in the parent folder of the project: `../telegram_data.db`

```bash
export DUCKDB_PATH=/absolute/path/to/telegram_data.db
```

**Logging:** `logs/mcp_server.log` (rotating, 10MB × 5)


Notes:

- `phone` can be with or without `+` and is used to select partitioned tables.

## Optional: Expose over SSE

<details>
<summary>Advanced Setup: HTTP/SSE Server</summary>

Run the stdio server and expose it over SSE using a small proxy:

```bash
# Install the proxy (Rust)
cargo install rmcp-proxy

# Start SSE proxy on localhost:8787, wrapping the stdio MCP server
RUST_LOG=error mcp-proxy --sse-port 8787 -- python -m terrorblade.mcp.server
```

Create a reverse SSH tunnel from the VPS to your local machine:

```bash
# On your local machine
ssh -N -R 127.0.0.1:8788:localhost:8787 vps_user@vps_host

# Optional: keepalive
# autossh -M 0 -N -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -R 127.0.0.1:8788:localhost:8787 vps_user@vps_host
```

On the VPS, you can proxy `http://127.0.0.1:8788` via Nginx (example):

```nginx
server {
  listen 80;
  server_name your-domain.example;

  location /mcp/ {
    proxy_pass http://127.0.0.1:8788/;
    proxy_set_header Host $host;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
  }
}
```

Later, add OAuth in front of this location (e.g., with `oauth2-proxy`) and have clients send an `Authorization: Bearer ...` header. The `mcp-proxy` supports adding headers when acting as an SSE client if you later reverse the direction.

</details>

## Prerequisites

- Python 3.12+
- DuckDB database from Terrorblade processing
- Optional: `uv` for package management

## Alternatives

### MotherDuck MCP
An MCP server implementation that interacts with DuckDB and MotherDuck databases, providing SQL analytics capabilities to AI Assistants and IDEs with inclusion of SQL analytics and data sharing. Main set of tools is `query` which is contextually the same as terrorblade idea, but lacking the `VectorSearch` which is being added to the DuckDB via `vss` extension.
[MotherDuck MCP](https://github.com/motherduckdb/mcp-server-motherduck)

### Telegram MCP Server by DLHellMe

[Telegram MCP Server by DLHellMe](https://github.com/DLHellMe/telegram-mcp-server)
Specialized for Telegram Data Extraction from public chats. for data scraping
This server provides both web scraping and direct API access to Telegram content.

### Telegram MCP by chigwell

[Telegram MCP by chigwell](https://github.com/chigwell/telegram-mcp)
Every major Telegram/Telethon feature is available as a tool which makes it a good alternative to the terrorblade for general purpose and hopefully more stable and reliable (for API calls)

And yet, there is no `VectorSearch` tool available and it's not clear if it's possible to extract the whole chat history into database with it.

### Telegram MCP by chaindead

[Telegram MCP by chaindead](https://github.com/chaindead/telegram-mcp)

Capabilities:

- Get current account information (tool: tg_me)
- List dialogs with optional unread filter (tool: tg_dialogs) ("Summarize all my unread Telegram messages")
- Mark dialog as read (tool: tg_read)
- Retrieve messages from specific dialog (tool: tg_dialog) "Check non-critical unread messages and give me a brief overview"
- Send draft messages to any dialog (tool: tg_send)

Looks cool, but it's not clear if it possible to work with whole chat history.

## Terrorblade MCP Benefits (AI written abstract)

**What makes Terrorblade MCP unique:**

- **🔍 Semantic Vector Search:** Uses embeddings for meaning-based search, not just text matching
- **📊 Conversation Clustering:** Groups related messages into meaningful conversation threads  
- **🗄️ Personal Chat Focus:** Designed for comprehensive personal message history analysis
- **⚡ HNSW Indexing:** Fast similarity search using DuckDB's Vector Similarity Search (VSS) extension
- **🔗 Cluster Context:** Shows conversation snippets around found messages for better understanding
- **📈 Analytics-Ready:** Structured data storage optimized for personal communication analysis
- **🛡️ Privacy-First:** Works with locally stored data, no external API calls for search operations

Unlike alternatives that focus on real-time API operations or public data scraping, Terrorblade MCP specializes in deep analysis of your own message history with advanced semantic understanding.

