# Terrorblade MCP Server

Exposes semantic vector search and cluster tools over the Model Context Protocol (MCP) for Telegram message analysis stored in DuckDB.

## Prerequisites

- Python 3.12+
- A DuckDB database produced by Terrorblade (e.g., `telegram_data.db`)
- Optional: `uv` to run via `uv run`/`uvx`

## Configuration

- Database path resolution order:
  1. Explicit `db_path` parameter (unless set to `"auto"` or `"default"`)
  2. Environment variable `DUCKDB_PATH`
  3. Default path in the parent folder of the project: `../telegram_data.db`

- Environment variable:

```bash
env | grep DUCKDB_PATH || true
export DUCKDB_PATH=/absolute/path/to/telegram_data.db
```

- Logging:
  - File: `logs/mcp_server.log` (rotating, 10 MB x 5)
  - Console: INFO level

## Run locally (stdio)

If you installed the project per the main README, a console script is available:

```bash
# Using the installed entrypoint
terrorblade-mcp

# Or directly via Python
python -m terrorblade.mcp.server

# Or with uv
uv run terrorblade-mcp
# Or
uvx terrorblade-mcp
```

The server will log to `logs/mcp_server.log` and stdout.

## Tools

- `vector_search(phone, query, top_k=10, chat_id=None, similarity_threshold=0.0, include_cluster_messages=True, db_path="auto") -> {"results": [...], "stats": {...}}`
- `cluster_search(phone, query, top_k=50, max_clusters=10, similarity_threshold=0.0, db_path="auto") -> [ ...clusters ]`
- `get_cluster(phone, chat_id, group_id, db_path="auto") -> [ ...messages ]`
- `random_large_cluster(phone, min_size=10, db_path="auto") -> [ ...messages ]`

Notes:
- Pass `db_path` explicitly or keep `db_path="auto"` to use `DUCKDB_PATH` or the default `../telegram_data.db`.
- `phone` can be with or without `+` and is used to select partitioned tables.

## Using with Cursor / Claude Desktop (stdio)

Add a server configuration pointing to your Python/venv and the entrypoint:

```json
{
  "mcpServers": {
    "terrorblade": {
      "command": "/absolute/path/to/python",
      "args": ["-m", "terrorblade.mcp.server"],
      "env": {
        "DUCKDB_PATH": "/absolute/path/to/telegram_data.db"
      }
    }
  }
}
```

## Expose locally over SSE and proxy to VPS

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

---

# MCP server for Telegram personal chats data analysis

Built with DuckDB and `vss` extension.

# Alternatives

## MotherDuck MCP
An MCP server implementation that interacts with DuckDB and MotherDuck databases, providing SQL analytics capabilities to AI Assistants and IDEs with inclusion of SQL analytics and data sharing. Main set of tools is `query` which is contextually the same as terrorblade idea, but lacking the `VectorSearch` which is being added to the DuckDB via `vss` extension.
[MotherDuck MCP](https://github.com/motherduckdb/mcp-server-motherduck)

## Telegram MCP Server by DLHellMe

[Telegram MCP Server by DLHellMe](https://github.com/DLHellMe/telegram-mcp-server)
Specialized for Telegram Data Extraction from public chats. for data scraping
This server provides both web scraping and direct API access to Telegram content.

## Telegram MCP by chigwell

[Telegram MCP by chigwell](https://github.com/chigwell/telegram-mcp)
Every major Telegram/Telethon feature is available as a tool which makes it a good alternative to the terrorblade for general purpose and hopefully more stable and reliable (for API calls)

And yet, there is no `VectorSearch` tool available and it's not clear if it's possible to extract the whole chat history into database with it.

## Telegram MCP by chaindead

[Telegram MCP by chaindead](https://github.com/chaindead/telegram-mcp)

Capabilities:

- Get current account information (tool: tg_me)
- List dialogs with optional unread filter (tool: tg_dialogs) ("Summarize all my unread Telegram messages")
- Mark dialog as read (tool: tg_read)
- Retrieve messages from specific dialog (tool: tg_dialog) "Check non-critical unread messages and give me a brief overview"
- Send draft messages to any dialog (tool: tg_send)

Looks cool, but it's not clear if it possible to work with whole chat history.

# Total differences

- It's focused on public chats analysis and search around public information avalilable for the user.
- It's not focused on personal chats analysis and search around personal information avalilable for the user.
- It's not focused on the vector search and clustering of the personal chats data.
- It's not focused on the personal chats data analysis and search around personal information avalilable for the user.

