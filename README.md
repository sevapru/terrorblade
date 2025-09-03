<div align="center">
  <a href="https://github.com/sevapru/terrorblade">
    <img alt="terrorblade" width="120" src="docs/images/terrorblade_logo.png">
  </a>
</div>

# Terrorblade

A unified data extraction and parsing platform for messaging platforms, featuring Telegram message processing, data standardization, and analytics preparation capabilities.

## Linux, Windows, macOS

```bash
curl -fsSL https://raw.githubusercontent.com/sevapru/terrorblade/refs/heads/main/scripts/install.sh | bash
```

<details>
<summary>Installation Steps</summary>

The installer will:


- ✅ Set up Python environment with `uv`
- ✅ Install all dependencies using unified requirements
- ✅ Configure security scanning tools
- ✅ Create convenient activation scripts
- ✅ Verify the installation

</details>

### Prerequisites


#### Required

- Python 3.12+
- DuckDB CLI (for database operations)

- Python 3.12+
- DuckDB CLI (for database operations)

#### Optional 

- Telegram API credentials (API ID and API Hash)  
- CUDA-compatible GPU (for GPU-accelerated features)
  

## Manual Installation

```bash
git clone https://github.com/sevapru/terrorblade.git
cd terrorblade
make install
```

After installation:


```bash
cd ~/terrorblade
source .venv/bin/activate
make help                  # See all available commands
make test                  # Verify your setup
make security              # Run security scans
cp .env.example .env       # Configure your local variables
```



## Implemented Features

- **Data Extraction:**
  - Asynchronous message fetching using Telethon API
  - JSON archive processing from Telegram Desktop exports
  - Incremental updates to avoid re-downloading existing messages
  - Support for media messages and reply tracking
  
- **Data Storage & Management:**
  - DuckDB storage for efficient message management
  - Standardized schema across different messaging platforms
  - Comprehensive logging system
  - Rate limiting protection with automatic retry
  
- **Data Processing & Analytics Preparation:**
  - GPU acceleration for data processing (optional)
  - Message preprocessing and cleaning
  - Embedding generation for semantic search capabilities
  - Conversation clustering and grouping
  - Advanced data analysis pipeline (Thoth module - in development)

<details>
<summary>Development Phases</summary>

### **Phase 1: Data Ingestion & Processing**  


| Module | Mythological Figure | Function | Status |  
|--------|---------------------|----------|--------|  
| **Terrorblade** | Demon | Data extraction and parsing (Telegram, WhatsApp, VK/Instagram/Facebook) | ✅ Released (Telegram) |  
| **Thoth** | Egyptian Scribe God | Topic analysis & visualization | 🔄 Coming Soon |  

### **Phase 2: Multi-Platform Expansion**  


| Module | Mythological Figure | Function | Status |  
|--------|---------------------|----------|--------|  
| **Argus** | All-Seeing Giant (Greek) | Multi-platform data extraction (WhatsApp, VK, Instagram, Facebook) | Planned (Q4 2025) |  
| **Themis**/**Nemesis** | Goddess of Justice/Retribution (Greek) | Advanced analytics & pattern detection | Planned (Q1 2026) |  

### **Phase 3: Ethics & Infrastructure**  


| Module | Mythological Figure | Function | Status |  
|--------|---------------------|----------|--------|  
| **Iris** | Rainbow Messenger (Greek) | Data visualization & interactive dashboards | Planned (Q2 2025) |  
| **Janus** | Two-Faced God of Gates (Roman) | Ethical data anonymization & access control | Planned (Q3 2025) |  

### **Phase 4: Specialized Expansion**  


- **Hephaestus**: AI customization toolkit (train models on niche slang/contexts).  
- **Hypnos**: Sleep/fatigue analysis via activity timelines.  
- **Eris**: Community stress-testing through controlled chaos (A/B message testing).

- **Eris**: Community stress-testing through controlled chaos (A/B message testing).

</details>

---

## Quick Start Demo


### Processing messages from extracted archive


This method is much safer from the perspective of account access and implementation since you upload your messages directly with machine-readable JSON.

#### Step 1: Export your Telegram messages as JSON

1. **Open Telegram Desktop** (this feature is not available on mobile apps)
2. **Go to Settings** → **Advanced** → **Export Telegram data**
3. **Configure export settings:**
   - ✅ Check "Personal chats"
   - **Format**: Select **"Machine-readable JSON"**
   - **Media**: You can uncheck all media types to speed up export (we only need text messages)
   - **Size limit**: Doesn't matter, chill
4. **Click "Export"** and choose a location to save the files
5. **Wait for export to complete** - this may take several minutes depending on your message history size
6. **Locate the `result.json` file** in the exported folder

#### Step 2: Process the JSON file with Terrorblade

Once you have the `result.json` file, you can process it using our JSON processor:

```python
from terrorblade.examples.create_db_from_tg_json import create_db_from_telegram_json

phone = "1234567890"  # Replace with your phone number (numbers only, no spaces or special characters required)
json_file_path = "/path/to/your/result.json" # Your telegram archive
db_path = "telegram_data.db"# Defaults to "telegram_data.db" in current directory)

create_db_from_telegram_json(phone, json_file_path, db_path)
```

**Parameters explained:**

- `phone` (required): Your phone number as a string with numbers only. This is used as an identifier in the database and doesn't need to match your actual Telegram number exactly.
- `json_file_path` (required): Full path to your exported `result.json` file
- `db_path` (optional): Where to save the database file. Defaults to `"telegram_data.db"` in the current directory.

The processor will:

- Create a DuckDB database with your messages
- Parse and clean the message data
- Generate embeddings for semantic search
- Group messages into conversation clusters
- Display a summary of processed data

## Vector Search

Once your messages are processed and stored in the database, you can perform semantic search to find relevant conversations using the vector search functionality. The system uses embeddings to understand the meaning behind your search terms and finds contextually similar messages.

### Features

- **Semantic Search**: Find messages by meaning, not just exact text matches
- **Cluster Context**: See conversation snippets around found messages for better understanding
- **HNSW Indexing**: Fast similarity search using DuckDB's VSS extension
- **Multi-keyword Support**: Search for multiple terms at once

### Quick Search Example

Use the simplified vector search example to find messages containing specific topics:

```bash
python terrorblade/examples/vector_search_example.py "поплава" --db telegram_data.db --phone 1234567890
```


Arguments:

- `keywords`: One or more keywords to search for in the vector store
- `--db`: Path to your DuckDB database file
- `--phone`: Your phone number identifier
- `--top-k`: Number of results per keyword (default: 10)
  
<details>
<summary>Possible Output</summary>

```bash
```bash
(terrorblade) ┌─[seva@*****] - [~/сode/terrorblade/terrorblade/examples] - [Mon Aug 04, 18:12]
└─[$]> python vector_search_example.py "паплава" --phone 79992004210 --db telegram_data.db
Database: 928899 embeddings, 298 chats

HNSW Index Statistics
==================================================
Index Name: idx_embeddings_79992004210
Table: chat_embeddings_79992004210
Type: HNSW
Indexed Rows: 928,899
Estimated Memory: 4082.08 MB
Key Columns: embeddings
Unique: False

Performance Estimates:
   Search complexity: O(log(928,899))
==================================================
Keyword: 'паплава'
============================================================
shape: (3, 8)
┌────────────┬───────────┬────────────┬─────────┬───────────────┬───────────┬─────────────────────┬────────────────────────────────────────────────────────────────────────┐
│ message_id ┆ chat_id   ┆ similarity ┆ cluster ┆ from_name     ┆ chat_name ┆ date                ┆ context_snippet                                                        │
│ ---        ┆ ---       ┆ ---        ┆ ---     ┆ ---           ┆ ---       ┆ ---                 ┆ ---                                                                    │
│ i64        ┆ i64       ┆ f64        ┆ str     ┆ str           ┆ str       ┆ datetime[μs]        ┆ str                                                                    │
╞════════════╪═══════════╪════════════╪═════════╪═══════════════╪═══════════╪═════════════════════╪════════════════════════════════════════════════════════════════════════╡
│ 578768     ┆ 335211685 ┆ 1.0        ┆ 6       ┆ Seva ✨       ┆ Максим    ┆ 2022-05-20 20:35:32 ┆ Максим Колмаков: Потом прогревается и норм                             │
│            ┆           ┆            ┆         ┆               ┆           ┆                     ┆ Seva ✨: хек                                                           │
│            ┆           ┆            ┆         ┆               ┆           ┆                     ┆ Seva ✨: Ну заодно посмотрят                                           │
│            ┆           ┆            ┆         ┆               ┆           ┆                     ┆ Seva ✨: может скажут                                                  │
│            ┆           ┆            ┆         ┆               ┆           ┆                     ┆ Seva ✨: что пиздец                                                    │
│            ┆           ┆            ┆         ┆               ┆           ┆                     ┆ >>> Seva ✨: паплава                                                   │
│            ┆           ┆            ┆         ┆               ┆           ┆                     ┆ Максим Колмаков: Да не, это обычная тема на этих движках под 60к пр... │
│            ┆           ┆            ┆         ┆               ┆           ┆                     ┆ Максим Колмаков: А чтобы потом мне отец не сказал вот ты хуйню сдел... │
│            ┆           ┆            ┆         ┆               ┆           ┆                     ┆ Максим Колмаков...                                                     │
│            ┆           ┆            ┆         ┆               ┆           ┆                     ┆                                                                        │
│ 301237     ┆ 246090345 ┆ 0.9502     ┆ 54      ┆ Seva ✨       ┆ Родион    ┆ 2020-04-01 11:20:53 ┆ Родион Спирин: Пошёл нахуй                                             │
│            ┆           ┆            ┆         ┆               ┆           ┆                     ┆ Родион Спирин: Ебанный блять                                           │
│            ┆           ┆            ┆         ┆               ┆           ┆                     ┆ Seva ✨: У тебя уже это                                                │
│            ┆           ┆            ┆         ┆               ┆           ┆                     ┆ Seva ✨: Паплава                                                       │
│            ┆           ┆            ┆         ┆               ┆           ┆                     ┆ Родион Спирин: Че                                                      │
│            ┆           ┆            ┆         ┆               ┆           ┆                     ┆ >>> Seva ✨: Паплава                                                   │
│            ┆           ┆            ┆         ┆               ┆           ┆                     ┆ Родион Спирин: Че                                                      │
│            ┆           ┆            ┆         ┆               ┆           ┆                     ┆ Seva ✨: Че?                                                           │
│            ┆           ┆            ┆         ┆               ┆           ┆                     ┆ Родион Спирин: Паплава                                                 │
│            ┆           ┆            ┆         ┆               ┆           ┆                     ┆ Родион Спирин: Что это блять                                           │
│            ┆           ┆            ┆         ┆               ┆           ┆                     ┆ Seva ✨: Именно она                                                    │
└────────────┴───────────┴────────────┴─────────┴───────────────┴───────────┴─────────────────────┴────────────────────────────────────────────────────────────────────────┘
```

### Understanding the Results

- **Similarity**: Cosine similarity score (0-1, higher = more similar)
- **Cluster**: Conversation cluster ID or "No cluster" for standalone messages
- **Match Type**: Shows if this is the original similarity match or a related cluster member
- **Context Snippet**: Shows 5 messages before and after the found message within the same conversation cluster
- **>>> Symbol**: Marks the exact message that matched your search

</details>

## MCP Server

Terrorblade ships an MCP server exposing vector search and cluster retrieval tools (compatible with Cursor, Claude, etc.).

### Run locally (stdio)

```bash
uv run terrorblade-mcp
```

### Tools

- `vector_search` — semantic search over messages with optional cluster snippets
- `get_cluster` — fetch messages for a specific `group_id` in a `chat_id`
- `random_large_cluster` — return a random large conversation cluster

Inputs mirror the database usage:

- `db_path`: path to DuckDB file (e.g., `telegram_data.db`)
- `phone`: user phone identifier (with or without `+`)

For Cursor MCP setup, add a server with command `uv` and args like:

```bash
uvx terrorblade-mcp
```

Refer to Cursor’s MCP docs for configuration details.

## MCP Server

Terrorblade ships an MCP server exposing vector search and cluster retrieval tools (compatible with Cursor, Claude, etc.).

### Run locally (stdio)

```bash
uv run terrorblade-mcp
```

### Tools

- `vector_search` — semantic search over messages with optional cluster snippets
- `get_cluster` — fetch messages for a specific `group_id` in a `chat_id`
- `random_large_cluster` — return a random large conversation cluster

Inputs mirror the database usage:

- `db_path`: path to DuckDB file (e.g., `telegram_data.db`)
- `phone`: user phone identifier (with or without `+`)

For Cursor MCP setup, add a server with command `uv` and args like:

```bash
uvx terrorblade-mcp
```

Refer to Cursor’s MCP docs for configuration details.

## Processing messages directly from Telegram API

### Before you start
### Before you start

>
>⚠️ **WARNING**:
>
> **This example is designed for small accounts with limited message history. Using it on accounts with large message histories may:**
>
> - **Trigger Telegram's rate limits**
> - **Cause your account to be temporarily disconnected**
> - **Require re-authentication on all devices**
> - **Result in temporary loss of access to your account**
>
>If you have a large message history, consider using a test account first.

In order to continue with this example you need to be able to obtain creditnails from the Telegram
<details>
<summary>Obtaining Telegram API Credentials</summary>

To use direct Telegram message synchronization, you'll need to obtain API credentials:

1. Visit [https://my.telegram.org/apps](https://my.telegram.org/apps)
2. Log in with your Telegram account
3. Click "Create application"
4. Fill in the required fields:
   - App title: Choose any name
   - Short name: Choose a short identifier
   - Platform: Desktop
   - Description: Brief description of your use case
5. After creation, you'll receive:
   - `api_id`: A numeric value
   - `api_hash`: A 32-character hexadecimal string
6. Add these values to your `.env` file:

   ```bash

   ```bash
   API_ID=your_api_id
   API_HASH=your_api_hash
   ```

Note: Keep these credentials secure and never share them publicly.

</details>

This example:

1. Initialize a DuckDB database to store your Telegram messages
2. Connect to Telegram using your phone number (you'll need to input auth code)
3. Download your message history
4. Process messages to:
   - Calculate embeddings for semantic search
   - Group messages into conversation clusters
   - Store everything in the database

The process may take some time depending on your message history size. Progress will be shown in the console.

**Note from Maintainers**: I am actively working on implementing rate limiting and batch processing to make the tool safer for accounts with larger message histories. In the meantime, please exercise caution when using this tool with accounts containing extensive message histories.

**Known Issue**: In case of API failure, recent messages from the last few minutes may be deleted from your Telegram account.

#### Try it on your own risk

After you have placed your credentials in the `.env` file, you can run the preprocessor which will organize your data within the DuckDB.

```python
from terrorblade.examples.create_db_from_tg_account import run_processor

phone = "+1234567890"  # Replace with your actual phone number (include country code)
run_processor(phone)
```

## Database Schema

For detailed information about the database structure, vector search capabilities, and DuckDB integration, see the comprehensive [Database Schema Documentation](docs/DATABASE_SCHEMA.md).

## GPU Acceleration

This project includes support for GPU acceleration through NVIDIA RAPIDS libraries.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.