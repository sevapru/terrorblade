<div align="center">
  <a href="https://github.com/sevapru/terrorblade">
    <img alt="terrorblade" width="240" src="fun/terrorblade_logo.png">
  </a>
</div>

# Terrorblade

A unified data extraction and parsing platform for messaging platforms, featuring Telegram message processing, data standardization, and analytics preparation capabilities.


## Linux, Windows, macOS
```bash
curl -fsSL https://raw.githubusercontent.com/sevapru/terrorblade/main/scripts/install.sh | bash
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
* Python 3.12+
* DuckDB CLI (for database operations)

#### Optional 
* Telegram API credentials (API ID and API Hash)  
* CUDA-compatible GPU (for GPU-accelerated features)
  

## Manual Installation

Full installation guide and troubleshooting: [INSTALL.md](fun/generated/v0.2.0/INSTALL.md)

```bash
git clone https://github.com/sevapru/terrorblade.git
cd terrorblade
make install
```

After installation:
```bash
cd ~/terrorblade
source .venv/bin/activate  # or: ./activate.sh
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
| **Argus** | All-Seeing Giant (Greek) | Multi-platform data extraction (WhatsApp, VK, Instagram, Facebook) | Planned (Q4 2024) |  
| **Themis**/**Nemesis** | Goddess of Justice/Retribution (Greek) | Advanced analytics & pattern detection | Planned (Q1 2025) |  

### **Phase 3: Ethics & Infrastructure**  
| Module | Mythological Figure | Function | Status |  
|--------|---------------------|----------|--------|  
| **Iris** | Rainbow Messenger (Greek) | Data visualization & interactive dashboards | Planned (Q2 2025) |  
| **Janus** | Two-Faced God of Gates (Roman) | Ethical data anonymization & access control | Planned (Q3 2025) |  

### **Phase 4: Specialized Expansion**  
- **Hephaestus**: AI customization toolkit (train models on niche slang/contexts).  
- **Hypnos**: Sleep/fatigue analysis via activity timelines.  
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


### Processing messages directly from Telegram API

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
   ```
   API_ID=your_api_id
   API_HASH=your_api_hash
   ```

Note: Keep these credentials secure and never share them publicly.
</details>

####  Before you start
⚠️ **WARNING**: 

**This example is designed for small accounts with limited message history. Using it on accounts with large message histories may:**
- **Trigger Telegram's rate limits**
- **Cause your account to be temporarily disconnected**
- **Require re-authentication on all devices**
- **Result in temporary loss of access to your account**

If you have a large message history, consider using a test account first.

This example is: 

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

The parser creates several tables for each user (where phone number is used as an identifier):

### Users Table

- `phone`: VARCHAR (Primary Key)
- `last_update`: TIMESTAMP
- `first_seen`: TIMESTAMP

### Messages Table (`messages_{phone}`)

- `message_id`: BIGINT
- `chat_id`: BIGINT
- `date`: TIMESTAMP
- `text`: TEXT
- `from_id`: BIGINT
- `reply_to_message_id`: BIGINT
- `media_type`: TEXT
- `file_name`: TEXT
- `from_name`: TEXT
- `chat_name`: TEXT
- `forwarded_from`: TEXT

### Message Clusters Table (`message_clusters_{phone}`)

- `message_id`: BIGINT
- `chat_id`: BIGINT
- `group_id`: INTEGER
Primary Key: (message_id, chat_id)

### Chat Embeddings Table (`chat_embeddings_{phone}`)

- `message_id`: BIGINT
- `chat_id`: BIGINT
- `embedding`: DOUBLE[]
Primary Key: (message_id, chat_id)

## GPU Acceleration

This project includes support for GPU acceleration through NVIDIA RAPIDS libraries. To enable GPU features:

1. Ensure you have a CUDA-compatible GPU
2. Install with CUDA support: `make install-cuda`
3. Set `USE_CUDA=true` in your `.env` file

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
