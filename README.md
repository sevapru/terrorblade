# **Chat Analysis Tool**

A Python-based Telegram message parser that allows you to fetch and store messages from Telegram chats using the Telegram API. The parser stores messages in a DuckDB database for efficient querying and analysis.



## Implemented Features

- Asynchronous message fetching using Telethon
- Incremental updates to avoid re-downloading existing messages
- DuckDB storage for efficient message management
- Support for media messages and reply tracking
- Comprehensive logging system
- Rate limiting protection with automatic retry
- GPU acceleration for data processing (optional)
- Semantic search in messages
- Message sentiment analysis
- Topic modeling and clustering
- Advanced data analysis through the Thoth package


## **Development Roadmap**  

### **Phase 1: Core Analytics**  
| Module | Mythological Figure | Function | Status |  
|--------|---------------------|----------|--------|  
| **Terrorblade** | Demon | Behavioral pattern extraction | ✅ Released |  
| **Thoth** | Egyptian Scribe God | Topic analysis & visualization | 🚧 In Development |  

<details>
<summary>Future Development Phases</summary>

### **Phase 2: Security & Observation**  
| Module | Mythological Figure | Function | Status |  
|--------|---------------------|----------|--------|  
| **Argus** | All-Seeing Giant (Greek) | Cross-platform monitoring (Telegram, WhatsApp, etc.) | Planned (Q4 2024) |  
| **Themis**/**Nemesis** | Goddess of Justice/Retribution (Greek) | Threat detection (terrorism, extremism patterns) | Planned (Q1 2025) |  

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


## Prerequisites

* Python 3.12+
* DuckDB CLI (for database operations)

Optional 
* Telegram API credentials (API ID and API Hash)  
* CUDA-compatible GPU (for GPU-accelerated features)
  

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

## Installation

> **Note**: See the [Docker Usage](#docker-usage) section below for docker installation instructions.

The project uses `uv` for fast Python package management and virtual environment handling.

1. Clone the repository:

```bash
git clone git@github.com:sevapru/terrorblade.git
cd terrorblade
```

2. Set up the environment:
```bash
# Copy environment configuration
cp .env.example .env

# Edit .env file with your credentials
# Required: API_ID, API_HASH
# Optional: DUCKDB_PATH, LOG_LEVEL, LOG_FILE, LOG_DIR
```

3. Run the installation:

```bash
make install
```

The installation process will:

- Check for required system dependencies (DuckDB CLI)
- Install `uv` if not present
- Set up a virtual environment
- Install required Python packages
- Verify configuration files

## Quick Start Demo

### Processing messages directly from Telegram API

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



```python
from terrorblade.examples.create_db_from_tg_account import run_processor

phone = "+1234567890"  # Replace with your actual phone number (include country code)
run_processor(phone)
```

**Note from Maintainers**: We are actively working on implementing rate limiting and batch processing to make the tool safer for accounts with larger message histories. In the meantime, please exercise caution when using this tool with accounts containing extensive message histories.

**Fun bug**: in case of failure your messages for the last few minutes will be deleted (and your firends will be sad about it)

### Processing messages from extracted archive
This method is much safer from the perspective of account access and implementation since you upload your messages directly with machine-readable JSON. 



### 2. Basic Analysis

```python
from terrorblade.analysis import MessageAnalyzer

# Initialize analyzer
analyzer = MessageAnalyzer()

# Get basic statistics
stats = analyzer.get_chat_statistics(chat_id)
print(f"Total messages: {stats.total_messages}")
print(f"Active users: {stats.unique_users}")
print(f"Media messages: {stats.media_count}")
```

## Additional Examples
### 3. Semantic Search

```python
from terrorblade.search import SemanticSearcher

# Initialize searcher
searcher = SemanticSearcher()

# Search for semantically similar messages
query = "What do you think about AI?"
results = searcher.search(query, limit=5)

for msg in results:
    print(f"Score: {msg.score:.2f} | Message: {msg.text}")
```

### 4. Advanced Analysis with Thoth

```python
from thoth import find_most_common_token, find_most_common_topic

# Find the most common token in embeddings
token, count = find_most_common_token("telegram_data.db")
print(f"Most common token: {token} (count: {count})")

# Find the most common topic cluster
topic, size = find_most_common_topic("telegram_data.db")
print(f"Most common topic: {topic} (size: {size})")
```

### 5. Topic Modeling

```python
from terrorblade.analysis import TopicModeler

# Initialize topic modeler
modeler = TopicModeler()

# Extract topics from chat
topics = modeler.extract_topics(chat_id)

for topic in topics:
    print(f"Topic: {topic.name}")
    print(f"Keywords: {', '.join(topic.keywords)}")
```

### 6. GPU-Accelerated Analysis (if CUDA is enabled)

```python
from terrorblade.analysis import GPUMessageAnalyzer

# Initialize GPU-accelerated analyzer
analyzer = GPUMessageAnalyzer()

# Perform clustering on messages
clusters = analyzer.cluster_messages(chat_id)
```

## Development Commands

- `make lint`: Run code formatting (black), import sorting (isort), and type checking (mypy)
- `make test`: Run test suite with pytest
- `make clean`: Clean up temporary files, caches, and virtual environments

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
- `from`: TEXT
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


# Docker Usage

The project includes a `Dockerfile` to build and run `Terrorblade` and `Thoth` services in containers. This uses a multi-stage build approach with `uv` for dependency management.

### Prerequisites

- Docker installed and running.

### Building the Images

To build the Docker images, navigate to the project root directory (where the `Dockerfile` is located) and run:

```bash
# Build the development image (includes debugging tools)
docker build --target dev -t terrorblade-dev .

# Build the production image for Terrorblade
docker build --target terrorblade_prod -t terrorblade-prod .

# Build the production image for Thoth
docker build --target thoth_prod -t thoth-prod .
```

### Running the Containers

#### Terrorblade Service

To run the `Terrorblade` service (production):

```bash
docker run -d --name terrorblade \\
  -v ./data:/app/data \\ # Mount a volume for DuckDB data (adjust path if needed)
  -v ./path/to/your/.env:/app/.env \\ # Mount your .env file
  terrorblade-prod
```

To run the `Terrorblade` service in development mode with debugging enabled on port 5678:

```bash
docker run -d --name terrorblade-dev \\
  -p 5678:5678 \\
  -v $(pwd):/app \\ # Mount current directory for live code changes
  -v ./path/to/your/.env:/app/.env \\ # Mount your .env file
  terrorblade-dev
```

Attach your debugger to port `5678`.

#### Thoth Service

**Note:** The `Thoth` service entrypoint in the `Dockerfile` is currently a placeholder. You will need to update the `CMD` in the `thoth_prod` stage of the `Dockerfile` to correctly start your Thoth application (e.g., if it's a web server like Flask/Gunicorn).

Assuming you have updated the `Thoth` entrypoint, you can run it (production):

```bash
docker run -d --name thoth \\
  # Add necessary port mappings if Thoth is a web service, e.g., -p 8000:8000
  # Add volume mounts if Thoth needs to access data or config files
  # -v ./data:/app/data # Example: If Thoth reads the same DuckDB
  # -v ./path/to/thoth/config:/app/config # Example: Thoth specific config
  thoth-prod
```

### Networking and Data Sharing

- **DuckDB Access**: If `Thoth` needs to access the DuckDB database managed by the `Terrorblade` container, you will need to ensure the database file is accessible. This can be achieved by:
    - Mounting the same host directory (containing the DuckDB file) as a volume into both containers.
    - If DuckDB is run in server mode within the `Terrorblade` container, configure networking (e.g., a Docker network) so `Thoth` can connect to it.
- **Loki Logging**: The `Dockerfile` includes comments regarding Loki integration. To fully implement this, you would typically:
    - Configure your application's logger (`terrorblade/utils/logger.py`) to output logs in a structured format (e.g., JSON).
    - Run a log shipper like Promtail, either as a sidecar container or on the host, configured to send logs from your containers to Loki.

### Using Docker Compose (Recommended)

For managing multi-container applications like this, Docker Compose is highly recommended. You can create a `docker-compose.yml` file to define and run both `Terrorblade` and `Thoth` services, manage networks, volumes, and environment variables more easily.

**Example `docker-compose.yml` structure:**

```yaml
version: '3.8'
services:
  terrorblade:
    build:
      context: .
      target: terrorblade_prod # or dev for development
    container_name: terrorblade
    volumes:
      - ./data:/app/data
      - ./your.env:/app/.env # Ensure your .env file is correctly named and present
    # ports:
      # - "5678:5678" # If running dev target and need debugger access
    environment:
      # Define environment variables here or use env_file
      - DUCKDB_PATH=/app/data/telegram_data.db # Example

  thoth:
    build:
      context: .
      target: thoth_prod
    container_name: thoth
    ports:
      - "8000:8000" # Example if Thoth runs a web server on port 8000
    volumes:
      - ./data:/app/data # If Thoth needs access to the same DuckDB data
    depends_on:
      - terrorblade # Optional: if Thoth depends on Terrorblade starting first
    environment:
      # Thoth specific environment variables
      - DUCKDB_PATH=/app/data/telegram_data.db # Example

volumes:
  data:
    # You can define a named volume for persistent data
```

To run with Docker Compose:

```bash
docker-compose up -d
```

To stop:

```bash
docker-compose down
```
