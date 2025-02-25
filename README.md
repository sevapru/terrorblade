# **Chat Analysis Tool**

A Python-based Telegram message parser that allows you to fetch and store messages from Telegram chats using the Telegram API. The parser stores messages in a DuckDB database for efficient querying and analysis.


## **Development Roadmap**  

### **Phase 1: Core Analytics**  
| Module | Mythological Figure | Function | Status |  
|--------|---------------------|----------|--------|  
| **Terrorblade** | Demon | Behavioral pattern extraction | ✅ Released |  
| **Thoth** | Egyptian Scribe God | Topic analysis & visualization | 🚧 In Development |  

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

---

## **Future Vision**  
### **Phase 4: Specialized Expansion**  
- **Hephaestus**: AI customization toolkit (train models on niche slang/contexts).  
- **Hypnos**: Sleep/fatigue analysis via activity timelines.  
- **Eris**: Community stress-testing through controlled chaos (A/B message testing).  

---

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

## Prerequisites

- Python 3.12+
- DuckDB CLI (for database operations)
- CUDA-compatible GPU (optional, for GPU-accelerated features)
- Telegram API credentials (API ID and API Hash)

## Installation

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

3. Choose your installation type:

For basic installation (CPU only):

```bash
make install
```

For development (includes testing and linting tools):

```bash
make dev
```

For GPU-accelerated features:

```bash
make install-cuda
```

For installation with Thoth analysis package:

```bash
make install-thoth
```

The installation process will:

- Check for required system dependencies (DuckDB CLI)
- Install `uv` if not present
- Set up a virtual environment
- Install required Python packages
- Verify configuration files

## Quick Start Demo

### 1. Message Loading

```python
from terrorblade.data.loaders import TelegramLoader

# Initialize the loader
loader = TelegramLoader()

# Load messages from a specific chat
chat_id = "your_chat_id"
messages = await loader.load_messages(chat_id)
```

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

## License

Creative Commons Attribution-NonCommercial 4.0 International License

Copyright (c) 2024 Vsevolod Prudius

This work is licensed under the Creative Commons Attribution-NonCommercial 4.0
International License. To view a copy of this license, visit
<http://creativecommons.org/licenses/by-nc/4.0/> or send a letter to Creative Commons,
PO Box 1866, Mountain View, CA 94042, USA.
