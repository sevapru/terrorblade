# Thoth

A data analysis package for analyzing Telegram message embeddings with vector database support.

## Features

- **Vector Database Integration**: Store and query message embeddings using Qdrant vector database (local or remote)
- **Topic Analysis**: Identify key topics and themes across conversations
- **User Behavior Analysis**: Analyze how users interact and what topics they engage with
- **Cross-Chat Analysis**: Compare patterns across different chat groups
- **Semantic Search**: Find messages by meaning rather than keyword matching
- **Topic Evolution Tracking**: See how topics change over time
- **Environment-Based Configuration**: Configure the analyzer using a `.env` file
- **Comprehensive Logging**: Built-in logging system with custom levels

## Installation

```bash
pip install thoth
```

### Requirements

- Python 3.12+
- Dependencies listed in `pyproject.toml`

## Configuration

Thoth can be configured using environment variables or a `.env` file. Copy the `.env.template` file to `.env` and modify the settings:

```bash
cp .env.template .env
# Edit .env with your preferred text editor
```

Key configuration options:

- **Database settings**: Path to DuckDB file and phone number
- **Qdrant settings**: Local or remote mode, paths/URLs, and API keys
- **Embedding model**: Model name and vector size
- **Logging**: Log level and file path

## Usage

### Programmatic Usage

```python
from thoth import ThothAnalyzer
from datetime import datetime, timedelta

# Initialize with config from .env file
analyzer = ThothAnalyzer()

# Or specify configuration directly
analyzer = ThothAnalyzer(
    db_path="path/to/telegram_data.db",
    phone="+1234567890",
    qdrant_url="http://my-qdrant-server:6333",  # For remote Qdrant
    qdrant_api_key="my-api-key"                 # If authentication is required
)

# Import data from DuckDB
analyzer.import_from_duckdb()

# Analyze topics
topics = analyzer.analyze_topics(
    chat_id=123456789,
    n_topics=5,
    date_from=datetime.now() - timedelta(days=30)
)

# Analyze topic sentiment
sentiment = analyzer.analyze_topic_sentiment(topic_id=1)

# Search for semantically similar messages
results = analyzer.search(
    query="What do you think about the new policy?",
    chat_id=123456789,
    limit=5
)
```

### Command Line Usage

```bash
# Import data
thoth --db-path telegram_data.db --phone +1234567890 import

# Analyze topic activity
thoth topics --action common --chat-id 123456789

# Search for messages
thoth search --query "artificial intelligence" --chat-id 123456789

# Get help
thoth --help
```

## Architecture

### Modular Structure

- **Data Import**: Import messages from DuckDB into vector database
- **Vector Storage**: Store embeddings in Qdrant (local or remote)
- **Embedding Generation**: Convert messages to embeddings using SentenceTransformers
- **Analysis Modules**: Topic analysis, user behavior, sentiment analysis
- **Command Interface**: CLI for common operations

### Vector Database Structure

- **Messages Collection**: Stores all message embeddings with metadata
- **Chat-Specific Collections**: Dedicated collections for each chat group
- **Topic Collection**: Stores topic embeddings for faster lookup

### Logging System

Thoth includes a comprehensive logging system with a custom "NICE" level between INFO and WARNING. The NICE level is used to log important quantitative metrics at the end of critical processing stages.

## Remote Qdrant Support

Thoth now supports both local and remote Qdrant instances:

- **Local Mode**: Uses a file-based Qdrant instance for simple deployment
- **Remote Mode**: Connects to a remote Qdrant server for:
  - Larger datasets
  - Shared access
  - Better performance
  - Horizontal scaling

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 