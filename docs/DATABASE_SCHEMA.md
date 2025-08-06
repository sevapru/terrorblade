# Database Schema & Architecture

Terrorblade uses **DuckDB** as its primary database engine, providing a modern, high-performance analytical database that combines the simplicity of SQLite with the power of columnar storage and vector operations. The system implements a **centralized storage architecture** where each user's data is organized into separate tables using phone numbers as identifiers.

## 🏗️ Architecture Overview

### Centralized Storage Design

The database follows a **multi-tenant architecture** where each user's data is isolated in separate tables:

- **User Isolation**: Each phone number gets its own set of tables
- **Schema Consistency**: All users share the same table structure defined in `terrorblade/data/dtypes.py`
- **Scalable Design**: Easy to add new users without affecting existing data
- **Vector Integration**: Native support for embeddings and semantic search

### DuckDB Integration

Terrorblade leverages DuckDB's advanced features:

- **Columnar Storage**: Optimized for analytical queries and large datasets
- **Vector Operations**: Native support for embeddings via the VSS extension
- **HNSW Indexing**: High-performance similarity search using Hierarchical Navigable Small World graphs
- **ACID Compliance**: Reliable data integrity and transaction support
- **Zero Configuration**: No server setup required, file-based operation

## 📊 Core Tables

### Users Table

Central registry of all users in the system:

| Column | Type | Description |
|--------|------|-------------|
| `phone` | VARCHAR | Primary key - user's phone number identifier |
| `last_update` | TIMESTAMP | Last time user data was synchronized |
| `first_seen` | TIMESTAMP | When user was first added to the system |

### Messages Table (`messages_{phone}`)

Primary storage for all message data with standardized schema:

| Column | Type | Description |
|--------|------|-------------|
| `message_id` | BIGINT | Unique Telegram message identifier |
| `chat_id` | BIGINT | Chat/conversation identifier |
| `date` | TIMESTAMP | Message timestamp with timezone |
| `text` | TEXT | Message content (cleaned and processed) |
| `from_id` | BIGINT | Sender's unique identifier |
| `reply_to_message_id` | BIGINT | ID of message being replied to |
| `media_type` | TEXT | Type of attached media (photo, video, etc.) |
| `file_name` | TEXT | Name of attached file |
| `from_name` | TEXT | Display name of sender |
| `chat_name` | TEXT | Name of chat/conversation |
| `forwarded_from` | TEXT | Source of forwarded messages |

### Message Clusters Table (`message_clusters_{phone}`)

Groups related messages into conversation clusters:

| Column | Type | Description |
|--------|------|-------------|
| `message_id` | BIGINT | References message in messages table |
| `chat_id` | BIGINT | Chat identifier |
| `group_id` | INTEGER | Cluster identifier (messages in same cluster) |

**Primary Key**: `(message_id, chat_id)`

### Chat Embeddings Table (`chat_embeddings_{phone}`)

Vector storage for semantic search capabilities:

| Column | Type | Description |
|--------|------|-------------|
| `message_id` | BIGINT | References message in messages table |
| `chat_id` | BIGINT | Chat identifier |
| `embedding` | FLOAT[768] | 768-dimensional embedding vector |

**Primary Key**: `(message_id, chat_id)`

## 🔍 Vector Search Architecture

### HNSW Indexing

Terrorblade implements high-performance vector search using DuckDB's VSS extension:

```sql
-- HNSW index for fast similarity search
CREATE INDEX idx_embeddings_{phone} 
ON chat_embeddings_{phone}(embeddings) 
USING HNSW(embeddings);
```

**Features:**

- **Sub-second Search**: HNSW algorithm provides logarithmic search complexity
- **Cosine Similarity**: Optimized for semantic similarity matching
- **Configurable Parameters**: Adjustable index size and search precision
- **Persistent Storage**: Indexes are automatically persisted to disk

### Vector Operations

The system supports multiple vector operations:

- **Similarity Search**: Find most similar messages to a query
- **Distance Search**: Find messages within a distance threshold
- **Batch Operations**: Process multiple queries efficiently
- **Cluster Context**: Retrieve conversation context around matches

### Search Capabilities

```python
# Semantic search for messages
vector_store.similarity_search(
    query_vector=embedding,
    top_k=10,
    similarity_threshold=0.7
)
```

## 🛠️ Database Management

### Session Management

Terrorblade includes a dedicated session manager for Telegram authentication:

| Column | Type | Description |
|--------|------|-------------|
| `phone` | VARCHAR | User's phone number |
| `session_data` | TEXT | Encrypted session string |
| `created_at` | TIMESTAMP | When session was created |
| `last_used` | TIMESTAMP | Last authentication time |

### Data Types & Schema

All data types are centrally defined in [`terrorblade/data/dtypes.py`](../terrorblade/data/dtypes.py):

```python
TELEGRAM_SCHEMA = {
    "message_id": {
        "polars_type": pl.Int64,
        "db_type": "BIGINT",
        "description": "Unique identifier for the message"
    },
    "embeddings": {
        "polars_type": pl.Array(pl.Float32, shape=768),
        "db_type": "FLOAT[768]",
        "description": "F32 embeddings array with fixed length 768"
    }
    ...
}
```

**Benefits:**

- **Type Safety**: Consistent data types across all operations
- **Documentation**: Self-documenting schema with descriptions
- **Maintainability**: Single source of truth for data structure
- **API Integration**: Automatic schema generation for APIs

## 📈 Performance Features

### Optimized Queries

The database implements several performance optimizations:

- **Indexed Searches**: Primary keys and foreign keys are indexed
- **Partitioned Data**: Messages separated by user for faster access
- **Columnar Storage**: Efficient compression and query performance
- **Vector Indexing**: HNSW indexes for sub-second similarity search

### Memory Management

- **Lazy Loading**: Indexes loaded on-demand
- **Connection Pooling**: Efficient database connection management
- **Batch Operations**: Bulk inserts and updates for better performance
- **Automatic Cleanup**: Session and temporary data cleanup

## 🔧 Database Operations

### Connection Management

```python
# Initialize database with read-only option
db = TelegramDatabase("telegram_data.db", read_only=True)

# Vector store with automatic VSS extension loading
vector_store = VectorStore("telegram_data.db", phone="1234567890")
```

### Data Operations

```python
# Add messages in batches
db.add_messages(phone="1234567890", messages_df=polars_df)

# Get user statistics
stats = db.get_user_stats(phone="1234567890")

# Perform vector search
results = vector_store.similarity_search(query_vector, top_k=10)
```

### Index Management

```python
# Create HNSW index for vector search
vector_store.create_hnsw_index()

# Get index statistics
stats = vector_store.get_index_stats()
vector_store.print_index_stats()
```

## Security & Privacy

### Data Isolation

- **User Separation**: Each user's data is completely isolated
- **Phone-based Partitioning**: Tables are separated by phone number
- **No Cross-User Access**: Impossible to access other users' data

### Session Security

- **Encrypted Storage**: Session data is encrypted in the database
- **Automatic Cleanup**: Old sessions are automatically removed
- **Access Control**: Session access is restricted to the owner

## 📚Additional Resources

- **[DuckDB Documentation](https://duckdb.org/docs/)**: Official DuckDB guides
- **[VSS Extension](https://github.com/duckdb/duckdb-vss)**: Vector similarity search
- **[HNSW Algorithm](https://arxiv.org/abs/1603.09320)**: Hierarchical Navigable Small World
- **[Polars Integration](https://pola.rs/)**: Fast DataFrame operations


## Planned Features

- **Multi-Platform Support**: WhatsApp, VK, Instagram schemas
- **Advanced Analytics**: Dashboards and insights
- **Distributed Storage**: Support for larger group-based datasets
- **API Integration**: RESTful endpoints for database access
