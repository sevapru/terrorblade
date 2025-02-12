# Chat Analysis Tool

A Python-based Telegram message parser that allows you to fetch and store messages from Telegram chats using the Telegram API. The parser stores messages in a DuckDB database for efficient querying and analysis.

## Features

- Asynchronous message fetching using Telethon
- Incremental updates to avoid re-downloading existing messages
- DuckDB storage for efficient message management
- Support for media messages and reply tracking
- Comprehensive logging system
- Rate limiting protection with automatic retry

## Prerequisites

- Python 3.12+
- CUDA-compatible GPU (for GPU-accelerated features)
- Telegram API credentials (API ID and API Hash)

## Installation

1. Clone the repository:
```bash
git clone git@github.com:sevapru/terrorblade.git
cd terrorblade
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root with your Telegram credentials:
```env
API_ID=your_api_id
API_HASH=your_api_hash
PHONE=your_phone_number # Not necessary for json parsing
```

## Usage

Run the parser with:
```bash
python src/data/loaders/telegram/parse_telegram_client.py
```

The script will:
1. Connect to Telegram using your credentials
2. Fetch messages from your chats
3. Store them in a DuckDB database (`telegram_data.db`)

## Database Schema

The parser creates two main tables:

### Users Table
- `phone`: VARCHAR (Primary Key)
- `last_update`: TIMESTAMP

### Messages Table
- `id`: BIGINT
- `chat_id`: BIGINT
- `date`: TIMESTAMP
- `text`: TEXT
- `from_id`: BIGINT
- `reply_to_msg_id`: BIGINT
- `media_type`: TEXT
- `file_name`: TEXT
- `from`: TEXT

## GPU Acceleration

This project includes support for GPU acceleration through NVIDIA RAPIDS libraries. Make sure you have a compatible CUDA installation to use these features.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License
