# Terrorblade v0.2.0 Release Notes

**Release Date**: July 2025  
**Version**: 0.2.0  
**Previous Version**: 0.1.0  

## Overview

Terrorblade v0.2.0 represents a comprehensive data extraction and parsing platform for messaging platforms, with a complete focus shift from analytics to data ingestion and standardization. This release establishes Terrorblade as the foundational data processing engine for multi-platform messaging data.

## Core Architecture Changes

### Data Processing Pipeline
- **TelegramPreprocessor**: Full-featured preprocessing engine for Telegram data
- **TextPreprocessor**: Base class with embedding generation and clustering capabilities
- **DuckDB Integration**: High-performance database backend for message storage
- **Standardized Schema**: Unified data schema across different messaging platforms

### Database Management System
- **TelegramDatabase**: Complete database management with user-specific table creation
- **Session Management**: Robust session handling for database connections
- **Schema Versioning**: Centralized schema definitions in `terrorblade.data.dtypes`
- **Multi-user Support**: User-specific table structures with phone number identifiers

## New Features

### Data Extraction Methods
| Method | Source | Status | Description |
|--------|--------|--------|-------------|
| **JSON Archive Processing** | Telegram Desktop Export | ✅ Complete | Safe offline processing of exported message archives |
| **Telethon API Integration** | Telegram Client API | ✅ Complete | Direct API access with rate limiting and retry logic |
| **Incremental Updates** | Both Sources | ✅ Complete | Avoid re-downloading existing messages |
| **Media Message Support** | Both Sources | ✅ Complete | Handle media files and reply tracking |

### Data Storage Architecture
```
Database Schema:
├── users                    # User registry with phone numbers
├── messages_{phone}         # User-specific message tables
├── message_clusters_{phone} # Conversation grouping data
└── chat_embeddings_{phone}  # Semantic embeddings for search
```

### Processing Capabilities
- **Embedding Generation**: Multi-language sentence transformers for semantic search
- **Conversation Clustering**: Time-window based message grouping
- **GPU Acceleration**: CUDA support for large-scale processing
- **Batch Processing**: Configurable batch sizes for memory management

## Security Infrastructure

### Comprehensive Security Scanning
| Tool | Purpose | Scope | Output Format |
|------|---------|-------|---------------|
| **Bandit** | Code security analysis | Python source code | JSON + Text reports |
| **Safety** | Dependency vulnerability scanning | pip packages | JSON + Text reports |
| **pip-audit** | Advanced vulnerability detection | Python packages | JSON + SBOM |
| **Semgrep** | Static analysis patterns | Source code | JSON + Text reports |

### Security Features
- **Automated Daily Scans**: GitHub Actions scheduled security checks
- **Multi-format Reports**: JSON, text, and SBOM (Software Bill of Materials)
- **Thoth-aware Scanning**: Conditional scanning based on module presence
- **CI/CD Integration**: Security gates in pull requests

## Development Infrastructure

### Build System Overhaul
```makefile
Commands Reduced: 25+ → 7 essential commands
- make install     # Complete development environment setup
- make test       # Comprehensive test suite execution
- make check      # Code quality checks (black, isort, ruff, mypy)
- make security   # Full security vulnerability scanning
- make requirements # Dependency management and compilation
- make clean      # Artifact cleanup
- make show-info  # Project status and information
```

### Code Quality Standards
- **Black**: Code formatting (100 character line length)
- **isort**: Import organization with custom first-party recognition
- **Ruff**: Fast Python linting with 147 enabled rules
- **MyPy**: Static type checking with strict configuration
- **Pylint**: Additional code quality analysis

### Testing Framework
- **Pytest**: Primary testing framework with coverage reporting
- **Test Discovery**: Automatic test directory detection
- **Telegram API Testing**: Specialized tests for API flood errors
- **Preprocessor Testing**: Comprehensive workflow validation

## Installation System

### One-Liner Installation
```bash
# Primary installation method
curl -fsSL https://raw.githubusercontent.com/sevapru/terrorblade/main/scripts/install.sh | bash

# With custom settings
INSTALL_DIR="$HOME/custom-terrorblade" curl -fsSL https://raw.githubusercontent.com/sevapru/terrorblade/main/scripts/install.sh | bash
```

### Cross-Platform Support
| Platform | Package Manager | Status |
|----------|----------------|--------|
| **Ubuntu/Debian** | apt | ✅ Supported |
| **CentOS/RHEL/Fedora** | dnf/yum | ✅ Supported |
| **macOS** | Homebrew | ✅ Supported |
| **Windows** | WSL2 | ✅ Supported |

### Dependency Management
- **UV Package Manager**: Fast dependency resolution and installation
- **Platform-specific Compilation**: Requirements compiled per target platform
- **CUDA Support**: Optional GPU acceleration packages
- **Development Dependencies**: Separate dev requirements with testing tools

## Configuration Management

### Environment Configuration
```bash
# Required for Telegram API access
API_ID=your_telegram_api_id
API_HASH=your_telegram_api_hash

# Optional configurations
LOG_LEVEL=INFO
LOG_FILE=telegram_preprocessor.log
LOG_DIR=logs
USE_CUDA=false
```

### Database Configuration
- **Default Database**: `telegram_data.db` in current directory
- **Custom Paths**: User-configurable database locations
- **User Tables**: Phone number-based table naming convention
- **Schema Migration**: Automatic table initialization and updates

## API Reference

### Core Classes
```python
# Primary data processing
from terrorblade.data.preprocessing.TelegramPreprocessor import TelegramPreprocessor
from terrorblade.data.database.telegram_database import TelegramDatabase

# JSON archive processing
from terrorblade.examples.create_db_from_tg_json import create_db_from_telegram_json

# Direct API processing
from terrorblade.examples.create_db_from_tg_account import run_processor
```

### Database Schema
```sql
-- Core tables created per user
users (
    phone VARCHAR PRIMARY KEY,
    last_update TIMESTAMP,
    first_seen TIMESTAMP
)

messages_{phone} (
    message_id BIGINT,
    chat_id BIGINT,
    date TIMESTAMP,
    text TEXT,
    from_id BIGINT,
    reply_to_message_id BIGINT,
    media_type TEXT,
    file_name TEXT,
    from_name TEXT,
    chat_name TEXT,
    forwarded_from TEXT
)
```

## Performance Specifications

### Processing Capabilities
- **Batch Size**: 2000 messages per batch (configurable)
- **Embedding Model**: paraphrase-multilingual-mpnet-base-v2
- **GPU Support**: CUDA acceleration for embedding generation
- **Memory Management**: Efficient Polars DataFrame operations
- **Database Engine**: DuckDB for high-performance analytics

### Scalability Features
- **Incremental Processing**: Only process new messages
- **Time-window Clustering**: Configurable conversation grouping (default: 5 minutes)
- **Large Dataset Support**: Optimized for extensive message histories
- **Rate Limiting**: Built-in protection against API limits

## Breaking Changes

### Project Focus Shift
- **From**: "AI platform for data analysis and machine learning"
- **To**: "Data extraction and parsing platform for messaging platforms"

### Module Responsibility
- **Terrorblade**: Now exclusively responsible for data extraction and parsing
- **Analytics Features**: Moved to upcoming Thoth module
- **Security Scanning**: Enhanced but focused on code security, not content analysis

### Development Phases Update
| Phase | Previous Focus | New Focus |
|-------|---------------|-----------|
| **Phase 1** | Core Analytics | Data Ingestion & Processing |
| **Phase 2** | Security & Observation | Multi-Platform Expansion |
| **Terrorblade Role** | Behavioral pattern extraction | Data extraction and parsing |

## File Structure Changes

### Simplified Makefile System
```
make/
├── common.mk       # Shared utilities and color definitions
├── requirements.mk # Dependency management (60% size reduction)
├── security.mk     # Security scanning (62% size reduction)
└── test.mk         # Testing framework (simplified structure)
```

### Documentation Organization
```
fun/generated/v0.2.0/
├── INSTALL.md              # Comprehensive installation guide
├── REQUIREMENTS.md         # Dependency specifications
├── SIMPLIFICATION_SUMMARY.md # Build system improvements
└── semgrep_report.txt      # Security analysis results
```

## Known Issues and Limitations

### Telegram API Limitations
- **Account Risk**: Direct API usage may trigger rate limits on large accounts
- **Message Deletion**: API failures may cause recent message deletion
- **Authentication Required**: Phone number and API credentials needed

### Thoth Module Status
- **Status**: In development, not included in v0.2.0
- **Integration**: Conditional installation based on directory presence
- **Future Release**: Planned for upcoming versions

## Migration Guide

### From v0.1.0 to v0.2.0
1. **Update Installation**:
   ```bash
   curl -fsSL https://raw.githubusercontent.com/sevapru/terrorblade/main/scripts/install.sh | bash
   ```

2. **Update Usage Patterns**:
   ```python
   # New JSON processing (recommended)
   from terrorblade.examples.create_db_from_tg_json import create_db_from_telegram_json
   create_db_from_telegram_json("1234567890", "/path/to/result.json")
   
   # API processing (use with caution)
   from terrorblade.examples.create_db_from_tg_account import run_processor
   run_processor("+1234567890")
   ```

3. **Database Schema**: Automatic migration on first run

## Technical Specifications

### Dependencies
```toml
Core Dependencies:
- polars (DataFrame operations)
- telethon (Telegram API client)
- duckdb (Database engine)
- sentence-transformers (Embedding generation)
- python-dotenv (Configuration management)

Development Dependencies:
- pytest + pytest-cov (Testing)
- black + isort + ruff + mypy (Code quality)
- bandit + safety + pip-audit + semgrep (Security)
```

### System Requirements
- **Python**: 3.9+ (3.12+ recommended)
- **Memory**: 2GB+ RAM for processing
- **Storage**: 2GB+ free disk space
- **GPU**: Optional CUDA-compatible GPU for acceleration

## Roadmap

### Immediate Next Steps (Q1 2025)
- **Thoth Module**: Topic analysis and visualization
- **WhatsApp Support**: Multi-platform data extraction
- **Enhanced Security**: Content analysis and pattern detection

### Long-term Vision (2025)
- **Argus Module**: Multi-platform monitoring (WhatsApp, VK, Instagram, Facebook)
- **Themis/Nemesis**: Advanced analytics and pattern detection
- **Iris**: Data visualization and interactive dashboards

## Acknowledgments

This release represents a fundamental shift toward establishing Terrorblade as a robust data infrastructure platform. The focus on data extraction, standardization, and security provides a solid foundation for future analytics modules.

---

**Installation**: `curl -fsSL https://raw.githubusercontent.com/sevapru/terrorblade/main/scripts/install.sh | bash`  
**Documentation**: [GitHub Repository](https://github.com/sevapru/terrorblade)  
**Issues**: [GitHub Issues](https://github.com/sevapru/terrorblade/issues) 