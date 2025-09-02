# Terrorblade TUI - Modular Architecture

This directory contains the optimized, modular implementation of the Terrorblade Cluster Analysis TUI.

## Architecture Overview

The TUI has been refactored into separate, focused modules:

```
terrorblade/tui/
├── modules/
│   ├── __init__.py              # Module exports
│   ├── chat_selector.py         # Chat loading, filtering, selection
│   ├── message_analyzer.py      # Cluster analysis, summarization
│   ├── logging_terminal.py      # Terminal logging component
│   └── optimized_tui.py         # Main TUI orchestrating modules
└── README.md                    # This file
```

## Modules

### ChatSelector
**File:** `chat_selector.py`

Handles all chat-related operations:
- Loading chats from database
- Filtering and searching chats
- Chat selection management
- UI updates for chat list

**Key Methods:**
- `load_chats()` - Load chats from database
- `search_chats()` - Filter chats by search term
- `select_chat_by_index()` - Select chat by list position
- `update_chat_list_display()` - Update UI with filtered chats

### MessageAnalyzer
**File:** `message_analyzer.py`

Handles all message analysis operations:
- Loading clusters for chats
- Analyzing cluster details
- Generating AI summaries
- Database operations for summaries

**Key Methods:**
- `load_clusters()` - Load clusters for a chat
- `analyze_cluster()` - Analyze cluster details
- `generate_summary()` - Generate AI summary
- `select_cluster_by_index()` - Select cluster by table position

### LoggingTerminal
**File:** `logging_terminal.py`

Provides terminal-like logging display:
- Captures Python logging messages
- Displays last N messages in UI
- Automatically formats and truncates messages
- Thread-safe message handling

**Key Methods:**
- `add_message()` - Add new log message
- `get_terminal_widget()` - Get widget for UI placement
- `clear()` - Clear all messages

### OptimizedClusterAnalysisTUI
**File:** `optimized_tui.py`

Main TUI application that orchestrates the modules:
- Initializes and coordinates modules
- Handles UI layout and events
- Provides user interaction flow
- Manages application lifecycle

## Usage

The modular TUI is used through the main entry point:

```bash
# Run the optimized TUI
python terrorblade/examples/cluster_analysis_simple_tui.py --phone +79992004210
```

## Key Improvements

1. **Separation of Concerns:** Each module has a single, well-defined responsibility
2. **Better Maintainability:** Smaller, focused modules are easier to modify and test
3. **Improved Reusability:** Modules can be reused independently
4. **Enhanced Logging:** Real-time logging terminal shows application status
5. **Cleaner Code Flow:** Optimized event handling and data flow
6. **Stable Quality:** Better error handling and state management

## Logging Terminal

The logging terminal at the bottom shows the last 3 log messages:
- Automatically captures Python logging
- Shows INFO, WARNING, and ERROR messages
- Messages are truncated for readability
- Real-time updates during application execution

## Event Flow

1. **Startup:** TUI initializes modules and loads chats
2. **Chat Selection:** User selects chat → ChatSelector updates state
3. **Cluster Loading:** User clicks "Load Clusters" → MessageAnalyzer loads clusters
4. **Cluster Selection:** User selects cluster → Auto-analyzes cluster
5. **Summary Generation:** User generates summary → MessageAnalyzer handles AI processing
6. **Data Persistence:** User saves summary → Database operations handled by MessageAnalyzer

## Dependencies

The modular architecture requires the same dependencies as the original TUI:

- textual
- polars
- openai (optional, for AI summaries)
- terrorblade core modules

## Configuration

Configuration is handled through command-line arguments:

- `--phone`: Phone number for database access
- `--db-path`: Database path (default: auto)

## Error Handling

Each module includes comprehensive error handling:
- Database connection errors
- Missing data scenarios
- Network/API failures
- Invalid user selections

Errors are logged and displayed both in the logging terminal and as user notifications.
