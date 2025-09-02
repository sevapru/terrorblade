"""
Logging Terminal Component for Terrorblade TUI

Provides a terminal-like display for recent log messages.
"""

import logging
from collections import deque

from textual.widgets import Static


class LoggingTerminal:
    """A terminal-like component for displaying recent log messages."""

    def __init__(self, max_lines: int = 3):
        self.max_lines = max_lines
        self.log_messages: deque[str] = deque(maxlen=max_lines)
        self.terminal_widget: Static = Static("", classes="logging-terminal")
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Setup logging handler to capture log messages."""
        # Create formatter
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        self.log_handler = TUITextLogHandler(self)
        logging.getLogger().addHandler(self.log_handler)

    def add_message(self, message: str) -> None:
        """Add a new log message to the terminal."""
        self.log_messages.append(message)
        self._update_display()

    def _update_display(self) -> None:
        """Update the terminal display with current messages."""
        if self.terminal_widget:
            display_lines = []
            for i, msg in enumerate(self.log_messages):
                if len(msg) > 80:
                    msg = msg[:77] + "..."
                display_lines.append(f"[{i + 1}] {msg}")
            while len(display_lines) < self.max_lines:
                display_lines.append("")

            display_text = "\n".join(display_lines)
            self.terminal_widget.update(display_text)

    def get_terminal_widget(self) -> Static:
        """Return the terminal widget for direct placement."""
        return self.terminal_widget

    def clear(self) -> None:
        """Clear all log messages."""
        self.log_messages.clear()
        self._update_display()


class TUITextLogHandler(logging.Handler):
    """Custom logging handler that sends messages to the TUI terminal."""

    def __init__(self, terminal: LoggingTerminal):
        super().__init__()
        self.terminal = terminal

    def emit(self, record: logging.LogRecord) -> None:
        """Handle a log record by sending it to the terminal."""
        try:
            formatted_message = self.format(record)
            if " - " in formatted_message:
                parts = formatted_message.split(" - ", 2)
                if len(parts) >= 3:
                    formatted_message = f"{parts[1]}: {parts[2]}"
            self.terminal.add_message(formatted_message)
        except Exception:
            pass
