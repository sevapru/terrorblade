"""
Chat Selection Module for Terrorblade TUI

Handles chat loading, filtering, and selection operations.
"""

import logging
from collections.abc import Callable
from typing import Any

from textual.containers import Container
from textual.widgets import Label, ListItem, ListView

logger = logging.getLogger(__name__)


class ChatSelector:
    """Handles chat selection and filtering operations."""

    def __init__(self, analyzer: Any, on_chat_selected: Callable[[dict[str, Any]], None]) -> None:
        self.analyzer = analyzer
        self.on_chat_selected = on_chat_selected
        self.chats_data: list[dict[str, Any]] = []
        self.filtered_chats: list[dict[str, Any]] = []
        self.current_chat: dict[str, Any] | None = None

    def get_ui_components(self) -> list[Container]:
        """Return the UI components for chat selection."""
        return [
            Container(classes="search-box"),
            Container(classes="chat-list-container"),
            Container(classes="button-bar"),
        ]

    def load_chats(self, callback: Callable[[list[dict[str, Any]]], None]) -> None:
        """Load chats from database - to be called from main thread."""
        if not self.analyzer:
            logger.error("Analyzer not initialized")
            return

        try:
            logger.info("Loading chats from database...")
            chats_df = self.analyzer.get_chats_list()

            if len(chats_df) > 0:
                chats_data = chats_df.to_dicts()
                logger.info(f"Loaded {len(chats_data)} chats")
                callback(chats_data)
            else:
                logger.warning("No chats found in database")
                callback([])
        except Exception as e:
            logger.error(f"Error loading chats: {e}")
            callback([])

    def update_chats_data(self, chats_data: list[dict[str, Any]]) -> None:
        """Update internal chat data and UI."""
        self.chats_data = chats_data
        self.filtered_chats = chats_data.copy()

        # Don't auto-select to avoid recursion
        # Let the UI handle selection explicitly

    def search_chats(self, search_term: str) -> None:
        """Filter chats based on search term."""
        if not search_term:
            self.filtered_chats = self.chats_data.copy()
        else:
            self.filtered_chats = []
            for chat in self.chats_data:
                chat_name = chat.get("chat_name", "").lower()
                chat_id = str(chat.get("chat_id", "")).lower()

                if search_term.lower() in chat_name or search_term.lower() in chat_id:
                    self.filtered_chats.append(chat)

        logger.info(f"Filtered to {len(self.filtered_chats)} chats")

    def update_chat_list_display(self, chat_list: ListView) -> None:
        """Update the chat list display with current filtered chats."""
        try:
            logger.info(f"Updating chat list with {len(self.filtered_chats)} chats")
            chat_list.clear()

            for _i, chat in enumerate(self.filtered_chats):
                chat_name = chat.get("chat_name", "Unknown")
                message_count = chat.get("message_count", 0)
                participant_count = chat.get("participant_count", 0)

                # Add indicator for currently selected chat
                if self.current_chat and chat.get("chat_id") == self.current_chat.get("chat_id"):
                    display_text = f"🔸 {chat_name} ({message_count:,} msgs, {participant_count} users)"
                else:
                    display_text = f"{chat_name} ({message_count:,} msgs, {participant_count} users)"

                chat_list.append(ListItem(Label(display_text, classes="chat-item")))

            logger.info(f"Successfully added {len(self.filtered_chats)} items to chat list")
        except Exception as e:
            logger.error(f"Error updating chat list: {e}")

    def select_chat_by_index(self, index: int) -> dict[str, Any] | None:
        """Select a chat by list index."""
        if index is not None and index < len(self.filtered_chats):
            self.current_chat = self.filtered_chats[index]
            logger.info(f"Chat selected via index: {self.current_chat.get('chat_name', 'Unknown')} (index: {index})")
            return self.current_chat
        else:
            logger.warning(f"Invalid chat selection - index: {index}, filtered count: {len(self.filtered_chats)}")
            return None

    def get_current_chat(self) -> dict[str, Any] | None:
        """Get the currently selected chat."""
        return self.current_chat

    def get_filtered_chats(self) -> list[dict[str, Any]]:
        """Get the filtered chats list."""
        return self.filtered_chats
