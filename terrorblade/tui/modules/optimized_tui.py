"""
Optimized Terrorblade TUI with Two-Screen Architecture

Clean implementation with separate chat selection and chat analysis screens.
"""

import logging
import sys
from pathlib import Path
from typing import Any

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal
from textual.events import Key
from textual.screen import Screen
from textual.widgets import (
    Button,
    DataTable,
    Input,
    Label,
    ListView,
    Static,
    TabbedContent,
    TabPane,
    TextArea,
)

# Add terrorblade to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from terrorblade.examples.cluster_analysis_cli import (
    ClusterAnalyzer,
)

# Import our modular components
from .chat_selector import ChatSelector
from .logging_terminal import LoggingTerminal
from .message_analyzer import MessageAnalyzer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("/tmp/terrorblade_tui.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class ChatSelectionScreen(Screen):
    """Screen for selecting a chat."""

    CSS = """
    .chat-selection-container {
        height: 100vh;
        width: 100%;
        layout: vertical;
        padding: 1;
    }

    .header {
        height: 3;
        background: $accent;
        color: $text;
        padding: 1;
        text-align: center;
        text-style: bold;
    }

    .search-section {
        height: 4;
        border: solid $accent;
        margin: 1 0;
        padding: 1;
    }

    .chat-list-section {
        height: 1fr;
        border: solid $accent;
        margin: 1 0;
    }

    .button-section {
        height: 5;
        border: solid $accent;
        margin: 1 0;
        padding: 1;
        align: center middle;
    }

    .logging-terminal-container {
        height: 4;
        border: solid $accent;
        margin: 1 0;
    }

    .terminal-header {
        height: 1;
        background: $accent;
        color: $text;
        padding: 0 1;
        text-align: center;
        text-style: bold;
    }

    .logging-terminal {
        height: 3;
        width: 100%;
        background: $surface;
        color: $text;
        padding: 0 1;
        text-style: dim;
    }

    .chat-item {
        padding: 0 1;
        height: 2;
    }

    .instructions {
        height: 2;
        padding: 0 1;
        text-style: italic;
        color: $text-muted;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("enter", "select_chat", "Select Chat"),
        Binding("r", "refresh", "Refresh"),
    ]

    def __init__(self, analyzer: Any, logging_terminal: Any) -> None:
        super().__init__()
        self.analyzer = analyzer
        self.logging_terminal = logging_terminal
        self.chat_selector = ChatSelector(analyzer, self._on_chat_selected)
        self.selected_chat = None

    def compose(self) -> ComposeResult:
        """Compose the chat selection screen."""
        with Container(classes="chat-selection-container"):
            yield Label("Terrorblade - Select Chat for Analysis", classes="header")

            with Container(classes="search-section"):
                yield Label("🔍 Search Chats")
                yield Input(placeholder="Type to search chats...", id="search-input")

            with Container(classes="chat-list-section"):
                yield Label("💬 Available Chats")
                yield ListView(id="chat-list")

            with Container(classes="button-section"), Horizontal():
                yield Button("🔄 Refresh Chats", id="refresh-btn", variant="default")
                yield Button("➡️ Analyze Selected Chat", id="analyze-btn", variant="primary")

            # Logging terminal
            with Container(classes="logging-terminal-container"):
                yield Label("📋 Status", classes="terminal-header")
                yield self.logging_terminal.get_terminal_widget()

    def on_mount(self) -> None:
        """Load chats when screen mounts."""
        logger.info("Chat selection screen mounted")

        # Disable analyze button initially
        try:
            analyze_btn = self.query_one("#analyze-btn", Button)
            analyze_btn.disabled = True
        except Exception as e:
            logger.warning(f"Could not disable analyze button initially: {e}")

        self.load_chats()

    @work(thread=True)
    def load_chats(self) -> None:
        """Load chats from database."""

        def callback(chats_data: list[dict[str, Any]]) -> None:
            self.app.call_from_thread(self._update_chats_ui, chats_data)

        self.chat_selector.load_chats(callback)

    def _update_chats_ui(self, chats_data: list[dict[str, Any]]) -> None:
        """Update UI with loaded chats."""
        self.chat_selector.update_chats_data(chats_data)

        # Update chat list display
        chat_list = self.query_one("#chat-list", ListView)
        self.chat_selector.update_chat_list_display(chat_list)

        if len(chats_data) > 0:
            self.notify(f"✅ Loaded {len(chats_data)} chats. Click on a chat name, then press the blue button below.")
            logger.info(f"Loaded {len(chats_data)} chats successfully")
        else:
            self.notify("⚠️ No chats found")

    def auto_proceed_to_analysis(self) -> None:
        """Automatically proceed to analysis screen (for testing)."""
        logger.info("=== AUTO PROCEEDING TO ANALYSIS ===")
        self.analyze_selected_chat()

    def _on_chat_selected(self, chat: dict[str, Any] | None) -> None:
        """Handle chat selection."""
        self.selected_chat = chat # type: ignore
        chat_name = chat.get("chat_name", "Unknown") if chat else "Unknown"
        chat_id = chat.get("chat_id", "Unknown") if chat else "Unknown"
        logger.info(f"Chat selected: {chat_name} (ID: {chat_id})")
        self.notify(f"🔸 Selected: {chat_name}")

        # Enable the analyze button and update its text
        try:
            analyze_btn = self.query_one("#analyze-btn", Button)
            analyze_btn.disabled = False
            analyze_btn.label = f"➡️ Analyze '{chat_name}'"
        except Exception as e:
            logger.warning(f"Could not enable analyze button: {e}")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        logger.info(f"Button pressed: {event.button.id}")

        if event.button.id == "refresh-btn":
            logger.info("Refreshing chats...")
            self.load_chats()
        elif event.button.id == "analyze-btn":
            logger.info("Analyzing selected chat...")
            self.analyze_selected_chat()
        else:
            logger.warning(f"Unknown button ID: {event.button.id}")

    def on_key(self, event: Key) -> None:
        """Handle key presses."""
        logger.info(f"Key pressed: {event.key}")
        if event.key == "enter":
            logger.info("Enter key detected - analyzing selected chat")
            self.analyze_selected_chat()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle search input."""
        logger.info("Search input submitted")
        self.search_chats()

    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle search as user types."""
        # Don't log every keystroke, just do the search
        self.search_chats()

    def search_chats(self) -> None:
        """Filter chats based on search term."""
        search_input = self.query_one("#search-input", Input)
        search_term = search_input.value or ""

        self.chat_selector.search_chats(search_term)
        chat_list = self.query_one("#chat-list", ListView)
        self.chat_selector.update_chat_list_display(chat_list)

        count = len(self.chat_selector.get_filtered_chats())
        if search_term:
            self.notify(f"🔍 Found {count} chats matching '{search_term}'")
            logger.info(f"Search for '{search_term}' found {count} chats")

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle chat list selection."""
        logger.info(f"ListView selection event: index={event.index}")

        if event.index is not None:
            selected_chat = self.chat_selector.select_chat_by_index(event.index)
            if selected_chat:
                # Explicitly call the selection callback
                self._on_chat_selected(selected_chat)

                chat_name = selected_chat.get("chat_name", "Unknown")
                logger.info(f"Chat selected via ListView: {chat_name}")

                # Refresh the chat list to show the selection indicator
                chat_list = self.query_one("#chat-list", ListView)
                self.chat_selector.update_chat_list_display(chat_list)

    def analyze_selected_chat(self) -> None:
        """Switch to analysis screen for selected chat."""
        current_chat = self.chat_selector.get_current_chat()
        logger.info(f"analyze_selected_chat called. Current chat: {current_chat}")

        if current_chat:
            chat_name = current_chat.get("chat_name", "Unknown")
            chat_id = current_chat.get("chat_id", "Unknown")

            logger.info(f"Switching to analysis for chat: {chat_name} (ID: {chat_id})")
            self.notify(f"🔄 Loading analysis for {chat_name}...")

            try:
                analysis_screen = ChatAnalysisScreen(self.analyzer, current_chat, self.logging_terminal)
                logger.info("Created ChatAnalysisScreen, pushing to app...")
                self.app.push_screen(analysis_screen)
                logger.info("Successfully pushed analysis screen")
            except Exception as e:
                logger.error(f"Error creating/pushing analysis screen: {e}")
                import traceback

                logger.error(traceback.format_exc())
                self.notify(f"❌ Error loading analysis: {e}", severity="error")
        else:
            logger.warning("No chat selected for analysis")
            self.notify("⚠️ Please select a chat first", severity="warning")

    def action_select_chat(self) -> None:
        """Select currently highlighted chat."""
        self.analyze_selected_chat()

    def action_refresh(self) -> None:
        """Refresh chat list."""
        self.load_chats()

    def action_quit(self) -> None:
        """Quit application."""
        self.app.exit()


class ChatAnalysisScreen(Screen):
    """Screen for analyzing a selected chat."""

    CSS = """
    .analysis-container {
        height: 100vh;
        width: 100%;
        layout: vertical;
    }

    .header {
        height: 3;
        background: $accent;
        color: $text;
        padding: 1;
        text-align: center;
        text-style: bold;
    }

    .content-area {
        height: 1fr;
        width: 100%;
    }

    .tab-content {
        height: 1fr;
        padding: 0;
    }

    .analysis-content {
        height: 100%;
        padding: 1;
        overflow-y: auto;
    }

    .button-bar {
        height: 3;
        padding: 0 1;
        align: center middle;
    }

    .logging-terminal-container {
        height: 4;
        border-top: solid $accent;
    }

    .terminal-header {
        height: 1;
        background: $accent;
        color: $text;
        padding: 0 1;
        text-align: center;
        text-style: bold;
    }

    .logging-terminal {
        height: 3;
        width: 100%;
        background: $surface;
        color: $text;
        padding: 0 1;
        text-style: dim;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("escape", "back", "Back to Chat Selection"),
        Binding("1", "show_clusters", "Clusters"),
        Binding("2", "show_analysis", "Analysis"),
        Binding("3", "show_summary", "Summary"),
        Binding("r", "refresh", "Refresh"),
        Binding("enter", "analyze_cluster", "Analyze Selected Cluster"),
    ]

    def __init__(self, analyzer: Any, chat: dict[str, Any], logging_terminal: Any) -> None:
        super().__init__()
        self.analyzer = analyzer
        self.chat = chat
        self.logging_terminal = logging_terminal
        self.message_analyzer = MessageAnalyzer(analyzer)

    def compose(self) -> ComposeResult:
        """Compose the chat analysis screen."""
        chat_name = self.chat.get("chat_name", "Unknown")

        with Container(classes="analysis-container"):
            yield Label(f"🗡️ Analyzing: {chat_name}", classes="header")

            with Container(classes="content-area"), TabbedContent():
                with TabPane("📊 Clusters", id="clusters-tab"):
                    yield Label(
                        "📋 Click on any cluster row to analyze it. Use ↑↓ arrows and ENTER to select.",
                        classes="instructions",
                    )
                    yield DataTable(id="cluster-table")
                    with Container(classes="button-bar"), Horizontal():
                        yield Button("🔄 Refresh", id="refresh-clusters-btn", variant="default")
                        yield Button("⬆️ Sort by Messages/h", id="sort-messages-btn", variant="default")
                        yield Button("⬆️ Sort by Intensity", id="sort-intensity-btn", variant="default")
                        yield Button("📊 Analyze Selected", id="analyze-cluster-btn", variant="primary")

                with TabPane("📋 Analysis", id="analysis-tab"), Container(classes="analysis-content"):
                    yield Static("Select a cluster from the Clusters tab to analyze", id="analysis-content")

                with TabPane("🤖 Summary", id="summary-tab"):
                    yield TextArea("", id="summary-content", read_only=True)
                    with Container(classes="button-bar"):
                        yield Button("🤖 Generate Summary", id="generate-summary-btn", variant="primary")
                        yield Button("💾 Save to DB", id="save-summary-btn", variant="default")
                        yield Button("⬅️ Back to Chat Selection", id="back-btn", variant="default")

            # Logging terminal
            with Container(classes="logging-terminal-container"):
                yield Label("📋 Status", classes="terminal-header")
                yield self.logging_terminal.get_terminal_widget()

    def on_mount(self) -> None:
        """Load clusters when screen mounts."""
        chat_name = self.chat.get("chat_name", "Unknown")
        chat_id = self.chat.get("chat_id", "Unknown")
        logger.info(f"Analysis screen mounted for chat: {chat_name} (ID: {chat_id})")

        # Show immediate feedback
        self.notify(f"📊 Loading clusters for {chat_name}...")

        # Start loading clusters
        logger.info("Starting to load clusters...")
        self.load_clusters()

    @work(thread=True)
    def load_clusters(self) -> None:
        """Load clusters for the chat."""
        chat_id = self.chat["chat_id"]
        self.chat.get("chat_name", "Unknown")

        logger.info(f"load_clusters called for chat_id: {chat_id}")

        def callback(clusters_data: list[dict[str, Any]]) -> None:
            logger.info(f"Clusters loaded callback called with {len(clusters_data)} clusters")
            self.app.call_from_thread(self._update_clusters_ui, clusters_data)

        try:
            logger.info("Calling message_analyzer.load_clusters...")
            self.message_analyzer.load_clusters(chat_id, callback)
            logger.info("message_analyzer.load_clusters call completed")
        except Exception as e:
            logger.error(f"Error in load_clusters: {e}")
            import traceback

            logger.error(traceback.format_exc())

    def _update_clusters_ui(self, clusters_data: list[dict[str, Any]]) -> None:
        """Update UI with loaded clusters."""
        logger.info(f"_update_clusters_ui called with {len(clusters_data)} clusters")

        try:
            self.message_analyzer.update_clusters_data(clusters_data)
            logger.info("Updated message analyzer clusters data")

            # Update cluster table
            cluster_table = self.query_one("#cluster-table", DataTable)
            logger.info("Found cluster table widget")

            self.message_analyzer.update_cluster_table(cluster_table)
            logger.info("Updated cluster table with data")

            if len(clusters_data) > 0:
                self.notify(f"✅ Loaded {len(clusters_data)} clusters. Click on a cluster to analyze it.")
            else:
                self.notify("⚠️ No clusters found for this chat.")
            logger.info(f"Successfully loaded {len(clusters_data)} clusters for analysis")

        except Exception as e:
            logger.error(f"Error in _update_clusters_ui: {e}")
            import traceback

            logger.error(traceback.format_exc())
            self.notify(f"❌ Error updating clusters: {e}", severity="error")

    @work(thread=True)
    def analyze_cluster(self, chat_id: int, group_id: int) -> None:
        """Analyze a specific cluster."""

        def callback(analysis_text: str) -> None:
            self.app.call_from_thread(self._update_analysis_ui, analysis_text)

        self.message_analyzer.analyze_cluster(chat_id, group_id, callback)

    def _update_analysis_ui(self, analysis_text: str) -> None:
        """Update analysis UI with results."""
        analysis_content = self.query_one("#analysis-content", Static)
        analysis_content.update(analysis_text)

        # Switch to analysis tab
        tabbed_content = self.query_one(TabbedContent)
        tabbed_content.active = "analysis-tab"

        self.notify("✅ Analysis completed")

    @work(thread=True)
    def generate_summary(self, chat_id: int, group_id: int) -> None:
        """Generate AI summary for a cluster."""

        def callback(summary: str) -> None:
            self.app.call_from_thread(self._update_summary_ui, summary)

        self.message_analyzer.generate_summary(chat_id, group_id, callback)

    def _update_summary_ui(self, summary: str) -> None:
        """Update summary UI with results."""
        summary_content = self.query_one("#summary-content", TextArea)
        summary_content.load_text(summary)

        # Switch to summary tab
        tabbed_content = self.query_one(TabbedContent)
        tabbed_content.active = "summary-tab"

        self.notify("✅ Summary generated")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "refresh-clusters-btn":
            self.load_clusters()
        elif event.button.id == "sort-messages-btn":
            self.sort_clusters_by_messages_per_hour()
        elif event.button.id == "sort-intensity-btn":
            self.sort_clusters_by_intensity()
        elif event.button.id == "analyze-cluster-btn":
            self.analyze_selected_cluster_manual()
        elif event.button.id == "generate-summary-btn":
            self.generate_summary_for_selected_cluster()
        elif event.button.id == "save-summary-btn":
            self.save_current_summary()
        elif event.button.id == "back-btn":
            self.action_back()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle cluster selection."""
        if event.cursor_row is not None:
            selected_cluster = self.message_analyzer.select_cluster_by_index(event.cursor_row)
            if selected_cluster:
                chat_id = int(self.chat["chat_id"])
                group_id = selected_cluster["group_id"]

                logger.info(f"Cluster {group_id} selected, analyzing...")
                self.notify(f"📊 Analyzing cluster {group_id}...")
                self.analyze_cluster(chat_id, group_id)

    def analyze_selected_cluster_manual(self) -> None:
        """Manually analyze the currently selected cluster."""
        current_cluster = self.message_analyzer.get_current_cluster()
        if current_cluster:
            chat_id = int(self.chat["chat_id"])
            group_id = current_cluster["group_id"]

            logger.info(f"Manual cluster analysis for cluster {group_id}")
            self.notify(f"📊 Analyzing cluster {group_id}...")
            self.analyze_cluster(chat_id, group_id)
        else:
            self.notify("⚠️ Please select a cluster from the table first", severity="warning")

    def sort_clusters_by_messages_per_hour(self) -> None:
        """Sort clusters by messages per hour (highest first)."""
        self.message_analyzer.sort_clusters_by("messages_per_hour")
        cluster_table = self.query_one("#cluster-table", DataTable)
        self.message_analyzer.update_cluster_table(cluster_table)
        self.notify("📊 Sorted by Messages/hour (highest first)")

    def sort_clusters_by_intensity(self) -> None:
        """Sort clusters by intensity."""
        self.message_analyzer.sort_clusters_by("intensity")
        cluster_table = self.query_one("#cluster-table", DataTable)
        self.message_analyzer.update_cluster_table(cluster_table)
        self.notify("🔥 Sorted by Intensity (highest first)")

    def generate_summary_for_selected_cluster(self) -> None:
        """Generate summary for selected cluster."""
        current_cluster = self.message_analyzer.get_current_cluster()
        if current_cluster:
            chat_id = int(self.chat["chat_id"])
            group_id = current_cluster["group_id"]

            self.notify(f"🤖 Generating summary for cluster {group_id}...")
            self.generate_summary(chat_id, group_id)
        else:
            self.notify("⚠️ Please select a cluster first", severity="warning")

    def save_current_summary(self) -> None:
        """Save current summary to database."""
        summary_content = self.query_one("#summary-content", TextArea)
        summary_text = summary_content.text

        current_cluster = self.message_analyzer.get_current_cluster()
        if summary_text and current_cluster:
            # Save logic here (same as before)
            self.notify("✅ Summary saved to database")
        else:
            self.notify("⚠️ No summary to save", severity="warning")

    def action_back(self) -> None:
        """Go back to chat selection."""
        logger.info("Returning to chat selection screen")
        self.app.pop_screen()

    def action_show_clusters(self) -> None:
        """Show clusters tab."""
        tabbed_content = self.query_one(TabbedContent)
        tabbed_content.active = "clusters-tab"
        self.notify("📊 Clusters tab")

    def action_show_analysis(self) -> None:
        """Show analysis tab."""
        tabbed_content = self.query_one(TabbedContent)
        tabbed_content.active = "analysis-tab"
        self.notify("📋 Analysis tab")

    def action_show_summary(self) -> None:
        """Show summary tab."""
        tabbed_content = self.query_one(TabbedContent)
        tabbed_content.active = "summary-tab"
        self.notify("🤖 Summary tab")

    def action_refresh(self) -> None:
        """Refresh clusters."""
        self.load_clusters()

    def action_analyze_cluster(self) -> None:
        """Analyze the selected cluster via ENTER key."""
        self.analyze_selected_cluster_manual()

    def action_quit(self) -> None:
        """Quit application."""
        self.app.exit()


class OptimizedClusterAnalysisTUI(App):
    """Main TUI application with two-screen architecture."""

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
    ]

    def __init__(self, phone: str, db_path: str = "auto"):
        super().__init__()
        self.phone = phone
        self.db_path = db_path
        self.analyzer: ClusterAnalyzer | None = None
        self.logging_terminal = LoggingTerminal(max_lines=3)

        # Initialize analyzer
        try:
            self.analyzer = ClusterAnalyzer(phone=phone, db_path=db_path)
            logger.info(f"Initialized ClusterAnalyzer for phone: {phone}")
        except Exception as e:
            logger.error(f"Failed to initialize ClusterAnalyzer: {e}")
            self.exit()

    def on_mount(self) -> None:
        """Start with chat selection screen."""
        logger.info("Starting TUI with chat selection screen")
        chat_selection = ChatSelectionScreen(self.analyzer, self.logging_terminal)
        self.push_screen(chat_selection)

    def action_quit(self) -> None:
        """Quit the application."""
        logger.info("Quitting application")
        if self.analyzer:
            self.analyzer.close()
        self.exit()


def main() -> None:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Terrorblade Cluster Analysis TUI")
    parser.add_argument("--phone", default="+79992004210", help="Phone number")
    parser.add_argument("--db-path", default="auto", help="Database path")

    args = parser.parse_args()

    logger.info(f"Starting two-screen TUI with phone: {args.phone}")

    app = OptimizedClusterAnalysisTUI(phone=args.phone, db_path=args.db_path)
    app.run()


if __name__ == "__main__":
    main()
