"""
Message Analysis Module for Terrorblade TUI

Handles cluster analysis, summarization, and database operations.
"""

import logging
import os
from collections.abc import Callable
from importlib.util import find_spec
from typing import Any

from textual.widgets import DataTable

logger = logging.getLogger(__name__)

# Check for optional dependencies
OPENAI_AVAILABLE = False
if find_spec("openai") is not None:
    OPENAI_AVAILABLE = True



class MessageAnalyzer:
    """Handles message cluster analysis and AI summarization."""

    def __init__(self, analyzer: Any) -> None:
        self.analyzer = analyzer
        self.clusters_data: list[dict[str, Any]] = []
        self.current_cluster: dict[str, Any] | None = None

    def load_clusters(self, chat_id: int, callback: Callable[[list[dict[str, Any]]], None]) -> None:
        """Load clusters for a specific chat - to be called from main thread."""
        if not self.analyzer:
            logger.error("Analyzer not initialized")
            return

        try:
            logger.info(f"Loading clusters for chat_id: {chat_id}")
            clusters_df = self.analyzer.get_large_clusters(chat_id=chat_id, min_size=10)

            if len(clusters_df) > 0:
                clusters_data = clusters_df.to_dicts()
                logger.info(f"Loaded {len(clusters_data)} clusters")
                callback(clusters_data)
            else:
                logger.warning("No large clusters found")
                callback([])
        except Exception as e:
            logger.error(f"Error loading clusters: {e}")
            callback([])

    def update_clusters_data(self, clusters_data: list[dict[str, Any]]) -> None:
        """Update internal clusters data."""
        self.clusters_data = clusters_data

    def update_cluster_table(self, cluster_table: DataTable) -> None:
        """Update the cluster table display with current clusters."""
        try:
            logger.info(f"Updating clusters table with {len(self.clusters_data)} clusters")
            cluster_table.clear(columns=True)

            # Add columns (simple format for compatibility)
            cluster_table.add_columns(
                "Group ID",
                "Chat Name",
                "Messages",
                "Participants",
                "Start Time",
                "Duration (h)",
                "Messages/h",
                "Intensity",
            )

            # Add rows with safe formatting
            for cluster in self.clusters_data:
                try:
                    # Ensure all values are strings and handle None values
                    group_id = str(cluster.get("group_id", "N/A"))
                    chat_name = str(cluster.get("chat_name", "Unknown"))[:30]
                    message_count = f"{cluster.get('message_count', 0):,}"
                    participant_count = str(cluster.get("participant_count", 0))
                    start_time = str(cluster.get("start_time_str", "N/A"))[:20]
                    duration_hours = f"{float(cluster.get('duration_hours', 0)):.1f}"
                    messages_per_hour = f"{float(cluster.get('messages_per_hour', 0)):.1f}"
                    intensity = str(cluster.get("intensity", "N/A"))[:10]

                    cluster_table.add_row(
                        group_id,
                        chat_name,
                        message_count,
                        participant_count,
                        start_time,
                        duration_hours,
                        messages_per_hour,
                        intensity,
                    )
                except Exception as e:
                    logger.warning(f"Error adding cluster row: {e}")

            logger.info(f"Successfully added {len(self.clusters_data)} rows to cluster table")
        except Exception as e:
            logger.error(f"Error updating clusters table: {e}")

    def analyze_cluster(self, chat_id: int, group_id: int, callback: Callable[[str], None]) -> None:
        """Analyze a specific cluster - to be called from main thread."""
        if not self.analyzer:
            logger.error("Analyzer not initialized")
            return

        try:
            logger.info(f"Analyzing cluster {group_id} in chat {chat_id}")
            stats = self.analyzer.analyze_cluster_details(chat_id, group_id)

            if "error" not in stats:
                # Format analysis for display
                analysis_text = self._format_cluster_analysis(stats)
                callback(analysis_text)
                logger.info("Cluster analysis completed")
            else:
                error_msg = stats["error"]
                logger.error(f"Error analyzing cluster: {error_msg}")
                callback(f"❌ Error analyzing cluster: {error_msg}")
        except Exception as e:
            logger.error(f"Error analyzing cluster: {e}")
            callback(f"❌ Error analyzing cluster: {e}")

    def _format_cluster_analysis(self, stats: dict[str, Any]) -> str:
        """Format cluster analysis data for display."""
        analysis_text = f"""
📊 Cluster Analysis - Group {stats["cluster_id"]} in {stats["chat_name"]}
{"=" * 60}

📝 Messages: {stats["message_count"]:,}
👥 Participants: {stats["participant_count"]}
⏰ Duration: {stats["duration_hours"]:.2f} hours
🚀 Intensity: {stats["messages_per_hour"]:.1f} messages/hour
🕐 Period: {stats["start_time"]} → {stats["end_time"]}

👥 Participant Breakdown:
"""

        # Add participant breakdown
        if "participants" in stats and len(stats["participants"]) > 0:
            for _, row in stats["participants"].iter_rows(named=True):
                analysis_text += f"  {row['from_name']}: {row['message_count']} messages\n"

        return analysis_text

    def generate_summary(self, chat_id: int, group_id: int, callback: Callable[[str], None]) -> None:
        """Generate AI summary for a cluster - to be called from main thread."""
        if not OPENAI_AVAILABLE:
            logger.warning("OpenAI not available")
            callback("⚠️ OpenAI not available. Install with: pip install openai")
            return

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not found in environment")
            callback("⚠️ OPENAI_API_KEY not found in environment variables")
            return

        if not self.analyzer:
            logger.error("Analyzer not initialized")
            callback("❌ Analyzer not initialized")
            return

        try:
            logger.info(f"Generating summary for cluster {group_id} in chat {chat_id}")

            # Get cluster text for summarization
            cluster_text = self.analyzer.get_cluster_summary_data(chat_id, group_id)

            # Generate summary
            from terrorblade.examples.cluster_analysis_cli import summarize_cluster_with_openai

            summary = summarize_cluster_with_openai(cluster_text, api_key)

            callback(summary)
            logger.info("Summary generated successfully")
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            callback(f"❌ Error generating summary: {e}")

    def select_cluster_by_index(self, index: int) -> dict[str, Any] | None:
        """Select a cluster by table index."""
        if index is not None and index < len(self.clusters_data):
            self.current_cluster = self.clusters_data[index]
            logger.info(f"Cluster selected: {self.current_cluster.get('group_id', 'Unknown')}")
            return self.current_cluster
        else:
            logger.warning(f"Invalid cluster selection - index: {index}, clusters count: {len(self.clusters_data)}")
            return None

    def get_current_cluster(self) -> dict[str, Any] | None:
        """Get the currently selected cluster."""
        return self.current_cluster

    def sort_clusters_by(self, sort_key: str) -> None:
        """Sort clusters by the specified key (highest first)."""
        try:
            if sort_key == "messages_per_hour":
                self.clusters_data.sort(key=lambda x: float(x.get("messages_per_hour", 0)), reverse=True)
            elif sort_key == "intensity":
                def intensity_to_number(intensity_str: str) -> int:
                    if isinstance(intensity_str, str):
                        if "Very High" in intensity_str:
                            return 4
                        elif "High" in intensity_str:
                            return 3
                        elif "Medium" in intensity_str:
                            return 2
                        elif "Low" in intensity_str:
                            return 1
                        else:
                            return 0
                    return 0

                self.clusters_data.sort(key=lambda x: intensity_to_number(x.get("intensity", "")), reverse=True)
            else:
                # Default: sort by message count
                self.clusters_data.sort(key=lambda x: int(x.get("message_count", 0)), reverse=True)

            logger.info(f"Sorted clusters by {sort_key}")
        except Exception as e:
            logger.error(f"Error sorting clusters: {e}")

    def get_clusters_data(self) -> list[dict[str, Any]]:
        """Get the clusters data."""
        return self.clusters_data
