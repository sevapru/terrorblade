#!/usr/bin/env python3
"""
Long Message Analysis Tool for Terrorblade

Finds groups of long messages sent consecutively within time windows.
Includes quantile analysis, activity heatmaps, and search parameter recommendations.

Usage:
    python analyze_shitposts.py
    python analyze_shitposts.py --interactive
"""

import argparse
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from dotenv import load_dotenv
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from terrorblade.data.database.telegram_database import TelegramDatabase
from terrorblade.examples.prompts.promptinator import Promptinator, analyze_dialogue_with_llm
from terrorblade.examples.prompts.provider_configs import (
    PROVIDER_MODEL_COMBINATIONS,
    get_available_prompts,
    get_all_provider_model_pairs,
    format_provider_model_display,
    validate_provider_model,
    get_recommended_models
)
from terrorblade.utils.config import get_db_path

load_dotenv()

@dataclass
class Config:
    """Application configuration constants."""

    DEFAULT_MIN_WORDS: int = 10
    DEFAULT_PHONE: str = "+79992004210"
    DEFAULT_PROVIDER: str = "openrouter"
    DEFAULT_MODEL: str = "google/gemini-2.5-pro"
    DEFAULT_PROMPT: str = "prompt_1.md"

    # Display settings
    MAX_CHAT_NAME_LENGTH: int = 24
    MAX_TEXT_PREVIEW_LENGTH: int = 60
    DEFAULT_MAX_ROWS: int = 20

    # Menu options
    MENU_OPTIONS = {
        "1": "Analyze Cluster",
        "2": "Parameters",
        "3": "Word Analysis",
        "4": "Sort Groups",
        "5": "Chat Selection",
        "6": "LLM Configuration",
        "7": "Exit",
    }

    SORT_OPTIONS = {
        "1": ("total_words", "Words (descending)"),
        "2": ("message_count", "Messages (descending)"),
        "3": ("start_time", "Time (newest first)"),
    }


@dataclass
class SearchParams:
    """Search parameters structure."""

    min_words: int = Config.DEFAULT_MIN_WORDS
    min_consecutive: int = 5
    time_window_hours: int = 1
    overlap: int = 10
    chat_id: str | None = None


@dataclass
class LLMConfig:
    """LLM configuration structure."""

    provider: str = Config.DEFAULT_PROVIDER
    model: str = Config.DEFAULT_MODEL
    prompt_file: str = Config.DEFAULT_PROMPT
    temperature: float = 0.7


class LLMManager:
    """Management of LLM providers and prompts for dialogue analysis."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.promptinator: Promptinator | None = None
        self._initialize_llm()

    def _initialize_llm(self) -> None:
        """Initialize LLM provider with current configuration."""
        try:
            self.promptinator = Promptinator(
                provider=self.config.provider,
                model=self.config.model
            )
        except Exception as e:
            print(f"Warning: Failed to initialize {self.config.provider}: {e}")
            self.promptinator = None

    def update_config(self, provider: str | None = None, model: str | None = None,
                     prompt_file: str | None = None, temperature: float | None = None) -> bool:
        """Update LLM configuration and reinitialize if needed."""
        config_changed = False

        if provider and provider != self.config.provider:
            self.config.provider = provider
            config_changed = True
        if model and model != self.config.model:
            self.config.model = model
            config_changed = True
        if prompt_file:
            self.config.prompt_file = prompt_file
        if temperature is not None:
            self.config.temperature = temperature

        if config_changed:
            self._initialize_llm()

        return self.promptinator is not None

    def analyze_dialogue(self, group: dict[str, Any], messages_text: list[str]) -> str:
        """Analyze dialogue using configured LLM and prompt."""
        if not self.promptinator:
            return "Error: LLM not properly initialized. Check your API keys and configuration."

        try:
            return analyze_dialogue_with_llm(
                group_data=group,
                messages_text=messages_text,
                promptinator=self.promptinator,
                prompt_file=self.config.prompt_file
            )
        except Exception as e:
            return f"Error analyzing dialogue: {e}"

    def get_provider_info(self) -> str:
        """Get current provider information for display."""
        if not self.promptinator:
            return f"❌ {self.config.provider}:{self.config.model} (Not connected)"

        info = self.promptinator.get_provider_info()
        status = "✅" if info["available"] else "❌"
        return f"{status} {info['provider']}:{info['model']} | Prompt: {self.config.prompt_file}"


class SentimentAnalyser:
    def __init__(self, phone: str, db_path: str = "auto"):
        self.phone = phone.replace("+", "")
        self.db = TelegramDatabase(db_path=get_db_path(db_path), read_only=True)
        self.console = Console()
        self.messages_table = f"messages_{self.phone}"
        self.chat_names_table = f"chat_names_{self.phone}"
        self.user_names_table = f"user_names_{self.phone}"
        # Initialize with default LLM configuration
        self.llm_config = LLMConfig()
        self.llm_manager = LLMManager(self.llm_config)
        self.config = Config()

        # Initialize with default SearchParams
        self.params = SearchParams()

        self._cached_messages_df: pl.DataFrame | None = None
        self._cached_quantiles: dict[str, float | int] | None = None
        self._cached_chat_id: str | None = None  # Track which chat the cache is for
        self._cached_filtered_df: pl.DataFrame | None = None  # Filtered by current threshold
        self._cached_threshold: int | None = None  # Current threshold used for filtering

    def _invalidate_cache(self) -> None:
        """Invalidate cached data when parameters change."""
        self._cached_messages_df = None
        self._cached_quantiles = None
        self._cached_chat_id = None
        self._cached_filtered_df = None
        self._cached_threshold = None

    def _get_group_text_preview(self, group_id: int, all_messages_df: pl.DataFrame) -> str:
        """Get text preview from the first message in a group."""
        try:
            group_messages = all_messages_df.filter(pl.col("group_id") == group_id)
            if group_messages.is_empty():
                return "No messages available"

            # Get the first message (earliest by date)
            first_message = group_messages.sort("date").head(1)
            if first_message.is_empty():
                return "No messages available"

            text = first_message["text"][0]
            if text:
                # Clean up the text and limit length
                preview = text.replace("\n", " ").strip()
                return preview[:100] + ("..." if len(preview) > 100 else "")
            else:
                return "No text content"
        except Exception:
            return "Error getting text"

    def _get_filtered_messages_df(self, chat_id: str | None = None, threshold: int | None = None) -> pl.DataFrame:
        """Get filtered messages DataFrame, using cache when possible."""
        if threshold is None:
            threshold = self.params.min_words

        if (self._cached_filtered_df is not None and
            self._cached_threshold == threshold and
            self._cached_chat_id == chat_id):
            return self._cached_filtered_df

        # Fetch and filter data
        messages_df, _ = self.analyze_word_quantiles(chat_id=chat_id)
        filtered_df = messages_df.filter(pl.col("word_count") > threshold)

        # Cache the result
        self._cached_filtered_df = filtered_df
        self._cached_threshold = threshold
        self._cached_chat_id = chat_id

        return filtered_df

    def interactive_mode(
        self,
        groups_df: pl.DataFrame,
        all_messages_df: pl.DataFrame,
        params: SearchParams,
    ) -> None:
        while True:
            self.display_groups(groups_df, all_messages_df=all_messages_df)
            self.show_menu()
            choice = self._get_user_choice(7)

            if choice == "1":
                self._handle_cluster_analysis(groups_df, all_messages_df)
            elif choice == "2":
                groups_df, all_messages_df, params = self._handle_parameters(
                    params
                )
            elif choice == "3":
                self._handle_word_analysis()
            elif choice == "4":
                groups_df = self._handle_sorting(groups_df)
            elif choice == "5":
                groups_df, all_messages_df, params = self._handle_chat_selection(
                    groups_df, all_messages_df, params
                )
            elif choice == "6":
                self._handle_llm_configuration()
            elif choice == "7":
                break

    def analyze_word_quantiles(
        self, chat_id: str | None = None, use_cache: bool = True
    ) -> tuple[pl.DataFrame, dict[str, float]]:
        """
        Analyze word quantiles for all chats or a specific chat using optimized SQL.

        Args:
            chat_id: If provided, analyze only messages from this chat. If None, analyze all chats.
            use_cache: If True and chat_id is None, use cached results for all chats analysis.

        Returns:
            Tuple of (messages_df, quantiles_dict)
        """
        if (
            use_cache
            and self._cached_messages_df is not None
            and self._cached_quantiles is not None
            and self._cached_chat_id == chat_id  # Ensure cache is for the same chat filter
        ):
            return self._cached_messages_df, self._cached_quantiles

        try:
            chat_filter = f"AND m.chat_id = {chat_id}" if chat_id else ""
            
            # Optimized SQL with word counting done in database
            sql = f"""
            SELECT 
                m.text, 
                m.date, 
                m.chat_id::BIGINT,
                -- Optimized word counting in SQL
                CASE 
                    WHEN TRIM(m.text) = '' THEN 0
                    ELSE LENGTH(TRIM(m.text)) - LENGTH(REPLACE(TRIM(m.text), ' ', '')) + 1
                END AS word_count,
                EXTRACT(year FROM m.date) AS year,
                EXTRACT(month FROM m.date) AS month
            FROM {self.messages_table} m
            WHERE m.text IS NOT NULL 
              AND LENGTH(TRIM(m.text)) > 0 
              {chat_filter}
            ORDER BY m.date
            """

            messages_df = pl.from_arrow(self.db.db.execute(sql).arrow())
            if len(messages_df) == 0:
                return pl.DataFrame(), {}
            quantiles = {
                "min": messages_df["word_count"].min(),
                "q25": messages_df["word_count"].quantile(0.25),
                "median": messages_df["word_count"].quantile(0.5),
                "q75": messages_df["word_count"].quantile(0.75),
                "q90": messages_df["word_count"].quantile(0.9),
                "q95": messages_df["word_count"].quantile(0.95),
                "q99.7": messages_df["word_count"].quantile(0.997),
                "max": messages_df["word_count"].max(),
                "mean": messages_df["word_count"].mean(),
            }

            self._cached_messages_df = messages_df
            self._cached_quantiles = quantiles
            self._cached_chat_id = chat_id

            return messages_df, quantiles

        except Exception as e:
            self.console.print(f"Error analyzing quantiles: {e}", style="red")
            return pl.DataFrame(), {}

    def display_word_analysis(
        self,
        interactive: bool = False,
        chat_id: str | None = None,
        min_words: int = 0,
    ) -> None:
        """Display full word distribution analysis."""

        if interactive and not chat_id:
            self.console.print("Select analysis area:", style="bold cyan")
            chat_id = self._get_chat_selection()


        threshold = min_words if min_words > 0 else self.params.min_words
        messages_df = self._get_filtered_messages_df(chat_id=chat_id or None, threshold=threshold)

        if messages_df.is_empty():
            self.console.print(
                f"❌ No messages with minimum {threshold} words",
                style="red",
            )
            return

        self.create_activity_heatmap(messages_df)
        self._update_visualisation(messages_df)

    def create_activity_heatmap(self, messages_df: pl.DataFrame) -> None:
        if len(messages_df) == 0:
            return

        activity_df = (
            messages_df.group_by(["year", "month"])
            .agg([
                pl.col("word_count").mean().alias("avg_words"),
                pl.col("word_count").count().alias("message_count"),
            ])
            .sort(["year", "month"])
        )

        if len(activity_df) == 0:
            return

        years = sorted(activity_df["year"].unique().to_list())
        months = list(range(1, 13))
        matrix = np.zeros((len(years), 12))

        for row in activity_df.iter_rows(named=True):
            year_idx = years.index(row["year"])
            month_idx = row["month"] - 1
            if row["message_count"] > 0:
                matrix[year_idx, month_idx] = row["avg_words"]

        self._print_heatmap(
            matrix, years, months, "Average words per message",
            main_color="green", threshold=0.0, emoji="🔥"
        )

    def create_threshold_heatmap(self, messages_df: pl.DataFrame) -> None:
        if len(messages_df) == 0:
            return

        activity_df = (
            messages_df.group_by(["year", "month"])
            .agg([
                pl.col("word_count").mean().alias("avg_words"),
                pl.col("word_count").max().alias("max_words"),
                pl.col("word_count").count().alias("message_count"),
            ])
            .sort(["year", "month"])
        )

        if len(activity_df) == 0:
            return

        years = sorted(activity_df["year"].unique().to_list())
        months = list(range(1, 13))
        matrix = np.zeros((len(years), 12))

        for row in activity_df.iter_rows(named=True):
            year_idx = years.index(row["year"])
            month_idx = row["month"] - 1
            if (
                row["message_count"] > 0
                and row["max_words"] > self.params.min_words
            ):
                matrix[year_idx, month_idx] = (
                    row["max_words"] - self.params.min_words
                )
        self._print_heatmap(
            matrix, years, months,
            f"(found groups with threshold)",
            main_color="yellow", threshold=self.params.min_words, emoji="🟡"
        )

    def create_quantiles_bar_chart(
        self,
        quantiles: dict[str, float],
        messages_df: pl.DataFrame | None = None,
    ) -> None:
        if not quantiles:
            return

        threshold_info = (
            f" (Current threshold: {self.params.min_words} words)"
            if self.params.min_words > 0
            else ""
        )
        self.console.print(
            f"\n📊 Message word quantiles from full dataset{threshold_info}",
            style="bold green",
        )
        if messages_df is None:
            messages_df = self._cached_messages_df

        if messages_df is None:
            self.console.print(
                "❌ No data for message word quantiles analysis", style="red"
            )
            return

        word_counts = messages_df["word_count"]
        quantile_values = [
            self.params.min_words,
            float(word_counts.quantile(0.25) or 0),
            float(word_counts.quantile(0.5) or 0),
            float(word_counts.quantile(0.75) or 0),
            float(word_counts.quantile(0.9) or 0),
            float(word_counts.quantile(0.95) or 0),
            float(word_counts.quantile(0.997) or 0),
            float(word_counts.max() or 0),
        ]

        max_val = max(quantile_values)
        bar_width = 40

        data = []
        for _i, (label, value) in enumerate(zip([f"{self.params.min_words}", "Q25", "Q50", "Q75", "Q90", "Q95", "Q99.7", "MAX"], quantile_values, strict=False)):
            bar_length = int((value / max_val) * bar_width) if max_val > 0 else 0
            bar = "█" * bar_length + "░" * (bar_width - bar_length)
            data.append((label, value, bar))

        table = Table(show_header=True, box=box.SIMPLE)
        table.add_column("Quantile", style="cyan", width=10)
        table.add_column("Value", justify="right", style="yellow", width=8)
        table.add_column("Chart", style="blue", width=45)

        for label, value, bar in data:
            table.add_row(label, f"{value:.1f}", bar)

        self.console.print(table)

    def find_long_message_groups(
        self,
        min_words: int | None = None,
        min_consecutive: int = 5,
        time_window_hours: int = 1,
        overlap: int = 10,
        chat_id: str | None = None,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        if min_words is None:
            min_words = self.params.min_words

        try:
            chat_filter = f"AND m.chat_id = {chat_id}" if chat_id else ""
            
            sql = f"""
            WITH latest_names AS (
                SELECT chat_id, chat_name 
                FROM (
                    SELECT chat_id, chat_name, 
                           ROW_NUMBER() OVER (PARTITION BY chat_id ORDER BY COALESCE(last_seen, first_seen) DESC) AS rn
                    FROM {self.chat_names_table}
                ) t WHERE rn = 1
            ),
            messages_with_words AS (
                SELECT
                    m.message_id::BIGINT as message_id,
                    m.chat_id::BIGINT as chat_id,
                    COALESCE(ln.chat_name, 'Unknown') as chat_name,
                    m.text,
                    m.from_id::BIGINT as from_id,
                    m.date,
                    CASE
                        WHEN TRIM(m.text) = '' THEN 0
                        ELSE LENGTH(TRIM(m.text)) - LENGTH(REPLACE(TRIM(m.text), ' ', '')) + 1
                    END AS word_count
                FROM {self.messages_table} m
                LEFT JOIN latest_names ln ON m.chat_id = ln.chat_id
                WHERE m.text IS NOT NULL
                  AND LENGTH(TRIM(m.text)) > 0
                  {chat_filter}
            ),
            long_messages AS (
                SELECT *,
                       ROW_NUMBER() OVER (PARTITION BY chat_id ORDER BY date) as rn
                FROM messages_with_words
                WHERE word_count >= {min_words}
            ),
            time_diffs AS (
                SELECT *,
                       COALESCE(date - LAG(date, 1) OVER (PARTITION BY chat_id ORDER BY date),
                               INTERVAL '0 hours') as time_diff
                FROM long_messages
            ),
            message_groups AS (
                SELECT *,
                       SUM(CASE WHEN time_diff > INTERVAL '{time_window_hours} hours'
                                THEN 1 ELSE 0 END)
                       OVER (PARTITION BY chat_id ORDER BY date ROWS UNBOUNDED PRECEDING) as temp_group_id
                FROM time_diffs
            ),
            consecutive_groups AS (
                SELECT chat_id, temp_group_id, chat_name,
                       COUNT(*) as message_count,
                       MIN(date) as start_time,
                       MAX(date) as end_time,
                       SUM(word_count) as total_words,
                       COUNT(DISTINCT from_id) as participants,
                       AVG(word_count::DOUBLE) as avg_words_per_message,
                       ARRAY_AGG(message_id ORDER BY date) as message_ids
                FROM message_groups
                GROUP BY chat_id, temp_group_id, chat_name
                HAVING COUNT(*) >= {min_consecutive}
            )
            SELECT 
                ROW_NUMBER() OVER (ORDER BY start_time) - 1 as group_id,
                chat_id,
                chat_name,
                start_time,
                end_time,
                message_count,
                total_words,
                participants,
                avg_words_per_message,
                message_ids
            FROM consecutive_groups
            ORDER BY start_time
            """

            # Execute query and convert to DataFrame
            groups_result = self.db.db.execute(sql).arrow()
            groups_df = pl.from_arrow(groups_result)

            if groups_df.is_empty():
                return pl.DataFrame(), pl.DataFrame()
            
            # Get detailed messages for each group with overlap
            all_messages_list = []
            
            if len(groups_df) > 0:
                for group_row in groups_df.iter_rows(named=True):
                    group_id = group_row["group_id"]
                    chat_id_val = group_row["chat_id"]
                    start_time = group_row["start_time"]
                    end_time = group_row["end_time"]
                    
                    # Get core messages and context with single query
                    messages_sql = f"""
                    WITH core_messages AS (
                        SELECT message_id::BIGINT as message_id, chat_id::BIGINT as chat_id, text, from_id::BIGINT as from_id, date,
                               CASE
                                   WHEN TRIM(text) = '' THEN 0
                                   ELSE LENGTH(TRIM(text)) - LENGTH(REPLACE(TRIM(text), ' ', '')) + 1
                               END AS word_count,
                               'core' as message_type
                        FROM {self.messages_table}
                        WHERE chat_id = {chat_id_val}
                          AND date >= '{start_time}'
                          AND date <= '{end_time}'
                          AND text IS NOT NULL
                    ),
                    overlap_before AS (
                        SELECT message_id::BIGINT as message_id, chat_id::BIGINT as chat_id, text, from_id::BIGINT as from_id, date,
                               CASE
                                   WHEN TRIM(text) = '' THEN 0
                                   ELSE LENGTH(TRIM(text)) - LENGTH(REPLACE(TRIM(text), ' ', '')) + 1
                               END AS word_count,
                               'overlap' as message_type
                        FROM {self.messages_table}
                        WHERE chat_id = {chat_id_val}
                          AND date < '{start_time}'
                          AND text IS NOT NULL
                        ORDER BY date DESC
                        LIMIT {overlap}
                    ),
                    overlap_after AS (
                        SELECT message_id::BIGINT as message_id, chat_id::BIGINT as chat_id, text, from_id::BIGINT as from_id, date,
                               CASE
                                   WHEN TRIM(text) = '' THEN 0
                                   ELSE LENGTH(TRIM(text)) - LENGTH(REPLACE(TRIM(text), ' ', '')) + 1
                               END AS word_count,
                               'overlap' as message_type
                        FROM {self.messages_table}
                        WHERE chat_id = {chat_id_val}
                          AND date > '{end_time}'
                          AND text IS NOT NULL
                        ORDER BY date ASC
                        LIMIT {overlap}
                    )
                    SELECT *, {group_id} as group_id
                    FROM (
                        SELECT * FROM overlap_before
                        UNION ALL
                        SELECT * FROM core_messages
                        UNION ALL
                        SELECT * FROM overlap_after
                    ) 
                    ORDER BY date
                    """
                    
                    messages_result = self.db.db.execute(messages_sql).arrow()
                    group_messages_df = pl.from_arrow(messages_result)
                    if not group_messages_df.is_empty():
                        all_messages_list.append(group_messages_df)

            # Combine all messages
            all_messages_df = (
                pl.concat(all_messages_list, how="vertical")
                if all_messages_list
                else pl.DataFrame()
            )

            return groups_df, all_messages_df

        except Exception as e:
            self.console.print(f"Error in find_long_message_groups: {e}", style="red")
            return pl.DataFrame(), pl.DataFrame()

    def display_groups(
        self, groups_df: pl.DataFrame, sort_by: str = "total_words", all_messages_df: pl.DataFrame | None = None,
    ) -> None:
        """Display groups using original field names."""
        if groups_df.is_empty():
            self.console.print("❌ No long message groups found.", style="red")
            return

        # Sort DataFrame
        display_df = groups_df.sort(sort_by, descending=True)

        # Create display DataFrame with original field names
        final_df = display_df.with_columns(
            [
                pl.int_range(1, len(display_df) + 1).alias("group_num"),
                pl.col("chat_name").str.slice(
                    0, self.config.MAX_CHAT_NAME_LENGTH
                ),
                (
                    pl.col("start_time").dt.strftime("%Y-%m-%d %H:%M")
                    + " → "
                    + pl.col("end_time").dt.strftime("%Y-%m-%d %H:%M")
                ).alias("time_range"),
                pl.col("avg_words_per_message").round(1),
                # Add text preview from first message in each group
                pl.when(all_messages_df is not None)
                .then(
                    pl.struct(["group_id"]).map_elements(
                        lambda x: self._get_group_text_preview(x["group_id"], all_messages_df),
                        return_dtype=pl.String
                    )
                )
                .otherwise(pl.lit("No messages available"))
                .alias("text_preview"),
            ]
        ).select(
            [
                "group_num",
                "chat_name",
                "time_range",
                "message_count",
                "total_words",
                "avg_words_per_message",
                "text_preview",
            ]
        )

        self._print_dataframe(
            final_df,
            f"🔥 Found {len(groups_df)} long message groups",
            max_rows=self.config.DEFAULT_MAX_ROWS,
        )

    def generate_summary(self, group: dict[str, Any]) -> str:
        """Generate summary using configured LLM provider."""
        try:
            messages_text = []
            for row in group["group_messages"].iter_rows(named=True):
                timestamp = row["date"].strftime("%H:%M") if hasattr(row["date"], "strftime") else str(row["date"])
                messages_text.append(f"[{timestamp}] ({row['word_count']}w): {row['text']}")

            self.console.print(f"🤖 Analyzing with {self.llm_manager.get_provider_info()}...", style="dim")

            return self.llm_manager.analyze_dialogue(group, messages_text)
        except Exception as e:
            return f"Error: {e}"

    def show_menu(self) -> None:
        """Display main menu options."""
        self.console.print("\nOptions:", style="bold cyan")
        for key, value in self.config.MENU_OPTIONS.items():
            self.console.print(f"  {key}) {value}")

    def _get_user_choice(self, max_option: int, allow_esc: bool = False) -> str:
        """Get user input with ESC support."""
        esc_info = " (ESC to cancel)" if allow_esc else ""
        choice = input(f"Choice (1-{max_option}){esc_info}: ").strip().lower()
        return choice

    def _get_chat_selection(self) -> str | None:
        try:
            sql = f"""
            SELECT DISTINCT m.chat_id, COALESCE(cn.chat_name, 'Unknown') as chat_name,
                   COUNT(*) as message_count
            FROM {self.messages_table} m
            LEFT JOIN {self.chat_names_table} cn ON m.chat_id = cn.chat_id
            WHERE m.text IS NOT NULL AND LENGTH(TRIM(m.text)) > 0
            GROUP BY m.chat_id, cn.chat_name
            ORDER BY message_count DESC
            LIMIT 20
            """

            chats_result = self.db.db.execute(sql).arrow()
            chats_df = pl.from_arrow(chats_result)
            if chats_df.is_empty():
                return None

            self.console.print("\n📋 Available chats:", style="bold blue")
            self.console.print("0. All chats (general statistics)")
            for i in range(len(chats_df)):
                chat_name = chats_df["chat_name"][i][:40]
                msg_count = chats_df["message_count"][i]
                self.console.print(
                    f"{i+1}. {chat_name} ({msg_count:,} messages)"
                )

            choice = input(f"\nSelect chat (0-{len(chats_df)}): ").strip()

            if choice == "0":
                return None
            elif choice.isdigit() and 1 <= int(choice) <= len(chats_df):
                return str(chats_df["chat_id"][int(choice) - 1])
            else:
                self.console.print("❌ Invalid choice", style="red")
                return None

        except Exception as e:
            self.console.print(f"Error selecting chat: {e}", style="red")
            return None

    def _get_quantile_mapping(
        self, matrix: np.ndarray
    ) -> tuple[list[float], np.ndarray]:
        """Get quantile-based color mapping."""
        non_zero_values = matrix[matrix > 0]
        if len(non_zero_values) == 0:
            return [0, 0, 0, 0, 0], np.zeros_like(matrix)

        quantiles = [
            0,
            float(np.quantile(non_zero_values, 0.25)),
            float(np.quantile(non_zero_values, 0.5)),
            float(np.quantile(non_zero_values, 0.75)),
            float(np.max(non_zero_values)),
        ]

        # Map values to intensity levels based on quantiles
        intensity_matrix = np.zeros_like(matrix, dtype=int)
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                val = matrix[i, j]
                if val == 0:
                    intensity_matrix[i, j] = 0
                elif val <= quantiles[1]:
                    intensity_matrix[i, j] = 1
                elif val <= quantiles[2]:
                    intensity_matrix[i, j] = 2
                elif val <= quantiles[3]:
                    intensity_matrix[i, j] = 3
                else:
                    intensity_matrix[i, j] = 4

        return quantiles, intensity_matrix



    def _handle_cluster_analysis(
        self, groups_df: pl.DataFrame, all_messages_df: pl.DataFrame
    ) -> None:
        """Handle cluster analysis with ESC support."""
        if groups_df.is_empty():
            return

        self.console.print("\n🔍 Cluster Analysis", style="bold cyan")
        cluster_choice = self._get_user_choice(len(groups_df), allow_esc=True)

        if cluster_choice == "esc":
            self.console.print("Cancelled.", style="dim")
            return

        try:
            group_num = int(cluster_choice)
            if 1 <= group_num <= len(groups_df):
                # Use the group_num as displayed (1-based index in the sorted display)
                # But we need to get the actual group from the sorted DataFrame
                sorted_groups = groups_df.sort("total_words", descending=True)
                group_row = sorted_groups.row(group_num - 1, named=True)
                group_id = group_row["group_id"]
                group_messages = all_messages_df.filter(
                    pl.col("group_id") == group_id
                )

                self.console.print(f"\nSelected: {group_row['chat_name']}")

                preview_messages = group_messages.head(5).select(
                    [
                        pl.col("date").dt.strftime("%Y-%m-%d %H:%M").alias("time"),
                        pl.col("word_count"),
                        pl.col("text")
                        .str.slice(0, self.config.MAX_TEXT_PREVIEW_LENGTH)
                        .alias("text_preview"),
                    ]
                )

                self._print_dataframe(
                    preview_messages,
                    f"Group #{group_num} Preview ({group_row['message_count']} messages, {group_row['total_words']} words)",
                    max_rows=5,
                )

                # Ask for confirmation before sending to OpenAI
                self.console.print("\n🤖 Send this cluster to OpenAI for analysis?", style="bold yellow")
                self.console.print("  • Press 'y' to proceed with AI analysis")
                self.console.print("  • Press any other key to cancel")
                confirm = input("Choice (y/N): ").strip().lower()

                if confirm != 'y':
                    self.console.print("❌ Analysis cancelled", style="dim")
                    return

                # Check if LLM is properly configured
                if not self.llm_manager.promptinator:
                    self.console.print(
                        "❌ LLM not configured. Please check your API keys and go to LLM Configuration.", style="red"
                    )
                    return

                group_dict = {
                    "chat_name": group_row["chat_name"],
                    "message_count": group_row["message_count"],
                    "total_words": group_row["total_words"],
                    "participants": group_row["participants"],
                    "avg_words_per_message": group_row[
                        "avg_words_per_message"
                    ],
                    "group_messages": group_messages,
                }

                self.console.print(f"🤖 Analyzing with {self.llm_manager.config.provider}...", style="bold cyan")
                summary = self.generate_summary(group_dict)
                self.console.print(
                    Panel(
                        summary, title="🤖 AI Analysis", border_style="blue"
                    )
                )

        except (ValueError, IndexError):
            self.console.print("❌ Invalid selection", style="red")

    def _handle_parameters(
        self, params: SearchParams
    ) -> tuple[pl.DataFrame, pl.DataFrame, SearchParams]:
        """Handle parameter configuration with recommendations."""
        self.console.print("\n⚙️ Parameter Configuration", style="bold cyan")
        self.console.print(
            f"Current: min_words={params.min_words}, chat={'All' if not params.chat_id else params.chat_id}"
        )
        new_words = input(f"Min words (current: {params.min_words}): ").strip()
        if new_words.isdigit():
            params.min_words = int(new_words)
        self.console.print("\nSelect chat scope:", style="bold cyan")
        selected_chat = self._get_chat_selection()
        params.chat_id = selected_chat


        self._get_search_recommendations(self._cached_quantiles)

        self.console.print("🔄 Recalculating...", style="yellow")

        # Invalidate cache when parameters change
        self._invalidate_cache()

        groups_df, all_messages_df = self.find_long_message_groups(
            min_words=params.min_words,
            min_consecutive=params.min_consecutive,
            time_window_hours=params.time_window_hours,
            overlap=params.overlap,
            chat_id=params.chat_id,
        )

        # Update analyzer's params to reflect changes
        self.params = params

        if not all_messages_df.is_empty():
            all_messages_df = all_messages_df.with_columns(
                [
                    pl.col("date").dt.year().alias("year"),
                    pl.col("date").dt.month().alias("month"),
                ]
            )

            # Recalculate quantiles after parameter changes
            _, quantiles = self.analyze_word_quantiles(chat_id=params.chat_id, use_cache=False)

            self.console.print(
                f"\n📊 Word analysis (current threshold: {self.params.min_words} words)",
                style="bold cyan",
            )
            self.create_activity_heatmap(all_messages_df)
            self._update_visualisation(all_messages_df)
        else:
            self.console.print(
                "❌ No data for analysis after changing parameters",
                style="red",
            )

        return groups_df, all_messages_df, params

    def _handle_word_analysis(self) -> None:
        """Handle word analysis with chat selection."""
        self.console.print(
            f"\n📊 Word analysis (current threshold: {self.params.min_words} words)",
            style="bold cyan",
        )
        if (
            self._cached_messages_df is not None
            and not self._cached_messages_df.is_empty()
        ):
            self._update_visualisation(self._cached_messages_df)
        else:
            self.display_word_analysis(interactive=True, min_words=0)

    def _handle_sorting(self, groups_df: pl.DataFrame) -> pl.DataFrame:
        """Handle group sorting."""
        if groups_df.is_empty():
            return groups_df

        self.console.print("\n📊 Sort Groups:", style="bold cyan")
        for key, (_field, desc) in self.config.SORT_OPTIONS.items():
            self.console.print(f"  {key}) {desc}")

        sort_choice = self._get_user_choice(len(self.config.SORT_OPTIONS))

        if sort_choice in self.config.SORT_OPTIONS:
            sort_field, _ = self.config.SORT_OPTIONS[sort_choice]
            return groups_df.sort(sort_field, descending=True)
        else:
            self.console.print("❌ Invalid choice", style="red")
            return groups_df

    def _handle_chat_selection(
        self, groups_df: pl.DataFrame, all_messages_df: pl.DataFrame, params: SearchParams
    ) -> tuple[pl.DataFrame, pl.DataFrame, SearchParams]:
        """Handle chat selection and recalculate groups."""
        self.console.print("\n💬 Chat Selection", style="bold cyan")
        self.console.print(
            f"Current chat: {'All chats' if not params.chat_id else params.chat_id}"
        )
        
        selected_chat = self._get_chat_selection()
        params.chat_id = selected_chat

        self.console.print("🔄 Recalculating with new chat selection...", style="yellow")

        # Invalidate cache when chat selection changes
        self._invalidate_cache()

        groups_df, all_messages_df = self.find_long_message_groups(
            min_words=params.min_words,
            min_consecutive=params.min_consecutive,
            time_window_hours=params.time_window_hours,
            overlap=params.overlap,
            chat_id=params.chat_id,
        )

        # Update analyzer's params to reflect changes
        self.params = params

        if not all_messages_df.is_empty():
            all_messages_df = all_messages_df.with_columns(
                [
                    pl.col("date").dt.year().alias("year"),
                    pl.col("date").dt.month().alias("month"),
                ]
            )

            # Recalculate quantiles after chat selection changes
            _, quantiles = self.analyze_word_quantiles(chat_id=params.chat_id, use_cache=False)

            self.console.print(
                f"\n📊 Word analysis (current threshold: {self.params.min_words} words)",
                style="bold cyan",
            )
            self.create_activity_heatmap(all_messages_df)
            self._update_visualisation(all_messages_df)
        else:
            self.console.print(
                "❌ No data for analysis with selected chat",
                style="red",
            )
        
        return groups_df, all_messages_df, params

    def _handle_llm_configuration(self) -> None:
        """Handle LLM provider and model configuration."""
        self.console.print("\n🤖 LLM Configuration", style="bold cyan")
        self.console.print(f"Current: {self.llm_manager.get_provider_info()}")

        while True:
            self.console.print("\nConfiguration Options:", style="bold blue")
            self.console.print("  1) Change Provider & Model")
            self.console.print("  2) Change Prompt File")
            self.console.print("  3) View Available Prompts")
            self.console.print("  4) View Recommended Models")
            self.console.print("  5) Test Current Configuration")
            self.console.print("  6) Back to Main Menu")

            choice = self._get_user_choice(6)

            if choice == "1":
                self._configure_provider_model()
            elif choice == "2":
                self._configure_prompt_file()
            elif choice == "3":
                self._show_available_prompts()
            elif choice == "4":
                self._show_recommended_models()
            elif choice == "5":
                self._test_llm_configuration()
            elif choice == "6":
                break

    def _configure_provider_model(self) -> None:
        """Configure LLM provider and model."""
        self.console.print("\n📋 Available Providers:", style="bold blue")

        providers = list(PROVIDER_MODEL_COMBINATIONS.keys())
        for i, provider in enumerate(providers, 1):
            config = PROVIDER_MODEL_COMBINATIONS[provider]
            self.console.print(f"  {i}) {config['display_name']} ({provider})")

        choice = input(f"\nSelect provider (1-{len(providers)}): ").strip()

        if not choice.isdigit() or not (1 <= int(choice) <= len(providers)):
            self.console.print("❌ Invalid choice", style="red")
            return

        selected_provider = providers[int(choice) - 1]
        provider_config = PROVIDER_MODEL_COMBINATIONS[selected_provider]

        self.console.print(f"\n📋 Available models for {provider_config['display_name']}:", style="bold blue")
        models = provider_config["models"]
        for i, model in enumerate(models, 1):
            self.console.print(f"  {i}) {model}")

        model_choice = input(f"\nSelect model (1-{len(models)}): ").strip()

        if not model_choice.isdigit() or not (1 <= int(model_choice) <= len(models)):
            self.console.print("❌ Invalid choice", style="red")
            return

        selected_model = models[int(model_choice) - 1]

        # Try to initialize with new configuration
        success = self.llm_manager.update_config(
            provider=selected_provider,
            model=selected_model
        )

        if success:
            self.console.print(f"✅ Successfully configured: {selected_provider}:{selected_model}", style="green")
        else:
            self.console.print(f"❌ Failed to configure {selected_provider}:{selected_model}. Check your API key: {provider_config['env_key']}", style="red")

    def _configure_prompt_file(self) -> None:
        """Configure prompt file."""
        available_prompts = get_available_prompts()

        if not available_prompts:
            self.console.print("❌ No prompt files found in prompts directory", style="red")
            return

        self.console.print("\n📋 Available Prompts:", style="bold blue")
        for i, prompt in enumerate(available_prompts, 1):
            self.console.print(f"  {i}) {prompt}")

        choice = input(f"\nSelect prompt (1-{len(available_prompts)}): ").strip()

        if not choice.isdigit() or not (1 <= int(choice) <= len(available_prompts)):
            self.console.print("❌ Invalid choice", style="red")
            return

        selected_prompt = available_prompts[int(choice) - 1]
        self.llm_manager.update_config(prompt_file=selected_prompt)
        self.console.print(f"✅ Prompt file changed to: {selected_prompt}", style="green")

    def _show_available_prompts(self) -> None:
        """Show available prompt files with previews."""
        available_prompts = get_available_prompts()

        if not available_prompts:
            self.console.print("❌ No prompt files found", style="red")
            return

        self.console.print("\n📋 Available Prompt Files:", style="bold blue")

        for prompt in available_prompts:
            try:
                from pathlib import Path
                prompt_path = Path(__file__).parent / "prompts" / prompt
                content = prompt_path.read_text(encoding='utf-8')
                preview = content[:100].replace('\n', ' ') + "..." if len(content) > 100 else content
                self.console.print(f"  • {prompt}: {preview}", style="dim")
            except Exception:
                self.console.print(f"  • {prompt}: (Unable to read)", style="dim")

    def _show_recommended_models(self) -> None:
        """Show recommended models for different use cases."""
        recommendations = get_recommended_models()

        self.console.print("\n💡 Recommended Models:", style="bold magenta")
        for use_case, model in recommendations.items():
            self.console.print(f"  • {use_case.replace('_', ' ').title()}: {model}")

    def _test_llm_configuration(self) -> None:
        """Test current LLM configuration with a simple query."""
        if not self.llm_manager.promptinator:
            self.console.print("❌ LLM not configured", style="red")
            return

        self.console.print(f"\n🧪 Testing {self.llm_manager.get_provider_info()}...", style="yellow")

        try:
            test_response = self.llm_manager.promptinator.query(
                "Respond with exactly: 'Configuration test successful!'",
                temperature=0.1
            )

            if test_response.error:
                self.console.print(f"❌ Test failed: {test_response.error}", style="red")
            else:
                self.console.print(f"✅ Test successful!", style="green")
                self.console.print(f"Response: {test_response.content[:100]}...", style="dim")
                if test_response.tokens_used:
                    if test_response.input_tokens and test_response.output_tokens:
                        self.console.print(f"Tokens: {test_response.input_tokens:,}/{test_response.output_tokens:,} (in/out)", style="dim")
                    else:
                        self.console.print(f"Tokens used: {test_response.tokens_used}", style="dim")
        except Exception as e:
            self.console.print(f"❌ Test failed: {e}", style="red")

    def _update_visualisation(self, messages_df: pl.DataFrame) -> None:
        """Update all three charts: quantiles, activity and threshold."""
        if messages_df is None or len(messages_df) == 0:
            self.console.print(
                "❌ No data for updating charts", style="red"
            )
            return

        self._cached_messages_df = messages_df

        self.create_threshold_heatmap(messages_df)
        self.create_quantiles_bar_chart(self._cached_quantiles, messages_df)


        self._get_search_recommendations(self._cached_quantiles)

    def _get_search_recommendations(self, quantiles: dict[str, float | int] | None) -> None:
        """Generate suggestions for search parameters."""
        if not quantiles:
            return

        self.console.print(
            "\n💡 Recommendations for search parameters", style="bold magenta"
        )
        self.console.print(f"  • Just long messages: --min-words {int(quantiles['q95'])}", style="dim")
        self.console.print(f"  • Unique longread messages: --min-words {int(quantiles['q99.7'])}", style="dim")

    def _print_dataframe(self, df: pl.DataFrame, title: str = "Data", max_rows: int = 20) -> None:
        if len(df) == 0:
            self.console.print(f"❌ {title}: no data available", style="red")
            return

        self.console.print(f"\n📋 {title} ({len(df)} records)", style="bold blue")

        with pl.Config(tbl_rows=max_rows, tbl_hide_column_data_types=True, tbl_hide_dataframe_shape=True):
            self.console.print(df)

        if len(df) > max_rows:
            self.console.print(f"... and {len(df) - max_rows} more records", style="dim")

    def _print_heatmap(
        self,
        matrix: np.ndarray,
        years: list[int],
        months: list[int],
        title: str,
        main_color: str = "green",
        threshold: float = 0.0,
        emoji: str = "🔥",
    ) -> None:
        self.console.print(f"\n{emoji} {title}", style=f"bold {main_color}")

        quantiles, intensity_matrix = self._get_quantile_mapping(matrix)
        intensity_chars = [" ", "░", "▒", "▓", "█"]
        color_styles = self._generate_color_gradient(main_color)

        month_names = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
        header = "      " + "  ".join(f"{m:^2}" for m in month_names)
        self.console.print(header, style="dim")

        for i, year in enumerate(years):
            row_parts = [f"{year} "]
            for j in range(12):
                intensity_idx = int(intensity_matrix[i, j])
                char = intensity_chars[intensity_idx]
                style = color_styles[intensity_idx]

                triple_char = char * 3

                if style:
                    row_parts.append(f"[{style}]{triple_char}[/]")
                else:
                    row_parts.append(f"{triple_char}")
                if j < 11:
                    row_parts.append(" ")

            row_str = "".join(row_parts)
            self.console.print(row_str, markup=True)

        max_val = np.max(matrix) if np.max(matrix) > 0 else 1
        threshold = threshold if threshold == 0 else self.params.min_words
        threshold_info = f" (threshold: {threshold:.1f})" if threshold > 0 else ""
        legend = f"Range: {threshold:.1f} - {max_val:.1f} words/message{threshold_info}"
        self.console.print(f"\n{legend}", style="dim")

        intensity_legend = "Intensity: "
        for i, (char, style, range_val) in enumerate(
            zip(intensity_chars, color_styles, quantiles, strict=False)
        ):
            if style and i > 0:
                intensity_legend += f"[{style}]{char}[/]({range_val:.1f}) "
            elif i == 0:
                intensity_legend += f"{char}(0) "
        self.console.print(intensity_legend, markup=True, style="dim")

    def _generate_color_gradient(self, main_color: str) -> list[str]:
        return [
            "",
            f"dim {main_color}",
            main_color,
            f"bright_{main_color}",
            f"bold bright_{main_color}"
        ]

def main() -> None:
    config = Config()
    parser = argparse.ArgumentParser(
        description="Long Message Analysis Tool for Telegram chats with LLM integration",
        epilog=f"Example: python analyze_dialogues.py --phone {config.DEFAULT_PHONE} --provider openrouter --model google/gemini-2.5-pro",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--phone",
        default=config.DEFAULT_PHONE,
        help="Phone number (e.g., +1234567890)",
    )
    parser.add_argument(
        "--interactive", default=True, action="store_true", help="Launch interactive mode"
    )

    # LLM Configuration arguments
    llm_group = parser.add_argument_group("LLM Configuration")
    llm_group.add_argument(
        "--provider",
        default=config.DEFAULT_PROVIDER,
        choices=list(PROVIDER_MODEL_COMBINATIONS.keys()),
        help=f"LLM provider (default: {config.DEFAULT_PROVIDER})"
    )
    llm_group.add_argument(
        "--model",
        help="LLM model (if not specified, uses provider default)"
    )
    llm_group.add_argument(
        "--prompt",
        default=config.DEFAULT_PROMPT,
        help=f"Prompt file to use (default: {config.DEFAULT_PROMPT})"
    )
    llm_group.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="LLM temperature (default: 0.7)"
    )
    llm_group.add_argument(
        "--list-providers",
        action="store_true",
        help="List available providers and models"
    )
    llm_group.add_argument(
        "--list-prompts",
        action="store_true",
        help="List available prompt files"
    )

    args = parser.parse_args()

    # Handle list commands
    if args.list_providers:
        console = Console()
        console.print("🤖 Available LLM Providers and Models:", style="bold cyan")
        for provider, config in PROVIDER_MODEL_COMBINATIONS.items():
            console.print(f"\n{config['display_name']} ({provider}):", style="bold blue")
            console.print(f"  Environment key: {config['env_key']}")
            console.print(f"  Default model: {config.get('default_model', 'N/A')}")
            console.print("  Available models:")
            for model in config['models']:
                console.print(f"    • {model}")
        return

    if args.list_prompts:
        console = Console()
        prompts = get_available_prompts()
        console.print("📋 Available Prompt Files:", style="bold cyan")
        if not prompts:
            console.print("❌ No prompt files found", style="red")
        else:
            for prompt in prompts:
                console.print(f"  • {prompt}")
        return

    # Initialize analyzer
    analyzer = SentimentAnalyser(args.phone)

    # Configure LLM based on CLI arguments
    model = args.model
    if not model and args.provider in PROVIDER_MODEL_COMBINATIONS:
        model = PROVIDER_MODEL_COMBINATIONS[args.provider].get("default_model")

    # Validate model if specified
    if model and not validate_provider_model(args.provider, model):
        analyzer.console.print(f"❌ Invalid model '{model}' for provider '{args.provider}'", style="red")
        analyzer.console.print("Use --list-providers to see available combinations", style="dim")
        return

    # Update LLM configuration
    success = analyzer.llm_manager.update_config(
        provider=args.provider,
        model=model,
        prompt_file=args.prompt,
        temperature=args.temperature
    )

    if not success:
        analyzer.console.print(f"❌ Failed to initialize {args.provider}. Check your API key.", style="red")
        analyzer.console.print(f"Required environment variable: {PROVIDER_MODEL_COMBINATIONS[args.provider]['env_key']}", style="dim")

    analyzer.console.print(f"🤖 LLM Configuration: {analyzer.llm_manager.get_provider_info()}", style="dim green")

    try:
        _, quantiles = analyzer.analyze_word_quantiles()

        # Initialize SearchParams with quantiles
        params = SearchParams(
            min_words=max(int(quantiles["q99.7"]), config.DEFAULT_MIN_WORDS),
            min_consecutive=5,
            time_window_hours=1,
            overlap=10,
            chat_id=None,
        )

        # Set the params in the analyzer instance
        analyzer.params = params

        groups_df, all_messages_df = analyzer.find_long_message_groups(
            min_words=params.min_words,
            min_consecutive=params.min_consecutive,
            time_window_hours=params.time_window_hours,
            overlap=params.overlap,
            chat_id=params.chat_id,
        )

        analyzer.console.print(
            f"\n📊 Messages Analysis (current threshold: {params.min_words} words)",
            style="bold cyan",
        )
        analyzer.display_word_analysis(min_words=params.min_words)

        if not args.interactive and not groups_df.is_empty():
            analyzer.display_groups(groups_df)
            analyzer.console.print(
                    "\n💡 Use --interactive for interactive mode", style="dim"
                )
            return

        analyzer.interactive_mode(groups_df, all_messages_df, params)


    except KeyboardInterrupt:
        analyzer.console.print("\n👋 Goodbye!", style="yellow")
    finally:
        analyzer.db.close()

if __name__ == "__main__":
    main()
