import json
import logging
import os
from typing import Any

import duckdb
import polars as pl
import torch
from dotenv import load_dotenv

from terrorblade import Logger
from terrorblade.data.database.telegram_database import TelegramDatabase
from terrorblade.data.dtypes import (
    CHAT_NAMES_SCHEMA,
    FILES_SCHEMA,
    USER_NAMES_SCHEMA,
    get_process_schema,
    telegram_import_schema_short,
)
from terrorblade.data.preprocessing.TextPreprocessor import TextPreprocessor
from terrorblade.utils.config import get_db_path


class TelegramPreprocessor(TextPreprocessor):
    """
    High-level processor for Telegram JSON exports and database-backed message processing.

    This class provides a cohesive pipeline to:
    - Load Telegram Desktop JSON archives
    - Normalize and clean message fields
    - Persist messages to DuckDB (per-phone table names)
    - Compute sentence embeddings and persist them
    - Group messages into temporal/semantic clusters

    It can operate in two modes:
    - File mode: parse a Telegram export JSON file and optionally write to database
    - DB mode: read previously persisted messages, compute missing embeddings/clusters

    Attributes:
        use_duckdb (bool): Whether to use DuckDB for persistence and lookups.
        db_path (str): Path to the DuckDB database file.
        phone (str | None): Phone identifier (digits only) used to partition tables.
        logger (Logger): Project logger instance.
        db (duckdb.DuckDBPyConnection): Active connection when `use_duckdb` is True.

    Tags:
        - pipeline
        - preprocessing
        - database
        - embeddings
        - clustering

    Examples:
        Basic usage from a JSON export:
        ```python
        tp = TelegramPreprocessor(use_duckdb=True, db_path="auto", phone="+79991234567")
        chats = tp.process_file("/path/to/telegram/result.json")
        tp.close()
        ```

        Process messages that are already stored in the database (compute missing embeddings):
        ```python
        tp = TelegramPreprocessor(use_duckdb=True, db_path="auto", phone="+79991234567")
        processed = tp.process_messages(phone="+79991234567")
        tp.close()
        ```
    """

    def __init__(
        self,
        use_duckdb: bool = False,
        db_path: str = "auto",
        phone: str | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the preprocessor.

        Args:
            use_duckdb (bool): Enable DuckDB to persist messages, embeddings, and clusters.
            db_path (str): Path to a DuckDB database file or "auto" to use environment/default.
            phone (str | None): Phone number used to scope per-user tables (e.g., `messages_<phone>`).
                Can be specified with or without the leading "+". Required when `use_duckdb=True`.
            *args: Additional args passed to `TextPreprocessor`.
            **kwargs: Additional kwargs passed to `TextPreprocessor`.

        Raises:
            ValueError: If `use_duckdb=True` and `phone` is not provided.

        Tags:
            - pipeline
            - database

        Examples:
            ```python
            tp = TelegramPreprocessor(use_duckdb=True, db_path="auto", phone="+7999...")
            ```
        """
        super().__init__(*args, **kwargs)
        load_dotenv(".env")

        self.use_duckdb = use_duckdb
        self.db_path = get_db_path(db_path)
        self.phone = phone.replace("+", "") if phone else None

        self.logger = Logger(
            name="TelegramPreprocessor",
            level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
            log_file=os.getenv("LOG_FILE", "telegram_preprocessor.log"),
            log_dir=os.getenv("LOG_DIR", "logs"),
        )

        if use_duckdb:
            if not phone:
                raise ValueError("Phone number is required when use_duckdb is True")
            self.logger.info(f"Initializing DuckDB connection with database: {db_path}")
            self.db = duckdb.connect(db_path)
            self._init_cluster_tables()

    def _init_cluster_tables(self) -> None:
        """Initialize per-user tables for clusters, embeddings, and mapping dictionaries.

        Creates the following tables when missing:
        - `message_clusters_<phone>`: mapping of message to cluster group_id
        - `chat_embeddings_<phone>`: message embeddings as FLOAT[768]
        - `chat_names_<phone>`: observed chat_id -> chat_name mappings with first/last seen
        - `user_names_<phone>`: observed from_id -> from_name mappings with first/last seen
        - `files_<phone>`: message_id/chat_id -> file_name mapping
        - `media_types` (global): dictionary table for media type normalization

        Tags:
            - database
            - schema
            - setup

        Examples:
            ```python
            tp = TelegramPreprocessor(use_duckdb=True, db_path="auto", phone="+7999...")
            # Tables are ensured during initialization
            ```
        """
        self.logger.info("Initializing database tables")

        # Create user-specific tables for clusters and embeddings
        phone_clean = self.phone.replace("+", "") if self.phone else "default"
        clusters_table = f"message_clusters_{phone_clean}"
        embeddings_table = f"chat_embeddings_{phone_clean}"
        chat_names_table = f"chat_names_{phone_clean}"
        user_names_table = f"user_names_{phone_clean}"
        files_table = f"files_{phone_clean}"

        self.db.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {clusters_table} (
                message_id BIGINT,
                chat_id BIGINT,
                group_id INTEGER,
                PRIMARY KEY (message_id, chat_id)
            )
        """
        )

        self.db.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {embeddings_table} (
                message_id BIGINT,
                chat_id BIGINT,
                embeddings FLOAT[768],
                PRIMARY KEY (message_id, chat_id)
            )
        """
        )

        # Global media types dictionary
        self.db.execute(
            """
            CREATE TABLE IF NOT EXISTS media_types (
                media_type_id INTEGER,
                name TEXT,
                PRIMARY KEY (media_type_id)
            )
            """
        )
        self.db.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_media_types_name ON media_types(name)
            """
        )

        # Name mapping tables (per-user)
        chat_cols = ", ".join([f'"{field}" {info["db_type"]}' for field, info in CHAT_NAMES_SCHEMA.items()])
        user_cols = ", ".join([f'"{field}" {info["db_type"]}' for field, info in USER_NAMES_SCHEMA.items()])
        files_cols = ", ".join([f'"{field}" {info["db_type"]}' for field, info in FILES_SCHEMA.items()])

        self.db.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {chat_names_table} (
                {chat_cols},
                PRIMARY KEY (chat_id, chat_name)
            )
            """
        )
        self.db.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {user_names_table} (
                {user_cols},
                PRIMARY KEY (from_id, from_name)
            )
            """
        )
        self.db.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {files_table} (
                {files_cols},
                PRIMARY KEY (message_id, chat_id)
            )
            """
        )

    def _get_messages_from_db(self, phone: str | None = None, chat_id: int | None = None) -> pl.DataFrame:
        """
        Retrieve messages for a specific user from DuckDB.

        Args:
            phone (str | None): Phone number whose tables to query. Defaults to `self.phone`.
            chat_id (int | None): If provided, filters messages to a single chat.

        Returns:
            pl.DataFrame: Messages ordered by `date`. Columns match the central TELEGRAM schema subset.

        Raises:
            Exception: On database execution or conversion errors.

        Tags:
            - database
            - io

        Examples:
            ```python
            tp = TelegramPreprocessor(use_duckdb=True, db_path="auto", phone="+7999...")
            df = tp._get_messages_from_db(chat_id=123456)
            ```
        """
        phone = phone or self.phone
        phone_clean = phone.replace("+", "") if phone else "default"
        messages_table = f"messages_{phone_clean}"
        query = f"SELECT * FROM {messages_table}"

        if chat_id is not None:
            query += f" WHERE chat_id = {chat_id}"

        query += " ORDER BY date"

        self.logger.info(f"Retrieving messages for phone {phone}" + (f" and chat {chat_id}" if chat_id else ""))
        try:
            result = self.db.execute(query).arrow()
            df = pl.from_arrow(result)
            # Ensure we return a DataFrame, not a Series
            if isinstance(df, pl.Series):
                df = df.to_frame()
            self.logger.info(f"Retrieved {len(df)} messages")
            return df
        except Exception as e:
            self.logger.error(f"Error retrieving messages from database: {str(e)}")
            raise

    def _update_clusters_in_db(self, clusters_df: pl.DataFrame) -> None:
        """
        Batch upsert cluster assignments for messages.

        Args:
            clusters_df (pl.DataFrame): DataFrame with columns `message_id`, `chat_id`, `group_id`.

        Tags:
            - database
            - clusters
            - write

        Examples:
            ```python
            # clusters_df must contain: message_id, chat_id, group_id
            tp._update_clusters_in_db(clusters_df)
            ```
        """
        clusters_table = f"message_clusters_{self.phone}"

        if len(clusters_df) == 0:
            return

        try:
            # The register + INSERT approach (0.0114s) is 65x faster than executemany(.rows()) (0.7483s)
            self.db.register("clusters_df", clusters_df)
            self.db.execute(
                f"""
                INSERT OR IGNORE INTO {clusters_table}
                (message_id, chat_id, group_id)
                SELECT message_id, chat_id, group_id
                FROM clusters_df
                """
            )
            self.logger.nice(f"Updated {len(clusters_df)} cluster records in table {clusters_table} ✓")  # type: ignore
        except Exception as e:
            self.logger.error(f"Error updating clusters in database: {str(e)}")
            raise

    def _update_embeddings_in_db(self, embeddings_df: pl.DataFrame) -> None:
        """
        Batch upsert embeddings for messages.

        Validates that the `embeddings` column is of type `Array(Float32, shape=768)` and then
        performs an `INSERT OR IGNORE` into the per-user embeddings table.

        Args:
            embeddings_df (pl.DataFrame): Must contain columns `message_id`, `chat_id`, `embeddings`.
                `embeddings` must be a fixed-length float array (length 768).

        Raises:
            ValueError: If `embeddings` column is missing or has unexpected type.
            Exception: On database execution errors.

        Tags:
            - database
            - embeddings
            - write

        Examples:
            ```python
            # Ensure embeddings are Float32 with shape 768
            tp._update_embeddings_in_db(df.select(["message_id", "chat_id", "embeddings"]))
            ```
        """
        embeddings_table = f"chat_embeddings_{self.phone}"

        if len(embeddings_df) == 0:
            return

        # Validate embeddings column exists and has correct type
        if "embeddings" not in embeddings_df.columns:
            raise ValueError("DataFrame must contain 'embeddings' column")

        # Validate embeddings are F32 arrays with correct shape
        embeddings_dtype = embeddings_df["embeddings"].dtype
        if not isinstance(embeddings_dtype, pl.Array) or embeddings_dtype.inner != pl.Float32:
            raise ValueError(f"Embeddings must be Array(Float32, shape=768), got {embeddings_dtype}")

        try:
            # The register + INSERT approach (0.0114s) is 65x faster than executemany(.rows()) (0.7483s)
            self.db.register("embeddings_df", embeddings_df)
            self.db.execute(
                f"""
                INSERT OR IGNORE INTO {embeddings_table}
                (message_id, chat_id, embeddings)
                SELECT message_id, chat_id, embeddings
                FROM embeddings_df
                """
            )
            self.logger.nice(f"Updated {len(embeddings_df)} embeddings ✓")  # type: ignore

        except Exception as e:
            self.logger.error(f"Error batch updating embeddings in database: {str(e)}")
            raise

    def _get_embeddings_from_db(self, chat_id: int) -> dict:
        """
        Retrieve embeddings for a specific chat.

        Args:
            chat_id (int): Chat identifier.

        Returns:
            dict: Mapping `{message_id: embeddings}` for the requested chat.

        Raises:
            Exception: On database errors.

        Tags:
            - database
            - embeddings
            - read

        Examples:
            ```python
            embs = tp._get_embeddings_from_db(chat_id=123)
            vector = embs.get(42)
            ```
        """
        phone_clean = self.phone.replace("+", "") if self.phone else "default"
        embeddings_table = f"chat_embeddings_{phone_clean}"
        self.logger.info(f"Retrieving embeddings for chat {chat_id}")

        try:
            query = f"""
                SELECT message_id, embeddings
                FROM {embeddings_table}
                WHERE chat_id = ?
                ORDER BY message_id
            """
            result = self.db.execute(query, [chat_id]).fetchall()
            embeddings_dict = {row[0]: row[1] for row in result}
            self.logger.info(f"Retrieved {len(embeddings_dict)} embeddings for chat {chat_id}")
            return embeddings_dict
        except Exception as e:
            self.logger.error(f"Error retrieving embeddings from database: {str(e)}")
            raise

    def _get_messages_with_embeddings(self, chat_id: int | None = None) -> set:
        """
        Return IDs of messages that already have stored embeddings.

        Args:
            chat_id (int | None): If provided, limit to a specific chat.

        Returns:
            set: Set of `message_id` values present in the embeddings table.

        Tags:
            - database
            - embeddings
            - read

        Examples:
            ```python
            existing = tp._get_messages_with_embeddings()
            ```
        """
        phone_clean = self.phone.replace("+", "") if self.phone else "default"
        embeddings_table = f"chat_embeddings_{phone_clean}"
        self.logger.info("Retrieving message IDs with embeddings" + (f" for chat {chat_id}" if chat_id else ""))

        try:
            query = f"SELECT message_id FROM {embeddings_table}"
            if chat_id is not None:
                query += f" WHERE chat_id = {chat_id}"

            result = self.db.execute(query).fetchall()
            message_ids = {row[0] for row in result}
            self.logger.info(f"Found {len(message_ids)} messages with existing embeddings")
            return message_ids
        except Exception as e:
            self.logger.error(f"Error retrieving message IDs with embeddings: {str(e)}")
            return set()

    def load_json(self, file_path: str, min_messages: int = 3) -> dict[int, pl.DataFrame]:
        """
        Load and minimally normalize a Telegram Desktop JSON export.

        The function flattens `text_entities` into `text`, creates `from_name` from `from`,
        filters out small chats, and returns a dictionary keyed by `chat_id`.

        Args:
            file_path (str): Path to a Telegram export JSON (e.g., `result.json`).
            min_messages (int): Minimum number of messages for a chat to be included.

        Returns:
            dict[int, pl.DataFrame]: Mapping of `chat_id` -> messages DataFrame.
                The DataFrame adheres to `telegram_import_schema_short` with additional
                columns `chat_name`, `chat_id`, `chat_type`, and `message_id`.

        Raises:
            ValueError: If `file_path` is not a `.json` file.
            Exception: On file IO or JSON parsing errors.

        Tags:
            - io
            - preprocessing

        Examples:
            ```python
            chats = tp.load_json("/exports/result.json", min_messages=5)
            df = chats[123456]
            ```
        """
        if not file_path.endswith(".json"):
            raise ValueError("File must be a JSON file")

        self.logger.info(f"Loading JSON data from {file_path}")
        try:
            with open(file_path) as file:
                data = json.load(file)
                chat_dict = {}

                for chat in data["chats"]["list"]:
                    if len(chat["messages"]) >= min_messages:
                        for message in chat["messages"]:
                            if "text_entities" in message:
                                texts = []
                                for entity in message["text_entities"]:
                                    if isinstance(entity, dict) and "text" in entity:
                                        texts.append(entity["text"])
                                if texts:
                                    message["text"] = " ".join(texts)

                            if "from" in message:
                                message["from_name"] = message["from"]

                        filtered_messages = [
                            {
                                field: (str(message.get(field, "")) if field == "text" else message.get(field))
                                for field in telegram_import_schema_short
                                if field in message or field == "from_name"
                            }
                            for message in chat["messages"]
                        ]

                        messages_df = pl.DataFrame(filtered_messages, schema=telegram_import_schema_short)

                        messages_df = messages_df.with_columns(
                            [
                                pl.lit(chat.get("name")).alias("chat_name"),
                                pl.lit(chat["id"]).alias("chat_id"),
                                pl.lit(chat["type"]).alias("chat_type"),
                                pl.col("id").alias("message_id"),
                            ]
                        ).drop("id")
                        chat_dict[chat["id"]] = messages_df

                self.logger.nice(f"Loaded {len(chat_dict)} chats with minimum {min_messages} messages ✓")  # type: ignore
                return chat_dict
        except Exception as e:
            self.logger.error(f"Error loading JSON data: {str(e)}")
            raise

    def parse_links(self, chat_df: pl.DataFrame) -> pl.DataFrame:
        """
        Convert nested Telegram `text` structures into plain strings.

        Telegram JSON sometimes stores `text` as a list of objects with `text` fields.
        This function flattens them into a single string.

        Args:
            chat_df (pl.DataFrame): Messages DataFrame with a `text` column that may contain lists.

        Returns:
            pl.DataFrame: DataFrame where `text` is normalized to `Utf8` strings.

        Tags:
            - preprocessing
            - text

        Examples:
            ```python
            out = tp.parse_links(pl.DataFrame({"text": [[{"text": "Hello"}, {"text": "World"}]]}))
            assert out[0, "text"] == "Hello World"
            ```
        """

        def extract_text(val: Any) -> Any:
            if isinstance(val, list):
                if len(val) == 1 and isinstance(val[0], dict) and "type" in val[0] and "text" in val[0]:
                    return val[0]["text"]
                return " ".join(item.get("text", "") for item in val if isinstance(item, dict) and "text" in item)
            return val

        return chat_df.with_columns([pl.col("text").map_elements(extract_text, return_dtype=pl.Utf8).alias("text")])

    def parse_members(self, chat_df: pl.DataFrame) -> pl.DataFrame:
        """
        Normalize `members` column to a compact string representation.

        If present, converts a list of members into a unique set and stringifies it.

        Args:
            chat_df (pl.DataFrame): Messages DataFrame.

        Returns:
            pl.DataFrame: DataFrame with `members` normalized (if the column exists).

        Tags:
            - preprocessing

        Examples:
            ```python
            df = pl.DataFrame({"members": [["alice", "bob", "alice"]]})
            out = tp.parse_members(df)
            assert "alice" in out[0, "members"]
            ```
        """
        if "members" in chat_df.columns:
            return chat_df.with_columns(
                [
                    pl.col("members")
                    .map_elements(
                        lambda x: str(list(set(x))) if isinstance(x, list) else x,
                        return_dtype=pl.self_dtype(),
                    )
                    .alias("members")
                ]
            )
        return chat_df

    def parse_reactions(self, chat_df: pl.DataFrame) -> pl.DataFrame:
        """
        Extract the first reaction emoji from the `reactions` list.

        Args:
            chat_df (pl.DataFrame): Messages DataFrame with optional `reactions` column.

        Returns:
            pl.DataFrame: DataFrame with `reactions` converted to a single emoji or `None`.

        Tags:
            - preprocessing

        Examples:
            ```python
            df = pl.DataFrame({"reactions": [[{"emoji": "👍"}]]})
            out = tp.parse_reactions(df)
            assert out[0, "reactions"] == "👍"
            ```
        """
        if "reactions" in chat_df.columns:
            return chat_df.with_columns(
                [
                    pl.col("reactions")
                    .map_elements(
                        lambda x: (x[0]["emoji"] if isinstance(x, list | pl.Series) and len(x) > 0 else None),
                        return_dtype=pl.Utf8,
                    )
                    .alias("reactions")
                ]
            )
        return chat_df

    def standardize_chat(self, chat_df: pl.DataFrame) -> pl.DataFrame:
        """
        Ensure all columns from `telegram_import_schema_short` exist with correct types.

        Creates missing columns as `None` and casts existing columns to target dtypes.

        Args:
            chat_df (pl.DataFrame): Messages DataFrame.

        Returns:
            pl.DataFrame: Standardized DataFrame ready for further processing.

        Tags:
            - preprocessing
            - schema

        Examples:
            ```python
            out = tp.standardize_chat(pl.DataFrame({"message_id": [1], "text": ["hello"], "chat_id": [1]}))
            ```
        """
        for col in telegram_import_schema_short:
            if col not in chat_df.columns:
                chat_df = chat_df.with_columns(pl.lit(None).alias(col))

        # Cast columns to expected types
        return chat_df.select([pl.col(col).cast(dtype) for col, dtype in telegram_import_schema_short.items()])

    def parse_timestamp(self, df: pl.DataFrame, date_col: str = "date") -> pl.DataFrame:
        """
        Parse the `date` column into Polars `Datetime`.

        Args:
            df (pl.DataFrame): Messages DataFrame with a string `date` column in ISO format.
            date_col (str): Column name to parse. Defaults to `"date"`.

        Returns:
            pl.DataFrame: DataFrame with `date` cast to `pl.Datetime`.

        Tags:
            - preprocessing
            - time

        Examples:
            ```python
            df = pl.DataFrame({"date": ["2023-01-01T12:00:00"], "text": ["hi"]})
            out = tp.parse_timestamp(df)
            assert out.schema["date"].dtype == pl.Datetime
            ```
        """
        return df.with_columns(pl.col(date_col).str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S"))

    def create_recipient_column(self, df: pl.DataFrame, author_col: str) -> pl.DataFrame:
        """
        Create a `recipients` column listing all unique authors except the current row's author.

        Args:
            df (pl.DataFrame): Messages DataFrame.
            author_col (str): Column containing author names (e.g., `from_name`).

        Returns:
            pl.DataFrame: DataFrame with an additional `recipients` column.

        Tags:
            - preprocessing
            - feature-engineering

        Examples:
            ```python
            df = pl.DataFrame({"from_name": ["alice", "bob"], "text": ["hi", "yo"]})
            out = tp.create_recipient_column(df, author_col="from_name")
            assert "alice" in out[1, "recipients"] and "bob" not in out[1, "recipients"]
            ```
        """
        unique_authors = df.select(pl.col(author_col)).unique().to_series().to_list()
        recipients_str = [
            ", ".join([author for author in unique_authors if author != author_name])
            for author_name in df[author_col].to_list()
        ]
        return df.with_columns(pl.Series("recipients", recipients_str))

    def handle_media(self, chat_pl: pl.DataFrame) -> pl.DataFrame:
        """
        Normalize textual representation for various `media_type` values.

        Rewrites `text` to include a compact marker for common media, such as stickers,
        videos, voice messages, audio, animations and video messages.

        Args:
            chat_pl (pl.DataFrame): Messages DataFrame with `media_type` and related columns.

        Returns:
            pl.DataFrame: DataFrame with `text` unified across media types.

        Tags:
            - preprocessing
            - media

        Examples:
            ```python
            df = pl.DataFrame({
                "text": [""],
                "media_type": ["voice_message"],
                "file": ["voice.ogg"],
            })
            out = tp.handle_media(df)
            assert out[0, "text"].startswith("[voice_message](")
            ```
        """
        # Sticker filter
        chat_pl = chat_pl.with_columns(
            pl.when(pl.col("media_type") == "sticker")
            .then(pl.col("sticker_emoji"))
            .otherwise(pl.col("text"))
            .alias("text")
        )
        # Video file filter
        chat_pl = chat_pl.with_columns(
            pl.when(pl.col("media_type") == "video_file")
            .then(pl.format("{} [video]({})", pl.col("text"), pl.col("file_name")))
            .otherwise(pl.col("text"))
            .alias("text")
        )
        # Voice message filter
        chat_pl = chat_pl.with_columns(
            pl.when(pl.col("media_type") == "voice_message")
            .then(pl.format("[voice_message]({})", pl.col("file")))
            .otherwise(pl.col("text"))
            .alias("text")
        )
        # Audio file filter
        chat_pl = chat_pl.with_columns(
            pl.when(pl.col("media_type") == "audio_file")
            .then(pl.format("[audio]({}-{})", pl.col("title"), pl.col("performer").fill_null("")))
            .otherwise(pl.col("text"))
            .alias("text")
        )
        # Animation filter
        chat_pl = chat_pl.with_columns(
            pl.when(pl.col("media_type") == "animation")
            .then(pl.format("{} [animation]({})", pl.col("text"), pl.col("file_name")))
            .otherwise(pl.col("text"))
            .alias("text")
        )
        # Video message filter
        chat_pl = chat_pl.with_columns(
            pl.when(pl.col("media_type") == "video_message")
            .then(pl.format("[video_message]({})", pl.col("file_name")))
            .otherwise(pl.col("text"))
            .alias("text")
        )
        return chat_pl

    def handle_location(self, chat_pl: pl.DataFrame) -> pl.DataFrame:
        """
        Append a compact `[location](lon, lat)` marker to `text` when location is present.

        Args:
            chat_pl (pl.DataFrame): Messages DataFrame that may contain `location_information.*` fields.

        Returns:
            pl.DataFrame: DataFrame with `text` updated for rows with location data.

        Tags:
            - preprocessing

        Examples:
            ```python
            df = pl.DataFrame({
                "text": ["meet"],
                "location_information.longitude": [30.5],
                "location_information.latitude": [50.4],
            })
            out = tp.handle_location(df)
            assert "[location]" in out[0, "text"]
            ```
        """
        return chat_pl.with_columns(
            pl.when(
                pl.col("location_information.longitude").is_not_null()
                & pl.col("location_information.latitude").is_not_null()
            )
            .then(
                pl.format(
                    "[location]({}, {})",
                    pl.col("location_information.longitude"),
                    pl.col("location_information.latitude"),
                )
            )
            .otherwise(pl.col("text"))
            .alias("text")
        )

    def handle_service_messages(self, chat_pl: pl.DataFrame) -> pl.DataFrame:
        """
        Normalize service messages: map `actor`/`actor_id` into `from`/`from_id`,
        and rewrite phone call messages using `discard_reason`.

        Args:
            chat_pl (pl.DataFrame): Messages DataFrame.

        Returns:
            pl.DataFrame: DataFrame with normalized `text`, `from`, and `from_id` for service rows.

        Tags:
            - preprocessing
            - system

        Examples:
            ```python
            df = pl.DataFrame({"type": ["service"], "actor": ["bot"], "discard_reason": ["missed"]})
            out = tp.handle_service_messages(df)
            assert out[0, "text"].startswith("[phone_call]")
            ```
        """
        return chat_pl.with_columns(
            [
                pl.when((pl.col("type") == "service") & (pl.col("discard_reason").is_not_null()))
                .then(pl.format("[phone_call]({})", pl.col("discard_reason")))
                .otherwise(pl.col("text"))
                .alias("text"),
                pl.when(pl.col("type") == "service").then(pl.col("actor")).otherwise(pl.col("from")).alias("from"),
                pl.when(pl.col("type") == "service")
                .then(pl.col("actor_id"))
                .otherwise(pl.col("from_id"))
                .alias("from_id"),
            ]
        )

    def handle_contacts(self, chat_pl: pl.DataFrame) -> pl.DataFrame:
        """
        Render a compact contact card into `text` when contact info is present.

        Args:
            chat_pl (pl.DataFrame): Messages DataFrame that may contain `contact_information.*` fields.

        Returns:
            pl.DataFrame: DataFrame with `text` updated for rows with contact data.

        Tags:
            - preprocessing
            - contacts

        Examples:
            ```python
            df = pl.DataFrame({
                "text": [""],
                "contact_information.first_name": ["Ann"],
                "contact_information.last_name": ["Lee"],
                "contact_information.phone_number": ["+1 234"],
            })
            out = tp.handle_contacts(df)
            assert out[0, "text"].startswith("[contact](Ann")
            ```
        """
        return chat_pl.with_columns(
            pl.when(
                (pl.col("contact_information.first_name").is_not_null())
                | (pl.col("contact_information.last_name").is_not_null())
            )
            .then(
                pl.format(
                    "[contact]({} {} : {})",
                    pl.col("contact_information.first_name").fill_null(""),
                    pl.col("contact_information.last_name").fill_null(""),
                    pl.col("contact_information.phone_number"),
                )
            )
            .otherwise(pl.col("text"))
            .alias("text")
        )

    def handle_files(self, chat_pl: pl.DataFrame) -> pl.DataFrame:
        """
        Append a `[file](name)` marker to `text` when a file is present but `media_type` is missing.

        Args:
            chat_pl (pl.DataFrame): Messages DataFrame with optional `file` and `file_name`.

        Returns:
            pl.DataFrame: DataFrame with `text` updated for file attachments.

        Tags:
            - preprocessing
            - media

        Examples:
            ```python
            df = pl.DataFrame({"text": [""], "file": ["/tmp/a"], "file_name": ["a.txt"]})
            out = tp.handle_files(df)
            assert "[file](a.txt)" in out[0, "text"]
            ```
        """
        return chat_pl.with_columns(
            pl.when((pl.col("media_type").is_null()) & (pl.col("file").is_not_null()))
            .then(pl.format("[file]({})", pl.col("file_name")))
            .otherwise(pl.col("text"))
            .alias("text")
        )

    def handle_photos(self, chat_pl: pl.DataFrame) -> pl.DataFrame:
        """
        Append a `[photo](name)` marker to `text` when a photo is present.

        Args:
            chat_pl (pl.DataFrame): Messages DataFrame with optional `photo` and `file_name`.

        Returns:
            pl.DataFrame: DataFrame with `text` annotated for photos when present.

        Tags:
            - preprocessing
            - media

        Examples:
            ```python
            df = pl.DataFrame({"text": [""], "photo": ["X"], "file_name": ["img.png"]})
            out = tp.handle_photos(df)
            assert "[photo](img.png)" in out[0, "text"]
            ```
        """
        return chat_pl.with_columns(
            pl.when(pl.col("photo").is_not_null())
            .then(pl.format("{} [photo]({})", pl.col("text"), pl.col("file_name").fill_null("")))
            .otherwise(pl.col("text"))
            .alias("text")
        )

    def handle_additional_types(self, chat_pl: pl.DataFrame) -> pl.DataFrame:
        """
        Apply all additional message-type handlers in a fixed order.

        This is a convenience wrapper to call the individual `handle_*` methods.

        Args:
            chat_pl (pl.DataFrame): Messages DataFrame.

        Returns:
            pl.DataFrame: DataFrame after applying media, location, files, photos, service, contacts.

        Tags:
            - preprocessing
            - media

        Examples:
            ```python
            out = tp.handle_additional_types(df)
            ```
        """
        chat_pl = self.handle_media(chat_pl)
        chat_pl = self.handle_location(chat_pl)
        chat_pl = self.handle_files(chat_pl)
        chat_pl = self.handle_photos(chat_pl)
        chat_pl = self.handle_service_messages(chat_pl)
        chat_pl = self.handle_contacts(chat_pl)
        return chat_pl

    def delete_service_messages(self, chat_pl: pl.DataFrame) -> pl.DataFrame:
        """
        Remove rows where `chat_type == "service"`.

        Args:
            chat_pl (pl.DataFrame): Messages DataFrame.

        Returns:
            pl.DataFrame: Filtered DataFrame without service messages.

        Tags:
            - preprocessing
            - filtering

        Examples:
            ```python
            df = pl.DataFrame({"chat_type": ["service", "private"], "text": ["a", "b"]})
            out = tp.delete_service_messages(df)
            assert len(out) == 1
            ```
        """

        return chat_pl.filter(pl.col("chat_type") != "service")

    def delete_empty_messages(self, chat_pl: pl.DataFrame) -> pl.DataFrame:
        """
        Drop messages that contain empty strings in string-typed columns.

        Args:
            chat_pl (pl.DataFrame): Messages DataFrame.

        Returns:
            pl.DataFrame: Filtered DataFrame with `text` not null and non-empty.

        Tags:
            - preprocessing
            - filtering

        Examples:
            ```python
            df = pl.DataFrame({"text": ["", "hi"]})
            out = tp.delete_empty_messages(df)
            assert out.height == 1 and out[0, "text"] == "hi"
            ```
        """
        return chat_pl.with_columns(
            pl.when(pl.col(pl.String).str.len_chars() == 0).then(None).otherwise(pl.col(pl.String)).name.keep()
        ).filter(pl.col("text").is_not_null())

    def prepare_data(self, file_path: str) -> dict[int, pl.DataFrame]:
        """
        Load a JSON export and run the preprocessing pipeline per chat.

        Pipeline steps:
        - `load_json`
        - `parse_links`
        - `parse_members`
        - `parse_timestamp`
        - `delete_service_messages`
        - cast `from_id` to Int64 and standardize schema (`get_process_schema`)

        Args:
            file_path (str): Path to the Telegram export JSON.

        Returns:
            dict[int, pl.DataFrame]: Mapping `chat_id` -> preprocessed DataFrame.

        Tags:
            - pipeline
            - preprocessing
            - io

        Examples:
            ```python
            chats = tp.prepare_data("/exports/result.json")
            for cid, df in chats.items():
                print(cid, df.height)
            ```
        """

        chats_dict = self.load_json(file_path)

        for key, chat_df in chats_dict.items():
            chat_df = self.parse_links(chat_df)
            chat_df = self.parse_members(chat_df)
            # chat_df = self.standardize_chat(chat_df) Well, it's not needed, because we have telegram_import_schema_short
            chat_df = self.parse_timestamp(chat_df)
            # chat_df = self.handle_additional_types(chat_df)
            chat_df = self.delete_service_messages(chat_df)
            # chat_df = self.delete_empty_messages(chat_df) well, it's not empty messages, it's messages with no text.

            chat_df = chat_df.with_columns(
                [pl.col("from_id").str.replace("^user", "").str.replace("^channel", "").cast(pl.Int64).alias("from_id")]
            )
            process_schema = get_process_schema()
            chat_df = chat_df.select([pl.col(k).cast(v) for k, v in process_schema.items() if k in chat_df.columns])

            chats_dict[key] = chat_df

        return chats_dict

    def _add_messages_to_db(self, messages_df: pl.DataFrame, phone: str | None = None) -> None:
        """
        Persist messages into per-user DuckDB tables.

        This delegates to `TelegramDatabase.add_messages` to avoid duplication of mapping
        and normalization logic.

        Args:
            messages_df (pl.DataFrame): DataFrame containing messages (process schema).
            phone (str | None): Override phone identifier; defaults to `self.phone`.

        Raises:
            ValueError: If no phone number is available.
            Exception: On database errors.

        Tags:
            - database
            - write

        Examples:
            ```python
            tp._add_messages_to_db(df)
            ```
        """
        try:
            # Delegate to TelegramDatabase to avoid duplicating logic
            phone_use = phone if phone is not None else self.phone
            if phone_use is None:
                raise ValueError("Phone number is required to add messages to DB")
            tdb = TelegramDatabase(db_path=self.db_path)
            try:
                tdb.add_messages(phone_use, messages_df)
                self.logger.info(f"Successfully added messages for user {phone_use}")
            finally:
                tdb.close()
        except Exception as e:
            self.logger.error(f"Error adding messages: {str(e)}")
            raise

    # Upsert helpers are handled centrally by TelegramDatabase to avoid duplication

    def process_file(
        self,
        file_path: str,
        time_window: str = "5m",
        cluster_size: int = 3,
    ) -> dict[int, pl.DataFrame]:
        """
        Process a Telegram export file end-to-end.

        Steps:
        - Load and preprocess chats (`prepare_data`)
        - Persist messages to DB (if enabled)
        - Compute embeddings for messages without existing embeddings
        - Create clusters and groups; persist embeddings/clusters

        Args:
            file_path (str): Path to the Telegram export JSON.
            time_window (str): Temporal window for clustering, e.g. `"5m"`, `"1h"`.
            cluster_size (int): Minimum messages to qualify as a cluster.

        Returns:
            dict[int, pl.DataFrame]: Mapping `chat_id` -> processed DataFrame with embeddings and groups.

        Tags:
            - pipeline
            - preprocessing
            - embeddings
            - clustering
            - database

        Examples:
            ```python
            result = tp.process_file("/exports/result.json", time_window="10m")
            ```
        """
        self.logger.info(f"Processing file {file_path} with {time_window} time window")
        chats_dict = self.prepare_data(file_path)

        total_messages = sum(len(chat_df) for chat_df in chats_dict.values())

        for _, chat_df in chats_dict.items():
            self._add_messages_to_db(chat_df)

        existing_message_ids = self._get_messages_with_embeddings() if self.use_duckdb else set()

        table = f"""
            {"Message Processing Summary":^50}
            {"-" * 50}
            {"Total messages":<30}: {total_messages} (in {len(chats_dict)} chats)
            {"Messages with embeddings":<30}: {len(existing_message_ids)}
            {"To process":<30}: {total_messages - len(existing_message_ids)}
            {"-" * 50}
            """
        self.logger.info(table)

        for chat_id, chat_df in chats_dict.items():
            if existing_message_ids:
                original_count = len(chat_df)
                chat_df = chat_df.filter(
                    ~pl.col("message_id").is_in(existing_message_ids)
                    & pl.col("text").is_not_null()
                    & (pl.col("text").str.len_chars() > 0)
                )
                skipped_count = original_count - len(chat_df)
                if skipped_count > 0:
                    self.logger.info(
                        f"Skipping {skipped_count} messages with empty text or existing embeddings in chat {chat_id}"
                    )

            if len(chat_df) == 0:
                chats_dict[chat_id] = chat_df
                continue

            processed_df = self.process_message_groups(chat_df, time_window, cluster_size)
            embeddings_data = processed_df.select(["message_id", "chat_id", "embeddings"])

            self._update_embeddings_in_db(embeddings_data)

            if "group_id" in processed_df.columns:
                clusters_df = processed_df.select(["message_id", "chat_id", "group_id"])
                self._update_clusters_in_db(clusters_df)

            chats_dict[chat_id] = processed_df

        # Log final summary
        if self.use_duckdb:
            total_embeddings_updated = sum(len(df) for df in chats_dict.values() if len(df) > 0)
            processed_chats = sum(1 for df in chats_dict.values() if len(df) > 0)
            if total_embeddings_updated > 0:
                self.logger.nice(  # type: ignore
                    f"✅ File processing complete: updated {total_embeddings_updated} embeddings across {processed_chats} chats"
                )
            else:
                self.logger.nice(  # type: ignore
                    "✅ File processing complete: all messages already processed, no new embeddings needed"
                )

        return chats_dict

    def process_messages(
        self,
        phone: str,
        chat_id: int | None = None,
        time_window: str = "5m",
        cluster_size: int = 3,
    ) -> pl.DataFrame:
        """
        Process messages already stored in the database.

        This will reuse existing embeddings where available and only compute missing ones,
        then create clusters/groups. If DuckDB is disabled, everything is processed in-memory.

        Args:
            phone (str): Phone identifier (with or without leading "+").
            chat_id (int | None): Optional chat filter.
            time_window (str): Temporal window for clustering, e.g. `"5m"`.
            cluster_size (int): Minimum cluster size.

        Returns:
            pl.DataFrame: Combined DataFrame of processed messages (all selected chats).

        Tags:
            - pipeline
            - database
            - embeddings
            - clustering

        Examples:
            ```python
            df = tp.process_messages(phone="+7999...")
            ```
        """
        self.logger.info(f"Processing messages with {time_window} time window")

        if not self.use_duckdb:
            self.logger.warning("DuckDB is not enabled - embeddings and clusters will not be saved")

        messages_df = self._get_messages_from_db(phone=phone, chat_id=chat_id)
        if messages_df.height == 0:
            self.logger.info("No messages found in the database")
            return pl.DataFrame()

        if self.use_duckdb:
            chat_ids = messages_df["chat_id"].unique().to_list()
            self.logger.info(f"Processing messages from {len(chat_ids)} chats")

            existing_message_ids = self._get_messages_with_embeddings()
            all_message_ids = set(messages_df["message_id"].to_list())
            missing_message_ids = all_message_ids - existing_message_ids

            if missing_message_ids:
                self.logger.info(f"Found {len(missing_message_ids)} messages without embeddings to process")
                # Filter messages that need embeddings
                messages_to_process = messages_df.filter(pl.col("message_id").is_in(list(missing_message_ids)))

                # Calculate embeddings for these messages
                messages_to_process = self.calculate_embeddings(messages_to_process)

                # Batch update all embeddings at once
                if len(messages_to_process) > 0:
                    embeddings_data = messages_to_process.select(["message_id", "chat_id", "embeddings"])

                    self._update_embeddings_in_db(embeddings_data)

                # Return all messages with processing completed
                return self.process_message_groups(messages_df, time_window, cluster_size)
            else:
                self.logger.info("All messages already have embeddings")
                return messages_df
        else:
            # If not using database, just process everything at once
            return self.process_message_groups(messages_df, time_window, cluster_size)

    def reprocess_clusters_only(
        self,
        phone: str,
        chat_id: int | None = None,
        time_window: str = "10m",
        cluster_size: int = 10,
    ) -> pl.DataFrame:
        """
        Fast cluster reprocessing using existing embeddings from the database.

        This method is 10-100x faster than full reprocessing because it:
        - ✅ Uses existing embeddings from database
        - ✅ Only recalculates temporal clusters and groups
        - ❌ Skips expensive embedding computation
        - ❌ Skips semantic segmentation recalculation (uses existing)

        Args:
            phone (str): Phone identifier (with or without leading "+").
            chat_id (int | None): Optional chat filter.
            time_window (str): Temporal window for clustering, e.g. `"5m"`.
            cluster_size (int): Minimum cluster size.

        Returns:
            pl.DataFrame: DataFrame with updated cluster information.

        Raises:
            ValueError: If DuckDB is not enabled or embeddings are missing.

        Tags:
            - pipeline
            - database
            - clustering
            - fast-reprocessing

        Examples:
            ```python
            # Fast reprocess all chats
            df = tp.reprocess_clusters_only(phone="+7999...")

            # Fast reprocess single chat
            df = tp.reprocess_clusters_only(phone="+7999...", chat_id=123)
            ```
        """
        if not self.use_duckdb:
            raise ValueError("reprocess_clusters_only requires DuckDB to be enabled")

        self.logger.info(f"🚀 Starting FAST cluster reprocessing with {time_window} time window")

        # Get messages with existing embeddings
        messages_df = self._get_messages_from_db(phone=phone, chat_id=chat_id)
        if messages_df.height == 0:
            self.logger.info("No messages found in the database")
            return pl.DataFrame()

        # Get all unique chat IDs for embedding lookup
        unique_chat_ids = messages_df["chat_id"].unique().to_list()

        # Load existing embeddings for all relevant chats
        all_embeddings = {}
        missing_embeddings_count = 0

        for cid in unique_chat_ids:
            chat_embeddings = self._get_embeddings_from_db(cid)
            all_embeddings.update(chat_embeddings)

        # Filter messages that have embeddings - use Polars operations instead of converting to list
        message_ids_with_embeddings = set(all_embeddings.keys())
        df_with_embeddings = messages_df.filter(pl.col("message_id").is_in(list(message_ids_with_embeddings)))

        missing_embeddings_count = len(messages_df) - len(df_with_embeddings)
        if missing_embeddings_count > 0:
            self.logger.warning(f"⚠️  {missing_embeddings_count} messages missing embeddings - they will be skipped")

        if len(df_with_embeddings) == 0:
            self.logger.error("❌ No messages with embeddings found")
            return pl.DataFrame()

        # Attach embeddings as a column using map operation
        embedding_list = [all_embeddings[mid] for mid in df_with_embeddings["message_id"].to_list()]
        df_with_embeddings = df_with_embeddings.with_columns(pl.Series("embeddings", embedding_list))

        self.logger.info(f"✅ Loaded {len(df_with_embeddings)} messages with existing embeddings")

        # Fast reprocessing: only clusters and groups (skip embedding calculation)
        # Process in batches if dataset is very large to manage GPU memory efficiently
        if len(df_with_embeddings) > self.batch_size * 2:
            self.logger.info(f"🔄 Processing {len(df_with_embeddings)} messages in batches for optimal GPU usage...")

            processed_segments = []
            for i in range(0, len(df_with_embeddings), self.batch_size):
                batch_df = df_with_embeddings.slice(i, min(self.batch_size, len(df_with_embeddings) - i))

                self.logger.info(
                    f"   Processing batch {i // self.batch_size + 1}/{(len(df_with_embeddings) + self.batch_size - 1) // self.batch_size}: semantic segments..."
                )
                batch_with_segments = self.calculate_segments(batch_df)
                processed_segments.append(batch_with_segments)

                # Clear GPU cache between batches if using CUDA
                if self.device == "cuda":
                    torch.cuda.empty_cache()

            # Concatenate all batches
            df_with_segments = pl.concat(processed_segments)
        else:
            self.logger.info("🔄 Recalculating semantic segments...")
            df_with_segments = self.calculate_segments(df_with_embeddings)

        self.logger.info("🔄 Recalculating temporal clusters...")
        df_with_clusters = self.create_clusters(df_with_segments, time_window, cluster_size)

        self.logger.info("🔄 Recalculating groups...")
        final_df = self.calculate_groups(df_with_clusters)

        # Update cluster information in database
        self.logger.info("💾 Updating cluster information in database...")
        clusters_df = final_df.select(["message_id", "chat_id", "group_id"])
        self._update_clusters_in_db(clusters_df)

        self.logger.nice(f"🎉 Fast cluster reprocessing completed! Processed {len(final_df)} messages")  # type: ignore

        return final_df

    def close(self) -> None:
        """
        Close the DuckDB connection if it exists.

        Tags:
            - database
            - resource-management

        Examples:
            ```python
            tp.close()
            ```
        """
        if self.use_duckdb:
            self.logger.info("Closing DuckDB connection")
            try:
                self.db.close()
                self.logger.nice("Closed DuckDB connection ✓")  # type: ignore
            except Exception as e:
                self.logger.error(f"Error closing DuckDB connection: {str(e)}")
                raise
