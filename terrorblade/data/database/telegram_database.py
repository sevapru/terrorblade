import logging
import os
import random
from dataclasses import dataclass
from datetime import datetime

import duckdb
import polars as pl
from dotenv import load_dotenv

from terrorblade import Logger
from terrorblade.data.dtypes import (
    CHAT_NAMES_SCHEMA,
    FILES_SCHEMA,
    FORWARDED_SOURCES_SCHEMA,
    MEDIA_TYPES_SCHEMA,
    TELEGRAM_SCHEMA,
    USER_NAMES_SCHEMA,
)
from terrorblade.utils.config import get_db_path

load_dotenv(".env")


@dataclass
class ChatStats:
    """Aggregate statistics for a chat.

    Attributes:
        chat_id (int): Chat identifier
        chat_name (str): Latest known chat name
        message_count (int): Number of messages in the chat
        cluster_count (int): Number of clusters in the chat
        avg_cluster_size (float): Average cluster size
        largest_cluster_size (int): Size of the largest cluster

    Tags:
        - stats
        - database
    """

    chat_id: int
    chat_name: str
    message_count: int
    cluster_count: int
    avg_cluster_size: float
    largest_cluster_size: int

    def __str__(self) -> str:
        return f"ChatStats(chat_id={self.chat_id}, \
            chat_name={self.chat_name}, \
            message_count={self.message_count}, \
            cluster_count={self.cluster_count}, \
            avg_cluster_size={self.avg_cluster_size}, \
            largest_cluster_size={self.largest_cluster_size})"


@dataclass
class UserStats:
    """Aggregate statistics for a user across all chats.

    Attributes:
        phone (str): Phone identifier
        total_messages (int): Total messages across all chats
        total_chats (int): Number of chats
        largest_chat (tuple[int, str, int] | None): (chat_id, chat_name, message_count)
        largest_cluster (tuple[int, str, int] | None): (chat_id, chat_name, cluster_size)
        chat_stats (dict[int, ChatStats]): Per-chat stats

    Tags:
        - stats
        - database
    """

    phone: str
    total_messages: int
    total_chats: int
    largest_chat: tuple[int, str, int] | None  # (chat_id, chat_name, message_count)
    largest_cluster: tuple[int, str, int] | None  # (chat_id, chat_name, cluster_size)
    chat_stats: dict[int, ChatStats]  # chat_id -> ChatStats

    def __str__(self) -> str:
        return f"UserStats(phone={self.phone}, \
            total_messages={self.total_messages}, \
            total_chats={self.total_chats}, \
            largest_chat={self.largest_chat}, \
            largest_cluster={self.largest_cluster}, \
            chat_stats={self.chat_stats})"


class TelegramDatabase:
    """
    High-level database access layer for Terrorblade.

    Responsibilities:
    - Initialize global and per-user tables
    - Insert messages and auxiliary dictionaries (media types, forwarded sources)
    - Maintain chat/user name mappings with first/last seen
    - Provide querying utilities for stats and cluster retrieval

    Attributes:
        db_path (str): Path to DuckDB database
        read_only (bool): Open in read-only mode if True
        db (duckdb.DuckDBPyConnection): Active DB connection
        logger (Logger): Project logger

    Tags:
        - database
        - schema
        - io
    """

    def __init__(self, db_path: str = "auto", read_only: bool = False) -> None:
        """
        Initialize the chat database interface.

        Args:
            db_path (str): Path to the DuckDB database file or "auto" to use environment/default
            read_only (bool): Whether to open database in read-only mode

        Raises:
            Exception: On connection failure

        Tags:
            - database
            - setup
        """
        self.db_path = get_db_path(db_path)
        self.read_only = read_only

        self.logger = Logger(
            name="ChatDatabaseInterface",
            level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
            log_file=os.getenv("LOG_FILE", "chat_db.log"),
            log_dir=os.getenv("LOG_DIR", "logs"),
        )

        try:
            if read_only:
                self.db = duckdb.connect(db_path, read_only=True)
            else:
                self.db = duckdb.connect(db_path)
                self._init_database()
        except Exception as e:
            self.logger.error(f"Error connecting to database: {str(e)}")
            raise

    def _init_database(self) -> None:
        """Initialize global tables if they don't exist and backfill `users` rows.

        Creates:
        - `users`
        - `media_types` (+ unique index on name)
        - `forwarded_sources` (+ unique index on name)
        - Backfills `users` from existing `messages_*` tables

        Tags:
            - database
            - schema
            - setup
        """
        try:
            self.db.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    phone VARCHAR PRIMARY KEY,
                    last_update TIMESTAMP,
                    first_seen TIMESTAMP
                )
            """
            )

            # Global media types dictionary
            media_cols = ", ".join([f'"{field}" {info["db_type"]}' for field, info in MEDIA_TYPES_SCHEMA.items()])
            self.db.execute(
                f"""
                CREATE TABLE IF NOT EXISTS media_types (
                    {media_cols},
                    PRIMARY KEY (media_type_id)
                )
            """
            )
            self.db.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_media_types_name ON media_types(name)
                """
            )

            # Global forwarded sources dictionary
            fwd_cols = ", ".join([f'"{field}" {info["db_type"]}' for field, info in FORWARDED_SOURCES_SCHEMA.items()])
            self.db.execute(
                f"""
                CREATE TABLE IF NOT EXISTS forwarded_sources (
                    {fwd_cols},
                    PRIMARY KEY (forwarded_from_id)
                )
            """
            )
            self.db.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_forwarded_sources_name ON forwarded_sources(name)
                """
            )

            table_list = self.db.execute("SHOW TABLES").fetchall()
            existing_tables = [table[0] for table in table_list]

            message_tables = [table for table in existing_tables if table.startswith("messages_")]
            for table in message_tables:
                phone = "+" + table.replace("messages_", "")
                result = self.db.execute(
                    f"""
                    SELECT MIN(date) FROM {table}
                """
                ).fetchone()
                first_seen = result[0] if result else None

                self.db.execute(
                    """
                    INSERT OR IGNORE INTO users (phone, last_update, first_seen)
                    SELECT ?, CURRENT_TIMESTAMP, ?
                    WHERE NOT EXISTS (SELECT 1 FROM users WHERE phone = ?)
                """,
                    [phone, first_seen, phone],
                )

            self.logger.info("Database initialization completed")
        except Exception as e:
            self.logger.error(f"Error initializing database: {str(e)}")
            raise

    def _ensure_user_exists(self, phone: str) -> bool:
        """
        Ensure per-user tables exist; optionally register user row.

        Args:
            phone (str): User's phone number

        Returns:
            bool: True if all user tables exist, else False

        Raises:
            Exception: On database failures

        Tags:
            - database
            - validation
        """
        try:
            messages_table = f"messages_{phone.replace('+', '')}"
            clusters_table = f"message_clusters_{phone.replace('+', '')}"
            chat_names_table = f"chat_names_{phone.replace('+', '')}"
            user_names_table = f"user_names_{phone.replace('+', '')}"
            files_table = f"files_{phone.replace('+', '')}"

            table_list = self.db.execute("SHOW TABLES").fetchall()
            existing_tables = [table[0] for table in table_list]

            if (
                messages_table not in existing_tables
                or clusters_table not in existing_tables
                or chat_names_table not in existing_tables
                or user_names_table not in existing_tables
                or files_table not in existing_tables
            ):
                self.logger.warning(f"Required tables for user {phone} do not exist")
                return False

            if not self.read_only:
                self.db.execute(
                    """
                    INSERT OR IGNORE INTO users (phone, last_update)
                    VALUES (?, CURRENT_TIMESTAMP)
                """,
                    [phone],
                )

            return True
        except Exception as e:
            self.logger.error(f"Error ensuring user exists: {str(e)}")
            raise

    def get_user_count(self) -> int:
        """
        Get the total number of users in the database.

        Returns:
            int: Number of users

        Raises:
            Exception: On query failure

        Tags:
            - stats
            - database
        """
        try:
            if not self.read_only:
                self._init_database()

            result = self.db.execute("SELECT COUNT(*) FROM users").fetchone()
            return result[0] if result else 0
        except Exception as e:
            self.logger.error(f"Error getting user count: {str(e)}")
            raise

    def get_all_users(self) -> list[str]:
        """
        Get list of all user phone numbers in the database.

        Returns:
            list[str]: List of phone numbers

        Raises:
            Exception: On query failure

        Tags:
            - stats
            - database
        """
        try:
            if not self.read_only:
                self._init_database()

            result = self.db.execute("SELECT phone FROM users ORDER BY phone").fetchall()
            return [row[0] for row in result]
        except Exception as e:
            self.logger.error(f"Error getting users: {str(e)}")
            raise

    def _latest_chat_names_cte(self, chat_names_table: str) -> str:
        """Return a CTE SQL string that selects the latest chat name per chat_id."""
        return f"""
        WITH latest_names AS (
            SELECT chat_id, chat_name
            FROM (
                SELECT chat_id, chat_name,
                       ROW_NUMBER() OVER (PARTITION BY chat_id ORDER BY COALESCE(last_seen, first_seen) DESC) AS rn
                FROM {chat_names_table}
            ) t
            WHERE rn = 1
        )
        """

    def get_user_stats(self, phone: str) -> UserStats | None:
        """
        Compute user-wide statistics and chat summaries.

        Args:
            phone (str): User's phone number

        Returns:
            UserStats | None: Dataclass with user stats or None if user tables missing

        Raises:
            Exception: On query failure

        Tags:
            - stats
            - database
        """
        try:
            if not self._ensure_user_exists(phone):
                return None

            messages_table = f"messages_{phone.replace('+', '')}"
            clusters_table = f"message_clusters_{phone.replace('+', '')}"
            chat_names_table = f"chat_names_{phone.replace('+', '')}"

            total_messages = self.db.execute(
                f"""
                SELECT COUNT(*) FROM {messages_table}
            """
            ).fetchone()
            if total_messages is None:
                return None
            total_messages = total_messages[0]

            latest_names_cte = self._latest_chat_names_cte(chat_names_table)

            chat_stats = self.db.execute(
                f"""
                {latest_names_cte}
                SELECT
                    m.chat_id,
                    ln.chat_name,
                    COUNT(*) as message_count
                FROM {messages_table} m
                LEFT JOIN latest_names ln ON m.chat_id = ln.chat_id
                GROUP BY m.chat_id, ln.chat_name
                ORDER BY message_count DESC
            """
            ).fetchall()

            largest_chat = chat_stats[0] if chat_stats else None
            largest_chat_tuple = (largest_chat[0], largest_chat[1], largest_chat[2]) if largest_chat else None

            largest_cluster = self.db.execute(
                f"""
                {latest_names_cte}
                SELECT
                    m.chat_id,
                    ln.chat_name,
                    COUNT(*) as cluster_size
                FROM {clusters_table} c
                JOIN {messages_table} m ON c.message_id = m.message_id AND c.chat_id = m.chat_id
                LEFT JOIN latest_names ln ON m.chat_id = ln.chat_id
                GROUP BY c.group_id, m.chat_id, ln.chat_name
                ORDER BY cluster_size DESC
                LIMIT 1
            """
            ).fetchone()

            largest_cluster_tuple = (
                (largest_cluster[0], largest_cluster[1], largest_cluster[2]) if largest_cluster else None
            )

            chat_stats_dict = {}
            for chat_id, chat_name, message_count in chat_stats:
                cluster_stats = self.db.execute(
                    f"""
                    SELECT
                        COUNT(DISTINCT c.group_id) as cluster_count,
                        AVG(cluster_size) as avg_size,
                        MAX(cluster_size) as max_size
                    FROM (
                        SELECT group_id, COUNT(*) as cluster_size
                        FROM {clusters_table}
                        WHERE chat_id = ?
                        GROUP BY group_id
                    ) c
                """,
                    [chat_id],
                ).fetchone()

                chat_stats_dict[chat_id] = ChatStats(
                    chat_id=chat_id,
                    chat_name=chat_name,
                    message_count=message_count,
                    cluster_count=cluster_stats[0] if cluster_stats else 0,
                    avg_cluster_size=cluster_stats[1] if cluster_stats else 0,
                    largest_cluster_size=cluster_stats[2] if cluster_stats else 0,
                )

            return UserStats(
                phone=phone,
                total_messages=total_messages,
                total_chats=len(chat_stats),
                largest_chat=largest_chat_tuple,
                largest_cluster=largest_cluster_tuple,
                chat_stats=chat_stats_dict,
            )
        except Exception as e:
            self.logger.error(f"Error getting user stats for {phone}: {str(e)}")
            raise

    def get_random_large_cluster(self, phone: str, min_size: int = 10) -> pl.DataFrame | pl.Series | None:
        """
        Retrieve a random cluster whose size is at least `min_size` across all chats.

        Args:
            phone (str): User's phone number
            min_size (int): Minimum cluster size to consider

        Returns:
            pl.DataFrame | pl.Series | None: Cluster messages or None if not found

        Raises:
            Exception: On query failure

        Tags:
            - database
            - clusters
        """
        try:
            if not self._ensure_user_exists(phone):
                return None

            messages_table = f"messages_{phone.replace('+', '')}"
            clusters_table = f"message_clusters_{phone.replace('+', '')}"

            large_clusters = self.db.execute(
                f"""
                WITH cluster_sizes AS (
                    SELECT
                        group_id,
                        chat_id,
                        COUNT(*) as size
                    FROM {clusters_table}
                    GROUP BY group_id, chat_id
                    HAVING COUNT(*) >= ?
                )
                SELECT
                    cs.group_id,
                    cs.chat_id,
                    cs.size
                FROM cluster_sizes cs
            """,
                [min_size],
            ).fetchall()

            if not large_clusters:
                self.logger.info(f"No clusters found with size >= {min_size}")
                return None

            selected_cluster = random.choice(large_clusters)
            group_id, chat_id, size = selected_cluster

            where_clause = "WHERE c.group_id = ? AND c.chat_id = ?"
            sql = self.__get_join_messages_clusters_sql(messages_table, clusters_table, where_clause)

            cluster_messages = self.db.execute(
                sql,
                [group_id, chat_id],
            ).arrow()

            return pl.from_arrow(cluster_messages)
        except Exception as e:
            self.logger.error(f"Error getting random large cluster: {str(e)}")
            raise

    def get_chat_stats(self, phone: str, chat_id: int) -> ChatStats | None:
        """
        Return statistics for a specific chat.

        Args:
            phone (str): User's phone number
            chat_id (int): Chat ID

        Returns:
            ChatStats | None: Stats for the chat, or None if not found

        Raises:
            Exception: On query failure

        Tags:
            - stats
            - database
        """
        try:
            if not self._ensure_user_exists(phone):
                return None

            messages_table = f"messages_{phone.replace('+', '')}"
            clusters_table = f"message_clusters_{phone.replace('+', '')}"
            chat_names_table = f"chat_names_{phone.replace('+', '')}"

            latest_names_cte = self._latest_chat_names_cte(chat_names_table)

            chat_info = self.db.execute(
                f"""
                {latest_names_cte}
                SELECT
                    m.chat_id,
                    ln.chat_name,
                    COUNT(*) as message_count
                FROM {messages_table} m
                LEFT JOIN latest_names ln ON m.chat_id = ln.chat_id
                WHERE m.chat_id = ?
                GROUP BY m.chat_id, ln.chat_name
            """,
                [chat_id],
            ).fetchone()

            if not chat_info:
                self.logger.warning(f"Chat {chat_id} not found for user {phone}")
                return None

            cluster_stats = self.db.execute(
                f"""
                SELECT
                    COUNT(DISTINCT c.group_id) as cluster_count,
                    AVG(cluster_size) as avg_size,
                    MAX(cluster_size) as max_size
                FROM (
                    SELECT group_id, COUNT(*) as cluster_size
                    FROM {clusters_table}
                    WHERE chat_id = ?
                    GROUP BY group_id
                ) c
            """,
                [chat_id],
            ).fetchone()

            return ChatStats(
                chat_id=chat_info[0],
                chat_name=chat_info[1],
                message_count=chat_info[2],
                cluster_count=cluster_stats[0] if cluster_stats else 0,
                avg_cluster_size=cluster_stats[1] if cluster_stats else 0,
                largest_cluster_size=cluster_stats[2] if cluster_stats else 0,
            )
        except Exception as e:
            self.logger.error(f"Error getting chat stats for chat {chat_id}, user {phone}: {str(e)}")
            raise

    def init_user_tables(self, phone: str) -> None:
        """
        Create per-user messages, clusters, name mapping, and files tables.

        Args:
            phone (str): User's phone number

        Raises:
            Exception: On creation failure

        Tags:
            - database
            - schema
            - setup

        Examples:
            ```python
            db.init_user_tables("+7999...")
            ```
        """
        try:
            messages_table = f"messages_{phone.replace('+', '')}"
            clusters_table = f"message_clusters_{phone.replace('+', '')}"
            chat_names_table = f"chat_names_{phone.replace('+', '')}"
            user_names_table = f"user_names_{phone.replace('+', '')}"
            files_table = f"files_{phone.replace('+', '')}"

            # Ensure global tables
            self._init_database()

            # Messages table
            columns = [f'"{field}" {info["db_type"]}' for field, info in TELEGRAM_SCHEMA.items()]
            self.db.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {messages_table} (
                {", ".join(columns)},
                PRIMARY KEY (message_id, chat_id)
            )
            """
            )

            # Clusters table
            self.db.execute(self.__get_create_clusters_table_sql(clusters_table))

            # Name mapping tables
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

            self.db.execute(
                """
                INSERT OR REPLACE INTO users (phone, last_update, first_seen)
                VALUES (?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """,
                [phone],
            )

            self.logger.info(f"Initialized tables for user {phone}")
        except Exception as e:
            self.logger.error(f"Error initializing user tables: {str(e)}")
            raise

    def _upsert_media_types(self, names: list[str]) -> dict[str, int]:
        """Ensure media types exist and return mapping name->id.

        Args:
            names (list[str]): Media type names.

        Returns:
            dict[str, int]: Mapping from media name to assigned integer id.

        Tags:
            - database
            - dictionary
        """
        if not names:
            return {}
        # Read existing mapping
        rows = self.db.execute("SELECT media_type_id, name FROM media_types").fetchall()
        name_to_id: dict[str, int] = {name: mtid for mtid, name in rows}
        # Assign new ids
        new_names = [n for n in set(names) if n is not None and n not in name_to_id]
        if new_names:
            # Compute next id
            max_id_row = self.db.execute("SELECT COALESCE(MAX(media_type_id), 0) FROM media_types").fetchone()
            next_id = (max_id_row[0] or 0) + 1  # type: ignore # TODO: check if this is correct
            to_insert = [(next_id + i, n) for i, n in enumerate(sorted(new_names))]
            self.db.executemany("INSERT OR IGNORE INTO media_types(media_type_id, name) VALUES (?, ?)", to_insert)
            for mid, n in to_insert:
                name_to_id[n] = mid
        return name_to_id

    def _upsert_forwarded_sources(self, names: list[str]) -> dict[str, int]:
        """Ensure forwarded source names exist and return mapping name->id.

        Args:
            names (list[str]): Forwarded source names.

        Returns:
            dict[str, int]: Mapping name->id.

        Tags:
            - database
            - dictionary
        """
        if not names:
            return {}
        rows = self.db.execute("SELECT forwarded_from_id, name FROM forwarded_sources").fetchall()
        name_to_id: dict[str, int] = {name: fid for fid, name in rows}
        new_names = [n for n in set(names) if n is not None and n not in name_to_id]
        if new_names:
            max_id_row = self.db.execute("SELECT COALESCE(MAX(forwarded_from_id), 0) FROM forwarded_sources").fetchone()
            next_id = (max_id_row[0] or 0) + 1  # type: ignore # TODO: check if this is correct
            to_insert = [(next_id + i, n) for i, n in enumerate(sorted(new_names))]
            self.db.executemany(
                "INSERT OR IGNORE INTO forwarded_sources(forwarded_from_id, name) VALUES (?, ?)",
                to_insert,
            )
            for fid, n in to_insert:
                name_to_id[n] = fid
        return name_to_id

    def _convert_string_column_to_mapping(
        self,
        df: pl.DataFrame,
        source_column: str,
        target_column: str,
        mapping: dict[str, int],
    ) -> pl.DataFrame:
        """Convert string column to integer using provided mapping.

        Args:
            df (pl.DataFrame): DataFrame to transform
            source_column (str): Name of the source string column
            target_column (str): Name of the target integer column
            mapping (dict[str, int]): String to integer mapping

        Returns:
            pl.DataFrame: DataFrame with the target column added/updated

        Tags:
            - database
            - conversion
        """
        if not mapping:
            return df.with_columns(pl.lit(None).cast(pl.Int32).alias(target_column))

        return df.with_columns(
            pl.col(source_column)
            .map_elements(
                lambda x: mapping.get(x) if isinstance(x, str) else None,
                return_dtype=pl.Int64,
                skip_nulls=False,
            )
            .cast(pl.Int32)
            .alias(target_column)
        )

    def _upsert_name_mappings(self, phone: str, messages_df: pl.DataFrame) -> None:
        """Insert/update chat and user name mappings based on messages_df.

        Args:
            phone (str): Phone identifier
            messages_df (pl.DataFrame): Messages DataFrame containing `chat_id`, `chat_name`, `from_id`, `from_name`, `date`.

        Tags:
            - database
            - dictionary
        """
        phone_clean = phone.replace("+", "")
        chat_names_table = f"chat_names_{phone_clean}"
        user_names_table = f"user_names_{phone_clean}"

        if "date" not in messages_df.columns:
            return

        # Chat names: chat_id, chat_name
        if "chat_id" in messages_df.columns and "chat_name" in messages_df.columns:
            chat_pairs = messages_df.select(["chat_id", "chat_name", "date"]).drop_nulls(
                subset=["chat_id", "chat_name"]
            )
            if chat_pairs.height > 0:
                agg = chat_pairs.group_by(["chat_id", "chat_name"]).agg(
                    [
                        pl.col("date").min().alias("first_seen"),
                        pl.col("date").max().alias("last_seen"),
                    ]
                )
                self.db.register("chat_pairs", agg)
                self.db.execute(
                    f"""
                    INSERT OR REPLACE INTO {chat_names_table}(chat_id, chat_name, first_seen, last_seen)
                    SELECT chat_id, chat_name, first_seen, last_seen FROM chat_pairs
                    """
                )

                # Validation: log chats with multiple distinct names
                dupes = self.db.execute(
                    f"""
                    SELECT chat_id, COUNT(DISTINCT chat_name) AS name_count
                    FROM {chat_names_table}
                    GROUP BY chat_id
                    HAVING COUNT(DISTINCT chat_name) > 1
                    """
                ).fetchall()
                for _row in dupes:
                    self.logger.warning("Chat {_row[0]} has {_row[1]} distinct names")

        # User names: from_id, from_name
        if "from_id" in messages_df.columns and "from_name" in messages_df.columns:
            user_pairs = messages_df.select(["from_id", "from_name", "date"]).drop_nulls(
                subset=["from_id", "from_name"]
            )
            if user_pairs.height > 0:
                agg_u = user_pairs.group_by(["from_id", "from_name"]).agg(
                    [
                        pl.col("date").min().alias("first_seen"),
                        pl.col("date").max().alias("last_seen"),
                    ]
                )
                self.db.register("user_pairs", agg_u)
                self.db.execute(
                    f"""
                    INSERT OR REPLACE INTO {user_names_table}(from_id, from_name, first_seen, last_seen)
                    SELECT from_id, from_name, first_seen, last_seen FROM user_pairs
                    """
                )

        # Files: message_id, chat_id, file_name
        files_table = f"files_{phone_clean}"
        if {"message_id", "chat_id", "file_name"}.issubset(set(messages_df.columns)):
            files_df = messages_df.select(["message_id", "chat_id", "file_name"]).drop_nulls(
                subset=["message_id", "chat_id", "file_name"]
            )
            if files_df.height > 0:
                self.db.register("files_df", files_df)
                self.db.execute(
                    f"""
                    INSERT OR REPLACE INTO {files_table}(message_id, chat_id, file_name)
                    SELECT message_id, chat_id, file_name FROM files_df
                    """
                )

    def add_messages(self, phone: str, messages_df: pl.DataFrame) -> None:
        """
        Insert messages into a user's `messages_<phone>` table.

        Args:
            phone (str): User's phone number
            messages_df (pl.DataFrame): DataFrame containing messages following `TELEGRAM_SCHEMA`

        Raises:
            Exception: On insertion failure

        Tags:
            - database
            - write
        """
        try:
            if not self._ensure_user_exists(phone):
                return

            messages_table = f"messages_{phone.replace('+', '')}"

            self.logger.info(f"Adding {messages_df.height} messages for user {phone}")

            # Ensure all schema columns exist in DataFrame with correct types
            for field, info in TELEGRAM_SCHEMA.items():
                if field not in messages_df.columns:
                    default_val = None
                    messages_df = messages_df.with_columns(pl.lit(default_val).cast(info["polars_type"]).alias(field))

            # Map media_type strings -> ints via media_types table
            if "media_type" in messages_df.columns:
                media_names = [
                    x for x in messages_df.select("media_type").to_series().drop_nulls().to_list() if isinstance(x, str)
                ]
                mapping = self._upsert_media_types(media_names)
                messages_df = self._convert_string_column_to_mapping(messages_df, "media_type", "media_type", mapping)

            # Map forwarded_from -> forwarded_from_id
            if "forwarded_from" in messages_df.columns:
                fwd_names = [
                    x
                    for x in messages_df.select("forwarded_from").to_series().drop_nulls().to_list()
                    if isinstance(x, str)
                ]
                fwd_map = self._upsert_forwarded_sources(fwd_names)
                messages_df = self._convert_string_column_to_mapping(
                    messages_df, "forwarded_from", "forwarded_from_id", fwd_map
                )

            # Upsert name mappings
            self._upsert_name_mappings(phone, messages_df)

            # Register and insert into messages table using schema order
            # Drop legacy string cols not present in TELEGRAM_SCHEMA
            cols_to_keep = list(TELEGRAM_SCHEMA.keys())
            # Ensure DataFrame has those columns in any order
            missing = [c for c in cols_to_keep if c not in messages_df.columns]
            for c in missing:
                messages_df = messages_df.with_columns(pl.lit(None).cast(TELEGRAM_SCHEMA[c]["polars_type"]).alias(c))
            messages_df = messages_df.select(cols_to_keep)
            self.db.register("messages_df", messages_df)
            self.db.execute(
                f"""INSERT OR IGNORE INTO {messages_table} ({", ".join(list(TELEGRAM_SCHEMA.keys()))}) SELECT {", ".join(list(TELEGRAM_SCHEMA.keys()))} FROM messages_df"""
            )

            first_seen = self.db.execute(
                f"""
                SELECT MIN(date) FROM {messages_table}
                """
            ).fetchone()
            first_seen = first_seen[0] if first_seen else None

            self.db.execute(
                """
                INSERT OR REPLACE INTO users (phone, last_update, first_seen)
                VALUES (?, ?, ?)
                """,
                [phone, datetime.now(), first_seen or datetime.now()],
            )

            self.logger.info(f"Successfully added messages for user {phone}")
        except Exception as e:
            self.logger.error(f"Error adding messages: {str(e)}")
            raise

    def get_largest_cluster_messages(self, phone: str) -> pl.DataFrame | pl.Series | None:
        """
        Get messages from the largest cluster for a specific user.

        Args:
            phone (str): User's phone number

        Returns:
            pl.DataFrame | None: DataFrame containing messages from the largest cluster

        Tags:
            - database
            - clusters
        """
        try:
            if not self._ensure_user_exists(phone):
                return None

            messages_table = f"messages_{phone.replace('+', '')}"
            clusters_table = f"message_clusters_{phone.replace('+', '')}"
            largest_cluster = self.db.execute(
                f"""
                WITH cluster_sizes AS (
                    SELECT
                        group_id,
                        chat_id,
                        COUNT(*) as size
                    FROM {clusters_table}
                    GROUP BY group_id, chat_id
                )
                SELECT
                    group_id,
                    chat_id,
                    size
                FROM cluster_sizes
                ORDER BY size DESC
                LIMIT 1
            """
            ).fetchone()

            if not largest_cluster:
                self.logger.info("No clusters found")
                return None

            group_id, chat_id, size = largest_cluster
            where_clause = "WHERE c.group_id = ? AND c.chat_id = ?"
            sql = self.__get_join_messages_clusters_sql(messages_table, clusters_table, where_clause)

            cluster_messages = self.db.execute(
                sql,
                [group_id, chat_id],
            ).arrow()

            return pl.from_arrow(cluster_messages)
        except Exception as e:
            self.logger.error(f"Error getting largest cluster messages: {str(e)}")
            raise

    def print_user_summary(self, phone: str) -> None | UserStats:
        """
        Print a concise summary of a user's message and cluster activity.

        Args:
            phone (str): User's phone number

        Returns:
            UserStats | None: The computed stats if available

        Tags:
            - stats
            - reporting
        """
        user_count = self.get_user_count()
        self.logger.info(f"\nTotal users in database: {user_count}")
        users = self.get_all_users()
        self.logger.info(f"Users: {users}")

        if user_stats := self.get_user_stats(phone):
            self.logger.info(f"\nStats for user {phone}:")
            self.logger.info(f"Total messages: {user_stats.total_messages}")
            self.logger.info(f"Total chats: {user_stats.total_chats}")
            if user_stats and user_stats.chat_stats:
                sorted_chats = sorted(user_stats.chat_stats.values(), key=lambda x: x.message_count, reverse=True)
                top_3_chats = sorted_chats[:3]

                self.logger.info(f"Top {len(top_3_chats)} largest chats:")
                for i, chat in enumerate(top_3_chats, 1):
                    self.logger.info(f"  {i}. {chat.chat_name} ({chat.message_count} messages)")

            if user_stats.largest_cluster and user_stats.largest_cluster[0] is not None:
                self.logger.info(
                    f"Largest cluster: {user_stats.largest_cluster[1]} ({user_stats.largest_cluster[2]} messages)"
                )

            cluster = self.get_random_large_cluster(phone, min_size=5)
            if cluster is not None and not cluster.is_empty():
                self.logger.info(f"\nRandom cluster size: {len(cluster)}")
                self.logger.info("Sample messages from cluster:")
                self.logger.info(cluster.select(["text", "date"]).head(3))  # type: ignore

            largest_cluster = self.get_largest_cluster_messages(phone)
            if largest_cluster is not None and not largest_cluster.is_empty():
                self.logger.info(f"\nLargest cluster messages ({len(largest_cluster)} messages):")
                self.logger.info(largest_cluster.select(["text", "date"]).head(5))  # type: ignore

            if (
                user_stats
                and user_stats.largest_chat
                and user_stats.largest_chat[0] is not None
                and (chat_stats := self.get_chat_stats(phone, user_stats.largest_chat[0]))
            ):
                self.logger.info(f"\nStats for largest chat {chat_stats.chat_name}:")
                self.logger.info(f"Total messages: {chat_stats.message_count}")
                if chat_stats.avg_cluster_size is not None:
                    self.logger.info(f"Average cluster size: {chat_stats.avg_cluster_size:.2f}")
                if chat_stats.largest_cluster_size is not None:
                    self.logger.info(f"Largest cluster size: {chat_stats.largest_cluster_size}")
            return user_stats
        else:
            self.logger.info(f"No data found for user {phone}")
            return None

    def close(self) -> None:
        """Close the database connection.

        Tags:
            - database
            - resource-management
        """
        try:
            self.db.close()
            self.logger.info("Database connection closed")
        except Exception as e:
            self.logger.error(f"Error closing database connection: {str(e)}")
            raise

    def get_max_message_id(self, phone: str, chat_id: int) -> int:
        """
        Get the maximum `message_id` for a chat.

        Args:
            phone (str): User's phone number
            chat_id (int): Chat ID

        Returns:
            int: Maximum message_id or -1 if no messages found

        Tags:
            - database
            - stats
        """
        try:
            if self.read_only and not self._ensure_user_exists(phone):
                self.logger.info(f"User {phone} does not exist in database")
                return -1

            messages_table = f"messages_{phone.replace('+', '')}"

            table_list = self.db.execute("SHOW TABLES").fetchall()
            existing_tables = [table[0] for table in table_list]

            if messages_table not in existing_tables:
                self.logger.info(f"Table {messages_table} does not exist yet")
                return -1

            result = self.db.execute(
                f"""
                SELECT MAX(message_id) FROM {messages_table}
                WHERE chat_id = ?
                """,
                [chat_id],
            ).fetchone()

            if result and result[0] is not None:
                self.logger.info(f"Found max message_id {result[0]} for chat {chat_id}, user {phone}")
                return result[0]

            self.logger.info(f"No messages found for chat {chat_id}, user {phone}")
            return -1
        except Exception as e:
            self.logger.error(f"Error getting max message_id for chat {chat_id}, user {phone}: {str(e)}")
            return -1

    def __get_create_clusters_table_sql(self, table_name: str) -> str:
        """Generate SQL for creating a clusters table."""
        return f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            message_id BIGINT,
            chat_id BIGINT,
            group_id INTEGER,
            PRIMARY KEY (message_id, chat_id)
        )
        """

    def __get_join_messages_clusters_sql(
        self,
        messages_table: str,
        clusters_table: str,
        where_clause: str = "",
        order_by: str = "m.date",
    ) -> str:
        """Generate SQL for joining messages and clusters tables."""
        column_list = ", ".join([f"m.{field}" for field in TELEGRAM_SCHEMA])
        sql = f"""
        SELECT
            {column_list},
            c.group_id
        FROM {messages_table} m
        JOIN {clusters_table} c ON m.message_id = c.message_id AND m.chat_id = c.chat_id
        {where_clause}
        """
        if order_by:
            sql += f" ORDER BY {order_by}"
        return sql

    def __get_messages_select_sql(self, messages_table: str, where_clause: str = "", order_by: str = "date") -> str:
        """Generate SQL for selecting messages from the messages table."""
        column_list = ", ".join(TELEGRAM_SCHEMA.keys())
        sql = f"""
        SELECT
            {column_list}
        FROM {messages_table}
        {where_clause}
        """
        if order_by:
            sql += f" ORDER BY {order_by}"
        return sql
