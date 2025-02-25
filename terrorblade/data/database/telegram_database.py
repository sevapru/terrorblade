import logging
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import duckdb
import polars as pl
from dotenv import load_dotenv

from terrorblade import Logger

load_dotenv(".env")


@dataclass
class ChatStats:
    chat_id: int
    chat_name: str
    message_count: int
    cluster_count: int
    avg_cluster_size: float
    largest_cluster_size: int


@dataclass
class UserStats:
    phone: str
    total_messages: int
    total_chats: int
    largest_chat: Optional[Tuple[int, str, int]]  # (chat_id, chat_name, message_count)
    largest_cluster: Optional[Tuple[int, str, int]]  # (chat_id, chat_name, cluster_size)
    chat_stats: Dict[int, ChatStats]  # chat_id -> ChatStats


class TelegramDatabase:
    def __init__(self, db_path: str = "telegram_data.db", read_only: bool = False):
        """
        Initialize the chat database interface.

        Args:
            db_path (str): Path to the DuckDB database file
            read_only (bool): Whether to open database in read-only mode
        """
        self.db_path = db_path
        self.read_only = read_only

        # Initialize logger
        self.logger = Logger(
            name="ChatDatabaseInterface",
            level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
            log_file=os.getenv("LOG_FILE", "chat_db.log"),
            log_dir=os.getenv("LOG_DIR", "logs"),
        )

        try:
            # Connect to database with appropriate mode
            if read_only:
                self.db = duckdb.connect(db_path, read_only=True)
            else:
                self.db = duckdb.connect(db_path)
                self._init_database()
        except Exception as e:
            self.logger.error(f"Error connecting to database: {str(e)}")
            raise

    def _init_database(self) -> None:
        """Initialize necessary database tables if they don't exist"""
        try:
            # Create users table if it doesn't exist with first_seen field
            self.db.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    phone VARCHAR PRIMARY KEY,
                    last_update TIMESTAMP,
                    first_seen TIMESTAMP
                )
            """
            )

            # Scan for existing message tables to populate users
            table_list = self.db.execute("SHOW TABLES").fetchall()
            existing_tables = [table[0] for table in table_list]

            # Find all message tables and extract phone numbers
            message_tables = [table for table in existing_tables if table.startswith("messages_")]
            for table in message_tables:
                phone = "+" + table.replace("messages_", "")
                # Get first message date as first_seen
                first_seen = self.db.execute(
                    f"""
                    SELECT MIN(date) FROM {table}
                """
                ).fetchone()[0]

                # Add user if not exists
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
        Ensure user exists in the database.

        Args:
            phone (str): User's phone number

        Returns:
            bool: True if user exists or was created, False if tables don't exist
        """
        try:
            messages_table = f"messages_{phone.replace('+', '')}"
            clusters_table = f"message_clusters_{phone.replace('+', '')}"

            # Check if required tables exist
            table_list = self.db.execute("SHOW TABLES").fetchall()
            existing_tables = [table[0] for table in table_list]

            if messages_table not in existing_tables or clusters_table not in existing_tables:
                self.logger.warning(f"Required tables for user {phone} do not exist")
                return False

            # Add user to users table if not exists
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
        """
        try:
            # First ensure users table exists
            if not self.read_only:
                self._init_database()

            result = self.db.execute("SELECT COUNT(*) FROM users").fetchone()
            return result[0] if result else 0
        except Exception as e:
            self.logger.error(f"Error getting user count: {str(e)}")
            raise

    def get_all_users(self) -> List[str]:
        """
        Get list of all user phone numbers in the database.

        Returns:
            List[str]: List of phone numbers
        """
        try:
            # First ensure users table exists
            if not self.read_only:
                self._init_database()

            result = self.db.execute("SELECT phone FROM users ORDER BY phone").fetchall()
            return [row[0] for row in result]
        except Exception as e:
            self.logger.error(f"Error getting users: {str(e)}")
            raise

    def get_user_stats(self, phone: str) -> Optional[UserStats]:
        """
        Get comprehensive statistics for a specific user.

        Args:
            phone (str): User's phone number

        Returns:
            Optional[UserStats]: Dataclass containing user statistics or None if user doesn't exist
        """
        try:
            if not self._ensure_user_exists(phone):
                return None

            messages_table = f"messages_{phone.replace('+', '')}"
            clusters_table = f"message_clusters_{phone.replace('+', '')}"

            # Get total messages
            total_messages = self.db.execute(
                f"""
                SELECT COUNT(*) FROM {messages_table}
            """
            ).fetchone()
            if total_messages is None:
                return None
            total_messages = total_messages[0]

            # Get chat statistics
            chat_stats = self.db.execute(
                f"""
                SELECT 
                    chat_id,
                    chat_name,
                    COUNT(*) as message_count
                FROM {messages_table}
                GROUP BY chat_id, chat_name
                ORDER BY message_count DESC
            """
            ).fetchall()

            # Get largest chat
            largest_chat = chat_stats[0] if chat_stats else None
            largest_chat_tuple = (largest_chat[0], largest_chat[1], largest_chat[2]) if largest_chat else None

            # Get largest cluster
            largest_cluster = self.db.execute(
                f"""
                SELECT 
                    m.chat_id,
                    m.chat_name,
                    COUNT(*) as cluster_size
                FROM {clusters_table} c
                JOIN {messages_table} m ON c.message_id = m.message_id
                GROUP BY c.group_id, m.chat_id, m.chat_name
                ORDER BY cluster_size DESC
                LIMIT 1
            """
            ).fetchone()

            largest_cluster_tuple = (
                (largest_cluster[0], largest_cluster[1], largest_cluster[2]) if largest_cluster else None
            )

            # Compile chat statistics
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

    def get_random_large_cluster(self, phone: str, min_size: int = 10) -> Optional[pl.DataFrame]:
        """
        Get a random large message cluster from any dialog for the specified user.

        Args:
            phone (str): User's phone number
            min_size (int): Minimum cluster size to consider

        Returns:
            Optional[pl.DataFrame]: DataFrame containing cluster messages or None if no clusters found
        """
        try:
            if not self._ensure_user_exists(phone):
                return None

            messages_table = f"messages_{phone.replace('+', '')}"
            clusters_table = f"message_clusters_{phone.replace('+', '')}"

            # Get list of large clusters
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

            # Select random cluster
            selected_cluster = random.choice(large_clusters)
            group_id, chat_id, size = selected_cluster

            # Get messages from the selected cluster
            cluster_messages = self.db.execute(
                f"""
                SELECT 
                    m.*,
                    c.group_id
                FROM {messages_table} m
                JOIN {clusters_table} c ON m.message_id = c.message_id
                WHERE c.group_id = ? AND c.chat_id = ?
                ORDER BY m.date
            """,
                [group_id, chat_id],
            ).arrow()

            return pl.from_arrow(cluster_messages)
        except Exception as e:
            self.logger.error(f"Error getting random large cluster for {phone}: {str(e)}")
            raise

    def get_chat_stats(self, phone: str, chat_id: int) -> Optional[ChatStats]:
        """
        Get detailed statistics for a specific chat.

        Args:
            phone (str): User's phone number
            chat_id (int): Chat ID

        Returns:
            Optional[ChatStats]: Dataclass containing chat statistics or None if chat not found
        """
        try:
            if not self._ensure_user_exists(phone):
                return None

            messages_table = f"messages_{phone.replace('+', '')}"
            clusters_table = f"message_clusters_{phone.replace('+', '')}"

            # Get basic chat info
            chat_info = self.db.execute(
                f"""
                SELECT 
                    chat_id,
                    chat_name,
                    COUNT(*) as message_count
                FROM {messages_table}
                WHERE chat_id = ?
                GROUP BY chat_id, chat_name
            """,
                [chat_id],
            ).fetchone()

            if not chat_info:
                self.logger.warning(f"Chat {chat_id} not found for user {phone}")
                return None

            # Get cluster statistics
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
        Initialize tables for a new user.

        Args:
            phone (str): User's phone number
        """
        try:
            messages_table = f"messages_{phone.replace('+', '')}"
            clusters_table = f"message_clusters_{phone.replace('+', '')}"

            # Create messages table
            self.db.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {messages_table} (
                    message_id BIGINT,
                    chat_id BIGINT,
                    date TIMESTAMP,
                    text TEXT,
                    from_id BIGINT,
                    reply_to_message_id BIGINT,
                    media_type TEXT,
                    file_name TEXT,
                    from TEXT,
                    chat_name TEXT,
                    forwarded_from TEXT,
                    PRIMARY KEY (message_id, chat_id)
                )
            """
            )

            # Create clusters table
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

            self.logger.info(f"Initialized tables for user {phone}")
        except Exception as e:
            self.logger.error(f"Error initializing user tables: {str(e)}")
            raise

    def add_messages(self, phone: str, messages_df: pl.DataFrame) -> None:
        """
        Add messages to the database for a specific user.

        Args:
            phone (str): User's phone number
            messages_df (pl.DataFrame): DataFrame containing messages
        """
        try:
            messages_table = f"messages_{phone.replace('+', '')}"

            # Ensure user tables exist
            self.init_user_tables(phone)

            # Convert DataFrame to DuckDB table
            self.db.execute(
                f"""
                INSERT OR REPLACE INTO {messages_table}
                SELECT * FROM messages_df
            """
            )

            # Update user's last_update and first_seen
            first_seen = self.db.execute(
                f"""
                SELECT MIN(date) FROM {messages_table}
            """
            ).fetchone()[0]

            self.db.execute(
                """
                INSERT OR REPLACE INTO users (phone, last_update, first_seen)
                VALUES (?, CURRENT_TIMESTAMP, ?)
            """,
                [phone, first_seen],
            )

            self.logger.info(f"Added {len(messages_df)} messages for user {phone}")
        except Exception as e:
            self.logger.error(f"Error adding messages: {str(e)}")
            raise

    def get_largest_cluster_messages(self, phone: str) -> Optional[pl.DataFrame]:
        """
        Get messages from the largest cluster for a specific user.

        Args:
            phone (str): User's phone number

        Returns:
            Optional[pl.DataFrame]: DataFrame containing messages from largest cluster
        """
        try:
            if not self._ensure_user_exists(phone):
                return None

            messages_table = f"messages_{phone.replace('+', '')}"
            clusters_table = f"message_clusters_{phone.replace('+', '')}"

            # Find the largest cluster
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

            # Get messages from the largest cluster
            cluster_messages = self.db.execute(
                f"""
                SELECT 
                    m.*,
                    c.group_id
                FROM {messages_table} m
                JOIN {clusters_table} c ON m.message_id = c.message_id
                WHERE c.group_id = ? AND c.chat_id = ?
                ORDER BY m.date
            """,
                [group_id, chat_id],
            ).arrow()

            return pl.from_arrow(cluster_messages)
        except Exception as e:
            self.logger.error(f"Error getting largest cluster messages: {str(e)}")
            raise

    def print_user_summary(self, phone: str) -> None:
        """
        Print comprehensive summary for a specific user.
        Similar to the third_main function but as a class method.

        Args:
            phone (str): User's phone number
        """
        try:
            if user_stats := self.get_user_stats(phone):
                print(f"\nStats for user {phone}:")
                print(f"Total messages: {user_stats.total_messages}")
                print(f"Total chats: {user_stats.total_chats}")

                if user_stats.largest_chat[0]:
                    print(f"Largest chat: {user_stats.largest_chat[1]} ({user_stats.largest_chat[2]} messages)")

                if user_stats.largest_cluster[0]:
                    print(
                        f"Largest cluster: {user_stats.largest_cluster[1]} ({user_stats.largest_cluster[2]} messages)"
                    )

                # Get a random large cluster
                if cluster := self.get_random_large_cluster(phone, min_size=5):
                    print(f"\nRandom cluster size: {len(cluster)}")
                    print("Sample messages from cluster:")
                    print(cluster.select(["text", "date"]).head(3))

                # Get stats for the largest chat
                if user_stats.largest_chat[0]:
                    if chat_stats := self.get_chat_stats(phone, user_stats.largest_chat[0]):
                        print(f"\nStats for largest chat {chat_stats.chat_name}:")
                        print(f"Total messages: {chat_stats.message_count}")
                        print(f"Number of clusters: {chat_stats.cluster_count}")
                        print(f"Average cluster size: {chat_stats.avg_cluster_size:.2f}")
                        print(f"Largest cluster size: {chat_stats.largest_cluster_size}")
            else:
                print(f"No data found for user {phone}")
        except Exception as e:
            self.logger.error(f"Error printing user summary: {str(e)}")
            raise

    def close(self) -> None:
        """Close the database connection"""
        try:
            self.db.close()
            self.logger.info("Database connection closed")
        except Exception as e:
            self.logger.error(f"Error closing database connection: {str(e)}")
            raise
