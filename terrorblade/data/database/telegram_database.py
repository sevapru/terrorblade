import logging
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import duckdb
import polars as pl
from dotenv import load_dotenv
from datetime import datetime

from terrorblade import Logger
from terrorblade.data.dtypes import TELEGRAM_SCHEMA

load_dotenv(".env")


@dataclass
class ChatStats:
    chat_id: int
    chat_name: str
    message_count: int
    cluster_count: int
    avg_cluster_size: float
    largest_cluster_size: int

    def __str__(self):
        return f"ChatStats(chat_id={self.chat_id}, \
            chat_name={self.chat_name}, \
            message_count={self.message_count}, \
            cluster_count={self.cluster_count}, \
            avg_cluster_size={self.avg_cluster_size}, \
            largest_cluster_size={self.largest_cluster_size})"


@dataclass
class UserStats:
    phone: str
    total_messages: int
    total_chats: int
    largest_chat: Optional[Tuple[int, str, int]]  # (chat_id, chat_name, message_count)
    largest_cluster: Optional[Tuple[int, str, int]]  # (chat_id, chat_name, cluster_size)
    chat_stats: Dict[int, ChatStats]  # chat_id -> ChatStats

    def __str__(self):
        return f"UserStats(phone={self.phone}, \
            total_messages={self.total_messages}, \
            total_chats={self.total_chats}, \
            largest_chat={self.largest_chat}, \
            largest_cluster={self.largest_cluster}, \
            chat_stats={self.chat_stats})"


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

            # Get messages from the selected cluster using centralized SQL generation
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

            # Create messages table using the centralized schema
            # Quote column names to avoid reserved keyword issues
            columns = [f'"{field}" {info["db_type"]}' for field, info in TELEGRAM_SCHEMA.items()]
            self.db.execute(f"""
                CREATE TABLE IF NOT EXISTS {messages_table} (
                {", ".join(columns)},
                PRIMARY KEY (message_id, chat_id)
            )
            """)

            # Create clusters table
            self.db.execute(self.__get_create_clusters_table_sql(clusters_table))

            # Add user to users table if not exists
            self.db.execute(
                """
                INSERT OR IGNORE INTO users (phone, last_update, first_seen)
                VALUES (?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """,
                [phone],
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
            if not self._ensure_user_exists(phone):
                return

            messages_table = f"messages_{phone.replace('+', '')}"

            self.logger.info(f"Adding {messages_df.height} messages for user {phone}")

            # Check if from_name is missing in the DataFrame
            if "from_name" not in messages_df.columns:
                # Add from_name column with NULL values
                messages_df = messages_df.with_columns(pl.lit(None).alias("from_name"))
                self.logger.info(f"Added missing 'from_name' column to DataFrame for user {phone}")


            # Convert DataFrame to DuckDB table
            self.db.execute(
                f"""INSERT OR IGNORE INTO {messages_table} ({', '.join(list(TELEGRAM_SCHEMA.keys()))}) SELECT {', '.join(list(TELEGRAM_SCHEMA.keys()))} FROM messages_df"""
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
                VALUES (?, ?, ?)
                """,
                [phone, datetime.now(), first_seen or datetime.now()],
            )

            self.logger.info(f"Successfully added messages for user {phone}")
        except Exception as e:
            self.logger.error(f"Error adding messages: {str(e)}")
            raise

    def get_largest_cluster_messages(self, phone: str) -> Optional[pl.DataFrame]:
        """
        Get messages from the largest cluster for a specific user.

        Args:
            phone (str): User's phone number

        Returns:
            Optional[pl.DataFrame]: DataFrame containing messages from the largest cluster
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

            # Get messages from the largest cluster using centralized SQL generation
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

    def print_user_summary(self, phone: str) -> None:
        """
        Print comprehensive summary for a specific user.
        Similar to the third_main function but as a class method.

        Args:
            phone (str): User's phone number
        """
        try:
            if user_stats := self.get_user_stats(phone):
                print(user_stats)

                if user_stats.largest_chat:
                    print(f"Largest chat: {user_stats.largest_chat[1]} ({user_stats.largest_chat[2]} messages)")
                    if chat_stats := self.get_chat_stats(phone, user_stats.largest_chat[0]):
                        print(chat_stats)
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

    def get_max_message_id(self, phone: str, chat_id: int) -> int:
        """
        Get the maximum message_id for a specific chat for a specific user.

        Args:
            phone (str): User's phone number
            chat_id (int): Chat ID

        Returns:
            int: Maximum message_id or -1 if no messages found
        """
        try:
            if self.read_only and not self._ensure_user_exists(phone):
                self.logger.info(f"User {phone} does not exist in database")
                return -1
                
            messages_table = f"messages_{phone.replace('+', '')}"

            # Check if the table exists
            table_list = self.db.execute("SHOW TABLES").fetchall()
            existing_tables = [table[0] for table in table_list]

            if messages_table not in existing_tables:
                self.logger.info(f"Table {messages_table} does not exist yet")
                return -1

            # Get the maximum message_id - no need to change this as we're only querying one field
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

    def __get_join_messages_clusters_sql(self, messages_table: str, clusters_table: str, where_clause: str = "", order_by: str = "m.date") -> str:
        """Generate SQL for joining messages and clusters tables."""
        column_list = ", ".join([f"m.{field}" for field in TELEGRAM_SCHEMA.keys()])
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
