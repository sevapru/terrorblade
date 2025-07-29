import logging
import os

import duckdb

from terrorblade import Logger


class SessionManager:
    """
    Manages Telegram session storage and retrieval from a database.

    This class handles storing session strings in a DuckDB database,
    allowing sessions to be reused across runs without storing them as files.
    """

    def __init__(self, db_path: str = "telegram_sessions.db") -> None:
        """
        Initialize the session manager.

        Args:
            db_path (str): Path to the DuckDB database file for storing sessions
        """
        self.db_path = db_path

        # Initialize logger
        self.logger = Logger(
            name="SessionManager",
            level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
            log_file=os.getenv("LOG_FILE", "telegram.log"),
            log_dir=os.getenv("LOG_DIR", "logs"),
        )

        try:
            # Connect to database
            self.db = duckdb.connect(db_path)
            self._init_database()
        except Exception as e:
            self.logger.error(f"Error connecting to session database: {str(e)}")
            raise

    def _init_database(self) -> None:
        """Initialize the sessions table if it doesn't exist"""
        try:
            self.db.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    phone VARCHAR PRIMARY KEY,
                    session_data TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )
            self.logger.info("Session database initialized")
        except Exception as e:
            self.logger.error(f"Error initializing session database: {str(e)}")
            raise

    def get_session(self, phone: str | None) -> str | None:
        """
        Retrieve a session string for the given phone number.

        Args:
            phone (str): Phone number to retrieve session for

        Returns:
            Optional[str]: Session string if found, None otherwise
        """
        try:
            result = self.db.execute(
                "SELECT session_data FROM sessions WHERE phone = ?", [phone]
            ).fetchone()

            if result:
                self.logger.info(f"Retrieved existing session for phone {phone}")
                # Update last used timestamp
                self.db.execute(
                    "UPDATE sessions SET last_used = CURRENT_TIMESTAMP WHERE phone = ?", [phone]
                )
                return result[0]
            else:
                self.logger.info(f"No existing session found for phone {phone}")
                return None
        except Exception as e:
            self.logger.error(f"Error retrieving session: {str(e)}")
            return None

    def save_session(self, phone: str, session_string: str) -> bool:
        """
        Save a session string for the given phone number.

        Args:
            phone (str): Phone number to save session for
            session_string (str): Session string to save

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.db.execute(
                """
                INSERT OR REPLACE INTO sessions (phone, session_data, last_used)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            """,
                [phone, session_string],
            )

            self.logger.info(f"Saved session for phone {phone}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving session: {str(e)}")
            return False

    def delete_session(self, phone: str) -> bool:
        """
        Delete a session for the given phone number.

        Args:
            phone (str): Phone number to delete session for

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.db.execute("DELETE FROM sessions WHERE phone = ?", [phone])
            self.logger.info(f"Deleted session for phone {phone}")
            return True
        except Exception as e:
            self.logger.error(f"Error deleting session: {str(e)}")
            return False

    def list_sessions(self) -> list:
        """
        List all stored sessions.

        Returns:
            list: List of tuples (phone, created_at, last_used)
        """
        try:
            result = self.db.execute(
                """
                SELECT phone, created_at, last_used
                FROM sessions
                ORDER BY last_used DESC
            """
            ).fetchall()
            return result
        except Exception as e:
            self.logger.error(f"Error listing sessions: {str(e)}")
            return []

    def close(self) -> None:
        """Close the database connection"""
        try:
            self.db.close()
            self.logger.info("Session database connection closed")
        except Exception as e:
            self.logger.error(f"Error closing session database: {str(e)}")
