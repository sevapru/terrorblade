import asyncio
import logging
import os
from pathlib import Path

import polars as pl
from dotenv import load_dotenv
from telethon import TelegramClient
from telethon.errors import FloodWaitError
from telethon.sessions import StringSession
from telethon.tl.types import MessageService

from terrorblade import Logger
from terrorblade.data.database.session_manager import SessionManager
from terrorblade.data.database.telegram_database import TelegramDatabase
from terrorblade.data.dtypes import get_polars_schema

load_dotenv(".env")


class TelegramParser:
    def __init__(
        self,
        phone: str,
        api_id: str | None = None,
        api_hash: str | None = None,
        db: TelegramDatabase | None = None,
        session_db_path: str = "telegram_sessions.db",
    ):
        """
        Initialize Telegram parser.

        Args:
            api_id (str): API ID from Telegram
            api_hash (str): API Hash from Telegram
            phone (str): Phone number for authentication
            db (Optional[TelegramDatabase]): Database instance for storing messages
            session_db_path (str): Path to the session database
        """
        self.api_id = api_id or os.getenv("API_ID")
        self.api_hash = api_hash or os.getenv("API_HASH")
        self.phone = phone or os.getenv("PHONE")
        self.client = None
        self.session_db_path = session_db_path or str(
            Path(os.getenv("LOG_DIR", "data")) / "telegram_sessions.db"
        )
        self.db = db or TelegramDatabase()

        # Initialize session manager
        self.session_manager = SessionManager(db_path=self.session_db_path)

        self.logger = Logger(
            name="TelegramParser",
            level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
            log_file=os.getenv("LOG_FILE", "telegram.log"),
            log_dir=os.getenv("LOG_DIR", "logs"),
        )

    async def connect(self):
        """Connecting to Telegram API using stored session if available"""
        self.logger.info("Attempting to connect to Telegram API")

        # Try to get existing session
        session_string = self.session_manager.get_session(self.phone)

        if session_string:
            self.logger.info(f"Using existing session for phone {self.phone}")
            # Create client with existing session
            self.client = TelegramClient(StringSession(session_string), self.api_id, self.api_hash)
        else:
            self.logger.info(
                f"No existing session found for phone {self.phone}, creating new session"
            )
            # Create new client with StringSession
            self.client = TelegramClient(StringSession(), self.api_id, self.api_hash)

        try:
            # Connect to Telegram servers (doesn't log in yet)
            await self.client.connect()

            # Check if already authorized
            if not await self.client.is_user_authorized():
                self.logger.info("User not authorized. Starting authentication process...")

                # Request verification code
                await self.client.send_code_request(self.phone)
                self.logger.info(f"Verification code sent to {self.phone}")

                # Ask for the code
                verification_code = input(f"Enter the verification code sent to {self.phone}: ")

                # Sign in with the code
                await self.client.sign_in(self.phone, verification_code)
                self.logger.info("Successfully authenticated with Telegram API")
            else:
                self.logger.info("User already authorized")

            # Save the session string after successful connection
            if self.client.session:
                session_string = self.client.session.save()
                self.session_manager.save_session(self.phone, session_string)
                self.logger.info(f"Saved session for phone {self.phone}")

        except FloodWaitError as e:
            wait_time = e.seconds
            self.logger.warning(
                f"FloodWaitError: Need to wait {wait_time} seconds due to Telegram rate limiting"
            )
            await asyncio.sleep(wait_time)
            # Try to reconnect after waiting
            await self.connect()  # Recursive call to retry the connection

        except Exception as e:
            self.logger.error(f"Failed to connect to Telegram API: {str(e)}")
            raise
        finally:
            if self.db:
                self.db.init_user_tables(self.phone)

    async def get_dialogs(self, limit: int | None = None) -> list:
        """
        Getting list of dialogs.

        Args:
            limit (int, optional): Limit of dialogs

        Returns:
            List: List of dialogs
        """
        self.logger.info(f"Fetching dialogs (limit: {limit})")
        try:
            dialogs = await self.client.get_dialogs(limit=limit)
            self.logger.info(f"Successfully fetched {len(dialogs)} dialogs")
            return dialogs
        except Exception as e:
            self.logger.error(f"Error fetching dialogs: {str(e)}")
            raise

    async def get_chat_messages(
        self,
        chat_id: int,
        limit: int | None = None,
        min_id: int | None = None,
        dialog_name: str | None = None,
    ) -> pl.DataFrame | None:
        """
        Getting messages from specific chat.

        Args:
            chat_id (int): Chat ID
            limit (int, optional): Limit of messages
            min_id (int, optional): Get messages after specified ID
            dialog_name (str, optional): Name of the dialog

        Returns:
            Optional[pl.DataFrame]: DataFrame with messages or None if no messages found
        """
        self.logger.info(
            f"Fetching messages from chat {chat_id} (limit: {limit}, min_id: {min_id})"
        )
        messages = []
        if chat_id < 0:
            self.logger.info(f"Chat {chat_id} is a service chat, skipping")
            return None

        if min_id is None:
            min_id = -1
        try:
            if self.client is None:
                self.logger.error("Client is not initialized")
                return None

            async for message in self.client.iter_messages(chat_id, limit=limit, min_id=min_id):
                if isinstance(message, MessageService):
                    continue

                msg_dict = {
                    "message_id": message.id,
                    "date": message.date,
                    "from_id": message.from_id.user_id if message.from_id else None,
                    "text": message.text,
                    "chat_id": chat_id,
                    "reply_to_message_id": message.reply_to_msg_id,
                    "media_type": (message.media.__class__.__name__ if message.media else None),
                    "file_name": message.file.name if message.file else None,
                    "chat_name": dialog_name,
                    "forwarded_from": (message.fwd_from.from_name if message.fwd_from else None),
                }

                # Get information about sender
                if message.from_id and self.client is not None:
                    sender = await self.client.get_entity(message.from_id)
                    msg_dict["from_name"] = (
                        f"{sender.first_name} {sender.last_name if sender.last_name else ''}".strip()
                    )

                messages.append(msg_dict)

            if not messages:
                self.logger.info(f"No messages found in chat {chat_id}")
                return None

            # Get the centralized schema
            schema = get_polars_schema()

            # Create DataFrame with schema applied directly
            df = pl.DataFrame(
                messages,
                schema={
                    col_name: dtype for col_name, dtype in schema.items() if col_name in messages[0]
                },
                strict=False,
            )

            # Get unique sender IDs and fetch their information once
            unique_sender_ids = (
                df.filter(pl.col("from_id").is_not_null())["from_id"].unique().to_list()
            )
            sender_info = {}

            for sender_id in unique_sender_ids:
                try:
                    if self.client is not None:
                        sender = await self.client.get_entity(sender_id)
                        sender_info[sender_id] = (
                            f"{sender.first_name} {sender.last_name if sender.last_name else ''}".strip()
                        )
                except Exception as e:
                    self.logger.warning(f"Could not fetch info for sender {sender_id}: {str(e)}")
                    sender_info[sender_id] = ""

            # Apply the mapping to the DataFrame with explicit return type and skip_nulls=False
            df = df.with_columns(
                pl.when(pl.col("from_id").is_not_null())
                .then(
                    pl.col("from_id").map_elements(
                        lambda x: sender_info.get(x, ""), return_dtype=pl.Utf8, skip_nulls=False
                    )
                )
                .otherwise(pl.col("chat_name"))
                .alias("from_name")
            )

            self.logger.info(f"Successfully fetched {len(messages)} messages from chat {chat_id}")

            return df
        except Exception as e:
            self.logger.error(f"Error fetching messages from chat {chat_id}: {str(e)}")
            raise

    async def get_all_chats(
        self, limit_dialogs: int | None = None, limit_messages: int | None = None
    ) -> dict[int, pl.DataFrame]:
        """
        Get messages from all chats and optionally store them in database.

        Args:
            limit_dialogs (int, optional): Limit number of dialogs
            limit_messages (int, optional): Limit messages per dialog

        Returns:
            Dict[int, pl.DataFrame]: Dictionary mapping chat_id to messages DataFrame
        """
        self.logger.info(
            f"Fetching all chats (dialog limit: {limit_dialogs}, messages limit: {limit_messages})"
        )
        try:
            dialogs = await self.get_dialogs(limit=limit_dialogs)
            chats_dict = {}

            for dialog in dialogs:
                chat_id = dialog.id

                # Получаем максимальный message_id из базы данных для этого чата
                min_id = self.db.get_max_message_id(self.phone, chat_id) if self.db else -1

                # Передаем min_id в get_chat_messages
                df = await self.get_chat_messages(
                    chat_id, limit=limit_messages, min_id=min_id, dialog_name=dialog.name
                )

                # Получаем максимальный message_id из базы данных для этого чата
                min_id = self.db.get_max_message_id(self.phone, chat_id) if self.db else -1

                # Передаем min_id в get_chat_messages
                df = await self.get_chat_messages(
                    chat_id, limit=limit_messages, min_id=min_id, dialog_name=dialog.name
                )

                if df is not None and not df.is_empty():
                    self.logger.info(f"Added {len(df)} messages from chat {chat_id}")
                    chats_dict[chat_id] = df

                    # If database is provided, store messages
                    if self.db is not None:
                        self.db.add_messages(self.phone, df)
                elif min_id > -1 or min_id > -1:
                    self.logger.info(
                        f"No new messages found for chat {chat_id} since message_id {min_id}"
                    )

            self.logger.info(f"Successfully fetched messages from {len(chats_dict)} chats")
            return chats_dict

        except Exception as e:
            self.logger.error(f"Error fetching all chats: {str(e)}")
            raise

    async def close(self):
        """Close connections"""
        self.logger.info("Closing Telegram and database connections")
        try:
            if self.client:
                await self.client.disconnect()
                self.logger.info("Successfully closed Telegram connection")

            if self.db is not None:
                self.db.close()
                self.logger.info("Successfully closed database connection")

            # Close session manager
            self.session_manager.close()
            self.logger.info("Successfully closed session manager")

        except Exception as e:
            self.logger.error(f"Error closing connections: {str(e)}")
            raise


async def update_telegram_data(
    phone: str,
    db: TelegramDatabase,
    limit_dialogs: int | None = None,
    limit_messages: int | None = None,
) -> None:
    """
    Update function that can be called during service initialization to fetch
    and update all chats for a given phone number.

    Args:
        phone (str): Phone number to use for Telegram authentication
        limit_dialogs (int, optional): Limit number of dialogs to fetch
        limit_messages (int, optional): Limit number of messages per dialog

    Returns:
        None
    """
    # Setup path for session database
    session_db_path = str(Path(os.getenv("LOG_DIR", "data")) / "telegram_sessions.db")

    # Verify required parameters
    if not phone:
        raise ValueError("Phone number must be provided")

    parser = TelegramParser(phone=phone, db=db, session_db_path=session_db_path)

    try:
        # Connect to Telegram API
        await parser.connect()

        # Fetch all chats with optional limits
        await parser.get_all_chats(limit_dialogs=limit_dialogs, limit_messages=limit_messages)

        # Print summary for the user
        db.print_user_summary(phone)

    except Exception as e:
        logging.error(f"Error updating Telegram data: {str(e)}")
        raise
    finally:
        # Ensure all connections are properly closed
        await parser.close()


if __name__ == "__main__":
    phone_env = os.getenv("PHONE")
    if phone_env is None:
        raise ValueError("PHONE must be set in environment variables")
    # asyncio.run(main("+31627866359"))  # main function not defined
