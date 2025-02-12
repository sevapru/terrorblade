from telethon import TelegramClient, sync
from telethon.tl.types import MessageService
import polars as pl
from datetime import datetime, timezone
import os
from typing import Dict, List, Optional
import asyncio
from dotenv import load_dotenv
from telethon.errors import FloodWaitError
import logging
import duckdb
from terrorblade import Logger

load_dotenv('.env')

api_id = os.getenv('API_ID')
api_hash = os.getenv('API_HASH')
phone = os.getenv('PHONE')

class TelegramParser:
    def __init__(self, api_id: str, api_hash: str, phone: str):
        """
        Initialise Telegram parser.
        
        Args:
            api_id (str): API ID от Telegram
            api_hash (str): API Hash от Telegram
            phone (str): Номер телефона для аутентификации
        """
        self.api_id = api_id
        self.api_hash = api_hash
        self.phone = phone
        self.client = None
        
        self.logger = Logger(
            name="TelegramParser",
            level=logging.INFO,
            log_file="telegram.log",
            log_dir="logs"
        )

        self.db = duckdb.connect('telegram_data.db')
        self._init_database()

    def _init_database(self):
        """Initialize database tables if they don't exist"""
        self.logger.info("Initializing database tables")
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                phone VARCHAR PRIMARY KEY,
                last_update TIMESTAMP
            )
        """)
        messages_table = f"messages_{self.phone.replace('+', '')}"
        self.db.execute(f"""
            CREATE TABLE IF NOT EXISTS {messages_table} (
                id BIGINT,
                chat_id BIGINT,
                date TIMESTAMP,
                text TEXT,
                from_id BIGINT,
                reply_to_msg_id BIGINT,
                media_type TEXT,
                file_name TEXT,
                "from" TEXT,
                PRIMARY KEY (id, chat_id)
            )
        """)

    async def connect(self):
        """Connecting to Telegram API"""
        self.logger.info("Attempting to connect to Telegram API")
        self.client = TelegramClient('session_name', self.api_id, self.api_hash)
        try:
            await self.client.start(phone=self.phone)
            self.logger.info("Successfully connected to Telegram API")
        except FloodWaitError as e:
            wait_time = e.seconds
            self.logger.warning(f"FloodWaitError: Need to wait {wait_time} seconds due to Telegram rate limiting")
            await asyncio.sleep(wait_time)
            await self.client.start(phone=self.phone)
        except Exception as e:
            self.logger.error(f"Failed to connect to Telegram API: {str(e)}")
            raise

    async def get_dialogs(self, limit: Optional[int] = None) -> List:
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

    async def get_chat_messages(self, chat_id: int, limit: Optional[int] = None, min_id: Optional[int] = None) -> pl.DataFrame:
        """
        Getting messages from specific chat.
        
        Args:
            chat_id (int): ID chat
            limit (int, optional): Limit of messages
            min_id (int, optional): Get messages after specified ID
            
        Returns:
            pl.DataFrame: DataFrame with messages
        """
        self.logger.info(f"Fetching messages from chat {chat_id} (limit: {limit}, min_id: {min_id})")
        messages = []
        
        if min_id is None:
            min_id = -1
        try:
            async for message in self.client.iter_messages(chat_id, limit=limit, min_id=min_id):
                if isinstance(message, MessageService):
                    continue
                
                msg_dict = {
                    'id': message.id,
                    'date': message.date,
                    'from_id': message.from_id.user_id if message.from_id else None,
                    'text': message.text,
                    'chat_id': chat_id,
                    'reply_to_msg_id': message.reply_to_msg_id,
                    'media_type': message.media.__class__.__name__ if message.media else None,
                    'file_name': message.file.name if message.file else None,
                    'from': message.from_id.user_id if message.from_id else None
                }
                
                # Get information about sender
                if message.from_id:
                    sender = await self.client.get_entity(message.from_id)
                    msg_dict['from'] = f"{sender.first_name} {sender.last_name if sender.last_name else ''}".strip()
                
                messages.append(msg_dict)

            if not messages:
                self.logger.nice(f"No messages found in chat {chat_id}")
                return None

            df = pl.DataFrame(messages)
            self.logger.info(f"Successfully fetched {len(messages)} messages from chat {chat_id}")
            
            # Cast types to needed format
            df = df.with_columns([
                pl.col('date').cast(pl.Datetime),
                pl.col('from_id').cast(pl.Int64),
                pl.col('chat_id').cast(pl.Int64),
                pl.col('reply_to_msg_id').cast(pl.Int64),
                pl.col('text').cast(pl.Utf8),
                pl.col('media_type').cast(pl.Utf8),
                pl.col('file_name').cast(pl.Utf8), 
                pl.col('id').alias('message_id').cast(pl.Int64)
            ])

            return df
        except Exception as e:
            self.logger.error(f"Error fetching messages from chat {chat_id}: {str(e)}")
            raise

    async def get_all_chats(self, limit_dialogs: Optional[int] = None, 
                           limit_messages: Optional[int] = None) -> Dict[int, pl.DataFrame]:
        """
        Modified to handle incremental updates and DuckDB storage
        """
        self.logger.info(f"Fetching all chats (dialog limit: {limit_dialogs}, messages limit: {limit_messages})")
        try:
            # Check if user exists and get last update time
            result = self.db.execute(
                "SELECT last_update FROM users WHERE phone = ?", 
                [self.phone]
            ).fetchone()
            
            last_update = result[0] if result else None
            dialogs = await self.get_dialogs(limit=limit_dialogs)
            chats_dict = {}
            
            for dialog in dialogs:
                chat_id = dialog.id
                messages_table = f"messages_{self.phone.replace('+', '')}"
                
                # Get the latest message ID and count for this chat from user-specific table
                chat_stats = self.db.execute(f"""
                    SELECT MAX(id), COUNT(*) FROM {messages_table}
                    WHERE chat_id = ?
                """, [chat_id]).fetchone()
                latest_msg_id, existing_messages = chat_stats[0], chat_stats[1]
                
                self.logger.info(f"Chat {chat_id} has {existing_messages} existing messages")
                
                # If we have messages, only fetch older ones using max_id
                if latest_msg_id:
                    df = await self.get_chat_messages(
                        chat_id, 
                        min_id=latest_msg_id
                    )
                else:
                    df = await self.get_chat_messages(
                        chat_id, 
                        limit=limit_messages
                    )
                
                if df is not None and not df.is_empty():
                    self.logger.info(f"Added {len(df)} new messages to chat {chat_id}")
                    # Convert DataFrame to match database schema types
                    df = df.with_columns([
                        pl.col('message_id').cast(pl.Int64),
                        pl.col('chat_id').cast(pl.Int64),
                        pl.col('date').cast(pl.Datetime),
                        pl.col('from_id').cast(pl.Int64),
                        pl.col('text').cast(pl.Utf8),
                        pl.col('reply_to_msg_id').cast(pl.Int64),
                        pl.col('media_type').cast(pl.Utf8),
                        pl.col('file_name').cast(pl.Utf8),
                        pl.col('from').cast(pl.Utf8),
                    ])
                    
                    # Convert to DuckDB-compatible format
                    df_dict = df.to_dicts()
                    
                    # Insert using parameterized query
                    for row in df_dict:
                        existing = self.db.execute(f"""
                            SELECT id FROM {messages_table}
                            WHERE id = ?
                        """, [row['message_id']]).fetchone()
                        
                        if not existing:
                            self.db.execute(f"""
                                INSERT INTO {messages_table}
                                (id, chat_id, date, from_id, text, reply_to_msg_id, 
                                 media_type, file_name, "from")
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, [
                                row['message_id'],
                                row['chat_id'], 
                                row['date'],
                                row['from_id'],
                                row['text'],
                                row['reply_to_msg_id'],
                                row['media_type'],
                                row['file_name'],
                                row['from']
                            ])
                    chats_dict[chat_id] = df
            
            # Update last update time for user
            current_time = datetime.now(timezone.utc)
            self.db.execute("""
                INSERT OR REPLACE INTO users (phone, last_update)
                VALUES (?, ?)
            """, [self.phone, current_time])
            
            self.logger.info(f"Successfully fetched and stored messages from {len(chats_dict)} chats")
            return chats_dict
            
        except Exception as e:
            self.logger.error(f"Error fetching all chats: {str(e)}")
            raise

    async def close(self):
        """Закрытие соединения"""
        self.logger.info("Closing Telegram and database connections")
        try:
            await self.client.disconnect()
            self.db.close()
            self.logger.info("Successfully closed all connections")
        except Exception as e:
            self.logger.error(f"Error closing connections: {str(e)}")
            raise

async def main(phone: str):
    api_id = os.getenv('API_ID')
    api_hash = os.getenv('API_HASH')

    parser = TelegramParser(api_id, api_hash, phone)
    
    try:
        await parser.connect()
        chats = await parser.get_all_chats(limit_dialogs=None, limit_messages=None)
        
        for chat_id, df in chats.items():
            print(f"Chat ID: {chat_id}")
            print(df.head())
            print("\n")
            
    finally:
        await parser.close()

if __name__ == "__main__":
    asyncio.run(main(os.getenv('PHONE')))
