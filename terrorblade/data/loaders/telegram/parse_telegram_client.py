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
from terrorblade import Logger
from terrorblade.data.database.telegram_database import TelegramDatabase

load_dotenv('.env')

api_id = os.getenv('API_ID')
api_hash = os.getenv('API_HASH')
phone = os.getenv('PHONE')

class TelegramParser:
    def __init__(self, api_id: str, api_hash: str, phone: str, db: Optional[TelegramDatabase] = None):
        """
        Initialize Telegram parser.
        
        Args:
            api_id (str): API ID from Telegram
            api_hash (str): API Hash from Telegram
            phone (str): Phone number for authentication
            db (Optional[TelegramDatabase]): Database instance for storing messages
        """
        self.api_id = api_id
        self.api_hash = api_hash
        self.phone = phone
        self.client = None
        self.db = db
        
        self.logger = Logger(
            name="TelegramParser",
            level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
            log_file=os.getenv('LOG_FILE', 'telegram.log'),
            log_dir=os.getenv('LOG_DIR', 'logs')
        )

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

    async def get_chat_messages(self, chat_id: int, limit: Optional[int] = None, min_id: Optional[int] = None, dialog_name: str = None) -> pl.DataFrame:
        """
        Getting messages from specific chat.
        
        Args:
            chat_id (int): Chat ID
            limit (int, optional): Limit of messages
            min_id (int, optional): Get messages after specified ID
            dialog_name (str, optional): Name of the dialog
            
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
                    'message_id': message.id,
                    'date': message.date,
                    'from_id': message.from_id.user_id if message.from_id else None,
                    'text': message.text,
                    'chat_id': chat_id,
                    'reply_to_message_id': message.reply_to_msg_id,
                    'media_type': message.media.__class__.__name__ if message.media else None,
                    'file_name': message.file.name if message.file else None,
                    'from': message.from_id.user_id if message.from_id else None,
                    'chat_name': dialog_name,
                    'forwarded_from': message.fwd_from.from_name if message.fwd_from else None
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
                pl.col('reply_to_message_id').cast(pl.Int64),
                pl.col('text').cast(pl.Utf8),
                pl.col('media_type').cast(pl.Utf8),
                pl.col('file_name').cast(pl.Utf8),
                pl.col('forwarded_from').cast(pl.Utf8),
                pl.col('message_id').alias('message_id').cast(pl.Int64)
            ])

            return df
        except Exception as e:
            self.logger.error(f"Error fetching messages from chat {chat_id}: {str(e)}")
            raise

    async def get_all_chats(self, limit_dialogs: Optional[int] = None, 
                           limit_messages: Optional[int] = None) -> Dict[int, pl.DataFrame]:
        """
        Get messages from all chats and optionally store them in database.
        
        Args:
            limit_dialogs (int, optional): Limit number of dialogs
            limit_messages (int, optional): Limit messages per dialog
            
        Returns:
            Dict[int, pl.DataFrame]: Dictionary mapping chat_id to messages DataFrame
        """
        self.logger.info(f"Fetching all chats (dialog limit: {limit_dialogs}, messages limit: {limit_messages})")
        try:
            dialogs = await self.get_dialogs(limit=limit_dialogs)
            chats_dict = {}
            
            for dialog in dialogs:
                chat_id = dialog.id
                df = await self.get_chat_messages(
                    chat_id, 
                    limit=limit_messages,
                    dialog_name=dialog.name
                )
                
                if df is not None and not df.is_empty():
                    self.logger.info(f"Added {len(df)} messages from chat {chat_id}")
                    chats_dict[chat_id] = df
                    
                    # If database is provided, store messages
                    if self.db is not None:
                        self.db.add_messages(self.phone, df)
            
            self.logger.info(f"Successfully fetched messages from {len(chats_dict)} chats")
            return chats_dict
            
        except Exception as e:
            self.logger.error(f"Error fetching all chats: {str(e)}")
            raise

    async def close(self):
        """Close connections"""
        self.logger.info("Closing Telegram and database connections")
        try:
            await self.client.disconnect()
            self.logger.info("Successfully closed Telegram connection")
            
            if self.db is not None:
                self.db.close()
                self.logger.info("Successfully closed database connection")
        except Exception as e:
            self.logger.error(f"Error closing connections: {str(e)}")
            raise

async def main(phone: str):
    api_id = os.getenv('API_ID')
    api_hash = os.getenv('API_HASH')
    
    # Initialize database
    db = TelegramDatabase()
    parser = TelegramParser(api_id, api_hash, phone, db)
    
    try:
        await parser.connect()
        chats = await parser.get_all_chats(limit_dialogs=None, limit_messages=None)
        
        # Print summary for the user
        db.print_user_summary(phone)
            
    finally:
        await parser.close()
        db.close()

if __name__ == "__main__":
    asyncio.run(main(os.getenv('PHONE')))
