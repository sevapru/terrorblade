from typing import Any

import polars as pl

# Centralized schema for Telegram messages
# This is the single source of truth for all Telegram data operations
TELEGRAM_SCHEMA = {
    "message_id": {
        "polars_type": pl.Int64,
        "db_type": "BIGINT",
        "description": "Unique identifier for the message",
    },
    "date": {
        "polars_type": pl.Datetime,
        "db_type": "TIMESTAMP",
        "description": "Timestamp when message was sent",
    },
    "from_id": {
        "polars_type": pl.Int64,
        "db_type": "BIGINT",
        "description": "Unique identifier of the sender",
    },
    "text": {
        "polars_type": pl.Utf8,
        "db_type": "TEXT",
        "description": "Text content of the message",
    },
    "chat_id": {
        "polars_type": pl.Int64,
        "db_type": "BIGINT",
        "description": "Unique identifier for the chat",
    },
    "reply_to_message_id": {
        "polars_type": pl.Int64,
        "db_type": "BIGINT",
        "description": "Message ID this message is replying to",
    },
    "media_type": {
        "polars_type": pl.Utf8,
        "db_type": "TEXT",
        "description": "Type of media attached to the message",
    },
    "file_name": {
        "polars_type": pl.Utf8,
        "db_type": "TEXT",
        "description": "Name of the attached file",
    },
    "from_name": {
        "polars_type": pl.Utf8,
        "db_type": "TEXT",
        "description": "Name or username of the sender",
    },
    "chat_name": {
        "polars_type": pl.Utf8,
        "db_type": "TEXT",
        "description": "Name of the chat or conversation",
    },
    "forwarded_from": {
        "polars_type": pl.Utf8,
        "db_type": "TEXT",
        "description": "Source of forwarded messages",
    },
}


# Helper functions to extract specific type definitions
def get_polars_schema() -> dict[str, Any]:
    """Return the Polars schema dictionary for Telegram messages."""
    return {field: info["polars_type"] for field, info in TELEGRAM_SCHEMA.items()}


def get_duckdb_schema() -> dict[str, str]:
    """Return the DuckDB schema dictionary for Telegram messages."""
    return {field: info["db_type"] for field, info in TELEGRAM_SCHEMA.items()}


def get_field_descriptions() -> dict[str, str]:
    """Return a dictionary of field descriptions."""
    return {field: info["description"] for field, info in TELEGRAM_SCHEMA.items()}


def get_column_names() -> list[str]:
    """Return a list of column names from the schema in the consistent order."""
    return list(TELEGRAM_SCHEMA.keys())


# For FastAPI and JSON serialization
def get_schema_for_api() -> dict[str, dict[str, Any]]:
    """
    Get a schema definition suitable for use in FastAPI or JSON serialization.
    Converts Polars types to string representations.
    """
    api_schema = {}
    for field, info in TELEGRAM_SCHEMA.items():
        api_schema[field] = {
            "type": str(info["polars_type"]),
            "db_type": info["db_type"],
            "description": info["description"],
        }
    return api_schema


def get_process_schema() -> dict[str, Any]:
    """Return a schema suitable for processing that maps to the central schema."""
    process_schema = {}
    # Add fields that are common between process schema and central schema
    for field in [
        "chat_name",
        "date",
        "from_name",
        "text",
        "reply_to_message_id",
        "forwarded_from",
        "chat_id",
        "message_id",
        "from_id",
    ]:
        if field in TELEGRAM_SCHEMA:
            process_schema[field] = TELEGRAM_SCHEMA[field]["polars_type"]

    return process_schema


def create_message_template() -> dict[str, None]:
    """
    Create an empty message template with all fields from the TELEGRAM_SCHEMA.

    Returns:
        Dict[str, None]: A dictionary with all schema fields initialized to None
    """
    return dict.fromkeys(TELEGRAM_SCHEMA.keys())


def map_telethon_message_to_schema(
    message, chat_id: int, dialog_name: str | None = None
) -> dict[str, Any]:
    """
    Maps a Telethon message object to our centralized schema format.

    Args:
        message: Telethon message object
        chat_id: The ID of the chat
        dialog_name: Optional name of the dialog

    Returns:
        Dict[str, Any]: Message data in our schema format
    """
    return {
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
        "from_name": None,  # This will be filled separately after getting entity info
    }


# Telegram import schema for archive
telegram_import_schema = {
    "id": pl.Int64,
    "chat_name": pl.Utf8,
    "date": pl.Utf8,
    "from": pl.Utf8,
    "text": pl.Utf8,
    "type": pl.Utf8,
    "from_id": pl.Utf8,
    "reply_to_message_id": pl.Int64,
    "forwarded_from": pl.Utf8,  # user+ pl.Int64
    "chat_id": pl.Int64,
    # MEDIA INFORMATION
    "media_type": pl.Utf8,
    "file": pl.Utf8,
    "file_name": pl.Utf8,
    # Photo information
    "photo": pl.Utf8,
    # Sticker information
    "sticker_emoji": pl.Utf8,
    "mime_type": pl.Utf8,
    # Music information
    "performer": pl.Utf8,
    "title": pl.Utf8,
    # Location information
    "location_information.latitude": pl.Float64,
    "location_information.longitude": pl.Float64,
    # Contact information
    "contact_information.first_name": pl.Utf8,
    "contact_information.last_name": pl.Utf8,
    "contact_information.phone_number": pl.Utf8,
    # Misc
    "chat_type": pl.Utf8,
    "discard_reason": pl.Utf8,
    "actor_id": pl.Utf8,  # user+ pl.Int64
    "actor": pl.Utf8,
    "members": pl.Utf8,
    "action": pl.Utf8,  # Phone call status or system messages
    # "inviter": pl.Utf8, # No need to parse, only GROUP there
    # "reply_to_peer_id": pl.Utf8, # channel+ pl.Int64 # Practically useless since thereis reply_to_message_id
    # "via_bot": pl.Utf8, # Mainly coming from bots, we can ignore it
    # "media_spoiler": pl.Utf8, # THEME MEDIA SPOILER - only a few disctinct values
    # "emoticon": pl.Utf8, # THEME EMOJI
    # "message_id": pl.Int64, # Used only for pin messages
    # "period": pl.Int64, # Message timeout?
    # "saved_from": pl.Utf8, Duplicating from forwarded_from for saved messages
    # "thumbnail": pl.Utf8, # Duplicate for file
    # "contact_vcard": pl.Utf8, # Useless for parsing
    # "place_name": pl.Utf8, # We already have a location
    # "address": pl.Utf8, # We already have a location
    # "date_unixtime": pl.Utf8, Absolutely the same as date, but in unixtime and without timezone. Could be faster to parse, but later then
}


# Telegram import schema for archive (shortened and aligned with TELEGRAM_SCHEMA)
telegram_import_schema_short = {
    "id": pl.Int64,
    "message_id": pl.Int64,  # Maps to message_id in TELEGRAM_SCHEMA
    "date": pl.Utf8,
    "from_id": pl.Utf8,  # Keep as Utf8 for JSON import, will be cast later
    "text": pl.Utf8,
    "chat_id": pl.Int64,
    "reply_to_message_id": pl.Int64,
    "media_type": pl.Utf8,
    "file_name": pl.Utf8,
    "from_name": pl.Utf8,  # Use from_name instead of "from" to align with TELEGRAM_SCHEMA
    "chat_name": pl.Utf8,
    "forwarded_from": pl.Utf8,
}
