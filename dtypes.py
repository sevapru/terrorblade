import polars as pl

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
        "actor_id": pl.Utf8, # user+ pl.Int64
        "actor": pl.Utf8,
        "members": pl.Utf8,
        "action": pl.Utf8, # Phone call status or system messages
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

telegram_process_schema = {
        "chat_name": pl.Utf8,
        "date": pl.Datetime,
        "from": pl.Utf8,
        "text": pl.Utf8,
        "reply_to_message_id": pl.Int64,
        "forwarded_from": pl.Utf8,  # user+ pl.Int64 #need to parse
        "id": pl.Int64, # Unique message id
        "from_id": pl.Utf8, # Unique user id of sender (same as from)
        "chat_id": pl.Int64,
    }

# Base categories for dialogues
dialogue_categories = {
    "personal": ["family", "relationships", "health", "hobbies", "education"],
    "professional": ["work", "business", "career", "finance", "projects"],
    "social": ["politics", "society", "culture", "events", "news"],
    "entertainment": ["movies", "games", "music", "sports", "travel"],
    "technical": ["technology", "science", "engineering", "software", "hardware"]
}

# Base emotions
base_emotions = {
    "fear": "fear",
    "anger": "anger",
    "joy": "joy",
    "disgust": "disgust",
    "surprise": "surprise",
    "contempt": "contempt",
    "sadness": "sadness"
}