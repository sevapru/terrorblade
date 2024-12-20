import polars as pl

telegram_schema = {
        "id": pl.Int64,
        "type": pl.Utf8,
        "date": pl.Utf8,
        "date_unixtime": pl.Utf8,
        "from": pl.Utf8,
        "from_id": pl.Utf8,
        "file": pl.Utf8,
        "file_name": pl.Utf8,
        "thumbnail": pl.Utf8,
        "media_type": pl.Utf8,
        "sticker_emoji": pl.Utf8,
        "mime_type": pl.Utf8,
        "text": pl.Utf8,
        "reply_to_message_id": pl.Int64,
        "photo": pl.Utf8,
        "forwarded_from": pl.Utf8,  # user+ pl.Int64
        "chat_name": pl.Utf8,
        "chat_id": pl.Int64,
        "chat_type": pl.Utf8,
        "place_name": pl.Utf8,
        "location_information.latitude": pl.Float64,
        "contact_information.first_name": pl.Utf8,
        "location_information.longitude": pl.Float64,
        "performer": pl.Utf8,
        "action": pl.Utf8,
        "address": pl.Utf8,
        "contact_vcard": pl.Utf8,
        "contact_information.last_name": pl.Utf8,
        "reply_to_peer_id": pl.Utf8, # channel+ pl.Int64
        "via_bot": pl.Utf8,
        "discard_reason": pl.Utf8,
        "title": pl.Utf8,
        "message_id": pl.Int64,
        "actor_id": pl.Utf8, # user+ pl.Int64
        "actor": pl.Utf8,
        "contact_information.phone_number": pl.Utf8,
        "period": pl.Int64,
        "saved_from": pl.Utf8,
        "members": pl.Utf8,
        "media_spoiler": pl.Utf8,
        "inviter": pl.Utf8,
        "emoticon": pl.Utf8,
    }