import json
import logging
import os
from datetime import datetime
from typing import Any

import duckdb
import polars as pl
from dotenv import load_dotenv

from terrorblade import Logger
from terrorblade.data.dtypes import get_process_schema, telegram_import_schema_short
from terrorblade.data.preprocessing.TextPreprocessor import TextPreprocessor


class TelegramPreprocessor(TextPreprocessor):
    """
    Preprocesses Telegram chat data.
    """

    def __init__(
        self,
        use_duckdb: bool = False,
        db_path: str = "telegram_data.db",
        phone: str | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the TelegramPreprocessor with the specified input folder.

        Args:
            input_folder (str): Path to the folder containing input files.
            use_duckdb (bool): Whether to use DuckDB for data processing
            db_path (str): Path to DuckDB database file
            phone (str): Phone number for user-specific tables
        """
        super().__init__(*args, **kwargs)
        load_dotenv(".env")

        self.use_duckdb = use_duckdb
        self.phone = phone.replace("+", "") if phone else None

        self.logger = Logger(
            name="TelegramPreprocessor",
            level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
            log_file=os.getenv("LOG_FILE", "telegram_preprocessor.log"),
            log_dir=os.getenv("LOG_DIR", "logs"),
        )

        if use_duckdb:
            if not phone:
                raise ValueError("Phone number is required when use_duckdb is True")
            self.logger.info(f"Initializing DuckDB connection with database: {db_path}")
            self.db = duckdb.connect(db_path)
            self._init_cluster_tables()

    def _init_cluster_tables(self) -> None:
        """Initialize tables for storing cluster information"""
        self.logger.info("Initializing database tables")

        # Create user-specific tables for clusters and embeddings
        phone_clean = self.phone.replace("+", "") if self.phone else "default"
        clusters_table = f"message_clusters_{phone_clean}"
        embeddings_table = f"chat_embeddings_{phone_clean}"

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

        self.db.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {embeddings_table} (
                message_id BIGINT,
                chat_id BIGINT,
                embeddings DOUBLE[],
                PRIMARY KEY (message_id, chat_id)
            )
        """
        )

    def _get_messages_from_db(
        self, phone: str | None = None, chat_id: int | None = None
    ) -> pl.DataFrame:
        """
        Retrieve messages from DuckDB for a specific phone number and optionally chat_id

        Args:
            phone (str, optional): Phone number to retrieve messages for. Defaults to self.phone.
            chat_id (int, optional): Specific chat ID to filter messages

        Returns:
            pl.DataFrame: DataFrame containing messages
        """
        phone = phone or self.phone
        phone_clean = phone.replace("+", "") if phone else "default"
        messages_table = f"messages_{phone_clean}"
        query = f"SELECT * FROM {messages_table}"

        if chat_id is not None:
            query += f" WHERE chat_id = {chat_id}"

        query += " ORDER BY date"

        self.logger.info(
            f"Retrieving messages for phone {phone}" + (f" and chat {chat_id}" if chat_id else "")
        )
        try:
            result = self.db.execute(query).arrow()
            df = pl.from_arrow(result)
            # Ensure we return a DataFrame, not a Series
            if isinstance(df, pl.Series):
                df = df.to_frame()
            self.logger.info(f"Retrieved {len(df)} messages")
            return df
        except Exception as e:
            self.logger.error(f"Error retrieving messages from database: {str(e)}")
            raise

    def _update_clusters_in_db(self, clusters_df: pl.DataFrame) -> None:
        """
        Update cluster information in DuckDB using polars DataFrame

        Args:
            clusters_df (pl.DataFrame): DataFrame containing cluster information with columns: message_id, chat_id, group_id
        """
        clusters_table = f"message_clusters_{self.phone}"

        if len(clusters_df) == 0:
            return

        try:
            # The register + INSERT approach (0.0114s) is 65x faster than executemany(.rows()) (0.7483s)
            self.db.register("clusters_df", clusters_df)
            self.db.execute(
                f"""
                INSERT OR IGNORE INTO {clusters_table}
                (message_id, chat_id, group_id)
                SELECT message_id, chat_id, group_id
                FROM clusters_df
                """
            )
            self.logger.nice(f"Updated {len(clusters_df)} cluster records in table {clusters_table} ✓")  # type: ignore
        except Exception as e:
            self.logger.error(f"Error updating clusters in database: {str(e)}")
            raise

    def _update_embeddings_in_db(self, embeddings_df: pl.DataFrame) -> None:
        """
        Update embeddings information in DuckDB using a polars DataFrame

        Args:
            embeddings_df (pl.DataFrame): DataFrame with columns: message_id, chat_id, embeddings
        """
        embeddings_table = f"chat_embeddings_{self.phone}"

        if len(embeddings_df) == 0:
            return

        try:
            # The register + INSERT approach (0.0114s) is 65x faster than executemany(.rows()) (0.7483s)
            self.db.register("embeddings_df", embeddings_df)
            self.db.execute(
                f"""
                INSERT OR IGNORE INTO {embeddings_table}
                (message_id, chat_id, embeddings)
                SELECT message_id, chat_id, embeddings
                FROM embeddings_df
                """
            )
            self.logger.nice(f"Updated {len(embeddings_df)} embeddings ✓")  # type: ignore

        except Exception as e:
            self.logger.error(f"Error batch updating embeddings in database: {str(e)}")
            raise

    def _get_embeddings_from_db(self, chat_id: int) -> dict:
        """
        Retrieve embeddings for a specific chat from DuckDB

        Args:
            chat_id (int): ID of the chat

        Returns:
            dict: Dictionary mapping message_ids to their embeddings
        """
        phone_clean = self.phone.replace("+", "") if self.phone else "default"
        embeddings_table = f"chat_embeddings_{phone_clean}"
        self.logger.info(f"Retrieving embeddings for chat {chat_id}")

        try:
            query = f"""
                SELECT message_id, embeddings
                FROM {embeddings_table}
                WHERE chat_id = ?
                ORDER BY message_id
            """
            result = self.db.execute(query, [chat_id]).fetchall()
            embeddings_dict = {row[0]: row[1] for row in result}
            self.logger.info(f"Retrieved {len(embeddings_dict)} embeddings for chat {chat_id}")
            return embeddings_dict
        except Exception as e:
            self.logger.error(f"Error retrieving embeddings from database: {str(e)}")
            raise

    def _get_messages_with_embeddings(self, chat_id: int | None = None) -> set:
        """
        Retrieve message IDs that already have embeddings in the database.

        Args:
            chat_id (int, optional): Specific chat ID to filter. If None, gets all messages with embeddings.

        Returns:
            set: Set of message IDs that already have embeddings
        """
        phone_clean = self.phone.replace("+", "") if self.phone else "default"
        embeddings_table = f"chat_embeddings_{phone_clean}"
        self.logger.info(
            "Retrieving message IDs with embeddings" + (f" for chat {chat_id}" if chat_id else "")
        )

        try:
            query = f"SELECT message_id FROM {embeddings_table}"
            if chat_id is not None:
                query += f" WHERE chat_id = {chat_id}"

            result = self.db.execute(query).fetchall()
            message_ids = {row[0] for row in result}
            self.logger.info(f"Found {len(message_ids)} messages with existing embeddings")
            return message_ids
        except Exception as e:
            self.logger.error(f"Error retrieving message IDs with embeddings: {str(e)}")
            return set()

    def load_json(self, file_path: str, min_messages: int = 3) -> dict[int, pl.DataFrame]:
        """
        Loads chat data from a JSON file and filters chats with a minimum number of messages.

        Args:
            file_path (str): Path to the JSON file.
            min_messages (int): Minimum number of messages required to include a chat.

        Returns:
            Dict[int, pl.DataFrame]: Dictionary of chat DataFrames.
        """
        if not file_path.endswith(".json"):
            raise ValueError("File must be a JSON file")

        self.logger.info(f"Loading JSON data from {file_path}")
        try:
            with open(file_path) as file:
                data = json.load(file)
                chat_dict = {}

                for chat in data["chats"]["list"]:
                    if len(chat["messages"]) >= min_messages:
                        for message in chat["messages"]:
                            if "text_entities" in message:
                                texts = []
                                for entity in message["text_entities"]:
                                    if isinstance(entity, dict) and "text" in entity:
                                        texts.append(entity["text"])
                                if texts:
                                    message["text"] = " ".join(texts)

                            if "from" in message:
                                message["from_name"] = message["from"]

                        filtered_messages = [
                            {
                                field: (
                                    str(message.get(field, ""))
                                    if field == "text"
                                    else message.get(field)
                                )
                                for field in telegram_import_schema_short
                                if field in message or field == "from_name"
                            }
                            for message in chat["messages"]
                        ]

                        messages_df = pl.DataFrame(
                            filtered_messages, schema=telegram_import_schema_short
                        )

                        messages_df = messages_df.with_columns(
                            [
                                pl.lit(chat.get("name")).alias("chat_name"),
                                pl.lit(chat["id"]).alias("chat_id"),
                                pl.lit(chat["type"]).alias("chat_type"),
                                pl.col("id").alias("message_id"),
                            ]
                        ).drop("id")
                        chat_dict[chat["id"]] = messages_df

                self.logger.nice(f"Loaded {len(chat_dict)} chats with minimum {min_messages} messages ✓")  # type: ignore
                return chat_dict
        except Exception as e:
            self.logger.error(f"Error loading JSON data: {str(e)}")
            raise

    def parse_links(self, chat_df: pl.DataFrame) -> pl.DataFrame:
        """
        Parses links in the chat DataFrame.

        Args:
            chat_df (pl.DataFrame): DataFrame containing chat messages.

        Returns:
            pl.DataFrame: DataFrame with parsed links.
        """

        def extract_text(val: Any) -> Any:
            if isinstance(val, list):
                if (
                    len(val) == 1
                    and isinstance(val[0], dict)
                    and "type" in val[0]
                    and "text" in val[0]
                ):
                    return val[0]["text"]
                return " ".join(
                    item.get("text", "")
                    for item in val
                    if isinstance(item, dict) and "text" in item
                )
            return val

        return chat_df.with_columns(
            [pl.col("text").map_elements(extract_text, return_dtype=pl.Utf8).alias("text")]
        )

    def parse_members(self, chat_df: pl.DataFrame) -> pl.DataFrame:
        """
        Parses members in the chat DataFrame.

        Args:
            chat_df (pl.DataFrame): DataFrame containing chat messages.

        Returns:
            pl.DataFrame: DataFrame with parsed members.
        """
        if "members" in chat_df.columns:
            return chat_df.with_columns(
                [
                    pl.col("members")
                    .map_elements(lambda x: str(list(set(x))) if isinstance(x, list) else x)
                    .alias("members")
                ]
            )
        return chat_df

    def parse_reactions(self, chat_df: pl.DataFrame) -> pl.DataFrame:
        """
        Parses reactions in the chat DataFrame.

        Args:
            chat_df (pl.DataFrame): DataFrame containing chat messages.

        Returns:
            pl.DataFrame: DataFrame with parsed reactions.
        """
        if "reactions" in chat_df.columns:
            return chat_df.with_columns(
                [
                    pl.col("reactions")
                    .map_elements(
                        lambda x: (
                            x[0]["emoji"]
                            if isinstance(x, list | pl.Series) and len(x) > 0
                            else None
                        )
                    )
                    .alias("reactions")
                ]
            )
        return chat_df

    def standardize_chat(self, chat_df: pl.DataFrame) -> pl.DataFrame:
        """
        Standardize the chat DataFrame to match the expected schema.
        Casts columns to the expected types.

        Args:
            chat_df (pl.DataFrame): The chat DataFrame to standardize.

        Returns:
            pl.DataFrame: The standardized chat DataFrame.
        """
        for col in telegram_import_schema_short:
            if col not in chat_df.columns:
                chat_df = chat_df.with_columns(pl.lit(None).alias(col))

        # Cast columns to expected types
        return chat_df.select(
            [pl.col(col).cast(dtype) for col, dtype in telegram_import_schema_short.items()]
        )

    def parse_timestamp(self, df: pl.DataFrame, date_col: str = "date") -> pl.DataFrame:
        """
        Parses and formats the date and date_unixtime columns in the provided DataFrame.

        Process:
        - Parse the date column into a datetime object
        - Parse the date_unixtime column into a datetime object

        Args:
            df (pl.DataFrame): DataFrame containing the timestamps to parse.
            date_col (str): The name of the column containing date strings.
            date_unixtime_col (str): The name of the column containing Unix timestamps.

        Returns:
            pl.DataFrame: DataFrame with the date and date_unixtime columns parsed and formatted.
        """
        return df.with_columns(pl.col(date_col).str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S"))

    def create_recipient_column(self, df: pl.DataFrame, author_col: str) -> pl.DataFrame:
        """
        Creates a recipient string containing all other participants except the author.

        Process:
        - Extract unique authors and create a list of these authors
        - Create a string of recipients excluding the author for each row
        - Add the recipient column to the DataFrame

        Args:
            df (pl.DataFrame): DataFrame containing the author column to filter.
            author_col (str): The name of the column containing author names.

        Returns:
            pl.DataFrame: DataFrame with the recipient column added.
        """
        unique_authors = df.select(pl.col(author_col)).unique().to_series().to_list()
        recipients_str = [
            ", ".join([author for author in unique_authors if author != author_name])
            for author_name in df[author_col].to_list()
        ]
        return df.with_columns(pl.Series("recipients", recipients_str))

    def handle_media(self, chat_pl: pl.DataFrame) -> pl.DataFrame:
        """
        Modifies the 'text' column based on 'media_type' to unify message representation.

        1) Sticker filter: replaces message text with the 'sticker_emoji'.
        2) Video file filter: appends '[video](file_name)' to the text.
        3) Voice message filter: appends '[voice_message](file)' to the text.
        4) Audio file filter: replaces text with '[audio](performer-title)'.
        5) Animation filter: appends '[animation](file_name)' to the text.

        Args:
            chat_pl (pl.DataFrame): DataFrame containing chat messages.

        Returns:
            pl.DataFrame: DataFrame with modified 'text' column.
        """
        # Sticker filter
        chat_pl = chat_pl.with_columns(
            pl.when(pl.col("media_type") == "sticker")
            .then(pl.col("sticker_emoji"))
            .otherwise(pl.col("text"))
            .alias("text")
        )
        # Video file filter
        chat_pl = chat_pl.with_columns(
            pl.when(pl.col("media_type") == "video_file")
            .then(pl.format("{} [video]({})", pl.col("text"), pl.col("file_name")))
            .otherwise(pl.col("text"))
            .alias("text")
        )
        # Voice message filter
        chat_pl = chat_pl.with_columns(
            pl.when(pl.col("media_type") == "voice_message")
            .then(pl.format("[voice_message]({})", pl.col("file")))
            .otherwise(pl.col("text"))
            .alias("text")
        )
        # Audio file filter
        chat_pl = chat_pl.with_columns(
            pl.when(pl.col("media_type") == "audio_file")
            .then(pl.format("[audio]({}-{})", pl.col("title"), pl.col("performer").fill_null("")))
            .otherwise(pl.col("text"))
            .alias("text")
        )
        # Animation filter
        chat_pl = chat_pl.with_columns(
            pl.when(pl.col("media_type") == "animation")
            .then(pl.format("{} [animation]({})", pl.col("text"), pl.col("file_name")))
            .otherwise(pl.col("text"))
            .alias("text")
        )
        # Video message filter
        chat_pl = chat_pl.with_columns(
            pl.when(pl.col("media_type") == "video_message")
            .then(pl.format("[video_message]({})", pl.col("file_name")))
            .otherwise(pl.col("text"))
            .alias("text")
        )
        return chat_pl

    def handle_location(self, chat_pl: pl.DataFrame) -> pl.DataFrame:
        """
        Modifies the 'text' column based on the presence of location information.

        Appends '[location](longitude, latitude)' to the text if location information is present.

        Args:
            chat_pl (pl.DataFrame): DataFrame containing chat messages.

        Returns:
            pl.DataFrame: DataFrame with modified 'text' column.
        """
        return chat_pl.with_columns(
            pl.when(
                pl.col("location_information.longitude").is_not_null()
                & pl.col("location_information.latitude").is_not_null()
            )
            .then(
                pl.format(
                    "[location]({}, {})",
                    pl.col("location_information.longitude"),
                    pl.col("location_information.latitude"),
                )
            )
            .otherwise(pl.col("text"))
            .alias("text")
        )

    def handle_service_messages(self, chat_pl: pl.DataFrame) -> pl.DataFrame:
        """
        Modifies the 'text' and 'from' columns based on the message type.

        1) Replaces 'text' with 'discard_reason' for service messages with a discard reason.
        2) Replaces 'from' with 'actor' for service messages.

        Args:
            chat_pl (pl.DataFrame): DataFrame containing chat messages.

        Returns:
            pl.DataFrame: DataFrame with modified 'text' and 'from' columns.
        """
        return chat_pl.with_columns(
            [
                pl.when((pl.col("type") == "service") & (pl.col("discard_reason").is_not_null()))
                .then(pl.format("[phone_call]({})", pl.col("discard_reason")))
                .otherwise(pl.col("text"))
                .alias("text"),
                pl.when(pl.col("type") == "service")
                .then(pl.col("actor"))
                .otherwise(pl.col("from"))
                .alias("from"),
                pl.when(pl.col("type") == "service")
                .then(pl.col("actor_id"))
                .otherwise(pl.col("from_id"))
                .alias("from_id"),
            ]
        )

    def handle_contacts(self, chat_pl: pl.DataFrame) -> pl.DataFrame:
        """
        If contact information is present, appends a contact note to 'text'.
        """
        return chat_pl.with_columns(
            pl.when(
                (pl.col("contact_information.first_name").is_not_null())
                | (pl.col("contact_information.last_name").is_not_null())
            )
            .then(
                pl.format(
                    "[contact]({} {} : {})",
                    pl.col("contact_information.first_name").fill_null(""),
                    pl.col("contact_information.last_name").fill_null(""),
                    pl.col("contact_information.phone_number"),
                )
            )
            .otherwise(pl.col("text"))
            .alias("text")
        )

    def handle_files(self, chat_pl: pl.DataFrame) -> pl.DataFrame:
        """
        Modifies the 'text' column based on the presence of photo information.

        If media_type is missing and file is present, appends '[file](file)' to the text.
        If file starts with "(": appends '[file]' to the text.
        Args:
            chat_pl (pl.DataFrame): DataFrame containing chat messages.

        Returns:
            pl.DataFrame: DataFrame with modified 'text' column.
        """
        return chat_pl.with_columns(
            pl.when((pl.col("media_type").is_null()) & (pl.col("file").is_not_null()))
            .then(pl.format("[file]({})", pl.col("file_name")))
            .otherwise(pl.col("text"))
            .alias("text")
        )

    def handle_photos(self, chat_pl: pl.DataFrame) -> pl.DataFrame:
        """
        Modifies the 'text' column based on the presence of photo information.

        If media_type is missing and file is present, appends '[file](file)' to the text.
        If file starts with "(": appends '[file]' to the text.
        Args:
            chat_pl (pl.DataFrame): DataFrame containing chat messages.

        Returns:
            pl.DataFrame: DataFrame with modified 'text' column.
        """
        return chat_pl.with_columns(
            pl.when(pl.col("photo").is_not_null())
            .then(pl.format("{} [photo]({})", pl.col("text"), pl.col("file_name").fill_null("")))
            .otherwise(pl.col("text"))
            .alias("text")
        )

    def handle_additional_types(self, chat_pl: pl.DataFrame) -> pl.DataFrame:
        """
        Modifies the 'text' column based on additional message types.

        Args:
            chat_pl (pl.DataFrame): DataFrame containing chat messages.

        Returns:
            pl.DataFrame: DataFrame with modified 'text' column.
        """
        chat_pl = self.handle_media(chat_pl)
        chat_pl = self.handle_location(chat_pl)
        chat_pl = self.handle_files(chat_pl)
        chat_pl = self.handle_photos(chat_pl)
        chat_pl = self.handle_service_messages(chat_pl)
        chat_pl = self.handle_contacts(chat_pl)
        return chat_pl

    def delete_service_messages(self, chat_pl: pl.DataFrame) -> pl.DataFrame:
        """
        Deletes service messages from the chat DataFrame.

        Args:
            chat_pl (pl.DataFrame): DataFrame containing chat messages.

        Returns:
            pl.DataFrame: DataFrame with service messages removed.
        """

        return chat_pl.filter(pl.col("chat_type") != "service")

    def delete_empty_messages(self, chat_pl: pl.DataFrame) -> pl.DataFrame:
        """
        Delete messages with empty text field
        """
        return chat_pl.with_columns(
            pl.when(pl.col(pl.String).str.len_chars() == 0)
            .then(None)
            .otherwise(pl.col(pl.String))
            .name.keep()
        ).filter(pl.col("text").is_not_null())

    def prepare_data(self, file_path: str) -> dict[int, pl.DataFrame]:
        """
        Loads and prepares chat data from a JSON file.

        Args:
            file_path (str): Path to the JSON file

        Returns:
            pl.DataFrame: Processed and combined chat data
        """

        chats_dict = self.load_json(file_path)

        for key, chat_df in chats_dict.items():
            chat_df = self.parse_links(chat_df)
            chat_df = self.parse_members(chat_df)
            # chat_df = self.standardize_chat(chat_df) Well, it's not needed, because we have telegram_import_schema_short
            chat_df = self.parse_timestamp(chat_df)
            # chat_df = self.handle_additional_types(chat_df)
            chat_df = self.delete_service_messages(chat_df)
            # chat_df = self.delete_empty_messages(chat_df) well, it's not empty messages, it's messages with no text.

            chat_df = chat_df.with_columns(
                [
                    pl.col("from_id")
                    .str.replace("^user", "")
                    .str.replace("^channel", "")
                    .cast(pl.Int64)
                    .alias("from_id")
                ]
            )
            process_schema = get_process_schema()
            chat_df = chat_df.select(
                [pl.col(k).cast(v) for k, v in process_schema.items() if k in chat_df.columns]
            )

            chats_dict[key] = chat_df

        return chats_dict

    def _add_messages_to_db(self, messages_df: pl.DataFrame, phone: str | None = None) -> None:
        """
        Add messages to the database for a specific user.

        Args:
            phone (str): User's phone number
            messages_df (pl.DataFrame): DataFrame containing messages
        """
        try:
            phone = phone if phone is not None else self.phone
            messages_table = f"messages_{phone}"

            self.logger.info(f"Adding {messages_df.height} messages for user {phone}")

            if "from_name" not in messages_df.columns:
                messages_df = messages_df.with_columns(pl.lit(None).alias("from_name"))
                self.logger.info(f"Added missing 'from_name' column to DataFrame for user {phone}")

            self.db.register("messages_df", messages_df)
            column_names = [
                k
                for k in telegram_import_schema_short
                if k not in ("id", "media_type", "file_name")
            ]
            self.db.execute(
                f"""INSERT OR IGNORE INTO {messages_table} ({', '.join(column_names)}) SELECT {', '.join(column_names)} FROM messages_df"""
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

    def process_file(
        self,
        file_path: str,
        time_window: str = "5m",
        cluster_size: int = 3,
        big_cluster_size: int = 10,
    ) -> dict[int, pl.DataFrame]:
        """
        Process chats from a file.

        Args:
            file_path (str): Path to file to process.
            time_window (str, optional): Time window for clustering. Defaults to "5m".
            cluster_size (int, optional): Minimum cluster size. Defaults to 3.
            big_cluster_size (int, optional): Minimum big cluster size. Defaults to 10.

        Returns:
            Dict[int, pl.DataFrame]: Dictionary of chat dataframes.
        """
        self.logger.info(f"Processing file {file_path} with {time_window} time window")
        chats_dict = self.prepare_data(file_path)

        total_messages = sum(len(chat_df) for chat_df in chats_dict.values())

        for _, chat_df in chats_dict.items():
            self._add_messages_to_db(chat_df)

        existing_message_ids = self._get_messages_with_embeddings() if self.use_duckdb else set()

        table = f"""
            {'Message Processing Summary':^50}
            {'-'*50}
            {'Total messages':<30}: {total_messages} (in {len(chats_dict)} chats)
            {'Messages with embeddings':<30}: {len(existing_message_ids)}
            {'To process':<30}: {total_messages - len(existing_message_ids)}
            {'-'*50}
            """
        self.logger.info(table)

        for chat_id, chat_df in chats_dict.items():
            if existing_message_ids:
                original_count = len(chat_df)
                chat_df = chat_df.filter(
                    ~pl.col("message_id").is_in(existing_message_ids)
                    & pl.col("text").is_not_null()
                    & (pl.col("text").str.len_chars() > 0)
                )
                skipped_count = original_count - len(chat_df)
                if skipped_count > 0:
                    self.logger.info(
                        f"Skipping {skipped_count} messages with empty text or existing embeddings in chat {chat_id}"
                    )

            if len(chat_df) == 0:
                chats_dict[chat_id] = chat_df
                continue

            processed_df = self.process_message_groups(
                chat_df, time_window, cluster_size, big_cluster_size
            )
            embeddings_data = processed_df.select(["message_id", "chat_id", "embeddings"])

            self._update_embeddings_in_db(embeddings_data)

            if "group_id" in processed_df.columns:
                clusters_df = processed_df.select(["message_id", "chat_id", "group_id"])
                self._update_clusters_in_db(clusters_df)

            chats_dict[chat_id] = processed_df

        # Log final summary
        if self.use_duckdb:
            total_embeddings_updated = sum(len(df) for df in chats_dict.values() if len(df) > 0)
            processed_chats = sum(1 for df in chats_dict.values() if len(df) > 0)
            if total_embeddings_updated > 0:
                self.logger.nice(f"✅ File processing complete: updated {total_embeddings_updated} embeddings across {processed_chats} chats")  # type: ignore
            else:
                self.logger.nice("✅ File processing complete: all messages already processed, no new embeddings needed")  # type: ignore

        return chats_dict

    def process_messages(
        self,
        phone: str,
        chat_id: int | None = None,
        time_window: str = "5m",
        cluster_size: int = 3,
        big_cluster_size: int = 10,
    ) -> pl.DataFrame:
        """
        Process messages from the database, reusing existing embeddings and clusters when available.

        Args:
            phone (str): Phone number to retrieve messages for.
            chat_id (int, optional): Specific chat ID to filter when retrieving from DB. Defaults to None.
            time_window (str, optional): Time window for clustering. Defaults to "5m".
            cluster_size (int, optional): Minimum cluster size. Defaults to 3.
            big_cluster_size (int, optional): Minimum big cluster size. Defaults to 10.

        Returns:
            pl.DataFrame: Combined dataframe of processed messages.
        """
        self.logger.info(f"Processing messages with {time_window} time window")

        if not self.use_duckdb:
            self.logger.warning("DuckDB is not enabled - embeddings and clusters will not be saved")

        # Process from database - chat_id only affects this initial retrieval
        messages_df = self._get_messages_from_db(phone=phone, chat_id=chat_id)
        if messages_df.height == 0:
            self.logger.info("No messages found in the database")
            return pl.DataFrame()

        # Process all messages together instead of chat by chat
        if self.use_duckdb:
            # Get all unique chat IDs from retrieved messages
            chat_ids = messages_df["chat_id"].unique().to_list()
            self.logger.info(f"Processing messages from {len(chat_ids)} chats")

            # Get message IDs that already have embeddings
            existing_message_ids = self._get_messages_with_embeddings()

            # Identify which messages don't have embeddings
            all_message_ids = set(messages_df["message_id"].to_list())
            missing_message_ids = all_message_ids - existing_message_ids

            if missing_message_ids:
                self.logger.info(
                    f"Found {len(missing_message_ids)} messages without embeddings to process"
                )
                # Filter messages that need embeddings
                messages_to_process = messages_df.filter(
                    pl.col("message_id").is_in(list(missing_message_ids))
                )

                # Calculate embeddings for these messages
                messages_to_process = self.calculate_embeddings(messages_to_process)

                # Batch update all embeddings at once
                if len(messages_to_process) > 0:
                    embeddings_data = messages_to_process.select(
                        ["message_id", "chat_id", "embeddings"]
                    )

                    self._update_embeddings_in_db(embeddings_data)

                # Return all messages with processing completed
                return self.process_message_groups(
                    messages_df, time_window, cluster_size, big_cluster_size
                )
            else:
                self.logger.info("All messages already have embeddings")
                return messages_df
        else:
            # If not using database, just process everything at once
            return self.process_message_groups(
                messages_df, time_window, cluster_size, big_cluster_size
            )

    def close(self) -> None:
        """Close the DuckDB connection if it exists"""
        if self.use_duckdb:
            self.logger.info("Closing DuckDB connection")
            try:
                self.db.close()
                self.logger.nice("Closed DuckDB connection ✓")  # type: ignore
            except Exception as e:
                self.logger.error(f"Error closing DuckDB connection: {str(e)}")
                raise
