import os
import json
import polars as pl
import pandas as pd
from dtypes import telegram_import_schema, telegram_process_schema

from TextPreprocessor import TextPreprocessor

class TelegramPreprocessor(TextPreprocessor):
    """
    Preprocesses Telegram chat data.
    """
    def __init__(self, input_folder=None, *args, **kwargs):
        """
        Initializes the TelegramPreprocessor with the specified input folder.
        
        Args:
            input_folder (str): Path to the folder containing input files.
        """
        super().__init__(*args, **kwargs)
        self.input_folder = input_folder
        
    def load_json(self, file_path: str, min_messges=3) -> dict:
        """
        Loads chat data from a JSON file and filters chats with a minimum number of messages.
        
        Args:
            file_path (str): Path to the JSON file.
            min_messges (int): Minimum number of messages required to include a chat.
        
        Returns:
            dict: Dictionary of chat DataFrames.
        """
        with open(file_path) as file:
            data = json.load(file)
            chat_dict = {}
            for chat in data["chats"]["list"]:
                if len(chat["messages"]) >= min_messges:
                    chat_df = pd.json_normalize(chat["messages"])
                    chat_df["chat_name"] = chat.get("name", None)
                    chat_df["chat_id"] = chat["id"]
                    chat_df["chat_type"] = chat["type"]
                    chat_dict[chat["id"]] = chat_df
                
            return chat_dict

    def parse_links(self, chat_df):
        """
        Parses links in the chat DataFrame.
        
        Args:
            chat_df (pd.DataFrame): DataFrame containing chat messages.
        
        Returns:
            pd.DataFrame: DataFrame with parsed links.
        """
        for idx, val in chat_df['text'].items():
            if isinstance(val, list):
                if len(val) == 1 and isinstance(val[0], dict) and 'type' in val[0] and 'text' in val[0]:
                    chat_df.at[idx, 'type'] = val[0]['type']
                    chat_df.at[idx, 'text'] = val[0]['text']
                else:
                    extracted_texts = []
                    for item in val:
                        if isinstance(item, dict) and 'text' in item:
                            extracted_texts.append(item['text'])
                    chat_df.at[idx, 'text'] = ' '.join(extracted_texts)
        return chat_df

    def parse_members(self, chat_df):
        """
        Parses members in the chat DataFrame.
        
        Args:
            chat_df (pd.DataFrame): DataFrame containing chat messages.
        
        Returns:
            pd.DataFrame: DataFrame with parsed members.
        """
        if 'members' in chat_df.columns:
            all_members = set()
            for idx, val in chat_df['members'].items():
                if isinstance(val, list):
                    all_members.update(val)
            chat_df['members'] = str(list(all_members))
        return chat_df

    def parse_reactions(self, chat_df):
        """
        Parses reactions in the chat DataFrame.
        
        Args:
            chat_df (pd.DataFrame): DataFrame containing chat messages.
        
        Returns:
            pd.DataFrame: DataFrame with parsed reactions.
        """
        for reaction in chat_df['reactions']:
            if isinstance(reaction, pl.Series):
                reaction = reaction[0]['emoji']
        return chat_df

    def standartize_chat(self, chat: pd.DataFrame) -> pl.DataFrame:
        """
        Standardizes the chat DataFrame to match the telegram schema.
        
        Args:
            chat (pd.DataFrame): DataFrame containing chat messages.
        
        Returns:
            pl.DataFrame: Standardized DataFrame.
        """
        chat = chat.reindex(columns=telegram_import_schema.keys())
        return pl.DataFrame(chat).with_columns([
            pl.col(col).cast(dtype) for col, dtype in telegram_import_schema.items()
        ])
    
    def parse_timestamp(self, df, date_col: str = 'date') -> pl.DataFrame:
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
        return df.with_columns(
            pl.col(date_col).str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S")
        )

    def create_recipient_column(self, df, author_col):
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
            ', '.join([author for author in unique_authors if author != author_name]) 
            for author_name in df[author_col].to_list()
        ]
        return df.with_columns(
            pl.Series("recipients", recipients_str)
        )
    
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
                pl.col("location_information.longitude").is_not_null() &
                pl.col("location_information.latitude").is_not_null()
            )
            .then(pl.format("[location]({}, {})",
                            pl.col("location_information.longitude"),
                            pl.col("location_information.latitude")))
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
        return chat_pl.with_columns([
            pl.when(
                (pl.col("type") == "service") & (pl.col("discard_reason").is_not_null())
            )
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
            .alias("from_id")
        ])

    def handle_contacts(self, chat_pl: pl.DataFrame) -> pl.DataFrame:
        """
        If contact information is present, appends a contact note to 'text'.
        """
        return chat_pl.with_columns(
            pl.when(
                (pl.col("contact_information.first_name").is_not_null()) |
                (pl.col("contact_information.last_name").is_not_null())
            )
            .then(pl.format("[contact]({} {} : {})",
                            pl.col("contact_information.first_name").fill_null(""),
                            pl.col("contact_information.last_name").fill_null(""),
                            pl.col("contact_information.phone_number")))
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
            pl.when(
                (pl.col("media_type").is_null()) & (pl.col("file").is_not_null())
            )
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
            pl.when(
                (pl.col("photo").is_not_null())
            )
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
        return chat_pl.filter(pl.col("type") != "service")

    def prepare_data(self, file_path: str) -> pl.DataFrame:
        """
        Loads and prepares chat data from a JSON file.
        
        Args:
            file_path (str): Path to the JSON file
            
        Returns:
            pl.DataFrame: Processed and combined chat data
        """
        if not file_path.endswith('.json'):
            raise ValueError("File must be a JSON file")
            
        chats_dict = self.load_json(file_path)
        
        for key, chat_df in chats_dict.items():
            chat_df = self.parse_links(chat_df)
            chat_df = self.parse_members(chat_df)
            chat_pl = self.standartize_chat(chat_df)
            chat_pl = self.parse_timestamp(chat_pl)
            chat_pl = self.handle_additional_types(chat_pl)
            chat_pl = self.delete_service_messages(chat_pl)
            # Huge cut in columns
            chat_pl = chat_pl.select([pl.col(k).cast(v) for k, v in telegram_process_schema.items()])
            
            chats_dict[key] = chat_pl
            
        return chats_dict

    def process_chats(self, file_path: str, time_window: str, 
                      cluster_size: int = 3, big_cluster_size: int = 10) -> pl.DataFrame:
        """
        Processes a single chat file by creating clusters and calculating embeddings.
        
        Args:
            file_path (str): Path to the chat file
            time_window (str): Time window for clustering (e.g. "1h", "30m")
            cluster_size (int): Minimum cluster size
            big_cluster_size (int): Minimum size for big clusters
            
        Returns:
            pl.DataFrame: Processed chat data with clusters
        """
        chats = self.prepare_data(file_path)
        for key, chat_df in chats.items():
            messages = self.process_message_groups(chat_df, time_window, cluster_size, big_cluster_size)
            chats[key] = {
                'messages': messages,
                'embeddings': self.embeddings
            }
            
        return chats
