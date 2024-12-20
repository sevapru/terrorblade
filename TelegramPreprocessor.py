import os
import json
import polars as pl
import pandas as pd
from dtypes import telegram_schema

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
            for idx, val in chat_df['members'].items():
                if isinstance(val, list):
                    chat_df.at[idx, 'members'] = str(val)
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
        chat = chat.reindex(columns=telegram_schema.keys())
        return pl.DataFrame(chat).with_columns([
            pl.col(col).cast(dtype) for col, dtype in telegram_schema.items()
        ])
    
    def parse_timestamp(self, df, date_col: str = 'date', date_unixtime_col: str = 'date_unixtime') -> pl.DataFrame:
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
        return df.with_columns([
            pl.col(date_col).str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S"),
            pl.col(date_unixtime_col).cast(pl.Int64).cast(pl.Datetime)
        ])

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
        for key, chat_df in chats.values():
            clusters = self.create_clusters(chat_df, time_window, cluster_size, big_cluster_size)
            chats[key] = clusters
            
