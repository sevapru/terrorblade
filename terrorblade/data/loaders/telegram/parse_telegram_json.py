import json
import polars as pl
import pandas as pd
from src.data.dtypes import telegram_schema

def load_json(file_path: str) -> dict:
    with open(file_path) as file:
        data = json.load(file)
        chat_dict = {}
        for chat in data["chats"]["list"]:
            if len(chat["messages"]) >= 3:
                chat_df = pd.json_normalize(chat["messages"])
                chat_df["chat_name"] = chat.get("name", None)
                chat_df["chat_id"] = chat["id"]
                chat_df["chat_type"] = chat["type"]
                chat_dict[chat["id"]] = chat_df
            
        
        return chat_dict

def parse_links(chat_df):
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

def parse_members(chat_df):
    if 'members' in chat_df.columns:
        for idx, val in chat_df['members'].items():
            if isinstance(val, list):
                chat_df.at[idx, 'members'] = str(val)
    return chat_df

def parse_reactions(chat_df):
    for reaction in chat_df['reactions']:
        if isinstance(reaction, pl.Series):
            reaction = reaction[0]['emoji']
    return chat_df

def standartize_chat(chat: pd.DataFrame) -> pl.DataFrame:
    chat = chat.reindex(columns=telegram_schema.keys())
    return pl.DataFrame(chat).with_columns([
        pl.col(col).cast(dtype) for col, dtype in telegram_schema.items()
    ])


chats_dict = load_json('/home/seva/data/messages_json/result.json')
for key, chat_df in chats_dict.items():
    chat_df = parse_links(chat_df)
    chat_df = parse_members(chat_df)
    chat_df = standartize_chat(chat_df)
    chats_dict[key] = chat_df


all_chats = pl.concat(list(chats_dict.values()))
all_chats.write_parquet('/home/seva/data/all_chats.parquet')

