import json

import polars as pl

from terrorblade.data.dtypes import get_polars_schema


def load_json(file_path: str) -> dict[int, pl.DataFrame]:
    with open(file_path) as file:
        data = json.load(file)
        chat_dict = {}
        for chat in data["chats"]["list"]:
            if len(chat["messages"]) >= 3:
                chat_df = pl.json_normalize(chat["messages"])
                chat_df = chat_df.with_columns(
                    [
                        pl.lit(chat.get("name", None)).alias("chat_name"),
                        pl.lit(chat["id"]).alias("chat_id"),
                        pl.lit(chat["type"]).alias("chat_type"),
                    ]
                )
                chat_dict[chat["id"]] = chat_df

        return chat_dict


def parse_links(chat_df: pl.DataFrame) -> pl.DataFrame:
    text_series = chat_df["text"]
    for idx in range(len(text_series)):
        val = text_series[idx]
        if isinstance(val, list):
            if len(val) == 1 and isinstance(val[0], dict) and "type" in val[0] and "text" in val[0]:
                chat_df = (
                    chat_df.with_row_count("idx")
                    .with_columns(
                        [
                            pl.when(pl.col("idx") == idx)
                            .then(pl.lit(val[0]["type"]))
                            .otherwise(pl.col("type"))
                            .alias("type"),
                            pl.when(pl.col("idx") == idx)
                            .then(pl.lit(val[0]["text"]))
                            .otherwise(pl.col("text"))
                            .alias("text"),
                        ]
                    )
                    .drop("idx")
                )
            else:
                extracted_texts = []
                for item in val:
                    if isinstance(item, dict) and "text" in item:
                        extracted_texts.append(item["text"])
                chat_df = (
                    chat_df.with_row_count("idx")
                    .with_columns(
                        [
                            pl.when(pl.col("idx") == idx)
                            .then(pl.lit(" ".join(extracted_texts)))
                            .otherwise(pl.col("text"))
                            .alias("text")
                        ]
                    )
                    .drop("idx")
                )
    return chat_df


def parse_members(chat_df: pl.DataFrame) -> pl.DataFrame:
    if "members" in chat_df.columns:
        members_series = chat_df["members"]
        for idx in range(len(members_series)):
            val = members_series[idx]
            if isinstance(val, list):
                chat_df = (
                    chat_df.with_row_count("idx")
                    .with_columns(
                        [
                            pl.when(pl.col("idx") == idx)
                            .then(pl.lit(str(val)))
                            .otherwise(pl.col("members"))
                            .alias("members")
                        ]
                    )
                    .drop("idx")
                )
    return chat_df


def parse_reactions(chat_df: pl.DataFrame) -> pl.DataFrame:
    if "reactions" in chat_df.columns:
        chat_df = chat_df.with_columns(
            [pl.col("reactions").map_elements(lambda x: x[0]["emoji"] if isinstance(x, list) else x).alias("reactions")]
        )
    return chat_df


def standartize_chat(chat: pl.DataFrame) -> pl.DataFrame:
    """
    Standardize the chat DataFrame to match the expected schema.

    Args:
        chat (pl.DataFrame): The chat DataFrame to standardize.

    Returns:
        pl.DataFrame: The standardized chat DataFrame.
    """
    # Get the polars schema from the centralized schema definition
    polars_schema = get_polars_schema()

    # Create a new DataFrame with only the columns from our schema
    chat = chat.select([col for col in polars_schema if col in chat.columns])

    # Cast columns to their respective types
    return chat.with_columns([pl.col(col).cast(dtype) for col, dtype in polars_schema.items() if col in chat.columns])
