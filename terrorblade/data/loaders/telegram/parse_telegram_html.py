# type: ignore
# ignored because this is not the main workflow and someone else can contribute here if you need

from pathlib import Path

import polars as pl
from bs4 import BeautifulSoup

input_folder = Path("/home/seva/data/Maria")
data = []

for file_path in input_folder.iterdir():
    if file_path.name.startswith("messages") and file_path.name.endswith(".html"):
        with open(file_path, encoding="utf-8") as file:
            soup = BeautifulSoup(file, "html.parser")
            messages = soup.find_all("div", class_="body")
            print(len(messages))
            for message in messages:
                date_time = (
                    message.find("div", class_="pull_right date details")["title"]
                    if message.find("div", class_="pull_right date details")
                    else "Unknown"
                )
                author = (
                    message.find("div", class_="from_name").text.strip()
                    if message.find("div", class_="from_name")
                    else "Unknown"
                )

                if message.find("div", class_="text"):
                    text = message.find("div", class_="text").text.strip()
                    message_type = "text"
                elif message.find("div", class_="media_wrap"):
                    media = message.find("div", class_="media_wrap")

                    if media.find("a", class_="video_file_wrap"):
                        text = media.find("a", class_="video_file_wrap")["href"]
                        message_type = "video"
                    elif media.find("a", class_="photo_wrap"):
                        text = media.find("a", class_="photo_wrap")["href"]
                        message_type = "photo"
                    elif media.find("a", class_="media_voice_message"):
                        text = media.find("a", class_="media_voice_message")["href"]
                        message_type = "voice_message"
                    elif media.find("a", class_="media_audio_file"):
                        text = media.find("a", class_="media_audio_file")["href"]
                        message_type = "audio"
                    elif media.find("a", class_="sticker_wrap"):
                        text = media.find("a", class_="sticker_wrap")["href"]
                        message_type = "sticker"
                    else:
                        text = "Unknown media"
                        message_type = "media"
                else:
                    text = "Unknown"
                    message_type = "unknown"

                data.append((date_time, author, message_type, text))
                len(data)

df = pl.DataFrame(
    data,
    schema={
        "timestamp": pl.Utf8,
        "author": pl.Utf8,
        "message_type": pl.Utf8,
        "text": pl.Utf8,
    },
)
print(df.shape)
df.write_parquet(f"parquet/{input_folder}.parquet")
