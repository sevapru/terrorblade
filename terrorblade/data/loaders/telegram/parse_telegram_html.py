import os

import polars as pl
from bs4 import BeautifulSoup

# Папка, где находятся HTML файлы
input_folder = "/home/seva/data/Maria"  # Замените на путь к вашей папке

# Списки для хранения извлеченных данных
data = []

# Проходим по всем HTML файлам в директории
for file_name in os.listdir(input_folder):
    if file_name.startswith("messages") and file_name.endswith(".html"):
        file_path = os.path.join(input_folder, file_name)

        with open(file_path, "r", encoding="utf-8") as file:
            soup = BeautifulSoup(file, "html.parser")  # Замените 'html.parser' на 'html5lib'

            # Извлечение сообщений из HTML
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

# Создание DataFrame с использованием polars
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
# Сохранение DataFrame в формате Parquet
df.write_parquet(f"parquet/{input_folder}.parquet")

print("Данные успешно сохранены в telegram_chats.parquet")
