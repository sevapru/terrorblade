"""Example script to create a database from Telegram JSON export file."""

from terrorblade.data.database.telegram_database import TelegramDatabase
from terrorblade.data.preprocessing.TelegramPreprocessor import TelegramPreprocessor


def create_db_from_telegram_json(
    phone: str, json_file_path: str, db_path: str = "telegram_data.db"
) -> None:
    """
    Create a database from Telegram JSON export file.

    This function processes a Telegram JSON export file, extracts messages,
    and stores them in a DuckDB database for further analysis.

    Args:
        phone (str): Phone number (e.g., "91654321987")
        json_file_path (str): Path to the Telegram JSON export file
        db_path (str, optional): Path to the database file. Defaults to "telegram_data.db"

    Example:
        >>> from terrorblade.examples.create_db_from_tg_json import create_db_from_telegram_json
        >>> create_db_from_telegram_json("91654321987", "/path/to/result.json")
    """
    print(f"Creating database from Telegram JSON export for {phone}")

    db = TelegramDatabase(db_path=db_path)
    db.init_user_tables(phone)

    preprocessor = TelegramPreprocessor(use_duckdb=True, phone=phone, db_path=db_path)

    try:
        print(f"Processing JSON file: {json_file_path}")
        preprocessor.process_file(json_file_path)
        db.print_user_summary(phone)

    finally:
        preprocessor.close()
        db.close()


if __name__ == "__main__":
    create_db_from_telegram_json(
        phone="31627866359",
        json_file_path="/home/seva/data/messages_json/result.json",
        db_path="/home/seva/data/telegram_data_test.db",
    )
