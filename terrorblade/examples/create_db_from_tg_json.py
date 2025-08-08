"""Example script to create a database from Telegram JSON export file."""

import argparse

from terrorblade.data.database.telegram_database import TelegramDatabase
from terrorblade.data.preprocessing.TelegramPreprocessor import TelegramPreprocessor


def create_db_from_telegram_json(
    phone: str,
    json_file_path: str,
    db_path: str = "telegram_data.db",
    skip_embeddings: bool = False,
) -> None:
    """
    Create a database from Telegram JSON export file.

    This function processes a Telegram JSON export file, extracts messages,
    and stores them in a DuckDB database for further analysis.

    Args:
        phone (str): Phone number (e.g., "91654321987")
        json_file_path (str): Path to the Telegram JSON export file
        db_path (str, optional): Path to the database file. Defaults to "telegram_data.db"
        skip_embeddings (bool): If True, will not compute/store embeddings (faster)

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
        if skip_embeddings:
            # Only add messages to DB
            chats = preprocessor.prepare_data(json_file_path)
            for _, chat_df in chats.items():
                preprocessor._add_messages_to_db(
                    chat_df, phone=phone
                )  # noqa: SLF001 (intended internal use)
        else:
            preprocessor.process_file(json_file_path)
        db.print_user_summary(phone)

    finally:
        preprocessor.close()
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a database from Telegram JSON export file")
    parser.add_argument("phone", help="Phone number (e.g., '31627866359')")
    parser.add_argument("json_file_path", help="Path to the Telegram JSON export file")
    parser.add_argument(
        "-d",
        "--db-path",
        default="telegram_data.db",
        help="Path to the database file (default: telegram_data.db)",
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip embeddings creation for faster DB build",
    )

    args = parser.parse_args()

    create_db_from_telegram_json(
        phone=args.phone,
        json_file_path=args.json_file_path,
        db_path=args.db_path,
        skip_embeddings=args.skip_embeddings,
    )
