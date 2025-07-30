"""Example script to create a database from Telegram account using client API."""

import asyncio

from terrorblade.data.database.telegram_database import TelegramDatabase
from terrorblade.data.loaders.telegram.parse_telegram_client import update_telegram_data  # type: ignore
from terrorblade.data.preprocessing.TelegramPreprocessor import TelegramPreprocessor


def run_processor(phone_number: str) -> None:
    """
    Update Telegram data and then process it with TelegramPreprocessor.

    Args:
        phone_number (str): Phone number to use for Telegram authentication
    """
    print("Welcome to the Telegram Database!")

    # Initialize database
    db = TelegramDatabase()
    db.print_user_summary(phone_number)

    # Update Telegram data using the async function
    print(f"Updating Telegram data for {phone_number}")
    # Run the async function in a synchronous context
    asyncio.run(update_telegram_data(phone=phone_number, db=db))

    # Now process the updated data
    print(f"Calculating clusters and embeddings for {phone_number}")
    preprocessor = TelegramPreprocessor(use_duckdb=True, phone=phone_number)
    preprocessor.process_messages(phone=phone_number)
    preprocessor.close()

    print(f"Thank you for using the Telegram Database for {phone_number}")


if __name__ == "__main__":
    PHONE = "+31627866359"
    # Direct call to the now-synchronous function
    run_processor(PHONE)
