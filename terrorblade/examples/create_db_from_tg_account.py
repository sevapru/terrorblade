import asyncio

from terrorblade.data.database.telegram_database import TelegramDatabase
from terrorblade.data.loaders.telegram.parse_telegram_client import update_telegram_data
from terrorblade.data.preprocessing.TelegramPreprocessor import TelegramPreprocessor


def run_processor(phone: str):
    """
    Update Telegram data and then process it with TelegramPreprocessor.

    Args:
        phone (str): Phone number to use for Telegram authentication
    """
    print("Welcome to the Telegram Database!")

    # Initialize database
    db = TelegramDatabase()
    db.print_user_summary(phone)

    # Update Telegram data using the async function
    print(f"Updating Telegram data for {phone}")
    # Run the async function in a synchronous context
    asyncio.run(update_telegram_data(phone=phone, db=db))

    # Now process the updated data
    print(f"Calculating clusters and embeddings for {phone}")
    preprocessor = TelegramPreprocessor(use_duckdb=True, phone=phone)
    preprocessor.process_messages(phone=phone)
    preprocessor.close()

    print(f"Thank you for using the Telegram Database for {phone}")


if __name__ == "__main__":
    phone = "+31627866359"
    # Direct call to the now-synchronous function
    run_processor(phone)
