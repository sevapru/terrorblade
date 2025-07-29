import asyncio
import random
import time
import unittest
from typing import Any
from unittest.mock import patch

from terrorblade.data.database.telegram_database import TelegramDatabase
from terrorblade.data.loaders.telegram.parse_telegram_client import update_telegram_data  # type: ignore


class TestTelegramFloodErrors(unittest.TestCase):
    """Test cases for Telegram flood errors and rate limiting."""

    def setUp(self) -> None:
        """Set up test environment before each test."""
        self.phone = "+******"
        self.db = TelegramDatabase()

    @patch("terrorblade.data.loaders.telegram.parse_telegram_client.update_telegram_data")
    def test_flood_error_multiple_requests(self, mock_update_telegram_data: Any) -> None:
        """Test that multiple rapid requests trigger flood control."""
        # Configure the mock to raise a flood error after a few calls
        mock_update_telegram_data.side_effect = [
            None,  # First few calls succeed
            None,
            None,
            Exception("FLOOD_WAIT_X: A wait of X seconds is required"),  # Then we get a flood error
        ]

        num_requests = 5
        delay = 0.1

        # Make multiple requests in quick succession
        for _i in range(num_requests):
            try:
                asyncio.run(
                    update_telegram_data(
                        phone=self.phone, db=self.db, limit_messages=random.randint(100, 500)
                    )
                )
                time.sleep(delay)
            except Exception as e:
                # We expect to get an exception
                self.assertIn("FLOOD_WAIT", str(e))
                return

        # If we didn't get an exception, the test failed
        self.fail("Expected flood error was not raised")

    def test_flood_error_real_requests(self) -> None:
        """Test with real requests to Telegram API to observe actual flood control behavior.

        Note: This test makes actual API calls and may affect your Telegram account's rate limits.
        """
        num_requests = 10
        delay = 0.05

        # Make multiple requests in quick succession to trigger flood control
        for i in range(num_requests):
            try:
                asyncio.run(
                    update_telegram_data(
                        phone=self.phone, db=self.db, limit_messages=random.randint(100, 500)
                    )
                )
                time.sleep(delay)
            except Exception as e:
                # If we get a flood error, the test passes
                if "FLOOD_WAIT" in str(e):
                    print(f"Flood error triggered after {i+1} requests: {e}")
                    return
                else:
                    print(f"Non-flood error occurred: {e}")

        print("Completed all requests without triggering flood control")

    def test_large_request(self) -> None:
        """Test that a very large request triggers flood control or other limits."""
        limit = 5000

        try:
            # Run the async function with a large limit
            asyncio.run(update_telegram_data(phone=self.phone, db=self.db, limit_messages=limit))
            print("Large request completed successfully without errors")
        except Exception as e:
            # If we get an error related to request size or rate limiting, the test passes
            print(f"Error occurred with large request (limit={limit}): {e}")
            if "FLOOD_WAIT" in str(e) or "TOO_MANY" in str(e):
                return
            else:
                self.fail(f"Unexpected error: {e}")


if __name__ == "__main__":
    unittest.main()
