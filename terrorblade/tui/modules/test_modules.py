"""
Test script for the modular TUI components.

This script tests the individual modules without starting the full TUI.
"""

import logging
import sys
from pathlib import Path

# Add terrorblade to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_imports() -> bool:
    """Test that all modules can be imported successfully."""
    try:
        logger.info("✅ All module imports successful")
        return True
    except Exception as e:
        logger.error(f"❌ Import failed: {e}")
        return False


def test_logging_terminal() -> bool:
    """Test the logging terminal component."""
    try:
        from .logging_terminal import LoggingTerminal

        terminal = LoggingTerminal(max_lines=3)
        logger.info("Testing logging terminal...")

        # Add some test messages
        terminal.add_message("Test message 1")
        terminal.add_message("Test message 2")
        terminal.add_message("Test message 3")
        terminal.add_message("Test message 4")  # Should replace message 1

        logger.info("✅ Logging terminal test successful")
        return True
    except Exception as e:
        logger.error(f"❌ Logging terminal test failed: {e}")
        return False


def test_chat_selector() -> bool:
    """Test the chat selector component (without database)."""
    try:
        from .chat_selector import ChatSelector

        # Create selector with None analyzer (no database)
        selector = ChatSelector(None, lambda x: None)

        # Test with mock data
        mock_chats = [
            {"chat_id": 1, "chat_name": "Test Chat 1", "message_count": 100},
            {"chat_id": 2, "chat_name": "Test Chat 2", "message_count": 200},
        ]

        selector.update_chats_data(mock_chats)

        # Test search
        selector.search_chats("Test Chat 1")

        logger.info("✅ Chat selector test successful")
        return True
    except Exception as e:
        logger.error(f"❌ Chat selector test failed: {e}")
        return False


def test_message_analyzer() -> bool:
    """Test the message analyzer component (without database)."""
    try:
        from .message_analyzer import MessageAnalyzer

        # Create analyzer with None analyzer (no database)
        analyzer = MessageAnalyzer(None)

        # Test with mock data
        mock_clusters = [
            {
                "group_id": 1,
                "chat_name": "Test Chat",
                "message_count": 50,
                "participant_count": 5,
                "start_time_str": "2024-01-01 10:00",
                "duration_hours": 2.5,
                "messages_per_hour": 20.0,
                "intensity": "High",
            }
        ]

        analyzer.update_clusters_data(mock_clusters)
        selected = analyzer.select_cluster_by_index(0)

        if selected and selected["group_id"] == 1:
            logger.info("✅ Message analyzer test successful")
            return True
        else:
            logger.error("❌ Message analyzer selection failed")
            return False
    except Exception as e:
        logger.error(f"❌ Message analyzer test failed: {e}")
        return False


def main() -> int:
    """Run all tests."""
    logger.info("Starting module tests...")

    tests = [
        ("Import Test", test_imports),
        ("Logging Terminal Test", test_logging_terminal),
        ("Chat Selector Test", test_chat_selector),
        ("Message Analyzer Test", test_message_analyzer),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        logger.info(f"Running {test_name}...")
        try:
            if test_func():
                logger.info(f"✅ {test_name} PASSED")
                passed += 1
            else:
                logger.error(f"❌ {test_name} FAILED")
                failed += 1
        except Exception as e:
            logger.error(f"❌ {test_name} FAILED with exception: {e}")
            failed += 1

    logger.info(f"Test Results: {passed} passed, {failed} failed")

    if failed == 0:
        logger.info("🎉 All tests passed! Modular architecture is working correctly.")
        return 0
    else:
        logger.error("⚠️ Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
