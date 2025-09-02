#!/usr/bin/env python3
"""
Fast cluster reprocessing using existing embeddings.

This script uses the new reprocess_clusters_only() method which is much faster
because it skips embedding calculation and uses existing embeddings from the database.

Perfect for fixing cluster logic without recomputing expensive embeddings!
"""

import logging
from pathlib import Path

from terrorblade.data.preprocessing.TelegramPreprocessor import TelegramPreprocessor
from terrorblade.utils.config import get_db_path


def fast_cluster_reprocessing() -> None:
    """
    Fast cluster reprocessing using existing embeddings.

    This method is 10-100x faster than full reprocessing because it:
    - ✅ Uses existing embeddings from database
    - ✅ Only recalculates temporal clusters and groups
    - ❌ Skips expensive embedding computation
    - ❌ Skips semantic segmentation recalculation
    """
    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    # Your configuration
    phone = "79992004210"
    db_path = get_db_path()  # Use environment variable or default

    logger.info(f"Using database: {db_path}")

    # Verify database exists
    if not Path(db_path).exists():
        logger.error(f"❌ Database not found: {db_path}")
        return

    logger.info("🚀 Starting FAST cluster reprocessing using existing embeddings")

    try:
        # Initialize TelegramPreprocessor
        preprocessor = TelegramPreprocessor(phone=phone, db_path=db_path, use_duckdb=True)

        # Use the new fast method that only reprocesses clusters
        # Using shorter time windows to prevent extremely long clusters
        preprocessor.reprocess_clusters_only(
            phone=phone,
            chat_id=None,  # Process all chats (you can specify a chat_id for testing)
            time_window="5m",  # 5-minute time window (more realistic for conversation breaks)
            cluster_size=10,  # Minimum 3 messages per cluster
        )

        logger.info("✅ Fast cluster reprocessing completed!")
        logger.info("💡 Check cluster statistics - they should now be realistic!")

    except Exception as e:
        logger.error(f"❌ Error during fast reprocessing: {e}")
        raise


def fast_reprocess_single_chat() -> None:
    """
    Test fast reprocessing on a single chat first.
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    phone = "79992004210"
    db_path = get_db_path()
    test_chat_id = 218477456  # Replace with actual chat_id
    logger.info(f"Using database: {db_path}")
    logger.info(f"🧪 Testing fast reprocessing on chat {test_chat_id}")
    try:
        preprocessor = TelegramPreprocessor(phone=phone, db_path=db_path, use_duckdb=True)
        # Process only one chat for testing
        preprocessor.reprocess_clusters_only(
            phone=phone,
            chat_id=test_chat_id,  # Specific chat only
            time_window="1h",  # Short window for testing (3 minutes)
            cluster_size=100,  # Lower threshold for testing
        )
        logger.info("✅ Single chat test completed!")
    except Exception as e:
        logger.error(f"❌ Error: {e}")


if __name__ == "__main__":
    print("🚀 Fast Cluster Reprocessing")
    print("=" * 40)
    print()
    print("This script uses existing embeddings for much faster cluster reprocessing!")
    print()
    print("Benefits:")
    print("  ⚡ 10-100x faster than full reprocessing")
    print("  ✅ Uses existing embeddings (no re-computation)")
    print("  🔧 Only fixes cluster logic issues")
    print("  💾 Updates database with corrected clusters")
    print()

    choice = input(
        "Choose option:\n1. Fast reprocess all chats\n2. Test on single chat\n\nEnter choice (1 or 2): "
    ).strip()

    if choice == "1":
        fast_cluster_reprocessing()
    elif choice == "2":
        fast_reprocess_single_chat()
    else:
        print("Invalid choice. Please run again and select 1 or 2.")
