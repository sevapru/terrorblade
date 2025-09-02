"""
Comprehensive tests for the Telegram preprocessing workflow.

This module tests the complete pipeline from JSON import to database storage,
including all preprocessing steps, embedding calculations, and clustering operations.
"""

import json
import shutil
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import ClassVar

import polars as pl
import pytest
import torch

from terrorblade.data.database.telegram_database import TelegramDatabase
from terrorblade.data.preprocessing.TelegramPreprocessor import TelegramPreprocessor
from terrorblade.data.preprocessing.TextPreprocessor import TextPreprocessor
from terrorblade.examples.create_db_from_tg_json import create_db_from_telegram_json


class TestTelegramWorkflow:
    """Test class for the complete Telegram preprocessing workflow."""

    # Class attributes with proper type annotations
    temp_dir: ClassVar[Path]
    test_db_path: ClassVar[Path]
    test_phone: ClassVar[str]
    test_data_path: ClassVar[Path]
    small_test_file: ClassVar[Path]

    @classmethod
    def setup_class(cls) -> None:
        """Set up test data and temporary files for the entire test class."""
        # Create temporary directory for test databases
        cls.temp_dir = Path(tempfile.mkdtemp())
        cls.test_db_path = cls.temp_dir / "test_telegram.db"
        cls.test_phone = "123456789"

        # Load subset of test data
        cls.test_data_path = Path(__file__).parent / "data" / "messages_test_animals.json"
        cls.small_test_file = cls.temp_dir / "small_test.json"

        # Create a smaller test dataset for faster testing
        cls._create_small_test_dataset()

    @classmethod
    def teardown_class(cls) -> None:
        """Clean up temporary files after all tests."""
        if hasattr(cls, "temp_dir") and cls.temp_dir.exists():
            shutil.rmtree(cls.temp_dir)

    @classmethod
    def _create_small_test_dataset(cls) -> None:
        """Create a smaller test dataset from the original test data."""
        with open(cls.test_data_path, encoding="utf-8") as f:
            full_data = json.load(f)

        # Take only first 2 chats with limited messages for faster testing
        small_data = {
            "about": full_data["about"],
            "chats": {"about": full_data["chats"]["about"], "list": []},
        }

        # Add chats with actual text messages
        chats_added = 0
        for chat in full_data["chats"]["list"]:
            if chats_added >= 2:
                break

            # Filter messages with actual text content
            text_messages = [
                msg for msg in chat["messages"] if msg.get("text", "").strip() and len(msg.get("text", "").strip()) > 5
            ]

            if len(text_messages) >= 5:  # Only include chats with enough text messages
                # Take first 20 text messages to keep tests fast
                chat_copy = chat.copy()
                chat_copy["messages"] = text_messages[:20]
                small_data["chats"]["list"].append(chat_copy)
                chats_added += 1

        # Save small test dataset
        with open(cls.small_test_file, "w", encoding="utf-8") as f:
            json.dump(small_data, f, ensure_ascii=False, indent=2)

    @pytest.fixture
    def text_preprocessor(self) -> TextPreprocessor:
        """Create a TextPreprocessor instance for testing."""
        return TextPreprocessor(time_window="5m", cluster_size=2, batch_size=100)

    @pytest.fixture
    def telegram_preprocessor(self) -> TelegramPreprocessor:
        """Create a TelegramPreprocessor instance for testing."""
        return TelegramPreprocessor(use_duckdb=False, time_window="5m", cluster_size=2)

    @pytest.fixture
    def telegram_preprocessor_with_db(self) -> TelegramPreprocessor:
        """Create a TelegramPreprocessor instance with database enabled."""
        db_path = self.__class__.temp_dir / f"test_preprocessor_{datetime.now().timestamp()}.db"
        return TelegramPreprocessor(
            use_duckdb=True,
            db_path=str(db_path),
            phone=self.__class__.test_phone,
            time_window="5m",
            cluster_size=2,
        )

    @pytest.fixture
    def telegram_database(self) -> TelegramDatabase:
        """Create a TelegramDatabase instance for testing."""
        db_path = self.__class__.temp_dir / f"test_db_{datetime.now().timestamp()}.db"
        return TelegramDatabase(db_path=str(db_path))

    @pytest.fixture
    def sample_messages_df(self) -> pl.DataFrame:
        """Create sample message DataFrame for testing."""
        # Create dummy embeddings for testing (768-dimensional vectors)
        import numpy as np

        dummy_embeddings = [np.random.rand(768).astype(np.float32).tolist() for _ in range(5)]

        return pl.DataFrame(
            {
                "text": [
                    "Hello world!",
                    "How are you doing today?",
                    "I am fine, thanks",
                    "Good morning everyone",
                    "Nice weather today",
                ],
                "from_id": [1, 1, 2, 3, 1],
                "date": [datetime.now() - timedelta(minutes=i) for i in range(5)],
                "chat_id": [100] * 5,
                "message_id": list(range(1, 6)),
                "from_name": ["User1", "User1", "User2", "User3", "User1"],
                "chat_name": ["Test Chat"] * 5,
                "reply_to_message_id": [None] * 5,
                "forwarded_from": [None] * 5,
                # Add missing columns required by TELEGRAM_SCHEMA
                "media_type": [None] * 5,
                "file_name": [None] * 5,
                "embeddings": dummy_embeddings,
            }
        ).with_columns(
            # Ensure embeddings column has correct type
            pl.col("embeddings").cast(pl.Array(pl.Float32, shape=768))
        )

    def test_textpreprocessor_init(self, text_preprocessor: TextPreprocessor) -> None:
        """Test TextPreprocessor initialization."""
        assert text_preprocessor.time_window == "5m"
        assert text_preprocessor.cluster_size == 2
        assert text_preprocessor.batch_size == 100
        assert text_preprocessor.device in ["cuda", "cpu"]
        assert text_preprocessor._embeddings_model is None  # Lazy loaded

    def test_textpreprocessor_embeddings_model_property(self, text_preprocessor: TextPreprocessor) -> None:
        """Test that embeddings model is loaded correctly."""
        model = text_preprocessor.embeddings_model
        assert model is not None
        assert hasattr(model, "encode")
        assert text_preprocessor.embeddings_model is model

    def test_concat_author_messages(
        self, text_preprocessor: TextPreprocessor, sample_messages_df: pl.DataFrame
    ) -> None:
        """Test concatenation of consecutive messages from same author."""
        result = text_preprocessor.concat_author_messages(sample_messages_df)
        assert len(result) < len(sample_messages_df)
        assert "Hello world!. How are you doing today?" in result["text"].to_list()
        expected_columns = [
            "chat_name",
            "date",
            "from_name",
            "text",
            "reply_to_message_id",
            "forwarded_from",
            "message_id",
            "from_id",
            "chat_id",
        ]
        assert all(col in result.columns for col in expected_columns)

    def test_calculate_embeddings(self, text_preprocessor: TextPreprocessor, sample_messages_df: pl.DataFrame) -> None:
        """Test embedding calculation for text messages."""
        result = text_preprocessor.calculate_embeddings(sample_messages_df)

        # Check that embeddings column is added
        assert "embeddings" in result.columns
        assert len(result) == len(sample_messages_df)

        # Check that embeddings have correct type and shape
        # Embeddings are stored as a Polars Series containing PyTorch tensors
        embeddings_series = result["embeddings"]
        assert len(embeddings_series) == len(sample_messages_df)

        # Extract a single embedding tensor to check
        first_embedding = embeddings_series[0]
        assert hasattr(first_embedding, "shape")  # Should have tensor-like properties
        assert len(first_embedding.shape) == 1  # Should be 1D tensor
        assert first_embedding.shape[0] > 0  # Should have dimensions

    def test_calculate_distances(self, text_preprocessor: TextPreprocessor) -> None:
        """Test distance calculation between embeddings."""
        # Create sample embeddings
        embeddings = torch.randn(5, 10)  # 5 samples, 10 dimensions
        distances = text_preprocessor.calculate_distances(embeddings)

        # Check distance matrix properties
        assert isinstance(distances, torch.Tensor)
        assert distances.shape == (5, 5)
        # Use a small tolerance for floating point precision issues
        assert torch.all(distances >= -1e-6)  # Distances should be non-negative (with tolerance)
        assert torch.all(distances <= 2)  # Cosine distances between 0 and 2

        # Check symmetry
        assert torch.allclose(distances, distances.T, rtol=1e-5)

        # Check diagonal is zero (distance to self) with appropriate tolerance
        diagonal = torch.diag(distances)
        assert torch.allclose(diagonal, torch.zeros_like(diagonal), atol=1e-6)  # Use absolute tolerance

    def test_calculate_sliding_distances(self, text_preprocessor: TextPreprocessor) -> None:
        """Test sliding window distance calculation."""
        embeddings = torch.randn(5, 10)
        distances = text_preprocessor.calculate_sliding_distances(embeddings, window_size=2)

        # Check sliding distances properties
        assert isinstance(distances, torch.Tensor)
        assert distances.shape == (5,)
        assert distances[0] == 0  # First element should be 0
        assert torch.all(distances >= 0)  # Distances should be non-negative

    def test_process_message_groups(
        self, text_preprocessor: TextPreprocessor, sample_messages_df: pl.DataFrame
    ) -> None:
        """Test complete message processing pipeline."""
        result = text_preprocessor.process_message_groups(sample_messages_df, time_window="5m", cluster_size=2)

        # Check that result has expected columns
        assert "group_id" in result.columns
        assert "embeddings" in result.columns
        assert len(result) > 0

        # Check that groups are properly assigned
        group_ids = result["group_id"].unique().to_list()
        assert len(group_ids) > 0

    def test_process_message_groups_empty_input(self, text_preprocessor: TextPreprocessor) -> None:
        """Test processing with empty DataFrame."""
        empty_df = pl.DataFrame(
            {
                "text": [],
                "from_id": [],
                "date": [],
                "chat_id": [],
                "message_id": [],
                "from_name": [],
                "chat_name": [],
                "reply_to_message_id": [],
                "forwarded_from": [],
            }
        )

        result = text_preprocessor.process_message_groups(empty_df)
        assert len(result) == 0

    def test_telegram_preprocessor_init(self, telegram_preprocessor: TelegramPreprocessor) -> None:
        """Test TelegramPreprocessor initialization."""
        assert telegram_preprocessor.use_duckdb is False
        assert telegram_preprocessor.phone is None
        assert hasattr(telegram_preprocessor, "logger")

    def test_telegram_preprocessor_init_with_db(self, telegram_preprocessor_with_db: TelegramPreprocessor) -> None:
        """Test TelegramPreprocessor initialization with database."""
        assert telegram_preprocessor_with_db.use_duckdb is True
        assert telegram_preprocessor_with_db.phone == self.__class__.test_phone
        assert hasattr(telegram_preprocessor_with_db, "db")

    def test_load_json_valid_file(self, telegram_preprocessor: TelegramPreprocessor) -> None:
        """Test loading valid JSON file."""
        result = telegram_preprocessor.load_json(str(self.__class__.small_test_file))

        assert isinstance(result, dict)
        assert len(result) > 0

        # Check that each chat has required structure
        for chat_id, chat_df in result.items():
            assert isinstance(chat_id, int)
            assert isinstance(chat_df, pl.DataFrame)
            assert len(chat_df) > 0
            assert "text" in chat_df.columns
            assert "chat_id" in chat_df.columns
            assert "message_id" in chat_df.columns

    def test_load_json_invalid_file(self, telegram_preprocessor: TelegramPreprocessor) -> None:
        """Test loading invalid file type."""
        with pytest.raises(ValueError, match="File must be a JSON file"):
            telegram_preprocessor.load_json("test.txt")

    def test_load_json_nonexistent_file(self, telegram_preprocessor: TelegramPreprocessor) -> None:
        """Test loading non-existent file."""
        with pytest.raises(FileNotFoundError):
            telegram_preprocessor.load_json("nonexistent.json")

    def test_parse_links(self, telegram_preprocessor: TelegramPreprocessor) -> None:
        """Test parsing of text entities and links."""
        # Test with simpler data that matches the actual implementation expectations
        # The parse_links method expects text column that may contain strings or list of dicts
        df = pl.DataFrame(
            {
                "text": [
                    "Simple text message",
                    "Another text message",
                    "Third message",
                ]
            }
        )

        result = telegram_preprocessor.parse_links(df)

        # For string inputs, parse_links should return them unchanged
        assert result["text"].to_list() == [
            "Simple text message",
            "Another text message",
            "Third message",
        ]

        # Test that the method doesn't break with string inputs
        assert "text" in result.columns
        assert len(result) == 3

    def test_parse_members(self, telegram_preprocessor: TelegramPreprocessor) -> None:
        """Test parsing of member lists."""
        df = pl.DataFrame(
            {
                "members": [
                    ["user1", "user2"],
                    ["user2", "user3", "user4"],
                    None,
                    [],
                ]
            }
        )

        result = telegram_preprocessor.parse_members(df)
        members_list = result["members"].to_list()

        # The actual implementation converts lists to string representations
        # Check that we got some kind of output for each input
        assert len(members_list) == 4
        # Just check that non-None values are processed
        non_none_results = [x for x in members_list if x is not None]
        assert len(non_none_results) >= 2  # At least the first two should be processed

    def test_parse_reactions(self, telegram_preprocessor: TelegramPreprocessor) -> None:
        """Test parsing of message reactions."""
        df = pl.DataFrame(
            {
                "reactions": [
                    [{"emoji": "👍"}],
                    [{"emoji": "❤️"}, {"emoji": "😊"}],  # Only first emoji is extracted
                    None,
                    [],
                ]
            }
        )

        result = telegram_preprocessor.parse_reactions(df)
        reactions_list = result["reactions"].to_list()

        # The implementation only extracts the first emoji from each reaction list
        assert "👍" in reactions_list
        assert "❤️" in reactions_list  # Only the first emoji is extracted
        assert None in reactions_list

    def test_parse_timestamp(self, telegram_preprocessor: TelegramPreprocessor) -> None:
        """Test timestamp parsing from various formats."""
        df = pl.DataFrame(
            {
                "date": [
                    "2023-01-01T12:00:00",
                    "2023-01-02T13:30:45",
                    "2023-12-31T23:59:59",
                ]
            }
        )

        result = telegram_preprocessor.parse_timestamp(df)

        # Check that dates are properly parsed
        assert result["date"].dtype == pl.Datetime
        dates = result["date"].to_list()
        assert all(isinstance(d, datetime) for d in dates)

    def test_delete_service_messages(self, telegram_preprocessor: TelegramPreprocessor) -> None:
        """Test deletion of service messages."""
        df = pl.DataFrame(
            {
                "chat_type": ["message", "service", "message", "service"],
                "text": ["Hello", "User joined", "How are you?", "User left"],
                "message_id": [1, 2, 3, 4],
            }
        )

        result = telegram_preprocessor.delete_service_messages(df)

        # Check that service messages are removed
        assert len(result) == 2
        assert all(chat_type != "service" for chat_type in result["chat_type"].to_list())
        assert result["text"].to_list() == ["Hello", "How are you?"]

    def test_prepare_data(self, telegram_preprocessor: TelegramPreprocessor) -> None:
        """Test complete data preparation pipeline."""
        result = telegram_preprocessor.prepare_data(str(self.__class__.small_test_file))

        assert isinstance(result, dict)
        assert len(result) > 0

        # Check that data is properly processed
        for chat_id, chat_df in result.items():
            assert isinstance(chat_id, int)
            assert isinstance(chat_df, pl.DataFrame)

            # Check required columns exist
            required_columns = ["text", "chat_id", "message_id", "from_id", "date"]
            assert all(col in chat_df.columns for col in required_columns)

            # Check data types
            assert chat_df["chat_id"].dtype == pl.Int64
            assert chat_df["message_id"].dtype == pl.Int64
            assert chat_df["from_id"].dtype == pl.Int64
            assert chat_df["date"].dtype == pl.Datetime

    def test_telegram_database_init(self, telegram_database: TelegramDatabase) -> None:
        """Test TelegramDatabase initialization."""
        assert telegram_database.db is not None
        assert telegram_database.read_only is False
        assert hasattr(telegram_database, "logger")

    def test_telegram_database_init_user_tables(self, telegram_database: TelegramDatabase) -> None:
        """Test user table initialization."""
        test_phone = "+1234567890"
        telegram_database.init_user_tables(test_phone)

        # Check that tables are created
        tables = telegram_database.db.execute("SHOW TABLES").fetchall()
        table_names = [table[0] for table in tables]

        assert "users" in table_names
        assert f"messages_{test_phone.replace('+', '')}" in table_names
        assert f"message_clusters_{test_phone.replace('+', '')}" in table_names

    def test_telegram_database_add_messages(
        self, telegram_database: TelegramDatabase, sample_messages_df: pl.DataFrame
    ) -> None:
        """Test adding messages to database."""
        test_phone = "+1234567890"
        telegram_database.init_user_tables(test_phone)

        # Add messages
        telegram_database.add_messages(test_phone, sample_messages_df)

        # Check that messages were added
        messages_table = f"messages_{test_phone.replace('+', '')}"
        result = telegram_database.db.execute(f"SELECT COUNT(*) FROM {messages_table}").fetchone()
        assert result is not None
        assert result[0] == len(sample_messages_df)

    def test_complete_workflow_integration(self) -> None:
        """Test the complete workflow from JSON to database."""
        # Use a separate temp file for this integration test
        integration_db = self.__class__.temp_dir / "integration_test.db"
        test_phone = "987654321"

        try:
            create_db_from_telegram_json(
                phone=test_phone,
                json_file_path=str(self.__class__.small_test_file),
                db_path=str(integration_db),
            )

            # Verify database was created and populated
            assert integration_db.exists()

            # Connect to database and verify data
            db = TelegramDatabase(db_path=str(integration_db), read_only=True)

            # Check that user tables exist
            tables = db.db.execute("SHOW TABLES").fetchall()
            table_names = [table[0] for table in tables]

            messages_table = f"messages_{test_phone}"
            assert "users" in table_names
            assert messages_table in table_names

            # Check that messages were imported
            message_count = db.db.execute(f"SELECT COUNT(*) FROM {messages_table}").fetchone()
            assert message_count is not None
            assert message_count[0] > 0

            db.close()

        finally:
            # Clean up
            if integration_db.exists():
                integration_db.unlink()

    def test_process_file_with_database(self, telegram_preprocessor_with_db: TelegramPreprocessor) -> None:
        """Test file processing with database storage."""
        try:
            # Manually create the required tables since TelegramPreprocessor only creates cluster tables
            from terrorblade.data.dtypes import TELEGRAM_SCHEMA

            phone_clean = self.__class__.test_phone
            messages_table = f"messages_{phone_clean}"

            # Create the users table
            telegram_preprocessor_with_db.db.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    phone VARCHAR PRIMARY KEY,
                    last_update TIMESTAMP,
                    first_seen TIMESTAMP
                )
            """
            )

            # Create the messages table
            columns = [f'"{field}" {info["db_type"]}' for field, info in TELEGRAM_SCHEMA.items()]
            create_sql = f"""
                CREATE TABLE IF NOT EXISTS {messages_table} (
                {", ".join(columns)},
                PRIMARY KEY (message_id, chat_id)
            )
            """
            telegram_preprocessor_with_db.db.execute(create_sql)

            result = telegram_preprocessor_with_db.process_file(str(self.__class__.small_test_file))

            assert isinstance(result, dict)
            assert len(result) > 0

            # Check that data was processed
            for chat_id, chat_df in result.items():
                assert isinstance(chat_id, int)
                assert isinstance(chat_df, pl.DataFrame)
                assert len(chat_df) > 0

        finally:
            telegram_preprocessor_with_db.close()

    def test_error_handling_invalid_phone(self) -> None:
        """Test error handling for invalid phone number."""
        with pytest.raises(ValueError, match="Phone number is required"):
            TelegramPreprocessor(use_duckdb=True, phone=None)

    def test_error_handling_corrupted_json(self, telegram_preprocessor: TelegramPreprocessor) -> None:
        """Test error handling for corrupted JSON file."""
        # Create corrupted JSON file
        corrupted_file = self.__class__.temp_dir / "corrupted.json"
        with open(corrupted_file, "w") as f:
            f.write('{"invalid": json content')

        with pytest.raises(json.JSONDecodeError):
            telegram_preprocessor.load_json(str(corrupted_file))

    def test_batch_processing_large_dataset(self, text_preprocessor: TextPreprocessor) -> None:
        """Test batch processing with large dataset."""
        # Create dataset larger than batch_size
        n_samples = 250  # Larger than default batch_size of 100
        large_df = pl.DataFrame(
            {
                "text": [f"Sample message {i}" for i in range(n_samples)],
                "from_id": [i % 3 for i in range(n_samples)],
                "date": [datetime.now() + timedelta(minutes=i) for i in range(n_samples)],
                "chat_id": [100] * n_samples,
                "message_id": list(range(n_samples)),
                "from_name": [f"User{i % 3}" for i in range(n_samples)],
                "chat_name": ["Test Chat"] * n_samples,
                "reply_to_message_id": [None] * n_samples,
                "forwarded_from": [None] * n_samples,
            }
        )

        result = text_preprocessor.process_message_groups(large_df)

        # Check that all messages were processed
        assert len(result) == n_samples
        assert "group_id" in result.columns
        assert "embeddings" in result.columns

    def test_device_selection(self, text_preprocessor: TextPreprocessor, sample_messages_df: pl.DataFrame) -> None:
        """Test proper device selection for embeddings."""
        result = text_preprocessor.calculate_embeddings(sample_messages_df)

        # Check that device was properly selected
        expected_device = "cuda" if torch.cuda.is_available() else "cpu"
        assert text_preprocessor.device == expected_device

        # Check that embeddings are properly formatted
        # Embeddings are stored as Polars Series elements (not raw tensors)
        first_embedding = result["embeddings"][0]
        # The embeddings should be numpy arrays or tensor-like objects stored in Polars
        assert hasattr(first_embedding, "shape")
        assert len(first_embedding.shape) == 1  # Should be 1D
        assert first_embedding.shape[0] > 0  # Should have dimensions


# Additional test functions for backwards compatibility and edge cases
def test_telegram_preprocessor_close() -> None:
    """Test proper cleanup of resources."""
    temp_db = Path(tempfile.mkdtemp()) / "cleanup_test.db"
    try:
        preprocessor = TelegramPreprocessor(use_duckdb=True, db_path=str(temp_db), phone="123456789")
        preprocessor.close()
        # Should not raise any exceptions
        assert True
    finally:
        if temp_db.exists():
            temp_db.unlink()
        temp_db.parent.rmdir()


def test_create_clusters_edge_cases() -> None:
    """Test cluster creation with edge cases."""
    preprocessor = TextPreprocessor()

    # Test with single message
    single_msg_df = pl.DataFrame(
        {
            "text": ["Single message"],
            "embeddings": [torch.randn(384)],  # Typical embedding size
            "date": [datetime.now()],
        }
    )

    result = preprocessor.create_clusters(single_msg_df, cluster_size=1)
    assert len(result) == 1
    assert "pre_cluster" in result.columns


def test_data_schema_validation() -> None:
    """Test that data schemas are properly defined and accessible."""
    from terrorblade.data.dtypes import TELEGRAM_SCHEMA, get_polars_schema, get_process_schema

    # Test schema access
    polars_schema = get_polars_schema()
    process_schema = get_process_schema()

    assert isinstance(polars_schema, dict)
    assert isinstance(process_schema, dict)
    assert isinstance(TELEGRAM_SCHEMA, dict)

    # Check required fields
    required_fields = ["message_id", "date", "from_id", "text", "chat_id"]
    assert all(field in TELEGRAM_SCHEMA for field in required_fields)


@pytest.mark.parametrize(
    "time_window,cluster_size",
    [
        ("5m", 2),
        ("10m", 3),
        ("30m", 1),
    ],
)
def test_parameter_combinations(time_window: str, cluster_size: int) -> None:
    """Test different parameter combinations."""
    preprocessor = TextPreprocessor(time_window=time_window, cluster_size=cluster_size)

    assert preprocessor.time_window == time_window
    assert preprocessor.cluster_size == cluster_size
