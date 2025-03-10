import os
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from thoth.thoth.analyzer import ThothAnalyzer


class TestThothAnalyzerInit:
    """Tests for ThothAnalyzer initialization and configuration."""

    def test_init_with_defaults(self):
        """Test initialization with default values."""
        with patch("qdrant_client.QdrantClient"):
            analyzer = ThothAnalyzer(db_path="test.db", phone="+1234567890")

            assert analyzer.db_path == Path("test.db")
            assert analyzer.phone == "+1234567890"
            assert analyzer.embedding_model == "all-MiniLM-L6-v2"
            assert analyzer.vector_size == 384
            assert analyzer.qdrant_path is None  # Default is None, uses in-memory

    def test_init_with_custom_values(self):
        """Test initialization with custom values."""
        with patch("qdrant_client.QdrantClient"):
            analyzer = ThothAnalyzer(
                db_path="custom.db",
                phone="+9876543210",
                embedding_model="paraphrase-multilingual-mpnet-base-v2",
                vector_size=768,
                qdrant_path="./custom_qdrant",
            )

            assert analyzer.db_path == Path("custom.db")
            assert analyzer.phone == "+9876543210"
            assert analyzer.embedding_model == "paraphrase-multilingual-mpnet-base-v2"
            assert analyzer.vector_size == 768
            assert analyzer.qdrant_path == "./custom_qdrant"

    def test_init_handles_path_conversion(self):
        """Test that string paths are converted to Path objects."""
        with patch("qdrant_client.QdrantClient"):
            analyzer = ThothAnalyzer(db_path="path/to/db.db", phone="+1234567890")

            assert isinstance(analyzer.db_path, Path)
            assert str(analyzer.db_path) == "path/to/db.db"

    def test_init_validates_phone_format(self):
        """Test that phone number format is validated."""
        with patch("qdrant_client.QdrantClient"):
            # Should accept with + prefix
            analyzer = ThothAnalyzer(db_path="test.db", phone="+1234567890")
            assert analyzer.phone == "+1234567890"

            # Should accept without + prefix
            analyzer = ThothAnalyzer(db_path="test.db", phone="1234567890")
            assert analyzer.phone == "1234567890"

    def test_init_invalid_arguments(self):
        """Test that initialization fails with invalid arguments."""
        with patch("qdrant_client.QdrantClient"), pytest.raises(ValueError):
            # Missing required phone
            ThothAnalyzer(db_path="test.db", phone=None)

        with patch("qdrant_client.QdrantClient"), pytest.raises(ValueError):
            # Missing required db_path
            ThothAnalyzer(db_path=None, phone="+1234567890")

    @patch("qdrant_client.QdrantClient")
    def test_qdrant_client_initialization(self, mock_qdrant):
        """Test that Qdrant client is initialized correctly."""
        # Test in-memory initialization
        ThothAnalyzer(db_path="test.db", phone="+1234567890")
        mock_qdrant.assert_called_with(":memory:")

        mock_qdrant.reset_mock()

        # Test local path initialization
        ThothAnalyzer(db_path="test.db", phone="+1234567890", qdrant_path="./qdrant_path")
        mock_qdrant.assert_called_with(path="./qdrant_path")

    @patch("sentence_transformers.SentenceTransformer")
    def test_embedding_model_initialization(self, mock_transformer):
        """Test that embedding model is initialized correctly."""
        with patch("qdrant_client.QdrantClient"):
            analyzer = ThothAnalyzer(db_path="test.db", phone="+1234567890", embedding_model="custom-model")

            # Assert the model is initialized when get_embedding is called
            mock_transformer.assert_not_called()  # Not called during init

            # Force model initialization by calling get_embedding
            analyzer._embedding_model = mock_transformer.return_value  # Mock for testing
            analyzer.get_embedding("test text")

            mock_transformer.assert_called_once_with("custom-model")

    @patch("duckdb.connect")
    def test_duckdb_connection(self, mock_connect):
        """Test that DuckDB connection is established correctly."""
        with patch("qdrant_client.QdrantClient"):
            analyzer = ThothAnalyzer(db_path="test.db", phone="+1234567890")

            # Mock connection return value
            mock_connect.return_value = MagicMock()

            # Access the db property to trigger connection
            db = analyzer.db

            # Verify connection was made with correct parameters
            mock_connect.assert_called_once_with("test.db")
            assert db is not None
