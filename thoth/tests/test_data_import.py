import pytest
from unittest.mock import patch, Mock, MagicMock
import numpy as np
import pandas as pd
from pathlib import Path

from thoth.thoth.analyzer import ThothAnalyzer


class TestDataImport:
    """Tests for ThothAnalyzer data import functionality."""
    
    @pytest.fixture
    def mock_analyzer(self):
        """Create a mocked analyzer instance."""
        with patch('qdrant_client.QdrantClient') as mock_qdrant, \
             patch('duckdb.connect') as mock_duckdb, \
             patch('sentence_transformers.SentenceTransformer') as mock_transformer:
            
            # Setup mocks
            mock_connection = MagicMock()
            mock_duckdb.return_value = mock_connection
            mock_cursor = MagicMock()
            mock_connection.cursor.return_value = mock_cursor
            
            # Create analyzer with mocks
            analyzer = ThothAnalyzer(
                db_path="test.db",
                phone="+1234567890"
            )
            
            # Mock embedding model
            mock_model = MagicMock()
            mock_transformer.return_value = mock_model
            analyzer._embedding_model = mock_model
            mock_model.encode.return_value = np.random.rand(384)  # Random embedding vector
            
            yield analyzer
    
    @pytest.fixture
    def mock_telegram_data(self):
        """Create mock telegram data for testing."""
        return [
            {"chat_id": 123, "message_id": 1, "from_id": 456, "date": "2023-01-01 10:00:00", 
             "text": "Hello world", "reply_to_message_id": None},
            {"chat_id": 123, "message_id": 2, "from_id": 789, "date": "2023-01-01 10:05:00", 
             "text": "Hi there", "reply_to_message_id": 1},
            {"chat_id": 456, "message_id": 1, "from_id": 456, "date": "2023-01-02 11:00:00", 
             "text": "Different chat", "reply_to_message_id": None},
        ]
    
    def test_import_data_basic(self, mock_analyzer, mock_telegram_data):
        """Test basic data import functionality."""
        # Mock the DuckDB query result
        mock_cursor = mock_analyzer.db.cursor.return_value
        mock_cursor.fetchall.return_value = mock_telegram_data
        mock_cursor.description = [
            ("chat_id",), ("message_id",), ("from_id",), ("date",), 
            ("text",), ("reply_to_message_id",)
        ]
        
        # Mock the Qdrant client
        mock_analyzer.qdrant = MagicMock()
        
        # Call import data method
        with patch('pandas.DataFrame', return_value=pd.DataFrame(mock_telegram_data)):
            result = mock_analyzer.import_data()
        
        # Verify that the appropriate query was executed
        mock_cursor.execute.assert_called()
        query_arg = mock_cursor.execute.call_args[0][0]
        assert "SELECT" in query_arg
        assert "FROM messages" in query_arg
        assert "WHERE from_phone" in query_arg
        
        # Verify that collection creation was attempted
        assert mock_analyzer.qdrant.create_collection.called
        
        # Verify that points were upserted for each unique chat
        assert mock_analyzer.qdrant.upsert.called
    
    def test_import_data_with_chat_filter(self, mock_analyzer, mock_telegram_data):
        """Test data import with chat filter."""
        # Mock the DuckDB query result
        mock_cursor = mock_analyzer.db.cursor.return_value
        mock_cursor.fetchall.return_value = [mock_telegram_data[0], mock_telegram_data[1]]  # Only chat 123
        mock_cursor.description = [
            ("chat_id",), ("message_id",), ("from_id",), ("date",), 
            ("text",), ("reply_to_message_id",)
        ]
        
        # Mock the Qdrant client
        mock_analyzer.qdrant = MagicMock()
        
        # Call import data method with chat filter
        with patch('pandas.DataFrame', return_value=pd.DataFrame([mock_telegram_data[0], mock_telegram_data[1]])):
            result = mock_analyzer.import_data(chat_ids=[123])
        
        # Verify that the appropriate query was executed with chat filter
        mock_cursor.execute.assert_called()
        query_arg = mock_cursor.execute.call_args[0][0]
        assert "SELECT" in query_arg
        assert "FROM messages" in query_arg
        assert "WHERE from_phone" in query_arg
        assert "AND chat_id IN" in query_arg
        
        # Verify that collection creation was attempted
        assert mock_analyzer.qdrant.create_collection.called
        
        # Verify that points were upserted for the filtered chat
        assert mock_analyzer.qdrant.upsert.called
    
    def test_import_data_empty_result(self, mock_analyzer):
        """Test data import with empty result set."""
        # Mock the DuckDB query result to return empty
        mock_cursor = mock_analyzer.db.cursor.return_value
        mock_cursor.fetchall.return_value = []
        mock_cursor.description = [
            ("chat_id",), ("message_id",), ("from_id",), ("date",), 
            ("text",), ("reply_to_message_id",)
        ]
        
        # Mock the Qdrant client
        mock_analyzer.qdrant = MagicMock()
        
        # Call import data method
        with patch('pandas.DataFrame', return_value=pd.DataFrame([])):
            result = mock_analyzer.import_data()
        
        # Verify that the query was executed
        mock_cursor.execute.assert_called()
        
        # Verify that no points were upserted due to empty data
        assert not mock_analyzer.qdrant.upsert.called
    
    def test_get_embedding(self, mock_analyzer):
        """Test get_embedding method."""
        # Mock the embedding model to return a fixed vector
        test_vector = np.array([0.1, 0.2, 0.3])
        mock_analyzer._embedding_model.encode.return_value = test_vector
        
        # Get embedding for test text
        result = mock_analyzer.get_embedding("test text")
        
        # Verify model was called with correct text
        mock_analyzer._embedding_model.encode.assert_called_with("test text")
        
        # Verify result is the expected vector
        np.testing.assert_array_equal(result, test_vector)
    
    def test_get_embedding_batch(self, mock_analyzer):
        """Test get_embedding with batch of texts."""
        # Mock the embedding model to return fixed vectors for a batch
        test_texts = ["test1", "test2", "test3"]
        test_vectors = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9]
        ])
        mock_analyzer._embedding_model.encode.return_value = test_vectors
        
        # Get embeddings for test texts
        results = mock_analyzer.get_embedding(test_texts)
        
        # Verify model was called with correct texts
        mock_analyzer._embedding_model.encode.assert_called_with(test_texts)
        
        # Verify results are the expected vectors
        np.testing.assert_array_equal(results, test_vectors)
    
    def test_clear_data(self, mock_analyzer):
        """Test clear_data method."""
        # Mock the Qdrant client
        mock_analyzer.qdrant = MagicMock()
        
        # Call clear data method
        mock_analyzer.clear_data()
        
        # Verify that collection deletion was attempted for each chat
        assert mock_analyzer.qdrant.delete_collection.called 