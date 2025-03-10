from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from thoth.thoth.analyzer import ThothAnalyzer


class TestSearch:
    """Tests for ThothAnalyzer search methods."""

    @pytest.fixture
    def mock_analyzer(self):
        """Create a mocked analyzer instance with initialized dependencies."""
        with (
            patch("qdrant_client.QdrantClient") as mock_qdrant,
            patch("duckdb.connect") as mock_duckdb,
            patch("sentence_transformers.SentenceTransformer") as mock_transformer,
        ):

            # Setup mocks
            mock_connection = MagicMock()
            mock_duckdb.return_value = mock_connection
            mock_cursor = MagicMock()
            mock_connection.cursor.return_value = mock_cursor

            # Create analyzer with mocks
            analyzer = ThothAnalyzer(db_path="test.db", phone="+1234567890")

            # Mock embedding model
            mock_model = MagicMock()
            mock_transformer.return_value = mock_model
            analyzer._embedding_model = mock_model
            # Return a fixed embedding for testing
            mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])

            # Mock Qdrant client
            analyzer.qdrant = MagicMock()

            yield analyzer

    @pytest.fixture
    def mock_search_results(self):
        """Create mock search results."""
        return [
            MagicMock(
                payload={
                    "message_id": 1,
                    "chat_id": 123,
                    "from_id": 456,
                    "text": "This is a test message",
                    "date": "2023-01-01 10:00:00",
                },
                score=0.95,
            ),
            MagicMock(
                payload={
                    "message_id": 2,
                    "chat_id": 123,
                    "from_id": 789,
                    "text": "Another related message",
                    "date": "2023-01-01 10:05:00",
                },
                score=0.85,
            ),
            MagicMock(
                payload={
                    "message_id": 3,
                    "chat_id": 456,
                    "from_id": 456,
                    "text": "Less relevant message",
                    "date": "2023-01-01 10:10:00",
                },
                score=0.65,
            ),
        ]

    def test_semantic_search(self, mock_analyzer, mock_search_results):
        """Test semantic_search method."""
        # Mock Qdrant search
        mock_analyzer.qdrant.search.return_value = mock_search_results

        # Run the search
        results = mock_analyzer.semantic_search(query="test query", chat_id=123, limit=10)

        # Assertions
        assert len(results) == 3
        assert results[0]["message_id"] == 1
        assert results[0]["score"] == 0.95
        assert results[0]["text"] == "This is a test message"

        # Verify that the embedding model was called correctly
        mock_analyzer._embedding_model.encode.assert_called_once_with("test query")

        # Verify that the Qdrant search was called correctly
        mock_analyzer.qdrant.search.assert_called_once()
        call_args = mock_analyzer.qdrant.search.call_args
        assert call_args[1]["collection_name"] == "chat_123"
        assert call_args[1]["query_vector"] == [0.1, 0.2, 0.3]
        assert call_args[1]["limit"] == 10

    def test_semantic_search_all_chats(self, mock_analyzer, mock_search_results):
        """Test semantic_search across all chats."""
        # Mock list_collections to return multiple chat collections
        mock_analyzer.qdrant.list_collections.return_value = MagicMock(
            collections=[MagicMock(name="chat_123"), MagicMock(name="chat_456")]
        )

        # Mock Qdrant search to return different results for each collection
        def mock_search_fn(collection_name, **kwargs):
            if collection_name == "chat_123":
                return mock_search_results[:2]  # First 2 results for chat 123
            else:
                return [mock_search_results[2]]  # Last result for chat 456

        mock_analyzer.qdrant.search = mock_search_fn

        # Run the search across all chats
        results = mock_analyzer.semantic_search(query="test query", chat_id=None, limit=10)  # All chats

        # Assertions
        assert len(results) == 3

        # Results should be sorted by score
        assert results[0]["message_id"] == 1  # Highest score
        assert results[1]["message_id"] == 2
        assert results[2]["message_id"] == 3  # Lowest score

        # Verify that embedding model was called
        mock_analyzer._embedding_model.encode.assert_called_with("test query")

    def test_semantic_search_with_date_filter(self, mock_analyzer):
        """Test semantic_search with date filtering."""
        # Mock Qdrant search
        mock_analyzer.qdrant.search.return_value = []

        # Run the search with date filters
        mock_analyzer.semantic_search(query="test query", chat_id=123, start_date="2023-01-01", end_date="2023-01-31")

        # Verify Qdrant search was called with filter
        call_args = mock_analyzer.qdrant.search.call_args
        filter_args = call_args[1]["filter"]

        # Check filter includes date constraints
        assert "must" in filter_args
        date_filters = [f for f in filter_args["must"] if "range" in f]
        assert len(date_filters) > 0

    def test_semantic_search_with_user_filter(self, mock_analyzer):
        """Test semantic_search with user filtering."""
        # Mock Qdrant search
        mock_analyzer.qdrant.search.return_value = []

        # Run the search with user filter
        mock_analyzer.semantic_search(query="test query", chat_id=123, user_id=456)

        # Verify Qdrant search was called with filter
        call_args = mock_analyzer.qdrant.search.call_args
        filter_args = call_args[1]["filter"]

        # Check filter includes user constraint
        assert "must" in filter_args
        user_filters = [f for f in filter_args["must"] if "key" in f and f["key"] == "from_id"]
        assert len(user_filters) > 0
        assert user_filters[0]["match"]["value"] == 456

    def test_find_messages_by_keywords(self, mock_analyzer, mock_search_results):
        """Test find_messages_by_keywords method."""
        # Mock scroll response from Qdrant
        mock_analyzer.qdrant.scroll.return_value = (mock_search_results, None)

        # Run the method
        results = mock_analyzer.find_messages_by_keywords(keywords=["test", "message"], chat_id=123)

        # Assertions
        assert len(results) == 3
        assert results[0]["message_id"] == 1
        assert "test" in results[0]["text"].lower()
        assert "message" in results[0]["text"].lower()

        # Verify scroll was called correctly
        mock_analyzer.qdrant.scroll.assert_called_once()
        call_args = mock_analyzer.qdrant.scroll.call_args
        assert call_args[1]["collection_name"] == "chat_123"

    def test_find_messages_by_keywords_all_chats(self, mock_analyzer, mock_search_results):
        """Test find_messages_by_keywords across all chats."""
        # Mock list_collections to return multiple chat collections
        mock_analyzer.qdrant.list_collections.return_value = MagicMock(
            collections=[MagicMock(name="chat_123"), MagicMock(name="chat_456")]
        )

        # Mock scroll for different collections
        def mock_scroll_fn(collection_name, **kwargs):
            if collection_name == "chat_123":
                return (mock_search_results[:2], None)  # First 2 results for chat 123
            else:
                return ([mock_search_results[2]], None)  # Last result for chat 456

        mock_analyzer.qdrant.scroll = mock_scroll_fn

        # Run the method across all chats
        results = mock_analyzer.find_messages_by_keywords(keywords=["message"], chat_id=None)  # All chats

        # Assertions
        assert len(results) == 3
        # Results from both chats should be included
        chat_ids = [msg["chat_id"] for msg in results]
        assert 123 in chat_ids
        assert 456 in chat_ids

    def test_find_similar_messages(self, mock_analyzer, mock_search_results):
        """Test find_similar_messages method."""
        # Mock method to get message by ID
        mock_analyzer._get_message_by_id = MagicMock(
            return_value={
                "message_id": 100,
                "chat_id": 123,
                "text": "Reference message for similarity search",
                "from_id": 456,
                "date": "2023-01-01 09:00:00",
            }
        )

        # Mock Qdrant search
        mock_analyzer.qdrant.search.return_value = mock_search_results

        # Run the method
        results = mock_analyzer.find_similar_messages(chat_id=123, message_id=100, limit=5)

        # Assertions
        assert len(results) == 3
        assert results[0]["message_id"] == 1
        assert results[0]["score"] == 0.95

        # Verify that the embedding model was called with the reference message text
        mock_analyzer._embedding_model.encode.assert_called_with("Reference message for similarity search")

        # Verify that Qdrant search was called correctly
        mock_analyzer.qdrant.search.assert_called_once()

    def test_find_chat_context(self, mock_analyzer, mock_search_results):
        """Test find_chat_context method."""
        # Mock method to get message by ID
        mock_analyzer._get_message_by_id = MagicMock(
            return_value={
                "message_id": 2,
                "chat_id": 123,
                "text": "Another related message",
                "from_id": 789,
                "date": "2023-01-01 10:05:00",
            }
        )

        # Mock scroll for retrieving messages before/after
        def mock_scroll_fn(collection_name, **kwargs):
            filter_args = kwargs.get("filter", {})
            range_filters = filter_args.get("must", [])

            # For messages before the target
            if any(f.get("range", {}).get("gte") is None for f in range_filters if "range" in f):
                return ([mock_search_results[0]], None)
            # For messages after the target
            else:
                return ([mock_search_results[2]], None)

        mock_analyzer.qdrant.scroll = mock_scroll_fn

        # Run the method
        results = mock_analyzer.find_chat_context(chat_id=123, message_id=2, context_messages=1)

        # Assertions
        assert len(results) == 3  # Before + target + after
        assert results[0]["message_id"] == 1  # Message before
        assert results[1]["message_id"] == 2  # Target message
        assert results[2]["message_id"] == 3  # Message after

        # Verify that message was retrieved by ID
        mock_analyzer._get_message_by_id.assert_called_with(123, 2)
