from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from thoth.thoth.analyzer import ThothAnalyzer


class TestUserAnalysis:
    """Tests for ThothAnalyzer user analysis methods."""

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
            mock_model.encode.return_value = np.random.rand(384)  # Random embedding vector

            # Mock Qdrant client
            analyzer.qdrant = MagicMock()

            yield analyzer

    @pytest.fixture
    def mock_chat_data(self):
        """Create mock chat data with message points."""
        return pd.DataFrame(
            {
                "chat_id": [123, 123, 123, 123, 123],
                "message_id": [1, 2, 3, 4, 5],
                "from_id": [111, 222, 111, 333, 222],
                "date": pd.to_datetime(
                    [
                        "2023-01-01 10:00:00",
                        "2023-01-01 10:05:00",
                        "2023-01-01 10:10:00",
                        "2023-01-01 10:15:00",
                        "2023-01-01 10:20:00",
                    ]
                ),
                "text": [
                    "Hello everyone!",
                    "Hi there, how are you?",
                    "I'm doing well, thanks for asking",
                    "Let's discuss the project",
                    "Good idea, I have some suggestions",
                ],
                "reply_to_message_id": [None, 1, 2, None, 4],
            }
        )

    @pytest.fixture
    def mock_topics(self):
        """Create mock topic data."""
        return {
            "Topic 0": {
                "count": 2,
                "keywords": ["hello", "hi", "greetings"],
                "messages": [
                    {"message_id": 1, "from_id": 111, "text": "Hello everyone!"},
                    {"message_id": 2, "from_id": 222, "text": "Hi there, how are you?"},
                ],
            },
            "Topic 1": {
                "count": 2,
                "keywords": ["doing", "well", "thanks"],
                "messages": [
                    {"message_id": 3, "from_id": 111, "text": "I'm doing well, thanks for asking"},
                ],
            },
            "Topic 2": {
                "count": 2,
                "keywords": ["project", "discuss", "idea", "suggestions"],
                "messages": [
                    {"message_id": 4, "from_id": 333, "text": "Let's discuss the project"},
                    {"message_id": 5, "from_id": 222, "text": "Good idea, I have some suggestions"},
                ],
            },
        }

    def test_find_user_topics(self, mock_analyzer, mock_topics):
        """Test find_user_topics method."""
        # Mock find_common_topics method
        mock_analyzer.find_common_topics = MagicMock(return_value=mock_topics)

        # Run method
        result = mock_analyzer.find_user_topics()

        # Assertions
        assert isinstance(result, dict)
        assert 111 in result  # User 111 should be in results
        assert 222 in result  # User 222 should be in results
        assert 333 in result  # User 333 should be in results

        # Check user 111 has the right topics
        assert len(result[111]["topics"]) == 2
        assert "Topic 0" in result[111]["topics"]
        assert "Topic 1" in result[111]["topics"]

        # Check topic counts
        assert result[111]["topics"]["Topic 0"] == 1
        assert result[111]["topics"]["Topic 1"] == 1

        # Check user 222 has the right topics
        assert len(result[222]["topics"]) == 2
        assert "Topic 0" in result[222]["topics"]
        assert "Topic 2" in result[222]["topics"]

    def test_find_user_interactions(self, mock_analyzer, mock_chat_data):
        """Test find_user_interactions method."""
        # Mock Qdrant search results
        mock_analyzer.qdrant.scroll.return_value = (
            [
                MagicMock(payload={"message_id": m, "chat_id": c, "text": t, "from_id": f, "reply_to_message_id": r})
                for m, c, t, f, r in zip(
                    mock_chat_data["message_id"],
                    mock_chat_data["chat_id"],
                    mock_chat_data["text"],
                    mock_chat_data["from_id"],
                    mock_chat_data["reply_to_message_id"],
                )
            ],
            "scroll_id",
        )

        # Mock method for getting message by id
        original_message_method = mock_analyzer._get_message_by_id

        def mock_get_message(chat_id, message_id):
            """Mock method to get message by ID"""
            for i, (mid, cid, fid) in enumerate(
                zip(mock_chat_data["message_id"], mock_chat_data["chat_id"], mock_chat_data["from_id"])
            ):
                if mid == message_id and cid == chat_id:
                    return {"chat_id": cid, "message_id": mid, "from_id": fid}
            return None

        mock_analyzer._get_message_by_id = mock_get_message

        try:
            # Run method
            result = mock_analyzer.find_user_interactions()

            # Assertions
            assert isinstance(result, dict)
            assert "nodes" in result
            assert "links" in result

            # Verify nodes (users)
            assert len(result["nodes"]) == 3  # 3 users
            user_ids = [node["id"] for node in result["nodes"]]
            assert 111 in user_ids
            assert 222 in user_ids
            assert 333 in user_ids

            # Verify links (interactions)
            assert len(result["links"]) > 0
            for link in result["links"]:
                assert "source" in link
                assert "target" in link
                assert "value" in link
                assert link["source"] in user_ids
                assert link["target"] in user_ids

        finally:
            # Restore original method
            mock_analyzer._get_message_by_id = original_message_method

    def test_analyze_user_engagement(self, mock_analyzer, mock_chat_data):
        """Test analyze_user_engagement method."""
        # Mock Qdrant search results
        mock_analyzer.qdrant.scroll.return_value = (
            [
                MagicMock(
                    payload={
                        "message_id": m,
                        "chat_id": c,
                        "text": t,
                        "from_id": f,
                        "date": d.strftime("%Y-%m-%d %H:%M:%S"),
                        "reply_to_message_id": r,
                    }
                )
                for m, c, t, f, d, r in zip(
                    mock_chat_data["message_id"],
                    mock_chat_data["chat_id"],
                    mock_chat_data["text"],
                    mock_chat_data["from_id"],
                    mock_chat_data["date"],
                    mock_chat_data["reply_to_message_id"],
                )
            ],
            "scroll_id",
        )

        # Run method
        result = mock_analyzer.analyze_user_engagement()

        # Assertions
        assert isinstance(result, dict)
        assert 111 in result  # User 111 should be in results
        assert 222 in result  # User 222 should be in results
        assert 333 in result  # User 333 should be in results

        # Check engagement metrics for user 111
        assert "message_count" in result[111]
        assert "average_response_time" in result[111]
        assert "reply_rate" in result[111]
        assert "active_days" in result[111]

        # Verify message counts
        assert result[111]["message_count"] == 2  # User 111 has 2 messages
        assert result[222]["message_count"] == 2  # User 222 has 2 messages
        assert result[333]["message_count"] == 1  # User 333 has 1 message

    def test_analyze_user_influence(self, mock_analyzer, mock_chat_data):
        """Test analyze_user_influence method."""
        # Mock user interaction result
        mock_interactions = {
            "nodes": [
                {"id": 111, "name": "User 111"},
                {"id": 222, "name": "User 222"},
                {"id": 333, "name": "User 333"},
            ],
            "links": [
                {"source": 222, "target": 111, "value": 1},  # 222 replies to 111
                {"source": 111, "target": 222, "value": 1},  # 111 replies to 222
                {"source": 222, "target": 333, "value": 1},  # 222 replies to 333
            ],
        }

        # Mock find_user_interactions method
        mock_analyzer.find_user_interactions = MagicMock(return_value=mock_interactions)

        # Run method
        result = mock_analyzer.analyze_user_influence()

        # Assertions
        assert isinstance(result, dict)
        assert 111 in result  # User 111 should be in results
        assert 222 in result  # User 222 should be in results
        assert 333 in result  # User 333 should be in results

        # Check influence metrics
        assert "centrality" in result[111]
        assert "replies_received" in result[111]
        assert "replies_given" in result[111]
        assert "influence_score" in result[111]

        # Based on our mock data, user 222 should have highest influence (most active)
        assert result[222]["replies_given"] == 2  # User 222 replies to both 111 and 333
        assert result[111]["replies_given"] == 1  # User 111 replies to 222
        assert result[333]["replies_given"] == 0  # User 333 doesn't reply to anyone

    def test_analyze_user_behavior_patterns(self, mock_analyzer, mock_chat_data):
        """Test analyze_user_behavior_patterns method."""
        # Mock Qdrant search results
        mock_analyzer.qdrant.scroll.return_value = (
            [
                MagicMock(
                    payload={
                        "message_id": m,
                        "chat_id": c,
                        "text": t,
                        "from_id": f,
                        "date": d.strftime("%Y-%m-%d %H:%M:%S"),
                        "reply_to_message_id": r,
                    }
                )
                for m, c, t, f, d, r in zip(
                    mock_chat_data["message_id"],
                    mock_chat_data["chat_id"],
                    mock_chat_data["text"],
                    mock_chat_data["from_id"],
                    mock_chat_data["date"],
                    mock_chat_data["reply_to_message_id"],
                )
            ],
            "scroll_id",
        )

        # Run method
        result = mock_analyzer.analyze_user_behavior_patterns()

        # Assertions
        assert isinstance(result, dict)
        assert 111 in result  # User 111 should be in results
        assert 222 in result  # User 222 should be in results
        assert 333 in result  # User 333 should be in results

        # Check behavior metrics
        assert "message_frequency" in result[111]
        assert "reply_ratio" in result[111]
        assert "avg_message_length" in result[111]
        assert "active_hours" in result[111]
