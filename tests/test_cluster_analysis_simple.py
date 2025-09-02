#!/usr/bin/env python3
"""
Simplified test suite for Cluster Analysis CLI Tool

This module tests the core functionality with mocked database operations.
"""

import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import polars as pl
import pytest

# Add terrorblade to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from terrorblade.examples.cluster_analysis_cli import (
    display_chats_table,
    display_cluster_analysis,
    display_clusters_table,
    extract_story_from_cluster,
    summarize_cluster_with_openai,
)


class TestBasicFunctionality:
    """Test basic functionality without database dependencies."""

    def test_sample_data_creation(self) -> None:
        """Test that we can create sample data for testing."""
        sample_data = pl.DataFrame(
            {
                "chat_id": [-1001234567890, -1002345678901],
                "chat_name": ["Test Family Group", "Work Team"],
                "message_count": [2847, 1592],
                "participant_count": [8, 12],
            }
        )

        assert len(sample_data) == 2
        assert "chat_name" in sample_data.columns
        assert sample_data["message_count"].sum() == 4439

    def test_display_functions_with_empty_data(self, capsys: pytest.CaptureFixture) -> None:
        """Test display functions with empty data."""
        empty_df = pl.DataFrame()

        display_chats_table(empty_df)
        captured = capsys.readouterr()
        assert "No chats found." in captured.out

        display_clusters_table(empty_df)
        captured = capsys.readouterr()
        assert "No large clusters found." in captured.out

    def test_display_chats_table_with_data(self, capsys: pytest.CaptureFixture) -> None:
        """Test displaying chats table with sample data."""
        sample_data = pl.DataFrame(
            {
                "chat_id": [-1001234567890, -1002345678901],
                "chat_name": ["Test Family Group", "Work Team"],
                "message_count": [2847, 1592],
                "participant_count": [8, 12],
                "cluster_count": [45, 28],
                "max_cluster_size": [127, 89],
            }
        )

        display_chats_table(sample_data)
        captured = capsys.readouterr()

        assert "📋 Available Chats:" in captured.out
        assert "Test Family Group" in captured.out
        assert "Work Team" in captured.out

    def test_display_clusters_table_with_data(self, capsys: pytest.CaptureFixture) -> None:
        """Test displaying clusters table with sample data."""
        sample_data = pl.DataFrame(
            {
                "group_id": [15, 23],
                "chat_name": ["Test Family Group", "Work Team"],
                "message_count": [127, 89],
                "participant_count": [6, 8],
                "start_time_str": ["2024-01-15 09:30", "2024-01-14 14:15"],
                "duration_hours": [4.2, 6.8],
                "messages_per_hour": [30.2, 13.1],
                "intensity": ["🔥 Very High", "🔴 High"],
            }
        )

        display_clusters_table(sample_data)
        captured = capsys.readouterr()

        assert "🔍 Large Clusters Found:" in captured.out
        assert "🔥 Very High" in captured.out
        assert "🔴 High" in captured.out

    def test_display_cluster_analysis_with_data(self, capsys: pytest.CaptureFixture) -> None:
        """Test displaying cluster analysis with sample data."""
        stats = {
            "cluster_id": 15,
            "chat_name": "Test Family Group",
            "message_count": 127,
            "participant_count": 6,
            "duration_hours": 4.2,
            "messages_per_hour": 30.2,
            "start_time": datetime(2024, 1, 15, 9, 30),
            "end_time": datetime(2024, 1, 15, 13, 42),
            "peak_hour": {"date": datetime(2024, 1, 15).date(), "hour": 11, "messages": 45, "active_users": 4},
            "participants": pl.DataFrame(
                {
                    "from_id": [123, 456, 789],
                    "from_name": ["Alice", "Bob", "Charlie"],
                    "message_count": [45, 32, 28],
                    "avg_message_length": [87.3, 65.2, 92.1],
                }
            ),
            "time_analysis": pl.DataFrame(
                {"date_only": [datetime(2024, 1, 15).date()], "hour": [11], "messages": [45], "active_users": [4]}
            ),
        }

        display_cluster_analysis(stats)
        captured = capsys.readouterr()

        assert "📊 Cluster Analysis" in captured.out
        assert "Group 15" in captured.out
        assert "Test Family Group" in captured.out
        assert "📝 Messages: 127" in captured.out
        assert "👥 Participants: 6" in captured.out
        assert "Alice" in captured.out

    def test_display_cluster_analysis_with_error(self, capsys: pytest.CaptureFixture) -> None:
        """Test displaying cluster analysis with error."""
        stats = {"error": "Test error message"}
        display_cluster_analysis(stats)
        captured = capsys.readouterr()

        assert "❌ Test error message" in captured.out


class TestAIFunctions:
    """Test AI-related functions with mocked API calls."""

    @patch("terrorblade.examples.cluster_analysis_cli.OPENAI_AVAILABLE", True)
    @patch("terrorblade.examples.cluster_analysis_cli.openai")
    def test_summarize_cluster_with_openai_success(self, mock_openai: Mock) -> None:
        """Test successful cluster summarization."""
        # Mock the OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "This is a test summary of the cluster discussion."

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.OpenAI.return_value = mock_client

        cluster_text = "[2024-01-15 09:30:00] Alice: Hello everyone!\n[2024-01-15 09:35:00] Bob: How are you?"
        api_key = "test-api-key"

        result = summarize_cluster_with_openai(cluster_text, api_key)

        assert result == "This is a test summary of the cluster discussion."
        mock_openai.OpenAI.assert_called_once_with(api_key=api_key)
        mock_client.chat.completions.create.assert_called_once()

        # Verify the prompt structure
        call_args = mock_client.chat.completions.create.call_args
        assert call_args[1]["model"] == "gpt-4o-mini"
        assert len(call_args[1]["messages"]) == 2
        assert call_args[1]["messages"][0]["role"] == "system"
        assert call_args[1]["messages"][1]["role"] == "user"

    @patch("terrorblade.examples.cluster_analysis_cli.OPENAI_AVAILABLE", False)
    def test_summarize_cluster_without_openai(self) -> None:
        """Test cluster summarization when OpenAI is not available."""
        result = summarize_cluster_with_openai("test", "test-key")
        assert "OpenAI library not available" in result

    @patch("terrorblade.examples.cluster_analysis_cli.OPENAI_AVAILABLE", True)
    @patch("terrorblade.examples.cluster_analysis_cli.openai")
    def test_extract_story_third_person(self, mock_openai: Mock) -> None:
        """Test story extraction in third person."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Once upon a time, Alice and Bob had a conversation..."

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.OpenAI.return_value = mock_client

        cluster_text = "[2024-01-15 09:30:00] Alice: Hello everyone!"
        api_key = "test-api-key"

        result = extract_story_from_cluster(cluster_text, api_key, "third_person")
        assert result == "Once upon a time, Alice and Bob had a conversation..."

        # Check that it mentions third-person in the prompt
        call_args = mock_client.chat.completions.create.call_args
        prompt_text = call_args[1]["messages"][1]["content"]
        assert "third-person" in prompt_text.lower()

    @patch("terrorblade.examples.cluster_analysis_cli.OPENAI_AVAILABLE", True)
    @patch("terrorblade.examples.cluster_analysis_cli.openai")
    def test_extract_story_first_person(self, mock_openai: Mock) -> None:
        """Test story extraction in first person."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "I remember when Alice and I had a conversation..."

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.OpenAI.return_value = mock_client

        cluster_text = "[2024-01-15 09:30:00] Alice: Hello everyone!"
        api_key = "test-api-key"

        result = extract_story_from_cluster(cluster_text, api_key, "first_person")
        assert result == "I remember when Alice and I had a conversation..."

        # Check that it mentions first-person in the prompt
        call_args = mock_client.chat.completions.create.call_args
        prompt_text = call_args[1]["messages"][1]["content"]
        assert "first-person" in prompt_text.lower()

    @patch("terrorblade.examples.cluster_analysis_cli.OPENAI_AVAILABLE", True)
    @patch("terrorblade.examples.cluster_analysis_cli.openai")
    def test_ai_functions_with_api_error(self, mock_openai: Mock) -> None:
        """Test AI functions when API call fails."""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_openai.OpenAI.return_value = mock_client

        result = summarize_cluster_with_openai("test", "test-key")
        assert "Error generating summary" in result

        result = extract_story_from_cluster("test", "test-key")
        assert "Error generating story" in result

    @patch("terrorblade.examples.cluster_analysis_cli.OPENAI_AVAILABLE", True)
    @patch("terrorblade.examples.cluster_analysis_cli.openai")
    def test_ai_functions_with_empty_response(self, mock_openai: Mock) -> None:
        """Test AI functions when API returns empty response."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = None

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.OpenAI.return_value = mock_client

        result = summarize_cluster_with_openai("test", "test-key")
        assert result == "No summary generated"

        result = extract_story_from_cluster("test", "test-key")
        assert result == "No story generated"


class TestClusterAnalyzerMocked:
    """Test ClusterAnalyzer with fully mocked dependencies."""

    @patch("terrorblade.examples.cluster_analysis_cli.TelegramDatabase")
    @patch("terrorblade.examples.cluster_analysis_cli.get_db_path")
    def test_cluster_analyzer_init_mocked(self, mock_get_db_path: Mock, mock_db_class: Mock) -> None:
        """Test ClusterAnalyzer initialization with mocked database."""
        from terrorblade.examples.cluster_analysis_cli import ClusterAnalyzer

        # Mock the get_db_path to return a predictable path
        mock_get_db_path.return_value = "/tmp/test.db"

        # Mock the database instance
        mock_db_instance = Mock()
        mock_db_class.return_value = mock_db_instance

        analyzer = ClusterAnalyzer(phone="+1234567890", db_path="test.db")

        assert analyzer.phone == "+1234567890"
        assert analyzer.phone_clean == "1234567890"
        assert analyzer.messages_table == "messages_1234567890"
        assert analyzer.clusters_table == "message_clusters_1234567890"
        assert analyzer.chat_names_table == "chat_names_1234567890"
        assert analyzer.user_names_table == "user_names_1234567890"

        # Verify get_db_path was called with the provided db_path
        mock_get_db_path.assert_called_once_with("test.db")

        # Verify database was initialized correctly with the resolved path
        mock_db_class.assert_called_once_with(db_path="/tmp/test.db", read_only=True)

    @patch("terrorblade.examples.cluster_analysis_cli.TelegramDatabase")
    @patch("terrorblade.examples.cluster_analysis_cli.get_db_path")
    def test_find_chat_by_name_mocked(self, mock_get_db_path: Mock, mock_db_class: Mock) -> None:
        """Test finding chat by name with mocked database."""
        from terrorblade.examples.cluster_analysis_cli import ClusterAnalyzer

        # Mock the get_db_path to return a predictable path
        mock_get_db_path.return_value = "/tmp/test.db"

        # Mock the database instance
        mock_db_instance = Mock()
        mock_db_class.return_value = mock_db_instance

        # Mock the execute method
        mock_result = Mock()
        mock_result.fetchone.return_value = [-1001234567890]
        mock_db_instance.db.execute.return_value = mock_result

        analyzer = ClusterAnalyzer(phone="+1234567890", db_path="test.db")
        result = analyzer.find_chat_by_name("family")

        assert result == -1001234567890

        # Test no results
        mock_result.fetchone.return_value = None
        result = analyzer.find_chat_by_name("nonexistent")
        assert result is None

    @patch("terrorblade.examples.cluster_analysis_cli.TelegramDatabase")
    @patch("terrorblade.examples.cluster_analysis_cli.get_db_path")
    def test_get_cluster_summary_data_mocked(self, mock_get_db_path: Mock, mock_db_class: Mock) -> None:
        """Test getting cluster summary data with mocked database."""
        from terrorblade.examples.cluster_analysis_cli import ClusterAnalyzer

        # Mock the get_db_path to return a predictable path
        mock_get_db_path.return_value = "/tmp/test.db"

        # Mock the database instance
        mock_db_instance = Mock()
        mock_db_class.return_value = mock_db_instance

        # Create sample messages data
        sample_messages = pl.DataFrame(
            {
                "date": [datetime(2024, 1, 15, 9, 30), datetime(2024, 1, 15, 9, 35)],
                "from_name": ["Alice", "Bob"],
                "text": ["Hello everyone!", "How are you?"],
            }
        )

        # Mock the execute method
        mock_result = Mock()
        mock_result.arrow.return_value = sample_messages.to_arrow()
        mock_db_instance.db.execute.return_value = mock_result

        analyzer = ClusterAnalyzer(phone="+1234567890", db_path="test.db")
        result = analyzer.get_cluster_summary_data(chat_id=-1001234567890, group_id=15)

        assert isinstance(result, str)
        assert "Alice: Hello everyone!" in result
        assert "Bob: How are you?" in result
        assert "2024-01-15" in result

def test_polars_functionality() -> None:
    """Test that polars operations work as expected."""
    # Test DataFrame creation and manipulation
    df = pl.DataFrame({"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"], "count": [10, 20, 30]})

    assert len(df) == 3
    assert "name" in df.columns
    assert df["count"].sum() == 60

    # Test filtering and transformation
    filtered = df.filter(pl.col("count") > 15)
    assert len(filtered) == 2

    # Test with datetime
    df_time = pl.DataFrame({"timestamp": [datetime(2024, 1, 15, 9, 30), datetime(2024, 1, 15, 10, 0)], "value": [1, 2]})

    formatted = df_time.with_columns([pl.col("timestamp").dt.strftime("%Y-%m-%d %H:%M").alias("time_str")])

    assert "time_str" in formatted.columns
    assert "2024-01-15" in formatted["time_str"][0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
