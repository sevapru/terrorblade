"""
Tests for the animal message transformer script.
"""

import json
import tempfile
from pathlib import Path

from scripts.animal_message_transformer import transform_messages


class TestAnimalTransformer:
    """Test class for the animal message transformer."""

    def test_transform_messages_structure(self) -> None:
        """Test that transformed messages maintain the correct structure."""
        # Load original test data
        with open("tests/data/messages_test.json", encoding="utf-8") as f:
            original_data = json.load(f)

        # Transform the data
        transformed_data = transform_messages(original_data)

        # Check that the structure is maintained
        assert "chats" in transformed_data
        assert "list" in transformed_data["chats"]
        assert len(transformed_data["chats"]["list"]) > 0

        # Check that messages have the expected structure
        for chat in transformed_data["chats"]["list"]:
            assert "messages" in chat
            for message in chat["messages"]:
                assert "id" in message
                assert "type" in message
                assert "date" in message
                # Only check "from" and "text" for regular messages, not service messages
                if message["type"] == "message":
                    assert "from" in message
                    assert "text" in message

    def test_author_transformation(self) -> None:
        """Test that author names are properly transformed."""
        # Load original test data
        with open("tests/data/messages_test.json", encoding="utf-8") as f:
            original_data = json.load(f)

        # Transform the data
        transformed_data = transform_messages(original_data)

        # Check that author names are transformed
        original_authors = set()
        transformed_authors = set()

        for chat in original_data["chats"]["list"]:
            for message in chat["messages"]:
                if "from" in message:
                    original_authors.add(message["from"])

        for chat in transformed_data["chats"]["list"]:
            for message in chat["messages"]:
                if "from" in message:
                    transformed_authors.add(message["from"])

        # Check that we have animal-themed names
        animal_indicators = ["🦁", "🦅", "🐯", "🐬", "🐧", "🐺", "🦊"]
        has_animal_names = any(
            any(indicator in author for indicator in animal_indicators)
            for author in transformed_authors
        )

        assert has_animal_names, "Transformed authors should contain animal emojis"
        assert len(transformed_authors) > 0, "Should have transformed authors"

    def test_message_content_transformation(self) -> None:
        """Test that message content is transformed to animal themes."""
        # Load original test data
        with open("tests/data/messages_test.json", encoding="utf-8") as f:
            original_data = json.load(f)

        # Transform the data
        transformed_data = transform_messages(original_data)

        # Check that text messages are transformed
        original_texts = []
        transformed_texts = []

        for chat in original_data["chats"]["list"]:
            for message in chat["messages"]:
                if message.get("text", "").strip():
                    original_texts.append(message["text"])

        for chat in transformed_data["chats"]["list"]:
            for message in chat["messages"]:
                if message.get("text", "").strip():
                    transformed_texts.append(message["text"])

        # Check that we have animal-themed content
        animal_keywords = ["animal", "cool", "amazing", "incredible", "awesome", "fascinating"]
        has_animal_content = any(
            any(keyword in text.lower() for keyword in animal_keywords)
            for text in transformed_texts
        )

        assert has_animal_content, "Transformed messages should contain animal-related content"
        assert len(transformed_texts) > 0, "Should have transformed text messages"

    def test_transformed_data_file(self) -> None:
        """Test that the transformed data file exists and is valid JSON."""
        transformed_file = Path("tests/data/messages_test_animals.json")

        assert transformed_file.exists(), "Transformed data file should exist"

        # Check that it's valid JSON
        with open(transformed_file, encoding="utf-8") as f:
            data = json.load(f)

        assert "chats" in data, "Transformed file should have chats structure"
        assert "list" in data["chats"], "Transformed file should have chats list"

    def test_script_execution(self) -> None:
        """Test that the script can be executed successfully."""
        import subprocess
        import sys

        # Create a temporary test file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            test_data = {
                "chats": {
                    "list": [
                        {
                            "name": "Test Chat",
                            "messages": [
                                {
                                    "id": 1,
                                    "type": "message",
                                    "date": "2023-01-01T00:00:00",
                                    "from": "Test User",
                                    "text": "Hello world",
                                }
                            ],
                        }
                    ]
                }
            }
            json.dump(test_data, f)
            input_file = f.name

        output_file = input_file.replace(".json", "_animals.json")

        try:
            # Run the script
            result = subprocess.run(
                [sys.executable, "scripts/animal_message_transformer.py", input_file, output_file],
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0, f"Script should exit successfully: {result.stderr}"

            # Check that output file was created
            assert Path(output_file).exists(), "Output file should be created"

            # Check that the output is valid JSON
            with open(output_file) as f:
                output_data = json.load(f)

            assert "chats" in output_data, "Output should have chats structure"

        finally:
            # Clean up
            for file_path in [input_file, output_file]:
                if Path(file_path).exists():
                    Path(file_path).unlink()

    def test_backup_functionality(self) -> None:
        """Test that the backup functionality works."""
        import subprocess
        import sys

        # Create a temporary test file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            test_data = {
                "chats": {
                    "list": [
                        {
                            "name": "Test Chat",
                            "messages": [
                                {
                                    "id": 1,
                                    "type": "message",
                                    "date": "2023-01-01T00:00:00",
                                    "from": "Test User",
                                    "text": "Hello world",
                                }
                            ],
                        }
                    ]
                }
            }
            json.dump(test_data, f)
            input_file = f.name

        output_file = input_file.replace(".json", "_animals.json")
        backup_file = input_file + ".backup"

        try:
            # Run the script with backup flag
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/animal_message_transformer.py",
                    input_file,
                    output_file,
                    "--backup",
                ],
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0, f"Script should exit successfully: {result.stderr}"

            # Check that backup file was created
            assert Path(backup_file).exists(), "Backup file should be created"

            # Check that backup contains original data
            with open(backup_file) as f:
                backup_data = json.load(f)

            assert backup_data == test_data, "Backup should contain original data"

        finally:
            # Clean up
            for file_path in [input_file, output_file, backup_file]:
                if Path(file_path).exists():
                    Path(file_path).unlink()
