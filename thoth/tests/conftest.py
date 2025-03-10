import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(scope="session")
def test_db_path():
    """Create a temporary database file path for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        yield tmp.name


@pytest.fixture(scope="session")
def test_phone():
    """Return a test phone number."""
    return "+1234567890"


@pytest.fixture(scope="session")
def test_qdrant_path():
    """Create a temporary directory for Qdrant storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_qdrant():
    """Create a mock Qdrant client."""
    with patch("qdrant_client.QdrantClient") as mock:
        mock_client = MagicMock()
        mock.return_value = mock_client
        yield mock_client


@pytest.fixture
def mock_transformer():
    """Create a mock SentenceTransformer."""
    with patch("sentence_transformers.SentenceTransformer") as mock:
        mock_model = MagicMock()
        mock.return_value = mock_model
        yield mock_model
