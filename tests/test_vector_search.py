"""Comprehensive tests for VectorSearch functionality.

Tests cover default pipeline with test data, different input parameters,
and exceptional cases following test_preprocessor formatting example.
"""

import shutil
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from terrorblade.data.database.telegram_database import TelegramDatabase
from terrorblade.data.database.vector_store import VectorStore
from terrorblade.data.preprocessing.TelegramPreprocessor import TelegramPreprocessor


class TestVectorSearch:
    """Comprehensive test suite for VectorSearch functionality."""

    @classmethod
    def setup_class(cls) -> None:
        """Set up temporary directory and test data for all tests."""
        cls.temp_dir = Path(tempfile.mkdtemp(prefix="terrorblade_vector_test_"))
        cls.test_phone = "+1234567890"
        cls.phone_clean = cls.test_phone.replace("+", "")

        # Create test data
        cls._create_test_data()

    @classmethod
    def teardown_class(cls) -> None:
        """Clean up temporary files after all tests."""
        if hasattr(cls, "temp_dir") and cls.temp_dir.exists():
            shutil.rmtree(cls.temp_dir)

    @classmethod
    def _create_test_data(cls) -> None:
        """Create test data for vector search tests."""
        # Create sample embeddings and messages for testing
        cls.sample_texts = [
            "Hello world, how are you?",
            "Good morning everyone!",
            "I love programming in Python",
            "Machine learning is fascinating",
            "The weather is nice today",
            "Let's go for a walk",
            "Data science and AI are exciting",
            "Neural networks are powerful tools",
            "Natural language processing rocks",
            "Vector databases are useful",
        ]

        # Create realistic embeddings for the texts
        cls.sample_embeddings = []
        for text in cls.sample_texts:
            # Create embeddings that have some semantic similarity
            # Similar texts get similar base vectors
            base_vector = np.random.rand(768).astype(np.float32)
            if "programming" in text.lower() or "python" in text.lower():
                base_vector[0:10] = 0.8  # Programming cluster
            elif "machine learning" in text.lower() or "ai" in text.lower() or "neural" in text.lower():
                base_vector[10:20] = 0.8  # ML cluster
            elif "weather" in text.lower() or "walk" in text.lower():
                base_vector[20:30] = 0.8  # Outdoor cluster

            cls.sample_embeddings.append(base_vector.tolist())

    @pytest.fixture
    def test_db_path(self) -> str:
        """Create a unique database path for each test."""
        return str(self.__class__.temp_dir / f"test_vector_{datetime.now().timestamp()}.db")

    @pytest.fixture
    def telegram_database(self, test_db_path: str) -> TelegramDatabase:
        """Create a TelegramDatabase instance for testing."""
        return TelegramDatabase(db_path=test_db_path)

    @pytest.fixture
    def sample_messages_df(self) -> pl.DataFrame:
        """Create sample message DataFrame with embeddings for testing."""
        return pl.DataFrame(
            {
                "text": self.__class__.sample_texts,
                "from_id": [i % 3 + 1 for i in range(10)],
                "date": [datetime.now() - timedelta(minutes=i * 5) for i in range(10)],
                "chat_id": [100 + (i % 2) for i in range(10)],  # Two different chats
                "message_id": list(range(1, 11)),
                "from_name": [f"User{i % 3 + 1}" for i in range(10)],
                "chat_name": ["Test Chat 1" if i % 2 == 0 else "Test Chat 2" for i in range(10)],
                "reply_to_message_id": [None] * 10,
                "forwarded_from": [None] * 10,
                "media_type": [None] * 10,
                "file_name": [None] * 10,
                "embeddings": self.__class__.sample_embeddings,
            }
        ).with_columns(pl.col("embeddings").cast(pl.Array(pl.Float32, shape=768)))

    @pytest.fixture
    def vector_store_with_data(
        self,
        test_db_path: str,
        telegram_database: TelegramDatabase,
        sample_messages_df: pl.DataFrame,
    ) -> VectorStore:
        """Create VectorStore with sample data populated."""
        # Initialize user tables and add data
        telegram_database.init_user_tables(self.__class__.test_phone)
        telegram_database.add_messages(self.__class__.test_phone, sample_messages_df)

        # Add embeddings to the embeddings table
        preprocessor = TelegramPreprocessor(use_duckdb=True, db_path=test_db_path, phone=self.__class__.test_phone)

        # Add embeddings directly
        embeddings_df = sample_messages_df.select(["message_id", "chat_id", "embeddings"])
        preprocessor._update_embeddings_in_db(embeddings_df)
        preprocessor.close()

        # Create and return VectorStore
        return VectorStore(db_path=test_db_path, phone=self.__class__.test_phone)

    @pytest.fixture
    def empty_vector_store(self, test_db_path: str, telegram_database: TelegramDatabase) -> VectorStore:
        """Create VectorStore with empty tables."""
        telegram_database.init_user_tables(self.__class__.test_phone)

        # Create the embeddings table that VectorStore expects
        preprocessor = TelegramPreprocessor(use_duckdb=True, db_path=test_db_path, phone=self.__class__.test_phone)
        preprocessor.close()

        return VectorStore(db_path=test_db_path, phone=self.__class__.test_phone)

    def test_vector_store_init(self, test_db_path: str, telegram_database: TelegramDatabase) -> None:
        """Test VectorStore initialization."""
        telegram_database.init_user_tables(self.__class__.test_phone)

        # Create the embeddings table that VectorStore expects
        preprocessor = TelegramPreprocessor(use_duckdb=True, db_path=test_db_path, phone=self.__class__.test_phone)
        preprocessor.close()

        vector_store = VectorStore(db_path=test_db_path, phone=self.__class__.test_phone)

        assert vector_store.db_path == test_db_path
        assert vector_store.phone == self.__class__.phone_clean
        assert vector_store.embeddings_table == f"chat_embeddings_{self.__class__.phone_clean}"
        assert vector_store.messages_table == f"messages_{self.__class__.phone_clean}"
        assert vector_store.clusters_table == f"message_clusters_{self.__class__.phone_clean}"
        assert vector_store.db is not None

        vector_store.close()

    def test_vector_store_init_invalid_phone(self, test_db_path: str, telegram_database: TelegramDatabase) -> None:
        """Test VectorStore initialization with phone number formatting."""
        telegram_database.init_user_tables("+987654321")

        # Create the embeddings table that VectorStore expects
        preprocessor = TelegramPreprocessor(use_duckdb=True, db_path=test_db_path, phone="+987654321")
        preprocessor.close()

        # Test phone number normalization
        vector_store = VectorStore(db_path=test_db_path, phone="+987654321")
        assert vector_store.phone == "987654321"  # Should strip the +

        vector_store.close()

    def test_vector_store_init_nonexistent_table(self, test_db_path: str) -> None:
        """Test VectorStore initialization with non-existent embeddings table."""
        # Don't create tables first - should fail gracefully
        with pytest.raises((ValueError, Exception)):
            VectorStore(db_path=test_db_path, phone=self.__class__.test_phone)

    def test_create_hnsw_index(self, vector_store_with_data: VectorStore) -> None:
        """Test HNSW index creation."""
        index_name = f"test_idx_embeddings_{self.__class__.phone_clean}"

        # Test index creation
        result = vector_store_with_data.create_hnsw_index(index_name=index_name)
        assert result is True

        # Verify index exists
        assert vector_store_with_data.check_index_exists(index_name)

        # Test index recreation without force
        result = vector_store_with_data.create_hnsw_index(index_name=index_name)
        assert result is False  # Should not recreate without force

        # Test forced recreation
        result = vector_store_with_data.create_hnsw_index(index_name=index_name, force_recreate=True)
        assert result is True

        vector_store_with_data.close()

    def test_create_hnsw_index_default_name(self, vector_store_with_data: VectorStore) -> None:
        """Test HNSW index creation with default name."""
        result = vector_store_with_data.create_hnsw_index()
        assert result is True

        default_index_name = f"idx_embeddings_{self.__class__.phone_clean}"
        assert vector_store_with_data.check_index_exists(default_index_name)

        vector_store_with_data.close()

    def test_index_stats(self, vector_store_with_data: VectorStore) -> None:
        """Test index statistics retrieval."""
        index_name = f"test_idx_embeddings_{self.__class__.phone_clean}"

        # Create index first
        vector_store_with_data.create_hnsw_index(index_name=index_name)

        # Test index stats
        stats = vector_store_with_data.get_index_stats(index_name)
        assert isinstance(stats, dict)
        assert stats["index_name"] == index_name
        assert stats["table_name"] == f"chat_embeddings_{self.__class__.phone_clean}"
        assert stats["index_type"] == "HNSW"
        assert stats["indexed_rows"] == 10  # We have 10 sample messages
        assert "estimated_memory_mb" in stats
        assert isinstance(stats["estimated_memory_mb"], float)

        vector_store_with_data.close()

    def test_check_index_exists(self, vector_store_with_data: VectorStore) -> None:
        """Test index existence checking."""
        index_name = f"test_idx_embeddings_{self.__class__.phone_clean}"

        # Should not exist initially
        assert not vector_store_with_data.check_index_exists(index_name)

        # Create index
        vector_store_with_data.create_hnsw_index(index_name=index_name)

        # Should exist now
        assert vector_store_with_data.check_index_exists(index_name)

        # Test with non-existent index
        assert not vector_store_with_data.check_index_exists("non_existent_index")

        vector_store_with_data.close()

    def test_cosine_similarity(self, vector_store_with_data: VectorStore) -> None:
        """Test cosine similarity calculation."""
        # Test identical vectors
        vector1 = [1.0] * 768
        vector2 = [1.0] * 768
        similarity = vector_store_with_data.cosine_similarity(vector1, vector2)
        assert abs(similarity - 1.0) < 0.001  # Should be 1.0 for identical vectors

        # Test orthogonal vectors
        vector1 = [1.0] + [0.0] * 767
        vector2 = [0.0] + [1.0] + [0.0] * 766
        similarity = vector_store_with_data.cosine_similarity(vector1, vector2)
        assert abs(similarity - 0.0) < 0.001  # Should be 0.0 for orthogonal vectors

        # Test opposite vectors
        vector1 = [1.0] * 768
        vector2 = [-1.0] * 768
        similarity = vector_store_with_data.cosine_similarity(vector1, vector2)
        assert abs(similarity - (-1.0)) < 0.001  # Should be -1.0 for opposite vectors

        vector_store_with_data.close()

    def test_cosine_distance(self, vector_store_with_data: VectorStore) -> None:
        """Test cosine distance calculation."""
        # Test identical vectors (distance should be 0)
        vector1 = [1.0] * 768
        vector2 = [1.0] * 768
        distance = vector_store_with_data.cosine_distance(vector1, vector2)
        assert abs(distance - 0.0) < 0.001

        # Test opposite vectors (distance should be 2)
        vector1 = [1.0] * 768
        vector2 = [-1.0] * 768
        distance = vector_store_with_data.cosine_distance(vector1, vector2)
        assert abs(distance - 2.0) < 0.001

        vector_store_with_data.close()

    def test_similarity_search(self, vector_store_with_data: VectorStore) -> None:
        """Test similarity search functionality."""
        # Create index for efficient search
        vector_store_with_data.create_hnsw_index()

        # Test search with sample embedding (should find similar vectors)
        query_vector = self.__class__.sample_embeddings[0]  # First embedding
        results = vector_store_with_data.similarity_search(query_vector, top_k=5)

        assert isinstance(results, list)
        assert len(results) <= 5
        assert len(results) > 0  # Should find at least the exact match

        # Check result format: (message_id, chat_id, similarity)
        for result in results:
            assert len(result) == 3
            assert isinstance(result[0], int)  # message_id
            assert isinstance(result[1], int)  # chat_id
            assert isinstance(result[2], float)  # similarity
            assert 0.0 <= result[2] <= 1.0  # Similarity should be in [0, 1]

        # Results should be sorted by similarity (descending)
        similarities = [result[2] for result in results]
        assert similarities == sorted(similarities, reverse=True)

        vector_store_with_data.close()

    def test_similarity_search_with_chat_filter(self, vector_store_with_data: VectorStore) -> None:
        """Test similarity search with chat_id filtering."""
        vector_store_with_data.create_hnsw_index()

        query_vector = self.__class__.sample_embeddings[0]

        # Test with specific chat_id
        results_chat_100 = vector_store_with_data.similarity_search(query_vector, top_k=10, chat_id=100)
        results_chat_101 = vector_store_with_data.similarity_search(query_vector, top_k=10, chat_id=101)

        # All results should have the specified chat_id
        for result in results_chat_100:
            assert result[1] == 100

        for result in results_chat_101:
            assert result[1] == 101

        # Should have results (both chats have data in our test setup)
        assert len(results_chat_100) > 0
        assert len(results_chat_101) > 0

        vector_store_with_data.close()

    def test_similarity_search_with_threshold(self, vector_store_with_data: VectorStore) -> None:
        """Test similarity search with similarity threshold."""
        vector_store_with_data.create_hnsw_index()

        query_vector = self.__class__.sample_embeddings[0]

        # Test with high threshold (should return fewer results)
        results_high_threshold = vector_store_with_data.similarity_search(
            query_vector, top_k=10, similarity_threshold=0.9
        )

        # Test with low threshold (should return more results)
        results_low_threshold = vector_store_with_data.similarity_search(
            query_vector, top_k=10, similarity_threshold=0.1
        )

        # High threshold should return fewer or equal results
        assert len(results_high_threshold) <= len(results_low_threshold)

        # All results should meet the threshold
        for result in results_high_threshold:
            assert result[2] >= 0.9

        for result in results_low_threshold:
            assert result[2] >= 0.1

        vector_store_with_data.close()

    def test_distance_search(self, vector_store_with_data: VectorStore) -> None:
        """Test distance-based search functionality."""
        vector_store_with_data.create_hnsw_index()

        query_vector = self.__class__.sample_embeddings[0]
        results = vector_store_with_data.distance_search(query_vector, top_k=5)

        assert isinstance(results, list)
        assert len(results) <= 5
        assert len(results) > 0

        # Check result format: (message_id, chat_id, distance)
        for result in results:
            assert len(result) == 3
            assert isinstance(result[0], int)  # message_id
            assert isinstance(result[1], int)  # chat_id
            assert isinstance(result[2], float)  # distance
            assert result[2] >= 0.0  # Distance should be non-negative

        # Results should be sorted by distance (ascending)
        distances = [result[2] for result in results]
        assert distances == sorted(distances)

        vector_store_with_data.close()

    def test_distance_search_with_threshold(self, vector_store_with_data: VectorStore) -> None:
        """Test distance search with distance threshold."""
        vector_store_with_data.create_hnsw_index()

        query_vector = self.__class__.sample_embeddings[0]

        # Test with low threshold (should return fewer results)
        results_low_threshold = vector_store_with_data.distance_search(query_vector, top_k=10, distance_threshold=0.5)

        # Test with high threshold (should return more results)
        results_high_threshold = vector_store_with_data.distance_search(query_vector, top_k=10, distance_threshold=1.5)

        # Low threshold should return fewer or equal results
        assert len(results_low_threshold) <= len(results_high_threshold)

        # All results should meet the threshold
        for result in results_low_threshold:
            assert result[2] <= 0.5

        for result in results_high_threshold:
            assert result[2] <= 1.5

        vector_store_with_data.close()

    def test_get_all_distances(self, vector_store_with_data: VectorStore) -> None:
        """Test getting distances to all messages."""
        query_vector = self.__class__.sample_embeddings[0]
        result_df = vector_store_with_data.get_all_distances(query_vector)

        assert isinstance(result_df, pl.DataFrame)
        assert not result_df.is_empty()
        assert set(result_df.columns) == {"message_id", "chat_id", "distance", "similarity"}

        # Should have results for all messages
        assert len(result_df) == 10  # We have 10 sample messages

        # Check data types
        assert result_df["message_id"].dtype == pl.Int64
        assert result_df["chat_id"].dtype == pl.Int64
        assert result_df["distance"].dtype == pl.Float64
        assert result_df["similarity"].dtype == pl.Float64

        # Distances should be non-negative
        assert all(result_df["distance"] >= 0)

        # Similarities should be between -1 and 1
        assert all(result_df["similarity"] >= -1)
        assert all(result_df["similarity"] <= 1)

        vector_store_with_data.close()

    def test_get_all_distances_with_chat_filter(self, vector_store_with_data: VectorStore) -> None:
        """Test getting distances with chat_id filter."""
        query_vector = self.__class__.sample_embeddings[0]

        # Test with specific chat_id
        result_df = vector_store_with_data.get_all_distances(query_vector, chat_id=100)

        assert isinstance(result_df, pl.DataFrame)
        assert not result_df.is_empty()

        # All results should have the specified chat_id
        assert all(result_df["chat_id"] == 100)

        # Should have fewer results than total
        assert len(result_df) < 10

        vector_store_with_data.close()

    def test_get_embedding(self, vector_store_with_data: VectorStore) -> None:
        """Test retrieving specific embeddings."""
        # Test retrieving existing embedding
        embedding = vector_store_with_data.get_embedding(message_id=1, chat_id=100)

        assert embedding is not None
        assert isinstance(embedding, list | tuple)  # DuckDB might return tuple
        assert len(embedding) == 768
        assert all(isinstance(x, float) for x in embedding)

        # Test retrieving non-existent embedding
        embedding = vector_store_with_data.get_embedding(message_id=999, chat_id=100)
        assert embedding is None

        # Test with non-existent chat_id
        embedding = vector_store_with_data.get_embedding(message_id=1, chat_id=999)
        assert embedding is None

        vector_store_with_data.close()

    def test_get_similar_messages_with_text(self, vector_store_with_data: VectorStore) -> None:
        """Test similarity search with text and metadata."""
        vector_store_with_data.create_hnsw_index()

        query_vector = self.__class__.sample_embeddings[0]
        result_df = vector_store_with_data.get_similar_messages_with_text(query_vector, top_k=5)

        assert isinstance(result_df, pl.DataFrame)
        assert not result_df.is_empty()

        expected_columns = {
            "message_id",
            "chat_id",
            "text",
            "chat_name",
            "from_name",
            "date",
            "similarity",
            "cluster_id",
            "text_preview",
        }
        assert set(result_df.columns) == expected_columns

        # Check that we have at most 5 results
        assert len(result_df) <= 5

        # Check similarity values are in correct range
        assert all(result_df["similarity"] >= 0)
        assert all(result_df["similarity"] <= 1)

        # Check that results are sorted by similarity (descending)
        similarities = result_df["similarity"].to_list()
        assert similarities == sorted(similarities, reverse=True)

        vector_store_with_data.close()

    def test_get_similar_messages_with_text_and_chat_filter(self, vector_store_with_data: VectorStore) -> None:
        """Test similarity search with text and chat_id filter."""
        vector_store_with_data.create_hnsw_index()

        query_vector = self.__class__.sample_embeddings[0]
        result_df = vector_store_with_data.get_similar_messages_with_text(query_vector, top_k=10, chat_id=100)

        assert isinstance(result_df, pl.DataFrame)

        if not result_df.is_empty():
            # All results should have the specified chat_id
            assert all(result_df["chat_id"] == 100)

        vector_store_with_data.close()

    def test_get_similar_messages_with_threshold(self, vector_store_with_data: VectorStore) -> None:
        """Test similarity search with similarity threshold."""
        vector_store_with_data.create_hnsw_index()

        query_vector = self.__class__.sample_embeddings[0]
        result_df = vector_store_with_data.get_similar_messages_with_text(
            query_vector, top_k=10, similarity_threshold=0.8
        )

        assert isinstance(result_df, pl.DataFrame)

        if not result_df.is_empty():
            # All results should meet the threshold
            assert all(result_df["similarity"] >= 0.8)

        vector_store_with_data.close()

    def test_get_table_stats(self, vector_store_with_data: VectorStore) -> None:
        """Test table statistics retrieval."""
        stats = vector_store_with_data.get_table_stats()

        assert isinstance(stats, dict)
        assert "total_embeddings" in stats
        assert "unique_chats" in stats
        assert "chat_breakdown" in stats

        assert stats["total_embeddings"] == 10  # We have 10 sample messages
        assert stats["unique_chats"] == 2  # We have 2 different chat_ids
        assert isinstance(stats["chat_breakdown"], dict)

        # Check chat breakdown
        chat_breakdown = stats["chat_breakdown"]
        assert 100 in chat_breakdown
        assert 101 in chat_breakdown
        assert sum(chat_breakdown.values()) == 10

        vector_store_with_data.close()

    def test_empty_table_operations(self, empty_vector_store: VectorStore) -> None:
        """Test operations on empty tables."""
        # Test search on empty table
        query_vector = [0.1] * 768
        results = empty_vector_store.similarity_search(query_vector, top_k=5)
        assert results == []

        # Test distance search on empty table
        results = empty_vector_store.distance_search(query_vector, top_k=5)
        assert results == []

        # Test get all distances on empty table
        result_df = empty_vector_store.get_all_distances(query_vector)
        assert result_df.is_empty()

        # Test get similar messages with text on empty table
        result_df = empty_vector_store.get_similar_messages_with_text(query_vector, top_k=5)
        assert result_df.is_empty()

        # Test table stats on empty table
        stats = empty_vector_store.get_table_stats()
        assert stats["total_embeddings"] == 0
        assert stats["unique_chats"] == 0
        assert stats["chat_breakdown"] == {}

        empty_vector_store.close()

    def test_invalid_vector_dimensions(self, vector_store_with_data: VectorStore) -> None:
        """Test error handling for invalid vector dimensions."""
        # Test with wrong dimension vector
        invalid_vector = [0.1] * 100  # Should be 768

        # These should handle errors gracefully
        results = vector_store_with_data.similarity_search(invalid_vector, top_k=5)
        assert results == []  # Should return empty list on error

        results = vector_store_with_data.distance_search(invalid_vector, top_k=5)
        assert results == []

        result_df = vector_store_with_data.get_all_distances(invalid_vector)
        assert result_df.is_empty()

        result_df = vector_store_with_data.get_similar_messages_with_text(invalid_vector, top_k=5)
        assert result_df.is_empty()

        vector_store_with_data.close()

    def test_vector_operations_edge_cases(self, vector_store_with_data: VectorStore) -> None:
        """Test vector operations with edge cases."""
        # Test with zero vector
        zero_vector = [0.0] * 768
        similarity = vector_store_with_data.cosine_similarity(zero_vector, zero_vector)
        # Zero vector similarity with itself may return -1.0 or NaN in DuckDB
        assert similarity in [0.0, -1.0] or str(similarity) == "nan"

        # Test with empty vectors (should handle gracefully)
        empty_vector = []
        similarity = vector_store_with_data.cosine_similarity(empty_vector, empty_vector)
        assert similarity == 0.0  # Should return default value

        vector_store_with_data.close()

    def test_index_operations_edge_cases(self, empty_vector_store: VectorStore) -> None:
        """Test index operations on empty tables."""
        # Test creating index on empty table (should work)
        result = empty_vector_store.create_hnsw_index()
        assert result is True

        # Test index stats on empty table
        stats = empty_vector_store.get_index_stats()
        assert "indexed_rows" in stats
        assert stats["indexed_rows"] == 0

        empty_vector_store.close()

    def test_parameter_combinations_similarity_search(self, vector_store_with_data: VectorStore) -> None:
        """Test similarity search with different parameter combinations."""
        vector_store_with_data.create_hnsw_index()
        query_vector = self.__class__.sample_embeddings[0]

        # Test different top_k values
        for top_k in [1, 3, 5, 10]:
            results = vector_store_with_data.similarity_search(query_vector, top_k=top_k)
            assert len(results) <= top_k

        # Test different threshold values
        for threshold in [0.0, 0.3, 0.5, 0.8, 0.95]:
            results = vector_store_with_data.similarity_search(query_vector, top_k=10, similarity_threshold=threshold)
            for result in results:
                assert result[2] >= threshold

        # Test with both chat_id filter and threshold
        results = vector_store_with_data.similarity_search(query_vector, top_k=5, chat_id=100, similarity_threshold=0.3)
        for result in results:
            assert result[1] == 100  # chat_id filter
            assert result[2] >= 0.3  # threshold

        vector_store_with_data.close()

    def test_parameter_combinations_distance_search(self, vector_store_with_data: VectorStore) -> None:
        """Test distance search with different parameter combinations."""
        vector_store_with_data.create_hnsw_index()
        query_vector = self.__class__.sample_embeddings[0]

        # Test different top_k values
        for top_k in [1, 3, 5, 10]:
            results = vector_store_with_data.distance_search(query_vector, top_k=top_k)
            assert len(results) <= top_k

        # Test different threshold values
        for threshold in [0.1, 0.5, 1.0, 1.5, 2.0]:
            results = vector_store_with_data.distance_search(query_vector, top_k=10, distance_threshold=threshold)
            for result in results:
                assert result[2] <= threshold

        vector_store_with_data.close()

    def test_close_operations(self, vector_store_with_data: VectorStore) -> None:
        """Test database connection closing."""
        # Ensure vector store is functional before closing
        stats = vector_store_with_data.get_table_stats()
        assert stats["total_embeddings"] > 0

        # Close the connection
        vector_store_with_data.close()

        # Operations after closing should handle errors gracefully
        # Note: Specific behavior depends on implementation
        # The close() method should not raise exceptions

    def test_database_error_handling(self, test_db_path: str) -> None:
        """Test error handling for database connection issues."""
        # Test with invalid database path
        invalid_path = "/invalid/path/database.db"

        with pytest.raises((OSError, Exception)):
            VectorStore(db_path=invalid_path, phone=self.__class__.test_phone)

    def test_print_index_stats_functionality(self, vector_store_with_data: VectorStore, capsys) -> None:
        """Test print_index_stats output (captures stdout)."""
        # Create index first
        vector_store_with_data.create_hnsw_index()

        # Call print function
        vector_store_with_data.print_index_stats()

        # Capture output
        captured = capsys.readouterr()

        # Verify expected content in output
        assert "HNSW Index Statistics" in captured.out
        assert "Index Name:" in captured.out
        assert "Indexed Rows:" in captured.out
        assert "Estimated Memory:" in captured.out

        vector_store_with_data.close()

    def test_print_index_stats_no_index(self, empty_vector_store: VectorStore, capsys) -> None:
        """Test print_index_stats when no index exists."""
        # Call print function without creating index
        empty_vector_store.print_index_stats()

        # Capture output
        captured = capsys.readouterr()

        # Should indicate no index exists
        assert "Index does not exist" in captured.out

        empty_vector_store.close()
