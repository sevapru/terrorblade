import logging
import os
from typing import Any

import duckdb
import polars as pl

from terrorblade import Logger


class VectorStore:
    """Vector store class for managing embeddings and performing semantic search using DuckDB VSS extension."""

    def __init__(self, db_path: str, phone: str) -> None:
        """Initialize the vector store with DuckDB connection."""
        self.db_path = db_path
        self.phone = phone.replace("+", "")
        self.embeddings_table = f"chat_embeddings_{self.phone}"
        self.messages_table = f"messages_{self.phone}"
        self.clusters_table = f"message_clusters_{self.phone}"
        self.chat_names_table = f"chat_names_{self.phone}"
        self.user_names_table = f"user_names_{self.phone}"

        self.logger = Logger(
            name="VectorStore",
            level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
            log_file=os.getenv("LOG_FILE", "vector_store.log"),
            log_dir=os.getenv("LOG_DIR", "logs"),
        )

        try:
            self.db = duckdb.connect(db_path)
            self._install_vss_extension()
            self._verify_embeddings_table()
        except Exception as e:
            self.logger.error(f"Error connecting to vector store database: {str(e)}")
            raise

    def _install_vss_extension(self) -> None:
        """Install and load DuckDB VSS extension for vector operations."""
        try:
            self.db.execute("INSTALL vss;")
            self.db.execute("LOAD vss;")
            self.db.execute("SET hnsw_enable_experimental_persistence = true;")
        except Exception as e:
            self.logger.error(f"Error loading VSS extension: {str(e)}")
            raise

    def _verify_embeddings_table(self) -> None:
        """Verify that the embeddings table exists and has the correct structure."""
        try:
            tables = self.db.execute("SHOW TABLES").fetchall()
            table_names = [table[0] for table in tables]

            if self.embeddings_table not in table_names:
                raise ValueError(f"Embeddings table {self.embeddings_table} does not exist")
            schema = self.db.execute(f"DESCRIBE {self.embeddings_table}").fetchall()
            expected_columns = {"message_id", "chat_id", "embeddings"}
            actual_columns = {col[0] for col in schema}

            if not expected_columns.issubset(actual_columns):
                missing = expected_columns - actual_columns
                raise ValueError(f"Missing columns in {self.embeddings_table}: {missing}")
        except Exception as e:
            self.logger.error(f"Error verifying embeddings table: {str(e)}")
            raise

    def _execute_query(self, query: str, params: list[Any], operation: str) -> Any:
        """Execute database query with unified error handling."""
        try:
            return self.db.execute(query, params)
        except Exception as e:
            self.logger.error(f"Error {operation}: {str(e)}")
            raise

    def _build_chat_filter(
        self, chat_id: int | None, base_params: list[Any]
    ) -> tuple[str, list[Any]]:
        """Build WHERE clause and parameters for chat_id filtering."""
        if chat_id is None:
            return "", base_params

        params = base_params.copy()
        params.append(chat_id)
        return "WHERE chat_id = ?" if "WHERE" not in str(base_params) else "AND chat_id = ?", params

    def _vector_operation(
        self, operation: str, vector1: list[float], vector2: list[float], default_value: float
    ) -> float:
        """Generic vector operation (similarity/distance) calculation."""
        try:
            result = self.db.execute(
                f"SELECT array_cosine_{operation}(?::FLOAT[768], ?::FLOAT[768])", [vector1, vector2]
            ).fetchone()
            return result[0] if result else default_value
        except Exception as e:
            self.logger.error(f"Error calculating cosine {operation}: {str(e)}")
            return default_value

    def _trigger_lazy_index_loading(self) -> None:
        """Trigger lazy loading of persisted HNSW index by accessing the table."""
        try:
            self.db.execute(f"SELECT COUNT(*) FROM {self.embeddings_table} LIMIT 1").fetchone()
        except Exception as e:
            self.logger.error(f"Error triggering lazy index loading: {str(e)}")

    def check_index_exists(self, index_name: str | None = None) -> bool:
        """Check if HNSW index exists for the embeddings table."""
        if index_name is None:
            index_name = f"idx_embeddings_{self.phone}"

        try:
            result = self._execute_query(
                "SELECT COUNT(*) FROM duckdb_indexes() WHERE index_name = ? AND table_name = ?",
                [index_name, self.embeddings_table],
                "checking index existence",
            ).fetchone()
            return result[0] > 0 if result else False
        except Exception:
            return False

    def get_index_stats(self, index_name: str | None = None) -> dict[str, Any]:
        """Get statistics about the HNSW index."""
        if index_name is None:
            index_name = f"idx_embeddings_{self.phone}"

        try:
            stats: dict[str, Any] = {}
            index_info = self._execute_query(
                "SELECT index_name, table_name, is_primary FROM duckdb_indexes() WHERE index_name = ? AND table_name = ?",
                [index_name, self.embeddings_table],
                "getting index info",
            ).fetchone()

            if index_info:
                stats.update(
                    {
                        "index_name": index_info[0],
                        "table_name": index_info[1],
                        "index_type": "HNSW",
                        "is_unique": False,
                        "is_primary": index_info[2],
                        "key_columns": "embeddings",
                    }
                )

                count_result = self._execute_query(
                    f"SELECT COUNT(*) FROM {self.embeddings_table}", [], "getting row count"
                ).fetchone()
                stats["indexed_rows"] = count_result[0] if count_result else 0

                vector_size_bytes = 768 * 4
                estimated_memory_mb = (stats["indexed_rows"] * vector_size_bytes * 1.5) / (
                    1024 * 1024
                )
                stats["estimated_memory_mb"] = round(estimated_memory_mb, 2)

            return stats
        except Exception as e:
            self.logger.error(f"Error getting index stats: {str(e)}")
            raise

    def print_index_stats(self, index_name: str | None = None) -> None:
        """Print detailed index statistics in a readable format."""
        if index_name is None:
            index_name = f"idx_embeddings_{self.phone}"

        print("\nHNSW Index Statistics")
        print("=" * 50)

        if not self.check_index_exists(index_name):
            print("Index does not exist")
            return

        stats = self.get_index_stats(index_name)

        if stats:
            print(f"Index Name: {stats.get('index_name', 'N/A')}")
            print(f"Table: {stats.get('table_name', 'N/A')}")
            print(f"Type: {stats.get('index_type', 'N/A')}")
            print(f"Indexed Rows: {stats.get('indexed_rows', 0):,}")
            print(f"Estimated Memory: {stats.get('estimated_memory_mb', 0)} MB")
            print(f"Key Columns: {stats.get('key_columns', 'N/A')}")
            print(f"Unique: {stats.get('is_unique', False)}")

            rows = stats.get("indexed_rows", 0)
            if rows > 0:
                print("\nPerformance Estimates:")
                print(f"   Search complexity: O(log({rows:,}))")
        else:
            print("Could not retrieve index statistics")

        print("=" * 50)

    def create_hnsw_index(
        self, index_name: str | None = None, force_recreate: bool = False
    ) -> bool:
        """Create HNSW index on embeddings column for fast vector similarity search."""
        if index_name is None:
            index_name = f"idx_embeddings_{self.phone}"

        try:
            self._trigger_lazy_index_loading()
            index_exists = self.check_index_exists(index_name)

            if index_exists and not force_recreate:
                return False

            if index_exists:
                self.db.execute(f"DROP INDEX IF EXISTS {index_name}")

            self._execute_query(
                f"CREATE INDEX {index_name} ON {self.embeddings_table} USING HNSW (embeddings) WITH (metric = 'cosine')",
                [],
                "creating HNSW index",
            )
            return True
        except Exception as e:
            self.logger.error(f"Error creating HNSW index: {str(e)}")
            raise

    def cosine_similarity(self, vector1: list[float], vector2: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        return self._vector_operation("similarity", vector1, vector2, 0.0)

    def cosine_distance(self, vector1: list[float], vector2: list[float]) -> float:
        """Calculate cosine distance between two vectors."""
        return self._vector_operation("distance", vector1, vector2, 2.0)

    def similarity_search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        chat_id: int | None = None,
        similarity_threshold: float = 0.0,
    ) -> list[tuple[int, int, float]]:
        """Perform similarity search using HNSW index."""
        where_clause, params = self._build_chat_filter(chat_id, [query_vector, top_k])

        if chat_id is not None:
            params = [query_vector, chat_id, top_k]

        query = f"""
            SELECT message_id, chat_id, array_cosine_similarity(embeddings, ?::FLOAT[768]) as similarity
            FROM {self.embeddings_table}
            {where_clause}
            ORDER BY similarity DESC
            LIMIT ?
        """

        try:
            results = self._execute_query(query, params, "performing similarity search").fetchall()
            filtered_results = [
                (msg_id, chat_id, sim)
                for msg_id, chat_id, sim in results
                if sim >= similarity_threshold
            ]
            return filtered_results
        except Exception:
            return []

    def distance_search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        chat_id: int | None = None,
        distance_threshold: float = 2.0,
    ) -> list[tuple[int, int, float]]:
        """Perform distance-based search using cosine distance."""
        where_clause, params = self._build_chat_filter(chat_id, [query_vector, top_k])

        if chat_id is not None:
            params = [query_vector, chat_id, top_k]

        query = f"""
            SELECT message_id, chat_id, array_cosine_distance(embeddings, ?::FLOAT[768]) as distance
            FROM {self.embeddings_table}
            {where_clause}
            ORDER BY distance ASC
            LIMIT ?
        """

        try:
            results = self._execute_query(query, params, "performing distance search").fetchall()
            filtered_results = [
                (msg_id, chat_id, dist)
                for msg_id, chat_id, dist in results
                if dist <= distance_threshold
            ]
            return filtered_results
        except Exception:
            return []

    def get_all_distances(
        self, query_vector: list[float], chat_id: int | None = None
    ) -> pl.DataFrame:
        """Calculate distances to all messages in the database."""
        where_clause, params = self._build_chat_filter(chat_id, [query_vector, query_vector])

        query = f"""
            SELECT message_id, chat_id,
                   array_cosine_distance(embeddings, ?::FLOAT[768]) as distance,
                   array_cosine_similarity(embeddings, ?::FLOAT[768]) as similarity
            FROM {self.embeddings_table}
            {where_clause}
            ORDER BY distance ASC
        """

        try:
            results = self._execute_query(query, params, "calculating all distances").fetchall()
            return pl.DataFrame(
                results, schema=["message_id", "chat_id", "distance", "similarity"], orient="row"
            )
        except Exception:
            return pl.DataFrame()

    def get_embedding(self, message_id: int, chat_id: int) -> list[float] | None:
        """Retrieve embedding for a specific message."""
        try:
            result = self._execute_query(
                f"SELECT embeddings FROM {self.embeddings_table} WHERE message_id = ? AND chat_id = ?",
                [message_id, chat_id],
                "retrieving embedding",
            ).fetchone()
            return result[0] if result else None
        except Exception:
            return None

    def get_similar_messages_with_text(
        self,
        query_vector: list[float] | str,
        top_k: int = 10,
        chat_id: int | None = None,
        similarity_threshold: float = 0.0,
        include_cluster_messages: bool = True,
    ) -> pl.DataFrame:
        """Perform similarity search and return DataFrame with message texts and cluster information."""

        empty_schema = [
            "message_id",
            "chat_id",
            "text",
            "chat_name",
            "from_name",
            "date",
            "similarity",
            "cluster_id",
            "text_preview",
        ]

        try:
            where_clause = ""
            params: list[Any] = [query_vector, query_vector, top_k]

            if chat_id is not None:
                where_clause = "AND e.chat_id = ?"
                params = [query_vector, query_vector, chat_id, top_k]

            query = f"""
                WITH latest_chat_names AS (
                    SELECT chat_id, chat_name FROM (
                        SELECT chat_id, chat_name,
                               ROW_NUMBER() OVER (PARTITION BY chat_id ORDER BY COALESCE(last_seen, first_seen) DESC) AS rn
                        FROM {self.chat_names_table}
                    ) t WHERE rn = 1
                ),
                latest_user_names AS (
                    SELECT from_id, from_name FROM (
                        SELECT from_id, from_name,
                               ROW_NUMBER() OVER (PARTITION BY from_id ORDER BY COALESCE(last_seen, first_seen) DESC) AS rn
                        FROM {self.user_names_table}
                    ) t WHERE rn = 1
                )
                SELECT e.message_id, e.chat_id, m.text, lcn.chat_name, lun.from_name, m.date,
                       array_cosine_similarity(e.embeddings, ?::FLOAT[768]) as similarity,
                       COALESCE(c.group_id, -1) as cluster_id
                FROM {self.embeddings_table} e
                JOIN {self.messages_table} m ON e.message_id = m.message_id AND e.chat_id = m.chat_id
                LEFT JOIN {self.clusters_table} c ON e.message_id = c.message_id AND e.chat_id = c.chat_id
                LEFT JOIN latest_chat_names lcn ON m.chat_id = lcn.chat_id
                LEFT JOIN latest_user_names lun ON m.from_id = lun.from_id
                WHERE array_cosine_similarity(e.embeddings, ?::FLOAT[768]) >= ?
                {where_clause}
                ORDER BY similarity DESC
                LIMIT ?
            """

            # Insert similarity_threshold before LIMIT param; order already correct
            if chat_id is not None:
                params = [query_vector, query_vector, similarity_threshold, chat_id, top_k]
            else:
                params = [query_vector, query_vector, similarity_threshold, top_k]

            results = self._execute_query(
                query, params, "performing similarity search with text"
            ).fetchall()

            if not results:
                return pl.DataFrame(schema=empty_schema)

            similarity_df = pl.DataFrame(
                results,
                schema=[
                    "message_id",
                    "chat_id",
                    "text",
                    "chat_name",
                    "from_name",
                    "date",
                    "similarity",
                    "cluster_id",
                ],
                orient="row",
            )

            if include_cluster_messages:
                text_previews = [
                    self._get_cluster_context_snippet(
                        row["message_id"], row["chat_id"], row["cluster_id"], row["text"]
                    )
                    for row in similarity_df.iter_rows(named=True)
                ]
                similarity_df = similarity_df.with_columns(pl.Series("text_preview", text_previews))
            else:
                similarity_df = similarity_df.with_columns(
                    pl.col("text").str.slice(0, 100).str.replace("\n", " ").alias("text_preview")
                )

            return similarity_df

        except Exception:
            return pl.DataFrame(schema=empty_schema)

    def _get_cluster_context_snippet(
        self,
        message_id: int,
        chat_id: int,
        cluster_id: int,
        original_text: str,
        context_size: int = 5,
    ) -> str:
        """Get a snippet of messages around the found message within the same cluster."""
        try:
            if cluster_id == -1:
                return (original_text or "")[:100].replace("\n", " ")

            cluster_messages = self._execute_query(
                f"""SELECT m.message_id, m.text, lun.from_name, m.date
                    FROM {self.messages_table} m
                    JOIN {self.clusters_table} c ON m.message_id = c.message_id AND m.chat_id = c.chat_id
                    LEFT JOIN (
                        SELECT from_id, from_name FROM (
                            SELECT from_id, from_name,
                                   ROW_NUMBER() OVER (PARTITION BY from_id ORDER BY COALESCE(last_seen, first_seen) DESC) AS rn
                            FROM {self.user_names_table}
                        ) t WHERE rn = 1
                    ) lun ON m.from_id = lun.from_id
                    WHERE c.group_id = ? AND m.chat_id = ?
                    ORDER BY m.date""",
                [cluster_id, chat_id],
                "getting cluster context",
            ).fetchall()

            if not cluster_messages:
                return (original_text or "")[:100].replace("\n", " ")

            target_position = next(
                (i for i, (msg_id, _, _, _) in enumerate(cluster_messages) if msg_id == message_id),
                None,
            )

            if target_position is None:
                return (original_text or "")[:100].replace("\n", " ")

            start_idx = max(0, target_position - context_size)
            end_idx = min(len(cluster_messages), target_position + context_size + 1)
            context_messages = cluster_messages[start_idx:end_idx]

            snippet_parts = []
            for msg_id, text, from_name, _ in context_messages:
                text_clean = (text or "").replace("\n", " ").strip()
                if len(text_clean) > 50:
                    text_clean = text_clean[:50] + "..."

                if msg_id == message_id:
                    snippet_parts.append(f">>> {from_name}: {text_clean}")
                else:
                    snippet_parts.append(f"{from_name}: {text_clean}")

            snippet = "\n".join(snippet_parts)
            if len(snippet) > 300:
                snippet = snippet[:300] + "..."

            return snippet + "\n"

        except Exception:
            return (original_text or "")[:100].replace("\n", " ")

    def get_table_stats(self) -> dict[str, Any]:
        """Get statistics about the embeddings table."""
        try:
            stats: dict[str, Any] = {}

            count_result = self._execute_query(
                f"SELECT COUNT(*) FROM {self.embeddings_table}", [], "getting table count"
            ).fetchone()
            stats["total_embeddings"] = count_result[0] if count_result else 0

            chats_result = self._execute_query(
                f"SELECT COUNT(DISTINCT chat_id) FROM {self.embeddings_table}",
                [],
                "getting unique chats",
            ).fetchone()
            stats["unique_chats"] = chats_result[0] if chats_result else 0

            chat_breakdown = self._execute_query(
                f"SELECT chat_id, COUNT(*) as message_count FROM {self.embeddings_table} GROUP BY chat_id ORDER BY message_count DESC",
                [],
                "getting chat breakdown",
            ).fetchall()
            stats["chat_breakdown"] = dict(chat_breakdown)

            return stats
        except Exception:
            return {}

    def close(self) -> None:
        """Close the database connection."""
        try:
            self.db.close()
        except Exception as e:
            self.logger.error(f"Error closing vector store connection: {str(e)}")
