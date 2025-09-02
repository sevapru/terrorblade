"""Simplified vector search example for keyword-based semantic search."""

import argparse

import polars as pl

from terrorblade.data.database.vector_store import VectorStore
from terrorblade.data.preprocessing.TextPreprocessor import TextPreprocessor

# Configure Polars for full text display
pl.Config.set_tbl_width_chars(1200)
pl.Config.set_fmt_str_lengths(1000)
pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_rows(100)


class VectorSearcher:
    """Simplified vector searcher for keyword-based queries."""

    def __init__(self, db_path: str, phone: str):
        """Initialize searcher with database and phone."""
        self.db_path = db_path
        self.phone = phone
        self._preprocessor: TextPreprocessor = TextPreprocessor()
        self._vector_store: VectorStore = VectorStore(db_path=self.db_path, phone=self.phone)

    def _setup(self) -> None:
        """Setup preprocessor and vector store."""
        self._vector_store.create_hnsw_index()

    def _cleanup(self) -> None:
        """Cleanup resources."""
        if self._vector_store:
            self._vector_store.close()

    def _encode_query(self, query_text: str) -> list[float]:
        """Convert query text to embedding vector."""
        query_embedding = self._preprocessor.embeddings_model.encode(
            [query_text],
            convert_to_tensor=True,
            normalize_embeddings=True,
            device=self._preprocessor.device,
        )
        return query_embedding.cpu().tolist()[0]

    def _format_results(self, df: pl.DataFrame) -> pl.DataFrame:
        """Format search results for display."""
        if len(df) == 0:
            return df

        return df.with_columns(
            [
                pl.col("similarity").round(4).alias("similarity"),
                pl.col("from_name").fill_null("Unknown").alias("from_name"),
                pl.col("chat_name").fill_null("N/A").alias("chat_name"),
                pl.when(pl.col("cluster_id") == -1)
                .then(pl.lit("No cluster"))
                .otherwise(pl.col("cluster_id").cast(pl.Utf8))
                .alias("cluster"),
                pl.col("text_preview").alias("context_snippet"),
            ]
        ).select(
            [
                "message_id",
                "chat_id",
                "similarity",
                "cluster",
                "from_name",
                "chat_name",
                "date",
                "context_snippet",
            ]
        )

    def search(self, keywords: str | list[str], top_k: int = 10) -> None:
        """
        Search for messages containing keywords.

        Args:
            keywords: Single keyword string or list of keywords
            top_k: Number of results to return per query
        """
        try:
            self._setup()
            if isinstance(keywords, str):
                keywords = [keywords]

            stats = self._vector_store.get_table_stats()
            print(f"Database: {stats.get('total_embeddings', 0)} embeddings, {stats.get('unique_chats', 0)} chats")
            self._vector_store.print_index_stats()

            for keyword in keywords:
                print(f"\n{'=' * 60}")
                print(f"Keyword: '{keyword}'")
                print("=" * 60)

                query_vector = self._encode_query(keyword)
                results_df = self._vector_store.get_similar_messages_with_text(
                    query_vector=query_vector, top_k=top_k, similarity_threshold=0.3
                )

                if len(results_df) > 0:
                    display_df = self._format_results(results_df)
                    with pl.Config(tbl_width_chars=1200, fmt_str_lengths=1000, tbl_cols=-1, tbl_rows=100):
                        print(display_df)
                else:
                    print("No results found.")

        finally:
            self._cleanup()


def main() -> None:
    """Main entry point for keyword search."""
    parser = argparse.ArgumentParser(description="Search messages by keywords")
    parser.add_argument("keywords", nargs="+", help="Keywords to search for")
    parser.add_argument("--db", required=True, help="Database path")
    parser.add_argument("--phone", required=True, help="Phone number")
    parser.add_argument("--top-k", type=int, default=10, help="Results per keyword (default: 10)")

    args = parser.parse_args()
    searcher = VectorSearcher(args.db, args.phone)
    searcher.search(args.keywords, args.top_k)


if __name__ == "__main__":
    main()
