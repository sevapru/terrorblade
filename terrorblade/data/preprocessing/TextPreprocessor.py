import os
from datetime import timedelta
from typing import Any

import polars as pl
import torch
from sentence_transformers import util

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


class TextPreprocessor:
    """
    Utilities for embedding text, temporal grouping, and conversation clustering.

    This class provides:
    - Embedding model management (lazily loaded SentenceTransformer)
    - Temporal clustering based on a time window
    - Semantic segmentation using sliding cosine distances
    - Group calculation for conversation threads

    Attributes:
        time_window (str): Default time window used for clustering (e.g., "5m", "1h").
        cluster_size (int): Minimum number of messages required to mark a cluster.
        batch_size (int): Batch size for embedding inference.
        squared_batch_size (int): Batch size for pairwise/square computations.
        device (str): "cuda" if available, otherwise "cpu".

    Tags:
        - preprocessing
        - embeddings
        - clustering
        - segmentation

    Examples:
        ```python
        tp = TextPreprocessor(time_window="10m", cluster_size=2)
        df = pl.DataFrame({
            "message_id": [1, 2],
            "chat_id": [10, 10],
            "from_id": [100, 100],
            "text": ["hello", "world"],
            "date": [pl.datetime(2024, 1, 1, 0, 0), pl.datetime(2024, 1, 1, 0, 1)],
        })
        out = tp.process_message_groups(df)
        ```
    """

    def __init__(
        self,
        time_window: str = "5m",
        cluster_size: int = 3,
        batch_size: int = 2000,
    ):
        """
        Initialize a `TextPreprocessor` with clustering and batching parameters.

        Args:
            time_window (str): Default temporal window for clustering (e.g., "5m", "1h").
            cluster_size (int): Minimum number of messages for a cluster.
            batch_size (int): Batch size for embedding inference.

        Tags:
            - preprocessing
            - embeddings
        """
        self.time_window = time_window
        self.cluster_size = cluster_size
        self._embeddings_model: Any | None = None
        self.batch_size = batch_size
        self.squared_batch_size = 1024
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Log GPU info if available
        if self.device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            print(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            print("💻 Using CPU for computations")

        # Adjust batch sizes based on device capabilities
        self._optimize_batch_sizes()

    @property
    def embeddings_model(self) -> Any:
        """
        Lazily instantiate and return the sentence-transformers model.

        Returns:
            Any: Loaded `SentenceTransformer` instance.

        Tags:
            - embeddings

        Examples:
            ```python
            model = tp.embeddings_model
            vec = model.encode(["hello"])
            ```
        """
        if self._embeddings_model is None:
            from sentence_transformers import SentenceTransformer

            self._embeddings_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
        return self._embeddings_model

    def concat_author_messages(self, df: pl.DataFrame, time_window_minutes: int = 5) -> pl.DataFrame:
        """
        Concatenate consecutive messages from the same author within a time window.

        The function groups runs of the same `from_id` where the time gap between adjacent
        messages is less than or equal to `time_window_minutes`, and joins `text` with ". ".

        Args:
            df (pl.DataFrame): DataFrame with columns `from_id`, `date`, `text`, `chat_id`, etc.
            time_window_minutes (int): Time window in minutes for concatenating consecutive messages.

        Returns:
            pl.DataFrame: New DataFrame with messages concatenated per run. Includes first/earliest
            fields for each run and drops the temporary grouping column.

        Tags:
            - preprocessing
            - grouping

        Examples:
            ```python
            out = tp.concat_author_messages(df, time_window_minutes=3)
            ```
        """
        time_window = timedelta(minutes=time_window_minutes)
        df = df.with_columns(
            [
                (pl.col("from_id") != pl.col("from_id").shift(1)).alias("author_changed"),
                ((pl.col("date") - pl.col("date").shift(1)) > time_window).fill_null(True).alias("time_exceeded"),
            ]
        )

        df = df.with_columns((pl.col("author_changed") | pl.col("time_exceeded")).alias("new_group"))

        df = df.with_columns(pl.col("new_group").cum_sum().alias("message_group"))

        df = df.group_by("message_group").agg(
            [
                pl.col("chat_name").first(),
                pl.col("date").min().alias("date"),
                pl.col("from_name").first().alias("from_name"),
                pl.col("text").str.join(". ").alias("text"),
                pl.col("reply_to_message_id").first().alias("reply_to_message_id"),
                pl.col("forwarded_from").first().alias("forwarded_from"),
                pl.col("message_id").alias("message_id"),
                pl.col("from_id").first().alias("from_id"),
                pl.col("chat_id").first().alias("chat_id"),
            ]
        )
        df = df.sort("date")
        df = df.drop("message_group")

        return df

    def create_clusters(
        self,
        df: pl.DataFrame,
        time_window: str | None = None,
        cluster_size: int = 3,
    ) -> pl.DataFrame:
        """
        Assign temporal clusters based on gaps larger than a time window.

        Args:
            df (pl.DataFrame): DataFrame containing at least a `date` column.
            time_window (str | None): Window string like "5m" or "1h". Defaults to `self.time_window`.
            cluster_size (int): Minimum number of messages for a cluster.

        Returns:
            pl.DataFrame: DataFrame augmented with `pre_cluster` and `cluster` (sized).

        Raises:
            ValueError: If `time_window` has an invalid format.

        Tags:
            - clustering
            - preprocessing

        Examples:
            ```python
            out = tp.create_clusters(df, time_window="10m", cluster_size=2)
            ```
        """
        # Convert the time window string to a timedelta object
        if time_window is None:
            time_window = self.time_window
        if time_window.endswith("h"):
            window_duration = timedelta(hours=int(time_window[:-1]))
        elif time_window.endswith("m"):
            window_duration = timedelta(minutes=int(time_window[:-1]))
        else:
            raise ValueError("time_window must be formatted as '1h' or '5m'.")
        time_diffs = df.with_columns((pl.col("date").diff().dt.total_seconds()).alias("time_diff"))

        breaks = (
            time_diffs["time_diff"].fill_null(window_duration.total_seconds() * 2) > window_duration.total_seconds()
        )
        cluster_ids = breaks.cum_sum().alias("pre_cluster")
        df = df.with_columns(cluster_ids)
        cluster_sizes = df.group_by("pre_cluster").agg(pl.len().alias("size"))
        df = df.join(cluster_sizes, on="pre_cluster").with_columns(
            pl.when(pl.col("size") > cluster_size)
            .then(pl.col("pre_cluster"))
            .otherwise(None)
            .alias("cluster"),  # Only qualify as a cluster if it contains more than 'cluster_size' messages
        )
        return df.drop(["size"])

    def calculate_distances(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise cosine-distance matrix for a batch of embeddings.

        Args:
            embeddings (torch.Tensor): 2D tensor of normalized embeddings (N x 768).

        Returns:
            torch.Tensor: Matrix of pairwise distances (N x N) on CPU.

        Tags:
            - embeddings
            - distance

        Examples:
            ```python
            emb = tp.embeddings_model.encode(["a", "b"], convert_to_tensor=True, normalize_embeddings=True)
            dist = tp.calculate_distances(emb)
            ```
        """

        embeddings = embeddings.to(self.device)
        similarities = torch.zeros((len(embeddings), len(embeddings)), device=self.device)

        for i in range(0, len(embeddings), self.squared_batch_size):
            end = min(i + self.squared_batch_size, len(embeddings))
            batch = embeddings[i:end]
            similarities[i:end] = util.cos_sim(batch, embeddings)

        distances = 1 - similarities
        return distances.cpu()

    def calculate_sliding_distances(self, embeddings: torch.Tensor, window_size: int = 5) -> torch.Tensor:
        """
        Compute average cosine distance within a sliding window for each embedding.

        For each index i, compare the i-th embedding to embeddings in the range
        `[i - window_size, i]` and compute `1 - mean(cos_sim)`. The first element is 0.

        Args:
            embeddings (torch.Tensor): 2D tensor of normalized embeddings (N x 768).
            window_size (int): Number of previous elements to consider in the window.

        Returns:
            torch.Tensor: 1D tensor of distances, length N, on CPU.

        Tags:
            - embeddings
            - distance
            - segmentation

        Examples:
            ```python
            emb = tp.embeddings_model.encode(["a", "b", "c"], convert_to_tensor=True, normalize_embeddings=True)
            d = tp.calculate_sliding_distances(emb, window_size=1)
            ```
        """
        embeddings = embeddings.to(self.device)
        if embeddings.shape[0] == 0:
            return torch.zeros(0, device=self.device)

        distances = torch.zeros(len(embeddings), device=self.device)

        gpu_batch_size = (
            min(self.squared_batch_size * 4, len(embeddings)) if self.device == "cuda" else self.squared_batch_size
        )

        for i in range(0, len(embeddings), gpu_batch_size):
            end = min(i + gpu_batch_size, len(embeddings))
            batch_indices = torch.arange(i, end, device=self.device)

            # Vectorized computation for better GPU utilization
            for idx in batch_indices:
                if idx == 0:  # Special case for first element
                    distances[idx] = 0
                    continue

                start = max(0, int(idx) - window_size)
                end_window = idx + 1

                main_embedding = embeddings[idx].unsqueeze(0)
                window_embeddings = embeddings[start:end_window]

                similarities = util.cos_sim(main_embedding, window_embeddings)
                distances[idx] = 1 - similarities.mean()

        return distances.cpu()

    def process_message_groups(
        self,
        df: pl.DataFrame,
        time_window: str | None = None,
        cluster_size: int = 1,
    ) -> pl.DataFrame:
        """
        Full pipeline: compute embeddings, temporal clusters, semantic segments, and group IDs.

        Args:
            df (pl.DataFrame): Messages DataFrame; must include `text` and `date`.
            time_window (str | None): Time window for clustering; defaults to class setting.
            cluster_size (int): Minimum cluster size.

        Returns:
            pl.DataFrame: Input DataFrame augmented with `embeddings`, cluster IDs, segments, and `group_id`.

        Tags:
            - pipeline
            - embeddings
            - clustering
            - segmentation

        Examples:
            ```python
            out = tp.process_message_groups(df, time_window="5m")
            ```
        """
        if len(df) == 0:
            return df

        if len(df) < self.batch_size:
            result = self._process_message_batch(df, time_window, cluster_size)
            return result

        processed_dfs = []

        for i in range(0, len(df), self.batch_size):
            batch_df = df.slice(i, min(self.batch_size, len(df) - i))

            batch_df = self.calculate_embeddings(batch_df)
            batch_df = self.calculate_segments(batch_df)

            processed_dfs.append(batch_df)

        # Concatenate all batches and calculate groups globally to ensure continuity
        combined_df = pl.concat(processed_dfs)

        # Now process temporal clusters and groups globally to ensure proper continuity
        combined_df = self.create_clusters(combined_df, time_window, cluster_size)
        final_result = self.calculate_groups(combined_df)

        return final_result

    def _process_message_batch(
        self,
        df: pl.DataFrame,
        time_window: str | None = None,
        cluster_size: int = 1,
    ) -> pl.DataFrame:
        """
        Process a batch of messages: embeddings → clusters → segments → groups.

        Args:
            df (pl.DataFrame): Batch DataFrame.
            time_window (str | None): Temporal window for clustering.
            cluster_size (int): Minimum cluster size.

        Returns:
            pl.DataFrame: Augmented DataFrame with processing results.

        Tags:
            - pipeline
            - embeddings
            - clustering
            - segmentation
        """
        # df = self.concat_author_messages(df)
        df = self.calculate_embeddings(df)

        df = self.create_clusters(df, time_window, cluster_size)
        df = self.calculate_segments(df)
        df = self.calculate_groups(df)
        return df

    def calculate_embeddings(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Encode message `text` into 768-d float embeddings and attach as a column.

        Args:
            df (pl.DataFrame): Must contain a `text` column; nulls are treated as empty strings.

        Returns:
            pl.DataFrame: DataFrame with `embeddings` column of dtype `Array(Float32, shape=768)`.

        Raises:
            ValueError: If the model returns a non-768-dimensional embedding.

        Tags:
            - embeddings

        Examples:
            ```python
            out = tp.calculate_embeddings(df)
            assert out.schema["embeddings"].inner == pl.Float32
            ```
        """
        texts = df["text"].fill_null("").to_list()

        embeddings = self.embeddings_model.encode(
            texts,
            batch_size=min(self.batch_size, len(texts)),
            show_progress_bar=True,
            convert_to_tensor=True,
            normalize_embeddings=True,
            device=self.device,
        )

        embeddings_f32 = embeddings.cpu().float()

        if embeddings_f32.shape[1] != 768:
            raise ValueError(f"Expected embeddings dimension 768, got {embeddings_f32.shape[1]}")

        return df.with_columns(pl.Series("embeddings", embeddings_f32))

    def calculate_groups(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Merge adjacent messages into `group_id`s based on semantic and temporal continuity.

        The group boundary is introduced when either `semantic_segment` changes or the
        `pre_cluster` changes.

        Args:
            df (pl.DataFrame): DataFrame with `semantic_segment` and `pre_cluster` columns.

        Returns:
            pl.DataFrame: DataFrame with `group_id` added and sort by `date`.

        Tags:
            - clustering
            - segmentation

        Examples:
            ```python
            out = tp.calculate_groups(df)
            ```
        """
        # Fixed: Use OR instead of AND - a group boundary occurs when EITHER changes
        group_changes = (df["semantic_segment"] != df["semantic_segment"].shift(1)) | (
            df["pre_cluster"] != df["pre_cluster"].shift(1)
        )
        df = df.with_columns(group_changes.cum_sum().alias("group_id")).drop(
            ["pre_cluster", "cluster", "semantic_segment"]
        )
        df[0, "group_id"] = 0
        return df.sort("date")

    def calculate_segments(
        self,
        df: pl.DataFrame,
        semantic_threshold: float | None = 0.7,
    ) -> pl.DataFrame:
        """
        Compute `semantic_segment` indices from sliding-window distances.

        Args:
            df (pl.DataFrame): DataFrame with an `embeddings` column.
            semantic_threshold (float | None): If None, uses the mean distance; otherwise
                marks a break when `distance > semantic_threshold`.

        Returns:
            pl.DataFrame: DataFrame with `semantic_segment` column added.

        Tags:
            - segmentation
            - embeddings

        Examples:
            ```python
            out = tp.calculate_segments(df)
            ```
        """
        # Convert to tensor and move to GPU immediately for better performance
        embeddings_tensor = torch.tensor(df["embeddings"], device=self.device)

        distances = self.calculate_sliding_distances(embeddings_tensor, 1)

        if semantic_threshold is None:
            semantic_threshold = 0.7  # float(torch.mean(distances))

        breaks = distances > semantic_threshold
        semantic_segment_ids = breaks.cumsum(dim=0)

        return df.with_columns(
            pl.Series("semantic_segment", semantic_segment_ids.cpu().numpy()).fill_null(strategy="forward")
        )

    ####
    # Visualisation #
    ####
    def show_cluster(self, df: pl.DataFrame, cluster_id: int | None) -> None:
        """
        Print all messages belonging to a specific cluster.

        Args:
            df (pl.DataFrame): Messages DataFrame containing a `cluster` column.
            cluster_id (int | None): Target cluster id.

        Tags:
            - visualization
            - clustering

        Examples:
            ```python
            tp.show_cluster(df, 1)
            ```
        """
        cluster_df = df.filter(pl.col("cluster") == cluster_id)
        print(cluster_df)

    def show_all_clusters(self, df: pl.DataFrame) -> None:
        """
        Print all clusters sequentially.

        Args:
            df (pl.DataFrame): Messages DataFrame containing a `cluster` column.

        Tags:
            - visualization
            - clustering
        """
        for cluster_id in df["cluster"].unique().to_list():
            self.show_cluster(df, cluster_id)

    def _optimize_batch_sizes(self) -> None:
        """
        Optimize batch sizes based on available GPU memory.

        Larger batches for GPUs with more memory to maximize throughput.
        """
        if self.device == "cuda":
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3

            if gpu_memory_gb >= 24:  # High-end GPU (RTX 4090, A100, etc.)
                self.squared_batch_size = 4096
                suggested_batch_size = 8000
            elif gpu_memory_gb >= 12:  # Mid-range GPU (RTX 3080, 4070, etc.)
                self.squared_batch_size = 2048
                suggested_batch_size = 4000
            elif gpu_memory_gb >= 8:
                self.squared_batch_size = 1024
                suggested_batch_size = 2000
            else:
                self.squared_batch_size = 512
                suggested_batch_size = 1000

            if suggested_batch_size > self.batch_size:
                print(
                    f"💡 GPU has {gpu_memory_gb:.1f}GB memory - consider increasing batch_size to {suggested_batch_size} for better performance"
                )
        else:
            self.squared_batch_size = 256
            print(f"📊 Using batch_size={self.batch_size}, squared_batch_size={self.squared_batch_size} for CPU")
