import os
from datetime import timedelta

import polars as pl
import torch
from sentence_transformers import util

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


class TextPreprocessor:
    """
    Preprocesses text data by removing special characters and converting to lowercase.
    """

    def __init__(
        self,
        time_window: str = "5m",
        cluster_size: int = 3,
        big_cluster_size: int = 10,
        batch_size: int = 2000,
    ):
        self.time_window = time_window
        self.cluster_size = cluster_size
        self.big_cluster_size = big_cluster_size
        self._embeddings_model = None
        self.batch_size = batch_size
        self.squared_batch_size = 1024
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def embeddings_model(self):
        if self._embeddings_model is None:
            from sentence_transformers import SentenceTransformer

            self._embeddings_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
        return self._embeddings_model

    def concat_author_messages(
        self, df: pl.DataFrame, time_window_minutes: int = 5
    ) -> pl.DataFrame:
        """
        Concatenates multiple messages from the same author if they are consecutive and within a specified time window.

        Args:
            df (pl.DataFrame): DataFrame containing the chat messages.
            time_window_minutes (int): Time window in minutes.

        Returns:
            pl.DataFrame: DataFrame with concatenated messages.
        """
        time_window = timedelta(minutes=time_window_minutes)
        df = df.with_columns(
            [
                (pl.col("from_id") != pl.col("from_id").shift(1)).alias("author_changed"),
                ((pl.col("date") - pl.col("date").shift(1)) > time_window)
                .fill_null(True)
                .alias("time_exceeded"),
            ]
        )

        df = df.with_columns(
            (pl.col("author_changed") | pl.col("time_exceeded")).alias("new_group")
        )

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
        cluster_size: int = 1,
        big_cluster_size: int = 10,
    ) -> pl.DataFrame:
        """
        Creates cluster assignments for messages that are considered part of the same conversation if they are
        sent within a specified time window proximity.

        Args:
            df (pl.DataFrame): DataFrame containing a 'timestamp' column.
            time_window (str): Time window for determining clusters, formatted as "1h" (for one hour) or "5m" (for five minutes).
            cluster_size (int): Minimum number of messages required to consider it a cluster.
            big_cluster_size (int): Minimum number of messages required to consider it a big cluster.

        Returns:
            pl.DataFrame: DataFrame augmented with a 'cluster' column indicating normal clusters and a 'big_cluster'
            column indicating big clusters.
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
            time_diffs["time_diff"].fill_null(window_duration.total_seconds() * 2)
            > window_duration.total_seconds()
        )
        cluster_ids = breaks.cum_sum().alias("pre_cluster")
        df = df.with_columns(cluster_ids)
        cluster_sizes = df.group_by("pre_cluster").agg(pl.len().alias("size"))
        df = df.join(cluster_sizes, on="pre_cluster").with_columns(
            pl.when(pl.col("size") > cluster_size)
            .then(pl.col("pre_cluster"))
            .otherwise(None)
            .alias(
                "cluster"
            ),  # Only qualify as a cluster if it contains more than 'cluster_size' messages
            pl.when(pl.col("size") > big_cluster_size)
            .then(True)
            .otherwise(False)
            .alias("is_big_cluster"),  # Mark big clusters
        )
        df = df.with_columns(
            pl.when(
                pl.col("is_big_cluster")
                & (pl.col("is_big_cluster").shift(1).is_null() | ~pl.col("is_big_cluster").shift(1))
            )
            .then(1)
            .otherwise(0)
            .cum_sum()
            .alias("big_cluster")
        )
        df = df.with_columns(
            pl.when(pl.col("is_big_cluster"))
            .then(pl.col("big_cluster"))
            .otherwise(None)  # Assign None to messages not in a big cluster
            .alias("big_cluster")
        )
        return df.drop(["is_big_cluster", "big_cluster", "size"])

    def calculate_distances(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Calculates distances based on cosine pair similarities between embeddings.

        Args:
            embeddings (torch.Tensor): Embeddings for the chat messages.

        Returns:
            torch.Tensor: Matrix of pairwise distances between embeddings.
        """

        embeddings = embeddings.to(self.device)
        similarities = torch.zeros((len(embeddings), len(embeddings)), device=self.device)

        for i in range(0, len(embeddings), self.squared_batch_size):
            end = min(i + self.squared_batch_size, len(embeddings))
            batch = embeddings[i:end]
            similarities[i:end] = util.cos_sim(batch, embeddings)

        distances = 1 - similarities
        return distances.cpu()

    def calculate_sliding_distances(
        self, embeddings: torch.Tensor, window_size: int = 5
    ) -> torch.Tensor:
        """
        Calculates sliding window distances for embeddings.

        Args:
        embeddings (torch.Tensor): Input embeddings
        window_size (int): Size of the sliding window

        Returns:
        torch.Tensor: Distances tensor
        """
        embeddings = embeddings.to(self.device)
        if embeddings.shape[0] == 0:
            return torch.zeros(0)

        distances = torch.zeros(len(embeddings), device=self.device)

        for i in range(0, len(embeddings), self.squared_batch_size):
            end = min(i + self.squared_batch_size, len(embeddings))
            batch_indices = torch.arange(i, end, device=self.device)

            starts = torch.clamp(batch_indices - window_size, min=0)
            ends = torch.clamp(batch_indices + 1, max=len(embeddings))

            for _j, (start, end, idx) in enumerate(zip(starts, ends, batch_indices, strict=False)):
                if idx == 0:  # Special case for first element
                    distances[idx] = 0
                    continue

                main_embedding = embeddings[idx].unsqueeze(0)
                window_embeddings = embeddings[start:end]
                similarities = util.cos_sim(main_embedding, window_embeddings)
                distances[idx] = 1 - similarities.mean()

        return distances.cpu()

    def process_message_groups(
        self,
        df: pl.DataFrame,
        time_window: str | None = None,
        cluster_size: int = 1,
        big_cluster_size: int = 10,
    ) -> pl.DataFrame:
        """
        Processes the provided DataFrame by creating clusters and calculating embeddings.

        Args:
            df (pl.DataFrame): DataFrame containing the chat messages.
            time_window (Optional[str]): Time window for clustering, e.g. "5m" or "1h". Defaults to None.
            cluster_size (int): Minimum size for a cluster. Defaults to 1.
            big_cluster_size (int): Minimum size for a big cluster. Defaults to 10.

        Returns:
            pl.DataFrame: DataFrame with clusters and embeddings added.
        """
        if len(df) == 0:
            return df

        if len(df) < self.batch_size:
            result = self._process_message_batch(df, time_window, cluster_size, big_cluster_size)
            return result

        processed_dfs = []
        for i in range(0, len(df), self.batch_size):
            batch_df = df.slice(i, min(self.batch_size, len(df) - i))
            processed_batch = self._process_message_batch(
                batch_df, time_window, cluster_size, big_cluster_size
            )
            processed_dfs.append(processed_batch)

        return pl.concat(processed_dfs)

    def _process_message_batch(
        self,
        df: pl.DataFrame,
        time_window: str | None = None,
        cluster_size: int = 1,
        big_cluster_size: int = 10,
    ) -> pl.DataFrame:
        """
        Process a batch of messages.
        """
        # df = self.concat_author_messages(df)
        df = self.calculate_embeddings(df)

        df = self.create_clusters(df, time_window, cluster_size, big_cluster_size)
        df = self.calculate_segments(df)
        df = self.calculate_groups(df)
        return df

    def calculate_embeddings(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculates embeddings for the provided texts using the provided model.

        Args:
            df (pl.DataFrame): DataFrame containing the chat messages.

        Returns:
            torch.Tensor: Tensor containing the embeddings for each text.
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
        return df.with_columns(pl.Series("embeddings", embeddings.cpu()))

    def calculate_groups(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Merges the segments with same semantic_segment number or either from the same cluster

        Args:
            df (pl.DataFrame): DataFrame containing the segments.

        Returns:
            pl.DataFrame: DataFrame with segments merged back in.
        """
        group_changes = (df["semantic_segment"] != df["semantic_segment"].shift(1)) & (
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
        semantic_threshold: float | None = None,
    ) -> pl.DataFrame:
        """
        Calculates the segments based on the provided distances.

        Args:
            df (pl.DataFrame): DataFrame containing the chat messages.

        Returns:
            pl.DataFrame: DataFrame with semantic_segment column added.
        """
        distances = self.calculate_sliding_distances(torch.Tensor(df["embeddings"]), 1)

        if semantic_threshold is None:
            semantic_threshold = float(torch.mean(distances))

        breaks = distances > semantic_threshold
        semantic_segment_ids = breaks.cumsum(dim=0)

        return df.with_columns(
            pl.Series("semantic_segment", semantic_segment_ids).fill_null(strategy="forward")
        )

    ####
    # Visualisation #
    ####
    def show_cluster(self, df: pl.DataFrame, cluster_id: int | None) -> None:
        """
        Displays the messages in the provided cluster.

        Args:
            df (pl.DataFrame): DataFrame containing the chat messages.
            cluster_id (int): ID of the cluster to display.
        """
        cluster_df = df.filter(pl.col("cluster") == cluster_id)
        print(cluster_df)

    def show_biggest_cluster(self, df: pl.DataFrame):
        """
        Displays the messages in the biggest cluster.

        Args:
            df (pl.DataFrame): DataFrame containing the chat messages.
        """
        biggest_cluster_id = (
            df.group_by("big_cluster")
            .agg(pl.len().alias("size"))
            .sort("size")["big_cluster"]
            .first()
        )
        self.show_cluster(df, biggest_cluster_id)  # type: ignore

    def show_all_clusters(self, df: pl.DataFrame) -> None:
        """
        Displays all clusters in the provided DataFrame.

        Args:
            df (pl.DataFrame): DataFrame containing the chat messages.
        """
        for cluster_id in df["cluster"].unique().to_list():
            self.show_cluster(df, cluster_id)
