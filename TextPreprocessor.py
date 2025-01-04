import polars as pl

from datetime import timedelta
import numpy as np
from sentence_transformers import SentenceTransformer, util
import cupy as cp
from cuml.cluster import HDBSCAN
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import torch
import seaborn as sns

class TextPreprocessor:
    """
    Preprocesses text data by removing special characters and converting to lowercase.
    """
    def __init__(self, time_window, cluster_size, big_cluster_size):
        self.embeddings_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
        self.time_window = time_window
        self.cluster_size = cluster_size
        self.big_cluster_size = big_cluster_size
        
        self.embeddings=None
        
    def concat_author_messages(self, df, time_window_minutes=5):
        """
        Concatenates multiple messages from the same author if they are consecutive and within a specified time window.

        Args:
            df (pl.DataFrame): DataFrame containing the chat messages.
            time_window_minutes (int): Time window in minutes.

        Returns:
            pl.DataFrame: DataFrame with concatenated messages.
        """
        time_window = timedelta(minutes=time_window_minutes)
        df = df.with_columns([
            (pl.col('from_id') != pl.col('from_id').shift(1)).alias('author_changed'),
            ((pl.col('date') - pl.col('date').shift(1)) > time_window).fill_null(True).alias('time_exceeded'),
        ])

        df = df.with_columns(
            (pl.col('author_changed') | pl.col('time_exceeded')).alias('new_group')
        )

        df = df.with_columns(
            pl.col('new_group').cum_sum().alias('message_group')
        )

        df = df.group_by('message_group').agg([
            pl.col('chat_name').first(),
            pl.col('date').min().alias('date'),
            pl.col('from').first().alias('from'),
            pl.col('text').str.join('. ').alias('text'),
            pl.col('reply_to_message_id').first().alias('reply_to_message_id'),
            pl.col('forwarded_from').first().alias('forwarded_from'),
            pl.col('id').alias('id'),
            pl.col('from_id').first().alias('from_id'),
            pl.col('chat_id').first().alias('chat_id'),
        ])
        df = df.sort('date')
        df = df.drop('message_group')

        return df
        
    def create_clusters(self, df, time_window=None, cluster_size=1, big_cluster_size=10):
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
        if time_window.endswith('h'):
            window_duration = timedelta(hours=int(time_window[:-1]))
        elif time_window.endswith('m'):
            window_duration = timedelta(minutes=int(time_window[:-1]))
        else:
            raise ValueError("time_window must be formatted as '1h' or '5m'.")

        # Create a column for time differences between consecutive messages
        time_diffs = df.with_columns(
            (pl.col("date").diff().dt.total_seconds()).alias("time_diff")
        )
        
        # Determine where new clusters should start based on time differences exceeding the window duration
        breaks = time_diffs['time_diff'].fill_null(window_duration.total_seconds() * 2) > window_duration.total_seconds()
        cluster_ids = breaks.cum_sum().alias("pre_cluster")
        df = df.with_columns(cluster_ids)
        
        # Calculate sizes of these pre-clusters
        cluster_sizes = df.group_by("pre_cluster").agg(pl.len().alias("size"))

        # Join sizes back to the main frame and determine qualifying clusters
        df = df.join(cluster_sizes, on="pre_cluster").with_columns(
            pl.when(pl.col("size") > cluster_size)
            .then(pl.col("pre_cluster"))
            .otherwise(None)
            .alias("cluster"),  # Only qualify as a cluster if it contains more than 'cluster_size' messages
            pl.when(pl.col("size") > big_cluster_size)
            .then(True)
            .otherwise(False)
            .alias("is_big_cluster")  # Mark big clusters
        )
        
        # Assign unique incremental indices to big clusters
        df = df.with_columns(
            pl.when(pl.col("is_big_cluster") & (pl.col("is_big_cluster").shift(1).is_null() | ~pl.col("is_big_cluster").shift(1)))
            .then(1)
            .otherwise(0)
            .cum_sum()
            .alias("big_cluster")  
        )
        
        # Ensure big cluster numbers are only assigned to big clusters
        df = df.with_columns(
            pl.when(pl.col("is_big_cluster"))
            .then(pl.col("big_cluster"))
            .otherwise(None)  # Assign None to messages not in a big cluster
            .alias("big_cluster")
        )
        
        # Cleaning up: drop temporary columns
        return df.drop(["is_big_cluster", "big_cluster", "size"])
    
    def find_simmilar_messages_in_chat(self, df, embeddings):
        """
        Finds similar messages in a chat based on the provided embeddings.
            
        Args:
        df (pl.DataFrame): DataFrame containing the chat messages.
        embeddings (torch.Tensor): Embeddings for the chat messages.
        """
        embeddings_cupy = cp.asarray(embeddings.cpu().numpy())
        
        clustering = HDBSCAN(
                min_cluster_size=5,
                metric='euclidean',
                cluster_selection_epsilon=0.5
            )
        cluster_labels = clustering.fit_predict(embeddings_cupy)
        df = df.with_columns(pl.Series('cluster_label', cp.asnumpy(cluster_labels)))
        df_split = df.group_by('cluster_label').agg([
                pl.col('author').first().alias('author'),
                pl.col('recipients').first().alias('author2'),
                pl.col('timestamp').min().alias('timestamp_start'),
                pl.col('timestamp').max().alias('timestamp_end'),
                pl.concat_str('text', separator='. ').alias('text')
            ])
    
        return df_split
    
    def calculate_embeddings(self, df):
        """
        Calculates embeddings for the provided texts using the provided model.
            
        Args:
        model (SentenceTransformer): SentenceTransformer model to use for embedding calculation.
        df (pl.DataFrame): DataFrame containing the chat messages.
        """
        return self.embeddings_model.encode(
            df['text'].to_list(),
            batch_size=1024,
            show_progress_bar=True,
            convert_to_tensor=True,
            normalize_embeddings=True,
            device='cuda'
        )
        
    def calculate_distances(self, embeddings):
        """
        Calculates distances based on cosine pair similarities between embeddings.
            
        Args:
        embeddings (torch.Tensor): Embeddings for the chat messages.
        """
        similarities = self.embeddings_model.similarity_pairwise(embeddings, embeddings)
        distances = 1 - similarities
        return distances.cpu()
    
    def calculate_sliding_distances(self, embeddings, window_size=5):
        distances = torch.zeros(len(embeddings), device=embeddings.device)
        for i in range(len(embeddings)):
            start = max(0, i - window_size)
            end = min(len(embeddings), i)
            main_embedding = embeddings[i].unsqueeze(0)
            window_embeddings = embeddings[start:end]
            similarities = util.cos_sim(main_embedding, window_embeddings)
            avg_similarity = similarities.mean()
            distances[i] = 1 - avg_similarity
            
        # What a great hotfix, happy new year 2025, my lads
        distances[0] = 0

        return distances.cpu()
            
    
    def calculate_groups(self, df):
        """
        Merges the segments with same semantic_segment number or either from the same cluster
        
        Args:
            df_split (pl.DataFrame): DataFrame containing the segments.
        
        Returns:
            pl.DataFrame: DataFrame with segments merged back in.
        """
        group_changes = (
            (df['semantic_segment'] != df['semantic_segment'].shift(1)) &
            (df['pre_cluster'] != df['pre_cluster'].shift(1))
        )
        df = df.with_columns(
            group_changes.cum_sum().alias('group')
        ).drop(["pre_cluster", "cluster", "semantic_segment"])
        df[0, 'group'] = 0
        return df.sort('date')
    
    def process_message_groups(self, df, time_window=None, cluster_size=1, big_cluster_size=10):
        """
        Processes the provided DataFrame by creating clusters and calculating embeddings.
        
        Args:
            df (pl.DataFrame): DataFrame containing the chat messages.
        
        Returns:
            pl.DataFrame: DataFrame with clusters and embeddings added.
        """
        df = self.concat_author_messages(df)
        df = self.create_clusters(df, time_window, cluster_size, big_cluster_size)
        self.embeddings = self.calculate_embeddings(df)
        df = self.calculate_segments(df, self.embeddings)
        df = self.calculate_groups(df)
        return df
    
    def show_cluster(self, df, cluster_id):
        """
        Displays the messages in the provided cluster.
        
        Args:
            df (pl.DataFrame): DataFrame containing the chat messages.
            cluster_id (int): ID of the cluster to display.
        """
        cluster_df = df.filter(pl.col("cluster") == cluster_id)
        print(cluster_df)
        
    def show_biggest_cluster(self, df):
        """
        Displays the messages in the biggest cluster.
        
        Args:
            df (pl.DataFrame): DataFrame containing the chat messages.
        """
        biggest_cluster_id = df.group_by("big_cluster").agg(pl.len().alias("size")).sort("size")["big_cluster"].first()
        self.show_cluster(df, biggest_cluster_id)
        
    def show_all_clusters(self, df):
        """
        Displays all clusters in the provided DataFrame.
        
        Args:
            df (pl.DataFrame): DataFrame containing the chat messages.
        """
        for cluster_id in df['cluster'].unique().to_list():
            self.show_cluster(df, cluster_id)
            

    def calculate_segments(self, df, embeddings, breakpoint_percentile_threshold=95, semantic_threshold=None):
        """
        Calculates the segments based on the provided distances.
        
        Args:
            df (pl.DataFrame): DataFrame containing the chat messages.
            distances (np.ndarray): Array of distances between the chat messages.
        
        Returns:
            list: List of dictionaries containing the segments.
        """
        distances = self.calculate_sliding_distances(embeddings, 1)

        
        if semantic_threshold is None:
            semantic_threshold = float(np.mean(distances.tolist()))
        
        breaks = distances > semantic_threshold
        semantic_segment_ids = breaks.cumsum(dim=0)
        
        return df.with_columns(
            pl.Series("semantic_segment", 
                      semantic_segment_ids.tolist()).fill_null(strategy="forward")
)
        
    
    def display_dialog_chunks(self, embeddings=None):
        if embeddings is None:
            embeddings = self.embeddings

        distances = self.calculate_sliding_distances(embeddings)

        # Initial threshold value
        init_threshold = 99  # Starting at 99%

        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 6))
        plt.subplots_adjust(bottom=0.25)
        y_upper_bound = 0.2

        # Plot function
        def plot_chunks(breakpoint_percentile_threshold):
            ax.clear()
            sns.lineplot(x=range(len(distances)), y=distances, label='Cosine Distance', ax=ax)
            ax.set_ylim(0, y_upper_bound)
            ax.set_xlim(0, len(distances))

            breakpoint_distance_threshold = np.percentile(distances, breakpoint_percentile_threshold)
            ax.axhline(y=breakpoint_distance_threshold, color='r', linestyle='-', label='Threshold')

            num_distances_above_threshold = np.sum(distances > breakpoint_distance_threshold)
            ax.text(len(distances)*0.01, y_upper_bound/50, f"{num_distances_above_threshold + 1} Chunks")

            indices_above_thresh = np.where(distances > breakpoint_distance_threshold)[0]
            boundaries = [0] + (indices_above_thresh + 1).tolist() + [embeddings.shape[0]]

            for i in range(len(boundaries)-1):
                start_idx = boundaries[i]
                end_idx = boundaries[i+1]
                ax.axvspan(start_idx, end_idx, facecolor=plt.cm.tab10(i % 10), alpha=0.25)
                ax.text(
                    x=(start_idx + end_idx) / 2,
                    y=breakpoint_distance_threshold + y_upper_bound / 20,
                    s=f"Chunk #{i+1}",
                    horizontalalignment='center',
                    rotation='vertical'
                )

            ax.set_title("Dialog Chunks Based on Embedding Breakpoints")
            ax.set_xlabel("Message Index")
            ax.set_ylabel("Cosine Distance")
            ax.legend()

        # Initial plot
        plot_chunks(init_threshold)

        # Slider
        axcolor = 'lightgoldenrodyellow'
        ax_thresh = plt.axes([0.15, 0.1, 0.65, 0.03], facecolor=axcolor)
        thresh_slider = Slider(
            ax=ax_thresh,
            label='Threshold (%)',
            valmin=0,
            valmax=100,
            valinit=init_threshold,
            valstep=1
        )

        # Update function
        def update(val):
            plot_chunks(thresh_slider.val)
            fig.canvas.draw_idle()

        thresh_slider.on_changed(update)

        plt.show()





