import os
import polars as pl

from datetime import timedelta
import numpy as np
from sentence_transformers import SentenceTransformer
import cupy as cp
from cuml.cluster import HDBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

import seaborn as sns
from ipywidgets import interact

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
        
    def create_clusters(self, df, time_window, cluster_size=3, big_cluster_size=10):
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
        if time_window.endswith('h'):
            window_duration = timedelta(hours=int(time_window[:-1]))
        elif time_window.endswith('m'):
            window_duration = timedelta(minutes=int(time_window[:-1]))
        else:
            raise ValueError("time_window must be formatted as '1h' or '5m'.")

        # Create a column for time differences between consecutive messages
        time_diffs = df.with_columns(
            (pl.col("timestamp").diff().dt.total_seconds()).alias("time_diff")
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
        return df.drop(["pre_cluster", "is_big_cluster"])
    
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
                pl.concat_str('text', separator=' ').alias('text')
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
            batch_size=256,
            show_progress_bar=True,
            convert_to_tensor=True,
            device='cuda'
        ).cpu()
    
    def calculate_distances(self, embeddings):
        """
        Calculates distances based on cosine similarities between embeddings.
            
        Args:
        embeddings (torch.Tensor): Embeddings for the chat messages.
        """
        similarities = cosine_similarity(embeddings[:-1], embeddings[1:]).diagonal()
        distances = 1 - similarities
        return distances
    
    def process_clusters(self, df, time_window, cluster_size=3, big_cluster_size=10):
        """
        Processes the provided DataFrame by creating clusters and calculating embeddings.
        
        Args:
            df (pl.DataFrame): DataFrame containing the chat messages.
        
        Returns:
            pl.DataFrame: DataFrame with clusters and embeddings added.
        """
        df = self.create_clusters(df, time_window, cluster_size, big_cluster_size)
        self.embeddings = self.calculate_embeddings(df)
        df_split = self.calculate_segments(df, self.embeddings)
        return df_split
    
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
        biggest_cluster_id = df.group_by("big_cluster").agg(pl.len().alias("size")).sort("size", reverse=True)["big_cluster"].first()
        self.show_cluster(df, biggest_cluster_id)
        
    def show_all_clusters(self, df):
        """
        Displays all clusters in the provided DataFrame.
        
        Args:
            df (pl.DataFrame): DataFrame containing the chat messages.
        """
        for cluster_id in df['cluster'].unique().to_list():
            self.show_cluster(df, cluster_id)
            
    def calculate_segments(self, df, embeddings, breakpoint_percentile_threshold=95):
        """
        Calculates the segments based on the provided distances.
        
        Args:
            df (pl.DataFrame): DataFrame containing the chat messages.
            distances (np.ndarray): Array of distances between the chat messages.
        
        Returns:
            list: List of dictionaries containing the segments.
        """
        distances = self.calculate_distances(embeddings)
        breakpoint_distance_threshold = np.percentile(distances, breakpoint_percentile_threshold)
        indices_above_thresh = np.where(distances > breakpoint_distance_threshold)[0]

        boundaries = [0] + (indices_above_thresh + 1).tolist() + [len(df)]
        segments = []
        for i in range(len(boundaries)-1):
            start_idx = boundaries[i]
            end_idx = boundaries[i+1]

            segment_df = df.slice(start_idx, end_idx - start_idx)

            segments.append({
                'author': segment_df['author'].first(),
                'author2': segment_df['recipients'].first(),
                'timestamp_start': segment_df['timestamp'].min(),
                'timestamp_end': segment_df['timestamp'].max(),
                'text': ' '.join(segment_df['text'].to_list())
            })
        return pl.DataFrame(segments)   
    
    def display_dialog_chunks(self, embeddings=None):
        if embeddings is None:
            embeddings = self.embeddings

        distances = self.calculate_distances(embeddings)

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
    
class TelegramPreprocessor(TextPreprocessor):
    """
    Preprocesses Telegram chat data.
    """
    def __init__(self, input_folder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_folder = input_folder
        
        
    def parse_timestamp(self, df, timestamp_col):
        """
        Parses and formats the timestamp column in the provided DataFrame.
        
        Process:
        - Extract the first 19 characters of the timestamp string
        - Parse the timestamp string into a datetime object

        Args:
            df (pl.DataFrame): DataFrame containing the timestamp to parse.
            timestamp_col (str): The name of the column containing timestamps.

        Returns:
            pl.DataFrame: DataFrame with the timestamp column parsed and formatted.
        """
        return df.with_columns(
            pl.col(timestamp_col)
            .str.slice(0, 19)
            .str.strptime(pl.Datetime, "%d.%m.%Y %H:%M:%S")
        )
        
           
    def clean_author(self, df, author_col):
        """
        Cleans the author column in the provided DataFrame.
        
        Process:
        - Remove ' via' and everything after
        - Remove newline characters
        - Remove whitespace from the start and end of the author names

        Args:
            df (pl.DataFrame): DataFrame containing the author column to clean.
            author_col (str): The name of the column containing author names.

        Returns:
            pl.DataFrame: DataFrame with the author column cleaned.
        """
        return df.with_columns(
            pl.col(author_col)
            .str.replace(r' via.*', '', literal=False)
            .str.strip_chars('\n')  
            .str.strip_chars()  # 
        )


    def create_recipient_column(self, df, author_col):
        """
        Creates a recipient string containing all other participants except the author.
        
        Process:
        - Extract unique authors and create a list of these authors
        - Create a string of recipients excluding the author for each row
        - Add the recipient column to the DataFrame

        Args:
            df (pl.DataFrame): DataFrame containing the author column to filter.
            author_col (str): The name of the column containing author names.

        Returns:
            pl.DataFrame: DataFrame with the recipient column added.
        """

        unique_authors = df.select(pl.col(author_col)).unique().to_series().to_list()
        recipients_str = [
            ', '.join([author for author in unique_authors if author != author_name]) 
            for author_name in df[author_col].to_list()
        ]
        return df.with_columns(
            pl.Series("recipients", recipients_str)
        )

        
    def prepare_data(self, file_name):
        """
        Loads and concatenates all Parquet files in the specified input folder.
        
        Process:
        - Load each Parquet file
        - Parse the timestamp column
        - Clean the author column
        - Create the recipient column
        
        Args:
            input_folder (str): Path to the folder containing the Parquet files.
        
        Returns:
            pl.DataFrame: DataFrame containing all the loaded data.
        """
        if file_name.endswith('.parquet'):
            df = pl.read_parquet(os.path.join(self.input_folder, file_name))
            df = self.parse_timestamp(df, "timestamp")
            df = self.clean_author(df, "author")
            df = self.create_recipient_column(df, "author")
            return df
        
    def process_chat(self, file_name, time_window, cluster_size=3, big_cluster_size=10):
        """
        Processes the provided DataFrame by creating clusters and calculating embeddings.
        
        Args:
            df (pl.DataFrame): DataFrame containing the chat messages.
        
        Returns:
            pl.DataFrame: DataFrame with clusters and embeddings added.
        """
        df = self.prepare_data(file_name=file_name)
        
        return self.process_clusters(df, time_window, cluster_size, big_cluster_size)
    
    def process_all_chats(self, input_folder, time_window, cluster_size=3, big_cluster_size=10):
        """
        Processes all chat data in the provided input folder.
        
        Args:
            input_folder (str): Path to the folder containing the Parquet files.
            time_window (str): Time window for determining clusters, formatted as "1h" (for one hour) or "5m" (for five minutes).
            cluster_size (int): Minimum number of messages required to consider it a cluster.
            big_cluster_size (int): Minimum number of messages required to consider it a big cluster.
        
        Returns:
            pl.DataFrame: DataFrame containing all the loaded and processed data.
        """
        self.input_folder = input_folder
        dataframes = []
        for file_name in os.listdir(input_folder):
            df = self.process_chat(file_name, time_window, cluster_size, big_cluster_size)
            dataframes.append(df)
        return pl.concat(dataframes)


def main():
    input_folder = '/home/seva/data/parquet'
    time_window = '30m'
    cluster_size = 3
    big_cluster_size = 10
    
    preprocessor = TelegramPreprocessor(input_folder=input_folder, 
                                        time_window=time_window, 
                                        cluster_size=cluster_size, 
                                        big_cluster_size=big_cluster_size)

    dataframes = []
    df_splits = []
    for file_name in os.listdir(input_folder):
        df = preprocessor.prepare_data(file_name)
        dataframes.append(df)
        df_split = preprocessor.process_chat(file_name, time_window, cluster_size, big_cluster_size)
        preprocessor.display_dialog_chunks()
        df_splits.append(df_split)

    all_data = pl.concat(dataframes)

    all_data = all_data.sort(by='timestamp')
    print(all_data.shape)

main()
