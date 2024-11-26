import os
import polars as pl

from datetime import timedelta
import numpy as np
from sentence_transformers import SentenceTransformer
import cupy as cp
from cuml.cluster import HDBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

input_folder = '/home/seva/data/parquet'  # Adjust the path to your folder


def parse_timestamp(df, timestamp_col):
    """
    Parses and formats the timestamp column in the provided DataFrame.

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
    
def clean_author(df, author_col):
    """
    Cleans the author column in the provided DataFrame.

    Args:
        df (pl.DataFrame): DataFrame containing the author column to clean.
        author_col (str): The name of the column containing author names.

    Returns:
        pl.DataFrame: DataFrame with the author column cleaned.
    """
    return df.with_columns(
        pl.col(author_col)
        .str.replace(r' via.*', '', literal=False)  # Remove ' via' and everything after
        .str.strip_chars('\n')  # Remove newline characters
        .str.strip_chars()  # Remove whitespace from the start and end
    )

def create_clusters(df, time_window, cluster_size=3, big_cluster_size=10):
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

def create_recipient_column(df, author_col):
    """
    Creates a recipient string containing all other participants except the author.

    Args:
        df (pl.DataFrame): DataFrame containing the author column to filter.
        author_col (str): The name of the column containing author names.

    Returns:
        pl.DataFrame: DataFrame with the recipient column added.
    """
    # Extract unique authors and create a list of these authors
    unique_authors = df.select(pl.col(author_col)).unique().to_series().to_list()
    
    # Create a string of recipients excluding the author for each row
    recipients_str = [
        ', '.join([author for author in unique_authors if author != author_name]) 
        for author_name in df[author_col].to_list()
    ]

    # Add the recipient column to the DataFrame
    return df.with_columns(
        pl.Series("recipients", recipients_str)
    )

def find_simmilar_messages_in_chat(df, embeddings):
    """
    Finds similar messages in a chat based on the provided embeddings.
        
    Args:
    df (pl.DataFrame): DataFrame containing the chat messages.
    embeddings (torch.Tensor): Embeddings for the chat messages.
    """
    # Convert the embeddings to a CuPy array
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
            
# Load and concatenate all Parquet files
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
dataframes = []
df_splits = []
for file_name in os.listdir(input_folder):
    if file_name.endswith('.parquet'):
        df = pl.read_parquet(os.path.join(input_folder, file_name))
        
        df = parse_timestamp(df, "timestamp")
        df = clean_author(df, "author")
        df = create_clusters(df, "30m", 3, 20)
        df = create_recipient_column(df, "author")
        texts = df['text'].to_list()
        embeddings = model.encode(
            texts,
            batch_size=256,
            show_progress_bar=True,
            convert_to_tensor=True,
            device='cuda'
        )
        embeddings = embeddings.cpu()
        similarities = cosine_similarity(embeddings[:-1], embeddings[1:]).diagonal()
        distances = 1 - similarities
        plt.figure(figsize=(12, 6))
        plt.plot(distances, label='Cosine Distance')
        y_upper_bound = 0.2
        plt.ylim(0, y_upper_bound)
        plt.xlim(0, len(distances))
        breakpoint_percentile_threshold = 95
        breakpoint_distance_threshold = np.percentile(distances, breakpoint_percentile_threshold)
        plt.axhline(y=breakpoint_distance_threshold, color='r', linestyle='-', label='Threshold')

        num_distances_above_threshold = np.sum(distances > breakpoint_distance_threshold)
        plt.text(len(distances)*0.01, y_upper_bound/50, f"{num_distances_above_threshold + 1} Chunks")

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
            
            plt.axvspan(start_idx, end_idx, facecolor=plt.cm.tab10(i % 10), alpha=0.25)
            plt.text(
                x=(start_idx + end_idx) / 2,
                y=breakpoint_distance_threshold + y_upper_bound / 20,
                s=f"Chunk #{i+1}",
                horizontalalignment='center',
                rotation='vertical'
            )
        
        if indices_above_thresh.size > 0:
            last_breakpoint = indices_above_thresh[-1] + 1
            if last_breakpoint < len(df):
                plt.axvspan(last_breakpoint, len(df), facecolor=plt.cm.tab10(len(indices_above_thresh) % 10), alpha=0.25)
                plt.text(
                    x=(last_breakpoint + len(df)) / 2,
                    y=breakpoint_distance_threshold + y_upper_bound / 20,
                    s=f"Chunk #{len(boundaries)-1}",
                    rotation='vertical'
                )
        
        plt.title("Dialog Chunks Based on Embedding Breakpoints")
        plt.xlabel("Message Index")
        plt.ylabel("Cosine Distance")
        plt.legend()
        plt.show()
        df_split = pl.DataFrame(segments)          
        dataframes.append(df)

all_data = pl.concat(dataframes)

all_data = all_data.sort(by='timestamp')

