import os
import polars as pl

from datetime import timedelta

input_folder = '~/data/parquet'  # Adjust the path to your folder

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
    
# Load and concatenate all Parquet files
dataframes = []
for file_name in os.listdir(input_folder):
    if file_name.endswith('.parquet'):
        df = pl.read_parquet(os.path.join(input_folder, file_name))
        
        df = parse_timestamp(df, "timestamp")
        df = clean_author(df, "author")
        df = create_clusters(df, "1h", 5, 100)
        df = create_recipient_column(df, "author")
        dataframes.append(df)

all_data = pl.concat(dataframes)

all_data = all_data.sort(by='timestamp')

