import pickle

import polars as pl

from terrorblade.data.database.telegram_database import TelegramDatabase
from terrorblade.data.preprocessing.TelegramPreprocessor import TelegramPreprocessor


def main():
    # file_path = '/home/seva/data/all_chats.parquet'

    # df = pl.read_parquet(file_path)
    # df.shape
    file_path = "/home/seva/data/messages_json/result.json"
    time_window = "30m"
    cluster_size = 1
    big_cluster_size = 10

    preprocessor = TelegramPreprocessor(
        time_window=time_window,
        cluster_size=cluster_size,
        big_cluster_size=big_cluster_size,
    )

    data = preprocessor.process_chats(file_path, time_window=time_window)
    print(data)

    combined_clusters = []
    embeddings_dict = {}
    for chat_id, chat_data in data.items():
        combined_clusters.append(chat_data["clusters"])
        embeddings_dict[chat_id] = chat_data["embeddings"]

    big_dataframe = pl.concat(combined_clusters, how="vertical")
    big_dataframe.write_parquet("/home/seva/data/all_chats.parquet")
    pickle.dump(embeddings_dict, open("/home/seva/data/all_embeddings.pkl", "wb"))


def alt_main():
    preprocessor = TelegramPreprocessor(use_duckdb=True, db_path="telegram_data.db")
    preprocessor.process_chats(phone="+31627866359")
    preprocessor.close()


def third_main():
    # Initialize the interface in read-only mode to avoid locking issues
    db = TelegramDatabase(read_only=True)

    try:
        # Get number of users
        user_count = db.get_user_count()
        print(f"\nTotal users in database: {user_count}")

        # Get all users
        users = db.get_all_users()
        print(f"Users: {users}")

        # Get stats for a specific user
        phone = "+31627866359"
        if user_stats := db.get_user_stats(phone):
            print(f"\nStats for user {phone}:")
            print(f"Total messages: {user_stats.total_messages}")
            print(f"Total chats: {user_stats.total_chats}")

            if user_stats.largest_chat[0]:
                print(f"Largest chat: {user_stats.largest_chat[1]} ({user_stats.largest_chat[2]} messages)")

            if user_stats.largest_cluster[0]:
                print(f"Largest cluster: {user_stats.largest_cluster[1]} ({user_stats.largest_cluster[2]} messages)")

            # Get a random large cluster
            if cluster := db.get_random_large_cluster(phone, min_size=5):
                print(f"\nRandom cluster size: {len(cluster)}")
                print("Sample messages from cluster:")
                print(cluster.select(["text", "date"]).head(3))

            # Get largest cluster messages
            if largest_cluster := db.get_largest_cluster_messages(phone):
                print(f"\nLargest cluster messages ({len(largest_cluster)} messages):")
                print(largest_cluster.select(["text", "date"]).head(5))

            # Get stats for the largest chat
            if user_stats.largest_chat[0]:
                if chat_stats := db.get_chat_stats(phone, user_stats.largest_chat[0]):
                    print(f"\nStats for largest chat {chat_stats.chat_name}:")
                    print(f"Total messages: {chat_stats.message_count}")
                    print(f"Number of clusters: {chat_stats.cluster_count}")
                    print(f"Average cluster size: {chat_stats.avg_cluster_size:.2f}")
                    print(f"Largest cluster size: {chat_stats.largest_cluster_size}")
        else:
            print(f"No data found for user {phone}")
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        # Don't forget to close the connection
        db.close()


if __name__ == "__main__":
    third_main()
