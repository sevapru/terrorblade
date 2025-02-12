import pickle
import polars as pl
from src.data.preprocessing.TelegramPreprocessor import TelegramPreprocessor

def main():
    # file_path = '/home/seva/data/all_chats.parquet'
                
    # df = pl.read_parquet(file_path)
    # df.shape
    file_path = '/home/seva/data/messages_json/result.json'
    time_window = '30m'
    cluster_size = 1
    big_cluster_size = 10

    preprocessor = TelegramPreprocessor(time_window=time_window, 
                                         cluster_size=cluster_size, 
                                         big_cluster_size=big_cluster_size)

    data = preprocessor.process_chats(file_path, time_window=time_window)
    print(data)

    #pickle.dump(data, open('/home/seva/data/all_chats.pkl', 'wb'))
    combined_clusters = []
    embeddings_dict = {}
    for chat_id, chat_data in data.items():
        combined_clusters.append(chat_data['clusters'])
        embeddings_dict[chat_id] = chat_data['embeddings']
        
    big_dataframe = pl.concat(combined_clusters, how="vertical")
    big_dataframe.write_parquet('/home/seva/data/all_chats.parquet')
    pickle.dump(embeddings_dict, open('/home/seva/data/all_embeddings.pkl', 'wb'))
    
    

main()