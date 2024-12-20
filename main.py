import os
from TelegramPreprocessor import TelegramPreprocessor

def main():
            

    file_path = '/home/seva/data/messages_json/result.json'
    time_window = '30m'
    cluster_size = 3
    big_cluster_size = 10

    preprocessor = TelegramPreprocessor(time_window=time_window, 
                                        cluster_size=cluster_size, 
                                        big_cluster_size=big_cluster_size)

    data = preprocessor.prepare_data(file_path)
    print(data)

main()