import polars as pl

from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired

df = pl.read_parquet('/home/seva/data/all_chats.parquet')

# Группируем по chat_name и group, объединяем тексты через дефис
grouped_df = df.group_by(['chat_name', 'group']).agg([
    pl.col('text').str.join(' - ').alias('text'),
    pl.col('date').min().alias('start_date'),
    pl.col('date').max().alias('end_date'),
    pl.len().alias('message_count')
]).sort(['chat_name', 'group'])

print(grouped_df)
zeroshot_topic_list = ["машина", "нравится", "сон", "работа", "деньги", "отдых", "путешествие"]

docs = grouped_df['text'].to_list()
topic_model = BERTopic(
    embedding_model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2", 
    min_topic_size=15,
    language="multilingual",
    zeroshot_topic_list=zeroshot_topic_list,
    zeroshot_min_similarity=.85,
    representation_model=KeyBERTInspired()
)

topics, _ = topic_model.fit_transform(docs)