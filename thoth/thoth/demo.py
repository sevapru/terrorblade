import os
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv

from thoth.analyzer import ThothAnalyzer

load_dotenv(Path().parent.parent / ".env")
# Or specify configuration directly
analyzer = ThothAnalyzer(db_path=os.getenv("DUCKDB_PATH"), phone="+31627866359")

# Import data from DuckDB
analyzer.import_from_duckdb()

# Analyze topics
topics = analyzer.analyze_topics(chat_id=123456789, n_topics=5, date_from=datetime.now() - timedelta(days=30))

# Analyze topic sentiment
sentiment = analyzer.analyze_topic_sentiment(topic_id=1)

# Search for semantically similar messages
results = analyzer.search(query="What do you think about the new policy?", chat_id=123456789, limit=5)
