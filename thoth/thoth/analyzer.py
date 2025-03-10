import os
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import duckdb
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import polars as pl
import qdrant_client
import scipy.stats as stats
import seaborn as sns
import torch
from qdrant_client.conversions import common_types as ct
from qdrant_client.http import models
from qdrant_client.http.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import DBSCAN, KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from textblob import TextBlob

from thoth.utils import NICE, Config


class ThothAnalyzer:
    """
    Thoth Analyzer for chat data analysis using vector embeddings and clustering techniques.

    This class provides methods for analyzing chat data using a vector database approach,
    enabling semantic search, topic analysis, sentiment analysis, and user interaction patterns.
    """

    def __init__(
        self,
        db_path: Optional[Path | str] = None,
        phone: Optional[str] = None,
        embedding_model: Optional[str] = None,
        vector_size: Optional[int] = None,
        qdrant_path: Optional[str] = None,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        config: Optional[Config] = None,
    ):
        """
        Initialize the ThothAnalyzer with needed connections and models.

        Args:
            db_path: Path to the DuckDB database file
            phone: Phone number used for user-specific tables
            embedding_model: Name of the sentence-transformers model to use
            vector_size: Size of embedding vectors to store
            qdrant_path: Path to the Qdrant database directory (for local mode)
            qdrant_url: URL for remote Qdrant instance
            qdrant_api_key: API key for remote Qdrant instance
            config: Configuration object (if provided, other parameters are ignored)
        """
        # Get configuration
        self._config = config or Config()
        self.logger = self._config.logger

        # Set parameters, with explicit parameters taking precedence over config
        self.db_path = Path(db_path) if db_path else Path(self._config.db_path)
        self.phone = phone if phone else self._config.phone
        if self.phone and self.phone.startswith("+"):
            self.phone = self.phone[1:]

        # Initialize embedding model
        self.embedding_model_name = embedding_model or self._config.embedding_model
        self.vector_size = vector_size or self._config.vector_size
        self._embedding_model = None  # Lazy-loaded to save memory when not needed

        # Initialize Qdrant client based on local or remote mode
        self.logger.info("Initializing Qdrant client")
        qdrant_config = {}

        # Override config with explicit parameters if provided
        if qdrant_url:
            qdrant_config["url"] = qdrant_url
            if qdrant_api_key:
                qdrant_config["api_key"] = qdrant_api_key
            self.logger.info(f"Using remote Qdrant at: {qdrant_url}")
        elif qdrant_path:
            qdrant_config["path"] = qdrant_path
            self.logger.info(f"Using local Qdrant at path: {qdrant_path}")
        else:
            # Use config settings
            qdrant_config = self._config.get_qdrant_config()

        # Initialize Qdrant client
        self.qdrant = qdrant_client.QdrantClient(**qdrant_config)

        # Collections we'll use
        collection_prefix = f"{self.phone}_" if self.phone else ""
        self.messages_collection = f"{collection_prefix}messages"
        self.embeddings_collection = f"{collection_prefix}embeddings"
        self.chat_collection_prefix = f"{collection_prefix}chat_"
        self.topic_collection = f"{collection_prefix}topics"

        self.logger.info("Initializing collections")
        start_time = time.time()
        self._init_collections()
        self.logger.nice("Collections initialized in %.2f seconds", time.time() - start_time)

    @property
    def embedding_model(self):
        """Lazy-load the embedding model when first needed"""
        if self._embedding_model is None:
            self.logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self._embedding_model = SentenceTransformer(self.embedding_model_name)
            self.logger.nice("Embedding model loaded with vector size: %d", self.vector_size)
        return self._embedding_model

    def _init_collections(self) -> None:
        """Initialize vector database collections if they don't exist."""
        try:
            # Check if we're using a remote Qdrant
            is_remote = hasattr(self.qdrant, "url") and self.qdrant.url is not None

            # Messages collection for storing message metadata
            try:
                self.qdrant.get_collection(self.messages_collection)
                self.logger.debug(f"Collection {self.messages_collection} already exists")
            except Exception as e:
                self.logger.info(f"Creating collection: {self.messages_collection}")
                self.qdrant.create_collection(
                    collection_name=self.messages_collection,
                    vectors_config=models.VectorParams(
                        size=self.vector_size,
                        distance=models.Distance.COSINE,
                    ),
                )

                # Only create payload indexes for remote Qdrant as they have no effect locally
                if is_remote:
                    self.logger.info(f"Creating payload indexes for {self.messages_collection}")
                    # Create needed payload indexes
                    self.qdrant.create_payload_index(
                        collection_name=self.messages_collection,
                        field_name="chat_id",
                        field_schema=models.PayloadSchemaType.INTEGER,
                    )
                    self.qdrant.create_payload_index(
                        collection_name=self.messages_collection,
                        field_name="date",
                        field_schema=models.PayloadSchemaType.DATETIME,
                    )

            # Similar setup for other collections
            try:
                self.qdrant.get_collection(self.topic_collection)
                self.logger.debug(f"Collection {self.topic_collection} already exists")
            except Exception:
                self.logger.info(f"Creating collection: {self.topic_collection}")
                self.qdrant.create_collection(
                    collection_name=self.topic_collection,
                    vectors_config=models.VectorParams(
                        size=self.vector_size,
                        distance=models.Distance.COSINE,
                    ),
                )
                if is_remote:
                    self.logger.info(f"Creating payload indexes for {self.topic_collection}")
                    self.qdrant.create_payload_index(
                        collection_name=self.topic_collection,
                        field_name="topic_id",
                        field_schema=models.PayloadSchemaType.INTEGER,
                    )
        except Exception as e:
            self.logger.error(f"Error initializing collections: {str(e)}")
            raise

    def import_from_duckdb(self) -> None:
        """Import messages from DuckDB and store as vectors in Qdrant."""
        if not self.db_path or not self.db_path.exists():
            self.logger.error(f"DuckDB file not found at {self.db_path}")
            raise FileNotFoundError(f"DuckDB file not found at {self.db_path}")

        self.logger.info(f"Importing data from DuckDB: {self.db_path}")
        start_time = time.time()

        # Connect to DuckDB
        con = duckdb.connect(str(self.db_path))

        # Get all chats
        chat_table = f"messages_{self.phone}"
        chat_query = f"SELECT chat_id, chat_name FROM {chat_table} ORDER BY chat_id"

        try:
            chats = con.execute(chat_query).fetchall()
            self.logger.info(f"Found {len(chats)} chats")

            # Process each chat
            for chat_id, chat_name in chats:
                self.logger.info(f"Processing chat: {chat_name} (ID: {chat_id})")
                chat_start_time = time.time()

                # Create collection for this chat if it doesn't exist
                chat_collection = f"{self.chat_collection_prefix}{chat_id}"
                try:
                    self.qdrant.get_collection(chat_collection)
                    self.logger.debug(f"Collection {chat_collection} already exists")
                except Exception:
                    self.logger.info(f"Creating collection: {chat_collection}")
                    self.qdrant.create_collection(
                        collection_name=chat_collection,
                        vectors_config=models.VectorParams(
                            size=self.vector_size,
                            distance=models.Distance.COSINE,
                        ),
                    )

                # Get messages for this chat
                msg_table = f"messages_{self.phone}" if self.phone else "messages"
                message_query = f"""
                SELECT id, date, from_id, text 
                FROM {msg_table} 
                WHERE chat_id = {chat_id} AND text != ''
                ORDER BY date
                """

                messages = con.execute(message_query).fetchall()
                self.logger.info(f"Found {len(messages)} messages in chat {chat_name}")

                if not messages:
                    continue

                # Generate and store embeddings for messages
                embeddings_batch = []
                texts = []

                for msg_id, date, from_id, text in messages:
                    if not text or text.isspace():
                        continue
                    texts.append(text)

                # Process in batches of 100 to avoid memory issues
                batch_size = 100
                total_processed = 0

                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i : i + batch_size]
                    batch_messages = messages[i : i + batch_size]

                    self.logger.debug(f"Processing batch of {len(batch_texts)} messages")
                    batch_start_time = time.time()

                    # Generate embeddings
                    batch_embeddings = self.embedding_model.encode(batch_texts)

                    # Prepare data for Qdrant
                    points = []
                    for j, embedding in enumerate(batch_embeddings):
                        msg_id, date, from_id, text = batch_messages[j]

                        # Add to message collection
                        points.append(
                            models.PointStruct(
                                id=msg_id,
                                vector=embedding.tolist(),
                                payload={
                                    "chat_id": chat_id,
                                    "chat_name": chat_name,
                                    "date": date.isoformat() if hasattr(date, "isoformat") else date,
                                    "from_id": from_id,
                                    "text": text,
                                },
                            )
                        )

                    # Store in Qdrant
                    self.qdrant.upsert(collection_name=chat_collection, points=points)

                    # Also store in main messages collection
                    self.qdrant.upsert(collection_name=self.messages_collection, points=points)

                    total_processed += len(batch_texts)
                    self.logger.debug(
                        f"Processed batch in {time.time() - batch_start_time:.2f} seconds. "
                        f"Total progress: {total_processed}/{len(texts)}"
                    )

                chat_processing_time = time.time() - chat_start_time
                self.logger.nice(
                    "Processed chat %s (%d) with %d messages in %.2f seconds (%.1f msgs/sec)",
                    chat_name,
                    chat_id,
                    len(messages),
                    chat_processing_time,
                    len(messages) / chat_processing_time if chat_processing_time > 0 else 0,
                )

            # Log completion metrics
            total_time = time.time() - start_time
            total_messages = sum(
                len(con.execute(f"SELECT id FROM {msg_table} WHERE chat_id = {chat_id} AND text != ''").fetchall())
                for chat_id, _ in chats
            )

            self.logger.nice(
                "Completed import of %d chats with %d messages in %.2f seconds (%.1f msgs/sec)",
                len(chats),
                total_messages,
                total_time,
                total_messages / total_time if total_time > 0 else 0,
            )

        except Exception as e:
            self.logger.error(f"Error importing data: {str(e)}")
            raise
        finally:
            con.close()

    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a text string using the model."""
        return self.embedding_model.encode(text)

    def add_message(self, message_id: int, chat_id: int, text: str, date: datetime, from_id: int) -> None:
        """Add a new message with embedding to the vector store."""
        vector = self.get_embedding(text)

        self.qdrant.upsert(
            collection_name=self.messages_collection,
            points=[
                PointStruct(
                    id=f"{chat_id}_{message_id}",
                    vector=vector.tolist(),
                    payload={
                        "message_id": message_id,
                        "chat_id": chat_id,
                        "date": date.isoformat(),
                        "from_id": from_id,
                        "text": text,
                    },
                )
            ],
        )

    def find_most_common_token(self) -> tuple[str, float]:
        """Find the most common token among all embeddings."""
        # Get embeddings and group by similarity
        result = self.qdrant.scroll(
            collection_name=self.messages_collection,
            limit=1000,
            with_vectors=True,
        )

        if not result or not result[0]:
            return ("No embeddings found", 0.0)

        # Cluster the vectors
        vectors = [point.vector for point in result[0]]
        kmeans = KMeans(n_clusters=min(10, len(vectors)), random_state=42)
        clusters = kmeans.fit_predict(vectors)

        # Find the most common cluster
        cluster_counter = Counter(clusters)
        most_common_cluster, count = cluster_counter.most_common(1)[0]

        return (f"Token cluster {most_common_cluster}", float(count))

    def find_most_common_topic(self, n_clusters: int = 10) -> tuple[str, float]:
        """Find the most common topic (cluster of embeddings)."""
        # Get all vectors
        result = self.qdrant.scroll(
            collection_name=self.messages_collection,
            limit=10000,
            with_vectors=True,
        )

        if not result or not result[0]:
            return ("No embeddings found", 0.0)

        vectors = [point.vector for point in result[0]]
        points = result[0]

        # Cluster using KMeans
        kmeans = KMeans(n_clusters=min(n_clusters, len(vectors)), random_state=42)
        clusters = kmeans.fit_predict(vectors)

        # Assign cluster to each point
        cluster_points = defaultdict(list)
        for i, cluster_id in enumerate(clusters):
            cluster_points[cluster_id].append(points[i])

        # Find most common cluster
        cluster_sizes = {cluster_id: len(points) for cluster_id, points in cluster_points.items()}
        most_common_cluster = max(cluster_sizes, key=cluster_sizes.get)

        # Extract text from the points in the most common cluster
        texts = [point.payload["text"] for point in cluster_points[most_common_cluster]]

        # Use TF-IDF to find the most representative words
        vectorizer = TfidfVectorizer(max_features=10)
        tfidf_matrix = vectorizer.fit_transform(texts)

        # Get top features
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = np.mean(tfidf_matrix.toarray(), axis=0)
        top_indices = tfidf_scores.argsort()[-5:][::-1]
        top_words = [feature_names[i] for i in top_indices]

        return (f"Topic: {', '.join(top_words)}", float(cluster_sizes[most_common_cluster]))

    def find_topic_evolution(self, time_window: str = "1 month") -> List[Dict[str, Any]]:
        """Track how topics evolve over time."""
        # Convert time window to timedelta
        if time_window.endswith("month"):
            window = timedelta(days=30 * int(time_window.split()[0]))
        elif time_window.endswith("day"):
            window = timedelta(days=int(time_window.split()[0]))
        elif time_window.endswith("hour"):
            window = timedelta(hours=int(time_window.split()[0]))
        else:
            window = timedelta(days=30)  # Default to 1 month

        # Get all messages with vectors
        result = self.qdrant.scroll(
            collection_name=self.messages_collection,
            limit=10000,
            with_vectors=True,
            with_payload=True,
        )

        if not result or not result[0]:
            return []

        points = result[0]

        # Create time-based bins
        dates = [datetime.fromisoformat(point.payload["date"]) for point in points]

        if not dates:
            return []

        min_date = min(dates)
        max_date = max(dates)

        # Create time bins
        bins = []
        current = min_date
        while current <= max_date:
            next_date = current + window
            bins.append((current, next_date))
            current = next_date

        # Group messages by time bins
        time_bins = defaultdict(list)
        for i, point in enumerate(points):
            date = dates[i]
            for bin_start, bin_end in bins:
                if bin_start <= date < bin_end:
                    time_bins[(bin_start, bin_end)].append((point.vector, point.payload))
                    break

        # Analyze each time bin
        results = []
        for (bin_start, bin_end), bin_points in time_bins.items():
            if len(bin_points) < 10:  # Skip bins with too few points
                continue

            vectors = [p[0] for p in bin_points]
            payloads = [p[1] for p in bin_points]

            # Cluster this time bin
            n_clusters = min(5, len(vectors) // 2)
            if n_clusters < 2:
                continue

            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(vectors)

            # Group by cluster
            cluster_texts = defaultdict(list)
            for i, cluster_id in enumerate(clusters):
                cluster_texts[cluster_id].append(payloads[i]["text"])

            # Get topics for each cluster
            topics = {}
            for cluster_id, texts in cluster_texts.items():
                vectorizer = TfidfVectorizer(max_features=5)
                try:
                    tfidf_matrix = vectorizer.fit_transform(texts)
                    feature_names = vectorizer.get_feature_names_out()
                    tfidf_scores = np.mean(tfidf_matrix.toarray(), axis=0)
                    top_indices = tfidf_scores.argsort()[-3:][::-1]
                    topics[cluster_id] = [feature_names[i] for i in top_indices]
                except:
                    # Handle empty vectors
                    topics[cluster_id] = ["(empty)"]

            results.append(
                {
                    "start_date": bin_start.isoformat(),
                    "end_date": bin_end.isoformat(),
                    "message_count": len(bin_points),
                    "topics": topics,
                    "dominant_topic": max(topics.items(), key=lambda x: len(cluster_texts[x[0]]))[1],
                }
            )

        return results

    def find_related_topics(self, threshold: float = 0.7) -> List[Tuple[int, int, float]]:
        """Find related topics based on embedding similarity."""
        # First, get topic clusters
        result = self.qdrant.scroll(
            collection_name=self.messages_collection,
            limit=10000,
            with_vectors=True,
            with_payload=True,
        )

        if not result or not result[0]:
            return []

        points = result[0]
        vectors = [point.vector for point in points]

        # Cluster to find topics
        n_clusters = min(20, len(vectors) // 10) if len(vectors) > 20 else 5
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(vectors)

        # Get cluster centroids
        centroids = kmeans.cluster_centers_

        # Calculate similarity between centroids
        similarities = []
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                similarity = cosine_similarity([centroids[i]], [centroids[j]])[0][0]
                if similarity >= threshold:
                    similarities.append((i, j, float(similarity)))

        return sorted(similarities, key=lambda x: x[2], reverse=True)

    def analyze_user_topics(self, min_messages: int = 10) -> Dict[str, List[int]]:
        """Analyze which topics each user participates in."""
        # Get all messages
        result = self.qdrant.scroll(
            collection_name=self.messages_collection,
            limit=10000,
            with_vectors=True,
            with_payload=True,
        )

        if not result or not result[0]:
            return {}

        points = result[0]
        vectors = [point.vector for point in points]

        # Cluster to find topics
        n_clusters = min(20, len(vectors) // 10) if len(vectors) > 20 else 5
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(vectors)

        # Group messages by user and count topics
        user_topics = defaultdict(list)
        for i, point in enumerate(points):
            user_id = str(point.payload.get("from_id", "unknown"))
            cluster_id = int(clusters[i])
            user_topics[user_id].append(cluster_id)

        # Filter users with minimum messages
        return {user: topic_list for user, topic_list in user_topics.items() if len(topic_list) >= min_messages}

    def analyze_topic_sentiment(self, topic_id: Optional[int] = None) -> Dict[int, float]:
        """Analyze sentiment for topics or a specific topic."""
        # This is a simplified implementation - production would use a proper sentiment model

        # Get all messages
        result = self.qdrant.scroll(
            collection_name=self.messages_collection,
            limit=10000,
            with_vectors=True,
            with_payload=True,
        )

        if not result or not result[0]:
            return {}

        points = result[0]
        vectors = [point.vector for point in points]

        # Cluster to find topics if topic_id not provided
        n_clusters = min(20, len(vectors) // 10) if len(vectors) > 20 else 5
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(vectors)

        # Simple sentiment based on a known sentiment axis
        # In a full implementation, we'd use a sentiment model
        sentiment_vector = np.array([0.1, 0.2, 0.3, 0.4, 0.5] + [0] * (self.vector_size - 5))

        topic_sentiments = {}
        for i, cluster_id in enumerate(clusters):
            if topic_id is not None and cluster_id != topic_id:
                continue

            # Calculate dot product with sentiment vector as rough sentiment
            sentiment = np.dot(vectors[i], sentiment_vector) / (
                np.linalg.norm(vectors[i]) * np.linalg.norm(sentiment_vector)
            )

            if cluster_id not in topic_sentiments:
                topic_sentiments[cluster_id] = []

            topic_sentiments[cluster_id].append(float(sentiment))

        # Average sentiments per topic
        return {topic: np.mean(sentiments) for topic, sentiments in topic_sentiments.items()}

    def analyze_topic_activity(self, time_window: str = "1 hour") -> Dict[int, Dict[str, int]]:
        """Analyze activity levels within topics across different times."""
        # Convert time window to hours for binning
        window_hours = 1
        if time_window.endswith("day"):
            window_hours = 24 * int(time_window.split()[0])
        elif time_window.endswith("hour"):
            window_hours = int(time_window.split()[0])

        # Get messages with timestamps
        result = self.qdrant.scroll(
            collection_name=self.messages_collection,
            limit=10000,
            with_vectors=True,
            with_payload=True,
        )

        if not result or not result[0]:
            return {}

        points = result[0]
        vectors = [point.vector for point in points]

        # Cluster messages into topics
        n_clusters = min(20, len(vectors) // 10) if len(vectors) > 20 else 5
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(vectors)

        # Create hour bins for a day cycle (0-23)
        activity = {cluster_id: {f"{h:02d}:00": 0 for h in range(24)} for cluster_id in range(n_clusters)}

        # Count messages per hour for each topic
        for i, point in enumerate(points):
            cluster_id = clusters[i]
            try:
                date = datetime.fromisoformat(point.payload["date"])
                hour = date.hour
                activity[cluster_id][f"{hour:02d}:00"] += 1
            except (ValueError, KeyError):
                continue

        return activity

    def analyze_user_interactions(self, min_interactions: int = 5) -> List[Dict[str, Any]]:
        """Analyze how users interact with each other through reply chains."""
        # Get all messages with reply information
        result = self.qdrant.scroll(
            collection_name=self.messages_collection,
            limit=10000,
            with_payload=True,
        )

        if not result or not result[0]:
            return []

        points = result[0]

        # Create a graph of user interactions
        user_graph = nx.DiGraph()

        # Process reply chains
        for point in points:
            from_id = point.payload.get("from_id")
            reply_to = point.payload.get("reply_to_message_id")
            chat_id = point.payload.get("chat_id")

            if not from_id or not reply_to:
                continue

            # Find the original message that was replied to
            try:
                original_msg = self.qdrant.retrieve(
                    collection_name=self.messages_collection,
                    ids=[f"{chat_id}_{reply_to}"],
                )

                if original_msg and original_msg.points:
                    to_id = original_msg.points[0].payload.get("from_id")

                    if to_id and from_id != to_id:
                        if not user_graph.has_edge(from_id, to_id):
                            user_graph.add_edge(from_id, to_id, weight=0)

                        user_graph[from_id][to_id]["weight"] += 1
            except:
                continue

        # Filter to only include edges with at least min_interactions
        significant_edges = [
            (u, v, d) for u, v, d in user_graph.edges(data=True) if d.get("weight", 0) >= min_interactions
        ]

        # Format results
        interactions = []
        for from_id, to_id, data in significant_edges:
            interactions.append(
                {
                    "from_user": str(from_id),
                    "to_user": str(to_id),
                    "interaction_count": data.get("weight", 0),
                }
            )

        return sorted(interactions, key=lambda x: x["interaction_count"], reverse=True)

    def extract_topic_keywords(self, top_n: int = 10) -> Dict[int, List[Tuple[str, float]]]:
        """Extract the most representative keywords for each topic cluster."""
        # Get all messages
        result = self.qdrant.scroll(
            collection_name=self.messages_collection,
            limit=10000,
            with_vectors=True,
            with_payload=True,
        )

        if not result or not result[0]:
            return {}

        points = result[0]
        vectors = [point.vector for point in points]
        texts = [point.payload.get("text", "") for point in points]

        # Cluster into topics
        n_clusters = min(20, len(vectors) // 10) if len(vectors) > 20 else 5
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(vectors)

        # Group texts by cluster
        cluster_texts = defaultdict(list)
        for i, cluster_id in enumerate(clusters):
            if texts[i]:
                cluster_texts[cluster_id].append(texts[i])

        # Extract keywords for each cluster using TF-IDF
        keywords = {}
        for cluster_id, texts in cluster_texts.items():
            if len(texts) < 3:  # Skip clusters with very few texts
                continue

            try:
                vectorizer = TfidfVectorizer(max_features=top_n * 2)
                tfidf_matrix = vectorizer.fit_transform(texts)
                feature_names = vectorizer.get_feature_names_out()

                # Get average TF-IDF score for each word across all documents in this cluster
                tfidf_scores = np.mean(tfidf_matrix.toarray(), axis=0)

                # Sort words by score and get top_n
                top_indices = tfidf_scores.argsort()[-top_n:][::-1]
                keywords[cluster_id] = [(feature_names[i], float(tfidf_scores[i])) for i in top_indices]
            except:
                keywords[cluster_id] = [("insufficient data", 0.0)]

        return keywords

    def analyze_topic_influence(self, min_replies: int = 5) -> Dict[int, float]:
        """Analyze which topics generate the most replies/engagement."""
        # Get all messages with reply information
        result = self.qdrant.scroll(
            collection_name=self.messages_collection,
            limit=10000,
            with_vectors=True,
            with_payload=True,
        )

        if not result or not result[0]:
            return {}

        points = result[0]
        vectors = [point.vector for point in points]

        # Create cluster assignments
        n_clusters = min(20, len(vectors) // 10) if len(vectors) > 20 else 5
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(vectors)

        # Map message IDs to their clusters
        message_clusters = {}
        for i, point in enumerate(points):
            chat_id = point.payload.get("chat_id")
            message_id = point.payload.get("message_id")
            if chat_id and message_id:
                message_clusters[(chat_id, message_id)] = clusters[i]

        # Count replies for each cluster
        cluster_replies = defaultdict(int)
        cluster_messages = defaultdict(int)

        for point in points:
            chat_id = point.payload.get("chat_id")
            reply_to = point.payload.get("reply_to_message_id")

            if not chat_id or not reply_to:
                continue

            # Find the cluster of the message being replied to
            replied_to_key = (chat_id, reply_to)
            if replied_to_key in message_clusters:
                original_cluster = message_clusters[replied_to_key]
                cluster_replies[original_cluster] += 1

        # Count total messages per cluster
        for i, cluster_id in enumerate(clusters):
            cluster_messages[cluster_id] += 1

        # Calculate influence score: replies per message
        influence = {}
        for cluster_id, message_count in cluster_messages.items():
            if message_count >= min_replies:
                reply_count = cluster_replies.get(cluster_id, 0)
                influence[cluster_id] = reply_count / message_count

        return influence

    def analyze_user_engagement(self) -> Dict[str, Dict[str, float]]:
        """Analyze user engagement patterns across different topics."""
        # Get all messages
        result = self.qdrant.scroll(
            collection_name=self.messages_collection,
            limit=10000,
            with_vectors=True,
            with_payload=True,
        )

        if not result or not result[0]:
            return {}

        points = result[0]
        vectors = [point.vector for point in points]

        # Cluster into topics
        n_clusters = min(20, len(vectors) // 10) if len(vectors) > 20 else 5
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(vectors)

        # Track user engagement metrics
        user_engagements = defaultdict(lambda: defaultdict(int))

        # Count messages, replies, and initiations per user per topic
        for i, point in enumerate(points):
            from_id = str(point.payload.get("from_id", "unknown"))
            cluster_id = clusters[i]
            cluster_key = f"topic_{cluster_id}"

            # Count message in topic
            user_engagements[from_id]["message_count"] += 1
            user_engagements[from_id][f"{cluster_key}_count"] += 1

            # Check if it's a reply
            if point.payload.get("reply_to_message_id"):
                user_engagements[from_id]["reply_count"] += 1
                user_engagements[from_id][f"{cluster_key}_replies"] += 1

        # Calculate engagement metrics
        engagement_metrics = {}
        for user_id, metrics in user_engagements.items():
            if metrics["message_count"] < 3:  # Skip users with very few messages
                continue

            # Calculate overall engagement ratio (replies/total)
            reply_ratio = metrics["reply_count"] / metrics["message_count"] if metrics["message_count"] > 0 else 0

            # Calculate per-topic engagement
            topic_engagement = {}
            for k in range(n_clusters):
                topic_key = f"topic_{k}"
                topic_count = metrics.get(f"{topic_key}_count", 0)
                topic_replies = metrics.get(f"{topic_key}_replies", 0)

                # Only include topics the user has participated in
                if topic_count > 0:
                    topic_ratio = topic_replies / topic_count
                    # Topic participation percentage
                    participation = topic_count / metrics["message_count"]
                    topic_engagement[topic_key] = {
                        "message_count": topic_count,
                        "reply_ratio": float(topic_ratio),
                        "participation": float(participation),
                    }

            engagement_metrics[user_id] = {
                "total_messages": metrics["message_count"],
                "overall_reply_ratio": float(reply_ratio),
                "topics": topic_engagement,
            }

        return engagement_metrics

    def build_topic_similarity_network(self, min_similarity: float = 0.5, max_connections: int = 100) -> nx.Graph:
        """Build a network of topics connected by their semantic similarity."""
        # Get all message vectors
        result = self.qdrant.scroll(
            collection_name=self.messages_collection,
            limit=10000,
            with_vectors=True,
            with_payload=True,
        )

        if not result or not result[0]:
            return nx.Graph()

        points = result[0]
        vectors = [point.vector for point in points]

        # Cluster into topics
        n_clusters = min(20, len(vectors) // 10) if len(vectors) > 20 else 5
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(vectors)

        # Get cluster centroids
        centroids = kmeans.cluster_centers_

        # Build a graph of topic similarities
        G = nx.Graph()

        # Add nodes for each topic
        for i in range(n_clusters):
            # Count messages in this cluster
            count = sum(1 for c in clusters if c == i)
            G.add_node(i, size=count, label=f"Topic {i}")

        # Add edges between topics based on cosine similarity
        edges = []
        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                similarity = float(cosine_similarity([centroids[i]], [centroids[j]])[0][0])
                if similarity >= min_similarity:
                    edges.append((i, j, similarity))

        # Sort edges by similarity and add top max_connections
        edges.sort(key=lambda x: x[2], reverse=True)
        for i, j, similarity in edges[:max_connections]:
            G.add_edge(i, j, weight=similarity)

        return G

    def predict_topic_trends(
        self, time_window: str = "1 day", prediction_days: int = 7
    ) -> Dict[int, List[Tuple[datetime, float]]]:
        """Predict how topics will trend in the near future."""
        # Parse time window
        if time_window.endswith("day"):
            window_days = int(time_window.split()[0])
            delta = timedelta(days=window_days)
        elif time_window.endswith("hour"):
            window_hours = int(time_window.split()[0])
            delta = timedelta(hours=window_hours)
        else:
            delta = timedelta(days=1)

        # Get all messages with timestamps
        result = self.qdrant.scroll(
            collection_name=self.messages_collection,
            limit=10000,
            with_vectors=True,
            with_payload=True,
        )

        if not result or not result[0]:
            return {}

        points = result[0]
        vectors = [point.vector for point in points]

        # Convert dates to datetime objects
        try:
            dates = [datetime.fromisoformat(point.payload["date"]) for point in points]
        except (ValueError, KeyError):
            return {}

        # Get the date range
        min_date = min(dates)
        max_date = max(dates)

        # Calculate number of bins
        num_bins = max(3, int((max_date - min_date) / delta))

        # Create time bins
        bins = []
        for i in range(num_bins):
            bin_start = min_date + i * delta
            bin_end = bin_start + delta
            bins.append((bin_start, bin_end))

        # Cluster the messages
        n_clusters = min(10, len(vectors) // 20) if len(vectors) > 20 else 5
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(vectors)

        # Count topic occurrences in each time bin
        topic_counts = {i: [] for i in range(n_clusters)}

        for bin_start, bin_end in bins:
            # Count occurrences of each topic in this bin
            bin_counts = {i: 0 for i in range(n_clusters)}

            for i, date in enumerate(dates):
                if bin_start <= date < bin_end:
                    cluster_id = clusters[i]
                    bin_counts[cluster_id] += 1

            # Store (bin_end, count) pairs
            for topic_id, count in bin_counts.items():
                topic_counts[topic_id].append((bin_end, count))

        # Create predictions using linear regression
        predictions = {}
        for topic_id, counts in topic_counts.items():
            if len(counts) < 3:  # Need enough data points
                continue

            # Prepare data for regression
            X = np.array([(d - min_date).total_seconds() for d, _ in counts]).reshape(-1, 1)
            y = np.array([c for _, c in counts])

            # Fit linear regression
            model = LinearRegression()
            model.fit(X, y)

            # Generate predictions
            future_dates = []
            last_date = max_date
            for i in range(1, prediction_days + 1):
                future_date = last_date + timedelta(days=i)
                future_dates.append(future_date)

            future_X = np.array([(d - min_date).total_seconds() for d in future_dates]).reshape(-1, 1)
            future_y = model.predict(future_X)

            # Store predictions
            predictions[topic_id] = [(date, max(0.0, float(count))) for date, count in zip(future_dates, future_y)]

        return predictions

    def analyze_cross_chat_topics(self, min_similarity: float = 0.7) -> List[Dict[str, Any]]:
        """Identify topics that appear across multiple chats."""
        # Get all messages with chat information
        result = self.qdrant.scroll(
            collection_name=self.messages_collection,
            limit=10000,
            with_vectors=True,
            with_payload=True,
        )

        if not result or not result[0]:
            return []

        points = result[0]
        vectors = [point.vector for point in points]

        # Extract chat IDs
        chat_ids = list(set(point.payload.get("chat_id") for point in points))

        if len(chat_ids) <= 1:
            return []  # Need multiple chats to compare

        # Cluster across all chats
        n_clusters = min(20, len(vectors) // 10) if len(vectors) > 20 else 5
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(vectors)

        # Get centroids
        centroids = kmeans.cluster_centers_

        # Map messages to chats and clusters
        chat_clusters = defaultdict(lambda: defaultdict(list))
        for i, point in enumerate(points):
            chat_id = point.payload.get("chat_id")
            if chat_id is not None:
                cluster_id = clusters[i]
                chat_clusters[chat_id][cluster_id].append(i)

        # Find cross-chat topics
        cross_chat_topics = []

        for cluster_id in range(n_clusters):
            # Check which chats have this topic
            chat_counts = {}
            for chat_id, chat_cluster_dict in chat_clusters.items():
                if cluster_id in chat_cluster_dict:
                    chat_counts[chat_id] = len(chat_cluster_dict[cluster_id])

            # If the topic appears in multiple chats
            if len(chat_counts) >= 2:
                # Get representative messages
                topic_messages = []
                for chat_id, count in chat_counts.items():
                    msg_indices = chat_clusters[chat_id][cluster_id][:3]  # Take first 3 messages
                    for idx in msg_indices:
                        topic_messages.append(
                            {
                                "chat_id": chat_id,
                                "text": points[idx].payload.get("text", ""),
                                "date": points[idx].payload.get("date", ""),
                            }
                        )

                # Get keywords for this topic
                topic_texts = [
                    points[i].payload.get("text", "")
                    for chat_id in chat_counts
                    for i in chat_clusters[chat_id][cluster_id]
                ]

                # Calculate top keywords
                if topic_texts:
                    try:
                        vectorizer = TfidfVectorizer(max_features=5)
                        tfidf_matrix = vectorizer.fit_transform(topic_texts)
                        feature_names = vectorizer.get_feature_names_out()
                        tfidf_scores = np.mean(tfidf_matrix.toarray(), axis=0)
                        top_indices = tfidf_scores.argsort()[-5:][::-1]
                        keywords = [feature_names[i] for i in top_indices]
                    except:
                        keywords = []
                else:
                    keywords = []

                cross_chat_topics.append(
                    {
                        "topic_id": int(cluster_id),
                        "chat_distribution": chat_counts,
                        "keywords": keywords,
                        "example_messages": topic_messages,
                        "total_messages": sum(chat_counts.values()),
                    }
                )

        # Sort by total message count
        return sorted(cross_chat_topics, key=lambda x: x["total_messages"], reverse=True)

    def analyze_topic_correlations(self, min_correlation: float = 0.5) -> List[Dict[str, Any]]:
        """Identify correlations between different topics in the chat data."""
        # Get all messages with timestamps
        result = self.qdrant.scroll(
            collection_name=self.messages_collection,
            limit=10000,
            with_vectors=True,
            with_payload=True,
        )

        if not result or not result[0]:
            return []

        points = result[0]
        vectors = [point.vector for point in points]

        # Convert all dates to datetime
        try:
            dates = [datetime.fromisoformat(point.payload["date"]) for point in points]
        except (ValueError, KeyError):
            return []

        # Get the date range
        min_date = min(dates)
        max_date = max(dates)

        # Create daily time bins
        delta = timedelta(days=1)
        bins = []
        current = min_date
        while current <= max_date:
            bin_end = current + delta
            bins.append((current, bin_end))
            current = bin_end

        # Cluster into topics
        n_clusters = min(15, len(vectors) // 20) if len(vectors) > 20 else 5
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(vectors)

        # Count daily occurrence of each topic
        daily_counts = {i: np.zeros(len(bins)) for i in range(n_clusters)}

        for i, (date, cluster_id) in enumerate(zip(dates, clusters)):
            for j, (bin_start, bin_end) in enumerate(bins):
                if bin_start <= date < bin_end:
                    daily_counts[cluster_id][j] += 1
                    break

        # Calculate correlation matrix between topics
        topic_corr = np.zeros((n_clusters, n_clusters))
        for i in range(n_clusters):
            for j in range(i, n_clusters):
                # Calculate Pearson correlation
                corr, _ = stats.pearsonr(daily_counts[i], daily_counts[j])
                topic_corr[i, j] = corr
                topic_corr[j, i] = corr  # Symmetric

        # Find significant correlations
        correlations = []
        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                if abs(topic_corr[i, j]) >= min_correlation:
                    # Get topic keywords
                    topic_i_texts = [points[k].payload.get("text", "") for k in range(len(points)) if clusters[k] == i]
                    topic_j_texts = [points[k].payload.get("text", "") for k in range(len(points)) if clusters[k] == j]

                    # Calculate keywords for both topics
                    keywords_i = self._get_topic_keywords(topic_i_texts, 3)
                    keywords_j = self._get_topic_keywords(topic_j_texts, 3)

                    correlations.append(
                        {
                            "topic_1": int(i),
                            "topic_2": int(j),
                            "correlation": float(topic_corr[i, j]),
                            "topic_1_keywords": keywords_i,
                            "topic_2_keywords": keywords_j,
                            "topic_1_message_count": sum(daily_counts[i]),
                            "topic_2_message_count": sum(daily_counts[j]),
                        }
                    )

        return sorted(correlations, key=lambda x: abs(x["correlation"]), reverse=True)

    def _get_topic_keywords(self, texts, n=5):
        """Helper method to extract keywords from topic texts."""
        if not texts:
            return []

        try:
            vectorizer = TfidfVectorizer(max_features=n)
            tfidf_matrix = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()
            tfidf_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            top_indices = tfidf_scores.argsort()[-n:][::-1]
            return [feature_names[i] for i in top_indices]
        except:
            return []

    def analyze_user_behavior_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Analyze user behavior patterns in messaging."""
        # Get all messages with user and timestamp information
        result = self.qdrant.scroll(
            collection_name=self.messages_collection,
            limit=10000,
            with_payload=True,
        )

        if not result or not result[0]:
            return {}

        points = result[0]

        # Group messages by user
        user_messages = defaultdict(list)
        for point in points:
            user_id = str(point.payload.get("from_id", "unknown"))
            try:
                date = datetime.fromisoformat(point.payload["date"])
                user_messages[user_id].append(
                    {
                        "date": date,
                        "text": point.payload.get("text", ""),
                        "reply_to": point.payload.get("reply_to_message_id"),
                    }
                )
            except (ValueError, KeyError):
                continue

        # Analyze patterns for each user
        user_patterns = {}
        for user_id, messages in user_messages.items():
            if len(messages) < 5:  # Skip users with too few messages
                continue

            # Sort messages by date
            messages.sort(key=lambda x: x["date"])

            # Calculate message frequency patterns
            hours = [msg["date"].hour for msg in messages]
            hour_counts = Counter(hours)

            # Calculate weekday patterns
            weekdays = [msg["date"].weekday() for msg in messages]
            weekday_counts = Counter(weekdays)

            # Calculate reply ratio
            reply_count = sum(1 for msg in messages if msg.get("reply_to"))
            reply_ratio = reply_count / len(messages) if messages else 0

            # Calculate average message length
            avg_length = np.mean([len(msg["text"]) for msg in messages]) if messages else 0

            # Calculate response time if user has reply chains
            reply_times = []
            for i, msg in enumerate(messages):
                if i > 0 and msg.get("reply_to"):
                    time_diff = (msg["date"] - messages[i - 1]["date"]).total_seconds()
                    if 0 < time_diff < 3600 * 24:  # Only include reasonable times (< 1 day)
                        reply_times.append(time_diff)

            avg_reply_time = np.mean(reply_times) if reply_times else None

            # Store user patterns
            user_patterns[user_id] = {
                "message_count": len(messages),
                "hourly_pattern": {str(h): count for h, count in hour_counts.items()},
                "weekday_pattern": {str(w): count for w, count in weekday_counts.items()},
                "reply_ratio": float(reply_ratio),
                "avg_message_length": float(avg_length),
                "avg_reply_time_seconds": float(avg_reply_time) if avg_reply_time is not None else None,
                "active_period_start": messages[0]["date"].isoformat(),
                "active_period_end": messages[-1]["date"].isoformat(),
            }

        return user_patterns

    def analyze_topic_lifecycle(self) -> Dict[int, Dict[str, Any]]:
        """Analyze the lifecycle of topics from emergence to decline."""
        # Get all messages with timestamps
        result = self.qdrant.scroll(
            collection_name=self.messages_collection,
            limit=10000,
            with_vectors=True,
            with_payload=True,
        )

        if not result or not result[0]:
            return {}

        points = result[0]
        vectors = [point.vector for point in points]

        # Attempt to parse dates
        date_point_map = []
        for point in points:
            try:
                date = datetime.fromisoformat(point.payload["date"])
                date_point_map.append((date, point))
            except (ValueError, KeyError):
                continue

        if not date_point_map:
            return {}

        # Sort by date
        date_point_map.sort(key=lambda x: x[0])

        # Get sorted dates and vectors
        sorted_dates = [item[0] for item in date_point_map]
        sorted_vectors = [point.vector for _, point in date_point_map]

        # Cluster into topics
        n_clusters = min(15, len(sorted_vectors) // 20) if len(sorted_vectors) > 20 else 5
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(sorted_vectors)

        # Create daily bins
        min_date = min(sorted_dates)
        max_date = max(sorted_dates)

        # Create daily bins
        delta = timedelta(days=1)
        current = min_date
        daily_bins = []

        while current <= max_date:
            next_day = current + delta
            daily_bins.append((current, next_day))
            current = next_day

        # Count daily occurrence of each topic
        daily_data = []
        for day_start, day_end in daily_bins:
            # Count each topic that day
            day_counts = {i: 0 for i in range(n_clusters)}
            day_total = 0

            for i, date in enumerate(sorted_dates):
                if day_start <= date < day_end:
                    cluster_id = clusters[i]
                    day_counts[cluster_id] += 1
                    day_total += 1

            if day_total > 0:  # Only include days with messages
                daily_data.append(
                    {"date": day_start.isoformat(), "total_messages": day_total, "topic_counts": day_counts}
                )

        # Analyze lifecycle for each topic
        topic_lifecycles = {}
        for topic_id in range(n_clusters):
            # Get topic keywords
            topic_texts = [
                date_point_map[i][1].payload.get("text", "")
                for i in range(len(date_point_map))
                if clusters[i] == topic_id
            ]
            keywords = self._get_topic_keywords(topic_texts, 5)

            # Extract this topic's daily data
            topic_daily_data = [
                {
                    "date": day["date"],
                    "count": day["topic_counts"][topic_id],
                    "percentage": (
                        day["topic_counts"][topic_id] / day["total_messages"] if day["total_messages"] > 0 else 0
                    ),
                }
                for day in daily_data
            ]

            # Analyze the lifecycle
            if topic_daily_data:
                lifecycle_analysis = self.analyze_single_topic_lifecycle(topic_daily_data)

                topic_lifecycles[topic_id] = {
                    "keywords": keywords,
                    "total_messages": sum(day["topic_counts"][topic_id] for day in daily_data),
                    "peak_date": lifecycle_analysis["peak_date"],
                    "emergence_date": lifecycle_analysis["emergence_date"],
                    "decline_date": lifecycle_analysis["decline_date"],
                    "lifecycle_stage": lifecycle_analysis["lifecycle_stage"],
                    "daily_data": topic_daily_data,
                }

        return topic_lifecycles

    def analyze_single_topic_lifecycle(self, daily_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the lifecycle of a single topic based on its daily activity."""
        if not daily_data:
            return {"peak_date": None, "emergence_date": None, "decline_date": None, "lifecycle_stage": "unknown"}

        # Find peak date
        peak_idx = max(range(len(daily_data)), key=lambda i: daily_data[i]["count"])
        peak_date = daily_data[peak_idx]["date"]

        # Find emergence (first date with significant activity)
        emergence_threshold = 0.1 * daily_data[peak_idx]["count"]
        emergence_idx = None
        for i in range(peak_idx, -1, -1):
            if daily_data[i]["count"] <= emergence_threshold:
                emergence_idx = i + 1
                break

        if emergence_idx is None:
            emergence_idx = 0

        emergence_date = daily_data[emergence_idx]["date"] if 0 <= emergence_idx < len(daily_data) else None

        # Find decline (last date with significant activity)
        decline_idx = None
        for i in range(peak_idx, len(daily_data)):
            if daily_data[i]["count"] <= emergence_threshold:
                decline_idx = i - 1
                break

        if decline_idx is None:
            decline_idx = len(daily_data) - 1

        decline_date = daily_data[decline_idx]["date"] if 0 <= decline_idx < len(daily_data) else None

        # Determine lifecycle stage
        latest_date = daily_data[-1]["date"]
        latest_count = daily_data[-1]["count"]

        if latest_count >= 0.8 * daily_data[peak_idx]["count"]:
            lifecycle_stage = "peak"
        elif daily_data[peak_idx]["date"] == latest_date:
            lifecycle_stage = "peak"
        elif peak_idx < len(daily_data) // 2:
            lifecycle_stage = "declining"
        elif latest_count < 0.2 * daily_data[peak_idx]["count"]:
            lifecycle_stage = "inactive"
        elif peak_idx > 0.7 * len(daily_data):
            lifecycle_stage = "emerging"
        else:
            lifecycle_stage = "active"

        return {
            "peak_date": peak_date,
            "emergence_date": emergence_date,
            "decline_date": decline_date,
            "lifecycle_stage": lifecycle_stage,
        }

    def compare_chats(self) -> List[Dict[str, Any]]:
        """Compare different chats based on topic distribution and messaging patterns."""
        # Get all messages
        result = self.qdrant.scroll(
            collection_name=self.messages_collection,
            limit=10000,
            with_vectors=True,
            with_payload=True,
        )

        if not result or not result[0]:
            return []

        points = result[0]
        vectors = [point.vector for point in points]

        # Extract chat IDs
        chat_ids = list(
            set(point.payload.get("chat_id") for point in points if point.payload.get("chat_id") is not None)
        )

        if len(chat_ids) <= 1:
            return []  # Need multiple chats to compare

        # Cluster all messages into topics
        n_clusters = min(20, len(vectors) // 10) if len(vectors) > 20 else 5
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(vectors)

        # Analyze each chat
        chat_analyses = {}
        for chat_id in chat_ids:
            # Filter messages for this chat
            chat_indices = [i for i, point in enumerate(points) if point.payload.get("chat_id") == chat_id]

            if not chat_indices:
                continue

            # Get topic distribution
            chat_clusters = [clusters[i] for i in chat_indices]
            topic_counts = Counter(chat_clusters)

            # Get active users
            user_counts = Counter(
                [points[i].payload.get("from_id") for i in chat_indices if points[i].payload.get("from_id") is not None]
            )

            # Calculate hourly activity pattern
            hours = []
            for i in chat_indices:
                try:
                    date = datetime.fromisoformat(points[i].payload["date"])
                    hours.append(date.hour)
                except (ValueError, KeyError):
                    continue

            hourly_pattern = Counter(hours)

            # Calculate message length stats
            message_lengths = [len(points[i].payload.get("text", "")) for i in chat_indices]
            avg_length = np.mean(message_lengths) if message_lengths else 0

            # Store analysis
            chat_analyses[chat_id] = {
                "message_count": len(chat_indices),
                "topic_distribution": {str(k): v for k, v in topic_counts.items()},
                "active_user_count": len(user_counts),
                "top_users": [str(user) for user, _ in user_counts.most_common(5)],
                "hourly_pattern": {str(h): count for h, count in hourly_pattern.items()},
                "avg_message_length": float(avg_length),
            }

        # Calculate similarities between chats
        chat_comparisons = []
        for i, chat1 in enumerate(chat_ids):
            if chat1 not in chat_analyses:
                continue

            for j in range(i + 1, len(chat_ids)):
                chat2 = chat_ids[j]
                if chat2 not in chat_analyses:
                    continue

                analysis1 = chat_analyses[chat1]
                analysis2 = chat_analyses[chat2]

                # Calculate topic distribution similarity
                topic_sim = self.calculate_chat_similarity(
                    analysis1["topic_distribution"], analysis2["topic_distribution"]
                )

                # Calculate hourly pattern similarity
                pattern_sim = self.calculate_pattern_similarity(
                    analysis1["hourly_pattern"], analysis2["hourly_pattern"]
                )

                # Overall similarity (weighted average)
                overall_sim = 0.6 * topic_sim + 0.4 * pattern_sim

                # Find common active users
                users1 = set(analysis1["top_users"])
                users2 = set(analysis2["top_users"])
                common_users = users1.intersection(users2)

                chat_comparisons.append(
                    {
                        "chat_id_1": str(chat1),
                        "chat_id_2": str(chat2),
                        "similarity": float(overall_sim),
                        "topic_similarity": float(topic_sim),
                        "activity_pattern_similarity": float(pattern_sim),
                        "common_active_users": len(common_users),
                        "common_user_list": list(common_users),
                    }
                )

        return sorted(chat_comparisons, key=lambda x: x["similarity"], reverse=True)

    def calculate_chat_similarity(self, metrics1: Dict[str, float], metrics2: Dict[str, float]) -> float:
        """Calculate similarity between chat topic distributions."""
        # Get all keys
        all_keys = set(metrics1.keys()).union(set(metrics2.keys()))

        # Create vectors with zeros for missing keys
        vec1 = np.array([float(metrics1.get(k, 0)) for k in all_keys])
        vec2 = np.array([float(metrics2.get(k, 0)) for k in all_keys])

        # Normalize
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 > 0:
            vec1 = vec1 / norm1
        if norm2 > 0:
            vec2 = vec2 / norm2

        # Cosine similarity
        dot_product = np.dot(vec1, vec2)
        return float(dot_product)

    def calculate_pattern_similarity(self, pattern1: Dict[int, int], pattern2: Dict[int, int]) -> float:
        """Calculate similarity between activity patterns."""
        # Ensure all hours are represented
        all_hours = {str(h) for h in range(24)}
        p1 = {h: pattern1.get(h, 0) for h in all_hours}
        p2 = {h: pattern2.get(h, 0) for h in all_hours}

        # Create vectors
        vec1 = np.array([float(p1[h]) for h in sorted(all_hours)])
        vec2 = np.array([float(p2[h]) for h in sorted(all_hours)])

        # Normalize
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 > 0:
            vec1 = vec1 / norm1
        if norm2 > 0:
            vec2 = vec2 / norm2

        # Cosine similarity
        dot_product = np.dot(vec1, vec2)
        return float(dot_product)

    def search_messages(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for messages semantically similar to a query."""
        # Get query embedding
        query_vector = self.get_embedding(query)

        # Perform semantic search
        search_result = self.qdrant.search(
            collection_name=self.messages_collection,
            query_vector=query_vector,
            limit=limit,
            with_payload=True,
        )

        if not search_result:
            return []

        # Format results
        results = []
        for scored_point in search_result:
            results.append(
                {
                    "text": scored_point.payload.get("text", ""),
                    "chat_id": scored_point.payload.get("chat_id"),
                    "message_id": scored_point.payload.get("message_id"),
                    "date": scored_point.payload.get("date"),
                    "from_id": scored_point.payload.get("from_id"),
                    "score": float(scored_point.score),
                }
            )

        return results

    def chat_search(self, chat_id: int, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for messages in a specific chat semantically similar to a query."""
        # Get query embedding
        query_vector = self.get_embedding(query)

        # Perform filtered semantic search
        search_result = self.qdrant.search(
            collection_name=self.messages_collection,
            query_vector=query_vector,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="chat_id",
                        match=models.MatchValue(value=chat_id),
                    )
                ]
            ),
            limit=limit,
            with_payload=True,
        )

        if not search_result:
            return []

        # Format results
        results = []
        for scored_point in search_result:
            results.append(
                {
                    "text": scored_point.payload.get("text", ""),
                    "message_id": scored_point.payload.get("message_id"),
                    "date": scored_point.payload.get("date"),
                    "from_id": scored_point.payload.get("from_id"),
                    "score": float(scored_point.score),
                }
            )

        return results

    def close(self) -> None:
        """Close connections to databases."""
        pass  # Qdrant Python client doesn't need explicit closing

    def analyze_topics(
        self,
        chat_id: Optional[int] = None,
        n_topics: int = 5,
        top_n_words: int = 10,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
    ) -> Dict:
        """
        Analyze topics from message vectors using clustering.

        Args:
            chat_id: Chat ID to analyze, or None for all chats
            n_topics: Number of topics to identify
            top_n_words: Number of top words to include per topic
            date_from: Start date for analysis
            date_to: End date for analysis

        Returns:
            Dictionary with topics and relevant information
        """
        self.logger.info(f"Analyzing topics for chat_id={chat_id} with {n_topics} topics")
        start_time = time.time()

        # Determine which collection to use
        collection_name = f"{self.chat_collection_prefix}{chat_id}" if chat_id else self.messages_collection

        try:
            # Check if collection exists
            self.qdrant.get_collection(collection_name)
        except Exception as e:
            self.logger.error(f"Collection {collection_name} not found: {str(e)}")
            raise ValueError(f"No data found for chat_id={chat_id}")

        # Build filter if date range is specified
        filter_obj = None
        if date_from or date_to:
            date_filter = {}
            if date_from:
                date_filter["gte"] = date_from.isoformat()
            if date_to:
                date_filter["lte"] = date_to.isoformat()

            filter_obj = models.Filter(must=[models.FieldCondition(key="date", range=models.Range(**date_filter))])

        # Get all points from collection
        self.logger.info("Retrieving message embeddings from vector store")
        retrieval_start = time.time()

        points = self.qdrant.scroll(
            collection_name=collection_name,
            limit=10000,  # Adjust based on expected data size
            with_vectors=True,
            with_payload=True,
            filter=filter_obj,
        )[
            0
        ]  # scroll returns (points, next_page_offset)

        retrieval_time = time.time() - retrieval_start
        self.logger.debug(f"Retrieved {len(points)} message embeddings in {retrieval_time:.2f} seconds")

        if not points:
            self.logger.warning("No messages found for topic analysis")
            return {"topics": []}

        # Extract vectors and texts
        vectors = np.array([point.vector for point in points])
        texts = [point.payload["text"] for point in points]

        # Perform KMeans clustering
        self.logger.info(f"Performing KMeans clustering with {n_topics} clusters")
        clustering_start = time.time()

        kmeans = KMeans(n_clusters=min(n_topics, len(vectors)), random_state=42)
        clusters = kmeans.fit_predict(vectors)

        clustering_time = time.time() - clustering_start
        self.logger.debug(f"KMeans clustering completed in {clustering_time:.2f} seconds")

        # Extract important words for each cluster using TF-IDF
        self.logger.info("Extracting important words for each topic using TF-IDF")
        tfidf_start = time.time()

        vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()

        tfidf_time = time.time() - tfidf_start
        self.logger.debug(f"TF-IDF processing completed in {tfidf_time:.2f} seconds")

        # Calculate cluster centers in original embedding space
        cluster_centers = kmeans.cluster_centers_

        # Find closest points to each center
        topics = []

        for cluster_idx in range(len(cluster_centers)):
            # Get messages in this cluster
            cluster_message_indices = np.where(clusters == cluster_idx)[0]
            cluster_messages = [texts[i] for i in cluster_message_indices]

            if not cluster_messages:
                continue

            # Get TF-IDF for this cluster
            cluster_tfidf = tfidf_matrix[cluster_message_indices]

            # Sum TF-IDF scores for each term in the cluster
            cluster_tfidf_sum = np.sum(cluster_tfidf, axis=0)

            # Get top terms
            top_term_indices = np.argsort(cluster_tfidf_sum.toarray()[0])[::-1][:top_n_words]
            top_terms = [feature_names[i] for i in top_term_indices]

            # Find representative messages (closest to center)
            distances = np.linalg.norm(vectors[cluster_message_indices] - cluster_centers[cluster_idx], axis=1)
            closest_idx = np.argsort(distances)[:3]  # Get 3 closest messages
            representative_messages = [cluster_messages[i] for i in closest_idx]

            # Calculate message count and date range
            message_count = len(cluster_message_indices)

            # Get dates if available
            dates = []
            for i in cluster_message_indices:
                date_str = points[i].payload.get("date")
                if date_str:
                    try:
                        if isinstance(date_str, str):
                            dates.append(datetime.fromisoformat(date_str))
                        else:
                            dates.append(date_str)
                    except (ValueError, TypeError):
                        pass

            date_range = None
            if dates:
                min_date = min(dates)
                max_date = max(dates)
                date_range = {
                    "start": min_date.isoformat() if hasattr(min_date, "isoformat") else str(min_date),
                    "end": max_date.isoformat() if hasattr(max_date, "isoformat") else str(max_date),
                }

            # Store topic information
            topics.append(
                {
                    "id": cluster_idx,
                    "terms": top_terms,
                    "messages": representative_messages,
                    "message_count": message_count,
                    "date_range": date_range,
                }
            )

        # Sort topics by message count
        topics.sort(key=lambda x: x["message_count"], reverse=True)

        total_time = time.time() - start_time
        self.logger.nice(
            "Topic analysis completed: identified %d topics from %d messages in %.2f seconds",
            len(topics),
            len(points),
            total_time,
        )

        for i, topic in enumerate(topics):
            terms_str = ", ".join(topic["terms"][:5])  # Show first 5 terms
            self.logger.nice("Topic %d: %s (%d messages)", i + 1, terms_str, topic["message_count"])

        return {"topics": topics}
