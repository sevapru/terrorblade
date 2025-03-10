from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import numpy as np
import pytest

from thoth.main import (
    analyze_cross_chat_topics,
    analyze_topic_activity,
    analyze_topic_influence,
    analyze_topic_sentiment,
    analyze_user_engagement,
    analyze_user_interactions,
    analyze_user_topics,
    build_topic_similarity_network,
    extract_topic_keywords,
    find_most_common_token,
    find_most_common_topic,
    find_related_topics,
    find_topic_evolution,
    predict_topic_trends,
)


@pytest.fixture
def test_db_path(tmp_path):
    """Create a test database with sample data."""
    db_path = tmp_path / "test.db"
    con = duckdb.connect(str(db_path))

    # Create tables
    con.execute(
        """
        CREATE TABLE messages (
            message_id BIGINT,
            chat_id BIGINT,
            date TIMESTAMP,
            text TEXT,
            "from" TEXT,
            reply_to_message_id BIGINT,
            embedding BLOB,
            sentiment_score DOUBLE,
            chat_name TEXT
        )
    """
    )

    con.execute(
        """
        CREATE TABLE message_clusters (
            message_id BIGINT,
            chat_id BIGINT,
            group_id INTEGER
        )
    """
    )

    # Insert sample data
    base_date = datetime.now() - timedelta(days=30)

    # Sample embeddings
    embedding1 = np.random.rand(384).astype(np.float32).tobytes()
    embedding2 = np.random.rand(384).astype(np.float32).tobytes()

    # Insert messages
    messages = [
        (1, 100, base_date, "Hello world", "user1", None, embedding1, 0.8, "chat1"),
        (2, 100, base_date + timedelta(hours=1), "Hi there", "user2", 1, embedding2, 0.5, "chat1"),
        (3, 101, base_date + timedelta(days=1), "Another chat", "user1", None, embedding1, 0.3, "chat2"),
    ]

    # Insert messages one at a time due to BLOB data
    for message in messages:
        con.execute(
            """
            INSERT INTO messages 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            [message],
        )

    # Insert clusters
    clusters = [
        (1, 100, 1),
        (2, 100, 1),
        (3, 101, 2),
    ]

    con.execute(
        """
        INSERT INTO message_clusters
        VALUES (?, ?, ?)
    """,
        clusters,
    )

    con.close()
    return db_path


def test_find_most_common_token(test_db_path):
    token, count = find_most_common_token(test_db_path)
    assert isinstance(token, str)
    assert isinstance(count, float)
    assert count > 0


def test_find_most_common_topic(test_db_path):
    topic, size = find_most_common_topic(test_db_path)
    assert isinstance(topic, str)
    assert isinstance(size, float)
    assert size > 0


def test_find_topic_evolution(test_db_path):
    evolution = find_topic_evolution(test_db_path)
    assert isinstance(evolution, list)
    if evolution:
        assert "cluster_id" in evolution[0]
        assert "size" in evolution[0]
        assert "duration" in evolution[0]


def test_find_related_topics(test_db_path):
    related = find_related_topics(test_db_path)
    assert isinstance(related, list)
    if related:
        assert len(related[0]) == 3
        assert isinstance(related[0][2], float)


def test_analyze_user_topics(test_db_path):
    topics = analyze_user_topics(test_db_path)
    assert isinstance(topics, dict)
    if topics:
        assert isinstance(list(topics.values())[0], list)


def test_analyze_topic_sentiment(test_db_path):
    sentiments = analyze_topic_sentiment(test_db_path)
    assert isinstance(sentiments, dict)
    if sentiments:
        assert isinstance(list(sentiments.values())[0], float)


def test_analyze_topic_activity(test_db_path):
    activity = analyze_topic_activity(test_db_path)
    assert isinstance(activity, dict)
    if activity:
        assert isinstance(list(activity.values())[0], dict)


def test_analyze_user_interactions(test_db_path):
    interactions = analyze_user_interactions(test_db_path)
    assert isinstance(interactions, list)
    if interactions:
        assert "user1" in interactions[0]
        assert "user2" in interactions[0]
        assert "interaction_count" in interactions[0]


def test_extract_topic_keywords(test_db_path):
    keywords = extract_topic_keywords(test_db_path)
    assert isinstance(keywords, dict)
    if keywords:
        assert isinstance(list(keywords.values())[0], list)


def test_analyze_topic_influence(test_db_path):
    influence = analyze_topic_influence(test_db_path)
    assert isinstance(influence, dict)
    if influence:
        assert isinstance(list(influence.values())[0], float)
        assert 0 <= list(influence.values())[0] <= 1


def test_analyze_user_engagement(test_db_path):
    engagement = analyze_user_engagement(test_db_path)
    assert isinstance(engagement, dict)
    if engagement:
        metrics = list(engagement.values())[0]
        assert "engagement_score" in metrics
        assert "topic_diversity" in metrics
        assert "activity_level" in metrics


def test_build_topic_similarity_network(test_db_path):
    network = build_topic_similarity_network(test_db_path)
    assert network.number_of_nodes() >= 0
    assert network.number_of_edges() >= 0
    if network.nodes:
        node = list(network.nodes())[0]
        assert "centrality" in network.nodes[node]
        assert "community" in network.nodes[node]


def test_predict_topic_trends(test_db_path):
    trends = predict_topic_trends(test_db_path)
    assert isinstance(trends, dict)
    if trends:
        predictions = list(trends.values())[0]
        assert isinstance(predictions, list)
        assert len(predictions[0]) == 2
        assert isinstance(predictions[0][0], datetime)


def test_analyze_cross_chat_topics(test_db_path):
    cross_chat = analyze_cross_chat_topics(test_db_path)
    assert isinstance(cross_chat, list)
    if cross_chat:
        assert "topic_id" in cross_chat[0]
        assert "num_chats" in cross_chat[0]
        assert "propagation_pattern" in cross_chat[0]
