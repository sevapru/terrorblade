import pytest
from unittest.mock import patch, Mock, MagicMock
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans

from thoth.thoth.analyzer import ThothAnalyzer


class TestTopicAnalysis:
    """Tests for ThothAnalyzer topic analysis methods."""
    
    @pytest.fixture
    def mock_analyzer(self):
        """Create a mocked analyzer instance with initialized dependencies."""
        with patch('qdrant_client.QdrantClient') as mock_qdrant, \
             patch('duckdb.connect') as mock_duckdb, \
             patch('sentence_transformers.SentenceTransformer') as mock_transformer:
            
            # Setup mocks
            mock_connection = MagicMock()
            mock_duckdb.return_value = mock_connection
            mock_cursor = MagicMock()
            mock_connection.cursor.return_value = mock_cursor
            
            # Create analyzer with mocks
            analyzer = ThothAnalyzer(
                db_path="test.db",
                phone="+1234567890"
            )
            
            # Mock embedding model
            mock_model = MagicMock()
            mock_transformer.return_value = mock_model
            analyzer._embedding_model = mock_model
            mock_model.encode.return_value = np.random.rand(384)  # Random embedding vector
            
            # Mock Qdrant client
            analyzer.qdrant = MagicMock()
            
            yield analyzer
    
    @pytest.fixture
    def mock_chat_data(self):
        """Create mock chat data with message points."""
        return pd.DataFrame({
            'chat_id': [123, 123, 123, 456, 456],
            'message_id': [1, 2, 3, 1, 2],
            'from_id': [111, 222, 111, 333, 111],
            'date': pd.to_datetime([
                '2023-01-01 10:00:00', 
                '2023-01-01 10:05:00',
                '2023-01-01 10:10:00',
                '2023-01-02 11:00:00',
                '2023-01-02 11:05:00'
            ]),
            'text': [
                'Hello, this is about topic A', 
                'Yes, I agree about topic A', 
                'Let\'s discuss topic B also',
                'I prefer topic C', 
                'Topic A is the best'
            ],
            'reply_to_message_id': [None, 1, None, None, 1]
        })
    
    @pytest.fixture
    def mock_embeddings(self):
        """Create mock embeddings for the chat data."""
        # 5 mock messages, 3 dimensions for simplicity
        return np.array([
            [0.1, 0.8, 0.1],  # Topic A  (message 1)
            [0.2, 0.7, 0.1],  # Topic A  (message 2)
            [0.1, 0.2, 0.7],  # Topic B  (message 3)
            [0.8, 0.1, 0.1],  # Topic C  (message 4)
            [0.2, 0.7, 0.1],  # Topic A  (message 5)
        ])
    
    def test_find_common_topics(self, mock_analyzer, mock_chat_data, mock_embeddings):
        """Test find_common_topics method."""
        # Mock search results from Qdrant
        mock_analyzer.qdrant.scroll.return_value = (
            [
                MagicMock(payload={'message_id': i, 'chat_id': c, 'text': t, 'from_id': f})
                for i, c, t, f in zip(
                    mock_chat_data['message_id'], 
                    mock_chat_data['chat_id'],
                    mock_chat_data['text'],
                    mock_chat_data['from_id']
                )
            ],
            'scroll_id'
        )
        
        # Mock embedding fetching
        mock_analyzer.get_embedding = MagicMock(return_value=mock_embeddings)
        
        # Mock KMeans clustering
        with patch('sklearn.cluster.KMeans') as mock_kmeans_class:
            mock_kmeans = MagicMock()
            mock_kmeans_class.return_value = mock_kmeans
            mock_kmeans.fit_predict.return_value = np.array([0, 0, 1, 2, 0])  # 3 clusters
            mock_kmeans.cluster_centers_ = np.array([
                [0.2, 0.7, 0.1],  # Topic A center
                [0.1, 0.2, 0.7],  # Topic B center
                [0.8, 0.1, 0.1],  # Topic C center
            ])
            
            # Run method
            result = mock_analyzer.find_common_topics(num_topics=3)
            
            # Assertions
            assert len(result) == 3
            assert all(topic in result for topic in ['Topic 0', 'Topic 1', 'Topic 2'])
            assert 'count' in result['Topic 0']
            assert 'keywords' in result['Topic 0']
            assert 'messages' in result['Topic 0']
            assert result['Topic 0']['count'] == 3  # 3 messages in cluster 0
    
    def test_find_topic_relationships(self, mock_analyzer, mock_chat_data, mock_embeddings):
        """Test find_topic_relationships method."""
        # Mock common topics result
        mock_topics = {
            'Topic 0': {
                'count': 3,
                'keywords': ['topic', 'A'],
                'messages': [
                    {'message_id': 1, 'text': 'Hello, this is about topic A', 'from_id': 111},
                    {'message_id': 2, 'text': 'Yes, I agree about topic A', 'from_id': 222},
                    {'message_id': 5, 'text': 'Topic A is the best', 'from_id': 111}
                ]
            },
            'Topic 1': {
                'count': 1,
                'keywords': ['topic', 'B'],
                'messages': [
                    {'message_id': 3, 'text': 'Let\'s discuss topic B also', 'from_id': 111}
                ]
            },
            'Topic 2': {
                'count': 1,
                'keywords': ['topic', 'C'],
                'messages': [
                    {'message_id': 4, 'text': 'I prefer topic C', 'from_id': 333}
                ]
            }
        }
        
        # Mock the find_common_topics method
        mock_analyzer.find_common_topics = MagicMock(return_value=mock_topics)
        
        # Mock the search response from Qdrant
        mock_analyzer.qdrant.scroll.return_value = (
            [
                MagicMock(payload={
                    'message_id': m, 
                    'chat_id': c, 
                    'text': t, 
                    'from_id': f,
                    'reply_to_message_id': r
                })
                for m, c, t, f, r in zip(
                    mock_chat_data['message_id'],
                    mock_chat_data['chat_id'],
                    mock_chat_data['text'],
                    mock_chat_data['from_id'],
                    mock_chat_data['reply_to_message_id']
                )
            ],
            'scroll_id'
        )
        
        # Run method
        result = mock_analyzer.find_topic_relationships()
        
        # Assertions
        assert isinstance(result, dict)
        assert 'nodes' in result
        assert 'links' in result
        assert len(result['nodes']) == 3  # 3 topics
        assert all('id' in node for node in result['nodes'])
    
    def test_analyze_topic_sentiment(self, mock_analyzer, mock_chat_data):
        """Test analyze_topic_sentiment method."""
        # Mock common topics result
        mock_topics = {
            'Topic 0': {
                'count': 3,
                'keywords': ['topic', 'A'],
                'messages': [
                    {'message_id': 1, 'text': 'Great topic A', 'from_id': 111},
                    {'message_id': 2, 'text': 'I love topic A', 'from_id': 222},
                    {'message_id': 5, 'text': 'Topic A is awesome', 'from_id': 111}
                ]
            },
            'Topic 1': {
                'count': 1,
                'keywords': ['topic', 'B'],
                'messages': [
                    {'message_id': 3, 'text': 'I dislike topic B', 'from_id': 111}
                ]
            }
        }
        
        # Mock the find_common_topics method
        mock_analyzer.find_common_topics = MagicMock(return_value=mock_topics)
        
        # Mock textblob sentiment analysis
        with patch('textblob.TextBlob') as mock_textblob:
            # Set up sentiment returns: first 3 positive, last negative
            mock_tb1 = MagicMock()
            mock_tb1.sentiment.polarity = 0.8
            mock_tb2 = MagicMock()
            mock_tb2.sentiment.polarity = 0.9
            mock_tb3 = MagicMock()
            mock_tb3.sentiment.polarity = 0.7
            mock_tb4 = MagicMock()
            mock_tb4.sentiment.polarity = -0.6
            
            mock_textblob.side_effect = [mock_tb1, mock_tb2, mock_tb3, mock_tb4]
            
            # Run method
            result = mock_analyzer.analyze_topic_sentiment()
            
            # Assertions
            assert isinstance(result, dict)
            assert 'Topic 0' in result
            assert 'Topic 1' in result
            assert 'sentiment' in result['Topic 0']
            assert 'sentiment' in result['Topic 1']
            assert result['Topic 0']['sentiment'] > 0  # Positive for topic A
            assert result['Topic 1']['sentiment'] < 0  # Negative for topic B
    
    def test_analyze_topic_activity(self, mock_analyzer, mock_chat_data):
        """Test analyze_topic_activity method."""
        # Mock the common topics result
        mock_topics = {
            'Topic 0': {
                'count': 3,
                'keywords': ['topic', 'A'],
                'messages': [
                    {'message_id': 1, 'date': '2023-01-01 10:00:00', 'text': 'Great topic A', 'from_id': 111},
                    {'message_id': 2, 'date': '2023-01-01 10:05:00', 'text': 'I love topic A', 'from_id': 222},
                    {'message_id': 5, 'date': '2023-01-02 11:05:00', 'text': 'Topic A is awesome', 'from_id': 111}
                ]
            },
            'Topic 1': {
                'count': 1,
                'keywords': ['topic', 'B'],
                'messages': [
                    {'message_id': 3, 'date': '2023-01-01 10:10:00', 'text': 'Let\'s discuss topic B', 'from_id': 111}
                ]
            }
        }
        
        # Mock find_common_topics method
        mock_analyzer.find_common_topics = MagicMock(return_value=mock_topics)
        
        # Mock Qdrant search to include dates
        mock_analyzer.qdrant.scroll.return_value = (
            [
                MagicMock(payload={
                    'message_id': m, 
                    'chat_id': c, 
                    'text': t, 
                    'from_id': f,
                    'date': d.strftime('%Y-%m-%d %H:%M:%S')
                })
                for m, c, t, f, d in zip(
                    mock_chat_data['message_id'],
                    mock_chat_data['chat_id'],
                    mock_chat_data['text'],
                    mock_chat_data['from_id'],
                    mock_chat_data['date']
                )
            ],
            'scroll_id'
        )
        
        # Run method
        result = mock_analyzer.analyze_topic_activity()
        
        # Assertions
        assert isinstance(result, dict)
        assert 'Topic 0' in result
        assert 'Topic 1' in result
        assert 'activity' in result['Topic 0']
        assert 'activity' in result['Topic 1']
        assert isinstance(result['Topic 0']['activity'], dict)
        # Should have activity for both dates
        assert len(result['Topic 0']['activity']) == 2
    
    def test_predict_topic_trends(self, mock_analyzer, mock_chat_data, mock_embeddings):
        """Test predict_topic_trends method."""
        # Mock Qdrant search results
        mock_analyzer.qdrant.scroll.return_value = (
            [
                MagicMock(payload={
                    'message_id': m, 
                    'chat_id': c, 
                    'text': t, 
                    'from_id': f,
                    'date': d.strftime('%Y-%m-%d %H:%M:%S')
                })
                for m, c, t, f, d in zip(
                    mock_chat_data['message_id'],
                    mock_chat_data['chat_id'],
                    mock_chat_data['text'],
                    mock_chat_data['from_id'],
                    mock_chat_data['date']
                )
            ],
            'scroll_id'
        )
        
        # Mock embedding fetching
        mock_analyzer.get_embedding = MagicMock(return_value=mock_embeddings)
        
        # Mock KMeans clustering
        with patch('sklearn.cluster.KMeans') as mock_kmeans_class:
            mock_kmeans = MagicMock()
            mock_kmeans_class.return_value = mock_kmeans
            mock_kmeans.fit_predict.return_value = np.array([0, 0, 1, 2, 0])  # 3 clusters
            
            # Mock LinearRegression
            with patch('sklearn.linear_model.LinearRegression') as mock_lr_class:
                mock_lr = MagicMock()
                mock_lr_class.return_value = mock_lr
                mock_lr.fit.return_value = mock_lr
                mock_lr.predict.return_value = np.array([5, 3, 1])  # Predictions for topics
                
                # Run method
                result = mock_analyzer.predict_topic_trends(time_window_days=30, prediction_days=7)
                
                # Assertions
                assert isinstance(result, dict)
                assert len(result) == 3  # 3 topics
                assert all(f'Topic {i}' in result for i in range(3))
                assert 'prediction' in result['Topic 0']
                
                # Check prediction format
                assert isinstance(result['Topic 0']['prediction'], list)
                assert len(result['Topic 0']['prediction']) == 7  # 7 days of predictions
    
    def test_analyze_cross_chat_topics(self, mock_analyzer, mock_chat_data, mock_embeddings):
        """Test analyze_cross_chat_topics method."""
        # Mock Qdrant search results
        mock_analyzer.qdrant.scroll.return_value = (
            [
                MagicMock(payload={
                    'message_id': m, 
                    'chat_id': c, 
                    'text': t, 
                    'from_id': f
                })
                for m, c, t, f in zip(
                    mock_chat_data['message_id'],
                    mock_chat_data['chat_id'],
                    mock_chat_data['text'],
                    mock_chat_data['from_id']
                )
            ],
            'scroll_id'
        )
        
        # Mock embedding fetching
        mock_analyzer.get_embedding = MagicMock(return_value=mock_embeddings)
        
        # Mock KMeans clustering
        with patch('sklearn.cluster.KMeans') as mock_kmeans_class:
            mock_kmeans = MagicMock()
            mock_kmeans_class.return_value = mock_kmeans
            mock_kmeans.fit_predict.return_value = np.array([0, 0, 1, 2, 0])  # 3 clusters
            
            # Mock TfidfVectorizer for keywords
            with patch('sklearn.feature_extraction.text.TfidfVectorizer') as mock_tfidf_class:
                mock_tfidf = MagicMock()
                mock_tfidf_class.return_value = mock_tfidf
                mock_tfidf.fit_transform.return_value = MagicMock()
                mock_tfidf.get_feature_names_out.return_value = np.array(['topic', 'a', 'b', 'c'])
                
                # Run method
                result = mock_analyzer.analyze_cross_chat_topics()
                
                # Assertions
                assert isinstance(result, list)
                assert len(result) > 0
                assert 'topic_id' in result[0]
                assert 'distribution' in result[0]
                assert 'keywords' in result[0]
                
                # Check distribution format
                assert isinstance(result[0]['distribution'], dict)
                assert all(str(chat_id) in result[0]['distribution'] for chat_id in set(mock_chat_data['chat_id']))
    
    def test_analyze_topic_lifecycle(self, mock_analyzer, mock_chat_data):
        """Test analyze_topic_lifecycle method."""
        # Mock common topics result
        mock_topics = {
            'Topic 0': {
                'count': 3,
                'keywords': ['topic', 'A'],
                'messages': [
                    {'message_id': 1, 'date': '2023-01-01 10:00:00', 'text': 'Great topic A', 'from_id': 111},
                    {'message_id': 2, 'date': '2023-01-01 10:05:00', 'text': 'I love topic A', 'from_id': 222},
                    {'message_id': 5, 'date': '2023-01-02 11:05:00', 'text': 'Topic A is awesome', 'from_id': 111}
                ]
            },
            'Topic 1': {
                'count': 1,
                'keywords': ['topic', 'B'],
                'messages': [
                    {'message_id': 3, 'date': '2023-01-01 10:10:00', 'text': 'Let\'s discuss topic B', 'from_id': 111}
                ]
            }
        }
        
        # Mock find_common_topics method
        mock_analyzer.find_common_topics = MagicMock(return_value=mock_topics)
        
        # Run method
        result = mock_analyzer.analyze_topic_lifecycle()
        
        # Assertions
        assert isinstance(result, dict)
        assert 'Topic 0' in result
        assert 'Topic 1' in result
        assert 'daily_counts' in result['Topic 0']
        assert 'emergence_date' in result['Topic 0']
        assert 'peak_date' in result['Topic 0']
        assert 'decline_date' in result['Topic 0'] 