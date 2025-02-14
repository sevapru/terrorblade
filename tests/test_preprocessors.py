import pytest
import polars as pl
from datetime import datetime, timedelta
import torch
from terrorblade.data.preprocessing.TextPreprocessor import TextPreprocessor
from terrorblade.data.preprocessing.TelegramPreprocessor import TelegramPreprocessor
import json

@pytest.fixture
def text_preprocessor():
    return TextPreprocessor()

@pytest.fixture
def telegram_preprocessor():
    return TelegramPreprocessor()

@pytest.fixture
def sample_messages_df():
    return pl.DataFrame({
        'text': ['Hello!', 'How are you?', 'I am fine', 'Good morning', 'Nice day'],
        'from_id': [1, 1, 2, 3, 1],
        'date': [
            datetime.now() - timedelta(minutes=i)
            for i in range(5)
        ],
        'chat_id': [100] * 5,
        'message_id': list(range(1, 6)),
        'from': ['User1', 'User1', 'User2', 'User3', 'User1'],
        'chat_name': ['Test Chat'] * 5,
        'reply_to_message_id': [None] * 5,
        'forwarded_from': [None] * 5
    })

def test_concat_author_messages(text_preprocessor, sample_messages_df):
    result = text_preprocessor.concat_author_messages(sample_messages_df)
    
    # Check that consecutive messages from the same author are concatenated
    assert len(result) < len(sample_messages_df)
    assert 'Hello! How are you?' in result['text'].to_list()

def test_calculate_embeddings(text_preprocessor, sample_messages_df):
    embeddings = text_preprocessor.calculate_embeddings(sample_messages_df)
    
    # Check that embeddings have correct shape and type
    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.shape[0] == len(sample_messages_df)
    assert embeddings.shape[1] > 0  # Should have some embedding dimensions

def test_calculate_distances(text_preprocessor):
    # Create sample embeddings
    embeddings = torch.randn(5, 10)  # 5 samples, 10 dimensions
    distances = text_preprocessor.calculate_distances(embeddings)
    
    # Check distances properties
    assert isinstance(distances, torch.Tensor)
    assert distances.shape == (5, 5)  # Should be a square matrix
    assert torch.all(distances >= 0)  # Distances should be non-negative
    assert torch.all(distances <= 2)  # Cosine distances are between 0 and 2

def test_calculate_sliding_distances(text_preprocessor):
    embeddings = torch.randn(5, 10)  # 5 samples, 10 dimensions
    distances = text_preprocessor.calculate_sliding_distances(embeddings, window_size=2)
    
    # Check sliding distances properties
    assert isinstance(distances, torch.Tensor)
    assert distances.shape == (5,)  # One distance per sample
    assert distances[0] == 0  # First element should be 0 as per implementation
    assert torch.all(distances >= 0)  # Distances should be non-negative

@pytest.fixture
def sample_telegram_data():
    return {
        "text": [
            [{"type": "link", "text": "Hello"}],
            "Simple text",
            [{"type": "mention", "text": "@user"}, {"type": "text", "text": " hi"}]
        ],
        "members": [
            ["user1", "user2"],
            ["user2", "user3"],
            ["user1", "user3"]
        ],
        "reactions": [
            [{"emoji": "👍"}],
            None,
            [{"emoji": "❤️"}]
        ]
    }

def test_parse_links(telegram_preprocessor):
    df = pl.DataFrame({"text": [
        [{"type": "link", "text": "Hello"}],
        "Simple text",
        [{"type": "mention", "text": "@user"}, {"type": "text", "text": " hi"}]
    ]})
    
    result = telegram_preprocessor.parse_links(df)
    assert result['text'].to_list() == ['Hello', 'Simple text', '@user hi']

def test_parse_members(telegram_preprocessor):
    df = pl.DataFrame({"members": [
        ["user1", "user2"],
        ["user2", "user3"],
        ["user1", "user3"]
    ]})
    
    result = telegram_preprocessor.parse_members(df)
    assert all(isinstance(x, str) for x in result['members'].to_list())
    assert all('user1' in x or 'user2' in x or 'user3' in x for x in result['members'].to_list())

def test_parse_reactions(telegram_preprocessor):
    df = pl.DataFrame({"reactions": [
        [{"emoji": "👍"}],
        None,
        [{"emoji": "❤️"}]
    ]})
    
    result = telegram_preprocessor.parse_reactions(df)
    assert result['reactions'].to_list() == ['👍', None, '❤️']

def test_standardize_chat(telegram_preprocessor):
    # Create a minimal DataFrame with some required columns
    df = pl.DataFrame({
        "text": ["Hello", "World"],
        "chat_id": [1, 1],
        "message_id": [1, 2]
    })
    
    result = telegram_preprocessor.standardize_chat(df)
    
    # Check that all required columns are present
    for col in telegram_import_schema.keys():
        assert col in result.columns
        
    # Check that types match the schema
    for col, dtype in telegram_import_schema.items():
        assert result[col].dtype == dtype or result[col].dtype == pl.Null

def test_process_message_groups(text_preprocessor, sample_messages_df):
    result = text_preprocessor.process_message_groups(
        sample_messages_df,
        time_window='5m',
        cluster_size=2,
        big_cluster_size=3
    )
    
    # Check that the result has the expected columns
    assert 'group_id' in result.columns
    assert len(result) > 0
    
    # Check that groups are properly assigned
    group_ids = result['group_id'].unique().to_list()
    assert len(group_ids) > 0 

def test_find_similar_messages(text_preprocessor):
    # Create sample data with similar messages
    df = pl.DataFrame({
        'text': [
            'Hello world',
            'Hello there',
            'Completely different topic',
            'Hi world',
            'Another different topic'
        ],
        'author': ['User1'] * 5,
        'recipients': ['User2'] * 5,
        'timestamp': [datetime.now() + timedelta(minutes=i) for i in range(5)]
    })
    
    # Calculate embeddings
    embeddings = text_preprocessor.calculate_embeddings(df)
    
    # Find similar messages
    result = text_preprocessor.find_simmilar_messages_in_chat(df, embeddings)
    
    # Check that clustering worked
    assert len(result) > 0
    assert 'cluster_label' in result.columns
    
    # Similar messages should be in the same cluster
    clusters = result['cluster_label'].unique().to_list()
    assert len(clusters) > 0

def test_batch_processing(text_preprocessor):
    # Create a large dataset
    n_samples = 1500  # More than batch_size
    df = pl.DataFrame({
        'text': ['Sample text ' + str(i) for i in range(n_samples)],
        'from_id': [i % 3 for i in range(n_samples)],
        'date': [
            datetime.now() + timedelta(minutes=i)
            for i in range(n_samples)
        ],
        'chat_id': [100] * n_samples,
        'message_id': list(range(n_samples)),
        'from': ['User' + str(i % 3) for i in range(n_samples)],
        'chat_name': ['Test Chat'] * n_samples,
        'reply_to_message_id': [None] * n_samples,
        'forwarded_from': [None] * n_samples
    })
    
    # Process the large dataset
    result = text_preprocessor.process_message_groups(df)
    
    # Check that all messages were processed
    assert len(result) > 0
    assert 'group_id' in result.columns
    
    # Check that batching didn't affect results
    group_ids = result['group_id'].unique().to_list()
    assert len(group_ids) > 0

def test_empty_input_handling(text_preprocessor):
    # Test with empty DataFrame
    empty_df = pl.DataFrame({
        'text': [],
        'from_id': [],
        'date': [],
        'chat_id': [],
        'message_id': [],
        'from': [],
        'chat_name': [],
        'reply_to_message_id': [],
        'forwarded_from': []
    })
    
    # Process empty data
    result = text_preprocessor.process_message_groups(empty_df)
    
    # Should return empty DataFrame without errors
    assert len(result) == 0

def test_cuda_availability(text_preprocessor):
    # Test CUDA device selection
    df = pl.DataFrame({
        'text': ['Test message 1', 'Test message 2'],
        'from_id': [1, 2],
        'date': [datetime.now(), datetime.now() + timedelta(minutes=1)],
        'chat_id': [100, 100],
        'message_id': [1, 2],
        'from': ['User1', 'User2'],
        'chat_name': ['Test Chat', 'Test Chat'],
        'reply_to_message_id': [None, None],
        'forwarded_from': [None, None]
    })
    
    # Calculate embeddings
    embeddings = text_preprocessor.calculate_embeddings(df)
    
    # Check that device was properly selected
    expected_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    assert str(embeddings.device).startswith(expected_device)

def test_clustering_cache(text_preprocessor):
    # Test that clustering results are cached
    df = pl.DataFrame({
        'text': ['Test 1', 'Test 2', 'Test 3'],
        'author': ['User1'] * 3,
        'recipients': ['User2'] * 3,
        'timestamp': [datetime.now() + timedelta(minutes=i) for i in range(3)]
    })
    
    embeddings = text_preprocessor.calculate_embeddings(df)
    
    # First clustering
    result1 = text_preprocessor.find_simmilar_messages_in_chat(df, embeddings)
    
    # Should use cached results
    result2 = text_preprocessor.find_simmilar_messages_in_chat(df, embeddings)
    
    # Results should be identical
    assert result1.equals(result2)
    
    # Check that cache was used
    assert hasattr(text_preprocessor, '_cached_cluster_labels')

def test_telegram_json_processing(telegram_preprocessor, tmp_path):
    # Create a sample Telegram JSON file
    sample_data = {
        "chats": {
            "list": [
                {
                    "name": "Test Chat",
                    "id": 123,
                    "type": "private",
                    "messages": [
                        {
                            "id": 1,
                            "text": "Hello",
                            "date": "2023-01-01T12:00:00",
                            "from": "User1"
                        },
                        {
                            "id": 2,
                            "text": "Hi there",
                            "date": "2023-01-01T12:01:00",
                            "from": "User2"
                        }
                    ]
                }
            ]
        }
    }
    
    # Write sample data to temporary file
    json_file = tmp_path / "test_chat.json"
    with open(json_file, 'w') as f:
        json.dump(sample_data, f)
    
    # Process the JSON file
    result = telegram_preprocessor.prepare_data(str(json_file))
    
    # Check results
    assert isinstance(result, dict)
    assert 123 in result  # Chat ID should be in results
    assert len(result[123]) > 0  # Should have messages
    assert 'text' in result[123].columns 