# Thoth Tests

This directory contains unit tests for the Thoth package.

## Test Structure

The tests are organized by functionality:

- `test_analyzer_init.py`: Tests for ThothAnalyzer initialization and configuration
- `test_data_import.py`: Tests for data import functionality
- `test_topic_analysis.py`: Tests for topic analysis methods
- `test_user_analysis.py`: Tests for user analysis methods
- `test_search.py`: Tests for search methods
- `conftest.py`: Shared fixtures and configuration

## Running Tests

To run the tests, navigate to the root directory of the package and run:

```bash
pytest
```

To run a specific test file:

```bash
pytest tests/test_analyzer_init.py
```

To run tests with verbose output:

```bash
pytest -v
```

## Test Coverage

To generate a test coverage report:

```bash
pytest --cov=thoth tests/
```

For a more detailed HTML coverage report:

```bash
pytest --cov=thoth --cov-report=html tests/
```

This will generate a directory named `htmlcov` containing the coverage report.

## Mocking

Most tests use mocking to avoid real database connections and external services. The primary mocks are:

- `qdrant_client.QdrantClient`: For vector database operations
- `sentence_transformers.SentenceTransformer`: For embedding generation
- `duckdb.connect`: For DuckDB database operations

These mocks are set up in the test fixtures to ensure consistent behavior across tests. 