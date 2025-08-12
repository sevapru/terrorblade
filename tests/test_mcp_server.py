import asyncio
import os
import tempfile
from contextlib import asynccontextmanager

import pytest
from fastmcp import Client


def _payload(result):
    # Prefer structured data if present (FastMCP >=2.10)
    if hasattr(result, "data") and result.data is not None:
        return result.data
    if hasattr(result, "structured_content") and result.structured_content is not None:
        return result.structured_content
    # Fallback: content may be text; return empty structure
    return {}


def _setup_min_db(db_path: str, phone: str) -> None:
    import duckdb

    phone_clean = phone.replace("+", "")
    messages = f"messages_{phone_clean}"
    embeddings = f"chat_embeddings_{phone_clean}"
    clusters = f"message_clusters_{phone_clean}"
    chat_names = f"chat_names_{phone_clean}"
    user_names = f"user_names_{phone_clean}"

    con = duckdb.connect(db_path)
    try:
        con.execute(
            f"""
            CREATE TABLE {messages} (
                message_id BIGINT,
                chat_id BIGINT,
                from_id BIGINT,
                text VARCHAR,
                date TIMESTAMP
            );
            CREATE TABLE {embeddings} (
                message_id BIGINT,
                chat_id BIGINT,
                embeddings FLOAT[768]
            );
            CREATE TABLE {clusters} (
                message_id BIGINT,
                chat_id BIGINT,
                group_id INTEGER
            );
            CREATE TABLE {chat_names} (
                chat_id BIGINT,
                chat_name VARCHAR,
                first_seen TIMESTAMP,
                last_seen TIMESTAMP
            );
            CREATE TABLE {user_names} (
                from_id BIGINT,
                from_name VARCHAR,
                first_seen TIMESTAMP,
                last_seen TIMESTAMP
            );
            """
        )

        con.execute(
            f"INSERT INTO {messages} VALUES (1, 10, 100, 'hello world', '2024-01-01 00:00:00')"
        )
        con.execute(
            f"INSERT INTO {messages} VALUES (2, 10, 101, 'machine learning is fun', '2024-01-01 00:01:00')"
        )
        con.execute(
            f"INSERT INTO {messages} VALUES (3, 10, 102, 'vector databases are useful', '2024-01-01 00:02:00')"
        )
        con.execute(f"INSERT INTO {clusters} VALUES (1, 10, 1)")
        con.execute(f"INSERT INTO {clusters} VALUES (2, 10, 1)")
        con.execute(f"INSERT INTO {clusters} VALUES (3, 10, 1)")

        con.execute(
            f"INSERT INTO {chat_names} VALUES (10, 'Test Chat', '2024-01-01 00:00:00', '2024-02-01 00:00:00')"
        )
        con.execute(
            f"INSERT INTO {user_names} VALUES (100, 'Alice', '2024-01-01 00:00:00', '2024-02-01 00:00:00')"
        )
        con.execute(
            f"INSERT INTO {user_names} VALUES (101, 'Bob', '2024-01-01 00:00:00', '2024-02-01 00:00:00')"
        )
        con.execute(
            f"INSERT INTO {user_names} VALUES (102, 'Carol', '2024-01-01 00:00:00', '2024-02-01 00:00:00')"
        )

        z = [0.0] * 768
        con.execute(f"INSERT INTO {embeddings} VALUES (1, 10, ?)", [z])
        con.execute(f"INSERT INTO {embeddings} VALUES (2, 10, ?)", [z])
        con.execute(f"INSERT INTO {embeddings} VALUES (3, 10, ?)", [z])
    finally:
        con.close()


@asynccontextmanager
async def _in_memory_mcp_server():
    from terrorblade.mcp.server import mcp

    async with Client(mcp) as c:
        yield c


@pytest.mark.asyncio
async def test_tools_are_registered():
    from terrorblade.mcp.server import mcp

    async with Client(mcp) as c:
        tools = await c.list_tools()
        tool_names = sorted([t.name for t in tools])
        assert "vector_search" in tool_names
        assert "cluster_search" in tool_names
        assert "get_cluster" in tool_names
        assert "random_large_cluster" in tool_names


@pytest.mark.asyncio
async def test_prompts_are_registered():
    from terrorblade.mcp.server import mcp

    async with Client(mcp) as c:
        prompts = await c.list_prompts()
        prompt_names = sorted([p.name for p in prompts])
        assert "vector_search_template" in prompt_names
        assert "cluster_summary_template" in prompt_names


@pytest.mark.asyncio
async def test_vector_search_tool_executes(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        db_path = os.path.join(td, "test.db")
        phone = "+123456"
        _setup_min_db(db_path, phone)

        from terrorblade.mcp import server as srv
        monkeypatch.setattr(srv, "_encode_query", lambda text: [0.0] * 768)

        async with _in_memory_mcp_server() as client:
            result = await client.call_tool(
                "vector_search",
                {
                    "db_path": db_path,
                    "phone": phone,
                    "query": "machine learning",
                    "top_k": 5,
                    "similarity_threshold": 0.0,
                    "include_cluster_messages": True,
                },
            )
            data = _payload(result)
            assert isinstance(data, dict)
            assert "results" in data and isinstance(data["results"], list)
            assert "stats" in data and isinstance(data["stats"], dict)


@pytest.mark.asyncio
async def test_cluster_endpoints(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        db_path = os.path.join(td, "test.db")
        phone = "+123456"
        _setup_min_db(db_path, phone)

        from terrorblade.mcp import server as srv
        monkeypatch.setattr(srv, "_encode_query", lambda text: [0.0] * 768)

        async with _in_memory_mcp_server() as client:
            r1 = await client.call_tool(
                "random_large_cluster", {"db_path": db_path, "phone": phone, "min_size": 1}
            )
            rows = _payload(r1)
            assert isinstance(rows, list)

            r2 = await client.call_tool(
                "get_cluster", {"db_path": db_path, "phone": phone, "chat_id": 10, "group_id": 1}
            )
            rows2 = _payload(r2)
            assert isinstance(rows2, list)

            r3 = await client.call_tool(
                "cluster_search",
                {
                    "db_path": db_path,
                    "phone": phone,
                    "query": "vector database",
                    "top_k": 10,
                    "max_clusters": 5,
                },
            )
            clusters = _payload(r3)
            assert isinstance(clusters, list)
            assert len(clusters) >= 0 