from __future__ import annotations

import contextlib

# --- Logging setup and helpers ---
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import polars as pl
from fastmcp import FastMCP


def _get_project_root() -> Path:
    # /.../terrorblade/terrorblade/mcp/server.py -> project root is parents[2]
    return Path(__file__).resolve().parents[2]


def _get_logs_dir() -> Path:
    logs_dir = _get_project_root() / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir


def _setup_logging() -> logging.Logger:
    logger = logging.getLogger("terrorblade.mcp.server")
    if logger.handlers:  # Already configured
        return logger

    logger.setLevel(logging.INFO)

    logs_dir = _get_logs_dir()
    log_file = logs_dir / "mcp_server.log"

    # Rotating file handler
    file_handler = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(file_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info("Logging initialized. Log file: %s", str(log_file))
    return logger


LOGGER = _setup_logging()


def _resolve_db_path(db_path: str | None) -> str:
    """
    Resolve the DuckDB database path.

    Priority:
    1) Explicit non-empty db_path parameter
    2) Environment variable DUCKDB_PATH
    3) Parent directory of the project: ../telegram_data.db
    """
    # 1) explicit parameter if provided and not a sentinel
    if db_path and db_path.strip().lower() not in {"auto", "default"}:
        resolved = str(Path(db_path).expanduser().resolve())
        return resolved

    # 2) env var
    env_path = os.getenv("DUCKDB_PATH")
    if env_path:
        resolved = str(Path(env_path).expanduser().resolve())
        return resolved

    # 3) default in parent of project
    parent_dir = _get_project_root().parent
    default_path = parent_dir / "telegram_data.db"
    return str(default_path)


mcp = FastMCP("Terrorblade MCP Server")


def _encode_query(text: str) -> list[float]:
    from terrorblade.data.preprocessing.TextPreprocessor import TextPreprocessor

    preproc = TextPreprocessor()
    embedding = preproc.embeddings_model.encode(
        [text], convert_to_tensor=True, normalize_embeddings=True, device=preproc.device
    )
    return embedding.cpu().tolist()[0]


def _df_to_rows(df: pl.DataFrame) -> list[dict[str, Any]]:

    if df.is_empty():
        return []
    # Ensure datetimes are ISO strings for JSON serialization
    df = df.with_columns(*[pl.col(col).dt.to_string() for col in df.columns if df.schema[col] == pl.Datetime])
    return df.to_dicts()


@mcp.prompt(name="vector_search_template")
def vector_search_template(query: str) -> str:
    """
    Template instructing the assistant to use the vector_search tool to find relevant messages.
    """
    return (
        "You are assisting with semantic search over Telegram chats.\n"
        "Use the `vector_search` tool with the provided query to find the most relevant messages.\n"
        "Provide concise results with chat name, author, date, similarity, and a compact snippet.\n\n"
        f"Query: {query}\n"
        "Return the top findings and any notable clusters."
    )


@mcp.prompt(name="cluster_summary_template")
def cluster_summary_template(chat_name: str, snippet: str) -> str:
    """
    Template for summarizing a conversation cluster snippet.
    """
    return (
        "You are summarizing a conversation cluster from a Telegram chat.\n"
        f"Chat: {chat_name}\n"
        "Snippet (ordered by time, `>>>` marks the most relevant message):\n"
        f"{snippet}\n\n"
        "Write a brief summary covering: topic, participants, and outcome."
    )


@mcp.tool(name="vector_search")
def vector_search(
    phone: str,
    query: str,
    top_k: int = 10,
    chat_id: int | None = None,
    similarity_threshold: float = 0.0,
    include_cluster_messages: bool = True,
    db_path: str = "auto",
) -> dict[str, Any]:
    """
    Perform semantic vector search over Telegram messages for a user.

    - db_path: DuckDB database path (use "auto" or "default" to use the default path)
    - phone: User phone identifier (e.g., +123456789 or 123456789)
    - query: Natural language query
    - top_k: Number of results to return
    - chat_id: Optional chat filter
    - similarity_threshold: Only return results with similarity >= threshold
    - include_cluster_messages: If true, include a compact snippet of cluster context
    """
    if not isinstance(phone, str) or not phone:
        raise ValueError("phone must be a non-empty string")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if top_k <= 0 or top_k > 1000:
        raise ValueError("top_k must be in the range 1..1000")
    if similarity_threshold < 0.0 or similarity_threshold > 1.0:
        raise ValueError("similarity_threshold must be between 0.0 and 1.0")

    real_db_path = _resolve_db_path(db_path)
    if not Path(real_db_path).exists():
        LOGGER.warning("DuckDB not found at %s; proceeding (it may be created by downstream code)", real_db_path)

    LOGGER.info(
        "vector_search: phone=%s top_k=%d chat_id=%s threshold=%.3f include_cluster_messages=%s db_path=%s",
        phone,
        top_k,
        str(chat_id),
        similarity_threshold,
        include_cluster_messages,
        real_db_path,
    )

    from terrorblade.data.database.vector_store import VectorStore

    vs = VectorStore(db_path=real_db_path, phone=phone)

    # Best-effort index creation (idempotent)
    with contextlib.suppress(Exception):
        vs.create_hnsw_index()

    query_vec = _encode_query(query)

    results_df = vs.get_similar_messages_with_text(
        query_vector=query_vec,
        top_k=top_k,
        chat_id=chat_id,
        similarity_threshold=similarity_threshold,
        include_cluster_messages=include_cluster_messages,
    )

    rows = _df_to_rows(results_df)
    stats = vs.get_table_stats()

    LOGGER.info("vector_search: returned %d rows", len(rows))

    with contextlib.suppress(Exception):
        vs.close()

    return {"results": rows, "stats": stats}


@mcp.tool(name="cluster_search")
def cluster_search(
    phone: str,
    query: str,
    top_k: int = 50,
    max_clusters: int = 10,
    similarity_threshold: float = 0.0,
    db_path: str = "auto",
) -> list[dict[str, Any]]:
    """
    Find the most relevant conversation clusters for a query by aggregating top vector hits.

    Returns cluster summaries including best similarity, number of hits, chat name, and a snippet.
    """
    if max_clusters <= 0 or max_clusters > 1000:
        raise ValueError("max_clusters must be in the range 1..1000")

    real_db_path = _resolve_db_path(db_path)
    if not Path(real_db_path).exists():
        LOGGER.warning("DuckDB not found at %s; proceeding (it may be created by downstream code)", real_db_path)

    LOGGER.info(
        "cluster_search: phone=%s top_k=%d max_clusters=%d threshold=%.3f db_path=%s",
        phone,
        top_k,
        max_clusters,
        similarity_threshold,
        real_db_path,
    )

    from terrorblade.data.database.vector_store import VectorStore

    vs = VectorStore(db_path=real_db_path, phone=phone)

    with contextlib.suppress(Exception):
        vs.create_hnsw_index()

    query_vec = _encode_query(query)

    df = vs.get_similar_messages_with_text(
        query_vector=query_vec,
        top_k=top_k,
        chat_id=None,
        similarity_threshold=similarity_threshold,
        include_cluster_messages=True,
    )

    rows = _df_to_rows(df)

    # Aggregate by (group_id, chat_id), ignoring -1 (no cluster)
    clusters: dict[tuple[int, int], dict[str, Any]] = {}
    for r in rows:
        cid = int(r.get("cluster_id", -1))
        if cid < 0:
            continue
        chat = int(r["chat_id"]) if "chat_id" in r else -1
        key = (cid, chat)
        current = clusters.get(key, {
            "group_id": cid,
            "chat_id": chat,
            "chat_name": r.get("chat_name"),
            "best_similarity": float(r.get("similarity", 0.0)),
            "hits": 0,
            "snippet": r.get("text_preview") or r.get("text"),
        })
        current["hits"] += 1
        sim = float(r.get("similarity", 0.0))
        if sim > float(current["best_similarity"]):
            current["best_similarity"] = sim
            # Prefer snippet from the most similar message
            current["snippet"] = r.get("text_preview") or r.get("text")
            current["chat_name"] = r.get("chat_name")
        clusters[key] = current

    # Sort by best similarity, then hits
    ordered = sorted(clusters.values(), key=lambda x: (x["best_similarity"], x["hits"]), reverse=True)

    LOGGER.info("cluster_search: formed %d clusters from %d rows", len(ordered), len(rows))

    with contextlib.suppress(Exception):
        vs.close()

    return ordered[:max_clusters]


@mcp.tool(name="get_cluster")
def get_cluster(phone: str, chat_id: int, group_id: int, db_path: str = "auto") -> list[dict[str, Any]]:
    """
    Retrieve all messages for a specific cluster (group_id) within a chat.

    - db_path: DuckDB database path
    - phone: User phone identifier
    - chat_id: Chat ID
    - group_id: Cluster group ID
    """
    if chat_id <= 0 or group_id < 0:
        raise ValueError("chat_id must be > 0 and group_id must be >= 0")

    import duckdb
    import polars as pl

    real_db_path = _resolve_db_path(db_path)
    LOGGER.info(
        "get_cluster: phone=%s chat_id=%d group_id=%d db_path=%s",
        phone,
        chat_id,
        group_id,
        real_db_path,
    )

    phone_clean = phone.replace("+", "")
    messages_table = f"messages_{phone_clean}"
    clusters_table = f"message_clusters_{phone_clean}"

    sql = f"""
        SELECT m.message_id, m.chat_id, m.text, m.from_id, m.date
        FROM {messages_table} m
        JOIN {clusters_table} c ON m.message_id = c.message_id AND m.chat_id = c.chat_id
        WHERE c.group_id = ? AND c.chat_id = ?
        ORDER BY m.date
    """

    try:
        con = duckdb.connect(real_db_path, read_only=True)
    except Exception:
        LOGGER.exception("Failed to open DuckDB in read-only mode at %s", real_db_path)
        raise

    try:
        df = pl.from_arrow(con.execute(sql, [group_id, chat_id]).arrow())
    finally:
        con.close()

    return _df_to_rows(df)  # type: ignore[arg-type]


@mcp.tool(name="random_large_cluster")
def random_large_cluster(phone: str, min_size: int = 10, db_path: str = "auto") -> list[dict[str, Any]]:
    """
    Retrieve a random large cluster (size >= min_size) across all chats for a user.
    Returns the full set of messages for that cluster.
    """
    if min_size <= 0:
        raise ValueError("min_size must be > 0")

    real_db_path = _resolve_db_path(db_path)
    LOGGER.info(
        "random_large_cluster: phone=%s min_size=%d db_path=%s",
        phone,
        min_size,
        real_db_path,
    )

    from terrorblade.data.database.telegram_database import TelegramDatabase

    tb = TelegramDatabase(db_path=real_db_path, read_only=True)
    df_or_none = tb.get_random_large_cluster(phone=phone, min_size=min_size)

    with contextlib.suppress(Exception):
        tb.close()

    if df_or_none is None:
        return []

    # Convert result to rows
    try:
        import polars as pl

        if isinstance(df_or_none, pl.Series):
            return _df_to_rows(df_or_none.to_frame())
        return _df_to_rows(df_or_none)
    except Exception:
        # Fallback: try to treat as iterable of dict-like rows
        try:
            return list(df_or_none)  # type: ignore[arg-type]
        except Exception:
            return []


def main() -> None:
    LOGGER.info("Starting Terrorblade MCP Server ...")
    mcp.run()


if __name__ == "__main__":
    main()
