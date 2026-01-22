"""Configuration utilities for terrorblade."""

import contextlib
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def get_project_root() -> Path:
    """Get the project root directory."""
    current_file = Path(__file__)
    return current_file.parent.parent.parent


def is_remote_db_path(db_path: str) -> bool:
    """
    Check if the database path is a remote connection string.

    Supported remote formats:
    - MotherDuck: md:database_name or md:?motherduck_token=...
    - HTTP/HTTPS: http://... or https://...
    - S3: s3://bucket/path

    Args:
        db_path: Database path or connection string

    Returns:
        True if this is a remote connection string
    """
    if not db_path:
        return False

    remote_prefixes = ("md:", "http://", "https://", "s3://", "gcs://", "az://")
    return db_path.strip().lower().startswith(remote_prefixes)


def get_remote_db_config() -> dict[str, str | None]:
    """
    Get remote DuckDB configuration from environment variables.

    Environment variables:
    - DUCKDB_REMOTE_URL: Remote database URL (MotherDuck, HTTP, S3)
    - MOTHERDUCK_TOKEN: MotherDuck authentication token
    - AWS_ACCESS_KEY_ID: AWS credentials for S3
    - AWS_SECRET_ACCESS_KEY: AWS credentials for S3
    - AWS_REGION: AWS region for S3

    Returns:
        Dictionary with remote database configuration
    """
    return {
        "remote_url": os.getenv("DUCKDB_REMOTE_URL"),
        "motherduck_token": os.getenv("MOTHERDUCK_TOKEN"),
        "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
        "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
        "aws_region": os.getenv("AWS_REGION", "us-east-1"),
    }


def get_db_path(db_path: str = "auto") -> str:
    """
    Resolve the database path from various sources.

    Priority order:
    1. Explicit parameter (if not "auto") - supports remote URLs
    2. Environment variable DUCKDB_REMOTE_URL (for remote connections)
    3. Environment variable DB_PATH (for local files)
    4. Default fallback path

    Supported formats:
    - Local file: /path/to/database.db or relative/path.db
    - MotherDuck: md:database_name or md:?motherduck_token=TOKEN
    - HTTP/HTTPS: https://example.com/database.db
    - S3: s3://bucket/path/database.db

    Args:
        db_path: Explicit path or "auto" to use environment/default

    Returns:
        Resolved database path or connection string
    """
    # 1) explicit parameter if provided and not a sentinel
    if db_path and db_path.strip().lower() not in {"auto", "default"}:
        # If it's a remote URL, return as-is
        if is_remote_db_path(db_path):
            return db_path.strip()

        # Local path handling
        path_obj = Path(db_path).expanduser()
        # Only resolve if not already absolute to avoid symlink resolution on macOS
        resolved = str(
            path_obj.resolve() if not path_obj.is_absolute() else path_obj
        )
        return resolved

    # 2) check for remote URL in environment
    remote_url = os.getenv("DUCKDB_REMOTE_URL")
    if remote_url and remote_url.strip():
        return remote_url.strip()

    # 3) env var for local path
    env_path = os.getenv("DB_PATH")
    if env_path:
        # Check if env var contains a remote URL
        if is_remote_db_path(env_path):
            return env_path.strip()

        path_obj = Path(env_path).expanduser()
        # Only resolve if not already absolute to avoid symlink resolution on macOS
        resolved = str(
            path_obj.resolve() if not path_obj.is_absolute() else path_obj
        )
        return resolved

    # 4) default in parent of project
    parent_dir = get_project_root().parent
    default_path = parent_dir / "telegram_data.db"
    return str(default_path)


def configure_duckdb_remote(conn) -> None:
    """
    Configure DuckDB connection for remote access (S3, HTTP, MotherDuck).

    This function loads necessary extensions and sets credentials
    based on environment variables.

    Args:
        conn: DuckDB connection object
    """
    remote_config = get_remote_db_config()

    # Install and load httpfs for HTTP/S3 support
    with contextlib.suppress(Exception):
        conn.execute("INSTALL httpfs; LOAD httpfs;")

    # Configure AWS credentials if present
    if (
        remote_config["aws_access_key_id"]
        and remote_config["aws_secret_access_key"]
    ):
        conn.execute(
            f"SET s3_access_key_id='{remote_config['aws_access_key_id']}';"
        )
        conn.execute(
            f"SET s3_secret_access_key='{remote_config['aws_secret_access_key']}';"
        )
        conn.execute(f"SET s3_region='{remote_config['aws_region']}';")

    # Configure MotherDuck token if present
    if remote_config["motherduck_token"]:
        conn.execute(
            f"SET motherduck_token='{remote_config['motherduck_token']}';"
        )


def get_db_connection_kwargs(db_path: str) -> dict:
    """
    Get appropriate connection kwargs based on database path type.

    Args:
        db_path: Database path or remote URL

    Returns:
        Dictionary of kwargs to pass to duckdb.connect()
    """
    kwargs: dict = {}

    if is_remote_db_path(db_path):
        # Remote connections typically need read_only=False for MotherDuck
        # and may need config for extensions
        if db_path.startswith("md:"):
            # MotherDuck connection
            kwargs["config"] = {
                "motherduck_token": os.getenv("MOTHERDUCK_TOKEN", "")
            }
        # HTTP/S3 connections are read-only by nature
        elif db_path.startswith(("http://", "https://", "s3://")):
            kwargs["read_only"] = True

    return kwargs
