"""Configuration utilities for terrorblade."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def get_project_root() -> Path:
    """Get the project root directory."""
    current_file = Path(__file__)
    return current_file.parent.parent.parent


def get_db_path(db_path: str = "auto") -> str:
    """
    Resolve the database path from various sources.

    Priority order:
    1. Explicit parameter (if not "auto")
    2. Environment variable DB_PATH
    3. Default fallback path

    Args:
        db_path: Explicit path or "auto" to use environment/default

    Returns:
        Resolved absolute path as string
    """
    # 1) explicit parameter if provided and not a sentinel
    if db_path and db_path.strip().lower() not in {"auto", "default"}:
        resolved = str(Path(db_path).expanduser().resolve())
        return resolved

    # 2) env var
    env_path = os.getenv("DB_PATH")
    if env_path:
        resolved = str(Path(env_path).expanduser().resolve())
        return resolved

    # 3) default in parent of project
    parent_dir = get_project_root().parent
    default_path = parent_dir / "telegram_data.db"
    return str(default_path)
