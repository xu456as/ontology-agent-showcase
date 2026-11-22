"""Database connection helper for the Ontology AI Agent demo.

Loads configuration from the project-root `.env` file via python-dotenv.
`DB_PATH` can be overridden by the environment variable; falls back to the
default `./data/banking_risk.db` relative to the project root.
"""
import os
import sqlite3

from dotenv import load_dotenv

# Resolve project root (one level above this `db` package) and load `.env`.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))

# Default DB path relative to project root, overridable via env var.
_DEFAULT_DB = os.path.join(_PROJECT_ROOT, "data", "banking_risk.db")
DB_PATH = os.path.abspath(os.environ.get("DB_PATH", _DEFAULT_DB))


def get_connection() -> sqlite3.Connection:
    """Return a sqlite3 connection with row factory enabled."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def list_tables(conn: sqlite3.Connection) -> list[str]:
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    )
    return [r["name"] for r in cur.fetchall()]


def table_schema(conn: sqlite3.Connection, table: str) -> list[dict]:
    """Return column info for a table."""
    cur = conn.execute(f"PRAGMA table_info({table})")
    return [dict(r) for r in cur.fetchall()]


def table_row_count(conn: sqlite3.Connection, table: str) -> int:
    return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
