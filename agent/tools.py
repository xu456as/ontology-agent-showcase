"""openjiuwen Tools for the BA assistant.

Two tools are registered with the ReActAgent:
  * ``get_schema``  — return table/column metadata from the SQLite DB
  * ``run_sql``     — execute a (read-only) SQL statement and return rows

Both are backed by ``db.get_connection``. Ontology extraction is NOT done via
tools: the JiuwenMemory kernel infers entities/relations from the conversation
automatically (see ``memory/graph_store.py``).
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

import pandas as pd

from openjiuwen.core.foundation.tool.base import Tool, ToolCard

from db import get_connection, list_tables, table_schema


# Tool input schemas use JSON Schema dicts (openjiuwen serialises them
# verbatim into the LLM function-calling definitions — Pydantic *classes*
# are not JSON-serialisable and break the token counter).
_SCHEMA_INPUT = {
    "type": "object",
    "properties": {
        "table_name": {
            "type": "string",
            "description": "Optional specific table name. Leave empty to return schema of all tables.",
        },
    },
}

_RUN_SQL_INPUT = {
    "type": "object",
    "properties": {
        "sql": {
            "type": "string",
            "description": "A single SQL SELECT statement to execute.",
        },
    },
    "required": ["sql"],
}


class GetSchemaTool(Tool):
    """Return the schema (tables + columns) of the banking risk database."""

    async def invoke(self, inputs: Any, **kwargs: Any) -> str:
        table_name = _extract(inputs, "table_name", "")
        try:
            conn = get_connection()
            tables = [table_name] if table_name else list_tables(conn)
            out = []
            for t in tables:
                cols = table_schema(conn, t)
                col_lines = ", ".join(f"{c['name']}({c['type']})" for c in cols)
                out.append(f"TABLE {t} ({col_lines})")
            conn.close()
            return "\n".join(out)
        except Exception as exc:  # noqa: BLE001
            return f"ERROR retrieving schema: {exc}"

    async def stream(self, inputs: Any, **kwargs: Any):
        yield await self.invoke(inputs, **kwargs)


class RunSqlTool(Tool):
    """Execute a read-only SQL SELECT and return up to 200 rows as JSON."""

    async def invoke(self, inputs: Any, **kwargs: Any) -> str:
        sql = _extract(inputs, "sql", "").strip()
        if not sql:
            return "ERROR: empty sql"
        # safety: only SELECT / WITH allowed, no writes
        head = sql.lstrip().split(None, 1)[0].upper() if sql.lstrip() else ""
        if head not in {"SELECT", "WITH"}:
            return "ERROR: only SELECT/WITH statements are allowed."
        try:
            conn = get_connection()
            df = pd.read_sql_query(sql, conn)
            conn.close()
            rows = df.head(200).to_dict(orient="records")
            return json.dumps({"row_count": len(df), "rows": rows}, ensure_ascii=False, default=str)
        except (sqlite3.Error, pd.errors.DatabaseError) as exc:
            return f"SQL ERROR: {exc}"

    async def stream(self, inputs: Any, **kwargs: Any):
        yield await self.invoke(inputs, **kwargs)


def _extract(inputs: Any, key: str, default: Any) -> Any:
    if isinstance(inputs, dict):
        return inputs.get(key, default)
    return getattr(inputs, key, default)


def make_tool_cards() -> list[tuple[ToolCard, Tool]]:
    schema_card = ToolCard(
        name="get_schema",
        description="Get the database schema: table names and their columns/types. Call this first to understand available data.",
        input_params=_SCHEMA_INPUT,
    )
    sql_card = ToolCard(
        name="run_sql",
        description="Execute a read-only SQL SELECT/WITH statement and return result rows. Use this to answer data questions.",
        input_params=_RUN_SQL_INPUT,
    )
    return [
        (schema_card, GetSchemaTool(card=schema_card)),
        (sql_card, RunSqlTool(card=sql_card)),
    ]
