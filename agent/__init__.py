"""Agent module — Banking BA assistant built on openjiuwen agent-core.

The agent is a ReActAgent that owns two tools:
  * ``get_schema`` — introspect the SQLite DB schema (tables + columns)
  * ``run_sql``    — execute a read-only SQL query and return rows

When no LLM credentials are configured the agent falls back to a deterministic
schema-aware assistant so the demo always runs end-to-end.
"""

from .ba_agent import BAAgent, get_ba_agent

__all__ = ["BAAgent", "get_ba_agent"]
