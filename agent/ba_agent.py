"""Banking BA assistant — openjiuwen ReActAgent with deterministic fallback.

The agent is a BA's apprentice: it reads the DBMS schema from the conversation
and generates business SQL.  When an LLM is configured (OPENAI_API_KEY etc.)
the real ``openjiuwen.core.single_agent.agents.react_agent.ReActAgent`` is
used; otherwise a deterministic, schema-aware fallback assistant answers so
the demo always runs.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import threading
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv

from .tools import make_tool_cards

load_dotenv()

SYSTEM_PROMPT = (
    "You are the assistant of a corporate banking Business Analyst (BA). "
    "Your job is to translate the BA's natural-language business questions "
    "into correct SQL queries against the bank's risk-monitoring database.\n\n"
    "Workflow:\n"
    "1. Call get_schema() to understand available tables and columns.\n"
    "2. Map the BA's business terms to columns. Business calibers the BA has "
    "taught you live in your long-term memory (JiuwenMemory), which is built "
    "automatically from your conversations — rely on it as background "
    "knowledge when interpreting terms.\n"
    "3. Call run_sql(sql) with a single SELECT/WITH statement.\n"
    "4. Explain the result to the BA in plain business language and cite the "
    "caliber you used.\n\n"
    "Conversational behavior:\n"
    "- This is a MULTI-TURN conversation. Use the conversation history to "
    "resolve follow-up questions, pronouns ('it', 'that customer', 'the same "
    "group') and references to earlier results or earlier SQL.\n"
    "- When the BA asks a short follow-up, prefer continuing the previous "
    "topic instead of restarting from scratch.\n"
    "- If a reference is ambiguous, state your interpretation in one short "
    "sentence before answering.\n\n"
    "Only output SELECT/WITH SQL, never modify data."
)


@dataclass
class AgentTurn:
    """One turn of the BA <-> agent conversation."""

    user: str
    assistant: str
    sql: str = ""
    result_preview: str = ""
    tool_trace: list[str] = field(default_factory=list)


class BAAgent:
    """Facade over the openjiuwen ReActAgent (or a deterministic fallback).

    Multi-turn conversation: the real LLM path keeps history per
    ``conversation_id`` through openjiuwen's session checkpointer (verified:
    follow-up turns see earlier turns). The deterministic fallback inherits
    context from the most recent turns itself.
    """

    def __init__(self) -> None:
        self._agent = None
        self._use_real_llm = False
        self._history: list[AgentTurn] = []
        self._conversation_id = "ba-demo-1"
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_lock = threading.Lock()
        self._init()

    # ------------------------------------------------------------------ init
    def _init(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY", "")
        api_base = os.getenv("OPENAI_API_BASE", "")
        model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        provider = os.getenv("LLM_PROVIDER", "openai")

        if not api_key or not api_base:
            self._use_real_llm = False
            return

        try:
            # Build the ReActAgent ON the agent's persistent event loop: its
            # async model client binds to the loop it was created on, and a
            # loop created on the main thread (no running loop) is never
            # run — later turns would fail with "Event loop is closed".
            loop = self._get_loop()
            future = asyncio.run_coroutine_threadsafe(
                self._build_agent(provider, api_key, api_base, model), loop
            )
            self._agent, self._use_real_llm = future.result(timeout=120)
        except Exception:  # noqa: BLE001
            self._use_real_llm = False

    async def _build_agent(
        self, provider: str, api_key: str, api_base: str, model: str
    ) -> tuple[Any, bool]:
        from openjiuwen.core.single_agent.agents.react_agent import (
            ReActAgent,
            ReActAgentConfig,
        )
        from openjiuwen.core.single_agent.schema.agent_card import AgentCard

        card = AgentCard(
            name="ba_assistant",
            description="Banking BA assistant that translates business questions into SQL.",
        )
        agent = ReActAgent(card=card)
        for tool_card, tool in make_tool_cards():
            agent._ability_manager.add_ability(tool_card, tool)

        cfg = (
            ReActAgentConfig()
            .configure_model_client(provider, api_key, api_base, model)
            .configure_prompt_template([{"role": "system", "content": SYSTEM_PROMPT}])
            .configure_context_engine(default_window_round_num=10)
            .configure_max_iterations(8)
        )
        agent.configure(cfg)
        return agent, True

    # ----------------------------------------------------------- conversation
    def new_conversation(self) -> None:
        """Start a fresh conversation: rotate the session id (LLM path) and
        clear the in-process history (fallback path)."""
        import uuid

        self._conversation_id = f"ba-demo-{uuid.uuid4().hex[:8]}"
        self._history.clear()

    @property
    def turns(self) -> int:
        return len(self._history)

    # ----------------------------------------------------------------- chat
    def _get_loop(self) -> asyncio.AbstractEventLoop:
        """Return the agent's long-lived background event loop.

        The ReActAgent and its OpenAI-compatible async client bind resources
        to the event loop they were first used on. Calling ``asyncio.run``
        per turn (new loop each time) fails on the second turn with
        ``RuntimeError: Event loop is closed`` — so every turn is submitted
        to one persistent loop instead.
        """
        if self._loop is not None and not self._loop.is_closed():
            return self._loop
        with self._loop_lock:
            if self._loop is None or self._loop.is_closed():
                self._loop = asyncio.new_event_loop()
                threading.Thread(
                    target=self._loop.run_forever,
                    daemon=True,
                    name="ba-agent-loop",
                ).start()
        return self._loop

    def chat(self, query: str) -> AgentTurn:
        """Synchronous entry point used by Streamlit."""
        loop = self._get_loop()
        turn = asyncio.run_coroutine_threadsafe(self._achat(query), loop).result(
            timeout=300
        )
        # Feed the BA's words to the JiuwenMemory kernel: its construction
        # pipeline infers entities/relations from the dialogue automatically.
        # Runs on the caller thread — the kernel's sync add() manages its own
        # event loop internally (must not run on the agent loop above).
        try:
            from memory.graph_store import get_bank_memory

            # Recent turns give the terminology extractor context for
            # pronoun resolution ("it", "that group", ...) in follow-ups.
            recent = "\n".join(
                f"BA: {t.user}\nAssistant: {t.assistant[:200]}"
                for t in self._history[-2:]
            )
            note = get_bank_memory().remember(query, context=recent)
            turn.tool_trace.append(f"[memory] remember(user turn) -> {note}")
        except Exception as exc:  # noqa: BLE001
            turn.tool_trace.append(f"[memory] remember failed: {exc}")
        self._history.append(turn)
        return turn

    async def _achat(self, query: str) -> AgentTurn:
        if self._use_real_llm and self._agent is not None:
            return await self._chat_with_llm(query)
        return self._chat_fallback(query)

    # ------------------------------------------------------------- real LLM
    async def _chat_with_llm(self, query: str) -> AgentTurn:
        try:
            result = await self._agent.invoke(
                {"query": query, "conversation_id": self._conversation_id}
            )
            output = (result or {}).get("output", "") if isinstance(result, dict) else str(result)
            return AgentTurn(user=query, assistant=output or "(no response)")
        except Exception as exc:  # noqa: BLE001
            return AgentTurn(
                user=query,
                assistant=f"LLM agent failed: {exc}. Falling back to schema-aware assistant.",
            )

    # ----------------------------------------------------------- fallback
    def _chat_fallback(self, query: str) -> AgentTurn:
        """Deterministic multi-turn assistant: context -> intent -> SQL ->
        execute -> explain.

        Keeps the demo fully functional without an LLM, including follow-up
        turns: short/pronoun questions inherit the previous turn's intent
        (and its SQL) instead of failing to match. Ontology extraction is
        handled by the JiuwenMemory kernel in ``chat()`` for every turn,
        regardless of which path produced the answer.
        """
        from db import get_connection

        q = query.lower()
        trace: list[str] = ["[tool] get_schema(table_name='') -> banking risk schema (6 tables)"]

        last = self._history[-1] if self._history else None
        sql = self._generate_sql(q)
        is_followup = False

        if sql is None and self._is_followup(query, last):
            # Follow-up turn ("what about its guarantors?"): continue the
            # previous topic by reusing its SQL.
            sql = last.sql if last else None
            is_followup = sql is not None
            trace.append("[context] follow-up detected -> continuing previous query intent")
        elif sql is None and last is not None:
            # Fall back to intent matching over the recent conversation
            # context (e.g. "and the groups?" after a warning-pool question).
            ctx = " ".join(t.user.lower() for t in self._history[-3:])
            sql = self._generate_sql(q + " " + ctx)
            if sql is not None:
                trace.append("[context] matched intent against recent conversation history")

        if sql is None:
            return AgentTurn(
                user=query,
                assistant=(
                    "I couldn't map your request to a known query pattern. "
                    "Try asking about the warning pool, customer exposure, "
                    "group exposure ranking, or guarantee-chain risk — or ask "
                    "a short follow-up about the previous result."
                ),
                tool_trace=trace,
            )

        trace.append(f"[tool] run_sql(sql='{sql[:80]}...')")
        preview = ""
        try:
            import pandas as pd

            conn = get_connection()
            df = pd.read_sql_query(sql, conn)
            conn.close()
            preview = df.head(10).to_string(index=False)
            row_count = len(df)
        except Exception as exc:  # noqa: BLE001
            return AgentTurn(
                user=query,
                assistant=f"SQL execution failed: {exc}",
                sql=sql,
                tool_trace=trace,
            )

        explanation = self._explain(q, row_count, is_followup=is_followup, last=last)
        return AgentTurn(
            user=query,
            assistant=explanation,
            sql=sql,
            result_preview=preview,
            tool_trace=trace,
        )

    @staticmethod
    def _is_followup(query: str, last: "AgentTurn | None") -> bool:
        """Heuristic follow-up detection: short questions or pronoun-bearing
        questions continue the previous topic."""
        if last is None:
            return False
        q = query.lower().strip()
        markers = (
            "it ", "its ", "it'", "them", "they ", "that ", "this ", "the same",
            "what about", "how about", "and the", "any ", "more ",
            "它", "它们", "上面", "刚才", "继续", "还有", "呢", "那些",
        )
        if any(m in q for m in markers):
            return True
        return len(q) < 30 and last.sql != ""

    def _explain(self, q: str, row_count: int, *, is_followup: bool = False,
                 last: "AgentTurn | None" = None) -> str:
        prefix = ""
        if is_followup and last is not None:
            topic = (last.user or "").strip().rstrip("?.!")
            prefix = f"Following up on \"{topic[:60]}\" — "

        if any(k in q for k in ("warning pool", "预警", "risky")):
            return (
                prefix
                + f"Found {row_count} customers in the warning pool. A customer enters the pool "
                "if it is overdue 90+ days, or if its guarantor is overdue 90+ days. "
                "Exposure is calculated as outstanding on-balance loans plus active off-balance "
                "bank guarantees (excluding settled/expired records), per your caliber."
            )
        if any(k in q for k in ("group exposure", "group risk", "top group")):
            return (
                prefix
                + f"Here are the groups ranked by total risk exposure. "
                "Exposure = outstanding loans + active guarantees, summed across all group members."
            )
        if any(k in q for k in ("exposure", "敞口")):
            return (
                prefix
                + f"Exposure breakdown for {row_count} customers. Each customer's exposure "
                "= on-balance outstanding loans + off-balance active guarantees."
            )
        if any(k in q for k in ("guarant", "担保")):
            return (
                prefix
                + f"Guarantee-chain risk view ({row_count} rows). When a guarantor falls into "
                "the overdue-90+ set, its guaranteed customers are flagged for cross-risk."
            )
        if any(k in q for k in ("core enterprise", "核心")):
            return (
                prefix
                + f"Core enterprises ({row_count}): a group is core when it has a group credit "
                "host unit AND at least 3 member units."
            )
        return prefix + f"Continuing the previous query — it returned {row_count} rows this time."

    # ------------------------------------------------------- SQL generation
    def _generate_sql(self, q: str) -> str | None:
        # Warning pool customers with exposure + guarantor cross-risk
        if any(k in q for k in ("warning pool", "预警", "risky customer", "risk list")):
            return _WARNING_POOL_SQL

        # Group-level exposure ranking
        if any(k in q for k in ("group exposure", "group risk", "top group", "largest exposure", "集团敞口")):
            return _GROUP_EXPOSURE_SQL

        # Customer exposure (on vs off balance)
        if any(k in q for k in ("exposure", "敞口", "risk exposure")):
            return _CUSTOMER_EXPOSURE_SQL

        # Guarantee chain risk
        if any(k in q for k in ("guarant", "担保", "guarantee chain", "cross risk")):
            return _GUARANTEE_CHAIN_SQL

        # Core enterprises
        if any(k in q for k in ("core enterprise", "核心企业", "core customer")):
            return _CORE_ENTERPRISE_SQL

        # No direct match — the caller decides how to continue (follow-up
        # inheritance, context matching, or a friendly clarification).
        return None


# ---------------------------------------------------------------------------
# Pre-built SQL patterns (used by the deterministic fallback)
# ---------------------------------------------------------------------------

_WARNING_POOL_SQL = """WITH overdue_set AS (
  SELECT cust_id FROM overdue WHERE overdue_days >= 90
),
warning_pool AS (
  SELECT cust_id FROM overdue_set
  UNION
  SELECT g.cust_id FROM guarantee_relation g
  JOIN overdue_set o ON g.guarantor_cust_id = o.cust_id
),
exposure AS (
  SELECT cust_id, SUM(balance) AS amt FROM loan
   WHERE loan_status='OUTSTANDING' GROUP BY cust_id
  UNION ALL
  SELECT cust_id, SUM(balance) FROM bank_guarantee
   WHERE g_status='ACTIVE' GROUP BY cust_id
)
SELECT c.cust_name AS customer, gm.group_name AS "group", c.host_branch AS branch,
       COALESCE(SUM(e.amt),0) AS exposure_wan,
       g.guarantor_cust_id AS guarantor,
       CASE WHEN g.guarantor_cust_id IN (SELECT cust_id FROM warning_pool)
            THEN 'guarantor also in warning pool' ELSE '' END AS cross_risk
FROM warning_pool w
JOIN ecif_customer c ON w.cust_id=c.cust_id
LEFT JOIN group_metadata gm ON c.group_id=gm.group_id
LEFT JOIN exposure e ON c.cust_id=e.cust_id
LEFT JOIN guarantee_relation g ON c.cust_id=g.cust_id
GROUP BY c.cust_name, gm.group_name, c.host_branch, g.guarantor_cust_id
ORDER BY exposure_wan DESC"""

_GROUP_EXPOSURE_SQL = """WITH exposure AS (
  SELECT cust_id, SUM(balance) AS amt FROM loan WHERE loan_status='OUTSTANDING' GROUP BY cust_id
  UNION ALL
  SELECT cust_id, SUM(balance) FROM bank_guarantee WHERE g_status='ACTIVE' GROUP BY cust_id
)
SELECT gm.group_name AS "group",
       COUNT(DISTINCT c.cust_id) AS members,
       COALESCE(SUM(e.amt),0) AS total_exposure_wan
FROM group_metadata gm
LEFT JOIN ecif_customer c ON c.group_id=gm.group_id
LEFT JOIN exposure e ON e.cust_id=c.cust_id
GROUP BY gm.group_name
ORDER BY total_exposure_wan DESC"""

_CUSTOMER_EXPOSURE_SQL = """WITH on_bal AS (
  SELECT cust_id, SUM(balance) AS on_balance FROM loan WHERE loan_status='OUTSTANDING' GROUP BY cust_id
), off_bal AS (
  SELECT cust_id, SUM(balance) AS off_balance FROM bank_guarantee WHERE g_status='ACTIVE' GROUP BY cust_id
)
SELECT c.cust_name,
       COALESCE(o.on_balance,0) AS on_balance_loan_wan,
       COALESCE(f.off_balance,0) AS off_balance_guarantee_wan,
       COALESCE(o.on_balance,0)+COALESCE(f.off_balance,0) AS total_exposure_wan
FROM ecif_customer c
LEFT JOIN on_bal o ON c.cust_id=o.cust_id
LEFT JOIN off_bal f ON c.cust_id=f.cust_id
ORDER BY total_exposure_wan DESC"""

_GUARANTEE_CHAIN_SQL = """WITH overdue_set AS (
  SELECT cust_id FROM overdue WHERE overdue_days >= 90
)
SELECT c.cust_name AS guaranteed_customer,
       g2.cust_name AS guarantor,
       gr.guarantee_amt AS guarantee_amt_wan,
       CASE WHEN gr.guarantor_cust_id IN (SELECT cust_id FROM overdue_set)
            THEN 'guarantor overdue 90+' ELSE 'ok' END AS guarantor_status
FROM guarantee_relation gr
JOIN ecif_customer c ON gr.cust_id=c.cust_id
JOIN ecif_customer g2 ON gr.guarantor_cust_id=g2.cust_id
ORDER BY guarantor_status DESC, guarantee_amt_wan DESC"""

_CORE_ENTERPRISE_SQL = """SELECT gm.group_name, gm.group_type,
       COUNT(c.cust_id) AS member_count,
       MAX(CASE WHEN c.is_group_host='Y' THEN 1 ELSE 0 END) AS has_host,
       CASE WHEN MAX(CASE WHEN c.is_group_host='Y' THEN 1 ELSE 0 END)=1
                 AND COUNT(c.cust_id)>=3 THEN 'Y' ELSE 'N' END AS is_core_enterprise
FROM group_metadata gm
LEFT JOIN ecif_customer c ON c.group_id=gm.group_id
GROUP BY gm.group_name, gm.group_type
ORDER BY is_core_enterprise DESC, member_count DESC"""


# -- module-level singleton -------------------------------------------------

_ba_agent: BAAgent | None = None


def get_ba_agent() -> BAAgent:
    global _ba_agent
    if _ba_agent is None:
        _ba_agent = BAAgent()
    return _ba_agent
