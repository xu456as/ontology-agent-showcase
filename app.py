"""
Ontology AI Agent — Corporate Credit Post-Loan Risk Monitoring Demo
===================================================================

A Streamlit-based engineering demo. The "DBMS" tab exposes the underlying
SQLite database: schema, data browser, and a free-form SQL console so the
audience (a bank BA) can verify the data the Agent reasons over.

Run:  streamlit run app.py
"""
import json
import os

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

from db import get_connection, list_tables, table_schema, table_row_count, DB_PATH
from agent import get_ba_agent

# Load project-root .env (db module also loads it; this makes APP_TITLE
# available at the top of app.py).
# override=False → any env var already set in the process (e.g. docker run -e,
# GitHub Actions env:, K8s envFrom) wins over .env file values. .env only fills
# in what's missing. Set LOAD_DOTENV_OVERRIDE=1 to flip back to override mode.
load_dotenv(
    override=os.environ.get("LOAD_DOTENV_OVERRIDE", "false").lower()
    in ("1", "true", "yes", "on")
)

APP_TITLE = os.environ.get("APP_TITLE", "Ontology AI Agent · Risk Monitoring Demo")


def _kg_graph_html(vis_nodes: list[dict], vis_links: list[dict]) -> str:
    """D3 force-directed graph (shared by the Index and Terminology sub-tabs).

    The component's script runs ONCE at iframe load. Streamlit tabs keep
    inactive panels at width 0, so the graph must be built lazily: a
    ResizeObserver builds it on the first non-zero container width (i.e.
    when its tab becomes visible), and is idempotent afterwards.
    String-concatenated HTML with placeholder replacement avoids f-string
    clashes with the JS template literals inside.
    """
    _nodes_json = json.dumps(vis_nodes)
    _links_json = json.dumps(vis_links)
    return (
        '<div id="graph-container" style="width:100%;height:520px;background:#0b1220;border-radius:10px;"></div>\n'
        '<script src="https://d3js.org/d3.v7.min.js"></script>\n'
        "<script>\n"
        "const nodes = __NODES__;\n"
        "const links = __LINKS__;\n"
        "const container = document.getElementById('graph-container');\n"
        "const height = 520;\n"
        "let built = false;\n"
        "function build(width){\n"
        "  if (built || !(width > 50)) return;\n"
        "  built = true;\n"
        "  const svg = d3.select('#graph-container').append('svg')\n"
        "      .attr('width', width).attr('height', height);\n"
        "  const g = svg.append('g');\n"
        "  svg.call(d3.zoom().scaleExtent([0.3,3]).on('zoom', e => g.attr('transform', e.transform)));\n"
        "  const sim = d3.forceSimulation(nodes)\n"
        "      .force('link', d3.forceLink(links).id(d=>d.id).distance(140))\n"
        "      .force('charge', d3.forceManyBody().strength(-420))\n"
        "      .force('center', d3.forceCenter(width/2, height/2))\n"
        "      .force('collide', d3.forceCollide(46));\n"
        "  const link = g.append('g').selectAll('line')\n"
        "      .data(links).enter().append('line')\n"
        "      .attr('stroke', '#4f8cff').attr('stroke-opacity', 0.6).attr('stroke-width', 2);\n"
        "  const linkLabel = g.append('g').selectAll('text')\n"
        "      .data(links).enter().append('text')\n"
        "      .attr('font-size', 10).attr('fill', '#22d3ee').attr('text-anchor','middle')\n"
        "      .text(d => d.relation);\n"
        "  const node = g.append('g').selectAll('g')\n"
        "      .data(nodes).enter().append('g')\n"
        "      .call(d3.drag().on('start',dragstart).on('drag',dragged).on('end',dragend));\n"
        "  node.append('circle').attr('r', 22)\n"
        "      .attr('fill', '#1a2342').attr('stroke', '#4f8cff').attr('stroke-width', 2);\n"
        "  node.append('text').attr('text-anchor','middle').attr('dy', 4)\n"
        "      .attr('font-size', 11).attr('fill','#eef3ff').text(d=>d.name);\n"
        "  sim.on('tick', ()=>{\n"
        "      link.attr('x1',d=>d.source.x).attr('y1',d=>d.source.y)\n"
        "          .attr('x2',d=>d.target.x).attr('y2',d=>d.target.y);\n"
        "      linkLabel.attr('x',d=>(d.source.x+d.target.x)/2).attr('y',d=>(d.source.y+d.target.y)/2 - 6);\n"
        "      node.attr('transform', d=>`translate(${d.x},${d.y})`);\n"
        "  });\n"
        "  function dragstart(event,d){if(!event.active)sim.alphaTarget(0.3).restart();d.fx=d.x;d.fy=d.y;}\n"
        "  function dragged(event,d){d.fx=event.x;d.fy=event.y;}\n"
        "  function dragend(event,d){if(!event.active)sim.alphaTarget(0);d.fx=null;d.fy=null;}\n"
        "}\n"
        "if (container.clientWidth > 50) build(container.clientWidth);\n"
        "new ResizeObserver(() => build(container.clientWidth)).observe(container);\n"
        "</script>\n"
    ).replace("__NODES__", _nodes_json).replace("__LINKS__", _links_json)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("🛡️ Ontology AI Agent")
st.sidebar.caption("Corporate Credit Post-Loan Risk Monitoring")
st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**Database**\n`{DB_PATH}`"
)
with st.sidebar.expander("Entity mapping legend", expanded=False):
    st.markdown(
        """
        Ontology entities → physical tables:
        - **Customer** → `ecif_customer`
        - **Group** → `group_metadata`
        - **On-balance loan** → `loan`
        - **Off-balance guarantee** → `bank_guarantee`
        - **Guarantor** → `guarantee_relation`
        - **Overdue ≥90d** → `overdue`
        """
    )

# The agent's graph memory starts EMPTY: entities and relations are built by
# the agent itself while conversing with the BA (Analysis tab).


# ---------------------------------------------------------------------------
# Main tabs
# ---------------------------------------------------------------------------
tab_dbms, tab_ontology, tab_analysis = st.tabs(
    ["🗄️ DBMS", "🧠 Ontology", "📊 Analysis"]
)

# ===========================================================================
# TAB 1: DBMS
# ===========================================================================
with tab_dbms:
    st.header("Database Management System")
    st.caption(
        "Inspect the schema, browse mock data, and run ad-hoc SQL queries "
        "against the local SQLite database."
    )

    sub_schema, sub_data, sub_sql = st.tabs(["Schema", "Data", "Custom SQL"])

    # ---- Sub-tab: Schema ----
    with sub_schema:
        st.subheader("Database Schema")
        conn = get_connection()
        tables = list_tables(conn)

        cols = st.columns(3)
        for i, tbl in enumerate(tables):
            cnt = table_row_count(conn, tbl)
            with cols[i % 3]:
                st.metric(label=tbl, value=f"{cnt} rows")

        st.markdown("---")
        for tbl in tables:
            with st.expander(f"📋 **{tbl}**", expanded=True):
                cols_meta = table_schema(conn, tbl)
                df = pd.DataFrame(cols_meta)[
                    ["cid", "name", "type", "notnull", "pk"]
                ].rename(
                    columns={
                        "cid": "#",
                        "name": "Column",
                        "type": "Type",
                        "notnull": "Not Null",
                        "pk": "PK",
                    }
                )
                df["Not Null"] = df["Not Null"].map({0: "", 1: "✓"})
                df["PK"] = df["PK"].map({0: "", 1: "🔑"})
                st.dataframe(df, width='stretch', hide_index=True)
        conn.close()

    # ---- Sub-tab: Data ----
    with sub_data:
        st.subheader("Data Browser")
        conn = get_connection()
        tables = list_tables(conn)
        sel = st.selectbox("Select a table", tables, index=0)

        if sel:
            cnt = table_row_count(conn, sel)
            st.caption(f"Showing **{cnt}** rows from `{sel}`")
            df = pd.read_sql_query(f"SELECT * FROM {sel}", conn)
            st.dataframe(df, width='stretch', hide_index=True, height=420)
        conn.close()

    # ---- Sub-tab: Custom SQL ----
    with sub_sql:
        st.subheader("Custom SQL Console")
        st.caption(
            "Write and execute your own SQL. Only `SELECT` statements are allowed "
            "to keep the demo database intact."
        )

        DEFAULT_SQL = """-- Example: customers in the warning pool
-- (overdue >= 90 days OR guarantor is in the overdue-90 set)
SELECT c.cust_name,
       c.host_branch,
       c.cust_status
FROM ecif_customer c
WHERE c.cust_id IN (
    SELECT cust_id FROM overdue WHERE overdue_days >= 90
    UNION
    SELECT g.cust_id
    FROM guarantee_relation g
    JOIN overdue o ON g.guarantor_cust_id = o.cust_id
    WHERE o.overdue_days >= 90
)
ORDER BY c.cust_name;"""

        query = st.text_area("SQL Query", value=DEFAULT_SQL, height=220, key="sql_editor")

        col_run, col_clear, col_preset = st.columns([1, 1, 3])
        with col_run:
            run = st.button("▶ Run Query", type="primary", width='stretch')
        with col_clear:
            if st.button("Clear", width='stretch'):
                st.session_state["sql_editor"] = ""
                st.rerun()

        presets = {
            "Warning pool customers": DEFAULT_SQL,
            "Customer exposure (loans + guarantees)": """SELECT c.cust_name,
       COALESCE(l.loan_bal, 0) AS loan_balance,
       COALESCE(g.guar_bal, 0) AS guarantee_balance,
       COALESCE(l.loan_bal, 0) + COALESCE(g.guar_bal, 0) AS total_exposure
FROM ecif_customer c
LEFT JOIN (SELECT cust_id, SUM(balance) AS loan_bal FROM loan WHERE loan_status='OUTSTANDING' GROUP BY cust_id) l
  ON c.cust_id = l.cust_id
LEFT JOIN (SELECT cust_id, SUM(balance) AS guar_bal FROM bank_guarantee WHERE g_status='ACTIVE' GROUP BY cust_id) g
  ON c.cust_id = g.cust_id
ORDER BY total_exposure DESC;""",
            "Group-level exposure": """SELECT gr.group_name,
       COALESCE(SUM(l.loan_bal), 0) + COALESCE(SUM(g.guar_bal), 0) AS group_exposure
FROM group_metadata gr
JOIN ecif_customer c ON c.group_id = gr.group_id
LEFT JOIN (SELECT cust_id, SUM(balance) AS loan_bal FROM loan WHERE loan_status='OUTSTANDING' GROUP BY cust_id) l
  ON c.cust_id = l.cust_id
LEFT JOIN (SELECT cust_id, SUM(balance) AS guar_bal FROM bank_guarantee WHERE g_status='ACTIVE' GROUP BY cust_id) g
  ON c.cust_id = g.cust_id
GROUP BY gr.group_name
ORDER BY group_exposure DESC;""",
            "Guarantor chain risk": """SELECT gr.cust_id AS customer,
       gr.guarantor_cust_id AS guarantor,
       o.overdue_days AS guarantor_overdue_days
FROM guarantee_relation gr
LEFT JOIN overdue o ON gr.guarantor_cust_id = o.cust_id
ORDER BY o.overdue_days DESC NULLS LAST;""",
        }
        with col_preset:
            chosen = st.selectbox("Load preset query", ["", *presets.keys()], index=0)
            if chosen:
                st.session_state["sql_editor"] = presets[chosen]
                st.rerun()

        if run:
            q = query.strip()
            if not q:
                st.error("Please enter a SQL query.")
            elif not q.lower().lstrip().startswith("select"):
                st.error("Only SELECT statements are allowed in this demo console.")
            else:
                try:
                    conn = get_connection()
                    df = pd.read_sql_query(q, conn)
                    conn.close()
                    st.success(f"Query returned **{len(df)}** row(s).")
                    st.dataframe(df, width='stretch', hide_index=True)
                except Exception as e:
                    st.error(f"SQL execution error:\n```\n{e}\n```")

# ===========================================================================
# TAB 2: Ontology — two isolated graphs (Index / Terminology) in one kernel
# ===========================================================================
with tab_ontology:
    st.header("Ontology Knowledge Base")
    st.caption(
        "Two **scope-isolated** graphs live in the JiuwenMemory kernel's graph "
        "store. Both start empty — chat with the assistant on the Analysis "
        "tab and watch them grow. No hand-coded rules: what gets learned is "
        "decided entirely by the kernel's LLM pipeline and the terminology "
        "extractor."
    )

    from memory.graph_store import get_bank_memory

    gm = get_bank_memory()

    top_l, top_r = st.columns([3, 1])
    with top_l:
        st.metric(
            "Memory backend",
            "JiuwenMemory kernel"
            + (" · LLM inference" if gm.llm_ready else " · keyword fallback"),
        )
        st.caption(f"💾 Persistent storage → {gm.storage_summary}")
    with top_r:
        if st.button(
            "🗑️ Reset memory",
            width='stretch',
            help="Tear the kernel down and assemble a fresh, empty memory (clears both graphs).",
        ):
            gm.reset()
            st.rerun()

    sub_index, sub_term = st.tabs(
        ["🗂️ Index — Jiuwen memory graph", "🏷️ Terminology — entity knowledge graph"]
    )

    # ---- SUB-TAB 1: Index (JiuwenMemory native) ----------------------------
    with sub_index:
        entities = gm.list_entities()
        relations = gm.list_relations()
        i1, i2, _ = st.columns([1, 1, 2])
        i1.metric("Memory units", len(entities))
        i2.metric("Unit relations", len(relations))
        st.caption(
            "The kernel's **native** graph: nodes are memory units (facts "
            "distilled from the dialogue), edges are the kernel's built-in "
            "association types (`corefers`, `caused_by`, `refers_to`, ...). "
            "Stored under scope `bank/ba`."
        )

        if not entities:
            st.info(
                "🫙 The agent's memory is empty — nothing has been learned yet.\n\n"
                "Go to the **Analysis** tab and teach the assistant something, e.g.:\n\n"
                "- *\"The warning pool is customers overdue 90+ days, or guaranteed by "
                "someone who is. Show me the pool.\"*\n"
                "- *\"How do I compute exposure for Prime Materials Co., Ltd.?\"*\n\n"
                "The agent will extract facts and associations from your words and "
                "persist them here."
            )
        else:
            st.subheader("Memory Units")
            entity_by_id = {e.id: e for e in entities}
            cols = st.columns(3)
            for i, ent in enumerate(entities):
                with cols[i % 3]:
                    with st.container(border=True):
                        st.markdown(f"**{ent.name}**")
                        st.caption(f"`{ent.id}` · label: `{ent.label}`")
                        st.write(ent.properties.get("description", ""))
                        extra = {
                            k: v
                            for k, v in ent.properties.items()
                            if k not in ("name", "description")
                        }
                        if extra:
                            st.json(extra, expanded=False)

            st.subheader("Association Graph")
            st.caption("Drag nodes to rearrange. Edges are kernel-derived unit associations.")
            vis_nodes = [{"id": e.id, "name": e.name, "label": e.label} for e in entities]
            vis_links = [
                {"source": r.source, "target": r.target, "relation": r.relation}
                for r in relations
            ]
            components.html(_kg_graph_html(vis_nodes, vis_links), height=540)

            st.subheader("Relations")
            rel_rows = []
            for r in relations:
                src = entity_by_id.get(r.source)
                tgt = entity_by_id.get(r.target)
                rel_rows.append(
                    {
                        "From": src.name if src else r.source,
                        "Relation": r.relation,
                        "To": tgt.name if tgt else r.target,
                        "Properties": r.properties,
                    }
                )
            st.dataframe(pd.DataFrame(rel_rows), width="stretch", hide_index=True)

    # ---- SUB-TAB 2: Terminology (entity knowledge graph) --------------------
    with sub_term:
        term_ents = gm.list_terminology_entities()
        term_rels = gm.list_terminology_relations()
        t1, t2, _ = st.columns([1, 1, 2])
        t1.metric("Entities", len(term_ents))
        t2.metric("Relations", len(term_rels))
        st.caption(
            "A classic **entity knowledge graph**: nodes are entities, edges "
            "are business predicates stated in the dialogue (A —predicate→ B), "
            "extracted by the LLM from each conversation turn and upserted "
            "with deduplication. Stored under scope `bank/ba/agent=terminology` "
            "— fully isolated from the Index graph."
        )

        if not term_ents:
            st.info(
                "🫙 The terminology graph is empty — no entities learned yet.\n\n"
                "State a fact with entities and a predicate on the **Analysis** tab, e.g.:\n\n"
                "- *\"Customer A belongs to Group B, and Customer A's guarantor is Customer C.\"*\n\n"
                "Each turn's entities and predicates become triplets here."
            )
        else:
            st.subheader("Entities")
            term_by_id = {e.id: e for e in term_ents}
            degree: dict[str, int] = {}
            for r in term_rels:
                degree[r.source] = degree.get(r.source, 0) + 1
                degree[r.target] = degree.get(r.target, 0) + 1
            cols = st.columns(4)
            for i, ent in enumerate(term_ents):
                with cols[i % 4]:
                    with st.container(border=True):
                        st.markdown(f"**{ent.name}**")
                        st.caption(
                            f"type: `{ent.label}` · {degree.get(ent.id, 0)} relation(s)"
                        )

            st.subheader("Knowledge Graph")
            st.caption("Drag nodes to rearrange. Edge labels are the business predicates.")
            vis_nodes = [{"id": e.id, "name": e.name, "label": e.label} for e in term_ents]
            vis_links = [
                {"source": r.source, "target": r.target, "relation": r.relation}
                for r in term_rels
            ]
            components.html(_kg_graph_html(vis_nodes, vis_links), height=540)

            st.subheader("Triplets")
            tri_rows = []
            for r in term_rels:
                src = term_by_id.get(r.source)
                tgt = term_by_id.get(r.target)
                tri_rows.append(
                    {
                        "From": src.name if src else r.source,
                        "Relation": r.relation,
                        "To": tgt.name if tgt else r.target,
                        "Evidence": r.properties.get("evidence", ""),
                    }
                )
            st.dataframe(pd.DataFrame(tri_rows), width="stretch", hide_index=True)



# ===========================================================================
# TAB 3: Analysis — conversational BA assistant (openjiuwen agent-core)
# ===========================================================================
with tab_analysis:
    st.header("Natural-Language Driven Analysis")
    st.caption(
        "Multi-turn chat with the Agent (built on openjiuwen agent-core). It "
        "remembers the conversation: ask follow-ups like *\"what about its "
        "guarantor chain?\"* or *\"and the groups?\"*. It reads the DBMS "
        "schema, maps business terms to columns, and generates + runs SQL. "
        "Every turn also feeds the JiuwenMemory kernel, which infers entities "
        "and relations from your words — see the Ontology tab."
    )

    agent = get_ba_agent()

    # Session controls
    ctrl1, ctrl2 = st.columns([3, 1])
    with ctrl1:
        if agent._use_real_llm:
            st.success(f"🧠 Real LLM connected — multi-turn session `{agent._conversation_id}`.")
        else:
            st.info(
                "🔧 Running in **schema-aware fallback mode** (no LLM configured). "
                "Set `OPENAI_API_KEY` / `OPENAI_API_BASE` / `LLM_MODEL` in `.env` to "
                "enable the real openjiuwen ReActAgent."
            )
    with ctrl2:
        if st.button("🧹 New conversation", width='stretch',
                     help="Start a fresh session: clears the chat and the agent's conversation memory."):
            agent.new_conversation()
            st.session_state["ba_chat"] = []
            st.rerun()

    # Suggested queries
    suggestions = [
        "Show me all customers in the warning pool",
        "Which groups have the largest risk exposure?",
        "List customer exposure split into on-balance and off-balance",
        "Show the guarantee-chain cross risk",
        "Which groups are core enterprises?",
    ]
    st.markdown("**Try asking:**")
    sugg_cols = st.columns(len(suggestions))
    for i, s in enumerate(suggestions):
        with sugg_cols[i]:
            if st.button(s, key=f"sg_{i}", width='stretch'):
                st.session_state["ba_chat_pending"] = s

    # Chat history in session state
    if "ba_chat" not in st.session_state:
        st.session_state["ba_chat"] = []
    if st.session_state["ba_chat"]:
        st.caption(f"💬 {sum(1 for m in st.session_state['ba_chat'] if m['role'] == 'user')} turn(s) in this session — the agent remembers all of them.")

    # Render history
    for msg in st.session_state["ba_chat"]:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(msg["content"])
            else:
                st.markdown(msg["content"])
                if msg.get("sql"):
                    with st.expander("📝 Generated SQL", expanded=False):
                        st.code(msg["sql"], language="sql")
                if msg.get("result"):
                    with st.expander("📊 Query result (top rows)", expanded=True):
                        st.text(msg["result"])
                if msg.get("trace"):
                    with st.expander("🔧 Tool trace", expanded=False):
                        for t in msg["trace"]:
                            st.code(t, language="text")

    # Input
    pending = st.session_state.pop("ba_chat_pending", "")
    user_input = st.chat_input("Ask the BA assistant a business question...", key="ba_chat_input")
    query = user_input or pending

    if query:
        st.session_state["ba_chat"].append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Agent is introspecting the schema and generating SQL..."):
                turn = agent.chat(query)

            st.markdown(turn.assistant)
            resp = {"role": "assistant", "content": turn.assistant}

            if turn.sql:
                with st.expander("📝 Generated SQL", expanded=False):
                    st.code(turn.sql, language="sql")
                resp["sql"] = turn.sql

            if turn.result_preview:
                with st.expander("📊 Query result (top rows)", expanded=True):
                    st.text(turn.result_preview)
                resp["result"] = turn.result_preview

            if turn.tool_trace:
                with st.expander("🔧 Tool trace", expanded=False):
                    for t in turn.tool_trace:
                        st.code(t, language="text")
                resp["trace"] = turn.tool_trace

            st.session_state["ba_chat"].append(resp)

