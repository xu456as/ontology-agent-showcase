"""JiuwenMemory kernel facade — the agent's long-term ontology memory.

Assembles the **full JiuwenMemory kernel** in-process via
``jiuwen_memory.api.assemble_runtime`` — the same entry openjiuwen's
``JiuwenMemoryProvider`` (SDK mode) uses — with **persistent local storage
backends**:

- ``kv_store``  → SQLite file (``data/memory/kv.db``): the true-source store
  for memory units; survives process restarts.
- ``vector_store`` → Milvus. By default an **embedded local Milvus**
  (milvus-lite, data under ``data/memory/milvus.db``); set ``MILVUS_URI`` in
  ``.env`` to point at a standalone/remote Milvus (e.g.
  ``http://localhost:19530``).
- ``graph_store`` → nano-graphrag GraphML files (``data/memory/graph/``):
  both the Index graph and the Terminology graph persist on disk.

It maintains TWO isolated graphs inside the kernel's graph store, separated
by ``Scope`` (the kernel's native isolation primitive — the scope key covers
org/space/user/agent/session, so distinct scopes never share nodes or edges):

1. **Index graph** (scope ``bank/ba``) — JiuwenMemory's native memory graph:
   ``BankMemory.remember`` → ``api.add(infer=True)`` distills the dialogue
   into semantic fact units; ``api.evolve(ASSOCIATE)`` then projects one node
   per memory unit plus kernel-derived unit-to-unit relations (corefers,
   caused_by, ...) into the store (``OrchestratingEvolver._persist_graph``).

2. **Terminology graph** (scope ``bank/ba/agent=terminology``) — a classic
   entity knowledge graph: nodes are named entities, edges are the business
   predicates stated in the dialogue (e.g. ``Customer A —belongs_to→
   Group B``). Triplets are extracted by an LLM call from each conversation
   turn (no hand-coded rules) and upserted directly into the kernel's graph
   store under the terminology scope.

This module only forwards conversation turns, drives the kernel's evolve
pipeline, performs the LLM terminology extraction, and exposes read views
over the two graph scopes for the Ontology tab.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv(
    override=os.environ.get("LOAD_DOTENV_OVERRIDE", "false").lower()
    in ("1", "true", "yes", "on")
)

from jiuwen_memory.api import (
    EvolveMode,
    assemble_runtime,
    internal_context,
)
from jiuwen_memory.common.security.authentication.authentication_impl.dev_authenticator import (
    DevAuthenticator,
)
from jiuwen_memory.common.type_def import Modality, Scope
from jiuwen_memory.construction.extractor_impl.llm_extractor import (
    _EXTRACT_SYSTEM_PROMPT as _KERNEL_EXTRACT_PROMPT,
)
from jiuwen_memory.storage.graph_impl.in_memory_graph_store import _skey
from jiuwen_memory.storage.types import Edge, Node

# Same scope convention as openjiuwen's JiuwenMemoryProvider SDK backend:
# org/user axes only — empty space/agent/session keeps the write inside the
# default space, which the dev identity is allowed to author.
SCOPE = Scope(org="bank", user="ba")

# The terminology (entity knowledge graph) lives under a DIFFERENT scope in
# the same kernel graph store — the scope key includes the agent axis, so the
# two graphs never share nodes/edges. Raw GraphStore writes bypass the memory
# API's security layer, so a distinct agent axis needs no extra identity.
TERMINOLOGY_SCOPE = Scope(org="bank", user="ba", agent="terminology")

# Process-internal caller identity. The kernel requires an authenticated
# actor whose identity covers the target scope; the dev authenticator is
# configured to speak as this demo's BA identity.
_AUTHENTICATOR = DevAuthenticator(actor=Scope(org=SCOPE.org, user=SCOPE.user))

_ASSOCIATE_JOB_TIMEOUT_S = 25.0

# Local persistent storage root. Everything the kernel writes lands under
# project/data/memory/ so the demo's memory survives process restarts.
_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "memory"

# Dimension of the kernel's default hashing embedder (log: "dim=64"). The
# milvus collection must match it.
_EMBEDDER_DIM = 64


def _module_present(name: str) -> bool:
    """Dependency presence check WITHOUT importing the package.

    ``import nano_graphrag`` eagerly pulls the whole GraphRAG pipeline (heavy,
    optional deps); the kernel's graph backend only needs the package present
    (its loader shim imports just the storage submodule).
    """
    try:
        from importlib.util import find_spec

        return find_spec(name) is not None
    except Exception:  # noqa: BLE001
        return False


def _is_local_uri(uri: str) -> bool:
    return not str(uri).lower().startswith(("http://", "https://"))


_MILVUS_CLIENT_PATCHED = False


def _patch_milvus_client_db_name() -> None:
    r"""Pin ``db_name='default'`` for local ``.db`` URIs on pymilvus MilvusClient.

    pymilvus >= 3.5 infers db_name from the URI's first path segment when the
    caller omits db_name — a local path like ``D:\...\milvus.db`` then
    becomes the db_name (``\workspace\...``) and milvus-lite rejects it. The
    kernel's milvus vector backend builds ``MilvusClient(uri)`` without
    db_name, so we default it for local file URIs only. Narrow, idempotent.
    """
    global _MILVUS_CLIENT_PATCHED
    if _MILVUS_CLIENT_PATCHED:
        return
    try:
        from pymilvus import MilvusClient
    except Exception:  # noqa: BLE001
        return
    if getattr(MilvusClient, "_ba_dbname_patched", False):
        _MILVUS_CLIENT_PATCHED = True
        return
    orig_init = MilvusClient.__init__

    def _patched_init(
        self,
        uri: str = "http://localhost:19530",
        user: str = "",
        password: str = "",
        db_name: str = "",
        token: str = "",
        timeout: float | None = None,
        **kwargs: Any,
    ) -> None:
        if db_name == "" and isinstance(uri, str) and uri.endswith(".db"):
            db_name = "default"
        orig_init(self, uri, user, password, db_name, token, timeout, **kwargs)

    MilvusClient.__init__ = _patched_init  # type: ignore[assignment]
    MilvusClient._ba_dbname_patched = True  # type: ignore[attr-defined]
    _MILVUS_CLIENT_PATCHED = True


def _patch_nano_graph_lenient_insert() -> None:
    """Make the nano-graphrag backend's insert idempotent.

    The backend treats an already-present node id or (source, target) edge
    pair as a ``ConflictError`` — but the kernel's associator re-projects the
    same relations on every ASSOCIATE run with fresh edge ids, which aborts
    the whole evolve job. Insert item-by-item instead and skip conflicts.
    """
    try:
        from jiuwen_memory.common.errors import ConflictError
        from jiuwen_memory.storage.graph_impl.nano_graphrag_graph import (
            NanoGraphRAGGraphStore,
        )
    except Exception:  # noqa: BLE001
        return
    if getattr(NanoGraphRAGGraphStore, "_ba_lenient_insert", False):
        return
    orig_insert = NanoGraphRAGGraphStore.insert

    def insert(self, scope, nodes=None, edges=None):  # noqa: ANN001, ANN202
        try:
            return orig_insert(self, scope, nodes=nodes, edges=edges)
        except ConflictError:
            # Batch aborted (possibly after partial writes): replay item by
            # item, skipping whatever is already in the graph.
            for n in nodes or []:
                try:
                    orig_insert(self, scope, nodes=[n])
                except ConflictError:
                    pass
            for e in edges or []:
                try:
                    orig_insert(self, scope, edges=[e])
                except ConflictError:
                    pass

    NanoGraphRAGGraphStore.insert = insert  # type: ignore[assignment]
    NanoGraphRAGGraphStore._ba_lenient_insert = True  # type: ignore[attr-defined]


def _build_config() -> dict[str, Any] | None:
    """Kernel assembly config.

    With LLM credentials in ``.env`` we plug an OpenAI-compatible client
    (e.g. DeepSeek) into the kernel's ``llm.default`` slot and switch the
    kernel's associator to its LLM implementation — so both fact extraction
    and relation inference run through a real model. Without credentials we
    return ``None`` and the kernel uses its built-in defaults (echo LLM,
    keyword extractor/associator).

    Persistent storage backends (local-first):
    - ``kv_store`` → SQLite: memory units survive restarts.
    - ``vector_store`` → Milvus. Default: embedded milvus-lite local file
      (``data/memory/milvus.db``). Set ``MILVUS_URI`` to target a
      standalone/remote Milvus (e.g. ``http://localhost:19530``).
    - ``graph_store`` → nano-graphrag: GraphML files under
      ``data/memory/graph/``, one namespace per scope.

    Any backend whose third-party dependency is missing silently falls back
    to the kernel's in-memory default, so the demo always boots.
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    api_base = os.getenv("OPENAI_API_BASE", "")
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    if not api_key or not api_base:
        return None
    cfg: dict[str, Any] = {
        "llm": {
            "default": {
                "target": "openai",
                "params": {
                    "llm_model": model,
                    "llm_base_url": api_base,
                    "llm_api_key": api_key,
                },
            },
        },
        "associator": {
            "default": {
                "target": "llm",
                "params": {
                    "llm": "default",
                    "feature_extractor": "default",
                    "embedder": "default",
                },
            },
        },
    }

    # ---- kv: SQLite (true source of memory units) --------------------------
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    cfg["kv_store"] = {
        "default": {
            "target": "sqlite",
            "params": {"db_path": str(_DATA_DIR / "kv.db")},
        },
    }

    # ---- vector: Milvus (embedded local file or standalone/remote) ---------
    milvus_uri = (os.getenv("MILVUS_URI", "") or str(_DATA_DIR / "milvus.db")).strip()
    local_lite = not milvus_uri.lower().startswith(("http://", "https://"))
    milvus_ready = _module_present("pymilvus") and (
        not local_lite or _module_present("milvus_lite")
    )
    if milvus_ready:
        _patch_milvus_client_db_name()
        # db_name is pinned via the patch above: the kernel builds
        # MilvusClient(uri) without db_name and pymilvus >= 3.5 would
        # otherwise derive it from the local file path.
        cfg["vector_store"] = {
            "default": {
                "target": "milvus",
                "params": {
                    "uri": milvus_uri,
                    "dim": _EMBEDDER_DIM,
                    "db_name": "default",
                },
            },
        }

    # ---- graph: nano-graphrag GraphML files (durable) -----------------------
    if _module_present("nano_graphrag") and _module_present("networkx"):
        _patch_nano_graph_lenient_insert()
        cfg["graph_store"] = {
            "default": {
                "target": "nano_graphrag",
                "params": {"working_dir": str(_DATA_DIR / "graph")},
            },
        }
    return cfg

# Generic triple-extraction instruction for the terminology graph. It defines
# the OUTPUT SHAPE only (entities + predicate edges) — no business concepts,
# no entity lists, no relation dictionaries: what to extract is entirely the
# LLM's decision from the dialogue.
_TERMINOLOGY_PROMPT = """\
You maintain a small entity knowledge graph (terminology) for a banking
business analyst assistant. From the LAST message (earlier turns are context
for pronoun resolution only — extract from the last message), extract:

- entities: concrete things or concepts the message names (customers, groups,
  products, guarantees, metrics, rules, departments, systems, ...), each with
  a short type label;
- relations: directed pairs of those entities connected by a predicate the
  message actually states, e.g.
  {"source": "Customer A", "relation": "belongs_to", "target": "Group B"}.

Rules:
- Never invent entities or relations the message does not state.
- relation: lowercase snake_case, 1-3 words, taken from the message's wording.
- Entity names: short form as written in the message, never whole sentences.
- If the message contains nothing extractable, return empty arrays.

Output ONLY JSON (no markdown fences, no explanation):
{"entities": [{"name": "...", "type": "..."}],
 "relations": [{"source": "...", "relation": "...", "target": "..."}]}
"""


@dataclass
class GraphNodeView:
    """A flat, serialisable view of a graph Node (for UI rendering)."""

    id: str
    label: str
    name: str
    properties: dict[str, Any]


@dataclass
class GraphEdgeView:
    """A flat, serialisable view of a graph Edge (for UI rendering)."""

    id: str
    source: str
    target: str
    relation: str
    properties: dict[str, Any]


def _node_display_name(node: Any) -> str:
    """Graph nodes carry the memory unit's content; show a short heading."""
    props = node.properties or {}
    content = str(props.get("content") or props.get("name") or node.id)
    return content[:48] + ("…" if len(content) > 48 else "")


def _norm_name(name: str) -> str:
    """Whitespace-collapse + lowercase normalisation for entity identity."""
    return " ".join(str(name).split()).lower()


def _entity_id(name: str) -> str:
    """Deterministic entity node id — same name always maps to one node."""
    return "ent-" + hashlib.sha1(_norm_name(name).encode()).hexdigest()[:12]


def _relation_id(source: str, relation: str, target: str) -> str:
    """Deterministic edge id — the same triplet never duplicates."""
    key = f"{_norm_name(source)}|{_norm_name(relation)}|{_norm_name(target)}"
    return "rel-" + hashlib.sha1(key.encode()).hexdigest()[:12]


def _parse_terminology_json(content: str) -> dict[str, Any]:
    """Tolerant JSON parsing of the extractor's reply.

    Accepts fenced output and surrounding chatter by slicing from the first
    ``{`` to the last ``}``. Validates the minimal shape; raises ValueError
    on garbage so the caller can degrade gracefully.
    """
    text = str(content or "")
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in terminology LLM reply")
    data = json.loads(text[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("terminology reply is not a JSON object")

    entities: list[dict[str, str]] = []
    for ent in data.get("entities") or []:
        if isinstance(ent, dict) and str(ent.get("name") or "").strip():
            entities.append({"name": str(ent["name"]).strip(), "type": str(ent.get("type") or "").strip()})
    relations: list[dict[str, str]] = []
    for rel in data.get("relations") or []:
        if (
            isinstance(rel, dict)
            and str(rel.get("source") or "").strip()
            and str(rel.get("relation") or "").strip()
            and str(rel.get("target") or "").strip()
        ):
            relations.append(
                {
                    "source": str(rel["source"]).strip(),
                    "relation": str(rel["relation"]).strip().lower(),
                    "target": str(rel["target"]).strip(),
                }
            )
    return {"entities": entities[:30], "relations": relations[:30]}


class BankMemory:
    """Facade over the in-process JiuwenMemory kernel.

    Conversation turns go in via :meth:`remember`; entities/relations the
    kernel infers are read out via :meth:`list_entities` /
    :meth:`list_relations`.
    """

    def __init__(self) -> None:
        self._runtime: Any | None = None
        self._api: Any | None = None
        self._graph: Any | None = None
        self._scope = SCOPE
        self._sk = _skey(SCOPE)
        self._term_scope = TERMINOLOGY_SCOPE
        self._term_sk = _skey(TERMINOLOGY_SCOPE)
        self._term_client: Any | None = None  # lazy OpenAI-compatible client
        self._infer = False
        self._llm_ready = False
        self._lock = threading.Lock()

    # -- lifecycle -----------------------------------------------------------
    def _ensure_init(self) -> None:
        if self._api is not None:
            return
        with self._lock:
            if self._api is not None:
                return
            cfg = _build_config()
            self._llm_ready = cfg is not None
            self._infer = self._llm_ready
            runtime = assemble_runtime(config=cfg)
            self._runtime = runtime
            self._api = runtime.api
            # The kernel's StoreManager exposes the shared graph store that
            # the associator's evolve pipeline projects entities/relations to.
            self._graph = runtime._kernel.storage.graph()

    def reset(self) -> None:
        """Tear the kernel down and wipe the persisted storage files.

        With persistent backends (sqlite kv / milvus / graphml) a bare
        teardown would leave the data on disk, so the storage directory is
        removed as well. A directory that is still held open by the OS (e.g.
        an in-process milvus-lite server) may survive on Windows; restarting
        the app then fully clears it.
        """
        with self._lock:
            self._runtime = None
            self._api = None
            self._graph = None
            self._infer = False
            self._llm_ready = False
            shutil.rmtree(_DATA_DIR, ignore_errors=True)

    @property
    def llm_ready(self) -> bool:
        self._ensure_init()
        return self._llm_ready

    @property
    def storage_summary(self) -> str:
        """Short description of the active storage backends (UI metric)."""
        self._ensure_init()
        parts = ["kv: sqlite"]
        cfg = _build_config() or {}
        if "vector_store" in cfg:
            uri = cfg["vector_store"]["default"]["params"]["uri"]
            kind = "milvus-lite (local file)" if _is_local_uri(uri) else "milvus standalone"
            parts.append(f"vector: {kind}")
        else:
            parts.append("vector: in-memory")
        parts.append("graph: graphml" if "graph_store" in cfg else "graph: in-memory")
        return " · ".join(parts)

    # -- write -----------------------------------------------------------------
    def remember(self, text: str, *, infer: bool | None = None, context: str = "") -> str:
        """Feed one conversation turn to the kernel (and the terminology graph).

        With ``infer=True`` the kernel runs its synchronous LLM extraction
        (dedup-aware), distilling the dialogue into semantic fact units.
        Afterwards an ASSOCIATE evolve job lets the kernel's associator derive
        relations across all accumulated units and project them into the
        Index graph. Separately, an LLM triple-extraction call upserts
        entities and predicate edges into the Terminology graph (its own
        scope). ``context`` carries earlier turns for pronoun resolution in
        the terminology prompt only. Returns a short summary for the trace.
        """
        self._ensure_init()
        text = (text or "").strip()
        if not text:
            return "empty text, nothing stored"
        use_infer = self._infer if infer is None else infer
        if use_infer:
            # Route the kernel's dynamic LLM extractor to its own built-in
            # extraction prompt (inline-text form is an officially supported
            # fallback: the registry miss uses the key value as prompt text).
            # The prompt contains no domain rules — the LLM decides what the
            # dialogue means.
            metadata = {
                "infer": "true",
                "_extract_prompt_default": _KERNEL_EXTRACT_PROMPT,
            }
        else:
            metadata = None
        security = internal_context(_AUTHENTICATOR)
        units = self._api.add(
            text,
            self._scope,
            Modality.TEXT,
            security=security,
            tags=["conversation", "user"],
            system_metadata=metadata,
        )
        n = len(units or [])
        note = f"{n} memory unit(s) extracted, infer={'on' if use_infer else 'off'}"
        if use_infer:
            note += f"; {self._associate(security)}"
        note += f"; {self._learn_terminology(text, context)}"
        return note

    def _associate(self, security: Any) -> str:
        """Run the kernel's ASSOCIATE evolve: associator derives relations
        across memory units; the evolver persists nodes + edges to the graph
        store. The job runs on the kernel's in-process scheduler; we poll it
        to completion so the Ontology tab is fresh right after the turn."""
        try:
            job_id = self._api.evolve(
                self._scope, EvolveMode.ASSOCIATE, security=security
            )
        except Exception as exc:  # noqa: BLE001
            return f"associate skipped ({type(exc).__name__})"
        if not job_id:
            return "associate job not scheduled"
        deadline = time.monotonic() + _ASSOCIATE_JOB_TIMEOUT_S
        while time.monotonic() < deadline:
            try:
                info = self._api.job_status(job_id, security=security)
            except Exception:  # noqa: BLE001
                return "associate job submitted"
            status = str(getattr(info, "status", "")).lower()
            if any(s in status for s in ("succeed", "complete", "done", "failed", "cancel")):
                return f"associate job {status}"
            time.sleep(0.5)
        return "associate job running in background (graph updates shortly)"

    # -- terminology graph (entity knowledge graph, separate scope) ------------
    def _openai_client(self) -> Any:
        if self._term_client is None:
            from openai import OpenAI

            self._term_client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY", ""),
                base_url=os.getenv("OPENAI_API_BASE", ""),
            )
        return self._term_client

    def _learn_terminology(self, text: str, context: str = "") -> str:
        """Extract entity/relation triplets from this turn via LLM and upsert
        them into the terminology scope of the kernel's graph store.

        The LLM decides WHAT to extract (no domain rules here); this method
        only normalises names into deterministic node ids so repeated mentions
        merge into one entity node and repeated triplets do not duplicate.
        """
        if not self._llm_ready:
            return "terminology skipped (no LLM configured)"
        try:
            content = (context or "").strip()
            user_text = text
            if content:
                user_text = (
                    f"[Earlier turns — context only]\n{content}\n\n"
                    f"[Last message — extract from this]\n{text}"
                )
            resp = self._openai_client().chat.completions.create(
                model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
                temperature=0,
                messages=[
                    {"role": "system", "content": _TERMINOLOGY_PROMPT},
                    {"role": "user", "content": user_text},
                ],
            )
            data = _parse_terminology_json(resp.choices[0].message.content)
            return self._persist_terminology(data, text)
        except Exception as exc:  # noqa: BLE001
            return f"terminology extraction failed ({type(exc).__name__})"

    def _persist_terminology(self, data: dict[str, Any], evidence: str) -> str:
        """Upsert extracted triplets into the terminology scope (id-deduped)."""
        existing_nodes = {n.id for n in self._scope_nodes(self._term_scope)}
        existing_edges = {e.id for e in self._scope_edges(self._term_scope)}

        nodes: dict[str, Node] = {}
        for ent in data["entities"]:
            eid = _entity_id(ent["name"])
            if eid in existing_nodes or eid in nodes:
                continue
            etype = str(ent.get("type") or "entity")
            nodes[eid] = Node(
                id=eid, label=etype, properties={"name": ent["name"], "type": etype}
            )
        edges: dict[str, Edge] = {}
        for rel in data["relations"]:
            # Endpoints missing from the entities list get a bare node so the
            # edge always connects two existing nodes.
            for name in (rel["source"], rel["target"]):
                eid = _entity_id(name)
                if eid not in existing_nodes and eid not in nodes:
                    nodes[eid] = Node(
                        id=eid, label="entity", properties={"name": name, "type": "entity"}
                    )
            rid = _relation_id(rel["source"], rel["relation"], rel["target"])
            if rid in existing_edges or rid in edges:
                continue
            edges[rid] = Edge(
                id=rid,
                source=_entity_id(rel["source"]),
                target=_entity_id(rel["target"]),
                relation=rel["relation"],
                properties={"evidence": evidence[:160]},
            )
        if nodes or edges:
            self._graph.insert(
                self._term_scope, nodes=list(nodes.values()), edges=list(edges.values())
            )
        return f"terminology: +{len(nodes)} entities, +{len(edges)} relations"

    # -- read --------------------------------------------------------------------

    # 给 Agent 的检索入口：用 BA 这轮的话当 query 去内核混合召回（关键词 + 向量 + 图），
    # 把命中的记忆单元拼成一段自然语言返回，供 Agent 当作 long-term context 使用。
    # 不匹配不报错，返回空串——Agent 当没这回事。
    def prefetch(self, query: str, top_k: int = 6) -> str:
        self._ensure_init()
        text = (query or "").strip()
        if not text:
            return ""
        try:
            from jiuwen_memory.api import (
                Context,
                DisclosureLevel,
                internal_context,
            )

            security = internal_context(_AUTHENTICATOR)
            ctx = Context(scope=self._scope)
            # L0=摘要(50-100字) 轻量，够用；max_tokens 自适应预算在 ADAPTIVE 模式下
            # 由内核自己选 L0/L1/L2，这里直接要 L0 摘要最省 token。
            result = self._api.search(
                text,
                ctx,
                security=security,
                top_k=min(max(top_k, 1), 20),
                disclosure=DisclosureLevel.L0,
            )
        except Exception as exc:  # noqa: BLE001
            return ""
        items = getattr(result, "items", None) or []
        if not items:
            return ""
        # 用 abstract(L0)，空就降级 overview(L1)，再空 content(L2)
        lines: list[str] = []
        seen: set[str] = set()
        for it in items:
            snippet = (
                getattr(it, "abstract", "")
                or getattr(it, "overview", "")
                or getattr(it, "content", "")
                or ""
            ).strip()
            snippet = " ".join(snippet.split())  # 折叠空白
            if not snippet or snippet in seen:
                continue
            seen.add(snippet)
            lines.append(snippet[:200])  # 单条硬上限，防被长单元撑爆
        if not lines:
            return ""
        # 只取前 6 条（top_k 默认值），并加一条简短标题让 Agent 知道这段是什么
        head = "## Long-term memory — what the BA has taught me so far\n"
        return head + "\n".join(f"- {ln}" for ln in lines[:top_k])

    def list_entities(self) -> list[GraphNodeView]:
        """Nodes of the INDEX graph (JiuwenMemory memory units)."""
        self._ensure_init()
        return self._list_nodes(self._scope)

    def list_relations(self) -> list[GraphEdgeView]:
        """Edges of the INDEX graph (kernel-derived unit associations)."""
        self._ensure_init()
        return self._list_edges(self._scope)

    def list_terminology_entities(self) -> list[GraphNodeView]:
        """Nodes of the TERMINOLOGY graph (entities)."""
        self._ensure_init()
        return self._list_nodes(self._term_scope)

    def list_terminology_relations(self) -> list[GraphEdgeView]:
        """Edges of the TERMINOLOGY graph (business predicate triplets)."""
        self._ensure_init()
        return self._list_edges(self._term_scope)

    def _scope_nodes(self, scope: Scope) -> list[Any]:
        """All nodes of one scope, across graph store backends.

        The GraphStore contract has no "list all" verb, so we read the
        backend's internals: in-memory keeps per-scope dicts; nano-graphrag
        keeps a networkx graph per scope namespace.
        """
        store = getattr(self._graph, "_store", self._graph)
        if hasattr(store, "_nodes"):  # in-memory backend
            return list(store._nodes.get(_skey(scope), {}).values())
        from jiuwen_memory.storage.graph_impl.nano_graphrag_graph import networkx_graph

        out = []
        for nid, data in networkx_graph(store._store(scope)).nodes(data=True):
            try:
                props = json.loads(data.get("properties") or "{}")
            except Exception:  # noqa: BLE001
                props = {}
            out.append(
                Node(id=str(nid), label=str(data.get("label") or ""), properties=props)
            )
        return out

    def _scope_edges(self, scope: Scope) -> list[Any]:
        """All edges of one scope, across graph store backends."""
        store = getattr(self._graph, "_store", self._graph)
        if hasattr(store, "_edges"):  # in-memory backend
            return list(store._edges.get(_skey(scope), {}).values())
        from jiuwen_memory.storage.graph_impl.nano_graphrag_graph import networkx_graph

        out = []
        for u, v, data in networkx_graph(store._store(scope)).edges(data=True):
            try:
                props = json.loads(data.get("properties") or "{}")
            except Exception:  # noqa: BLE001
                props = {}
            out.append(
                Edge(
                    id=str(data.get("id") or f"{u}->{v}"),
                    source=str(u),
                    target=str(v),
                    relation=str(data.get("relation") or ""),
                    properties=props,
                )
            )
        return out

    def _list_nodes(self, scope: Scope) -> list[GraphNodeView]:
        return [
            GraphNodeView(
                id=n.id,
                label=str(n.label),
                name=_node_display_name(n),
                properties=dict(n.properties or {}),
            )
            for n in self._scope_nodes(scope)
        ]

    def _list_edges(self, scope: Scope) -> list[GraphEdgeView]:
        return [
            GraphEdgeView(
                id=e.id,
                source=e.source,
                target=e.target,
                relation=str(e.relation),
                properties=dict(e.properties or {}),
            )
            for e in self._scope_edges(scope)
        ]


# -- module-level singleton ------------------------------------------------------

_bank_memory: BankMemory | None = None
_bank_memory_lock = threading.Lock()


def get_bank_memory() -> BankMemory:
    global _bank_memory
    if _bank_memory is None:
        with _bank_memory_lock:
            if _bank_memory is None:
                _bank_memory = BankMemory()
    return _bank_memory
