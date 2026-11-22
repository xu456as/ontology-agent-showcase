"""Memory module — the agent's long-term memory, powered by the JiuwenMemory
kernel (agent-memory).

The kernel assembles in-process (``jiuwen_memory.api.assemble_runtime``) and
infers entities/relations from the BA's conversation via its construction
pipeline (LLM extractor + associator). This package exposes a thin facade so
the rest of the app only depends on ``Node`` / ``Edge`` semantics.
"""

from .graph_store import BankMemory, get_bank_memory

__all__ = ["BankMemory", "get_bank_memory"]
