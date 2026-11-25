"""Smoke test: teach one caliber, then ask a question that depends on it.

Run:  python _smoke_prefetch.py

Scenario (pure BA business speak, no DB jargon):
  1. Teach: "On-balance exposure is outstanding principal of disbursed loans,
     off-balance is active bank guarantees. Total risk exposure = both added."
  2. Ask: "What's the total risk exposure for Everbright Manufacturing?"
  3. Verify that:
     - memory was written (Index graph has memory units)
     - prefetch returns the taught exposure caliber when queried
     - Agent's SQL uses the taught caliber (loan OUTSTANDING + guarantee ACTIVE)
"""

import os
import sys

# Make sure the project root is on sys.path so `from db import ...` works
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))

from memory.graph_store import get_bank_memory
from agent.ba_agent import get_ba_agent

# ---- Step 0: ensure the SQLite demo DB exists --------------------------------
from db import init_db as init_db_module
db_path = os.environ.get("DB_PATH", os.path.join(_PROJECT_ROOT, "data", "banking_risk.db"))
if not os.path.exists(os.path.abspath(db_path)):
    print("[smoke] SQLite DB not found, initializing...")
    init_db_module.init_db()

# ---- Step 1: start with a FRESH memory --------------------------------------
gm = get_bank_memory()
print(f"[smoke] storage: {gm.storage_summary}")
print(f"[smoke] LLM ready: {gm.llm_ready}")
gm.reset()

# ---- Step 2: teach one business caliber --------------------------------------
TEACH = (
    "On-balance exposure is the outstanding principal of loans we have "
    "actually disbursed. Off-balance exposure is the amount we have committed "
    "under active bank guarantees. Total risk exposure is the two added together."
)
print(f"\n[smoke] TEACHING: {TEACH[:70]}...")
turn1 = get_ba_agent().chat(TEACH)
print(f"  -> assistant:     {turn1.assistant[:140]}")
print(f"  -> tool_trace:    {turn1.tool_trace}")

# ---- Step 3: prefetch should return the taught caliber ----------------------
# Use a query semantically related to what we just taught.
pf = gm.prefetch("risk exposure for customers", top_k=6)
print(f"\n[smoke] PREFETCH result ({len(pf)} chars):")
print(pf if pf else "(empty — prefetch returned nothing)")

# ---- Step 4: ask a question that depends on the caliber taught ---------------
ASK = "What's the total risk exposure for Everbright Manufacturing Co., Ltd.?"
print(f"\n[smoke] ASKING: {ASK}")
turn2 = get_ba_agent().chat(ASK)
print(f"  -> assistant:     {turn2.assistant[:200]}")
print(f"  -> SQL:           {turn2.sql or '(fallback only, no SQL)'}")
print(f"  -> result preview (first 200 chars):")
print(turn2.result_preview[:200] if turn2.result_preview else "(no preview)")
print(f"  -> tool_trace:    {turn2.tool_trace}")

# ---- Step 5: verify ---------------------------------------------------------
checks: list[tuple[str, bool, str]] = []

# 5a. memory units exist after teaching
n_units = len(gm.list_entities())
checks.append(("memory units created", n_units > 0, f"got {n_units}"))

# 5b. prefetch returns something now we have memory
checks.append(("prefetch returns hits", bool(pf), "got hits" if pf else "empty"))

# 5c. Agent SQL (or explanation) uses the taught caliber
#     Look for "OUTSTANDING" or "ACTIVE" in SQL, or "on-balance"/"off-balance" in text
sql_text = (turn2.sql or "").upper()
assistant_text = turn2.assistant.lower()
used_caliber = any(kw in sql_text for kw in ("OUTSTANDING", "ACTIVE")) or any(
    kw in assistant_text for kw in ("on-balance", "off-balance", "outstanding loans", "active guarantees")
)
checks.append(("agent SQL/explanation uses taught caliber", used_caliber, "yes" if used_caliber else "no"))

# 5d. SQL actually ran (non-empty preview for this loan-only customer C001)
has_result = bool(turn2.result_preview.strip())
checks.append(("SQL execution returned rows", has_result, "yes" if has_result else "no"))

print("\n=== VERIFICATION ===")
ok = True
for label, passed, detail in checks:
    tag = "PASS" if passed else "FAIL"
    if not passed:
        ok = False
    print(f"[{tag}] {label}  ({detail})")

sys.exit(0 if ok else 1)
