#!/usr/bin/env bash
# Install every dependency group for ontology-agent-showcase.
#
# requirements.txt has packages that MUST pull their transitive deps
# (milvus-lite needs grpcio/protobuf, jiuwen_memory needs networkx, etc.).
# requirements-extras-nodeps.txt has packages that MUST NOT pull their own
# deps — nano-graphrag's Requires-Dist lists tiktoken/torch which the project
# never uses; we only need the package present so that jiuwen_memory's
# GraphML graph backend can import its storage shim.
#
# Run order matters: runtime deps land FIRST, so that a --no-deps package
# that relies on a requirement from requirements.txt (nano-graphrag uses
# networkx) imports cleanly at graph_backend init time.
#
# Usage:
#   scripts/install_deps.sh              # install only, no checks
#   scripts/install_deps.sh --verify     # install + find_spec import check
#   scripts/install_deps.sh --dry-run    # print pip commands, don't run
#   PYTHON=/path/to/python scripts/install_deps.sh --verify

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
DRY_RUN=false
VERIFY=false

usage() {
    cat <<'EOF'
Usage: scripts/install_deps.sh [OPTIONS]

Options:
  --python PATH   Python interpreter (default: $PYTHON, fallback python3)
  --verify        After installing, run importlib.util.find_spec checks
  --dry-run       Print pip commands only, do not execute
  -h, --help      Show this message
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --python)
            PYTHON="$2"; shift 2 ;;
        --verify)
            VERIFY=true; shift ;;
        --dry-run)
            DRY_RUN=true; shift ;;
        -h|--help)
            usage; exit 0 ;;
        *)
            echo "unknown flag: $1" >&2; usage; exit 2 ;;
    esac
done

pip() {
    if $DRY_RUN; then
        echo "$PYTHON -m pip $*"
    else
        $PYTHON -m pip "$@"
    fi
}

echo "==> [runtime] requirements.txt"
pip install --disable-pip-version-check --no-input -r requirements.txt

echo "==> [nodeps-extras] requirements-extras-nodeps.txt"
pip install --disable-pip-version-check --no-input --no-deps -r requirements-extras-nodeps.txt

if $VERIFY; then
    echo "-- verify --"
    # find_spec is the right check — it confirms the package is present
    # without importing it (nano_graphrag.__init__ imports tiktoken/torch
    # which we deliberately did NOT install). This matches what
    # memory/graph_store.py::_module_present does at runtime.
    $PYTHON -c '
import importlib.util, sys
modules = [
    ("runtime", "streamlit"),
    ("runtime", "pandas"),
    ("runtime", "openjiuwen"),
    ("runtime", "jiuwen_memory"),
    ("nodeps-extras", "nano_graphrag"),
]
ok_all = True
for group, mod in modules:
    ok = importlib.util.find_spec(mod) is not None
    tag = "ok" if ok else "MISSING"
    print("[%s] %s: %s" % (tag, group, mod))
    ok_all = ok_all and ok
sys.exit(0 if ok_all else 1)
'
fi

echo "[OK] dependency install complete."
