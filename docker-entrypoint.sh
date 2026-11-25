#!/usr/bin/env bash
# Container entrypoint: init SQLite DB on first run, then exec Streamlit.
# DB / memory state lives under /app/data/ — mount a volume there to persist.
set -euo pipefail

cd /app

DB_PATH="${DB_PATH:-./data/banking_risk.db}"
mkdir -p "$(dirname "$DB_PATH")"

if [ ! -f "$DB_PATH" ]; then
    echo "[entrypoint] ${DB_PATH} missing — initializing schema and mock data"
    python3 db/init_db.py
else
    echo "[entrypoint] ${DB_PATH} already present — skipping init"
fi

echo "[entrypoint] starting Streamlit on :${PORT}"
exec streamlit run app.py \
    --server.port="${PORT}" \
    --server.address=0.0.0.0 \
    --server.runOnSave=false \
    --server.fileWatcherType=none \
    --browser.gatherUsageStats=false
