# syntax=docker/dockerfile:1.6
# ontology-agent-showcase — Streamlit demo for a corporate credit risk agent.
#
# Build:
#   docker build -t ontology-agent:latest .
# Run (env passed in, memory persisted to host):
#   docker run -p 8501:8501 \
#     -e LLM_PROVIDER=openai \
#     -e OPENAI_API_KEY=sk-xxx \
#     -e OPENAI_API_BASE=https://api.openai.com/v1 \
#     -e LLM_MODEL=gpt-4o-mini \
#     -v $(pwd)/data:/app/data \
#     ontology-agent:latest

FROM python:3.12-slim AS base

# System deps — milvus-lite (embedded) and nanographrag need nothing extra,
# but streamlit's d3 iframe fetches from jsdelivr so ca-certificates must be
# present. curl is handy for debugging containers. tzdata avoids tz stubs.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl tzdata \
 && rm -rf /var/lib/apt/lists/* \
 && python -m pip install --no-cache-dir --upgrade pip

WORKDIR /app

# install_deps.sh uses $PYTHON (default python3). Make it explicit so the
# slim image (where the interpreter may be named python3, not python) is fine.
ENV PYTHON=python3

# ---- layer 1: dependency install (cached until any requirements file changes) ----
COPY requirements.txt ./
COPY requirements-extras-nodeps.txt ./
COPY scripts/install_deps.sh scripts/
RUN bash scripts/install_deps.sh \
 && python -c "import nano_graphrag, openjiuwen, jiuwen_memory, streamlit; print('deps ok')" \
 && rm -rf /root/.cache/pip

# ---- layer 2: app source ----
COPY . .

# ---- runtime setup ----
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8501

EXPOSE 8501

# entrypoint ensures the SQLite schema is initialized on first run (no-op
# on subsequent restarts), then exec's streamlit.
RUN chmod +x /app/docker-entrypoint.sh
ENTRYPOINT ["/app/docker-entrypoint.sh"]
