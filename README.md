# Ontology AI Agent — Corporate Credit Post-Loan Risk Monitoring Demo

A Streamlit-based engineering demo that showcases an ontology-driven AI Agent
positioned as a "bank BA's apprentice". The BA teaches the Agent business
rules in plain language; the Agent extracts entities and relationships,
persists them, and later answers natural-language data queries by reusing
that knowledge.

The demo has three tabs:

1. **DBMS** — SQLite schema, data browser, and free-form SQL console.
2. **Ontology** — the entities & relations the **JiuwenMemory kernel** infers
   from the conversation (LLM fact extraction + LLM association pipeline — no
   hand-coded rules), rendered as an interactive knowledge graph. The memory
   starts **empty** and grows as you chat — a *Reset memory* button lets you
   re-run the demo from scratch.
3. **Analysis** — a conversational BA assistant built on
   **openjiuwen agent-core (ReActAgent)**. It reads the DBMS schema and
   generates + runs SQL. Every turn is fed to the JiuwenMemory kernel, which
   distills it into semantic fact units and derives the relations between
   them into its graph store.

---

## Project Structure

```
ontology-agent-showcase/
├── app.py                  # Streamlit application entry point (3 tabs)
├── requirements.txt        # Python dependencies
├── .env                    # Runtime configuration (DB path, LLM creds, etc.)
├── .env.example            # Template for .env
├── README.md               # This file
├── db/
│   ├── __init__.py         # DB connection helpers (reads DB_PATH from .env)
│   └── init_db.py          # SQLite schema + mock data initialization
├── memory/
│   ├── __init__.py
│   └── graph_store.py      # GraphMemory facade (JiuwenMemory) + ontology seed
├── agent/
│   ├── __init__.py
│   ├── tools.py            # openjiuwen Tools: get_schema, run_sql
│   └── ba_agent.py         # BA assistant (ReActAgent + deterministic fallback)
└── data/
    └── banking_risk.db     # Generated SQLite database (run init_db.py)
```

---

## Prerequisites

- Python **3.11+** (openjiuwen requires 3.11 ≤ version < 3.14; tested on 3.12)
- pip

---

## Installation

### 1. Create a virtual environment

From the project root:

```bash
python -m venv .venv
```

### 2. Activate the virtual environment

**Windows (PowerShell):**

```powershell
.\.venv\Scripts\Activate.ps1
```

> If PowerShell blocks script execution, run:
> `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`

**Windows (CMD):**

```cmd
.venv\Scripts\activate.bat
```

**Linux / macOS:**

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Initialize the database

```bash
python db/init_db.py
```

This creates `data/banking_risk.db` with 6 tables and English mock data.

---

## Configuration (`.env`)

The application reads configuration from the project-root `.env` file via
`python-dotenv`. Copy `.env.example` to `.env` if needed and adjust values:

| Variable       | Default                         | Description                                   |
|----------------|---------------------------------|-----------------------------------------------|
| `DB_PATH`      | `./data/banking_risk.db`        | Path to the SQLite database file.             |
| `APP_TITLE`    | `Ontology AI Agent · Risk Monitoring Demo` | Browser tab / page title.          |
| `DEFAULT_PORT` | `8501`                          | Default Streamlit port (informational).       |
| `LOG_LEVEL`    | `INFO`                          | Logging level (not yet wired to app output).  |
| `LLM_PROVIDER` | `openai`                        | LLM provider for the openjiuwen ReActAgent.   |
| `OPENAI_API_KEY` | _(empty)_                     | LLM API key. Empty → fallback mode.           |
| `OPENAI_API_BASE` | _(empty)_                    | LLM API base URL (e.g. `https://api.openai.com/v1`). |
| `LLM_MODEL`    | `gpt-4o-mini`                   | Model name passed to openjiuwen.              |

`DB_PATH` may be relative to the project root or absolute.

### LLM vs fallback mode

The **Analysis** tab uses an `openjiuwen` `ReActAgent` with two tools:

- `get_schema` — introspects the SQLite schema
- `run_sql` — executes read-only SQL and returns rows

Ontology construction is **not** tool-driven: every turn is also fed to the
in-process **JiuwenMemory kernel** (`jiuwen_memory.api.assemble_runtime`, the
same entry openjiwen's `JiuwenMemoryProvider` SDK mode uses). When LLM
credentials are configured, the kernel's dynamic LLM extractor distills the
dialogue into semantic fact units and its LLM associator derives the
relations between units, projecting both into the kernel's graph store —
which the Ontology tab reads. Without credentials the kernel falls back to
its built-in keyword extractor/associator, so the demo still runs
end-to-end.

DeepSeek works out of the box: set `LLM_PROVIDER=DeepSeek`,
`OPENAI_API_BASE=https://api.deepseek.com/v1` and `LLM_MODEL=deepseek-chat`
(the provider name is case-sensitive and matches openjiuwen's
`ProviderType.DeepSeek`).

---

## Running the Application

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501**.

If `streamlit` is not found on PATH, use:

```bash
python -m streamlit run app.py
```

### docker run
```bash
docker run -d --restart unless-stopped `
  -p 8501:8501 `
  --env-file .\.env `
  -v D:/workspace/ontology_helper_case/ontology-agent-showcase/data:/app/data `
  ghcr.io/xu456as/ontology-agent-showcase:0.0.1
```


---

## Database Schema

Six tables model a corporate credit post-loan risk monitoring domain.

### `ecif_customer` — Customer master data

| Column         | Type    | Notes                       |
|----------------|---------|-----------------------------|
| `cust_id`      | TEXT PK | Customer identifier         |
| `cust_name`    | TEXT    | Customer name               |
| `group_id`     | TEXT FK | Group identifier            |
| `is_group_host`| TEXT    | `Y` / `N` — group sponsor   |
| `host_branch`  | TEXT    | Host branch                 |
| `industry`     | TEXT    | Industry                    |
| `cust_status`  | TEXT    | `NORMAL` / `WARNING` / `OVERDUE` |

### `group_metadata` — Group metadata

Static, descriptive attributes of a group. Member count and the
"core enterprise" flag are **not stored** — they are derived at query time
from `ecif_customer` (count of members per `group_id`, combined with the
`is_group_host = 'Y'` sponsor flag and a ≥3-member threshold).

| Column              | Type    | Notes                                   |
|---------------------|---------|-----------------------------------------|
| `group_id`          | TEXT PK | Group identifier                        |
| `group_name`        | TEXT    | Group name                              |
| `group_type`        | TEXT    | Group type                              |
| `group_description` | TEXT    | Short business description of the group |

### `loan` — On-balance sheet loans

| Column       | Type    | Notes                                |
|--------------|---------|--------------------------------------|
| `loan_id`    | TEXT PK | Loan identifier                      |
| `cust_id`    | TEXT FK | Customer                             |
| `balance`    | REAL    | Outstanding balance (10k CNY)        |
| `loan_status`| TEXT    | `OUTSTANDING` / `SETTLED`            |
| `start_date` | TEXT    | Disbursement date                    |
| `end_date`   | TEXT    | Maturity date                        |

### `bank_guarantee` — Off-balance sheet guarantees

| Column        | Type    | Notes                      |
|---------------|---------|----------------------------|
| `guarantee_id`| TEXT PK | Guarantee identifier       |
| `cust_id`     | TEXT FK | Customer                   |
| `balance`     | REAL    | Guarantee balance (10k CNY)|
| `g_status`    | TEXT    | `ACTIVE` / `EXPIRED`       |

### `guarantee_relation` — Who guarantees whom

| Column             | Type    | Notes                      |
|--------------------|---------|----------------------------|
| `rel_id`           | TEXT PK | Relation identifier        |
| `cust_id`          | TEXT FK | Guaranteed customer        |
| `guarantor_cust_id`| TEXT FK | Guarantor customer         |
| `guarantee_amt`    | REAL    | Guarantee amount (10k CNY) |

### `overdue` — Overdue records

| Column        | Type    | Notes                                      |
|---------------|---------|--------------------------------------------|
| `overdue_id`  | TEXT PK | Overdue record identifier                  |
| `cust_id`     | TEXT FK | Customer                                   |
| `overdue_days`| INTEGER | Days overdue                               |
| `overdue_amt` | REAL    | Overdue amount (10k CNY)                   |
| `five_class`  | TEXT    | Loan five-class (PASS/SPECIAL/SUBPAR/...)  |

---

## Business Scenario — The Warning Pool

The demo data tells a coherent risk story:

- **C010** (Prime Materials) — 120 days overdue → in warning pool
- **C002** (Everbright Trading) — 95 days overdue → in warning pool
- **C005** (Far East Construction) — guaranteed by C010 → in warning pool
- **C008** (Global Supply Chain) — guaranteed by C002 → in warning pool

**Warning pool rule** (as taught by the BA):
> A customer enters the warning pool if it is itself 90+ days overdue, **or**
> if its guarantor belongs to the 90+ day overdue set.

**Exposure rule:**
> Exposure = outstanding on-balance loan balance + active off-balance
> guarantee balance (excludes settled loans / expired guarantees).

---

## Custom SQL Console

The **DBMS → Custom SQL** sub-tab lets you run any `SELECT` query. Built-in
presets:

1. **Warning pool customers** — customers 90+ days overdue or guaranteed by
   such a customer.
2. **Customer exposure** — per-customer loan + guarantee exposure.
3. **Group-level exposure** — exposure aggregated by group, sorted descending.
4. **Guarantor chain risk** — each guarantee relation with the guarantor's
   overdue days.

Only `SELECT` statements are allowed to keep the demo database intact.

---

## Roadmap

- [x] Phase 1: DBMS tab — schema, data browser, custom SQL
- [ ] Phase 2: Ontology tab — entity definitions, relationship graph,
      entity→field mapping
- [ ] Phase 3: Analysis tab — natural-language query → semantic parsing →
      schema linking → generated SQL → result set + charts
- [ ] Phase 4: Full teaching → analysis closed-loop walkthrough
