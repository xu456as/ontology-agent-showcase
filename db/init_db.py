"""
Initialize the SQLite database for the Ontology AI Agent demo.

Creates 6 tables with English mock data that tells a coherent
credit-risk monitoring story:
  - C010 (Prime Materials Corp) and C002 (Everbright Trading Co)
    are 90+ days overdue themselves.
  - C005 (Far East Construction) is guaranteed by C010 -> warning pool.
  - C008 (Global Supply Chain) is guaranteed by C002 -> warning pool.
  - C011 (Prime Building Materials) is guaranteed by C005 -> warning pool.

Run:  python db/init_db.py
"""
import os
import sqlite3

from dotenv import load_dotenv

# Load .env from project root so DB_PATH stays in sync with app.py.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))

_DEFAULT_DB = os.path.join(_PROJECT_ROOT, "data", "banking_risk.db")
DB_PATH = os.path.abspath(os.environ.get("DB_PATH", _DEFAULT_DB))

# ---------------------------------------------------------------------------
# Schema definitions
# ---------------------------------------------------------------------------
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS ecif_customer (
    cust_id        TEXT PRIMARY KEY,
    cust_name      TEXT NOT NULL,
    group_id       TEXT,
    is_group_host  TEXT NOT NULL CHECK (is_group_host IN ('Y','N')),
    host_branch    TEXT,
    industry       TEXT,
    cust_status    TEXT
);

CREATE TABLE IF NOT EXISTS group_metadata (
    group_id          TEXT PRIMARY KEY,
    group_name        TEXT NOT NULL,
    group_type        TEXT,
    group_description TEXT
);

CREATE TABLE IF NOT EXISTS loan (
    loan_id      TEXT PRIMARY KEY,
    cust_id      TEXT NOT NULL,
    balance      REAL NOT NULL,
    loan_status  TEXT NOT NULL CHECK (loan_status IN ('OUTSTANDING','SETTLED')),
    start_date   TEXT,
    end_date     TEXT
);

CREATE TABLE IF NOT EXISTS bank_guarantee (
    guarantee_id TEXT PRIMARY KEY,
    cust_id      TEXT NOT NULL,
    balance      REAL NOT NULL,
    g_status     TEXT NOT NULL CHECK (g_status IN ('ACTIVE','EXPIRED'))
);

CREATE TABLE IF NOT EXISTS guarantee_relation (
    rel_id             TEXT PRIMARY KEY,
    cust_id            TEXT NOT NULL,
    guarantor_cust_id  TEXT NOT NULL,
    guarantee_amt      REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS overdue (
    overdue_id   TEXT PRIMARY KEY,
    cust_id      TEXT NOT NULL,
    overdue_days INTEGER NOT NULL,
    overdue_amt  REAL NOT NULL,
    five_class   TEXT
);
"""

# ---------------------------------------------------------------------------
# Mock data (English)
# ---------------------------------------------------------------------------
GROUPS = [
    ("G001", "Everbright Industrial Group", "Industrial", "Diversified industrial conglomerate spanning manufacturing, logistics, and new energy."),
    ("G002", "Far East Construction Group", "Industrial", "Construction and building materials group focused on infrastructure projects."),
    ("G003", "Global Commerce Group",       "Trading",    "Trading and retail group with supply chain and e-commerce subsidiaries."),
    ("G004", "Prime Materials Group",       "Industrial", "Chemical materials and building materials group."),
    ("G005", "RuiDa Technology Group",      "Technology", "Technology group covering electronics, software, and semiconductors."),
]

CUSTOMERS = [
    ("C001", "Everbright Manufacturing Co., Ltd.",  "G001", "Y", "Head Office",   "Manufacturing",  "NORMAL"),
    ("C002", "Everbright Trading Co., Ltd.",        "G001", "N", "Shanghai Br.", "Wholesale/Retail","WARNING"),
    ("C003", "Everbright Logistics Co., Ltd.",      "G001", "N", "Shanghai Br.", "Transportation", "NORMAL"),
    ("C004", "Everbright New Energy Co., Ltd.",     "G001", "N", "Shenzhen Br.", "New Energy",     "NORMAL"),
    ("C005", "Far East Construction Co., Ltd.",     "G002", "Y", "Beijing Br.",  "Construction",   "WARNING"),
    ("C006", "Far East Building Materials Co.",     "G002", "N", "Beijing Br.",  "Building Materials","NORMAL"),
    ("C007", "Global Department Store Co., Ltd.",   "G003", "Y", "Guangzhou Br.","Retail",         "NORMAL"),
    ("C008", "Global Supply Chain Co., Ltd.",       "G003", "N", "Guangzhou Br.","Supply Chain",   "WARNING"),
    ("C009", "Global E-Commerce Co., Ltd.",         "G003", "N", "Guangzhou Br.","Internet",       "NORMAL"),
    ("C010", "Prime Materials Co., Ltd.",           "G004", "Y", "Hangzhou Br.", "Chemical Materials","OVERDUE"),
    ("C011", "Prime Building Materials Co., Ltd.",  "G004", "N", "Hangzhou Br.", "Building Materials","NORMAL"),
    ("C012", "RuiDa Technology Co., Ltd.",          "G005", "Y", "Chengdu Br.",  "Electronics",    "NORMAL"),
    ("C013", "RuiDa Software Co., Ltd.",            "G005", "N", "Chengdu Br.",  "Software",       "NORMAL"),
    ("C014", "RuiDa Semiconductor Co., Ltd.",       "G005", "N", "Chengdu Br.",  "Semiconductor",  "NORMAL"),
]

LOANS = [
    ("L001", "C001", 12000, "OUTSTANDING", "2025-03-10", "2027-03-09"),
    ("L002", "C002",  6000, "OUTSTANDING", "2024-11-01", "2026-04-30"),
    ("L003", "C003",  4000, "OUTSTANDING", "2025-06-15", "2027-06-14"),
    ("L004", "C004",  8000, "OUTSTANDING", "2025-01-20", "2028-01-19"),
    ("L005", "C005",  8000, "OUTSTANDING", "2025-02-28", "2027-02-27"),
    ("L006", "C006",  5000, "OUTSTANDING", "2025-04-12", "2027-04-11"),
    ("L007", "C007",  9000, "OUTSTANDING", "2025-05-05", "2027-05-04"),
    ("L008", "C008",  5000, "OUTSTANDING", "2025-07-18", "2027-07-17"),
    ("L009", "C009",  3000, "OUTSTANDING", "2025-08-22", "2027-08-21"),
    ("L010", "C010",  5000, "OUTSTANDING", "2024-06-01", "2026-05-31"),
    ("L011", "C011",  4000, "OUTSTANDING", "2025-03-30", "2027-03-29"),
    ("L012", "C012",  7000, "OUTSTANDING", "2025-09-10", "2027-09-09"),
    ("L013", "C013",  6000, "OUTSTANDING", "2025-10-01", "2027-09-30"),
    ("L014", "C014",  4500, "OUTSTANDING", "2025-11-11", "2027-11-10"),
    ("L015", "C001",     0, "SETTLED",     "2023-01-01", "2024-12-31"),
]

GUARANTEES = [
    ("B001", "C001", 3000, "ACTIVE"),
    ("B002", "C003", 1000, "ACTIVE"),
    ("B003", "C004", 2000, "ACTIVE"),
    ("B004", "C005", 2000, "ACTIVE"),
    ("B005", "C007", 2500, "ACTIVE"),
    ("B006", "C008", 1500, "ACTIVE"),
    ("B007", "C009",  500, "ACTIVE"),
    ("B008", "C011", 1000, "ACTIVE"),
    ("B009", "C012", 2000, "ACTIVE"),
    ("B010", "C013", 1500, "ACTIVE"),
]

GUARANTEE_RELATIONS = [
    ("R001", "C005", "C010", 8000),  # Far East guaranteed by Prime Materials (overdue)
    ("R002", "C008", "C002", 5000),  # Global Supply guaranteed by Everbright Trading (overdue)
    ("R003", "C011", "C005", 4000),  # Prime Building guaranteed by Far East (in warning pool)
    ("R004", "C006", "C005", 3000),  # Far East BM guaranteed by Far East Construction
    ("R005", "C013", "C012", 2000),
    ("R006", "C003", "C001", 2000),
]

OVERDUE = [
    ("O001", "C010", 120, 5000, "DOUBTFUL"),
    ("O002", "C002",  95, 6000, "SUBPAR"),
    ("O003", "C016",  45,  800, "SPECIAL"),
    ("O004", "C018",  12,  300, "SPECIAL"),
]


def _insert_many(conn, table: str, rows: list[tuple]):
    """Insert rows with dynamically generated placeholders to avoid column mismatch."""
    if not rows:
        return
    n_cols = len(rows[0])
    placeholders = ",".join(["?"] * n_cols)
    sql = f"INSERT INTO {table} VALUES ({placeholders})"
    conn.executemany(sql, rows)
    conn.commit()


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA_SQL)

    _insert_many(conn, "group_metadata", GROUPS)
    _insert_many(conn, "ecif_customer", CUSTOMERS)
    _insert_many(conn, "loan", LOANS)
    _insert_many(conn, "bank_guarantee", GUARANTEES)
    _insert_many(conn, "guarantee_relation", GUARANTEE_RELATIONS)
    _insert_many(conn, "overdue", OVERDUE)

    # Sanity check
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    tables = [r[0] for r in cur.fetchall()]
    print(f"[OK] Database created at: {DB_PATH}")
    print(f"[OK] Tables ({len(tables)}): {', '.join(tables)}")
    for t in tables:
        cnt = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        print(f"     - {t}: {cnt} rows")

    conn.close()


if __name__ == "__main__":
    init_db()
