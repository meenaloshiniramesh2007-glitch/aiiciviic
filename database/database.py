"""
database/database.py — AI Civic Guardian
=========================================
Auto-detects environment:
  - Local dev  → SQLite  (no DATABASE_URL set)
  - Render     → PostgreSQL (DATABASE_URL set by Render)

All SQL is written in a dialect compatible with both databases.
The only differences handled here:
  - Placeholder: SQLite uses ?   PostgreSQL uses %s
  - AUTOINCREMENT: SQLite        SERIAL / BIGSERIAL: PostgreSQL
  - DATETIME: SQLite             TIMESTAMP: PostgreSQL
"""

import os
import sqlite3

# ── Detect which database to use ──────────────────────────────────────────────
DATABASE_URL = os.environ.get("DATABASE_URL", "")

# Render provides postgres:// but psycopg2 needs postgresql://
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

USE_POSTGRES = bool(DATABASE_URL)

# Local SQLite path (used when DATABASE_URL is not set)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SQLITE_PATH = os.path.join(BASE_DIR, "civic_guardian.db")


# ── Placeholder symbol ────────────────────────────────────────────────────────
# Use this everywhere instead of hardcoding ? or %s
PH = "%s" if USE_POSTGRES else "?"


def get_connection():
    """
    Return a database connection.
    SQLite connection has row_factory set so columns are accessible by name.
    PostgreSQL uses psycopg2 with a DictCursor-like wrapper (RowWrapper).
    """
    if USE_POSTGRES:
        import psycopg2
        import psycopg2.extras
        conn = psycopg2.connect(DATABASE_URL)
        conn.autocommit = False
        return _PgConnection(conn)
    else:
        conn = sqlite3.connect(SQLITE_PATH)
        conn.row_factory = sqlite3.Row
        return conn


# ─────────────────────────────────────────────────────────────────────────────
# PostgreSQL wrapper — makes psycopg2 behave like sqlite3 for our code
# (column access by name, .execute() returns cursor, .commit(), .close())
# ─────────────────────────────────────────────────────────────────────────────
class _PgRow:
    """Wraps a psycopg2 RealDictRow so it behaves like sqlite3.Row."""
    def __init__(self, row_dict):
        self._d = dict(row_dict) if row_dict else {}

    def __getitem__(self, key):
        return self._d[key]

    def __contains__(self, key):
        return key in self._d

    def get(self, key, default=None):
        return self._d.get(key, default)

    def keys(self):
        return self._d.keys()

    def __iter__(self):
        return iter(self._d.values())


class _PgCursor:
    """Wraps psycopg2 cursor so .fetchone() / .fetchall() return _PgRow."""
    def __init__(self, cursor):
        self._cur = cursor

    @property
    def lastrowid(self):
        self._cur.execute("SELECT lastval()")
        return self._cur.fetchone()[0]

    def fetchone(self):
        row = self._cur.fetchone()
        return _PgRow(row) if row else None

    def fetchall(self):
        return [_PgRow(r) for r in self._cur.fetchall()]

    def __iter__(self):
        for row in self._cur:
            yield _PgRow(row)


class _PgConnection:
    """Wraps psycopg2 connection to match sqlite3's interface."""
    def __init__(self, conn):
        self._conn = conn
        self._cur  = conn.cursor(cursor_factory=__import__("psycopg2").extras.RealDictCursor)

    def execute(self, sql, params=()):
        # Convert ? placeholders to %s for PostgreSQL
        pg_sql = sql.replace("?", "%s")
        self._cur.execute(pg_sql, params)
        return _PgCursor(self._cur)

    def executemany(self, sql, seq):
        pg_sql = sql.replace("?", "%s")
        self._cur.executemany(pg_sql, seq)

    def commit(self):
        self._conn.commit()

    def close(self):
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ─────────────────────────────────────────────────────────────────────────────
# Table creation — compatible SQL for both SQLite and PostgreSQL
# ─────────────────────────────────────────────────────────────────────────────
def _sqlite_sql():
    return {
        "users": """
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER  PRIMARY KEY AUTOINCREMENT,
                name          TEXT     NOT NULL,
                email         TEXT     NOT NULL UNIQUE,
                phone         TEXT,
                password      TEXT     NOT NULL,
                preferred_lang TEXT    DEFAULT 'en'
            )""",
        "complaints": """
            CREATE TABLE IF NOT EXISTS complaints (
                id                INTEGER  PRIMARY KEY AUTOINCREMENT,
                user_id           INTEGER,
                user_name         TEXT,
                user_phone        TEXT,
                department        TEXT     NOT NULL,
                category          TEXT,
                location          TEXT     NOT NULL,
                description       TEXT     NOT NULL,
                description_lang  TEXT     DEFAULT 'en',
                image             TEXT,
                video             TEXT,
                voice_transcript  TEXT,
                status            TEXT     NOT NULL DEFAULT 'Pending',
                priority          TEXT     DEFAULT 'Normal',
                ai_category       TEXT,
                ai_confidence     REAL,
                ai_image_label    TEXT,
                auto_letter       TEXT,
                letter_sent_at    DATETIME,
                submitted_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at        DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )""",
        "auto_letters": """
            CREATE TABLE IF NOT EXISTS auto_letters (
                id             INTEGER  PRIMARY KEY AUTOINCREMENT,
                complaint_id   INTEGER  NOT NULL,
                department     TEXT     NOT NULL,
                letter_text    TEXT     NOT NULL,
                generated_at   DATETIME DEFAULT CURRENT_TIMESTAMP,
                acknowledged   INTEGER  DEFAULT 0,
                FOREIGN KEY (complaint_id) REFERENCES complaints(id)
            )""",
    }


def _postgres_sql():
    return {
        "users": """
            CREATE TABLE IF NOT EXISTS users (
                id            SERIAL       PRIMARY KEY,
                name          TEXT         NOT NULL,
                email         TEXT         NOT NULL UNIQUE,
                phone         TEXT,
                password      TEXT         NOT NULL,
                preferred_lang TEXT        DEFAULT 'en'
            )""",
        "complaints": """
            CREATE TABLE IF NOT EXISTS complaints (
                id                SERIAL       PRIMARY KEY,
                user_id           INTEGER,
                user_name         TEXT,
                user_phone        TEXT,
                department        TEXT         NOT NULL,
                category          TEXT,
                location          TEXT         NOT NULL,
                description       TEXT         NOT NULL,
                description_lang  TEXT         DEFAULT 'en',
                image             TEXT,
                video             TEXT,
                voice_transcript  TEXT,
                status            TEXT         NOT NULL DEFAULT 'Pending',
                priority          TEXT         DEFAULT 'Normal',
                ai_category       TEXT,
                ai_confidence     REAL,
                ai_image_label    TEXT,
                auto_letter       TEXT,
                letter_sent_at    TIMESTAMP,
                submitted_at      TIMESTAMP    DEFAULT NOW(),
                updated_at        TIMESTAMP    DEFAULT NOW(),
                FOREIGN KEY (user_id) REFERENCES users(id)
            )""",
        "auto_letters": """
            CREATE TABLE IF NOT EXISTS auto_letters (
                id             SERIAL       PRIMARY KEY,
                complaint_id   INTEGER      NOT NULL,
                department     TEXT         NOT NULL,
                letter_text    TEXT         NOT NULL,
                generated_at   TIMESTAMP    DEFAULT NOW(),
                acknowledged   INTEGER      DEFAULT 0,
                FOREIGN KEY (complaint_id) REFERENCES complaints(id)
            )""",
    }


def create_tables():
    conn = get_connection()
    schemas = _postgres_sql() if USE_POSTGRES else _sqlite_sql()

    for table, ddl in schemas.items():
        conn.execute(ddl)

    # ── SQLite-only: add missing columns to existing DBs ─────────────────────
    if not USE_POSTGRES:
        _sqlite_migrate(conn)

    conn.commit()
    conn.close()
    mode = "PostgreSQL" if USE_POSTGRES else "SQLite"
    print(f"[DB] Tables created / verified. ({mode})")


def _sqlite_migrate(conn):
    """Add any new columns to existing SQLite tables without breaking them."""
    # users
    user_cols = [r[1] for r in conn.execute("PRAGMA table_info(users)").fetchall()]
    if "preferred_lang" not in user_cols:
        conn.execute("ALTER TABLE users ADD COLUMN preferred_lang TEXT DEFAULT 'en'")
        print("[DB] Added column: users.preferred_lang")

    # complaints
    comp_cols = [r[1] for r in conn.execute("PRAGMA table_info(complaints)").fetchall()]
    new_cols = {
        "video"            : "TEXT",
        "voice_transcript" : "TEXT",
        "description_lang" : "TEXT DEFAULT 'en'",
        "priority"         : "TEXT DEFAULT 'Normal'",
        "auto_letter"      : "TEXT",
        "letter_sent_at"   : "DATETIME",
        "user_phone"       : "TEXT",
    }
    for col, col_type in new_cols.items():
        if col not in comp_cols:
            conn.execute(f"ALTER TABLE complaints ADD COLUMN {col} {col_type}")
            print(f"[DB] Added column: complaints.{col}")
