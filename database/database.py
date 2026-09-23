"""
database/database.py — AI Civic Guardian
SQLite database connection and table creation.
"""

import sqlite3
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH  = os.path.join(BASE_DIR, "civic_guardian.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def create_tables():
    conn = get_connection()
    c = conn.cursor()

    # Users table
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            name         TEXT    NOT NULL,
            email        TEXT    NOT NULL UNIQUE,
            phone        TEXT,
            password     TEXT    NOT NULL,
            preferred_lang TEXT  DEFAULT 'en'
        )
    """)

    # Complaints table — with video, voice transcript, auto-letter columns
    c.execute("""
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
        )
    """)

    # Auto-letters table — one per complaint, tracks dispatch to department
    c.execute("""
        CREATE TABLE IF NOT EXISTS auto_letters (
            id             INTEGER  PRIMARY KEY AUTOINCREMENT,
            complaint_id   INTEGER  NOT NULL,
            department     TEXT     NOT NULL,
            letter_text    TEXT     NOT NULL,
            generated_at   DATETIME DEFAULT CURRENT_TIMESTAMP,
            acknowledged   INTEGER  DEFAULT 0,
            FOREIGN KEY (complaint_id) REFERENCES complaints(id)
        )
    """)

    # Migrate existing users table safely
    user_cols = [row[1] for row in c.execute("PRAGMA table_info(users)").fetchall()]
    if "preferred_lang" not in user_cols:
        c.execute("ALTER TABLE users ADD COLUMN preferred_lang TEXT DEFAULT 'en'")
        print("[DB] Added column: users.preferred_lang")

    # Migrate existing complaints table safely (add new columns if missing)
    existing_cols = [row[1] for row in c.execute("PRAGMA table_info(complaints)").fetchall()]
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
        if col not in existing_cols:
            c.execute(f"ALTER TABLE complaints ADD COLUMN {col} {col_type}")
            print(f"[DB] Added column: complaints.{col}")

    conn.commit()
    conn.close()
    print("[DB] Tables created / verified.")
