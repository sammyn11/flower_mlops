"""Tiny SQLite layer that records every uploaded training image."""
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from . import config as C


@contextmanager
def _conn():
    C.DATA_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(C.DB_PATH, timeout=30)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db():
    with _conn() as con:
        con.execute("""CREATE TABLE IF NOT EXISTS uploads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT, label TEXT, path TEXT,
            uploaded_at TEXT, used_in_version INTEGER)""")


def add_upload(filename, label, path):
    init_db()
    with _conn() as con:
        con.execute("INSERT INTO uploads (filename,label,path,uploaded_at,used_in_version) VALUES (?,?,?,?,NULL)",
                    (filename, label, str(path), datetime.utcnow().isoformat()))


def pending_count():
    init_db()
    with _conn() as con:
        return con.execute("SELECT COUNT(*) FROM uploads WHERE used_in_version IS NULL").fetchone()[0]


def total_count():
    init_db()
    with _conn() as con:
        return con.execute("SELECT COUNT(*) FROM uploads").fetchone()[0]


def counts_by_label():
    init_db()
    with _conn() as con:
        rows = con.execute("SELECT label, COUNT(*) c FROM uploads GROUP BY label").fetchall()
    return {r["label"]: r["c"] for r in rows}


def mark_used(version):
    init_db()
    with _conn() as con:
        con.execute("UPDATE uploads SET used_in_version=? WHERE used_in_version IS NULL", (version,))
