"""SQLite persistence helpers for users, scans, and safety violations."""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

ROOT_DIR = Path(__file__).resolve().parents[1]
DB_PATH = Path(os.getenv("SAFETY_DB_PATH", ROOT_DIR / "data" / "safety_monitoring.db"))


def get_connection() -> sqlite3.Connection:
    """Return a row-aware SQLite connection, creating its parent directory."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


@contextmanager
def connection_scope() -> Iterator[sqlite3.Connection]:
    connection = get_connection()
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def init_database() -> None:
    """Create tables and indexes when the application starts."""
    with connection_scope() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('admin', 'supervisor')),
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id),
                source_type TEXT NOT NULL CHECK(source_type IN ('image', 'video', 'webcam')),
                filename TEXT NOT NULL,
                timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                total_persons INTEGER NOT NULL DEFAULT 0,
                total_violations INTEGER NOT NULL DEFAULT 0,
                compliance_pct REAL NOT NULL DEFAULT 100
            );
            CREATE TABLE IF NOT EXISTS violations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_id INTEGER NOT NULL REFERENCES scans(id) ON DELETE CASCADE,
                violation_type TEXT NOT NULL,
                confidence REAL NOT NULL,
                frame_number INTEGER,
                timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                bbox TEXT,
                evidence_path TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_scans_timestamp ON scans(timestamp);
            CREATE INDEX IF NOT EXISTS idx_scans_user ON scans(user_id);
            CREATE INDEX IF NOT EXISTS idx_violations_scan ON violations(scan_id);
            CREATE INDEX IF NOT EXISTS idx_violations_timestamp ON violations(timestamp);
            """
        )
        columns = {row["name"] for row in connection.execute("PRAGMA table_info(violations)")}
        if "evidence_path" not in columns:
            connection.execute("ALTER TABLE violations ADD COLUMN evidence_path TEXT")


def create_user(username: str, email: str, password_hash: str, role: str = "supervisor") -> int:
    with connection_scope() as connection:
        cursor = connection.execute(
            "INSERT INTO users (username, email, password_hash, role) VALUES (?, ?, ?, ?)",
            (username.strip(), email.strip().lower(), password_hash, role),
        )
        return int(cursor.lastrowid)


def get_user_by_username(username: str) -> dict[str, Any] | None:
    with connection_scope() as connection:
        row = connection.execute("SELECT * FROM users WHERE username = ?", (username.strip(),)).fetchone()
        return dict(row) if row else None


def get_user_by_email(email: str) -> dict[str, Any] | None:
    with connection_scope() as connection:
        row = connection.execute("SELECT * FROM users WHERE email = ?", (email.strip().lower(),)).fetchone()
        return dict(row) if row else None


def ensure_default_admin(password_hash: str) -> None:
    """Create the bootstrap admin only when no admin account exists."""
    with connection_scope() as connection:
        exists = connection.execute("SELECT 1 FROM users WHERE role = 'admin' LIMIT 1").fetchone()
        if not exists:
            username = os.getenv("DEFAULT_ADMIN_USERNAME", "admin")
            email = os.getenv("DEFAULT_ADMIN_EMAIL", "admin@safety.local")
            connection.execute(
                "INSERT OR IGNORE INTO users (username, email, password_hash, role) VALUES (?, ?, ?, 'admin')",
                (username, email, password_hash),
            )


def save_scan(
    user_id: int,
    source_type: str,
    filename: str,
    total_persons: int,
    total_violations: int,
    compliance_pct: float,
    violations: list[dict[str, Any]],
) -> int:
    """Persist a scan and all its violation detections atomically."""
    with connection_scope() as connection:
        cursor = connection.execute(
            """INSERT INTO scans
               (user_id, source_type, filename, total_persons, total_violations, compliance_pct)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (user_id, source_type, filename, total_persons, total_violations, compliance_pct),
        )
        scan_id = int(cursor.lastrowid)
        connection.executemany(
            """INSERT INTO violations
               (scan_id, violation_type, confidence, frame_number, bbox, evidence_path)
               VALUES (?, ?, ?, ?, ?, ?)""",
            [
                (
                    scan_id,
                    item["violation_type"],
                    float(item.get("confidence", 0)),
                    item.get("frame_number"),
                    item.get("bbox"),
                    item.get("evidence_path"),
                )
                for item in violations
            ],
        )
        return scan_id


def delete_violation(violation_id: int) -> None:
    with connection_scope() as connection:
        row = connection.execute("SELECT scan_id FROM violations WHERE id = ?", (violation_id,)).fetchone()
        if not row:
            return
        connection.execute("DELETE FROM violations WHERE id = ?", (violation_id,))
        summary = connection.execute(
            """SELECT COUNT(*) AS count,
                      COUNT(DISTINCT COALESCE(frame_number, -1) || ':' || COALESCE(bbox, '')) AS subjects
               FROM violations WHERE scan_id = ?""",
            (row["scan_id"],),
        ).fetchone()
        scan = connection.execute(
            "SELECT total_persons FROM scans WHERE id = ?", (row["scan_id"],)
        ).fetchone()
        persons, count = int(scan["total_persons"]), int(summary["subjects"])
        compliance = 100.0 if persons <= 0 else max(0.0, 100.0 * (persons - min(persons, count)) / persons)
        connection.execute(
            "UPDATE scans SET total_violations = ?, compliance_pct = ? WHERE id = ?",
            (int(summary["count"]), compliance, row["scan_id"]),
        )


def fetch_frame(query: str, parameters: tuple[Any, ...] = ()):
    """Return query rows as a pandas DataFrame for charts and tables."""
    import pandas as pd

    with connection_scope() as connection:
        return pd.read_sql_query(query, connection, params=parameters)
