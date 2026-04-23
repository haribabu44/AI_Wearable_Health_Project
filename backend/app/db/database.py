"""
db/database.py
--------------
Improvements over v1:
  - Thread-safe connection pool via queue.Queue
  - create_table() called ONCE at startup — not on every operation
  - Typed return values (list[dict] instead of raw tuples)
  - Centralised schema — adding a column means editing one place
"""

import json
import logging
import queue
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Generator

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_pool: queue.Queue[sqlite3.Connection] = queue.Queue()
_pool_lock = threading.Lock()
_pool_initialised = False


def _make_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(
        settings.db_path,
        detect_types=sqlite3.PARSE_DECLTYPES,
        check_same_thread=False,
    )
    conn.row_factory = sqlite3.Row   # rows behave like dicts
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    """Called ONCE at application startup from main.py lifespan."""
    global _pool_initialised
    with _pool_lock:
        if _pool_initialised:
            return

        for _ in range(settings.db_pool_size):
            _pool.put(_make_connection())

        conn = _pool.get()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS health_records (
                    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_name         TEXT,
                    steps                REAL    NOT NULL,
                    temperature          REAL    NOT NULL,
                    spo2                 REAL    NOT NULL,
                    glucose              REAL    NOT NULL,
                    bp                   REAL    NOT NULL,
                    age                  INTEGER,
                    bmi                  REAL,
                    sex                  INTEGER,
                    predicted_heart_rate REAL,
                    anomaly_status       TEXT,
                    health_score         INTEGER,
                    alerts               TEXT,
                    disease_predictions  TEXT,
                    timestamp            TEXT    NOT NULL
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_timestamp ON health_records(timestamp)"
            )
            conn.commit()
        finally:
            _pool.put(conn)

        _pool_initialised = True
        logger.info("Database pool initialised (size=%d)", settings.db_pool_size)


@contextmanager
def get_conn() -> Generator[sqlite3.Connection, None, None]:
    """
    Context manager that borrows a connection from the pool and returns
    it when done — even if an exception is raised.
    """
    conn = _pool.get(timeout=5)
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    finally:
        _pool.put(conn)


def insert_record(data: dict[str, Any]) -> int:
    """Insert a health record and return the new row id."""
    with get_conn() as conn:
        cursor = conn.execute(
            """
            INSERT INTO health_records (
                patient_name, steps, temperature, spo2, glucose, bp,
                age, bmi, sex,
                predicted_heart_rate, anomaly_status, health_score,
                alerts, disease_predictions, timestamp
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                data.get("patient_name"),
                data["steps"],
                data["temperature"],
                data["spo2"],
                data["glucose"],
                data["bp"],
                data.get("age"),
                data.get("bmi"),
                data.get("sex"),
                data.get("predicted_heart_rate"),
                data.get("anomaly_status"),
                data.get("health_score"),
                json.dumps(data.get("alerts", [])),
                json.dumps(data.get("disease_predictions", {})),
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]


def get_records(limit: int = 100, offset: int = 0) -> list[dict]:
    """Return the most recent records as a list of plain dicts."""
    with get_conn() as conn:
        cursor = conn.execute(
            """
            SELECT * FROM health_records
            ORDER BY id DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        )
        rows = cursor.fetchall()
    return [dict(row) for row in rows]


def get_record_count() -> int:
    with get_conn() as conn:
        cursor = conn.execute("SELECT COUNT(*) FROM health_records")
        return cursor.fetchone()[0]
