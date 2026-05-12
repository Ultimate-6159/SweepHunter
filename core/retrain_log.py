"""
🗃️  retrain_log.py
==================
Log + query model retrain events to the same SQLite DB.

Schema:
  CREATE TABLE model_retrains (
    id INTEGER PRIMARY KEY,
    ts_utc TEXT,
    model_filename TEXT,
    model_hash TEXT,
    rows_trained INTEGER,
    cv_acc REAL,
    oos_acc REAL,
    accepted INTEGER,        -- 1=acceptance gate accepted, 0=rejected
    notes TEXT
  );

  Also adds: ALTER TABLE config_snapshots ADD COLUMN notes TEXT;
"""
from __future__ import annotations
import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .logger import get_logger
from .paths import db_path

log = get_logger("retrain")


def _db() -> Path:
    from .config import Config
    return db_path(Config.section("database").get("filename", "hyper_trades.sqlite"))


def ensure_schema() -> None:
    """สร้าง table + คอลัมน์ที่ขาดอยู่ — idempotent (รันซ้ำได้)"""
    with sqlite3.connect(str(_db())) as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS model_retrains (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts_utc TEXT NOT NULL,
                model_filename TEXT,
                model_hash TEXT,
                rows_trained INTEGER,
                cv_acc REAL,
                oos_acc REAL,
                accepted INTEGER DEFAULT 1,
                notes TEXT
            )
        """)
        # add notes column to config_snapshots if missing
        cols = [r[1] for r in c.execute("PRAGMA table_info(config_snapshots)").fetchall()]
        if "notes" not in cols:
            try:
                c.execute("ALTER TABLE config_snapshots ADD COLUMN notes TEXT")
                log.info("[migrate] added notes column to config_snapshots")
            except Exception as e:
                log.debug("ALTER skip: %s", e)
        c.commit()


def _hash_file(p: Path) -> str:
    try:
        return hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    except Exception:
        return ""


def log_retrain(model_path: Path, rows: int, cv_acc: float,
                oos_acc: float, accepted: bool = True,
                notes: str = "") -> int:
    """บันทึก retrain event ลง DB — เรียกหลัง joblib.dump() สำเร็จ"""
    ensure_schema()
    h = _hash_file(model_path) if model_path.exists() else ""
    with sqlite3.connect(str(_db())) as c:
        cur = c.execute("""
            INSERT INTO model_retrains
              (ts_utc, model_filename, model_hash, rows_trained, cv_acc, oos_acc, accepted, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now(timezone.utc).isoformat(),
            model_path.name, h, int(rows),
            float(cv_acc), float(oos_acc),
            1 if accepted else 0, notes,
        ))
        c.commit()
        rid = int(cur.lastrowid)
    log.info("[retrain] 🗃️ logged event id=%d hash=%s rows=%d oos=%.4f accepted=%s",
             rid, h, rows, oos_acc, accepted)
    return rid


def set_snapshot_notes(snapshot_id: int, notes: str) -> bool:
    """ใส่ notes ให้ snapshot ที่มีอยู่แล้ว"""
    ensure_schema()
    with sqlite3.connect(str(_db())) as c:
        cur = c.execute("UPDATE config_snapshots SET notes=? WHERE id=?",
                        (notes, snapshot_id))
        c.commit()
        return cur.rowcount > 0


def get_retrains(limit: int = 20) -> list:
    ensure_schema()
    with sqlite3.connect(str(_db())) as c:
        c.row_factory = sqlite3.Row
        return [dict(r) for r in c.execute(
            "SELECT * FROM model_retrains ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()]
