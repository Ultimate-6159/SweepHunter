"""
async_db_manager.py
====================
Asynchronous SQLite Logger + Selective Webhook Broadcaster
-----------------------------------------------------------
หลักการ (SRD):
  - SQLite write ทุกตัว enqueue ใน Queue, worker thread เขียนแบบ batch
    -> main loop (ความถี่ M1) ไม่ต้องรอ disk I/O
  - Webhook ส่งเฉพาะ event สำคัญ:
        1) series_closed      (รวบไม้ Martingale ปิดจบซีรีส์)
        2) summary_12h        (สรุปผลทุก 12 ชม.)
    เพื่อรักษา Zero-Touch (ไม่รบกวนผู้ใช้)
"""
from __future__ import annotations
import json
import queue
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional

import requests

from .config import Config
from .logger import get_logger
from .paths import db_path

log = get_logger("db")


SCHEMA = """
CREATE TABLE IF NOT EXISTS series (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    opened_at_utc TEXT NOT NULL,
    closed_at_utc TEXT,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,                    -- BUY/SELL
    steps INTEGER NOT NULL DEFAULT 1,
    total_volume REAL,
    avg_entry_price REAL,
    final_pnl REAL,
    status TEXT NOT NULL DEFAULT 'OPEN',   -- OPEN / CLOSED_TP / CLOSED_EQUITY_STOP / CLOSED_MANUAL
    notes TEXT
);

CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc TEXT NOT NULL,
    series_id INTEGER,
    step INTEGER,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    prediction INTEGER NOT NULL,
    confidence REAL,
    role TEXT,                             -- PRIMARY / RECOVERY
    spread_points REAL,
    atr REAL,
    ticket INTEGER,
    entry_price REAL,
    sl REAL,
    tp REAL,
    volume REAL,
    status TEXT NOT NULL DEFAULT 'PENDING',
    pnl REAL,
    closed_at_utc TEXT,
    close_price REAL,
    notes TEXT,
    FOREIGN KEY(series_id) REFERENCES series(id)
);
CREATE INDEX IF NOT EXISTS ix_decisions_series ON decisions(series_id);
CREATE INDEX IF NOT EXISTS ix_decisions_ticket ON decisions(ticket);
CREATE INDEX IF NOT EXISTS ix_decisions_status ON decisions(status);
CREATE INDEX IF NOT EXISTS ix_series_status ON series(status);
"""


# ============================================================ Async SQLite
class AsyncDBManager:
    """
    Producer/Consumer SQLite writer.
      - main thread เรียก insert_*/update_* แบบ non-blocking (enqueue เท่านั้น)
      - background worker drain queue + commit batch ทุก 0.5s
    Read query (ที่ block ได้) ยังคงเป็น sync (ใช้ context _conn()).
    """

    def __init__(self) -> None:
        fname = Config.section("database").get("filename", "hyper_trades.sqlite")
        self.path = db_path(fname)
        self._q: queue.Queue = queue.Queue(maxsize=10_000)
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._next_id_lock = threading.Lock()
        self._init_schema()
        self._worker = threading.Thread(target=self._run_worker, name="db-writer", daemon=True)
        self._worker.start()
        log.info("AsyncDB ready -> %s", self.path)

    # ---------------- low level
    @contextmanager
    def _conn(self):
        with self._lock:
            con = sqlite3.connect(self.path, timeout=10, isolation_level=None)
            con.row_factory = sqlite3.Row
            try:
                con.execute("PRAGMA journal_mode=WAL;")
                con.execute("PRAGMA synchronous=NORMAL;")
                yield con
            finally:
                con.close()

    def _init_schema(self) -> None:
        with self._conn() as c:
            c.executescript(SCHEMA)

    # ---------------- worker
    def _run_worker(self) -> None:
        batch: List[tuple] = []
        last_flush = time.time()
        while not self._stop.is_set() or not self._q.empty():
            try:
                item = self._q.get(timeout=0.2)
                batch.append(item)
            except queue.Empty:
                pass
            if batch and (len(batch) >= 50 or time.time() - last_flush > 0.5):
                self._flush(batch)
                batch = []
                last_flush = time.time()
        if batch:
            self._flush(batch)

    def _flush(self, batch: Iterable[tuple]) -> None:
        try:
            with self._conn() as c:
                c.execute("BEGIN")
                for sql, params in batch:
                    c.execute(sql, params)
                c.execute("COMMIT")
        except Exception as e:
            log.exception("DB flush failed: %s", e)

    def shutdown(self) -> None:
        self._stop.set()
        self._worker.join(timeout=5)

    # ---------------- ID allocation (sync — quick)
    def _alloc_id(self, table: str) -> int:
        with self._next_id_lock, self._conn() as c:
            cur = c.execute(f"SELECT COALESCE(MAX(id),0)+1 FROM {table}")
            return int(cur.fetchone()[0])

    # =========================================================== series API
    def open_series(self, symbol: str, side: str) -> int:
        sid = self._alloc_id("series")
        sql = ("INSERT INTO series (id, opened_at_utc, symbol, side, status) "
               "VALUES (?, ?, ?, ?, 'OPEN')")
        self._q.put((sql, (sid, datetime.now(timezone.utc).isoformat(), symbol, side.upper())))
        return sid

    def update_series(self, series_id: int, **fields) -> None:
        if not fields:
            return
        sets = ",".join(f"{k}=?" for k in fields)
        sql = f"UPDATE series SET {sets} WHERE id=?"
        self._q.put((sql, list(fields.values()) + [series_id]))

    def close_series(self, series_id: int, status: str, final_pnl: float,
                     total_volume: float, avg_entry_price: float, notes: str = "") -> None:
        self.update_series(
            series_id,
            status=status,
            final_pnl=float(final_pnl),
            total_volume=float(total_volume),
            avg_entry_price=float(avg_entry_price),
            closed_at_utc=datetime.now(timezone.utc).isoformat(),
            notes=notes,
        )

    # =========================================================== decision API
    def insert_decision(self, **fields) -> int:
        fields.setdefault("ts_utc", datetime.now(timezone.utc).isoformat())
        did = self._alloc_id("decisions")
        fields["id"] = did
        cols = ",".join(fields.keys())
        ph = ",".join("?" for _ in fields)
        sql = f"INSERT INTO decisions ({cols}) VALUES ({ph})"
        self._q.put((sql, list(fields.values())))
        return did

    def update_decision(self, decision_id: int, **fields) -> None:
        if not fields:
            return
        sets = ",".join(f"{k}=?" for k in fields)
        sql = f"UPDATE decisions SET {sets} WHERE id=?"
        self._q.put((sql, list(fields.values()) + [decision_id]))

    # =========================================================== read API
    def open_decisions(self) -> List[sqlite3.Row]:
        with self._conn() as c:
            return list(c.execute(
                "SELECT * FROM decisions WHERE status IN ('OPEN','PENDING') AND ticket IS NOT NULL"
            ))

    def count_settled(self) -> int:
        with self._conn() as c:
            return int(c.execute(
                "SELECT COUNT(*) FROM decisions WHERE status IN ('WIN','LOSS')"
            ).fetchone()[0])

    def count_orders_today_utc(self, only_primary: bool = True) -> int:
        midnight = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        sql = ("SELECT COUNT(*) FROM decisions WHERE ts_utc >= ? "
               "AND status NOT IN ('SKIP','ERROR','PENDING')")
        if only_primary:
            sql += " AND role='PRIMARY'"
        with self._conn() as c:
            return int(c.execute(sql, (midnight.isoformat(),)).fetchone()[0])

    def summary_since(self, since: datetime) -> Dict[str, Any]:
        with self._conn() as c:
            cur = c.execute(
                "SELECT COUNT(*) n, "
                "SUM(CASE WHEN final_pnl>0 THEN 1 ELSE 0 END) wins, "
                "SUM(CASE WHEN final_pnl<=0 THEN 1 ELSE 0 END) losses, "
                "COALESCE(SUM(final_pnl),0) pnl "
                "FROM series WHERE closed_at_utc >= ?",
                (since.isoformat(),),
            )
            r = cur.fetchone()
            n, wins, losses, pnl = (r[0] or 0), (r[1] or 0), (r[2] or 0), (r[3] or 0.0)
            cur2 = c.execute(
                "SELECT COUNT(*) FROM decisions WHERE ts_utc >= ? AND role='PRIMARY' "
                "AND status NOT IN ('SKIP','ERROR','PENDING')",
                (since.isoformat(),))
            n_orders = int(cur2.fetchone()[0])
        return {
            "since_utc": since.isoformat(),
            "series_closed": n,
            "series_wins": wins,
            "series_losses": losses,
            "series_winrate": (wins / n) if n else 0.0,
            "primary_orders_opened": n_orders,
            "total_pnl": float(pnl),
        }


# ============================================================ Webhook
class WebhookBroadcaster:
    """ส่งเฉพาะ event สำคัญ (series_closed, summary_12h)."""

    def __init__(self) -> None:
        self.cfg = Config.section("webhook")
        self._summary_lock = threading.Lock()
        self._next_summary = self._compute_next_summary()

    def _compute_next_summary(self) -> datetime:
        hrs = float(self.cfg.get("summary_interval_hours", 12))
        return datetime.now(timezone.utc) + timedelta(hours=hrs)

    def _post(self, payload: Dict[str, Any]) -> None:
        if not self.cfg.get("enabled"):
            return
        url = self.cfg.get("url")
        if not url:
            return
        try:
            requests.post(url, json=payload,
                          timeout=float(self.cfg.get("timeout_sec", 5)))
        except Exception as e:
            log.warning("Webhook failed: %s", e)

    def series_closed(self, data: Dict[str, Any]) -> None:
        body = {"event": "series_closed",
                "ts_utc": datetime.now(timezone.utc).isoformat(),
                "data": data}
        self._post(body)

    def maybe_send_summary(self, db: AsyncDBManager) -> bool:
        with self._summary_lock:
            now = datetime.now(timezone.utc)
            if now < self._next_summary:
                return False
            since = now - timedelta(hours=float(self.cfg.get("summary_interval_hours", 12)))
            summary = db.summary_since(since)
            self._post({"event": "summary_12h",
                        "ts_utc": now.isoformat(),
                        "data": summary})
            log.info("[SUMMARY] sent: %s", json.dumps(summary, default=str))
            self._next_summary = self._compute_next_summary()
            return True
