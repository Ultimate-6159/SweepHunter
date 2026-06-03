"""
fix_db_timestamps.py — แก้ closed_at_utc ใน DB ที่เก็บผิด (MT5 epoch ไม่ได้ลบ skew)

ปัญหา: deal.time จาก MT5 (Vantage) เร็วกว่า UTC จริง ~3 ชม.
  เก็บด้วย fromtimestamp(deal.time) → ได้เวลา broker แต่ label ว่า UTC
  รายงาน heatmap บวก +3 อีกที → เลื่อนไปชั่วโมง 18 แทน 15

Usage:
  python fix_db_timestamps.py           # dry-run
  python fix_db_timestamps.py --apply   # แก้ DB จริง (backup ก่อน)
"""
from __future__ import annotations

import json
import shutil
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent
DB = ROOT / "data/db/hyper_trades.sqlite"
APPLY = "--apply" in sys.argv


def load_offset() -> int:
    try:
        cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
        return int((cfg.get("session_weighting") or {}).get("broker_offset_hours", 3))
    except Exception:
        return 3


def main() -> int:
    if not DB.exists():
        print(f"[ERR] DB not found: {DB}")
        return 1

    off = load_offset()
    fix_delta = timedelta(hours=off)

    conn = sqlite3.connect(str(DB))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT id, ticket, closed_at_utc, pnl
        FROM decisions
        WHERE status IN ('WIN','LOSS') AND closed_at_utc IS NOT NULL AND ticket IS NOT NULL
        ORDER BY closed_at_utc DESC
        """
    ).fetchall()

    updates: list[tuple[str, int]] = []
    print(f"Broker offset: UTC+{off}")
    print(f"Will subtract {off}h from closed_at_utc ({len(rows)} rows)\n")
    print("Sample (before → after → broker slot after fix):")
    for i, r in enumerate(rows):
        old_s = str(r["closed_at_utc"]).replace("Z", "+00:00")
        old = datetime.fromisoformat(old_s)
        if old.tzinfo is None:
            old = old.replace(tzinfo=timezone.utc)
        new = old - fix_delta
        updates.append((new.isoformat(), int(r["id"])))
        if i < 8:
            bd = new + fix_delta
            print(
                f"  #{r['ticket']} ${r['pnl']:+.0f} | "
                f"{old.strftime('%H:%M')} (old) → {new.strftime('%H:%M')} UTC → "
                f"{bd.strftime('%H:%M')} broker"
            )

    if not APPLY:
        print(f"\n[DRY-RUN] ไม่ได้แก้ DB — รัน python fix_db_timestamps.py --apply")
        conn.close()
        return 0

    bak = DB.with_suffix(f".pre_timefix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.sqlite")
    shutil.copy2(DB, bak)
    print(f"\n[OK] Backup: {bak.name}")

    conn.executemany("UPDATE decisions SET closed_at_utc=? WHERE id=?", updates)
    conn.commit()
    conn.close()
    print(f"[OK] Updated {len(updates)} rows")
    print("     รัน analyze_lot_multipliers.bat + strategy_report.bat ใหม่")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
