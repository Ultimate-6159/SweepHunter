"""
🗄️ snapshot_config.py — บันทึก config snapshot ลง DB

ใช้ทุกครั้งที่เปลี่ยน config สำคัญ → จะสร้าง row ใหม่ในตาราง config_snapshots
พร้อม timestamp + hash → match กับ trades ภายหลังได้ว่า
"trade ID = N เกิดภายใต้ config snapshot ID = X"

ใช้สำหรับ A/B compare strategy:
  python snapshot_config.py "เปิด trail + RR 2.0"
"""
from __future__ import annotations
import hashlib
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CFG = ROOT / "config.json"
DB = ROOT / "data" / "db" / "hyper_trades.sqlite"

SCHEMA = """
CREATE TABLE IF NOT EXISTS config_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc TEXT NOT NULL,
    label TEXT,
    config_hash TEXT NOT NULL,
    risk_per_trade_pct REAL,
    sl_atr_mult REAL,
    tp_atr_mult REAL,
    smart_trailing_enabled INTEGER,
    be_trigger_atr REAL,
    series_loss_cap_pct REAL,
    series_loss_cap_action TEXT,
    hourly_lot_mult_enabled INTEGER,
    full_config_json TEXT
);
CREATE INDEX IF NOT EXISTS ix_config_ts ON config_snapshots(ts_utc);
"""


def main() -> int:
    label = sys.argv[1] if len(sys.argv) > 1 else "manual snapshot"

    if not CFG.exists():
        print(f"❌ {CFG} not found")
        return 1
    cfg = json.loads(CFG.read_text(encoding="utf-8"))
    raw = json.dumps(cfg, sort_keys=True)
    h = hashlib.sha256(raw.encode()).hexdigest()[:16]

    # extract key fields
    t = cfg.get("trading", {})
    a = cfg.get("account_scaling", {})
    s = cfg.get("smart_trailing", {})
    rf = cfg.get("risk_filters", {}).get("series_loss_cap", {})
    hl = cfg.get("hourly_lot_multiplier", {})

    DB.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(DB)) as conn:
        conn.executescript(SCHEMA)
        # check duplicate hash
        existing = conn.execute(
            "SELECT id, label, ts_utc FROM config_snapshots WHERE config_hash=? ORDER BY id DESC LIMIT 1",
            (h,)).fetchone()
        if existing:
            print(f"⚠️  Config นี้บันทึกแล้ว (id={existing[0]}, '{existing[1]}', {existing[2][:16]})")
            ans = input("บันทึกซ้ำ? (y/N): ").strip().lower()
            if ans != "y":
                print("ยกเลิก — ใช้ snapshot เดิม")
                return 0

        cur = conn.execute("""
            INSERT INTO config_snapshots
              (ts_utc, label, config_hash,
               risk_per_trade_pct, sl_atr_mult, tp_atr_mult,
               smart_trailing_enabled, be_trigger_atr,
               series_loss_cap_pct, series_loss_cap_action,
               hourly_lot_mult_enabled, full_config_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now(timezone.utc).isoformat(), label, h,
            a.get("risk_per_trade_pct"),
            t.get("sl_atr_mult"), t.get("tp_atr_mult"),
            int(s.get("enabled", False)), s.get("be_trigger_atr"),
            rf.get("max_loss_pct_of_balance"), rf.get("action", "close"),
            int(hl.get("enabled", False)), raw,
        ))
        sid = cur.lastrowid
        conn.commit()

    print(f"\n✅ Snapshot saved: id={sid}  hash={h}  label='{label}'")
    print(f"\n📋 Key settings:")
    print(f"   risk_per_trade  : {a.get('risk_per_trade_pct')}%")
    print(f"   SL/TP           : {t.get('sl_atr_mult')}/{t.get('tp_atr_mult')} (RR={t.get('tp_atr_mult')/t.get('sl_atr_mult'):.2f})")
    print(f"   smart_trailing  : {s.get('enabled')} (trigger={s.get('be_trigger_atr')}×ATR)")
    print(f"   loss_cap        : {rf.get('max_loss_pct_of_balance')}% / {rf.get('action')}")
    print(f"   hourly_lot_mult : {hl.get('enabled')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
