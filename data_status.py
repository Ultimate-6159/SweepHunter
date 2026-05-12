"""
📊 data_status.py
==================
Data Lineage Dashboard — รู้ทันทีว่า:
  • Snapshot ไหนใช้ได้ (กำไร) / Snapshot ไหนเสีย
  • เริ่มเก็บ data เมื่อไหร่ จบเมื่อไหร่
  • Model retrain ที่ไหนบ้าง (timestamp model file)
  • "data era" ปัจจุบัน — กำไรสะสมตั้งแต่ snapshot นั้น

Usage:
  python data_status.py              # full timeline
  python data_status.py --from 4     # from snapshot id 4
  python data_status.py --since 7d   # last 7 days
"""
from __future__ import annotations
import argparse
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# force UTF-8 stdout (Windows cp1252 default breaks emoji)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "data" / "db" / "hyper_trades.sqlite"
MODEL_PATH = ROOT / "data" / "models" / "xgb_hyper_model.pkl"


def fmt_time(ts: str) -> str:
    if not ts: return "?"
    return ts[:19].replace("T", " ")


def relative_age(ts_str: str) -> str:
    try:
        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        delta = datetime.now(timezone.utc) - ts
        if delta.days > 0:
            return f"{delta.days}d ago"
        h = delta.seconds // 3600
        m = (delta.seconds % 3600) // 60
        if h > 0: return f"{h}h{m}m ago"
        return f"{m}m ago"
    except Exception:
        return "?"


def cmd_status(args):
    if not DB_PATH.exists():
        print(f"❌ DB not found: {DB_PATH}")
        return 1
    # ensure schema (notes column + retrain table)
    try:
        from core.retrain_log import ensure_schema
        ensure_schema()
    except Exception:
        pass
    c = sqlite3.connect(str(DB_PATH))
    c.row_factory = sqlite3.Row

    # ---------- HEADER ----------
    now = datetime.now(timezone.utc)
    print("=" * 90)
    print(f"📊 DATA LINEAGE DASHBOARD     UTC: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 90)

    # Model status
    if MODEL_PATH.exists():
        m_ts = datetime.fromtimestamp(MODEL_PATH.stat().st_mtime, tz=timezone.utc)
        m_age = relative_age(m_ts.isoformat())
        size_kb = MODEL_PATH.stat().st_size / 1024
        print(f"🤖 Model:  {MODEL_PATH.name}  ({size_kb:.0f} KB)  trained {m_age}  [{fmt_time(m_ts.isoformat())}]")
    else:
        print("🤖 Model:  ❌ NOT FOUND")

    # Live snapshot
    snaps = c.execute("SELECT id, label, ts_utc FROM config_snapshots ORDER BY id DESC LIMIT 1").fetchall()
    if snaps:
        s = snaps[0]
        print(f"🟢 Active snapshot: #{s['id']}  ({relative_age(s['ts_utc'])})  {s['label'][:60]}")

    # Halt status
    try:
        import json as _json
        recov = _json.loads(Path("data/recovery_state.json").read_text())
        halt_ts = float(recov.get("halted_until_ts", 0))
        now_ts = datetime.now(timezone.utc).timestamp()
        if halt_ts > now_ts:
            wait = halt_ts - now_ts
            until = datetime.fromtimestamp(halt_ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            print(f"⏸️  HALTED until {until}  (wait {wait/3600:.1f}h)  → run 6_clear_halt.bat to reset")
    except Exception:
        pass

    # ---------- SNAPSHOT TIMELINE ----------
    where, params = "1=1", []
    if args.from_id:
        where = "id >= ?"; params = [int(args.from_id)]
    elif args.since:
        if args.since.endswith("d"):
            cutoff = (now - timedelta(days=int(args.since[:-1]))).isoformat()
        elif args.since.endswith("h"):
            cutoff = (now - timedelta(hours=int(args.since[:-1]))).isoformat()
        else:
            cutoff = args.since
        where = "ts_utc >= ?"; params = [cutoff]

    print("\n" + "=" * 90)
    print("📋 SNAPSHOT TIMELINE")
    print("=" * 90)
    print(f"{'#':<4} {'Started':<19} {'Age':<10} {'Trades':<8} {'WR':<6} {'Net':<10} {'$/tr':<8} {'Label'}")
    print("-" * 90)

    rows = c.execute(f"""
        SELECT cs.id, cs.ts_utc, cs.label, cs.config_hash,
               cs.risk_per_trade_pct, cs.smart_trailing_enabled,
               cs.series_loss_cap_action, cs.hourly_lot_mult_enabled,
               COALESCE(cs.notes, '') as notes,
               (SELECT COUNT(*) FROM decisions d WHERE d.config_snapshot_id=cs.id AND d.status IN ('WIN','LOSS')) as n,
               (SELECT SUM(CASE WHEN d.status='WIN' THEN 1 ELSE 0 END) FROM decisions d WHERE d.config_snapshot_id=cs.id) as w,
               (SELECT COALESCE(SUM(d.pnl),0) FROM decisions d WHERE d.config_snapshot_id=cs.id AND d.status IN ('WIN','LOSS')) as net,
               (SELECT MAX(d.ts_utc) FROM decisions d WHERE d.config_snapshot_id=cs.id) as last_trade
        FROM config_snapshots cs
        WHERE {where}
        ORDER BY cs.id
    """, params).fetchall()

    # Trades with NULL snapshot id
    null_row = c.execute("""
        SELECT COUNT(*) n,
            SUM(CASE WHEN status='WIN' THEN 1 ELSE 0 END) w,
            SUM(pnl) net
        FROM decisions WHERE status IN ('WIN','LOSS') AND config_snapshot_id IS NULL
    """).fetchone()
    if null_row['n'] > 0:
        wr = null_row['w'] / null_row['n'] * 100
        pertr = null_row['net'] / null_row['n']
        flag = "✅" if pertr > 0 else "❌"
        print(f"{'-':<4} {'(pre-snapshot)':<19} {'-':<10} {null_row['n']:<8} {wr:<5.0f}% ${null_row['net']:<+8.2f} ${pertr:<+6.2f}  📜 OLD baseline (pre-tracking) {flag}")

    for r in rows:
        n = r['n'] or 0
        wr = (r['w'] / n * 100) if n else 0
        pertr = (r['net'] / n) if n else 0
        flag = "✅" if r['net'] > 0 else ("❌" if r['net'] < 0 else "⚪")
        age = relative_age(r['ts_utc'])
        # mark active snapshot
        marker = "▶" if r['id'] == snaps[0]['id'] else " "
        label_short = (r['label'] or "?")[:50]
        print(f"{marker}#{r['id']:<3} {fmt_time(r['ts_utc']):<19} {age:<10} {n:<8} {wr:<5.0f}% ${r['net']:<+8.2f} ${pertr:<+6.2f}  {label_short} {flag}")
        if r['notes']:
            print(f"     📝 {r['notes']}")

    # ---------- CURRENT ERA STATS ----------
    print("\n" + "=" * 90)
    print("🟢 CURRENT ERA (active snapshot)")
    print("=" * 90)
    if snaps:
        sid = snaps[0]['id']
        r = c.execute("""
            SELECT
                COUNT(*) n,
                SUM(CASE WHEN status='WIN' THEN 1 ELSE 0 END) w,
                SUM(CASE WHEN status='LOSS' THEN 1 ELSE 0 END) l,
                SUM(pnl) net,
                AVG(CASE WHEN status='WIN' AND pnl>0 THEN pnl END) avgw,
                AVG(CASE WHEN status='LOSS' AND pnl<0 THEN pnl END) avgl,
                MIN(pnl) worst, MAX(pnl) best,
                MIN(ts_utc) first_t, MAX(ts_utc) last_t
            FROM decisions WHERE config_snapshot_id=? AND status IN ('WIN','LOSS')
        """, (sid,)).fetchone()
        if r['n']:
            wr = r['w']/r['n']*100
            rr = (r['avgw'] or 0)/abs(r['avgl'] or 1)
            print(f"  Snapshot #{sid}  {snaps[0]['label']}")
            print(f"  Started: {fmt_time(r['first_t'])}  ({relative_age(r['first_t'])})")
            print(f"  Last trade: {fmt_time(r['last_t'])}  ({relative_age(r['last_t'])})")
            print(f"  Trades: {r['n']}  W{r['w']}/L{r['l']}  WR={wr:.1f}%  RR={rr:.2f}")
            print(f"  Net: ${r['net']:+.2f}  avgW=${r['avgw'] or 0:.2f}  avgL=${r['avgl'] or 0:.2f}")
            print(f"  Range: best=${r['best']:+.2f}  worst=${r['worst']:+.2f}")
        else:
            print(f"  ⏳ Snapshot #{sid} — ยังไม่มีไม้ปิด (เพิ่งเปลี่ยน config)")

    # ---------- DATA QUALITY HINTS ----------
    print("\n" + "=" * 90)
    print("💡 DATA USAGE GUIDE")
    print("=" * 90)
    profitable = [r for r in rows if (r['net'] or 0) > 0 and (r['n'] or 0) >= 20]
    losing = [r for r in rows if (r['net'] or 0) < -20 and (r['n'] or 0) >= 20]
    if profitable:
        print(f"  ✅ Profitable snapshots (n>=20):  {', '.join('#'+str(r['id']) for r in profitable)}")
    if losing:
        print(f"  ❌ Losing snapshots (n>=20):       {', '.join('#'+str(r['id']) for r in losing)}")
    small = [r for r in rows if 0 < (r['n'] or 0) < 20]
    if small:
        print(f"  ⏳ Sample too small (<20):          {', '.join('#'+str(r['id']) for r in small)}")

    # ---------- MODEL RETRAIN TIMELINE ----------
    try:
        from core.retrain_log import get_retrains, ensure_schema
        ensure_schema()
        retrains = get_retrains(limit=10)
        if retrains:
            print("\n" + "=" * 90)
            print("🗃️  MODEL RETRAIN HISTORY (latest 10)")
            print("=" * 90)
            print(f"  {'ID':<4} {'When':<19} {'Age':<10} {'Hash':<18} {'OOS':<7} {'Rows':<8} Notes")
            for r in retrains:
                acc_em = "✅" if r['accepted'] else "❌"
                age = relative_age(r['ts_utc'])
                print(f"  #{r['id']:<3} {fmt_time(r['ts_utc']):<19} {age:<10} "
                      f"{(r['model_hash'] or '?'):<18} {r['oos_acc']:<6.4f} {r['rows_trained']:<8} {acc_em} {r['notes'] or ''}")
    except Exception as e:
        pass

    print("\n  📌 หลัง retrain → trades ใหม่ใช้ model ใหม่ → แยก 'data era' ตาม retrain id ได้")
    return 0


def main():
    ap = argparse.ArgumentParser(description="Data Lineage Dashboard")
    sub = ap.add_subparsers(dest="cmd")

    p_status = sub.add_parser("status", help="Show full timeline (default)")
    p_status.add_argument("--from", dest="from_id", type=int, help="Start from snapshot id")
    p_status.add_argument("--since", help="Time window e.g. 7d, 24h")

    p_note = sub.add_parser("note", help="Set notes on a snapshot")
    p_note.add_argument("snapshot_id", type=int)
    p_note.add_argument("text", help="Notes text (quote it)")

    p_retr = sub.add_parser("retrains", help="List model retrain events")
    p_retr.add_argument("--limit", type=int, default=20)

    # default = status
    args = ap.parse_args()
    if args.cmd is None:
        # parse as status
        args = ap.parse_args(["status"] + sys.argv[1:])

    if args.cmd == "status":
        sys.exit(cmd_status(args))
    elif args.cmd == "note":
        from core.retrain_log import set_snapshot_notes
        ok = set_snapshot_notes(args.snapshot_id, args.text)
        if ok:
            print(f"✅ Snapshot #{args.snapshot_id} notes updated.")
        else:
            print(f"❌ Snapshot #{args.snapshot_id} not found.")
        sys.exit(0 if ok else 1)
    elif args.cmd == "retrains":
        from core.retrain_log import get_retrains, ensure_schema
        ensure_schema()
        rows = get_retrains(args.limit)
        print(f"\n🗃️  MODEL RETRAIN LOG (latest {len(rows)})")
        print("-" * 90)
        print(f"  {'ID':<4} {'When':<19} {'Hash':<18} {'Rows':<8} {'CV':<7} {'OOS':<7} {'Acc':<5} Notes")
        for r in rows:
            acc = "✅" if r['accepted'] else "❌"
            print(f"  #{r['id']:<3} {fmt_time(r['ts_utc']):<19} {r['model_hash'] or '?':<18} "
                  f"{r['rows_trained']:<8} {r['cv_acc']:<6.4f} {r['oos_acc']:<6.4f} {acc:<5} {r['notes'] or ''}")
        sys.exit(0)


if __name__ == "__main__":
    main()
