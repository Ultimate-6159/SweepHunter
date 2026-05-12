"""ดู state ตอนนี้ทันที"""
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

DB = Path(r"C:\SweepHunter - Copy\data\db\hyper_trades.sqlite")
out = open(r"C:\SweepHunter - Copy\_now.txt", "w", encoding="utf-8")
def w(*a): out.write(" ".join(str(x) for x in a) + "\n")
c = sqlite3.connect(str(DB)); c.row_factory = sqlite3.Row

now = datetime.now(timezone.utc)
w(f"⏰ NOW UTC: {now.isoformat()}")
w(f"⏰ NOW BKK: {(now.timestamp() + 7*3600)}\n")

# ============== Snapshot history ==============
w("="*80); w("📋 CONFIG SNAPSHOTS (เรียง id)"); w("="*80)
rows = c.execute("SELECT id, ts_utc, label FROM config_snapshots ORDER BY id").fetchall()
for r in rows:
    w(f"  #{r['id']}  {r['ts_utc'][:19]}  {r['label'][:80]}")

# ============== Trades แต่ละ snapshot ==============
w("\n" + "="*80); w("📊 TRADES per snapshot"); w("="*80)
rows = c.execute("""
    SELECT
        config_snapshot_id sid,
        COUNT(*) n,
        SUM(CASE WHEN status='WIN' THEN 1 ELSE 0 END) w,
        SUM(CASE WHEN status='LOSS' THEN 1 ELSE 0 END) l,
        SUM(pnl) net,
        AVG(pnl) avg_pnl,
        MIN(ts_utc) first_ts,
        MAX(ts_utc) last_ts
    FROM decisions
    WHERE status IN ('WIN','LOSS')
    GROUP BY config_snapshot_id
    ORDER BY config_snapshot_id NULLS FIRST
""").fetchall()
for r in rows:
    sid_s = str(r['sid']) if r['sid'] else "NULL"
    wr = r['w']/r['n']*100 if r['n'] else 0
    flag = "✅" if r['net'] > 0 else "❌"
    last_short = r['last_ts'][:19] if r['last_ts'] else "?"
    w(f"  sid={sid_s:<5} n={r['n']:<4} W{r['w']:<3}/L{r['l']:<3} WR={wr:.0f}%  Net=${r['net']:+8.2f}  $/tr=${r['avg_pnl'] or 0:+5.2f}  จนถึง {last_short}  {flag}")

# ============== Trades ล่าสุด 20 ไม้ ==============
w("\n" + "="*80); w("🔍 LATEST 20 TRADES (newest first)"); w("="*80)
rows = c.execute("""
    SELECT id, ts_utc, side, role, status, pnl, volume, confidence, atr, config_snapshot_id sid
    FROM decisions
    WHERE status IN ('WIN','LOSS')
    ORDER BY id DESC LIMIT 20
""").fetchall()
for r in rows:
    side = "BUY" if (r["side"] is None and (r["pnl"] or 0)) else r["side"]
    s = (r["status"] or "?")[:5]
    pnl = r["pnl"] or 0
    em = "💚" if pnl > 0 else "💔"
    sid_s = str(r['sid']) if r['sid'] else "?"
    w(f"  {r['ts_utc'][:19]}  #{r['id']:<5} {(r['role'] or '?')[:7]:<7} sid={sid_s:<3} {em} ${pnl:+7.2f} vol={r['volume']:.2f} conf={r['confidence'] or 0:.2f}")

# ============== Open positions / series ==============
w("\n" + "="*80); w("🚨 OPEN STATE"); w("="*80)
rows = c.execute("SELECT * FROM decisions WHERE status IN ('OPEN','PENDING') ORDER BY id DESC LIMIT 5").fetchall()
if rows:
    for r in rows:
        w(f"  OPEN #{r['id']} ticket={r['ticket']} status={r['status']} entry={r['entry_price']} sl={r['sl']} tp={r['tp']}")
else:
    w("  ไม่มี OPEN/PENDING decisions ใน DB")

rows = c.execute("SELECT * FROM series WHERE status='OPEN' ORDER BY id DESC LIMIT 3").fetchall()
if rows:
    for r in rows:
        w(f"  Series OPEN #{r['id']} side={r['side']} steps={r['steps']} opened={r['opened_at_utc']}")
else:
    w("  ไม่มี series OPEN")

out.close(); print("done")
