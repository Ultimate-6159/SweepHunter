"""วิเคราะห์: change ไหนพังจริง + ทางที่ดีที่สุด"""
import sqlite3
from pathlib import Path

DB = Path(r"C:\SweepHunter - Copy\data\db\hyper_trades.sqlite")
out = open(r"C:\SweepHunter - Copy\_diag.txt", "w", encoding="utf-8")
def w(*a): out.write(" ".join(str(x) for x in a) + "\n")

c = sqlite3.connect(str(DB)); c.row_factory = sqlite3.Row

# ============= Daily PnL last 14 days =============
w("="*80); w("📊 DAILY PnL — 14 วันล่าสุด"); w("="*80)
rows = c.execute("""
    SELECT date(ts_utc) d, COUNT(*) n,
        SUM(CASE WHEN status='WIN' THEN 1 ELSE 0 END) w_n,
        SUM(pnl) net,
        AVG(CASE WHEN status='WIN' AND pnl>0 THEN pnl END) avg_w,
        AVG(CASE WHEN status='LOSS' AND pnl<0 THEN pnl END) avg_l,
        MIN(pnl) worst, MAX(pnl) best
    FROM decisions WHERE status IN ('WIN','LOSS')
    GROUP BY d ORDER BY d DESC LIMIT 14
""").fetchall()
w(f"\n  {'วันที่':<12} {'N':<4} {'WR':<5} {'Net':<9} {'avgW':<6} {'avgL':<7} {'worst':<8} {'best':<7}")
w("  " + "-"*72)
for r in rows:
    wr = r['w_n']/r['n']*100 if r['n'] else 0
    flag = "✅" if r['net'] > 0 else "❌"
    w(f"  {r['d']:<12} {r['n']:<4} {wr:<5.1f} ${r['net']:<+7.2f} ${r['avg_w'] or 0:<+5.1f} ${r['avg_l'] or 0:<+6.1f} ${r['worst']:<+7.1f} ${r['best']:<+6.1f} {flag}")

# ============= Worst series ทุกวัน =============
w("\n" + "="*80); w("💀 SERIES ที่เสียหนักที่สุด — 14 วันล่าสุด"); w("="*80)
rows = c.execute("""
    SELECT id, date(opened_at_utc) d, opened_at_utc, side, steps, final_pnl, status
    FROM series WHERE date(opened_at_utc) >= date('now', '-14 days')
        AND final_pnl < -30
    ORDER BY final_pnl LIMIT 15
""").fetchall()
for r in rows:
    h = r['opened_at_utc'][11:16]
    w(f"  {r['d']} {h}  #{r['id']:<4} {r['side']} steps={r['steps']:<2} ${r['final_pnl']:+8.2f}  {r['status']}")

# ============= ก่อน vs หลัง config changes =============
w("\n" + "="*80); w("🔄 เปรียบเทียบ: ก่อน vs หลัง changes ใหญ่"); w("="*80)
# Major change cutoffs (UTC):
# - Old config (BE trail, 1% risk): before 2026-05-08T16:00
# - Mid (smart_trail off, hourly mult, 1.5%): 2026-05-08T16:00 → 2026-05-09T04:30
# - New (BE 0.8 trail back + hourly mult): 2026-05-09T04:30 → 2026-05-11T01:30
# - Latest (BE 0.5 + disable_during_recovery + cap 4%): from 2026-05-11T01:30
buckets = [
    ("📍 OLD (1% risk + BE trail) — บ่อยๆ baseline",
     "2026-04-23", "2026-05-08T16:00:00"),
    ("🟡 MID (1.5% risk + no trail + hmult)",
     "2026-05-08T16:00:00", "2026-05-09T04:30:00"),
    ("🟠 NEW (BE 0.8 + hmult enabled in recovery)",
     "2026-05-09T04:30:00", "2026-05-11T02:00:00"),
    ("🔴 LATEST (BE 0.5 + hmult-recovery off + cap 4%)",
     "2026-05-11T02:00:00", "2030-01-01"),
]
for label, lo, hi in buckets:
    r = c.execute("""
        SELECT COUNT(*) n,
            SUM(CASE WHEN status='WIN' THEN 1 ELSE 0 END) w_n,
            SUM(pnl) net,
            AVG(CASE WHEN status='WIN' AND pnl>0 THEN pnl END) avg_w,
            AVG(CASE WHEN status='LOSS' AND pnl<0 THEN pnl END) avg_l,
            MIN(pnl) worst,
            SUM(volume) vol
        FROM decisions WHERE status IN ('WIN','LOSS')
            AND ts_utc >= ? AND ts_utc < ?
    """, (lo, hi)).fetchone()
    if not r["n"]: continue
    wr = r['w_n']/r['n']*100
    rr = (r['avg_w'] or 0)/abs(r['avg_l'] or 1)
    per_tr = r['net']/r['n']
    comm = (r['vol'] or 0)*6.0
    flag = "✅" if per_tr > 0 else "❌"
    w(f"\n  {label}")
    w(f"    Trades: {r['n']}  WR: {wr:.1f}%  Net: ${r['net']:+.2f}  $/trade: ${per_tr:+.2f}  RR: {rr:.2f}  comm: ${comm:.2f}  {flag}")
    w(f"    avgW: ${r['avg_w'] or 0:.2f}  avgL: ${r['avg_l'] or 0:.2f}  worst: ${r['worst']:+.2f}")

# ============= Step distribution: primary vs recovery =============
w("\n" + "="*80); w("🎯 PRIMARY vs RECOVERY trades — 14 วันล่าสุด"); w("="*80)
rows = c.execute("""
    SELECT role, COUNT(*) n,
        SUM(CASE WHEN status='WIN' THEN 1 ELSE 0 END) w_n,
        SUM(pnl) net,
        AVG(CASE WHEN status='WIN' AND pnl>0 THEN pnl END) avg_w,
        AVG(CASE WHEN status='LOSS' AND pnl<0 THEN pnl END) avg_l
    FROM decisions WHERE status IN ('WIN','LOSS')
        AND date(ts_utc) >= date('now','-14 days')
    GROUP BY role
""").fetchall()
for r in rows:
    wr = r['w_n']/r['n']*100 if r['n'] else 0
    w(f"  {r['role']:<10} n={r['n']:<4} WR={wr:.1f}%  Net=${r['net']:+8.2f}  avgW=${r['avg_w'] or 0:.2f}  avgL=${r['avg_l'] or 0:.2f}")

out.close(); print("done")
