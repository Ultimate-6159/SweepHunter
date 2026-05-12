"""วิเคราะห์ PnL ตาม confidence buckets — หา sweet spot"""
import sqlite3
DB = r'C:\SweepHunter - Copy\data\db\hyper_trades.sqlite'
out = open(r'C:\SweepHunter - Copy\_conf.txt', 'w', encoding='utf-8')
def w(*a): out.write(" ".join(str(x) for x in a)+"\n")
c = sqlite3.connect(DB); c.row_factory = sqlite3.Row

# === Bucket: ทุก trade ที่ settled ===
buckets = [
    ("0.50-0.55", 0.50, 0.55),
    ("0.55-0.60", 0.55, 0.60),
    ("0.60-0.65", 0.60, 0.65),
    ("0.65-0.70", 0.65, 0.70),
    ("0.70-0.75", 0.70, 0.75),
    ("0.75-0.80", 0.75, 0.80),
    ("0.80-0.90", 0.80, 0.90),
    ("0.90+",     0.90, 1.01),
]

w("="*100); w("📊 PnL by CONFIDENCE BUCKET (ทุก trade ที่ settled)"); w("="*100)
w(f"  {'Conf':<10} {'N':<5} {'WR':<6} {'AvgW':<7} {'AvgL':<8} {'RR':<5} {'$/tr':<8} {'Net':<10}  Verdict")
w("  " + "-"*95)
for label, lo, hi in buckets:
    r = c.execute("""
        SELECT COUNT(*) n,
            SUM(CASE WHEN status='WIN' THEN 1 ELSE 0 END) w,
            SUM(CASE WHEN status='LOSS' THEN 1 ELSE 0 END) l,
            SUM(pnl) net,
            AVG(CASE WHEN status='WIN' AND pnl>0 THEN pnl END) avgw,
            AVG(CASE WHEN status='LOSS' AND pnl<0 THEN pnl END) avgl
        FROM decisions
        WHERE status IN ('WIN','LOSS') AND confidence >= ? AND confidence < ?
    """, (lo, hi)).fetchone()
    if r['n'] == 0:
        continue
    wr = r['w']/r['n']*100
    rr = (r['avgw'] or 0) / abs(r['avgl'] or 1)
    pertr = r['net']/r['n']
    if pertr > 0.5: v = "✅ KEEP"
    elif pertr > -0.5: v = "⚠️ NEUTRAL"
    else: v = "❌ AVOID"
    w(f"  {label:<10} {r['n']:<5} {wr:<5.1f}% ${r['avgw'] or 0:<5.1f} ${r['avgl'] or 0:<6.1f} {rr:<4.2f} ${pertr:<+6.2f} ${r['net']:<+8.2f}  {v}")

# === แยก PRIMARY vs RECOVERY ===
w("\n" + "="*100); w("📊 PnL by CONF + ROLE (แยก PRIMARY/RECOVERY)"); w("="*100)
for role in ["PRIMARY", "RECOVERY"]:
    w(f"\n  ── ROLE: {role} ──")
    w(f"  {'Conf':<10} {'N':<5} {'WR':<6} {'AvgW':<7} {'AvgL':<8} {'$/tr':<8} {'Net':<10}")
    for label, lo, hi in buckets:
        r = c.execute("""
            SELECT COUNT(*) n, SUM(CASE WHEN status='WIN' THEN 1 ELSE 0 END) w,
                SUM(pnl) net,
                AVG(CASE WHEN status='WIN' AND pnl>0 THEN pnl END) avgw,
                AVG(CASE WHEN status='LOSS' AND pnl<0 THEN pnl END) avgl
            FROM decisions
            WHERE status IN ('WIN','LOSS') AND role=? AND confidence >= ? AND confidence < ?
        """, (role, lo, hi)).fetchone()
        if r['n'] < 5: continue
        wr = r['w']/r['n']*100
        pertr = r['net']/r['n']
        flag = "✅" if pertr > 0 else "❌"
        w(f"  {label:<10} {r['n']:<5} {wr:<5.1f}% ${r['avgw'] or 0:<5.1f} ${r['avgl'] or 0:<6.1f} ${pertr:<+6.2f} ${r['net']:<+8.2f}  {flag}")

# === แยก BUY vs SELL ===
w("\n" + "="*100); w("📊 PnL by CONF + SIDE"); w("="*100)
for pred, side_label in [(2, "BUY"), (0, "SELL")]:
    w(f"\n  ── SIDE: {side_label} ──")
    for label, lo, hi in buckets:
        r = c.execute("""
            SELECT COUNT(*) n, SUM(CASE WHEN status='WIN' THEN 1 ELSE 0 END) w,
                SUM(pnl) net
            FROM decisions
            WHERE status IN ('WIN','LOSS') AND prediction=? AND confidence >= ? AND confidence < ?
        """, (pred, lo, hi)).fetchone()
        if r['n'] < 5: continue
        wr = r['w']/r['n']*100
        pertr = r['net']/r['n']
        flag = "✅" if pertr > 0 else "❌"
        w(f"  {label:<10} {r['n']:<5} {wr:<5.1f}%  ${pertr:<+6.2f}  ${r['net']:<+8.2f}  {flag}")

# === Cumulative threshold analysis (>= X) ===
w("\n" + "="*100); w("📊 CUMULATIVE: ถ้าตั้ง min_confidence = X จะได้อะไร"); w("="*100)
w(f"  {'Threshold':<12} {'N kept':<8} {'WR':<6} {'$/tr':<8} {'Net':<10}  Verdict")
w("  " + "-"*70)
for thr in [0.50, 0.55, 0.58, 0.60, 0.62, 0.65, 0.68, 0.70, 0.72, 0.75, 0.80]:
    r = c.execute("""
        SELECT COUNT(*) n, SUM(CASE WHEN status='WIN' THEN 1 ELSE 0 END) w,
            SUM(pnl) net
        FROM decisions
        WHERE status IN ('WIN','LOSS') AND confidence >= ?
    """, (thr,)).fetchone()
    if r['n'] == 0: continue
    wr = r['w']/r['n']*100
    pertr = r['net']/r['n']
    flag = "✅ BEST" if pertr > 1 else ("✅" if pertr > 0 else "❌")
    w(f"  >={thr:<10.2f} {r['n']:<8} {wr:<5.1f}% ${pertr:<+6.2f} ${r['net']:<+8.2f}  {flag}")

out.close(); print("done")
