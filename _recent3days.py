"""วิเคราะห์เฉพาะ พุธ-ศุกร์ ล่าสุด (6-8 พ.ค. = หลัง AI retrain)"""
import sqlite3
from pathlib import Path

DB = Path(r"C:\SweepHunter - Copy\data\db\hyper_trades.sqlite")
out = open(r"C:\SweepHunter - Copy\_recent3days.txt", "w", encoding="utf-8")
def w(*a): out.write(" ".join(str(x) for x in a) + "\n")

c = sqlite3.connect(str(DB)); c.row_factory = sqlite3.Row

DATE_FROM = "2026-05-06"   # วันพุธ
DATE_TO   = "2026-05-08"   # วันศุกร์

# === Sanity check: data range ===
w("="*80)
w(f"📅 ANALYSIS WINDOW: {DATE_FROM} (พุธ) → {DATE_TO} (ศุกร์)")
w("   เฉพาะข้อมูลหลัง AI retrain ล่าสุด — ตัวเลขที่เชื่อถือได้")
w("="*80)

total = c.execute("""
    SELECT COUNT(*), SUM(pnl), SUM(CASE WHEN status='WIN' THEN 1 ELSE 0 END)
    FROM decisions WHERE status IN ('WIN','LOSS') 
    AND date(ts_utc) BETWEEN ? AND ?
""", (DATE_FROM, DATE_TO)).fetchone()
n, net, w_n = total
wr = w_n/n*100 if n else 0
w(f"\n  📊 Total: {n} trades   WR: {wr:.1f}%   Net: ${net or 0:+.2f}\n")

# === Daily breakdown ===
w("="*80)
w("📆 รายวัน")
w("="*80)
rows = c.execute("""
    SELECT date(ts_utc) d, COUNT(*) n,
        SUM(CASE WHEN status='WIN' THEN 1 ELSE 0 END) w_n,
        SUM(pnl) net,
        SUM(CASE WHEN pnl > 0 THEN pnl ELSE 0 END) gw,
        SUM(CASE WHEN pnl < 0 THEN pnl ELSE 0 END) gl
    FROM decisions WHERE status IN ('WIN','LOSS')
    AND date(ts_utc) BETWEEN ? AND ?
    GROUP BY d ORDER BY d
""", (DATE_FROM, DATE_TO)).fetchall()
for r in rows:
    wr = r['w_n']/r['n']*100
    pf = r['gw']/abs(r['gl']) if r['gl'] else 99
    flag = "✅" if r['net'] > 0 else "❌"
    w(f"  {r['d']}  n={r['n']:<3} WR={wr:5.1f}%  WIN${r['gw']:+7.0f}  LOSS${r['gl']:+7.0f}  NET${r['net']:+8.2f} PF={pf:.2f} {flag}")

# === PnL by Hour (recent only) ===
w("\n" + "="*80)
w("🕐 PnL by Hour — เฉพาะ พุธ-ศุกร์ ล่าสุด")
w("="*80)
rows = c.execute("""
    SELECT CAST(strftime('%H', ts_utc) AS INT) h,
        COUNT(*) n,
        SUM(CASE WHEN status='WIN' THEN 1 ELSE 0 END) w_n,
        SUM(pnl) net,
        SUM(CASE WHEN pnl > 0 THEN pnl ELSE 0 END) gw,
        SUM(CASE WHEN pnl < 0 THEN pnl ELSE 0 END) gl
    FROM decisions
    WHERE status IN ('WIN','LOSS')
    AND date(ts_utc) BETWEEN ? AND ?
    GROUP BY h ORDER BY h
""", (DATE_FROM, DATE_TO)).fetchall()

w(f"  {'UTC':<5} {'BKK':<5} {'N':<4} {'WR%':<6} {'WIN':<9} {'LOSS':<9} {'NET':<10} {'PF':<6}")
w("  " + "-"*68)
for r in rows:
    bkk = (r['h']+7) % 24
    wr = r['w_n']/r['n']*100
    pf = r['gw']/abs(r['gl']) if r['gl'] else 99
    flag = ""
    if r['net'] < -20: flag = "🔴 BLOCK"
    elif r['net'] < -5:  flag = "⚠️  WARN"
    elif r['net'] > 30: flag = "💎 PRIME"
    elif r['net'] > 10: flag = "✅ GOOD"
    w(f"  {r['h']:02d}    {bkk:02d}    {r['n']:<4} {wr:<6.1f} ${r['gw']:<+7.0f} ${r['gl']:<+7.0f} ${r['net']:<+8.2f} {pf:<5.2f} {flag}")

# === Worst series in window ===
w("\n" + "="*80)
w("💀 Worst series ใน 3 วันนี้")
w("="*80)
rows = c.execute("""
    SELECT id, opened_at_utc, side, steps, final_pnl, status FROM series
    WHERE date(opened_at_utc) BETWEEN ? AND ? AND final_pnl < -10
    ORDER BY final_pnl LIMIT 10
""", (DATE_FROM, DATE_TO)).fetchall()
for r in rows:
    h = int(r['opened_at_utc'][11:13])
    bkk = (h+7) % 24
    w(f"  {r['opened_at_utc'][:16]}  hour={h:02d}/{bkk:02d}BKK  #{r['id']:<4} {r['side']:<4} steps={r['steps']:<2} ${r['final_pnl']:+.2f} {r['status']}")

# === What-if Block hot zones (recent only) ===
w("\n" + "="*80)
w("🧪 SIMULATION (ใช้ข้อมูล 3 วันนี้): Block hour ที่ Net < -$15")
w("="*80)
losers = [r['h'] for r in rows[:0]]  # reset
losers = []
for r in c.execute("""
    SELECT CAST(strftime('%H', ts_utc) AS INT) h, SUM(pnl) net, COUNT(*) n
    FROM decisions WHERE status IN ('WIN','LOSS')
    AND date(ts_utc) BETWEEN ? AND ?
    GROUP BY h HAVING net < -15
""", (DATE_FROM, DATE_TO)).fetchall():
    losers.append(r['h'])

if losers:
    losers_csv = ",".join(map(str, losers))
    saved = c.execute(f"""
        SELECT SUM(pnl), COUNT(*) FROM decisions WHERE status IN ('WIN','LOSS')
        AND date(ts_utc) BETWEEN ? AND ?
        AND CAST(strftime('%H', ts_utc) AS INT) IN ({losers_csv})
    """, (DATE_FROM, DATE_TO)).fetchone()
    w(f"\n  🔴 Block hours UTC: {losers}  (BKK: {[(h+7)%24 for h in losers]})")
    w(f"  Net จากชั่วโมงเหล่านี้: ${saved[0]:+.2f}  ({saved[1]} trades)")
    w(f"  Net ปัจจุบัน 3 วัน:    ${net:+.2f}")
    w(f"  ✨ Net ถ้า block ตั้งแต่แรก: ${net - (saved[0] or 0):+.2f}")
    w(f"  💰 เงินที่ \"ไม่หาย\":         ${-(saved[0] or 0):+.2f}")
else:
    w("  ✅ ไม่มีชั่วโมงที่ Net < -$15 ใน 3 วันนี้ — strategy ทำงานดี!")

out.close()
print("done")
