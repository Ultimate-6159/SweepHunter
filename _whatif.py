"""What-if analysis: ถ้า block hot zones จะดีขึ้นแค่ไหน"""
import sqlite3
from pathlib import Path

DB = Path(r"C:\SweepHunter - Copy\data\db\hyper_trades.sqlite")
out = open(r"C:\SweepHunter - Copy\_whatif.txt", "w", encoding="utf-8")
def w(*a): out.write(" ".join(str(x) for x in a) + "\n")

c = sqlite3.connect(str(DB)); c.row_factory = sqlite3.Row

# PnL ทุกชั่วโมง
w("="*80)
w("📊 PnL by Hour — ทั้ง 24 ชม. (ตลอด 2-3 สัปดาห์)")
w("="*80)
rows = c.execute("""
    SELECT CAST(strftime('%H', ts_utc) AS INT) h,
           COUNT(*) n,
           SUM(CASE WHEN status='WIN' THEN 1 ELSE 0 END) w_n,
           SUM(pnl) net,
           SUM(CASE WHEN pnl > 0 THEN pnl ELSE 0 END) gross_w,
           SUM(CASE WHEN pnl < 0 THEN pnl ELSE 0 END) gross_l
    FROM decisions
    WHERE status IN ('WIN','LOSS')
    GROUP BY h ORDER BY h
""").fetchall()

w(f"  {'UTC':<6} {'BKK':<6} {'N':<5} {'WR%':<6} {'WIN':<10} {'LOSS':<10} {'NET':<10} {'PF':<6}")
w("  " + "-"*70)
for r in rows:
    bkk = (r['h']+7) % 24
    wr = r['w_n']/r['n']*100
    pf = r['gross_w']/abs(r['gross_l']) if r['gross_l'] else 99
    flag = ""
    if r['net'] < -30: flag = "🔴 BLOCK"
    elif r['net'] < -10: flag = "⚠️ WARN"
    elif r['net'] > 50: flag = "💎 PRIME"
    w(f"  {r['h']:02d}    {bkk:02d}    {r['n']:<5} {wr:<6.1f} ${r['gross_w']:<+8.0f} ${r['gross_l']:<+8.0f} ${r['net']:<+8.2f} {pf:<5.2f} {flag}")

# === Simulation: ถ้า block hours [5, 11, 14] ===
w("\n" + "="*80)
w("🧪 SIMULATION: ถ้า block hour [5, 11, 14] UTC ตั้งแต่ต้น")
w("="*80)

block_hours = [5, 11, 14]
all_pnl = c.execute("SELECT SUM(pnl) FROM decisions WHERE status IN ('WIN','LOSS')").fetchone()[0]
blocked_pnl = c.execute(f"""
    SELECT SUM(pnl) FROM decisions WHERE status IN ('WIN','LOSS')
    AND CAST(strftime('%H', ts_utc) AS INT) IN ({','.join(map(str,block_hours))})
""").fetchone()[0]
blocked_n = c.execute(f"""
    SELECT COUNT(*) FROM decisions WHERE status IN ('WIN','LOSS')
    AND CAST(strftime('%H', ts_utc) AS INT) IN ({','.join(map(str,block_hours))})
""").fetchone()[0]

w(f"  Net PnL ปัจจุบัน:           ${all_pnl:+.2f}")
w(f"  Net PnL จากชั่วโมงต้องห้าม:  ${blocked_pnl:+.2f}  ({blocked_n} trades)")
w(f"  ✨ Net PnL ถ้า block:        ${all_pnl - blocked_pnl:+.2f}")
w(f"  💰 ส่วนต่าง:                 ${-blocked_pnl:+.2f}  ← นี่คือเงินที่จะ \"ไม่หาย\"")
w(f"  📉 trades ที่หายไป:          {blocked_n} ไม้")

# Series blow-ups in those hours
big = c.execute(f"""
    SELECT id, opened_at_utc, side, final_pnl, status FROM series
    WHERE CAST(strftime('%H', opened_at_utc) AS INT) IN ({','.join(map(str,block_hours))})
    AND final_pnl < -20
    ORDER BY final_pnl
""").fetchall()
if big:
    w(f"\n  💀 Series ใหญ่ (>$20 loss) ในชั่วโมงเหล่านี้: {len(big)} ครั้ง")
    for r in big[:10]:
        h = int(r['opened_at_utc'][11:13])
        w(f"    {r['opened_at_utc'][:16]}  hour={h:02d}  #{r['id']:<4} {r['side']:<4} ${r['final_pnl']:+.2f}")

out.close()
print("done")
