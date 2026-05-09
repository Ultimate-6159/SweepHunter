"""วิเคราะห์ Smart Trailing — ช่วยจริงไหม + ขนาด WIN กระจายอย่างไร"""
import sqlite3
from pathlib import Path
from datetime import datetime, timezone, timedelta

DB = Path(r"C:\SweepHunter - Copy\data\db\hyper_trades.sqlite")
out = open(r"C:\SweepHunter - Copy\_trail_analysis.txt", "w", encoding="utf-8")
def w(*a): out.write(" ".join(str(x) for x in a) + "\n")

c = sqlite3.connect(str(DB)); c.row_factory = sqlite3.Row

# ============================================ Distribution of WIN sizes
w("="*80)
w("📊 DISTRIBUTION ของ WIN ทั้งหมด — ช่วยตอบว่า 'BE Trail ตัด TP บ่อยแค่ไหน?'")
w("="*80)
buckets = [
    (0, 0.50,    "0-50¢      🤏 micro"),
    (0.50, 2.0,  "50¢-$2     🐜 tiny (BE-ish)"),
    (2.0, 5.0,   "$2-$5      🐢 small"),
    (5.0, 10.0,  "$5-$10     ➖ normal"),
    (10.0, 20.0, "$10-$20    ✅ good"),
    (20.0, 50.0, "$20-$50    💎 prime"),
    (50.0, 999,  ">$50       🚀 jackpot"),
]
total_w = c.execute("SELECT COUNT(*),SUM(pnl) FROM decisions WHERE status='WIN'").fetchone()
total_n = total_w[0]
total_pnl = total_w[1] or 0
w(f"\n  Total WINs: {total_n}   Sum: ${total_pnl:+.2f}")
w(f"  {'Bucket':<32} {'N':<6} {'%':<6} {'Sum$':<10} {'%PnL':<6}")
w("  " + "-"*70)
for lo, hi, label in buckets:
    row = c.execute("SELECT COUNT(*), SUM(pnl) FROM decisions WHERE status='WIN' AND pnl >= ? AND pnl < ?", (lo, hi)).fetchone()
    n = row[0] or 0
    s = row[1] or 0
    pct_n = n/total_n*100 if total_n else 0
    pct_p = s/total_pnl*100 if total_pnl else 0
    bar = "█" * int(pct_n/2)
    w(f"  {label:<32} {n:<6} {pct_n:<5.1f}% ${s:<+8.0f} {pct_p:<5.1f}% {bar}")

# ============================================ Hypothesis: WINs < $2 = BE-saved
w("\n" + "="*80)
w("🔍 HYPOTHESIS: WIN < $2 อาจเป็นไม้ที่ BE Trail ปิดเมื่อราคาตี")
w("="*80)
small = c.execute("SELECT COUNT(*),SUM(pnl) FROM decisions WHERE status='WIN' AND pnl < 2.0").fetchone()
big   = c.execute("SELECT COUNT(*),SUM(pnl) FROM decisions WHERE status='WIN' AND pnl >= 2.0").fetchone()
w(f"\n  WIN < $2  : {small[0]} ไม้ รวม ${small[1] or 0:+.2f}  (เฉลี่ย ${(small[1] or 0)/(small[0] or 1):.2f}/ไม้)")
w(f"  WIN ≥ $2  : {big[0]} ไม้ รวม ${big[1] or 0:+.2f}  (เฉลี่ย ${(big[1] or 0)/(big[0] or 1):.2f}/ไม้)")
w(f"\n  ⚠️  ถ้าไม่มี BE Trail → ไม้ {small[0]} ตัวนี้อาจ:")
w(f"     • วิ่งถึง TP เต็ม → +$15-25 ต่อไม้ → +${small[0]*20:.0f}")
w(f"     • หรือกลับมา hit SL → -$15 ต่อไม้ → -${small[0]*15:.0f}")
w(f"     สมมติ 50/50 → คาดว่า upside ≈ +${small[0]*2.5:.0f}")

# ============================================ Sweet spot: WIN sizes vs LOSS sizes
w("\n" + "="*80)
w("⚖️  WIN avg vs LOSS avg — RR จริงเป็นเท่าไหร่")
w("="*80)
won = c.execute("SELECT COUNT(*),AVG(pnl),SUM(pnl) FROM decisions WHERE status='WIN' AND pnl > 0").fetchone()
lost = c.execute("SELECT COUNT(*),AVG(pnl),SUM(pnl) FROM decisions WHERE status='LOSS' AND pnl < 0").fetchone()
w(f"  WIN  : n={won[0]:<4} avg=${won[1]:.2f}  sum=${won[2]:+.0f}")
w(f"  LOSS : n={lost[0]:<4} avg=${lost[1]:.2f}  sum=${lost[2]:+.0f}")
rr = won[1]/abs(lost[1]) if lost[1] else 0
ev = (won[0]/(won[0]+lost[0]))*won[1] + (lost[0]/(won[0]+lost[0]))*lost[1]
w(f"  RR เฉลี่ยจริง  = {rr:.2f} (target 1.33 ตามที่ตั้ง)")
w(f"  EV ต่อไม้      = ${ev:+.2f}")

# ============================================ Commission analysis
w("\n" + "="*80)
w("💸 COMMISSION IMPACT — กิน PnL ไปกี่ %")
w("="*80)
COMM_PER_LOT = 6.0  # round-trip
total_vol = c.execute("SELECT SUM(volume) FROM decisions WHERE status IN ('WIN','LOSS')").fetchone()[0] or 0
total_comm = total_vol * COMM_PER_LOT
gross = c.execute("SELECT SUM(pnl) FROM decisions WHERE status IN ('WIN','LOSS')").fetchone()[0] or 0
w(f"\n  Total volume traded : {total_vol:.2f} lots")
w(f"  Commission ($6/lot) : ${total_comm:.2f}")
w(f"  Gross PnL           : ${gross:+.2f}")
w(f"  ⚠️  Commission/Gross : {total_comm/abs(gross)*100:.1f}%" if gross else "")
w(f"  → Net หลังหัก commission: ${gross - total_comm:+.2f}")
w(f"")
w(f"  ความหมาย: ถ้า commission ≥ 50% ของ PnL = Strategy แพ้ broker")
w(f"           ตอนนี้ {total_comm/abs(gross)*100:.0f}% → " + ("⚠️ สูง" if total_comm > 0.4*abs(gross) else "✅ OK"))

# ============================================ Recent vs Old comparison
w("\n" + "="*80)
w("📈 ก่อนปิด Trail (เก่า) vs หลังปิด (ใหม่) — มีไม้ไหนเปลี่ยนไหม?")
w("="*80)
# trailing was disabled around hour ~11 May 8 UTC (let's say cutoff = May 8 16:00 UTC)
cutoff = "2026-05-08T16:00:00"
old = c.execute("SELECT COUNT(*),AVG(pnl),SUM(pnl) FROM decisions WHERE status='WIN' AND ts_utc < ?", (cutoff,)).fetchone()
new = c.execute("SELECT COUNT(*),AVG(pnl),SUM(pnl) FROM decisions WHERE status='WIN' AND ts_utc >= ?", (cutoff,)).fetchone()
w(f"  ก่อน  ({cutoff[:10]} ก่อน): n={old[0]:<4} avg WIN ${old[1] or 0:.2f}")
w(f"  หลัง  ({cutoff[:10]} หลัง): n={new[0]:<4} avg WIN ${new[1] or 0:.2f}")
if new[0] and new[0] > 5:
    delta = (new[1] or 0) - (old[1] or 0)
    w(f"  Δ avg WIN: ${delta:+.2f}/ไม้  ({'✅ ใหญ่ขึ้น' if delta > 0 else '⚠️ เล็กลง'})")
else:
    w(f"  ⚠️ ข้อมูลใหม่ยังน้อยเกินไป ({new[0]} ไม้) — รอ 50+ ไม้ก่อนค่อยสรุป")

out.close()
print("done")
