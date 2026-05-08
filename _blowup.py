"""วิเคราะห์ว่าทำไมเสียหนัก — focus ช่วง 24 ชม. ล่าสุด"""
import sqlite3
from pathlib import Path
from datetime import datetime, timezone, timedelta

DB = Path(r"C:\SweepHunter - Copy\data\db\hyper_trades.sqlite")
out = open(r"C:\SweepHunter - Copy\_blowup_analysis.txt", "w", encoding="utf-8")
def w(*a): out.write(" ".join(str(x) for x in a) + "\n")

c = sqlite3.connect(str(DB)); c.row_factory = sqlite3.Row

# === 1. Recent series (เรียงตามเวลา) ===
w("="*80); w("📅 SERIES ใน 24 ชม. ล่าสุด (เรียงตามเวลา)"); w("="*80)
since = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
rows = c.execute("""
    SELECT s.id, s.opened_at_utc, s.closed_at_utc, s.side, s.steps,
           s.final_pnl, s.status, s.notes
    FROM series s
    WHERE s.opened_at_utc > ?
    ORDER BY s.opened_at_utc
""", (since,)).fetchall()
total_pnl = 0.0
for r in rows:
    pnl = r["final_pnl"] or 0
    total_pnl += pnl
    emoji = "💚" if pnl > 0 else ("💔" if pnl < 0 else "⚪")
    t = r["opened_at_utc"][11:19]
    status = (r["status"] or "OPEN")[:18]
    w(f"  {t} #{r['id']:<4} {r['side']:<4} steps={r['steps']:<2} {status:<20} {emoji} ${pnl:+.2f}")
w(f"\n  ➡️  รวม 24 ชม.: ${total_pnl:+.2f}  ({len(rows)} series)")

# === 2. Worst 10 series ทั้งหมด (เพื่อหา pattern blow-up) ===
w("\n" + "="*80); w("💀 WORST 10 SERIES — series ที่กินกำไรเยอะสุด"); w("="*80)
rows = c.execute("""
    SELECT id, opened_at_utc, closed_at_utc, side, steps, final_pnl, status
    FROM series WHERE final_pnl IS NOT NULL
    ORDER BY final_pnl ASC LIMIT 10
""").fetchall()
for r in rows:
    t = r["opened_at_utc"][:16].replace("T", " ")
    dur_min = ""
    if r["closed_at_utc"]:
        a = datetime.fromisoformat(r["opened_at_utc"].replace("Z","+00:00"))
        b = datetime.fromisoformat(r["closed_at_utc"].replace("Z","+00:00"))
        dur_min = f"{(b-a).total_seconds()/60:.0f}min"
    w(f"  {t}  #{r['id']:<4} {r['side']:<4} steps={r['steps']:<2} {dur_min:<8} "
      f"{r['status']:<25} ${r['final_pnl']:+.2f}")

# === 3. ละเอียด worst series — ดูทุกไม้ที่อยู่ในนั้น ===
worst = rows[0]
w(f"\n" + "="*80); w(f"🔬 ZOOM IN: worst series #{worst['id']} ({worst['side']}, {worst['final_pnl']:+.2f})")
w("="*80)
ds = c.execute("""
    SELECT step, role, ts_utc, prediction, confidence, volume, entry_price,
           pnl, status, atr, spread_points, notes
    FROM decisions WHERE series_id=? ORDER BY step
""", (worst["id"],)).fetchall()
for d in ds:
    side_name = {0:"SELL",1:"HOLD",2:"BUY"}[d["prediction"]]
    t = d["ts_utc"][11:19]
    w(f"  step{d['step']:<2} {d['role']:<8} {t}  {side_name} conf={d['confidence']:.2%} "
      f"lot={d['volume']:.3f}  atr={d['atr'] or 0:.2f}  spread={d['spread_points'] or 0:.0f}p  "
      f"→ {d['status']:<6} ${d['pnl'] or 0:+.2f}")

# === 4. Blow-up pattern: series ที่เสียหนัก เกิดที่ชั่วโมงไหน? ===
w("\n" + "="*80); w("🕐 BLOW-UP HOURS: ชั่วโมงที่เกิด series เสียหนักบ่อย")
w("="*80)
rows = c.execute("""
    SELECT CAST(strftime('%H', opened_at_utc) AS INT) h,
        COUNT(*) n_series,
        SUM(CASE WHEN final_pnl < -10 THEN 1 ELSE 0 END) n_blowup,
        SUM(final_pnl) total_pnl,
        AVG(final_pnl) avg_pnl
    FROM series WHERE final_pnl IS NOT NULL
    GROUP BY h
    ORDER BY total_pnl
""").fetchall()
w(f"  {'Hour UTC':<10} {'BKK':<6} {'Series':<8} {'Blowups':<10} {'Total':<12} {'Avg/series':<12}")
w("  " + "-"*68)
for r in rows[:8]:
    bkk = (r['h'] + 7) % 24
    w(f"  {r['h']:02d}:00 UTC  {bkk:02d}:00  {r['n_series']:<8} {r['n_blowup']:<10} "
      f"${r['total_pnl']:+8.2f}  ${r['avg_pnl']:+.2f}")

# === 5. ATR ratio ตอน blow-up ===
w("\n" + "="*80); w("⚡ ATR / Spread ตอน blow-up — มี outlier ไหม?")
w("="*80)
rows = c.execute("""
    SELECT d.atr, d.spread_points, d.confidence, s.final_pnl, s.id
    FROM decisions d JOIN series s ON s.id = d.series_id
    WHERE s.final_pnl < -10 AND d.role = 'PRIMARY' AND d.atr IS NOT NULL
    ORDER BY s.final_pnl
""").fetchall()
if rows:
    w(f"  ATR ของ PRIMARY ใน blow-up series:")
    atrs = [r["atr"] for r in rows[:10]]
    spreads = [r["spread_points"] or 0 for r in rows[:10]]
    confs = [r["confidence"] or 0 for r in rows[:10]]
    w(f"    avg ATR  = {sum(atrs)/len(atrs):.3f} (min={min(atrs):.3f} max={max(atrs):.3f})")
    w(f"    avg spread = {sum(spreads)/len(spreads):.1f}p")
    w(f"    avg confidence = {sum(confs)/len(confs)*100:.1f}%")

# === 6. ดูกำไรที่หายไปวันนี้ ===
w("\n" + "="*80); w("💸 วันนี้ทำได้เท่าไหร่ vs เสียเท่าไหร่")
w("="*80)
today = datetime.now(timezone.utc).date().isoformat()
rows = c.execute("""
    SELECT
      SUM(CASE WHEN pnl > 0 THEN pnl ELSE 0 END) gross_win,
      SUM(CASE WHEN pnl < 0 THEN pnl ELSE 0 END) gross_loss,
      SUM(pnl) net,
      COUNT(*) n,
      SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) n_win,
      SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) n_loss
    FROM decisions
    WHERE date(ts_utc) = ? AND status IN ('WIN','LOSS')
""", (today,)).fetchone()
if rows and rows["n"]:
    wr = rows["n_win"]/rows["n"]*100
    w(f"  วันนี้ ({today}):")
    w(f"    Trades: {rows['n']} (WIN {rows['n_win']} / LOSS {rows['n_loss']}, WR={wr:.1f}%)")
    w(f"    Gross WIN  : ${rows['gross_win']:+.2f}")
    w(f"    Gross LOSS : ${rows['gross_loss']:+.2f}")
    w(f"    NET        : ${rows['net']:+.2f}")
    if rows['gross_loss']:
        pf = rows['gross_win'] / abs(rows['gross_loss'])
        w(f"    Profit Factor: {pf:.2f}")

out.close()
print("✅ saved → _blowup_analysis.txt")
