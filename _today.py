"""สรุปผลเทรดวันนี้"""
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

DB = Path(r"C:\SweepHunter - Copy\data\db\hyper_trades.sqlite")
out = open(r"C:\SweepHunter - Copy\_today.txt", "w", encoding="utf-8")
def w(*a): out.write(" ".join(str(x) for x in a) + "\n")

c = sqlite3.connect(str(DB)); c.row_factory = sqlite3.Row
today = datetime.now(timezone.utc).date().isoformat()

# === สรุปวันนี้ ===
w("="*80); w(f"📅 สรุปการเทรดวันนี้ ({today} UTC)"); w("="*80)
r = c.execute("""
    SELECT COUNT(*) n,
        SUM(CASE WHEN status='WIN' THEN 1 ELSE 0 END) w_n,
        SUM(CASE WHEN status='LOSS' THEN 1 ELSE 0 END) l_n,
        SUM(CASE WHEN pnl > 0 THEN pnl ELSE 0 END) gw,
        SUM(CASE WHEN pnl < 0 THEN pnl ELSE 0 END) gl,
        SUM(pnl) net,
        SUM(volume) vol,
        AVG(CASE WHEN status='WIN' AND pnl>0 THEN pnl END) avg_w,
        AVG(CASE WHEN status='LOSS' AND pnl<0 THEN pnl END) avg_l
    FROM decisions WHERE status IN ('WIN','LOSS') AND date(ts_utc)=?
""", (today,)).fetchone()
if not r["n"]:
    w("\n  ⚠️ ยังไม่มีไม้ปิดวันนี้"); out.close(); print("done"); raise SystemExit

wr = r["w_n"]/r["n"]*100
pf = r["gw"]/abs(r["gl"]) if r["gl"] else 99
rr = r["avg_w"]/abs(r["avg_l"]) if r["avg_l"] else 0
comm = (r["vol"] or 0) * 6.0
w(f"\n  📊 Trades: {r['n']} ไม้   WIN {r['w_n']} / LOSS {r['l_n']}   WR={wr:.1f}%")
w(f"  💚 Gross WIN  : ${r['gw']:+.2f}   (avg ${r['avg_w'] or 0:.2f})")
w(f"  💔 Gross LOSS : ${r['gl']:+.2f}   (avg ${r['avg_l'] or 0:.2f})")
w(f"  ⚖️  RR เฉลี่ย   : {rr:.2f}   PF: {pf:.2f}")
w(f"  🏷️  Volume      : {r['vol']:.2f} lots → Commission ~${comm:.2f}")
w(f"  ─────────────────────────────────────────")
w(f"  📈 NET PnL     : ${r['net']:+.2f}")
w(f"  💰 NET-Commission : ${(r['net'] or 0) - comm:+.2f}")

# === Series breakdown ===
w("\n" + "="*80); w("📁 SERIES วันนี้"); w("="*80)
rows = c.execute("""
    SELECT id, opened_at_utc, closed_at_utc, side, steps, final_pnl, status
    FROM series WHERE date(opened_at_utc)=? ORDER BY id
""", (today,)).fetchall()
total_series_pnl = 0
clean = retried = stopped = 0
for s in rows:
    pnl = s["final_pnl"] or 0
    total_series_pnl += pnl
    em = "💚" if pnl > 0 else ("💔" if pnl < 0 else "⚪")
    st = (s["status"] or "OPEN")[:20]
    if "LOSS_CAP" in st or "MAX_STEP" in st: stopped += 1
    elif s["steps"] and s["steps"] > 1: retried += 1
    else: clean += 1
    t = s["opened_at_utc"][11:16]
    w(f"  {t}  #{s['id']:<4} {s['side']:<4} steps={s['steps']:<2} {st:<22} {em} ${pnl:+7.2f}")

w(f"\n  ✅ Clean win   : {clean}  (1-step TP)")
w(f"  ♻️  Recovery   : {retried}  (ใช้ 2+ steps)")
w(f"  🛑 Stopped     : {stopped}  (cap/max-steps)")

# === ชั่วโมงที่ดี/แย่ของวันนี้ ===
w("\n" + "="*80); w("🕐 PnL by Hour วันนี้"); w("="*80)
rows = c.execute("""
    SELECT CAST(strftime('%H',ts_utc) AS INT) h, COUNT(*) n,
        SUM(CASE WHEN status='WIN' THEN 1 ELSE 0 END) w_n,
        SUM(pnl) net
    FROM decisions WHERE status IN ('WIN','LOSS') AND date(ts_utc)=?
    GROUP BY h ORDER BY h
""", (today,)).fetchall()
for r2 in rows:
    bkk = (r2['h']+7) % 24
    wr2 = r2['w_n']/r2['n']*100
    flag = "🚀" if r2['net'] > 10 else ("⚠️" if r2['net'] < -10 else "")
    w(f"  {r2['h']:02d}UTC ({bkk:02d}BKK)  n={r2['n']:<3} WR={wr2:5.1f}%  ${r2['net']:+8.2f}  {flag}")

out.close(); print("done")
