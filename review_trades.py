"""
Past Trade Reviewer
===================
ทบทวนเทรดในอดีต — หาสาเหตุที่เสีย โดยเฉพาะ:
  • AI confidence สูงแต่ทาย "ผิด" (high conf LOSS)
  • Pattern ที่เกิดบ่อยตอนเสีย (regime/session/spread/atr)
  • Win streak vs Loss streak
  • Confusion matrix แยกตามช่วงเวลา

รัน: python review_trades.py
"""
import sys
import sqlite3
from collections import Counter
from pathlib import Path

# Force utf-8 output for Windows console
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DB_PATH = Path(__file__).parent / "data" / "db" / "hyper_trades.sqlite"


def main() -> None:
    if not DB_PATH.exists():
        print(f"❌ ไม่พบ DB: {DB_PATH}")
        return
    c = sqlite3.connect(str(DB_PATH))
    c.row_factory = sqlite3.Row
    cur = c.cursor()

    settled = cur.execute(
        "SELECT * FROM decisions WHERE status IN ('WIN','LOSS') "
        "ORDER BY ts_utc DESC LIMIT 500"
    ).fetchall()
    if not settled:
        print("❌ ยังไม่มีเทรดที่ปิดแล้ว")
        return

    n = len(settled)
    wins = [r for r in settled if r["status"] == "WIN"]
    losses = [r for r in settled if r["status"] == "LOSS"]
    wr = len(wins) / n * 100
    sum_pnl = sum(float(r["pnl"] or 0) for r in settled)
    avg_win = sum(float(r["pnl"] or 0) for r in wins) / len(wins) if wins else 0
    avg_loss = sum(float(r["pnl"] or 0) for r in losses) / len(losses) if losses else 0

    print(f"\n{'='*60}")
    print(f"📊 Trade Review — {n} trades ล่าสุด")
    print(f"{'='*60}")
    print(f"  Win Rate:      {wr:.1f}%  ({len(wins)}W / {len(losses)}L)")
    print(f"  Net P/L:       ${sum_pnl:+.2f}")
    print(f"  Avg Win:       ${avg_win:+.2f}")
    print(f"  Avg Loss:      ${avg_loss:+.2f}")
    if avg_loss != 0:
        print(f"  RR (real):     1 : {abs(avg_win/avg_loss):.2f}")
    if wr > 0 and avg_loss < 0:
        be_wr = abs(avg_loss) / (abs(avg_loss) + avg_win) * 100 if (avg_loss + avg_win) != 0 else 0
        print(f"  Break-even WR: {be_wr:.1f}%")

    # ── 1. High-conf LOSS (โอกาสที่ AI ผิดมาก)
    print(f"\n🚨 High-Confidence LOSSES (AI มั่นใจแต่ผิด):")
    high_conf_losses = [r for r in losses if (r["confidence"] or 0) >= 0.55]
    print(f"  พบ {len(high_conf_losses)}/{len(losses)} ไม้ ({len(high_conf_losses)/max(1,len(losses))*100:.1f}%)")
    for r in sorted(high_conf_losses, key=lambda x: -(x["confidence"] or 0))[:10]:
        side = "BUY" if r["prediction"] == 2 else "SELL" if r["prediction"] == 0 else "HOLD"
        print(f"    {r['ts_utc'][:16]} {side} conf={r['confidence']*100:.1f}% "
              f"atr={r['atr']:.2f} spread={r['spread_points']:.0f}p PnL=${r['pnl']:+.2f}")

    # ── 2. Confusion by direction
    print(f"\n📈 Win Rate แยกตาม Direction:")
    for pred, name in [(2, "BUY"), (0, "SELL")]:
        sub = [r for r in settled if r["prediction"] == pred]
        if sub:
            sub_wins = sum(1 for r in sub if r["status"] == "WIN")
            print(f"  {name}:  WR {sub_wins/len(sub)*100:.1f}%  ({sub_wins}/{len(sub)})")

    # ── 3. Win Rate by hour UTC
    print(f"\n🕐 Win Rate แยกตาม Hour UTC (top 6 ชั่วโมงที่เทรดเยอะสุด):")
    by_hour = {}
    for r in settled:
        h = r["ts_utc"][11:13]
        by_hour.setdefault(h, []).append(r)
    sorted_hours = sorted(by_hour.items(), key=lambda x: -len(x[1]))[:6]
    for h, rows in sorted_hours:
        w = sum(1 for r in rows if r["status"] == "WIN")
        pnl = sum(float(r["pnl"] or 0) for r in rows)
        print(f"  {h}:00 UTC  WR={w/len(rows)*100:.0f}%  trades={len(rows)}  pnl=${pnl:+.2f}")

    # ── 4. ATR regime impact
    print(f"\n⚡ Win Rate แยกตาม ATR (volatility regime):")
    atrs = sorted([float(r["atr"] or 0) for r in settled if r["atr"]])
    if atrs:
        q33 = atrs[len(atrs)//3]
        q66 = atrs[2*len(atrs)//3]
        groups = {"LOW": [], "MID": [], "HIGH": []}
        for r in settled:
            a = float(r["atr"] or 0)
            g = "LOW" if a < q33 else "MID" if a < q66 else "HIGH"
            groups[g].append(r)
        for g, rows in groups.items():
            if not rows: continue
            w = sum(1 for r in rows if r["status"] == "WIN")
            pnl = sum(float(r["pnl"] or 0) for r in rows)
            print(f"  {g} (ATR<{q33:.1f}/{q66:.1f}/+):  WR={w/len(rows)*100:.0f}%  trades={len(rows)}  pnl=${pnl:+.2f}")

    # ── 5. Spread impact
    print(f"\n📏 Win Rate แยกตาม Spread:")
    spreads = sorted([float(r["spread_points"] or 0) for r in settled if r["spread_points"]])
    if spreads:
        med = spreads[len(spreads)//2]
        wide = [r for r in settled if (r["spread_points"] or 0) > med]
        narrow = [r for r in settled if (r["spread_points"] or 0) <= med]
        for label, rows in [("กว้าง (>median)", wide), ("แคบ (<=median)", narrow)]:
            if not rows: continue
            w = sum(1 for r in rows if r["status"] == "WIN")
            pnl = sum(float(r["pnl"] or 0) for r in rows)
            print(f"  {label:20s}: WR={w/len(rows)*100:.0f}%  trades={len(rows)}  pnl=${pnl:+.2f}")

    # ── 6. Loss streak analysis
    print(f"\n💔 Loss Streaks:")
    streaks = []
    cur_streak = 0
    for r in reversed(settled):
        if r["status"] == "LOSS":
            cur_streak += 1
        else:
            if cur_streak > 0:
                streaks.append(cur_streak)
            cur_streak = 0
    if cur_streak > 0:
        streaks.append(cur_streak)
    if streaks:
        from collections import Counter
        cnt = Counter(streaks)
        for length in sorted(cnt.keys()):
            print(f"  เสีย {length} ไม้ติด: {cnt[length]} ครั้ง")
        print(f"  ➜ longest streak: {max(streaks)} ไม้")

    # ── 7. Recovery effectiveness
    print(f"\n♻️ Recovery Series Outcomes:")
    series = cur.execute(
        "SELECT status, COUNT(*) n, SUM(final_pnl) tot "
        "FROM series WHERE status LIKE 'CLOSED%' GROUP BY status"
    ).fetchall()
    for s in series:
        print(f"  {s['status']:25s}: {s['n']} ครั้ง  pnl รวม=${(s['tot'] or 0):+.2f}")

    # ── 8. Recommendations
    print(f"\n{'='*60}")
    print(f"💡 คำแนะนำจากข้อมูล")
    print(f"{'='*60}")
    if wr < 45:
        print("  ⚠️  WR ต่ำเกิน 45% — model ต้อง retrain หรือเพิ่ม features")
    if high_conf_losses and len(high_conf_losses)/max(1,len(losses)) > 0.4:
        print("  ⚠️  High-conf losses มาก > 40% — มี systematic bias, ต้อง investigate features")
    if avg_loss != 0 and abs(avg_win/avg_loss) < 0.8:
        print(f"  ⚠️  RR แย่ ({abs(avg_win/avg_loss):.2f}<0.8) — ปรับ TP/SL ratio")
    if streaks and max(streaks) >= 5:
        print(f"  ⚠️  Loss streak สูงสุด {max(streaks)} — risk of blow account, ลด max_steps")

    c.close()


if __name__ == "__main__":
    main()
