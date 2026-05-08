"""Quick log analysis — last 30 trades + SL/TP distance vs ATR"""
import sys, sqlite3
from pathlib import Path
try: sys.stdout.reconfigure(encoding="utf-8")
except: pass

DB = Path("data/db/hyper_trades.sqlite")
c = sqlite3.connect(str(DB))
c.row_factory = sqlite3.Row
cur = c.cursor()

print("=== LAST 30 TRADES (entry context) ===")
rows = cur.execute("""SELECT ts_utc, prediction, confidence, atr, spread_points, entry_price, sl, tp, pnl, status
                      FROM decisions WHERE status IN ('WIN','LOSS')
                      ORDER BY ts_utc DESC LIMIT 30""").fetchall()
for r in rows:
    side = 'BUY ' if r['prediction']==2 else 'SELL'
    sl_d = abs((r['entry_price'] or 0) - (r['sl'] or 0))
    tp_d = abs((r['entry_price'] or 0) - (r['tp'] or 0))
    pnl = r['pnl'] or 0
    print(f"{r['ts_utc'][11:16]} {r['status']:4} {side} conf={r['confidence']*100:.0f}% atr={r['atr']:.2f} sprd={r['spread_points']:.0f}p sl_dist={sl_d:.2f} tp_dist={tp_d:.2f} pnl=${pnl:+.2f}")

print("\n=== SL/TP distances vs ATR (WIN vs LOSS) ===")
for status in ['WIN','LOSS']:
    rows = cur.execute("""SELECT atr, ABS(entry_price-sl) sl_d, ABS(entry_price-tp) tp_d
                          FROM decisions WHERE status=? AND atr>0 AND entry_price>0""", (status,)).fetchall()
    if rows:
        n = len(rows)
        avg_atr = sum(r['atr'] for r in rows)/n
        avg_sl = sum(r['sl_d'] for r in rows)/n
        avg_tp = sum(r['tp_d'] for r in rows)/n
        print(f"  {status}: n={n} avg_atr={avg_atr:.2f} avg_sl={avg_sl:.2f}({avg_sl/avg_atr:.2f}xATR) avg_tp={avg_tp:.2f}({avg_tp/avg_atr:.2f}xATR)")

print("\n=== TIME-TO-RESOLVE (entry → close, in minutes) for last 50 ===")
rows = cur.execute("""SELECT ts_utc, closed_at_utc, status, pnl, ABS(entry_price-close_price) move
                      FROM decisions WHERE status IN ('WIN','LOSS') AND closed_at_utc IS NOT NULL
                      ORDER BY ts_utc DESC LIMIT 50""").fetchall()
from datetime import datetime
durations = []
for r in rows:
    try:
        t1 = datetime.fromisoformat(r['ts_utc'])
        t2 = datetime.fromisoformat(r['closed_at_utc'])
        d = (t2-t1).total_seconds()/60
        durations.append((r['status'], d, r['move'] or 0))
    except: pass
if durations:
    wins = [d for s,d,m in durations if s=='WIN']
    losses = [d for s,d,m in durations if s=='LOSS']
    if wins: print(f"  WIN  duration: avg={sum(wins)/len(wins):.1f}min  median={sorted(wins)[len(wins)//2]:.1f}min  min={min(wins):.1f}  max={max(wins):.1f}")
    if losses: print(f"  LOSS duration: avg={sum(losses)/len(losses):.1f}min  median={sorted(losses)[len(losses)//2]:.1f}min  min={min(losses):.1f}  max={max(losses):.1f}")

print("\n=== HIGH-CONF LOSSES (≥55%) — for pattern analysis ===")
rows = cur.execute("""SELECT ts_utc, prediction, confidence, atr, entry_price, close_price, pnl
                      FROM decisions WHERE status='LOSS' AND confidence>=0.55
                      ORDER BY ts_utc DESC LIMIT 15""").fetchall()
for r in rows:
    side = 'BUY ' if r['prediction']==2 else 'SELL'
    move = (r['close_price'] or 0) - (r['entry_price'] or 0)
    move_atr = move / r['atr'] if r['atr'] else 0
    print(f"  {r['ts_utc'][11:16]} {side} conf={r['confidence']*100:.0f}% entry={r['entry_price']:.2f} close={r['close_price']:.2f} move={move:+.2f} ({move_atr:+.2f}xATR) pnl=${r['pnl']:+.2f}")
