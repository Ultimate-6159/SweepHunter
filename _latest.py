"""ดู 20 trades ล่าสุดแบบละเอียด"""
import sqlite3
DB = r'C:\SweepHunter - Copy\data\db\hyper_trades.sqlite'
out = open(r'C:\SweepHunter - Copy\_latest.txt', 'w', encoding='utf-8')
def w(*a): out.write(" ".join(str(x) for x in a)+"\n")

c = sqlite3.connect(DB); c.row_factory = sqlite3.Row
rows = c.execute("""
    SELECT id, ts_utc, prediction, role, status, pnl, volume, confidence, atr, config_snapshot_id sid
    FROM decisions
    WHERE status IN ('WIN','LOSS')
    ORDER BY id DESC LIMIT 30
""").fetchall()

w(f"{'ID':<5} {'Time':<19} {'Side':<4} {'Role':<8} {'sid':<3} {'Status':<5} {'PnL':<8} {'Vol':<5} {'Conf':<5} {'ATR':<5}")
w("-"*90)
for r in rows:
    side = "BUY" if r['prediction']==2 else ("SELL" if r['prediction']==0 else "HOLD")
    sid = str(r['sid']) if r['sid'] else "?"
    em = "💚" if (r['pnl'] or 0) > 0 else "💔"
    w(f"#{r['id']:<4} {r['ts_utc'][:19]:<19} {side:<4} {(r['role'] or '?')[:7]:<8} {sid:<3} {r['status']:<5} {(r['pnl'] or 0):+7.2f} {r['volume']:.2f}  {(r['confidence'] or 0):.2f}  {(r['atr'] or 0):.2f} {em}")
out.close(); print("done")
