"""ดูเฉพาะหลัง retrain ครั้งล่าสุด"""
import sqlite3
DB = r'C:\SweepHunter - Copy\data\db\hyper_trades.sqlite'
out = open(r'C:\SweepHunter - Copy\_post.txt', 'w', encoding='utf-8')
def w(*a): out.write(" ".join(str(x) for x in a)+"\n")
c = sqlite3.connect(DB); c.row_factory = sqlite3.Row

# Snapshot 7 = หลัง retrain
rows = c.execute("""
    SELECT id, ts_utc, prediction, role, status, pnl, volume, confidence, atr, config_snapshot_id sid
    FROM decisions
    WHERE config_snapshot_id >= 7 AND status IN ('WIN','LOSS','OPEN','PENDING')
    ORDER BY id
""").fetchall()
w(f"📊 หลัง retrain (sid >= 7): {len(rows)} ไม้")
w("="*90)
w(f"{'ID':<5} {'Time':<19} {'Side':<4} {'Role':<7} {'sid':<3} {'Status':<7} {'PnL':<8} {'Vol':<5} {'Conf':<5} {'ATR':<5}")
w("-"*90)
total = 0; wins = 0; losses = 0
for r in rows:
    side = "BUY" if r['prediction']==2 else ("SELL" if r['prediction']==0 else "HOLD")
    em = "💚" if (r['pnl'] or 0) > 0 else ("💔" if (r['pnl'] or 0) < 0 else "⏳")
    pnl = r['pnl'] or 0
    total += pnl
    if r['status'] == 'WIN': wins += 1
    elif r['status'] == 'LOSS': losses += 1
    w(f"#{r['id']:<4} {r['ts_utc'][:19]:<19} {side:<4} {(r['role'] or '?')[:6]:<7} {r['sid']:<3} {r['status']:<7} {pnl:+7.2f} {r['volume']:.2f}  {(r['confidence'] or 0):.2f}  {(r['atr'] or 0):.2f} {em}")

w(f"\n📈 SUMMARY: {wins}W / {losses}L  Net = ${total:+.2f}")

# ALL snapshots compared (clean view)
w("\n" + "="*90); w("📊 SNAPSHOT TIMELINE"); w("="*90)
rows = c.execute("""
    SELECT config_snapshot_id sid, COUNT(*) n,
        SUM(CASE WHEN status='WIN' THEN 1 ELSE 0 END) w,
        SUM(CASE WHEN status='LOSS' THEN 1 ELSE 0 END) l,
        SUM(pnl) net,
        AVG(CASE WHEN status='WIN' AND pnl>0 THEN pnl END) avgw,
        AVG(CASE WHEN status='LOSS' AND pnl<0 THEN pnl END) avgl,
        MIN(pnl) worst
    FROM decisions WHERE status IN ('WIN','LOSS')
    GROUP BY sid ORDER BY sid NULLS FIRST
""").fetchall()
for r in rows:
    sid = str(r['sid']) if r['sid'] else "OLD"
    wr = r['w']/r['n']*100 if r['n'] else 0
    pertr = r['net']/r['n'] if r['n'] else 0
    flag = "✅" if r['net'] > 0 else "❌"
    w(f"  sid={sid:<4} n={r['n']:<4} W{r['w']:<3}/L{r['l']:<3} WR={wr:4.0f}%  Net=${r['net']:+8.2f}  $/tr=${pertr:+6.2f}  avgW=${r['avgw'] or 0:5.2f} avgL=${r['avgl'] or 0:6.2f} worst=${r['worst']:+7.2f} {flag}")

out.close(); print("done")
