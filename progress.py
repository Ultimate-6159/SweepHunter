"""Quick progress check"""
import sys, sqlite3
from datetime import datetime, timezone, timedelta
try: sys.stdout.reconfigure(encoding="utf-8")
except: pass

c = sqlite3.connect('data/db/hyper_trades.sqlite')
c.row_factory = sqlite3.Row
cur = c.cursor()

target = 500
total = cur.execute("SELECT COUNT(*) FROM decisions WHERE status IN ('WIN','LOSS')").fetchone()[0]
print(f'=== Data Collection Progress ===')
print(f'Settled trades: {total} / {target} ({total/target*100:.1f}%)')
bar_len = 40
filled = int(bar_len * min(1, total/target))
print(f'  [{"#"*filled}{"-"*(bar_len-filled)}]')
print(f'Remaining: {max(0,target-total)} trades')
print()

since24 = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
last24 = cur.execute("SELECT COUNT(*) FROM decisions WHERE status IN ('WIN','LOSS') AND ts_utc >= ?", (since24,)).fetchone()[0]
print(f'Last 24h: {last24} trades')
if last24 > 0:
    days_left = max(0,target-total) / last24
    print(f'Estimated days to reach {target}: {days_left:.1f} days')

since1h = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
last1h = cur.execute("SELECT COUNT(*) FROM decisions WHERE status IN ('WIN','LOSS') AND ts_utc >= ?", (since1h,)).fetchone()[0]
print(f'Last 1h:  {last1h} trades')

print()
first = cur.execute("SELECT MIN(ts_utc) FROM decisions WHERE status IN ('WIN','LOSS')").fetchone()[0]
last = cur.execute("SELECT MAX(ts_utc) FROM decisions WHERE status IN ('WIN','LOSS')").fetchone()[0]
print(f'First trade: {first}')
print(f'Last trade:  {last}')

print()
wins = cur.execute("SELECT COUNT(*) FROM decisions WHERE status='WIN'").fetchone()[0]
losses = cur.execute("SELECT COUNT(*) FROM decisions WHERE status='LOSS'").fetchone()[0]
total_pnl = cur.execute("SELECT SUM(pnl) FROM decisions WHERE status IN ('WIN','LOSS')").fetchone()[0] or 0
if wins+losses > 0:
    print(f'WIN: {wins} | LOSS: {losses} | WR: {wins/(wins+losses)*100:.1f}% | Net P/L: ${total_pnl:+.2f}')

# Pending (open) trades
pending = cur.execute("SELECT COUNT(*) FROM decisions WHERE status IN ('OPEN','PENDING')").fetchone()[0]
print(f'Pending (open): {pending} trades')
