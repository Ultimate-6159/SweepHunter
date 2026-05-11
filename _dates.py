import sqlite3
c = sqlite3.connect(r'C:\SweepHunter - Copy\data\db\hyper_trades.sqlite')
rows = c.execute("SELECT date(ts_utc) d, COUNT(*) n FROM decisions WHERE status IN ('WIN','LOSS') GROUP BY d ORDER BY d DESC LIMIT 7").fetchall()
with open(r'C:\SweepHunter - Copy\_dates.txt', 'w') as f:
    for r in rows:
        f.write(f"{r[0]}: {r[1]} trades\n")
