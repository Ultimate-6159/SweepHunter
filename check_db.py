import sqlite3
from datetime import datetime, timezone
from collections import defaultdict

db_path = r'C:\SweepHunter - 6159\data\db\hyper_trades.sqlite'
conn = sqlite3.connect(db_path)
cur = conn.cursor()

# ดู sample decisions
cur.execute("SELECT DISTINCT status FROM decisions LIMIT 20")
print("Decision statuses:", [r[0] for r in cur.fetchall()])

cur.execute("SELECT DISTINCT status FROM series LIMIT 20")
print("Series statuses:", [r[0] for r in cur.fetchall()])

cur.execute("SELECT COUNT(*) FROM decisions")
print("Total decisions:", cur.fetchone()[0])

cur.execute("SELECT COUNT(*) FROM series")
print("Total series:", cur.fetchone()[0])

# sample rows
cur.execute("SELECT id, ts_utc, pnl, status, closed_at_utc FROM decisions LIMIT 5")
print("\nSample decisions:")
for r in cur.fetchall():
    print(" ", r)

cur.execute("SELECT id, opened_at_utc, closed_at_utc, final_pnl, status FROM series LIMIT 5")
print("\nSample series:")
for r in cur.fetchall():
    print(" ", r)

conn.close()
