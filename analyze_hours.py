import sqlite3
from datetime import datetime, timezone
from collections import defaultdict

db_path = r'C:\SweepHunter - 6159\data\db\hyper_trades.sqlite'
conn = sqlite3.connect(db_path)
cur = conn.cursor()

# decisions: WIN/LOSS trades ที่ปิดแล้ว
cur.execute("""
    SELECT closed_at_utc, pnl, volume
    FROM decisions
    WHERE status IN ('WIN','LOSS') AND closed_at_utc IS NOT NULL AND pnl IS NOT NULL
""")
rows = cur.fetchall()
conn.close()

print(f"Total closed trades (WIN+LOSS): {len(rows)}")

OLD_BLOCKED = {0, 1, 2, 3, 4, 11, 12, 19, 20, 23}
NOW_BLOCKED  = {0, 1, 2, 3, 4}
UNBLOCKED    = OLD_BLOCKED - NOW_BLOCKED   # {11,12,19,20,23}

hour_stats = defaultdict(lambda: {'wins': 0, 'losses': 0, 'pnl': 0.0, 'trades': 0})

for closed_at, pnl, vol in rows:
    try:
        # closed_at_utc เป็น ISO string เช่น "2026-04-23T04:03:39.278797+00:00"
        dt = datetime.fromisoformat(str(closed_at))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        dt_utc = dt.astimezone(timezone.utc)
        h_utc = dt_utc.hour   # UTC hour จริง
        h_broker = (h_utc + 3) % 24  # broker hour (UTC+3)
        hour_stats[h_broker]['trades'] += 1
        hour_stats[h_broker]['pnl'] += pnl
        if pnl > 0:
            hour_stats[h_broker]['wins'] += 1
        else:
            hour_stats[h_broker]['losses'] += 1
    except Exception as e:
        pass

print(f"\n{'BrokerH':>8} {'UTC':>5} | {'Trades':>6} | {'WR%':>6} | {'Net PnL':>10} | Status")
print("-" * 76)
for h in sorted(hour_stats.keys()):
    s = hour_stats[h]
    wr = s['wins'] / s['trades'] * 100 if s['trades'] > 0 else 0
    utc_h = (h - 3) % 24
    if h in NOW_BLOCKED:
        status = "BLOCKED (ยังบล็อก)"
    elif h in UNBLOCKED:
        status = "<<< NOW OPEN (ปลดบล็อกแล้ว)"
    else:
        status = ""
    pnl_val = s['pnl']
    print(f"  B{h:02d} (UTC{utc_h:02d}) | {s['trades']:6d} | {wr:5.1f}% | ${pnl_val:+9.2f} | {status}")

print("\n=== ผล: ชั่วโมงที่เพิ่งปลดบล็อก ===")
total_open_pnl = 0.0
for h in sorted(UNBLOCKED):
    s = hour_stats.get(h, {'trades': 0, 'wins': 0, 'pnl': 0.0})
    utc_h = (h - 3) % 24
    wr = s['wins'] / s['trades'] * 100 if s['trades'] > 0 else 0
    total_open_pnl += s['pnl']
    if s['trades'] < 5:
        verdict = "⚠️  ข้อมูลน้อยเกินไปสรุปไม่ได้"
    elif s['pnl'] > 0 and wr >= 45:
        verdict = "✅ ดี"
    elif s['pnl'] < 0 and wr < 45:
        verdict = "❌ ควรบล็อกกลับ"
    else:
        verdict = "🟡 กลางๆ"
    print(f"  Broker {h:02d} (UTC {utc_h:02d}:xx) | {s['trades']:3d} trades | WR={wr:.1f}% | PnL=${s['pnl']:+.2f} | {verdict}")

print(f"\n  รวม PnL ชั่วโมงที่ปลดบล็อก: ${total_open_pnl:+.2f}")
print(f"  (ถ้าลบ = ควรบล็อกกลับ, ถ้าบวก = ดีที่ปลดบล็อก)")

print("\n=== ชั่วโมงยังบล็อกอยู่ [0-4] (broker) ===")
total_blocked_pnl = 0.0
for h in sorted(NOW_BLOCKED):
    s = hour_stats.get(h, {'trades': 0, 'wins': 0, 'pnl': 0.0})
    utc_h = (h - 3) % 24
    wr = s['wins'] / s['trades'] * 100 if s['trades'] > 0 else 0
    total_blocked_pnl += s['pnl']
    print(f"  Broker {h:02d} (UTC {utc_h:02d}:xx) | {s['trades']:3d} trades | WR={wr:.1f}% | PnL=${s['pnl']:+.2f}")
print(f"  รวม PnL ชั่วโมงที่ยังบล็อก: ${total_blocked_pnl:+.2f}")
