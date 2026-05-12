"""Simulate direction_flip rule on historical data."""
import sqlite3, sys
from datetime import datetime, timedelta

try: sys.stdout.reconfigure(encoding="utf-8")
except: pass

DB = r'C:\SweepHunter - Copy\data\db\hyper_trades.sqlite'
out = open(r'C:\SweepHunter - Copy\_flip.txt', 'w', encoding='utf-8')
def w(*a): out.write(" ".join(str(x) for x in a)+"\n")

c = sqlite3.connect(DB); c.row_factory = sqlite3.Row
rows = c.execute("""
    SELECT id, ts_utc, prediction, status, pnl
    FROM decisions WHERE status IN ('WIN','LOSS')
    ORDER BY ts_utc, id
""").fetchall()

w(f"📊 รวม trades settled: {len(rows)}\n" + "="*90)


def simulate(min_consec, cooldown_min):
    last_loss_side = None
    consec = 0
    blocked_side = None
    block_until = None
    blocked_count = 0; blocked_pnl = 0.0; blocked_w = 0; blocked_l = 0
    allowed_pnl = 0.0; total = 0.0
    triggers = []
    for r in rows:
        side = "BUY" if r['prediction'] == 2 else "SELL"
        pnl = r['pnl'] or 0
        ts = datetime.fromisoformat(r['ts_utc'])
        total += pnl
        if blocked_side == side and block_until and ts < block_until:
            blocked_count += 1; blocked_pnl += pnl
            if pnl > 0: blocked_w += 1
            else: blocked_l += 1
        else:
            allowed_pnl += pnl
        if pnl > 0:
            consec = 0; last_loss_side = None
            blocked_side = None; block_until = None
        else:
            if last_loss_side == side: consec += 1
            else:
                consec = 1; last_loss_side = side
            if consec >= min_consec and blocked_side != side:
                blocked_side = side
                block_until = ts + timedelta(minutes=cooldown_min)
                triggers.append((r['ts_utc'][:19], side, consec))
    return total, allowed_pnl, blocked_count, blocked_pnl, blocked_w, blocked_l, triggers


# === Main analysis with current settings ===
total, allowed, bc, bp, bw, bl, trigs = simulate(2, 30)
w(f"\n📈 SIMULATION (min_consec=2, cooldown=30min)\n" + "="*90)
w(f"  Total trades:                  {len(rows)}")
w(f"  Trades blocked by flip:        {bc}  ({bc/len(rows)*100:.1f}%)")
w(f"  Total Net WITHOUT flip:        ${total:+.2f}")
w(f"  Total Net WITH flip:           ${allowed:+.2f}")
w(f"  Δ from flip:                   ${allowed-total:+.2f}")
verdict = "✅ HELPED (saved $)" if allowed > total else "❌ HURT (cost opportunity)"
w(f"  → flip {verdict}\n")
w(f"  ── Blocked trades breakdown ──")
w(f"  Would-be WIN (blocked):    {bw} ไม้  → เสียโอกาสกำไร")
w(f"  Would-be LOSS (blocked):   {bl} ไม้  → ช่วยรอดขาดทุน")
w(f"  Net of blocked trades:     ${bp:+.2f}")
w(f"  Trigger events:            {len(trigs)} times")

# === Sensitivity test ===
w(f"\n📊 SENSITIVITY: ลองค่า min_consec ต่างๆ\n" + "="*90)
w(f"  {'min_consec':<12} {'cooldown':<10} {'Blocked':<10} {'Net w/flip':<14} {'Net w/o flip':<14} {'Δ':<10} verdict")
w(f"  " + "-"*88)
configs = [
    (2, 30), (2, 60), (2, 120),
    (3, 30), (3, 60),
    (4, 30), (4, 60),
    (5, 30),
]
for mc, cd in configs:
    t, a, b, _, _, _, _ = simulate(mc, cd)
    delta = a - t
    v = "✅" if delta > 0 else "❌"
    w(f"  {mc:<12} {cd:<10} {b:<10} ${a:<+12.2f} ${t:<+12.2f} ${delta:<+8.2f} {v}")

# === ดู blocked trades ที่ "เคยจะเป็น WIN" ===
w(f"\n🔍 BLOCKED TRADES (min_consec=2, cooldown=30): ตัวอย่าง 15 ไม้\n" + "="*90)
last_loss_side = None; consec = 0; blocked_side = None; block_until = None
samples = []
for r in rows:
    side = "BUY" if r['prediction'] == 2 else "SELL"
    pnl = r['pnl'] or 0
    ts = datetime.fromisoformat(r['ts_utc'])
    if blocked_side == side and block_until and ts < block_until:
        samples.append((r['ts_utc'][:19], side, pnl, r['id']))
    if pnl > 0:
        consec = 0; last_loss_side = None
        blocked_side = None; block_until = None
    else:
        if last_loss_side == side: consec += 1
        else: consec = 1; last_loss_side = side
        if consec >= 2 and blocked_side != side:
            blocked_side = side
            block_until = ts + timedelta(minutes=30)
w(f"  {'Time':<19} {'Side':<5} {'PnL':<10} {'ID':<6}")
for s in samples[-15:]:
    em = "💚" if s[2] > 0 else "💔"
    w(f"  {s[0]:<19} {s[1]:<5} ${s[2]:+8.2f}  #{s[3]:<5} {em}")

out.close(); print("done")
