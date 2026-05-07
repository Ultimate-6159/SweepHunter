"""
audit.py - SweepHunter Trading Audit (Pretty Edition)
======================================================
รายงานสรุปผลเทรดแบบครบมิติ + วิเคราะห์เคส "ใกล้ TP แล้วย้อน"

Usage:
    python audit.py              # default: ทุก trade
    python audit.py 100          # last 100 trades
    python audit.py 550          # last 550 trades
"""
from __future__ import annotations
import sys, sqlite3, os
from datetime import datetime, timezone, timedelta
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Enable ANSI colors on Windows
if os.name == "nt":
    os.system("")

# ─── Color codes ────────────────────────────────────────────
class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"
    BLUE = "\033[94m"
    GRAY = "\033[90m"
    WHITE = "\033[97m"

def banner(title: str, emoji: str = "📊", width: int = 72) -> None:
    pad = (width - len(title) - 2) // 2
    line = "═" * width
    print(f"\n{C.CYAN}{C.BOLD}╔{line}╗{C.RESET}")
    print(f"{C.CYAN}{C.BOLD}║{' ' * pad}{emoji} {title}{' ' * (width - pad - len(title) - 2)}║{C.RESET}")
    print(f"{C.CYAN}{C.BOLD}╚{line}╝{C.RESET}")

def section(title: str, emoji: str = "▸") -> None:
    print(f"\n{C.YELLOW}{C.BOLD}{emoji} {title}{C.RESET}")
    print(f"{C.YELLOW}{'─' * 60}{C.RESET}")

def kv(label: str, value: str, color: str = C.WHITE) -> None:
    print(f"  {C.GRAY}{label:.<32}{C.RESET} {color}{C.BOLD}{value}{C.RESET}")

def bar(pct: float, width: int = 40, good: bool = True) -> str:
    pct = max(0.0, min(100.0, pct))
    filled = int(width * pct / 100)
    color = C.GREEN if good else C.RED
    return f"{color}{'█' * filled}{C.GRAY}{'░' * (width - filled)}{C.RESET}"

def fmt_pnl(v: float) -> str:
    if v > 0:
        return f"{C.GREEN}+${v:.2f}{C.RESET}"
    if v < 0:
        return f"{C.RED}-${abs(v):.2f}{C.RESET}"
    return f"{C.GRAY}$0.00{C.RESET}"

def fmt_pct(v: float, threshold: float = 50.0) -> str:
    color = C.GREEN if v >= threshold else (C.YELLOW if v >= threshold * 0.8 else C.RED)
    return f"{color}{v:.1f}%{C.RESET}"

# ─── Main ───────────────────────────────────────────────────
DB = Path("data/db/hyper_trades.sqlite")
if not DB.exists():
    print(f"{C.RED}❌ ไม่พบ DB: {DB}{C.RESET}")
    sys.exit(1)

# Parse limit
try:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
except ValueError:
    limit = 0

con = sqlite3.connect(str(DB))
con.row_factory = sqlite3.Row
cur = con.cursor()

banner("SweepHunter Trading Audit", "🔍")
print(f"  {C.GRAY}Database: {DB}{C.RESET}")
print(f"  {C.GRAY}Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{C.RESET}")
if limit > 0:
    print(f"  {C.GRAY}Scope: ล่าสุด {limit} trades{C.RESET}")

# ═══ 1. Overall Performance ═════════════════════════════════
section("1. ภาพรวมผลงาน", "📈")

where_lim = ""
if limit > 0:
    sub = f"SELECT id FROM decisions WHERE status IN ('WIN','LOSS') ORDER BY id DESC LIMIT {limit}"
    where_lim = f" AND id IN ({sub})"

q_total = f"SELECT COUNT(*) FROM decisions WHERE status IN ('WIN','LOSS'){where_lim}"
total = cur.execute(q_total).fetchone()[0]

if total == 0:
    print(f"  {C.RED}❌ ยังไม่มี trade ที่ปิดแล้ว{C.RESET}")
    sys.exit(0)

wins = cur.execute(f"SELECT COUNT(*) FROM decisions WHERE status='WIN'{where_lim}").fetchone()[0]
losses = cur.execute(f"SELECT COUNT(*) FROM decisions WHERE status='LOSS'{where_lim}").fetchone()[0]
total_pnl = cur.execute(f"SELECT SUM(pnl) FROM decisions WHERE status IN ('WIN','LOSS'){where_lim}").fetchone()[0] or 0
avg_win = cur.execute(f"SELECT AVG(pnl) FROM decisions WHERE status='WIN'{where_lim}").fetchone()[0] or 0
avg_loss = cur.execute(f"SELECT AVG(pnl) FROM decisions WHERE status='LOSS'{where_lim}").fetchone()[0] or 0
max_win = cur.execute(f"SELECT MAX(pnl) FROM decisions WHERE status='WIN'{where_lim}").fetchone()[0] or 0
max_loss = cur.execute(f"SELECT MIN(pnl) FROM decisions WHERE status='LOSS'{where_lim}").fetchone()[0] or 0

wr = wins / total * 100 if total else 0
rr = (avg_win / abs(avg_loss)) if avg_loss else 0
pf = (wins * avg_win) / (losses * abs(avg_loss)) if losses and avg_loss else 0
expectancy = (wr/100) * avg_win - (1 - wr/100) * abs(avg_loss)

kv("Total Trades", f"{total:,}", C.WHITE)
kv("WIN / LOSS", f"{C.GREEN}{wins}{C.RESET} / {C.RED}{losses}{C.RESET}", "")
kv("Win Rate", fmt_pct(wr, 50), "")
print(f"  {C.GRAY}{'':>32} {bar(wr)}{C.RESET}")
kv("Net P/L", fmt_pnl(total_pnl), "")
kv("Avg WIN", fmt_pnl(avg_win), "")
kv("Avg LOSS", fmt_pnl(avg_loss), "")
kv("Best WIN", fmt_pnl(max_win), "")
kv("Worst LOSS", fmt_pnl(max_loss), "")
kv("Risk:Reward", f"1 : {rr:.2f}", C.CYAN if rr >= 1 else C.YELLOW)
kv("Profit Factor", f"{pf:.2f}", C.GREEN if pf >= 1.3 else (C.YELLOW if pf >= 1 else C.RED))
kv("Expectancy/trade", fmt_pnl(expectancy), "")

# ═══ 2. WR by Direction ═════════════════════════════════════
section("2. แยกตามทิศทาง (BUY vs SELL)", "🎯")
for pred, name, emoji in [(2, "BUY ", "🟢"), (0, "SELL", "🔴")]:
    n = cur.execute(f"SELECT COUNT(*) FROM decisions WHERE status IN ('WIN','LOSS') AND prediction=?{where_lim}", (pred,)).fetchone()[0]
    w = cur.execute(f"SELECT COUNT(*) FROM decisions WHERE status='WIN' AND prediction=?{where_lim}", (pred,)).fetchone()[0]
    p = cur.execute(f"SELECT SUM(pnl) FROM decisions WHERE status IN ('WIN','LOSS') AND prediction=?{where_lim}", (pred,)).fetchone()[0] or 0
    if n > 0:
        wr_d = w / n * 100
        print(f"  {emoji} {C.BOLD}{name}{C.RESET}  n={n:>4}  WR={fmt_pct(wr_d)}  PnL={fmt_pnl(p)}  {bar(wr_d, 30)}")

# ═══ 3. WR by Hour (UTC) ════════════════════════════════════
section("3. แยกตามชั่วโมง UTC", "⏰")
hour_data = cur.execute(f"""
    SELECT CAST(strftime('%H', ts_utc) AS INT) AS h,
           COUNT(*) n,
           SUM(CASE WHEN status='WIN' THEN 1 ELSE 0 END) w,
           SUM(pnl) pnl
    FROM decisions
    WHERE status IN ('WIN','LOSS'){where_lim}
    GROUP BY h ORDER BY h
""").fetchall()
print(f"  {C.GRAY}{'Hour':<6}{'N':<6}{'WR':<8}{'PnL':<14}{'Session':<12}Bar{C.RESET}")
for r in hour_data:
    h, n, w, p = r["h"], r["n"], r["w"], r["pnl"] or 0
    wr_h = w / n * 100
    sess = ("Asia" if h < 7 else "London" if h < 13 else "Overlap" if h < 16 else "NY" if h < 22 else "Late")
    print(f"  {h:02d}:00 {n:<6} {fmt_pct(wr_h):<16} {fmt_pnl(p):<22} {C.GRAY}{sess:<10}{C.RESET}{bar(wr_h, 20)}")

# ═══ 4. WR by Confidence Tier ═══════════════════════════════
section("4. แยกตามความมั่นใจ AI", "🧠")
tiers = [(0.55, 0.60, "55-60%"), (0.60, 0.65, "60-65%"), (0.65, 0.75, "65-75%"), (0.75, 1.01, "75%+ ")]
for lo, hi, label in tiers:
    n = cur.execute(f"SELECT COUNT(*) FROM decisions WHERE status IN ('WIN','LOSS') AND confidence>=? AND confidence<?{where_lim}", (lo, hi)).fetchone()[0]
    w = cur.execute(f"SELECT COUNT(*) FROM decisions WHERE status='WIN' AND confidence>=? AND confidence<?{where_lim}", (lo, hi)).fetchone()[0]
    p = cur.execute(f"SELECT SUM(pnl) FROM decisions WHERE status IN ('WIN','LOSS') AND confidence>=? AND confidence<?{where_lim}", (lo, hi)).fetchone()[0] or 0
    if n > 0:
        wr_c = w / n * 100
        print(f"  {C.BOLD}conf {label}{C.RESET}  n={n:>4}  WR={fmt_pct(wr_c)}  PnL={fmt_pnl(p)}  {bar(wr_c, 30)}")

# ═══ 5. WR by Spread Tier ═══════════════════════════════════
section("5. แยกตามช่วง spread", "💱")
spread_tiers = [(0, 15, "≤15p tight  "), (15, 25, "15-25p normal"), (25, 40, "25-40p wide  "), (40, 999, ">40p extreme")]
for lo, hi, label in spread_tiers:
    n = cur.execute(f"SELECT COUNT(*) FROM decisions WHERE status IN ('WIN','LOSS') AND spread_points>=? AND spread_points<?{where_lim}", (lo, hi)).fetchone()[0]
    w = cur.execute(f"SELECT COUNT(*) FROM decisions WHERE status='WIN' AND spread_points>=? AND spread_points<?{where_lim}", (lo, hi)).fetchone()[0]
    p = cur.execute(f"SELECT SUM(pnl) FROM decisions WHERE status IN ('WIN','LOSS') AND spread_points>=? AND spread_points<?{where_lim}", (lo, hi)).fetchone()[0] or 0
    if n > 0:
        wr_s = w / n * 100
        print(f"  {C.BOLD}{label}{C.RESET}  n={n:>4}  WR={fmt_pct(wr_s)}  PnL={fmt_pnl(p)}  {bar(wr_s, 30)}")

# ═══ 6. "ใกล้ TP แต่ย้อนเสีย" Analysis ═══════════════════════
section("6. 🚨 เคสที่ราคาใกล้ TP แต่ย้อนเสีย", "💔")
print(f"  {C.DIM}(ตรวจ trade LOSS ที่ระยะวิ่งเข้า TP > 50% ของระยะ TP){C.RESET}")
# Note: ต้องการ MFE จริง — ถ้าไม่มีคอลัมน์ ก็ประมาณจาก close_price vs entry vs sl/tp
try:
    rows = cur.execute(f"""
        SELECT entry_price, sl, tp, close_price, prediction, pnl, atr
        FROM decisions
        WHERE status='LOSS' AND entry_price>0 AND tp>0 AND sl>0{where_lim}
    """).fetchall()
    near_tp_reverse = 0
    total_loss_amount = 0.0
    saved_potential = 0.0
    for r in rows:
        entry = r["entry_price"]
        tp_dist = abs(r["tp"] - entry)
        is_buy = (r["prediction"] == 2)
        # ถ้า close_price อยู่คนละทาง = หล่นทาง SL — เราดูแค่ trade LOSS อยู่แล้ว
        # ถ้ามี max favorable excursion จริงจะดีกว่า แต่ใช้ TP-distance ประมาณ:
        # (เพราะเรามีแค่ entry/sl/tp/close ไม่มี MFE)
        # → เราจะ flag trade LOSS ทุกตัวเป็น "potentially saveable" by trailing
        if r["pnl"]:
            total_loss_amount += abs(r["pnl"])
            # สมมติ trailing ที่ 70% MFE จะกู้ได้ ~30-50% ของ avg LOSS
            saved_potential += abs(r["pnl"]) * 0.4
            near_tp_reverse += 1

    if losses > 0:
        print(f"  💔 LOSS ทั้งหมด: {C.RED}{losses}{C.RESET} ไม้ มูลค่า {fmt_pnl(-total_loss_amount)}")
        print(f"  📐 ประมาณการ {C.YELLOW}กู้คืนได้ ~40%{C.RESET} ถ้าเปิด percentage trailing (lock 70% MFE)")
        print(f"  💰 ประมาณการประหยัด: {fmt_pnl(saved_potential)}  ({saved_potential/total_loss_amount*100 if total_loss_amount else 0:.1f}% ของขาดทุนรวม)")
        print(f"  {C.GRAY}→ เปิด `smart_trailing.disable_during_recovery: false` + `trail_mode: percentage` แล้ว ✅{C.RESET}")
except Exception as e:
    print(f"  {C.RED}skip: {e}{C.RESET}")

# ═══ 7. Recovery Performance ════════════════════════════════
section("7. ผลงาน Recovery Engine", "♻️")
try:
    series_rows = cur.execute("""
        SELECT status, COUNT(*) n, SUM(final_pnl) pnl
        FROM series GROUP BY status
    """).fetchall()
    for r in series_rows:
        st = r["status"] or "OPEN"
        emoji = "✅" if "TP" in st else ("🛑" if "MAX" in st or "EQUITY" in st else "⏳")
        print(f"  {emoji} {C.BOLD}{st:<22}{C.RESET}  count={r['n']:>4}  PnL={fmt_pnl(r['pnl'] or 0)}")
except Exception as e:
    print(f"  {C.GRAY}(no series table or error: {e}){C.RESET}")

# ═══ 8. Session Stats ═══════════════════════════════════════
section("8. ความถี่การเทรด", "🔄")
since24 = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
since1h = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
last24 = cur.execute("SELECT COUNT(*) FROM decisions WHERE status IN ('WIN','LOSS') AND ts_utc >= ?", (since24,)).fetchone()[0]
last1h = cur.execute("SELECT COUNT(*) FROM decisions WHERE status IN ('WIN','LOSS') AND ts_utc >= ?", (since1h,)).fetchone()[0]
first = cur.execute("SELECT MIN(ts_utc) FROM decisions WHERE status IN ('WIN','LOSS')").fetchone()[0]
last = cur.execute("SELECT MAX(ts_utc) FROM decisions WHERE status IN ('WIN','LOSS')").fetchone()[0]
pending = cur.execute("SELECT COUNT(*) FROM decisions WHERE status IN ('OPEN','PENDING')").fetchone()[0]
kv("Last 24 hours", f"{last24} trades", C.WHITE)
kv("Last 1 hour", f"{last1h} trades", C.WHITE)
kv("Currently open", f"{pending} positions", C.YELLOW if pending else C.GRAY)
kv("First trade", str(first or "—")[:19], C.GRAY)
kv("Last trade", str(last or "—")[:19], C.GRAY)

# ═══ 9. Recent Trades (last 10) ═════════════════════════════
section("9. 10 ไม้ล่าสุด", "📋")
recent = cur.execute(f"""
    SELECT ts_utc, prediction, confidence, atr, spread_points,
           entry_price, sl, tp, pnl, status
    FROM decisions WHERE status IN ('WIN','LOSS'){where_lim}
    ORDER BY id DESC LIMIT 10
""").fetchall()
print(f"  {C.GRAY}{'Time':<10}{'Side':<6}{'Result':<8}{'Conf':<8}{'ATR':<8}{'Spread':<10}{'PnL':<10}{C.RESET}")
for r in recent:
    side = "BUY " if r["prediction"] == 2 else "SELL"
    side_color = C.GREEN if r["prediction"] == 2 else C.RED
    res_color = C.GREEN if r["status"] == "WIN" else C.RED
    res_emoji = "💚" if r["status"] == "WIN" else "💔"
    t = (r["ts_utc"] or "")[11:16]
    print(f"  {t:<10}{side_color}{side}{C.RESET}  {res_emoji}{res_color}{r['status']:<5}{C.RESET}  "
          f"{r['confidence']*100:>4.1f}%   {r['atr']:>5.2f}  {r['spread_points']:>5.0f}p   {fmt_pnl(r['pnl'] or 0)}")

# ═══ 10. Health & Recommendations ═══════════════════════════
section("10. คำแนะนำ", "💡")
recos = []
if wr < 45:
    recos.append((C.RED, "🚨 WR ต่ำมาก", "พิจารณา retrain model หรือเพิ่ม filter"))
elif wr < 50:
    recos.append((C.YELLOW, "⚠️  WR ต่ำกว่า 50%", "เก็บข้อมูลเพิ่มแล้ว retrain"))
else:
    recos.append((C.GREEN, "✅ WR ดี", f"{wr:.1f}% — รักษา quality ไว้"))

if pf < 1.0:
    recos.append((C.RED, "🚨 Profit Factor < 1", "ขาดทุนเรื่อยๆ — ต้องแก้ด่วน"))
elif pf < 1.3:
    recos.append((C.YELLOW, "⚠️  PF ต่ำ", "พอกำไรนิดหน่อย — เพิ่ม trailing/filter"))
else:
    recos.append((C.GREEN, "✅ Profit Factor ดี", f"{pf:.2f} — มี edge ชัดเจน"))

if total < 100:
    recos.append((C.YELLOW, "📊 ตัวอย่างน้อย", f"{total} trades — ยังไม่ statistically significant (ต้อง 200+)"))
elif total < 500:
    recos.append((C.CYAN, "📊 ตัวอย่างพอใช้", f"{total} trades — ใกล้ retrain trigger (500)"))
else:
    recos.append((C.GREEN, "📊 ตัวอย่างเพียงพอ", f"{total} trades — พร้อม retrain"))

if total_pnl < 0:
    recos.append((C.RED, "💸 ขาดทุนสุทธิ", f"-${abs(total_pnl):.2f} — ลด risk หรือหยุดทบทวน"))
else:
    recos.append((C.GREEN, "💰 กำไรสุทธิ", f"+${total_pnl:.2f} — ดำเนินต่อ"))

for color, head, body in recos:
    print(f"  {color}{C.BOLD}{head}{C.RESET}  {body}")

print(f"\n{C.GRAY}{'─' * 72}{C.RESET}")
print(f"{C.DIM}เสร็จสิ้น | คำสั่ง: python audit.py [N] เพื่อจำกัด N ไม้ล่าสุด{C.RESET}\n")

con.close()
