"""Analyze day x hour (broker TZ) — Hybrid rule: Profit Factor + WR.

Rule: trades >= MIN_TRADES AND (PF < PF_BLOCK OR WR < WR_BLOCK_PCT)
  PF = gross_wins/gross_losses (lot-size independent — safe to mix model configs)
  WR = secondary safety net

Usage: python analyze_slots.py [--dry-run] [--yes] [--since-snapshot=N]
  --dry-run : แสดงผลอย่างเดียว ไม่แก้ config.json (ไม่ถามยืนยัน)
  --yes     : apply โดยไม่ถามยืนยัน (สำหรับ automation)
  --since-snapshot=N : ใช้เฉพาะไม้ config_snapshot_id >= N (ยุด config ปัจจุบัน)
                       เกณฑ์รวมลดเหลือ 100 ไม้ (แทน 300)

ป้องกัน 3 ชั้น:
  1. ต้องมี trade รวม >= MIN_TOTAL_TRADES  (ข้อมูลพอก่อนตัดสิน)
  2. แต่ละ slot ต้องมี trades >= MIN_TRADES (sample ต่อ slot พอ)
  3. ถามยืนยันก่อน apply เสมอ (ยกเว้น --yes)
"""
import json
import sys
import sqlite3
import shutil
from collections import defaultdict
from datetime import datetime, timezone

DRY_RUN       = "--dry-run" in sys.argv
AUTO_YES      = "--yes" in sys.argv
SINCE_SNAPSHOT = 0
for _arg in sys.argv:
    if _arg.startswith("--since-snapshot="):
        SINCE_SNAPSHOT = int(_arg.split("=", 1)[1])
BROKER_OFFSET = 3
MIN_TRADES    = 8       # trades ต่อ slot ขั้นต่ำ (sample size)
MIN_TOTAL_TRADES = 300  # trades รวมทั้งหมดขั้นต่ำ ก่อนจะ update ได้
MIN_TOTAL_ERA  = 100    # เมื่อ --since-snapshot ใช้เกณฑ์ต่ำกว่า (ยุดเดียว)
PF_BLOCK      = 0.9     # block ถ้า PF < ค่านี้ (ขาดทุนสุทธิ)
# หมายเหตุ: ไม่ใช้ WR < X% เป็นเงื่อนไขเดี่ยวอีกต่อไป
#   เพราะระบบ TP=1.6×ATR, SL=0.8×ATR (RR 2:1) breakeven ที่ WR ~33%
#   WR=37-39% + PF>1.0 = ยังทำกำไรได้จริง ไม่ควรบล็อก
#   ใช้ PF เป็นตัวชี้วัดหลัก: PF>1.0 = slot ทำกำไรสุทธิ ไม่ว่า WR จะเท่าไหร่
DB            = "data/db/hyper_trades.sqlite"
CONFIG        = "config.json"
DOW           = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


def profit_factor(gw: float, gl: float) -> float:
    return gw / gl if gl > 0 else 999.0


def should_block(t: int, pf: float, wr: float) -> bool:
    # กฎ: PF เป็นตัวชี้วัดเดียว — PF < PF_BLOCK = slot ขาดทุนสุทธิ
    # ไม่บล็อกเพราะ WR ต่ำอย่างเดียว เพราะ RR 2:1 ทำให้ WR ต่ำยังกำไรได้
    return t >= MIN_TRADES and pf < PF_BLOCK


def should_unblock(t: int, pf: float) -> bool:
    # ปลดบล็อกเฉพาะเมื่อมี sample พอและ PF ดีขึ้นชัดเจน
    return t >= MIN_TRADES and pf >= PF_BLOCK


# ── load trades ───────────────────────────────────────────────────────────────
conn = sqlite3.connect(DB)
if SINCE_SNAPSHOT > 0:
    rows = conn.execute("""
        SELECT closed_at_utc, pnl FROM decisions
        WHERE status IN ('WIN','LOSS') AND closed_at_utc IS NOT NULL AND pnl IS NOT NULL
          AND config_snapshot_id >= ?
    """, (SINCE_SNAPSHOT,)).fetchall()
else:
    rows = conn.execute("""
        SELECT closed_at_utc, pnl FROM decisions
        WHERE status IN ('WIN','LOSS') AND closed_at_utc IS NOT NULL AND pnl IS NOT NULL
    """).fetchall()
conn.close()

min_total_eff = MIN_TOTAL_ERA if SINCE_SNAPSHOT > 0 else MIN_TOTAL_TRADES

slot = defaultdict(lambda: {"t": 0, "pnl": 0.0, "w": 0, "gw": 0.0, "gl": 0.0})
for closed_at, pnl in rows:
    dt = datetime.fromisoformat(str(closed_at))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)
    bh = (dt.hour + BROKER_OFFSET) % 24
    key = (dt.weekday(), bh)
    s = slot[key]
    s["t"] += 1
    s["pnl"] += pnl
    if pnl > 0:
        s["w"] += 1
        s["gw"] += pnl
    else:
        s["gl"] += abs(pnl)

total = len(rows)
SEP = "=" * 60

trade_dates = set()
recent_by_date: dict = defaultdict(lambda: {"t": 0, "pnl": 0.0})
for closed_at, pnl in rows:
    dt = datetime.fromisoformat(str(closed_at))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    trade_dates.add(dt.date())
    recent_by_date[dt.date()]["t"] += 1
    recent_by_date[dt.date()]["pnl"] += pnl

# ── หัวข้อ ────────────────────────────────────────────────────────────────────
print(SEP)
print("  SweepHunter — วิเคราะห์ช่วงเวลาที่ควรบล็อก (broker TZ)")
print(SEP)
print(f"  ข้อมูลที่ใช้   : {total} trades (ต้องการ >= {min_total_eff} ถึงจะ update ได้)")
if SINCE_SNAPSHOT > 0:
    print(f"  กรองยุด      : config_snapshot_id >= {SINCE_SNAPSHOT} (ข้อมูล config ปัจจุบันเท่านั้น)")
else:
    print(f"  กรองยุด      : ทั้งหมด (รวมข้อมูลเก่า — ใช้ --since-snapshot=14 สำหรับยุดปัจจุบัน)")
if trade_dates:
    d0, d1 = min(trade_dates), max(trade_dates)
    dow_seen = sorted({d.weekday() for d in trade_dates})
    print(f"  ช่วงวันที่   : {d0} → {d1} ({len(trade_dates)} วันทำการ)")
    print(f"  วันที่มีข้อมูล : {', '.join(DOW[d] for d in dow_seen)}")
print()
print("  ไม้รายวัน (ปิดแล้ว WIN/LOSS — อ่านจาก DB ทุกครั้งที่รัน, ไม่เก็บแยก):")
for d in sorted(recent_by_date.keys(), reverse=True)[:7]:
    s = recent_by_date[d]
    dow = DOW[d.weekday()]
    mark = " ← วันล่าสุด" if d == max(trade_dates) else ""
    print(f"    {d} ({dow})  {s['t']:3d} ไม้  PnL ${s['pnl']:+.2f}{mark}")
print("  (เสาร์-อาทิตย์ตลาดปิด → ไม่มีไม้ | ไม้ที่ยังเปิดอยู่ยังไม่นับจนกว่าจะปิด)")
print()
print(f"  กฎการบล็อก    : trades >= {MIN_TRADES} ไม้ และ PF < {PF_BLOCK}")
print(f"  ระบบ RR 2:1   : TP=1.6xATR / SL=0.8xATR  breakeven ที่ WR ~33%")
print(f"                  WR ต่ำ + PF>1.0 = ยังกำไร ไม่บล็อก  |  PF<1.0 = ขาดทุนสุทธิ บล็อก")
print()
print("  คำอธิบายคอลัมน์:")
print("    Tr    = จำนวน trade ที่ผ่านมาในช่วงนี้")
print("    WR%   = อัตราชนะ — แสดงข้อมูลเพิ่มเติม ไม่ใช่เกณฑ์บล็อกอีกต่อไป")
print("    PF    = Profit Factor = กำไรรวม / ขาดทุนรวม  (ไม่ขึ้นกับขนาด lot)")
print("            PF > 1.0 = slot ทำกำไรสุทธิ (เปิดเทรดได้แม้ WR จะต่ำ)")
print("            PF < 1.0 = slot ขาดทุนสุทธิ (ควรบล็อก)")
print("    PnL   = กำไร/ขาดทุนรวมของช่วงนี้")
print("    สถานะ:")
print("      [BLOCK] = ควรบล็อก  (PF < 0.9 ขาดทุนสุทธิ)")
print("      [watch] = น่าจับตา  (sample น้อย หรือ PF ใกล้เกณฑ์ 0.9-1.15)")
print("      (ว่าง)  = ปกติ ไม่ต้องทำอะไร")
print()

# ── Guard ─────────────────────────────────────────────────────────────────────
data_sufficient = total >= min_total_eff
if not data_sufficient:
    need_more = min_total_eff - total
    print(f"  ⚠  ข้อมูลยังไม่พอ: มี {total} trades ต้องการอีก {need_more} ไม้")
    print(f"     สามารถดูผลวิเคราะห์ได้ แต่ยังไม่อัปเดต config.json")
    print(f"     รอให้บอทเทรดเพิ่ม แล้วค่อยรันใหม่\n")

# ── ตารางผลวิเคราะห์ ──────────────────────────────────────────────────────────
by_dow: dict[int, list[int]] = defaultdict(list)
DOW_TH = ("จันทร์", "อังคาร", "พุธ", "พฤหัส", "ศุกร์", "เสาร์", "อาทิตย์")

print(f"  {'วัน':<4} {'ชม':>2} | {'Tr':>3} | {'WR%':>5} | {'PF':>5} | {'PnL':>10} | สถานะ")
print(f"  {'-'*4} {'-'*2}-+-{'-'*3}-+-{'-'*5}-+-{'-'*5}-+-{'-'*10}-+-{'-'*20}")

for (dow, h), s in sorted(slot.items()):
    wr  = s["w"] / s["t"] * 100 if s["t"] else 0
    pf  = profit_factor(s["gw"], s["gl"])
    pfs = f"{pf:.2f}" if pf < 99 else " inf"
    blocked = should_block(s["t"], pf, wr)
    day_th  = DOW_TH[dow]

    if blocked:
        # PF < 0.9 = ขาดทุนสุทธิ — บล็อก
        wr_note = f" | WR={wr:.0f}% ชนะน้อยด้วย" if wr < 40 else f" | WR={wr:.0f}% แม้ชนะ%พอใช้"
        by_dow[dow].append(h)
        print(f"  {day_th:<4} {h:02d} | {s['t']:3d} | {wr:5.1f}% | {pfs:>5} | ${s['pnl']:+9.2f} | [BLOCK] PF={pf:.2f} ขาดทุนสุทธิ{wr_note}")
    elif s["t"] >= MIN_TRADES // 2 and pf < 1.15:
        # PF 0.9-1.15 = กำไรนิดหน่อย หรือ sample ยังน้อย — เฝ้าดู
        if s["t"] < MIN_TRADES:
            warn = f"sample น้อย ({s['t']}/{MIN_TRADES}) รอข้อมูลเพิ่ม"
        elif pf < 1.0:
            warn = f"PF={pf:.2f} เกือบขาดทุน ระวัง"
        else:
            warn = f"PF={pf:.2f} กำไรน้อย จับตาดู"
        print(f"  {day_th:<4} {h:02d} | {s['t']:3d} | {wr:5.1f}% | {pfs:>5} | ${s['pnl']:+9.2f} | [watch] {warn}")

# ช่องขาดทุนหนักแต่ sample ยังไม่ครบ — อธิบายทำไม heatmap ดูแดงแต่ยังไม่ block
early_loss = [
    ((d, h), s)
    for (d, h), s in slot.items()
    if s["t"] < MIN_TRADES and s["pnl"] < -40
]
if early_loss:
    print(f"\n  ช่องขาดทุนหนัก แต่ยังไม่ block (sample < {MIN_TRADES} ไม้ — รอข้อมูล):")
    for (dow, h), s in sorted(early_loss, key=lambda x: x[1]["pnl"])[:8]:
        pf = profit_factor(s["gw"], s["gl"])
        print(f"    {DOW_TH[dow]} {h:02d}:xx  {s['t']}/{MIN_TRADES}ไม้  PnL ${s['pnl']:+.2f}  PF={pf:.2f}")

new_slots = [{"dow": d, "hours": sorted(hs)} for d, hs in sorted(by_dow.items())]
total_blocked = sum(len(s["hours"]) for s in new_slots)

# ── เปรียบเทียบกับ config ปัจจุบัน (ก่อนสรุป) ─────────────────────────────────
cfg = json.load(open(CONFIG, encoding="utf-8"))
old_slots = cfg["session_weighting"].get("blocked_slots", [])
old_set = {(s["dow"], h) for s in old_slots for h in s["hours"]}
new_set = {(s["dow"], h) for s in new_slots for h in s["hours"]}
removed_candidates = old_set - new_set

# ปลดบล็อกเฉพาะช่องที่มี sample ≥ MIN_TRADES และ PF ≥ เกณฑ์
kept_no_sample = set()
removed = set()
for dow, h in sorted(removed_candidates):
    s = slot.get((dow, h), {"t": 0, "gw": 0.0, "gl": 0.0})
    pf = profit_factor(s["gw"], s["gl"])
    if should_unblock(s["t"], pf):
        removed.add((dow, h))
    else:
        kept_no_sample.add((dow, h))

final_by_dow: dict[int, list] = defaultdict(list)
for dow, h in sorted(new_set | kept_no_sample):
    final_by_dow[dow].append(h)
for dow, h in sorted(old_set - removed):
    if h not in final_by_dow[dow]:
        final_by_dow[dow].append(h)
for dow in final_by_dow:
    final_by_dow[dow] = sorted(set(final_by_dow[dow]))

final_slots = [{"dow": d, "hours": final_by_dow[d]} for d in sorted(final_by_dow)]
final_set = {(s["dow"], h) for s in final_slots for h in s["hours"]}
total_blocked = len(final_set)

# ── สรุปผล ────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print(f"  สรุปผลวิเคราะห์")
print(SEP)
print(f"  ช่วงเวลาที่ควรบล็อกทั้งหมด : {total_blocked} slots ใน {len(final_slots)} วัน")
for s in final_slots:
    hrs = ", ".join(f"{h:02d}:xx" for h in s["hours"])
    print(f"    {DOW_TH[s['dow']]} : {hrs}")
if kept_no_sample:
    print(f"\n  คง block เดิม (ยังไม่มี sample ≥{MIN_TRADES} ในยุดนี้ — ไม่ปลด):")
    for dow, h in sorted(kept_no_sample):
        s = slot.get((dow, h), {"t": 0})
        print(f"    {DOW_TH[dow]} {h:02d}:xx  ({s['t']} ไม้ในยุดที่วิเคราะห์)")

# ── diff vs config ─────────────────────────────────────────────────────────────
added = final_set - old_set
removed = old_set - final_set

print(f"\n  เปรียบเทียบกับ config.json ที่ใช้อยู่ตอนนี้:")
if added or removed:
    for dow, h in sorted(added):
        print(f"    [+] เพิ่มบล็อก  {DOW_TH[dow]} {h:02d}:xx  (ข้อมูลใหม่พบว่าช่วงนี้ขาดทุน)")
    for dow, h in sorted(removed):
        s = slot.get((dow, h), {"t": 0, "gw": 0.0, "gl": 0.0})
        pf = profit_factor(s["gw"], s["gl"])
        print(f"    [-] ปลดบล็อก   {DOW_TH[dow]} {h:02d}:xx  (PF={pf:.2f}, {s['t']} ไม้ — ดีขึ้นแล้ว)")
else:
    print(f"    ไม่มีการเปลี่ยนแปลง — config.json ที่ใช้อยู่ตรงกับข้อมูลล่าสุดแล้ว")

# ── guard + confirmation ──────────────────────────────────────────────────────
print(f"\n{SEP}")

if DRY_RUN:
    print("  [dry-run] ไม่อัปเดต config.json")
    sys.exit(0)

if not data_sufficient:
    print(f"  ยังไม่อัปเดต config.json เพราะข้อมูลไม่พอ ({total} < {min_total_eff})")
    print(f"  รอบอทเทรดเพิ่มอีก {min_total_eff - total} ไม้ แล้วรันใหม่")
    sys.exit(0)

if not (added or removed):
    print("  config.json เป็นปัจจุบันแล้ว ไม่ต้องอัปเดต")
    sys.exit(0)

if not AUTO_YES:
    print(f"  จะอัปเดต config.json ไหม?")
    print(f"    เพิ่มบล็อก {len(added)} ช่วงเวลา | ปลดบล็อก {len(removed)} ช่วงเวลา")
    print(f"    (backup จะถูกบันทึกเป็น config.json.bak ก่อนเสมอ)")
    ans = input("  พิมพ์  y  แล้วกด Enter เพื่อยืนยัน  (อื่นๆ = ยกเลิก) : ").strip().lower()
    if ans != "y":
        print("\n  ยกเลิก — config.json ไม่ถูกแก้ไข")
        sys.exit(0)

# Apply
shutil.copy(CONFIG, CONFIG + ".bak")
cfg["session_weighting"]["blocked_slots"] = final_slots
cfg["session_weighting"]["_comment_slots"] = (
    f"Updated by analyze_slots.py ({total} trades"
    + (f", snapshot>={SINCE_SNAPSHOT}" if SINCE_SNAPSHOT else "")
    + f", rule: PF<{PF_BLOCK})"
)
with open(CONFIG, "w", encoding="utf-8") as f:
    json.dump(cfg, f, ensure_ascii=False, indent=2)

print(f"\n  อัปเดต config.json สำเร็จ  (backup: config.json.bak)")
print(f"  บล็อกทั้งหมด {total_blocked} slots | เพิ่ม {len(added)} | ปลด {len(removed)}")
print(f"\n  *** รีสตาร์ทบอทเพื่อให้การเปลี่ยนแปลงมีผล ***")

