"""Analyze day x hour (broker TZ) — hard block + watch (soft lot).

Hard block (skip bar): trades >= 8 AND PF < 0.9 AND net PnL < -$150
Watch (lot x0.5): 4 <= trades < 8 AND (PF < 1.0 OR net PnL < -$50), not hard-blocked

Usage: python analyze_slots.py [--dry-run] [--yes] [--since-snapshot=N]
"""
from __future__ import annotations

import json
import shutil
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.report_broker import broker_offset_from_config, broker_slot_from_closed_at

DRY_RUN = "--dry-run" in sys.argv
AUTO_YES = "--yes" in sys.argv
SINCE_SNAPSHOT = 0
for _arg in sys.argv:
    if _arg.startswith("--since-snapshot="):
        SINCE_SNAPSHOT = int(_arg.split("=", 1)[1])

BROKER_OFFSET = broker_offset_from_config()
MIN_TRADES = 8
MIN_TRADES_WATCH = MIN_TRADES // 2
MIN_TOTAL_TRADES = 300
MIN_TOTAL_ERA = 100
PF_BLOCK = 0.9
PNL_BLOCK = -150.0
PF_WATCH = 1.0
PNL_WATCH = -50.0
WATCH_LOT_MULT = 0.5

DB = "data/db/hyper_trades.sqlite"
CONFIG = "config.json"
DOW = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
DOW_TH = ("จันทร์", "อังคาร", "พุธ", "พฤหัส", "ศุกร์", "เสาร์", "อาทิตย์")
SEP = "=" * 60


def profit_factor(gw: float, gl: float) -> float:
    return gw / gl if gl > 0 else 999.0


def should_hard_block(t: int, pf: float, pnl: float) -> bool:
    return t >= MIN_TRADES and pf < PF_BLOCK and pnl < PNL_BLOCK


def should_watch(t: int, pf: float, pnl: float) -> bool:
    if t < MIN_TRADES_WATCH or t >= MIN_TRADES:
        return False
    return pf < PF_WATCH or pnl < PNL_WATCH


def should_unblock(t: int, pf: float) -> bool:
    return t >= MIN_TRADES and pf >= PF_BLOCK


def slots_to_config(by_dow: dict[int, list[int]]) -> list[dict]:
    return [{"dow": d, "hours": sorted(hs)} for d, hs in sorted(by_dow.items()) if hs]


def slot_set_from_config(slots: list) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for s in slots or []:
        for h in s.get("hours") or []:
            out.add((int(s["dow"]), int(h)))
    return out


def merge_slot_config(
    new_set: set[tuple[int, int]],
    old_set: set[tuple[int, int]],
    slot: dict,
    *,
    allow_keep_stale: bool,
) -> tuple[list[dict], set[tuple[int, int]], set[tuple[int, int]]]:
    """คืน (final_slots, added, removed) — allow_keep_stale สำหรับ blocked เท่านั้น."""
    removed_candidates = old_set - new_set
    kept_stale: set[tuple[int, int]] = set()
    removed: set[tuple[int, int]] = set()

    for dow, h in sorted(removed_candidates):
        s = slot.get((dow, h), {"t": 0, "gw": 0.0, "gl": 0.0, "pnl": 0.0})
        pf = profit_factor(s["gw"], s["gl"])
        if s["t"] == 0:
            removed.add((dow, h))
        elif should_unblock(s["t"], pf):
            removed.add((dow, h))
        elif allow_keep_stale:
            kept_stale.add((dow, h))
        else:
            removed.add((dow, h))

    final_by_dow: dict[int, list[int]] = defaultdict(list)
    for dow, h in sorted(new_set | (kept_stale if allow_keep_stale else set())):
        final_by_dow[dow].append(h)
    for dow, h in sorted(old_set - removed):
        if h not in final_by_dow[dow]:
            final_by_dow[dow].append(h)
    for dow in final_by_dow:
        final_by_dow[dow] = sorted(set(final_by_dow[dow]))

    final = slots_to_config(final_by_dow)
    final_set = {(s["dow"], h) for s in final for h in s["hours"]}
    added = final_set - old_set
    removed = old_set - final_set
    return final, added, removed


def load_slot_stats() -> tuple[dict, int, set, dict]:
    conn = sqlite3.connect(DB)
    if SINCE_SNAPSHOT > 0:
        rows = conn.execute(
            """
            SELECT closed_at_utc, pnl FROM decisions
            WHERE status IN ('WIN','LOSS') AND closed_at_utc IS NOT NULL AND pnl IS NOT NULL
              AND config_snapshot_id >= ?
            """,
            (SINCE_SNAPSHOT,),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT closed_at_utc, pnl FROM decisions
            WHERE status IN ('WIN','LOSS') AND closed_at_utc IS NOT NULL AND pnl IS NOT NULL
            """
        ).fetchall()
    conn.close()

    slot = defaultdict(lambda: {"t": 0, "pnl": 0.0, "w": 0, "gw": 0.0, "gl": 0.0})
    trade_dates: set = set()
    recent_by_date: dict = defaultdict(lambda: {"t": 0, "pnl": 0.0})

    for closed_at, pnl in rows:
        key = broker_slot_from_closed_at(str(closed_at), BROKER_OFFSET)
        if key is None:
            continue
        dow, h = key
        s = slot[(dow, h)]
        pnl_f = float(pnl)
        s["t"] += 1
        s["pnl"] += pnl_f
        if pnl_f > 0:
            s["w"] += 1
            s["gw"] += pnl_f
        else:
            s["gl"] += abs(pnl_f)

        dt = datetime.fromisoformat(str(closed_at).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        trade_dates.add(dt.date())
        recent_by_date[dt.date()]["t"] += 1
        recent_by_date[dt.date()]["pnl"] += pnl_f

    return slot, len(rows), trade_dates, recent_by_date


def main() -> int:
    slot, total, trade_dates, recent_by_date = load_slot_stats()
    min_total_eff = MIN_TOTAL_ERA if SINCE_SNAPSHOT > 0 else MIN_TOTAL_TRADES
    data_sufficient = total >= min_total_eff

    hard_by_dow: dict[int, list[int]] = defaultdict(list)
    watch_by_dow: dict[int, list[int]] = defaultdict(list)

    print(SEP)
    print("  SweepHunter — วิเคราะห์ช่วงเวลา (broker TZ): hard block + watch")
    print(SEP)
    print(f"  ข้อมูลที่ใช้   : {total} trades (ต้องการ >= {min_total_eff} ถึงจะ update ได้)")
    if SINCE_SNAPSHOT > 0:
        print(f"  กรองยุด      : config_snapshot_id >= {SINCE_SNAPSHOT}")
    print(f"  broker offset : UTC+{BROKER_OFFSET}")
    if trade_dates:
        d0, d1 = min(trade_dates), max(trade_dates)
        print(f"  ช่วงวันที่   : {d0} → {d1} ({len(trade_dates)} วันทำการ)")
    print()
    print(f"  HARD block    : n>={MIN_TRADES}, PF<{PF_BLOCK}, PnL<{PNL_BLOCK:.0f} → ข้ามแท่ง")
    print(f"  WATCH soft    : {MIN_TRADES_WATCH}<=n<{MIN_TRADES}, PF<{PF_WATCH} หรือ PnL<{PNL_WATCH:.0f}")
    print(f"                  → lot x{WATCH_LOT_MULT} (ไม่ข้ามแท่ง)")
    print()

    print(f"  {'วัน':<4} {'ชม':>2} | {'Tr':>3} | {'WR%':>5} | {'PF':>5} | {'PnL':>10} | สถานะ")
    print(f"  {'-'*4} {'-'*2}-+-{'-'*3}-+-{'-'*5}-+-{'-'*5}-+-{'-'*10}-+-{'-'*22}")

    for (dow, h), s in sorted(slot.items()):
        wr = s["w"] / s["t"] * 100 if s["t"] else 0
        pf = profit_factor(s["gw"], s["gl"])
        pfs = f"{pf:.2f}" if pf < 99 else " inf"
        pnl = s["pnl"]
        day_th = DOW_TH[dow]

        if should_hard_block(s["t"], pf, pnl):
            hard_by_dow[dow].append(h)
            print(
                f"  {day_th:<4} {h:02d} | {s['t']:3d} | {wr:5.1f}% | {pfs:>5} | "
                f"${pnl:+9.2f} | [BLOCK] hard"
            )
        elif should_watch(s["t"], pf, pnl):
            watch_by_dow[dow].append(h)
            print(
                f"  {day_th:<4} {h:02d} | {s['t']:3d} | {wr:5.1f}% | {pfs:>5} | "
                f"${pnl:+9.2f} | [WATCH] lot x{WATCH_LOT_MULT}"
            )
        elif s["t"] >= MIN_TRADES_WATCH and pf < 1.15:
            warn = f"sample {s['t']}/{MIN_TRADES}" if s["t"] < MIN_TRADES else f"PF={pf:.2f}"
            print(
                f"  {day_th:<4} {h:02d} | {s['t']:3d} | {wr:5.1f}% | {pfs:>5} | "
                f"${pnl:+9.2f} | [hint] {warn}"
            )

    new_hard = slots_to_config(hard_by_dow)
    new_watch = slots_to_config(watch_by_dow)
    new_hard_set = slot_set_from_config(new_hard)
    new_watch_set = slot_set_from_config(new_watch)
    # watch ไม่รวมช่องที่ hard block
    new_watch_set -= new_hard_set
    watch_by_dow2: dict[int, list[int]] = defaultdict(list)
    for dow, h in sorted(new_watch_set):
        watch_by_dow2[dow].append(h)
    new_watch = slots_to_config(watch_by_dow2)

    cfg = json.load(open(CONFIG, encoding="utf-8"))
    sw = cfg.setdefault("session_weighting", {})
    old_hard = sw.get("blocked_slots", [])
    old_watch = sw.get("watch_slots", [])
    old_hard_set = slot_set_from_config(old_hard)
    old_watch_set = slot_set_from_config(old_watch)

    final_hard, added_h, removed_h = merge_slot_config(
        new_hard_set, old_hard_set, slot, allow_keep_stale=True,
    )
    final_watch, added_w, removed_w = merge_slot_config(
        new_watch_set, old_watch_set, slot, allow_keep_stale=False,
    )
    # ลบ watch ที่ซ้ำกับ hard
    final_watch_set = slot_set_from_config(final_watch) - slot_set_from_config(final_hard)
    watch_by_final: dict[int, list[int]] = defaultdict(list)
    for dow, h in sorted(final_watch_set):
        watch_by_final[dow].append(h)
    final_watch = slots_to_config(watch_by_final)
    added_w = final_watch_set - old_watch_set
    removed_w = old_watch_set - final_watch_set

    print(f"\n{SEP}")
    print("  สรุป")
    print(SEP)
    print(f"  HARD block: {len(slot_set_from_config(final_hard))} slots")
    for s in final_hard:
        hrs = ", ".join(f"{h:02d}:xx" for h in s["hours"])
        print(f"    {DOW_TH[s['dow']]} : {hrs}")
    print(f"  WATCH: {len(slot_set_from_config(final_watch))} slots (lot x{WATCH_LOT_MULT})")
    for s in final_watch:
        hrs = ", ".join(f"{h:02d}:xx" for h in s["hours"])
        print(f"    {DOW_TH[s['dow']]} : {hrs}")

    print(f"\n  เปรียบเทียบ config:")
    for dow, h in sorted(added_h):
        print(f"    [+] HARD  {DOW_TH[dow]} {h:02d}:xx")
    for dow, h in sorted(removed_h):
        print(f"    [-] HARD  {DOW_TH[dow]} {h:02d}:xx")
    for dow, h in sorted(added_w):
        print(f"    [+] WATCH {DOW_TH[dow]} {h:02d}:xx")
    for dow, h in sorted(removed_w):
        print(f"    [-] WATCH {DOW_TH[dow]} {h:02d}:xx")

    if DRY_RUN:
        print("\n  [dry-run] ไม่อัปเดต config.json")
        return 0

    if not data_sufficient:
        print(f"\n  ยังไม่อัปเดต — ข้อมูลไม่พอ ({total} < {min_total_eff})")
        return 0

    if not (added_h or removed_h or added_w or removed_w):
        print("\n  config.json เป็นปัจจุบันแล้ว")
        return 0

    if not AUTO_YES:
        print(f"\n  จะอัปเดต config.json? (HARD +{len(added_h)}/-{len(removed_h)} WATCH +{len(added_w)}/-{len(removed_w)})")
        ans = input("  พิมพ์ y แล้ว Enter: ").strip().lower()
        if ans != "y":
            print("  ยกเลิก")
            return 0

    shutil.copy(CONFIG, CONFIG + ".bak")
    sw["blocked_slots"] = final_hard
    sw["watch_slots"] = final_watch
    sw["watch_lot_multiplier"] = WATCH_LOT_MULT
    sw["_comment_rule"] = (
        f"hard: n>={MIN_TRADES} PF<{PF_BLOCK} PnL<{PNL_BLOCK:.0f} skip bar | "
        f"watch: {MIN_TRADES_WATCH}<=n<{MIN_TRADES} PF<{PF_WATCH} or PnL<{PNL_WATCH:.0f} lot x{WATCH_LOT_MULT}"
    )
    sw["_comment_slots"] = (
        f"Updated analyze_slots.py ({total} trades"
        + (f", snap>={SINCE_SNAPSHOT}" if SINCE_SNAPSHOT else "")
        + ")"
    )
    with open(CONFIG, encoding="utf-8", mode="w") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    print("\n  อัปเดต config.json สำเร็จ (backup: config.json.bak)")
    print("  *** รีสตาร์ทบอท ***")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
