"""
วิเคราะห์ lot multiplier แยก วัน×ชั่วโมง (broker TZ) จาก DB
→ รายงาน HTML + ถามยืนยันก่อนอัปเดต config.json

Usage:
  python analyze_lot_multipliers.py [--dry-run] [--yes] [--since-snapshot=14]

  --dry-run           ดูรายงานอย่างเดียว ไม่ถามแก้ config
  --yes               อัปเดต config โดยไม่ถาม (automation)
  --since-snapshot=N  ใช้เฉพาะไม้ config_snapshot_id >= N (default 14)
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import sys
import webbrowser
from collections import defaultdict
from datetime import datetime, timedelta, timezone

DRY_RUN = "--dry-run" in sys.argv
AUTO_YES = "--yes" in sys.argv
SINCE_SNAPSHOT = 14
for _arg in sys.argv:
    if _arg.startswith("--since-snapshot="):
        SINCE_SNAPSHOT = int(_arg.split("=", 1)[1])

DB = "data/db/hyper_trades.sqlite"
CONFIG = "config.json"
REPORT_DIR = "data/reports"
BROKER_OFFSET = 3

MIN_TRADES_SLOT = 6       # sample ต่อช่อง (ทั้ง DB)
MIN_TRADES_SLOT_ERA = 4   # ยุด snapshot — ข้อมูลยังน้อย ใช้เกณฑ์ต่ำกว่า
MIN_TRADES_SLOT_PREVIEW = 3  # แสดงในรายงานอย่างเดียว (ยังไม่เขียน config)
MIN_TRADES_HOUR = 12      # sample รวมทุกวัน ต่อชั่วโมง
MIN_TRADES_HOUR_ERA = 8
MIN_TOTAL_TRADES = 300
MIN_TOTAL_ERA = 100

DOW = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
DOW_TH = ("จันทร์", "อังคาร", "พุธ", "พฤหัส", "ศุกร์", "เสาร์", "อาทิตย์")

MULT_LABELS = {
    2.0: "PRIME",
    1.5: "GOOD",
    1.2: "OK",
    1.0: "NEUTRAL",
    0.5: "WEAK",
}


def broker_slot(closed_at: str) -> tuple[int, int] | None:
    if not closed_at:
        return None
    s = str(closed_at).replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    bd = dt.astimezone(timezone.utc) + timedelta(hours=BROKER_OFFSET)
    return bd.weekday(), bd.hour


def profit_factor(gw: float, gl: float) -> float:
    return gw / gl if gl > 0 else 999.0


def _min_slot_apply() -> int:
    return MIN_TRADES_SLOT_ERA if SINCE_SNAPSHOT > 0 else MIN_TRADES_SLOT


def _min_hour_apply() -> int:
    return MIN_TRADES_HOUR_ERA if SINCE_SNAPSHOT > 0 else MIN_TRADES_HOUR


def suggest_mult(t: int, pf: float, pnl: float, *, for_config: bool = True) -> float | None:
    """
    None = ใช้ default 1.0 / fallback ชั่วโมง
    for_config=False → แสดง preview ในรายงาน (เกณฑ์ต่ำกว่า)
    """
    min_t = _min_slot_apply() if for_config else MIN_TRADES_SLOT_PREVIEW
    if t < min_t:
        return None
    # ชัดเจนว่าแย่
    if pf < 0.88 or pnl < -50:
        return 0.5
    # ชัดเจนว่าดี
    if pf >= 1.45 and pnl > 30:
        return 2.0
    if pf >= 1.22 and pnl > 15:
        return 1.5
    # sample น้อย (4–5 ไม้) — เขียน config เฉพาะสัญญาณแรง ไม่ใส่ 1.2 กลางๆ
    if for_config and t < 6:
        return None
    if pf >= 1.05:
        return 1.2
    if pf >= 0.95:
        return None  # กลางๆ ไม่ใส่ใน config
    return 0.5


def suggest_hour_mult(t: int, pf: float, pnl: float) -> float | None:
    if t < _min_hour_apply():
        return None
    return suggest_mult(t, pf, pnl, for_config=True)


def mult_color(m: float) -> str:
    if m >= 2.0:
        return "#14532d"
    if m >= 1.5:
        return "#166534"
    if m >= 1.2:
        return "#713f12"
    if m <= 0.5:
        return "#7f1d1d"
    return "#334155"


def load_trades() -> list[tuple[str, float]]:
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
    return rows


def build_stats(rows: list[tuple[str, float]]):
    slot: dict[tuple[int, int], dict] = defaultdict(
        lambda: {"t": 0, "pnl": 0.0, "w": 0, "gw": 0.0, "gl": 0.0}
    )
    hour_agg: dict[int, dict] = defaultdict(
        lambda: {"t": 0, "pnl": 0.0, "w": 0, "gw": 0.0, "gl": 0.0}
    )
    trade_dates: set = set()

    for closed_at, pnl in rows:
        pnl = float(pnl)
        key = broker_slot(closed_at)
        if key is None:
            continue
        dow, h = key
        s = slot[key]
        s["t"] += 1
        s["pnl"] += pnl
        if pnl > 0:
            s["w"] += 1
            s["gw"] += pnl
        else:
            s["gl"] += abs(pnl)

        ha = hour_agg[h]
        ha["t"] += 1
        ha["pnl"] += pnl
        if pnl > 0:
            ha["w"] += 1
            ha["gw"] += pnl
        else:
            ha["gl"] += abs(pnl)

        dt = datetime.fromisoformat(str(closed_at).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        trade_dates.add(dt.date())

    return slot, hour_agg, trade_dates


def propose_slot_multipliers(slot: dict) -> list[dict]:
    out = []
    for (dow, h), s in sorted(slot.items()):
        pf = profit_factor(s["gw"], s["gl"])
        m = suggest_mult(s["t"], pf, s["pnl"])
        if m is None or abs(m - 1.0) < 1e-6:
            continue
        wr = s["w"] / s["t"] * 100 if s["t"] else 0
        out.append({
            "dow": dow,
            "hour": h,
            "mult": m,
            "_note": f"{s['t']}t WR={wr:.0f}% PF={pf:.2f} PnL=${s['pnl']:+.0f}",
        })
    return out


def propose_hour_multipliers(hour_agg: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    for h, s in sorted(hour_agg.items()):
        pf = profit_factor(s["gw"], s["gl"])
        m = suggest_hour_mult(s["t"], pf, s["pnl"])
        if m is None or abs(m - 1.0) < 1e-6:
            continue
        out[str(h)] = m
    return out


def slot_mult_lookup(slots: list[dict]) -> dict[tuple[int, int], float]:
    m = {}
    for e in slots:
        m[(int(e["dow"]), int(e["hour"]))] = float(e["mult"])
    return m


def generate_html(
    slot: dict,
    hour_agg: dict,
    slot_props: list[dict],
    hour_props: dict[str, float],
    total: int,
    min_total: int,
    trade_dates: set,
    era_label: str,
    report_path: str,
) -> None:
    sm = slot_mult_lookup(slot_props)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    date_range = ""
    if trade_dates:
        date_range = f"{min(trade_dates)} → {max(trade_dates)}"

    rows_html = ""
    for dow in range(7):
        cells = f"<th class='dow'>{DOW[dow]}<br><small>{DOW_TH[dow]}</small></th>"
        for h in range(24):
            s = slot.get((dow, h), {"t": 0, "pnl": 0.0, "w": 0, "gw": 0.0, "gl": 0.0})
            pf = profit_factor(s["gw"], s["gl"])
            wr = (s["w"] / s["t"] * 100) if s["t"] else 0
            prop = sm.get((dow, h))
            if prop is not None:
                mult = prop
                src = "config"
            else:
                prev = suggest_mult(s["t"], pf, s["pnl"], for_config=False)
                if prev is not None:
                    mult = prev
                    src = "preview" if s["t"] < _min_slot_apply() else "แนะนำ"
                else:
                    mult = 1.0
                    src = "default"
            bg = mult_color(mult)
            lbl = MULT_LABELS.get(mult, str(mult))
            inner = (
                f"<div class='m'>{mult:.1f}x</div>"
                f"<div class='sub'>{lbl}</div>"
                if s["t"] or prop
                else "<div class='sub'>—</div>"
            )
            if s["t"]:
                inner += f"<div class='meta'>{s['t']}t ${s['pnl']:+.0f}</div>"
            title = (
                f"{DOW[dow]} {h:02d}:xx | {s['t']} trades | WR {wr:.0f}% | "
                f"PF {pf:.2f} | ${s['pnl']:+.2f} | mult {mult}x ({src})"
            )
            cells += f"<td style='background:{bg}' title='{title}'>{inner}</td>"
        rows_html += f"<tr>{cells}</tr>"

    hour_header = "<tr><th>Hour</th>" + "".join(
        f"<th>{h:02d}</th>" for h in range(24)
    ) + "</tr>"

    slot_table = ""
    for e in slot_props:
        dow, h = int(e["dow"]), int(e["hour"])
        slot_table += (
            f"<tr><td>{DOW[dow]} ({DOW_TH[dow]})</td><td>{h:02d}:xx</td>"
            f"<td><b>{e['mult']:.1f}x</b></td><td>{MULT_LABELS.get(e['mult'], '')}</td>"
            f"<td style='color:#94a3b8'>{e.get('_note', '')}</td></tr>"
        )

    hour_table = ""
    for h, m in sorted(hour_props.items(), key=lambda x: int(x[0])):
        s = hour_agg[int(h)]
        pf = profit_factor(s["gw"], s["gl"])
        hour_table += (
            f"<tr><td>{int(h):02d}:xx</td><td><b>{m:.1f}x</b></td>"
            f"<td>{s['t']}</td><td>{pf:.2f}</td><td>${s['pnl']:+.2f}</td></tr>"
        )

    sufficient = total >= min_total
    guard = (
        "<p class='ok'>ข้อมูลพอสำหรับอัปเดต config ได้</p>"
        if sufficient
        else f"<p class='warn'>ข้อมูลยังไม่พอ ({total} &lt; {min_total}) — ดูรายงานได้ แต่ไม่ควรอัปเดต config</p>"
    )

    html = f"""<!DOCTYPE html>
<html lang="th"><head><meta charset="UTF-8">
<title>Lot Multiplier Report — Era {SINCE_SNAPSHOT}+</title>
<style>
body{{font-family:'Segoe UI',sans-serif;background:#0f172a;color:#e2e8f0;margin:0;padding:16px}}
h1{{font-size:20px}} .sub{{color:#94a3b8;font-size:13px;margin-bottom:16px}}
table{{border-collapse:separate;border-spacing:2px;font-size:10px}}
th,td{{padding:4px;text-align:center}} th.dow{{background:#1e293b;min-width:52px}}
td .m{{font-weight:700;font-size:11px}} td .sub{{font-size:8px;opacity:.9}}
td .meta{{font-size:7px;color:#cbd5e1;margin-top:2px}}
.panel{{background:#1e293b;border-radius:10px;padding:14px;margin:16px 0;border:1px solid #334155}}
.panel h2{{font-size:14px;margin:0 0 10px}}
.tbl{{width:100%;border-collapse:collapse;font-size:12px}}
.tbl th,.tbl td{{padding:8px;border-bottom:1px solid #334155;text-align:left}}
.ok{{color:#4ade80}} .warn{{color:#fbbf24}}
.legend span{{display:inline-block;padding:4px 10px;border-radius:4px;margin:2px;font-size:11px}}
</style></head><body>
<h1>Lot Multiplier — วัน × ชั่วโมง (broker UTC+{BROKER_OFFSET})</h1>
<p class="sub">สร้าง {ts} · {era_label}{(' · ' + date_range) if date_range else ''} · {total} ไม้ปิด</p>
{guard}
<div class="legend">
  <span style="background:#14532d">2.0 PRIME</span>
  <span style="background:#166534">1.5 GOOD</span>
  <span style="background:#713f12">1.2 OK</span>
  <span style="background:#334155">1.0 NEUTRAL</span>
  <span style="background:#7f1d1d">0.5 WEAK</span>
</div>
<div class="panel"><h2>Heatmap ตัวคูณที่แนะนำ (ช่องเขียว=บูสต์ แดง=ลด lot)</h2>
<table>{hour_header}{rows_html}</table>
<p style="font-size:11px;color:#64748b;margin-top:8px">
  เกณฑ์เขียน config: ≥{_min_slot_apply()} ไม้/ช่อง · preview ≥{MIN_TRADES_SLOT_PREVIEW} · ชั่วโมงรวม ≥{_min_hour_apply()} ไม้
</p></div>
<div class="panel"><h2>slot_multipliers ที่จะเขียน config ({len(slot_props)} ช่อง)</h2>
<table class="tbl"><thead><tr><th>วัน</th><th>ชม.</th><th>คูณ</th><th>ระดับ</th><th>หลักฐาน</th></tr></thead>
<tbody>{slot_table or '<tr><td colspan=5>ไม่มี — ใช้ default 1.0</td></tr>'}</tbody></table></div>
<div class="panel"><h2>multipliers ชั่วโมงเดียว (fallback ทุกวัน, {len(hour_props)} ชม.)</h2>
<table class="tbl"><thead><tr><th>ชม.</th><th>คูณ</th><th>ไม้รวม</th><th>PF</th><th>PnL</th></tr></thead>
<tbody>{hour_table or '<tr><td colspan=5>ไม่เปลี่ยน — ใช้ค่าเดิมใน config</td></tr>'}</tbody></table></div>
<footer style="color:#64748b;font-size:11px;margin-top:24px">
  รันใหม่: analyze_lot_multipliers.bat · ดูอย่างเดียว: --dry-run
</footer></body></html>"""

    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)


def main() -> int:
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    rows = load_trades()
    total = len(rows)
    min_total = MIN_TOTAL_ERA if SINCE_SNAPSHOT > 0 else MIN_TOTAL_TRADES
    era_label = (
        f"snapshot ≥ {SINCE_SNAPSHOT}"
        if SINCE_SNAPSHOT > 0
        else "ทั้งหมด"
    )

    slot, hour_agg, trade_dates = build_stats(rows)
    slot_props = propose_slot_multipliers(slot)
    hour_props = propose_hour_multipliers(hour_agg)

    slots_filled = len(slot)
    slots_apply = sum(1 for v in slot.values() if v["t"] >= _min_slot_apply())
    slots_preview = sum(1 for v in slot.values() if MIN_TRADES_SLOT_PREVIEW <= v["t"] < _min_slot_apply())

    ts_file = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(REPORT_DIR, f"lot_multiplier_{ts_file}.html")
    generate_html(
        slot, hour_agg, slot_props, hour_props, total, min_total,
        trade_dates, era_label, report_path,
    )

    sep = "=" * 60
    print(sep)
    print("  SweepHunter — วิเคราะห์ Lot Multiplier (วัน × ชั่วโมง broker)")
    print(sep)
    print(f"  ข้อมูล        : {total} ไม้ (ต้องการ >= {min_total} ถึงจะแนะนำอัปเดต config)")
    print(f"  ยุด           : {era_label}")
    if trade_dates:
        print(f"  ช่วงวันที่    : {min(trade_dates)} → {max(trade_dates)}")
    print(f"  รายงาน HTML   : {os.path.abspath(report_path)}")
    print()
    print(f"  ทำไมตัวคูณน้อย?  {total} ไม้ ÷ สูงสุด 168 ช่อง(7วัน×24ชม.) ≈ {total/168:.1f} ไม้/ช่องเฉลี่ย")
    print(f"  ช่องที่มีไม้    : {slots_filled} ช่อง | ≥{_min_slot_apply()} ไม้พอเขียน config: {slots_apply} | preview {MIN_TRADES_SLOT_PREVIEW}–{_min_slot_apply()-1}: {slots_preview}")
    print(f"  (เกณฑ์เขียน config ตั้งใจสูงเพื่อไม่คูณมั่ว — รอ ~500+ ไม้จะได้ช่องเต็มขึ้น)")
    print()
    print(f"  slot_multipliers แนะนำ : {len(slot_props)} ช่อง (วัน×ชม. แยกกัน, ≥{_min_slot_apply()} ไม้)")
    print(f"  multipliers ชั่วโมง    : {len(hour_props)} ชม. (ทุกวันรวมกัน, ≥{_min_hour_apply()} ไม้)")
    print()
    print(f"  {'วัน':<6} {'ชม':>3} | {'คูณ':>5} | {'Tr':>3} | {'PF':>5} | {'PnL':>9} | หมายเหตุ")
    print(f"  {'-'*6} {'-'*3}-+-{'-'*5}-+-{'-'*3}-+-{'-'*5}-+-{'-'*9}-+-{'-'*12}")
    for e in slot_props:
        s = slot[(e["dow"], e["hour"])]
        pf = profit_factor(s["gw"], s["gl"])
        print(
            f"  {DOW[e['dow']]:<6} {e['hour']:02d} | {e['mult']:4.1f}x | {s['t']:3d} | "
            f"{pf:5.2f} | ${s['pnl']:+8.2f} | {MULT_LABELS.get(e['mult'], '')}"
        )
    if hour_props:
        print()
        print("  ชั่วโมง fallback (ทุกวัน):")
        for h, m in sorted(hour_props.items(), key=lambda x: int(x[0])):
            s = hour_agg[int(h)]
            pf = profit_factor(s["gw"], s["gl"])
            print(f"    {int(h):02d}:xx → {m:.1f}x  ({s['t']} ไม้  PF={pf:.2f}  ${s['pnl']:+.2f})")

    try:
        webbrowser.open(f"file:///{os.path.abspath(report_path).replace(chr(92), '/')}")
    except Exception:
        pass

    if DRY_RUN:
        print(f"\n  [dry-run] ไม่แก้ config.json")
        return 0

    if total < min_total:
        print(f"\n  ยังไม่อัปเดต config — ข้อมูลไม่พอ ({total} < {min_total})")
        print(f"  รอเพิ่มอีก {min_total - total} ไม้ แล้วรันใหม่")
        return 0

    cfg = json.load(open(CONFIG, encoding="utf-8"))
    hl = cfg.setdefault("hourly_lot_multiplier", {})
    old_slots = hl.get("slot_multipliers", [])
    old_hours = hl.get("multipliers", {})

    def norm_slots(sl):
        if isinstance(sl, dict):
            return {
                (int(k.split("_")[0]), int(k.split("_")[1])): round(float(v), 4)
                for k, v in sl.items()
            }
        out = {}
        for e in sl or []:
            if "dow" in e and "hour" in e:
                out[(int(e["dow"]), int(e["hour"]))] = round(float(e["mult"]), 4)
        return out

    def norm_hours_map(h: dict) -> dict[int, float]:
        out: dict[int, float] = {}
        for k, v in (h or {}).items():
            try:
                out[int(k)] = round(float(v), 4)
            except (TypeError, ValueError):
                continue
        return out

    old_s = norm_slots(old_slots)
    new_s = {
        (int(e["dow"]), int(e["hour"])): round(float(e["mult"]), 4)
        for e in slot_props
    }
    old_h = norm_hours_map(old_hours)
    # เทียบผลหลัง merge จริง (ไม่ใช่แค่ keys ใน hour_props — กัน false positive)
    merged_hours = dict(old_hours or {})
    merged_hours.update(hour_props or {})
    target_h = norm_hours_map(merged_hours)

    slot_changed = old_s != new_s
    hour_changed = target_h != old_h

    if not slot_changed and not hour_changed:
        print("\n  config.json ตรงกับผลวิเคราะห์แล้ว — ไม่มีการเปลี่ยนแปลง ไม่ถามอัปเดต")
        return 0

    print(f"\n{sep}")
    print("  เปลี่ยนแปลงที่จะเขียน config:")
    any_diff = False
    for k, v in sorted(new_s.items()):
        if old_s.get(k) != v:
            any_diff = True
            tag = "ใหม่" if k not in old_s else f"{old_s[k]:.1f}x→{v:.1f}x"
            print(f"    slot {DOW[k[0]]} {k[1]:02d}:xx  × {v:.1f}  ({tag})")
    for k in sorted(old_s.keys() - new_s.keys()):
        any_diff = True
        print(f"    slot {DOW[k[0]]} {k[1]:02d}:xx  ลบ (กลับ default/fallback)")
    for h, v in sorted(target_h.items()):
        if old_h.get(h) != v:
            any_diff = True
            print(f"    hour {h:02d}:xx (ทุกวัน)  × {v:.1f}")
    if not any_diff:
        print("\n  config.json ตรงกับผลวิเคราะห์แล้ว — ไม่มีการเปลี่ยนแปลง ไม่ถามอัปเดต")
        return 0

    if not AUTO_YES:
        print()
        print("  อัปเดต hourly_lot_multiplier ใน config.json ไหม?")
        print("  (backup → config.json.bak | รีสตาร์ทบอทหลังแก้)")
        ans = input("  พิมพ์  y  แล้ว Enter = ยืนยัน  (อื่นๆ = ยกเลิก) : ").strip().lower()
        if ans != "y":
            print("\n  ยกเลิก — config ไม่ถูกแก้")
            return 0

    shutil.copy(CONFIG, CONFIG + ".bak")
    hl["slot_multipliers"] = slot_props
    if hour_props:
        # merge: อัปเดตเฉพาะชั่วโมงที่วิเคราะห์ได้ คงค่าเดิมชั่วโมงอื่น
        merged = dict(old_hours or {})
        merged.update(hour_props)
        hl["multipliers"] = merged
    hl["_comment_data"] = (
        f"analyze_lot_multipliers.py {datetime.now():%Y-%m-%d %H:%M} "
        f"({total} trades, {era_label})"
    )
    with open(CONFIG, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    print(f"\n  อัปเดต config.json สำเร็จ (backup: config.json.bak)")
    print(f"  slot_multipliers: {len(slot_props)} ช่อง")
    print("  *** รีสตาร์ทบอทเพื่อให้ตัวคูณมีผล ***")
    return 0


if __name__ == "__main__":
    sys.exit(main())
