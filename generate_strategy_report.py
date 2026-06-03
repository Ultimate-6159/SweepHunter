"""
generate_strategy_report.py
===========================
รายงานเว็บ: บล็อก วัน×ชั่วโมง (broker) + Recovery lot multiply

Usage: python generate_strategy_report.py [--since-snapshot=N]
  --since-snapshot=N : ใช้เฉพาะไม้ config_snapshot_id >= N (default 14 = ยุด config ปัจจุบัน)

Output: data/reports/strategy_<timestamp>.html
"""
from __future__ import annotations

import json
import math
import sqlite3
import sys
import webbrowser
from collections import defaultdict
from datetime import datetime, timezone, timedelta

from core.report_broker import (
    HM_NOW_CSS,
    HM_NOW_LEGEND,
    broker_slot_from_closed_at,
    hm_cell_classes,
    now_slot,
    now_slot_label,
)
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "data/db/hyper_trades.sqlite"
CONFIG_PATH = ROOT / "config.json"
OUT_DIR = ROOT / "data/reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DOW_NAMES = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
MIN_TRADES_BLOCK = 8
PF_BLOCK = 0.9
MIN_TRADES_WATCH = 4
PF_WATCH = 1.15
DEFAULT_SINCE_SNAPSHOT = 14

SINCE_SNAPSHOT = DEFAULT_SINCE_SNAPSHOT
for _arg in sys.argv:
    if _arg.startswith("--since-snapshot="):
        SINCE_SNAPSHOT = int(_arg.split("=", 1)[1])
    elif _arg == "--all":
        SINCE_SNAPSHOT = 0


def load_config() -> dict:
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return {}


def is_blocked(dow: int, hour: int, cfg_sw: dict) -> tuple[bool, str]:
    for slot in cfg_sw.get("blocked_slots") or []:
        if int(slot.get("dow", -1)) == dow:
            hours = [int(h) for h in slot.get("hours") or []]
            if hour in hours:
                return True, "slot"
    for h in cfg_sw.get("blocked_hours_broker") or []:
        if hour == int(h):
            return True, "every day"
    return False, ""


def fmt_money(v: float) -> str:
    cls = "pos" if v >= 0 else "neg"
    return f"<span class='{cls}'>${v:+.2f}</span>"


def profit_factor(gw: float, gl: float) -> float:
    return gw / gl if gl > 0 else 999.0


def cell_class(pnl: float, n: int, pf: float, blocked: bool) -> str:
    parts = ["hm-cell"]
    if blocked:
        parts.append("blocked")
    if n == 0:
        parts.append("empty")
    elif n >= MIN_TRADES_BLOCK and pf < PF_BLOCK:
        parts.append("bad")
    elif n >= MIN_TRADES_WATCH and pf < PF_WATCH:
        parts.append("watch")
    elif pnl > 50 and n >= MIN_TRADES_WATCH:
        parts.append("good")
    elif pnl < 0:
        parts.append("weak")
    return " ".join(parts)


def sim_recovery_lot(
    base: float,
    debt: float,
    cum_vol: float,
    step: int,
    lot_mult: float,
    vol_mult: float,
    max_rec_mult: float,
    net_per_lot: float,
    balance_cap_lot: float,
) -> dict:
    geo = base * (lot_mult ** step)
    recovery_lot = (debt + 0.5) / net_per_lot if net_per_lot > 0 else base
    volume_floor = cum_vol * vol_mult
    rec_cap = base * max_rec_mult if max_rec_mult > 0 else 999.0
    raw = max(geo, recovery_lot, volume_floor)
    capped = min(raw, balance_cap_lot, rec_cap)
    which = "geo"
    if abs(raw - recovery_lot) < 1e-9 or recovery_lot >= geo and recovery_lot >= volume_floor:
        which = "recover"
    if volume_floor >= geo and volume_floor >= recovery_lot:
        which = "volume"
    if capped < raw - 1e-9:
        which += "+cap"
    return {
        "step": step,
        "geo": geo,
        "recover": recovery_lot,
        "volume": volume_floor,
        "raw": raw,
        "final": capped,
        "driver": which,
        "rec_cap": rec_cap,
    }


def main() -> None:
    if not DB_PATH.exists():
        print(f"[ERR] DB not found: {DB_PATH}")
        sys.exit(1)

    cfg = load_config()
    cfg_sw = cfg.get("session_weighting") or {}
    cfg_r = cfg.get("recovery") or {}
    cfg_as = cfg.get("account_scaling") or {}
    broker_off = int(cfg_sw.get("broker_offset_hours", 3))

    blocked_slots_cfg = cfg_sw.get("blocked_slots") or []
    blocked_hours_cfg = cfg_sw.get("blocked_hours_broker") or []
    lot_mult = float(cfg_r.get("lot_multiplier", 1.25))
    vol_mult = float(cfg_r.get("profit_volume_multiplier", 1.05))
    max_rec_mult = float(cfg_r.get("max_recovery_lot_multiplier", 2.0))
    risk_pct = float(cfg_as.get("risk_per_trade_pct", 1.5))
    max_lot_pct = float(cfg_as.get("max_lot_pct_of_balance", 25.0))

    con = sqlite3.connect(str(DB_PATH))
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    if SINCE_SNAPSHOT > 0:
        rows = cur.execute("""
            SELECT closed_at_utc, pnl, volume, role, step, series_id, status, atr
            FROM decisions
            WHERE status IN ('WIN','LOSS') AND closed_at_utc IS NOT NULL AND pnl IS NOT NULL
              AND config_snapshot_id >= ?
        """, (SINCE_SNAPSHOT,)).fetchall()
        era_label = f"config_snapshot_id ≥ {SINCE_SNAPSHOT} (ยุด config ปัจจุบัน)"
    else:
        rows = cur.execute("""
            SELECT closed_at_utc, pnl, volume, role, step, series_id, status, atr
            FROM decisions
            WHERE status IN ('WIN','LOSS') AND closed_at_utc IS NOT NULL AND pnl IS NOT NULL
        """).fetchall()
        era_label = "ทั้งหมด (รวมข้อมูลเก่า — ใช้ strategy_report.bat สำหรับยุดปัจจุบัน)"

    # ---------- slot stats (dow × broker hour) ----------
    slot: dict[tuple[int, int], dict] = defaultdict(
        lambda: {"t": 0, "pnl": 0.0, "w": 0, "gw": 0.0, "gl": 0.0}
    )
    hour_only: dict[int, dict] = defaultdict(lambda: {"t": 0, "pnl": 0.0, "w": 0})

    trade_dates: set = set()

    for r in rows:
        key = broker_slot_from_closed_at(str(r["closed_at_utc"]), broker_off)
        if key is None:
            continue
        dow, bh = key
        dt = datetime.fromisoformat(str(r["closed_at_utc"]).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        trade_dates.add(dt.date())
        pnl = float(r["pnl"])
        slot[(dow, bh)]["t"] += 1
        slot[(dow, bh)]["pnl"] += pnl
        hour_only[bh]["t"] += 1
        hour_only[bh]["pnl"] += pnl
        if pnl > 0:
            slot[(dow, bh)]["w"] += 1
            slot[(dow, bh)]["gw"] += pnl
            hour_only[bh]["w"] += 1
        else:
            slot[(dow, bh)]["gl"] += abs(pnl)

    date_range = ""
    if trade_dates:
        d0, d1 = min(trade_dates), max(trade_dates)
        date_range = f" · {d0} → {d1} ({len(trade_dates)} วัน)"

    total_trades = len(rows)
    total_pnl = sum(float(r["pnl"]) for r in rows)

    # ---------- recovery by step ----------
    step_stats: dict[int, dict] = defaultdict(
        lambda: {"t": 0, "pnl": 0.0, "w": 0, "vol_sum": 0.0, "vol_max": 0.0}
    )
    series_steps: dict[int, list] = defaultdict(list)
    primary_vol = []
    recovery_vol = []

    for r in rows:
        vol = float(r["volume"] or 0)
        st = int(r["step"] or 1)
        role = r["role"] or "PRIMARY"
        if role == "RECOVERY":
            recovery_vol.append(vol)
            step_stats[st]["t"] += 1
            step_stats[st]["pnl"] += float(r["pnl"])
            step_stats[st]["vol_sum"] += vol
            step_stats[st]["vol_max"] = max(step_stats[st]["vol_max"], vol)
            if float(r["pnl"]) > 0:
                step_stats[st]["w"] += 1
            if r["series_id"]:
                series_steps[int(r["series_id"])].append(
                    (st, vol, float(r["pnl"]), r["status"])
                )
        else:
            primary_vol.append(vol)

    # estimate net $ per lot at TP (from winning recovery trades)
    win_recovers = [
        float(r["pnl"]) / float(r["volume"])
        for r in rows
        if r["role"] == "RECOVERY" and r["status"] == "WIN"
        and float(r["volume"] or 0) > 0 and float(r["pnl"] or 0) > 0
    ]
    net_per_lot = (
        sum(win_recovers) / len(win_recovers) if win_recovers else 80.0
    )

    # balance proxy from latest — use 1000 if unknown
    balance = 1000.0
    try:
        import MetaTrader5 as mt5
        if mt5.initialize():
            acc = mt5.account_info()
            if acc and acc.balance > 0:
                balance = float(acc.balance)
            mt5.shutdown()
    except Exception:
        pass

    base_lot_est = max(0.01, balance * risk_pct / 100.0 / max(net_per_lot * 0.8, 1.0))
    balance_cap_lot = max(
        0.01,
        balance * max_lot_pct / 100.0 / max(net_per_lot * 0.5, 1.0),
    )

    con.close()

    # ---------- heatmap HTML ----------
    cur_dow, cur_h = now_slot(broker_off)
    now_lbl = now_slot_label(broker_off)
    hm_rows = ""
    for dow in range(7):
        row_cls = "hm-now-row" if dow == cur_dow else ""
        cells = f"<th class='dow'>{DOW_NAMES[dow]}</th>"
        for h in range(24):
            s = slot[(dow, h)]
            n, pnl = s["t"], s["pnl"]
            pf = profit_factor(s["gw"], s["gl"])
            wr = (s["w"] / n * 100) if n else 0
            blk, blk_type = is_blocked(dow, h, cfg_sw)
            cls = cell_class(pnl, n, pf, blk)
            cls += " " + hm_cell_classes(dow, h, cur_dow, cur_h)
            if blk:
                lbl = "ทุกวัน" if blk_type == "every day" else DOW_NAMES[dow]
                badge = f"<span class='badge' title='{blk_type}'>BLOCK<br><small>{lbl}</small></span>"
            else:
                badge = ""
            inner = (
                f"{badge}<div class='pnl'>{fmt_money(pnl) if n else ('—' if not blk else 'no trades')}</div>"
                f"<div class='meta'>{n}t · {wr:.0f}%</div>"
                if n or blk
                else "<div class='meta'>—</div>"
            )
            title = (
                f"{DOW_NAMES[dow]} {h:02d}:xx broker | {n} trades | WR {wr:.1f}% | "
                f"PF {pf:.2f} | ${pnl:+.2f}" + (f" | blocked ({blk_type})" if blk else "")
            )
            cells += f"<td class='{cls}' title='{title}'>{inner}</td>"
        hm_rows += f"<tr class='{row_cls}'>{cells}</tr>"

    # hour summary row
    hour_header = "<tr><th>Hour</th>" + "".join(
        f"<th class='{hm_cell_classes(None, h, cur_dow, cur_h, col_only=True)}'>{h:02d}</th>"
        for h in range(24)
    ) + "</tr>"

    # suggested blocks (same rule as analyze_slots.py)
    suggest_rows = ""
    watch_rows = ""
    for (dow, h), s in sorted(slot.items(), key=lambda x: (x[1]["pnl"], -x[1]["t"])):
        pf = profit_factor(s["gw"], s["gl"])
        wr = s["w"] / s["t"] * 100 if s["t"] else 0
        blk, _ = is_blocked(dow, h, cfg_sw)
        if s["t"] >= MIN_TRADES_BLOCK and pf < PF_BLOCK:
            status = "✓ บล็อกแล้ว" if blk else "→ ควรบล็อก"
            suggest_rows += f"""<tr>
            <td><b>{DOW_NAMES[dow]}</b></td><td>{h:02d}:xx</td>
            <td>{s['t']}</td><td>{wr:.1f}%</td><td>{pf:.2f}</td><td>{fmt_money(s['pnl'])}</td>
            <td>{status}</td></tr>"""
        elif s["t"] >= MIN_TRADES_WATCH and pf < PF_WATCH:
            watch_rows += f"""<tr>
            <td>{DOW_NAMES[dow]}</td><td>{h:02d}:xx</td>
            <td>{s['t']}/{MIN_TRADES_BLOCK}</td><td>{wr:.1f}%</td><td>{pf:.2f}</td>
            <td>{fmt_money(s['pnl'])}</td></tr>"""

    # config blocked list
    cfg_block_rows = ""
    for slot in blocked_slots_cfg:
        d = int(slot.get("dow", 0))
        hrs = ", ".join(f"{int(x):02d}" for x in sorted(slot.get("hours") or []))
        cfg_block_rows += f"<tr><td>{DOW_NAMES[d]}</td><td>{hrs}</td><td>เฉพาะวัน</td></tr>"
    for h in sorted(blocked_hours_cfg):
        cfg_block_rows += f"<tr><td>ทุกวัน</td><td>{int(h):02d}:xx</td><td>blocked_hours_broker</td></tr>"

    # recovery step table
    step_rows = ""
    max_step = max(step_stats.keys()) if step_stats else 1
    for st in range(1, max_step + 1):
        s = step_stats.get(st, {"t": 0, "pnl": 0.0, "w": 0, "vol_sum": 0.0, "vol_max": 0.0})
        if s["t"] == 0:
            continue
        wr = s["w"] / s["t"] * 100
        avg_vol = s["vol_sum"] / s["t"]
        step_rows += f"""<tr>
            <td><b>Step {st}</b></td><td>{s['t']}</td>
            <td>{avg_vol:.3f}</td><td>{s['vol_max']:.3f}</td>
            <td>{wr:.1f}%</td><td>{fmt_money(s['pnl'])}</td></tr>"""

    avg_primary = sum(primary_vol) / len(primary_vol) if primary_vol else 0.01
    avg_recovery = sum(recovery_vol) / len(recovery_vol) if recovery_vol else 0.0
    ratio = avg_recovery / avg_primary if avg_primary > 0 else 0

    # simulation: debt $80, cum vol 0.6, steps 1-5
    sim_debt = 80.40
    sim_cum_vol = 0.60
    sim_rows = ""
    for st in range(1, 6):
        for label, mult, mrec in [
            ("config ปัจจุบัน", lot_mult, max_rec_mult),
            ("conservative", 1.15, 1.5),
            ("เดิม (อันตราย)", 1.5, 99.0),
        ]:
            r = sim_recovery_lot(
                base_lot_est, sim_debt, sim_cum_vol, st,
                mult, vol_mult, mrec, net_per_lot, balance_cap_lot,
            )
            sim_rows += f"""<tr>
                <td>{label}</td><td>{st}</td>
                <td>{r['geo']:.3f}</td><td>{r['recover']:.3f}</td>
                <td>{r['volume']:.3f}</td>
                <td><b>{r['final']:.3f}</b></td><td>{r['driver']}</td>
                <td>{r['rec_cap']:.3f}</td></tr>"""

    # bar chart recovery volume by step (inline CSS bars)
    vol_bars = ""
    max_avg = max(
        (step_stats[st]["vol_sum"] / step_stats[st]["t"])
        for st in step_stats
        if step_stats[st]["t"] > 0
    ) if step_stats else 1
    for st in sorted(step_stats.keys()):
        s = step_stats[st]
        if s["t"] == 0:
            continue
        avg_v = s["vol_sum"] / s["t"]
        pct = min(100, avg_v / max_avg * 100) if max_avg > 0 else 0
        vol_bars += f"""<div class='vbar-row'>
            <span class='vbar-label'>Step {st}</span>
            <div class='vbar'><div style='width:{pct:.0f}%'></div></div>
            <span class='vbar-val'>{avg_v:.3f} lot (max {s['vol_max']:.3f})</span>
        </div>"""

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    generated = datetime.now().strftime("%Y%m%d_%H%M%S")
    net_color = "#10b981" if total_pnl >= 0 else "#ef4444"

    html = f"""<!DOCTYPE html>
<html lang="th">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SweepHunter Strategy — Block &amp; Lot Report</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #0f172a; color: #e2e8f0; padding: 16px; line-height: 1.45; }}
.wrap {{ max-width: 1400px; margin: 0 auto; }}
header {{ background: #1e293b; border: 1px solid #334155; padding: 24px; border-radius: 10px; margin-bottom: 16px; }}
header h1 {{ font-size: 22px; color: #f8fafc; margin-bottom: 6px; }}
header .sub {{ color: #94a3b8; font-size: 13px; }}
.kpi {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; margin-bottom: 16px; }}
.kpi .box {{ background: #1e293b; border: 1px solid #334155; padding: 14px; border-radius: 8px; }}
.kpi .lbl {{ font-size: 11px; color: #64748b; text-transform: uppercase; }}
.kpi .val {{ font-size: 22px; font-weight: 700; margin-top: 4px; }}
section {{ background: #1e293b; border: 1px solid #334155; padding: 20px; border-radius: 10px; margin-bottom: 16px; }}
section h2 {{ font-size: 16px; color: #f1f5f9; margin-bottom: 12px; padding-bottom: 8px; border-bottom: 1px solid #334155; }}
section h3 {{ font-size: 13px; color: #94a3b8; margin: 14px 0 8px; font-weight: 600; }}
.pos {{ color: #34d399; font-weight: 600; }}
.neg {{ color: #f87171; font-weight: 600; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
th {{ background: #0f172a; color: #94a3b8; padding: 8px; text-align: left; font-size: 11px; text-transform: uppercase; }}
td {{ padding: 8px; border-bottom: 1px solid #334155; }}
tr:hover td {{ background: #273549; }}
.config-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
@media (max-width: 900px) {{ .config-grid {{ grid-template-columns: 1fr; }} }}
.config-grid dl {{ display: grid; grid-template-columns: 160px 1fr; gap: 4px 12px; font-size: 13px; }}
.config-grid dt {{ color: #64748b; }}
.config-grid dd {{ color: #e2e8f0; }}
.hm-wrap {{ overflow-x: auto; }}
.hm {{ border-collapse: collapse; font-size: 10px; min-width: 1200px; }}
.hm th, .hm td {{ padding: 4px 2px; text-align: center; border: 1px solid #334155; }}
.hm th.dow {{ min-width: 42px; background: #0f172a; color: #cbd5e1; font-size: 11px; }}
.hm th:not(.dow) {{ color: #64748b; font-weight: 500; }}
.hm-cell {{ min-width: 52px; vertical-align: top; }}
.hm-cell.empty {{ background: #1e293b; color: #475569; }}
.hm-cell.good {{ background: #064e3b; }}
.hm-cell.bad {{ background: #7f1d1d; }}
.hm-cell.watch {{ background: #713f12; }}
.hm-cell.weak {{ background: #422006; }}
.hm-cell.blocked {{ outline: 2px solid #fbbf24; background: #1c1917 !important; }}
.hm-cell .pnl {{ font-size: 10px; font-weight: 700; }}
.hm-cell .meta {{ font-size: 9px; color: #94a3b8; }}
{HM_NOW_CSS}
.badge {{ display: block; font-size: 8px; background: #fbbf24; color: #1c1917; border-radius: 3px; margin-bottom: 2px; font-weight: 700; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 12px; font-size: 12px; color: #94a3b8; margin-bottom: 10px; }}
.legend span {{ display: inline-flex; align-items: center; gap: 6px; }}
.swatch {{ width: 14px; height: 14px; border-radius: 3px; display: inline-block; }}
.vbar-row {{ display: flex; align-items: center; gap: 10px; margin: 6px 0; font-size: 13px; }}
.vbar {{ flex: 1; max-width: 400px; height: 18px; background: #334155; border-radius: 4px; overflow: hidden; }}
.vbar > div {{ height: 100%; background: #3b82f6; }}
.vbar-label {{ width: 56px; color: #94a3b8; }}
.vbar-val {{ color: #cbd5e1; min-width: 160px; }}
.note {{ font-size: 12px; color: #64748b; margin-top: 8px; }}
footer {{ text-align: center; color: #475569; font-size: 11px; padding: 20px; }}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>SweepHunter — Block Slots &amp; Recovery Lot Report</h1>
    <div class="sub">Generated {ts} · Broker UTC+{broker_off} · {total_trades} closed trades · {era_label}{date_range}</div>
  </header>

  <div class="kpi">
    <div class="box"><div class="lbl">Net P/L (ยุดปัจจุบัน)</div>
      <div class="val" style="color:{net_color}">${total_pnl:+.2f}</div></div>
    <div class="box"><div class="lbl">Trades</div><div class="val">{total_trades}</div></div>
    <div class="box"><div class="lbl">Avg PRIMARY lot</div><div class="val">{avg_primary:.3f}</div></div>
    <div class="box"><div class="lbl">Avg RECOVERY lot</div><div class="val">{avg_recovery:.3f}</div></div>
    <div class="box"><div class="lbl">Recovery / Primary</div><div class="val">{ratio:.1f}×</div></div>
    <div class="box"><div class="lbl">Est. base lot</div><div class="val">{base_lot_est:.3f}</div></div>
  </div>

  <section>
    <h2>⚙️ Config ปัจจุบัน (config.json)</h2>
    <div class="config-grid">
      <div>
        <h3>Session / Block</h3>
        <dl>
          <dt>broker_offset_hours</dt><dd>{broker_off}</dd>
          <dt>blocked_slots</dt><dd>{len(blocked_slots_cfg)} วัน</dd>
          <dt>blocked_hours_broker</dt><dd>{blocked_hours_cfg}</dd>
        </dl>
      </div>
      <div>
        <h3>Recovery lot</h3>
        <dl>
          <dt>lot_multiplier</dt><dd>{lot_mult}</dd>
          <dt>profit_volume_multiplier</dt><dd>{vol_mult}</dd>
          <dt>max_recovery_lot_multiplier</dt><dd>{max_rec_mult}× base</dd>
          <dt>risk_per_trade_pct</dt><dd>{risk_pct}%</dd>
          <dt>max_lot_pct_of_balance</dt><dd>{max_lot_pct}% → cap ~{balance_cap_lot:.2f} lot</dd>
          <dt>global_equity_stop_pct</dt><dd>{cfg_r.get('global_equity_stop_pct', 0)}</dd>
        </dl>
      </div>
    </div>
    <h3>รายการบล็อกใน config</h3>
    <table><thead><tr><th>วัน</th><th>ชั่วโมง broker</th><th>ประเภท</th></tr></thead>
    <tbody>{cfg_block_rows or '<tr><td colspan=3>—</td></tr>'}</tbody></table>
  </section>

  <section>
    <h2>🗓️ Heatmap วัน × ชั่วโมง (broker)</h2>
    <div class="legend">
      <span><i class="swatch" style="background:#064e3b"></i> ดี (PnL&gt;$50, n≥{MIN_TRADES_WATCH})</span>
      <span><i class="swatch" style="background:#7f1d1d"></i> ควร block (n≥{MIN_TRADES_BLOCK}, PF&lt;{PF_BLOCK})</span>
      <span><i class="swatch" style="background:#713f12"></i> watch (n≥{MIN_TRADES_WATCH}, PF&lt;{PF_WATCH})</span>
      <span><i class="swatch" style="background:#422006"></i> ขาดทุน sample น้อย (ยังไม่ block)</span>
      <span><i class="swatch" style="outline:2px solid #fbbf24;background:#1c1917"></i> BLOCK ใน config</span>
      {HM_NOW_LEGEND}
    </div>
    <p class="note">เวลา = broker (UTC+{broker_off}). <b>ตอนนี้: {now_lbl}</b> · ข้อมูล = {era_label}{date_range}.
    สีแดง = เกณฑ์เดียวกับ <code>analyze_slots_current.bat</code> (≥{MIN_TRADES_BLOCK} ไม้ + PF&lt;{PF_BLOCK}).
    ช่องแดงใน heatmap เก่า (1–3 ไม้) ไม่ใช่เกณฑ์ block — รอ sample ก่อน.</p>
    <div class="hm-wrap">
      <table class="hm">
        {hour_header}
        {hm_rows}
      </table>
    </div>
  </section>

  <section>
    <h2>⚠️ ช่องที่ควรบล็อก (≥{MIN_TRADES_BLOCK} ไม้ + PF&lt;{PF_BLOCK})</h2>
    <table><thead><tr>
      <th>วัน</th><th>ชม.</th><th>Trades</th><th>WR%</th><th>PF</th><th>Net P/L</th><th>สถานะ config</th>
    </tr></thead><tbody>{suggest_rows or '<tr><td colspan=7>ยังไม่มีช่องครบเกณฑ์ block</td></tr>'}</tbody></table>
    <h3>เฝ้าดู (≥{MIN_TRADES_WATCH} ไม้, PF&lt;{PF_WATCH}, ยังไม่ครบ {MIN_TRADES_BLOCK} ไม้)</h3>
    <table><thead><tr>
      <th>วัน</th><th>ชม.</th><th>Trades</th><th>WR%</th><th>PF</th><th>Net P/L</th>
    </tr></thead><tbody>{watch_rows or '<tr><td colspan=6>—</td></tr>'}</tbody></table>
  </section>

  <section>
    <h2>📈 Recovery Lot — สถิติจาก DB</h2>
    <p class="note">net $/lot โดยประมาณ (จาก WIN recovery): <b>${net_per_lot:.1f}</b> · balance ใช้ประมาณ: ${balance:.0f}</p>
    <div class="vbar-row" style="margin-bottom:12px">{vol_bars or '<span class=note>ไม่มี RECOVERY trades</span>'}</div>
    <table><thead><tr>
      <th>Step</th><th>Trades</th><th>Avg lot</th><th>Max lot</th><th>WR%</th><th>Net P/L</th>
    </tr></thead><tbody>{step_rows or '<tr><td colspan=6>—</td></tr>'}</tbody></table>
  </section>

  <section>
    <h2>🧮 จำลอง Lot Recovery (หนี้ ${sim_debt:.2f}, cum vol {sim_cum_vol:.2f})</h2>
    <p class="note">เปรียบเทียบ config ปัจจุบัน vs conservative vs แบบเก่า (ไม่มี max_recovery cap)</p>
    <table><thead><tr>
      <th>โหมด</th><th>Step</th><th>Geo</th><th>Recover</th><th>Vol×</th>
      <th>Final lot</th><th>Driver</th><th>Rec cap</th>
    </tr></thead><tbody>{sim_rows}</tbody></table>
  </section>

  <footer>SweepHunter · generate_strategy_report.py · รันใหม่: strategy_report.bat</footer>
</div>
</body>
</html>"""

    out = OUT_DIR / f"strategy_{generated}.html"
    out.write_text(html, encoding="utf-8")
    print(f"[OK] Strategy report: {out}")
    print(f"     Era: {era_label}")
    print(f"     Trades: {total_trades} | Net: ${total_pnl:+.2f}")
    try:
        webbrowser.open(out.resolve().as_uri())
        print("[OK] Opening in browser...")
    except Exception as e:
        print(f"[WARN] Could not open browser: {e}")


if __name__ == "__main__":
    main()
