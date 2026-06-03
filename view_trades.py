"""
SweepHunter Trade Dashboard
Reads hyper_trades.sqlite -> generates trades_dashboard.html -> opens browser

Usage: python view_trades.py [--since-snapshot=N] [--all]
  default: --since-snapshot=14 (ยุด config ปัจจุบัน — ตรง analyze_slots_current)
"""
import json
import os
import sqlite3
import sys
import webbrowser
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from core.report_broker import (
    HM_NOW_CSS,
    HM_NOW_LEGEND,
    broker_slot_from_closed_at,
    now_slot_label,
)

DB = "data/db/hyper_trades.sqlite"
OUT = "data/trades_dashboard.html"
BROKER_OFFSET = 3
DOW = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
DOW_TH = ("จันทร์", "อังคาร", "พุธ", "พฤหัส", "ศุกร์", "เสาร์", "อาทิตย์")
DOW_TH_SHORT = ("จ", "อ", "พ", "พฤ", "ศ", "ส", "อา")
DEFAULT_SINCE_SNAPSHOT = 14


def broker_slot(closed_at: str) -> tuple[int, int] | None:
    return broker_slot_from_closed_at(closed_at, BROKER_OFFSET)


SINCE_SNAPSHOT = DEFAULT_SINCE_SNAPSHOT
for _arg in sys.argv:
    if _arg.startswith("--since-snapshot="):
        SINCE_SNAPSHOT = int(_arg.split("=", 1)[1])
    elif _arg == "--all":
        SINCE_SNAPSHOT = 0

# ── load recovery state ────────────────────────────────────────────────────────
def load_recovery():
    try:
        with open("data/recovery_state.json", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def load_blocked_slots():
    try:
        with open("config.json", encoding="utf-8") as f:
            cfg = json.load(f)
        return cfg.get("session_weighting", {}).get("blocked_slots", [])
    except Exception:
        return []

# ── query DB ──────────────────────────────────────────────────────────────────
conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row

if SINCE_SNAPSHOT > 0:
    trades_raw = conn.execute("""
        SELECT d.id, d.ts_utc, d.series_id, d.step, d.symbol,
               d.confidence, d.atr, d.spread_points, d.volume, d.ticket,
               d.entry_price, d.sl, d.tp, d.close_price,
               d.status, d.pnl, d.closed_at_utc, d.notes,
               CASE d.prediction WHEN 2 THEN 'BUY' WHEN 0 THEN 'SELL' ELSE 'HOLD' END side,
               cs.risk_per_trade_pct, cs.sl_atr_mult
        FROM decisions d
        LEFT JOIN config_snapshots cs ON cs.id = d.config_snapshot_id
        WHERE d.status IN ('WIN','LOSS')
          AND d.config_snapshot_id >= ?
        ORDER BY d.ts_utc
    """, (SINCE_SNAPSHOT,)).fetchall()
    era_label = f"snapshot ≥ {SINCE_SNAPSHOT} (ยุด config ปัจจุบัน)"
else:
    trades_raw = conn.execute("""
        SELECT d.id, d.ts_utc, d.series_id, d.step, d.symbol,
               d.confidence, d.atr, d.spread_points, d.volume, d.ticket,
               d.entry_price, d.sl, d.tp, d.close_price,
               d.status, d.pnl, d.closed_at_utc, d.notes,
               CASE d.prediction WHEN 2 THEN 'BUY' WHEN 0 THEN 'SELL' ELSE 'HOLD' END side,
               cs.risk_per_trade_pct, cs.sl_atr_mult
        FROM decisions d
        LEFT JOIN config_snapshots cs ON cs.id = d.config_snapshot_id
        WHERE d.status IN ('WIN','LOSS')
        ORDER BY d.ts_utc
    """).fetchall()
    era_label = "ทั้งหมด (รวมข้อมูลเก่า)"

series_raw = conn.execute("""
    SELECT id, opened_at_utc, closed_at_utc, symbol, side,
           steps, total_volume, avg_entry_price, final_pnl, status, notes
    FROM series ORDER BY opened_at_utc DESC LIMIT 200
""").fetchall()

retrains_raw = conn.execute("""
    SELECT ts_utc, rows_trained, cv_acc, oos_acc, accepted, notes
    FROM model_retrains ORDER BY ts_utc DESC LIMIT 50
""").fetchall()

if SINCE_SNAPSHOT > 0:
    open_raw = conn.execute("""
        SELECT d.id, d.ts_utc, d.series_id, d.step, d.symbol,
               d.confidence, d.volume, d.ticket, d.entry_price, d.sl, d.tp,
               d.status, d.notes,
               CASE d.prediction WHEN 2 THEN 'BUY' WHEN 0 THEN 'SELL' ELSE 'HOLD' END side
        FROM decisions d
        WHERE d.status IN ('OPEN','PENDING')
          AND d.config_snapshot_id >= ?
        ORDER BY d.ts_utc DESC
    """, (SINCE_SNAPSHOT,)).fetchall()
else:
    open_raw = conn.execute("""
        SELECT d.id, d.ts_utc, d.series_id, d.step, d.symbol,
               d.confidence, d.volume, d.ticket, d.entry_price, d.sl, d.tp,
               d.status, d.notes,
               CASE d.prediction WHEN 2 THEN 'BUY' WHEN 0 THEN 'SELL' ELSE 'HOLD' END side
        FROM decisions d
        WHERE d.status IN ('OPEN','PENDING')
        ORDER BY d.ts_utc DESC
    """).fetchall()

conn.close()

gen_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

rec = load_recovery()
blocked_slots = load_blocked_slots()

# ── build datasets ─────────────────────────────────────────────────────────────
trades = []
equity_curve = []
cum_pnl = 0.0
total_win = total_loss = 0
buy_win = buy_loss = sell_win = sell_loss = 0
heatmap = defaultdict(lambda: {"t": 0, "w": 0, "pnl": 0.0, "gw": 0.0, "gl": 0.0})

for r in trades_raw:
    pnl = float(r["pnl"] or 0)
    cum_pnl += pnl
    ts = r["ts_utc"] or ""
    closed = r["closed_at_utc"] or ""
    side = r["side"]
    won = r["status"] == "WIN"

    # heatmap (broker TZ — วัน+ชั่วโมง broker ตรงกับบอท)
    try:
        slot = broker_slot(closed)
        if slot is not None:
            key = slot
            s = heatmap[key]
            s["t"] += 1
            s["pnl"] += pnl
            if won:
                s["w"] += 1
                s["gw"] += pnl
            else:
                s["gl"] += abs(pnl)
    except Exception:
        pass

    if won:
        total_win += 1
        if side == "BUY":
            buy_win += 1
        else:
            sell_win += 1
    else:
        total_loss += 1
        if side == "BUY":
            buy_loss += 1
        else:
            sell_loss += 1

    equity_curve.append({"t": ts[:16].replace("T", " "), "pnl": round(cum_pnl, 2)})
    trades.append({
        "id": r["id"],
        "ts": ts[:16].replace("T", " "),
        "closed": closed[:16].replace("T", " ") if closed else "",
        "closed_iso": closed or "",
        "side": side,
        "vol": float(r["volume"] or 0),
        "conf": round(float(r["confidence"] or 0) * 100, 1),
        "atr": round(float(r["atr"] or 0), 2),
        "sprd": float(r["spread_points"] or 0),
        "entry": float(r["entry_price"] or 0),
        "close": float(r["close_price"] or 0),
        "sl": float(r["sl"] or 0),
        "tp": float(r["tp"] or 0),
        "status": r["status"],
        "pnl": round(pnl, 2),
        "step": r["step"] or 1,
        "series": r["series_id"] or 0,
        "risk": round(float(r["risk_per_trade_pct"] or 0), 2),
    })

# heatmap json
hm_data = {}
for (dow, h), s in heatmap.items():
    wr = s["w"] / s["t"] * 100 if s["t"] else 0
    pf = (s["gw"] / s["gl"]) if s["gl"] > 0 else 999
    hm_data[f"{dow}_{h}"] = {
        "t": s["t"], "wr": round(wr, 1), "pnl": round(s["pnl"], 2),
        "pf": round(pf, 2)
    }

# series
series_list = []
for r in series_raw:
    series_list.append({
        "id": r["id"],
        "opened": (r["opened_at_utc"] or "")[:16].replace("T", " "),
        "closed": (r["closed_at_utc"] or "")[:16].replace("T", " "),
        "side": r["side"] or "",
        "steps": r["steps"] or 0,
        "pnl": round(float(r["final_pnl"] or 0), 2),
        "status": r["status"] or "",
    })

# open / pending (live positions — not in closed-trade stats)
open_list = []
for r in open_raw:
    open_list.append({
        "id": r["id"],
        "ts": (r["ts_utc"] or "")[:16].replace("T", " "),
        "side": r["side"] or "HOLD",
        "vol": float(r["volume"] or 0),
        "conf": round(float(r["confidence"] or 0) * 100, 1),
        "entry": float(r["entry_price"] or 0),
        "sl": float(r["sl"] or 0),
        "tp": float(r["tp"] or 0),
        "status": r["status"] or "OPEN",
        "step": r["step"] or 1,
        "series": r["series_id"] or 0,
        "ticket": r["ticket"] or "",
    })

# retrains
retrain_list = []
for r in retrains_raw:
    retrain_list.append({
        "ts": (r["ts_utc"] or "")[:16].replace("T", " "),
        "rows": r["rows_trained"] or 0,
        "cv": round(float(r["cv_acc"] or 0) * 100, 1),
        "oos": round(float(r["oos_acc"] or 0) * 100, 1),
        "ok": bool(r["accepted"]),
        "notes": (r["notes"] or "")[:80],
    })

# summary stats
n = len(trades)
wr_pct = total_win / n * 100 if n else 0
net_pnl = round(cum_pnl, 2)
avg_win  = (sum(t["pnl"] for t in trades if t["pnl"] > 0) / max(total_win, 1))
avg_loss = (sum(abs(t["pnl"]) for t in trades if t["pnl"] < 0) / max(total_loss, 1))
rr = avg_win / avg_loss if avg_loss else 0
gross_w  = sum(t["pnl"] for t in trades if t["pnl"] > 0)
gross_l  = sum(abs(t["pnl"]) for t in trades if t["pnl"] < 0)
pf_total = gross_w / gross_l if gross_l else 999

global_debt = float(rec.get("global_debt_usd", 0))
cum_loss_active = float(rec.get("cumulative_loss_usd", 0))
consec = int(rec.get("consecutive_losses", 0))

trade_dates = set()
for t in trades:
    if t["closed"]:
        try:
            trade_dates.add(t["closed"][:10])
        except Exception:
            pass
date_range = ""
if trade_dates:
    date_range = f" · {min(trade_dates)} → {max(trade_dates)}"

now_broker = datetime.now(timezone.utc) + timedelta(hours=BROKER_OFFSET)
now_slot = {"dow": now_broker.weekday(), "hour": now_broker.hour}
now_lbl = now_slot_label(BROKER_OFFSET)

recent_slot_trades = []
for t in reversed(trades[-15:]):
    slot = broker_slot(t.get("closed_iso", ""))
    if slot is None:
        continue
    dow, bh = slot
    recent_slot_trades.append({
        "id": t["id"],
        "pnl": t["pnl"],
        "closed": t["closed"],
        "slot": f"{DOW[dow]} {bh:02d}:xx",
        "dow": dow,
        "hour": bh,
    })

# ── HTML ───────────────────────────────────────────────────────────────────────
def pnl_color(v):
    if v > 0: return "#22c55e"
    if v < 0: return "#ef4444"
    return "#94a3b8"

HTML = f"""<!DOCTYPE html>
<html lang="th">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SweepHunter — แดชบอร์ดเทรด</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#0f172a;color:#e2e8f0;font-family:'Segoe UI','Leelawadee UI',sans-serif;font-size:13px;line-height:1.5}}
h1{{background:linear-gradient(135deg,#6366f1,#8b5cf6);padding:14px 20px;font-size:18px;letter-spacing:.5px}}
h1 span{{font-size:12px;opacity:.85;margin-left:12px;font-weight:normal}}
.tabs{{display:flex;background:#1e293b;border-bottom:2px solid #334155;flex-wrap:wrap}}
.tab{{padding:10px 18px;cursor:pointer;border-bottom:2px solid transparent;margin-bottom:-2px;transition:.2s;font-size:13px}}
.tab:hover{{background:#334155}}
.tab.active{{border-bottom-color:#6366f1;color:#818cf8;font-weight:600}}
.page{{display:none;padding:16px;max-width:1400px;margin:0 auto}}
.page.active{{display:block}}
.cards{{display:flex;flex-wrap:wrap;gap:12px;margin-bottom:16px}}
.card{{background:#1e293b;border-radius:10px;padding:14px 18px;min-width:150px;flex:1;border:1px solid #334155}}
.card-label{{font-size:12px;color:#94a3b8;font-weight:600}}
.card-value{{font-size:22px;font-weight:700;margin-top:4px}}
.card-sub{{font-size:11px;color:#64748b;margin-top:4px;line-height:1.4}}
.green{{color:#22c55e}}.red{{color:#ef4444}}.yellow{{color:#f59e0b}}.blue{{color:#60a5fa}}.purple{{color:#a78bfa}}
.panel{{background:#1e293b;border-radius:10px;padding:14px;margin-bottom:14px;border:1px solid #334155}}
.panel h2{{font-size:14px;font-weight:600;color:#e2e8f0;margin-bottom:8px}}
.panel .desc{{font-size:12px;color:#64748b;margin-bottom:12px;line-height:1.55}}
.help-box{{background:#0f172a;border:1px solid #334155;border-left:3px solid #6366f1;border-radius:8px;padding:12px 14px;margin-bottom:14px;font-size:12px;color:#94a3b8;line-height:1.65}}
.help-box b{{color:#cbd5e1}}
.help-box ul{{margin:6px 0 0 18px}}
table{{width:100%;border-collapse:collapse;font-size:12px}}
th{{background:#0f172a;padding:7px 8px;text-align:left;color:#64748b;font-weight:600;white-space:nowrap;cursor:pointer;position:sticky;top:0}}
th:hover{{color:#e2e8f0}}
td{{padding:6px 8px;border-bottom:1px solid #334155}}
tr:hover td{{background:#273549}}
.badge{{display:inline-block;padding:2px 7px;border-radius:4px;font-size:11px;font-weight:600}}
.badge-win{{background:#14532d;color:#4ade80}}
.badge-loss{{background:#450a0a;color:#f87171}}
.badge-buy{{background:#1e3a5f;color:#60a5fa}}
.badge-sell{{background:#3b1a40;color:#c084fc}}
.badge-hold{{background:#334155;color:#94a3b8}}
.badge-ok{{background:#14532d;color:#4ade80}}
.badge-no{{background:#450a0a;color:#f87171}}
input[type=text]{{background:#0f172a;border:1px solid #334155;border-radius:6px;padding:6px 10px;color:#e2e8f0;width:220px;margin-right:8px}}
select{{background:#0f172a;border:1px solid #334155;border-radius:6px;padding:6px 8px;color:#e2e8f0;margin-right:8px}}
.tbl-wrap{{max-height:480px;overflow-y:auto;border-radius:8px;border:1px solid #334155}}
.filters{{margin-bottom:10px;display:flex;flex-wrap:wrap;gap:6px;align-items:center}}
.chart-wrap{{position:relative;height:260px}}
#heatmap-tbl td, #heatmap-tbl th{{padding:0}}
#heatmap-tbl td:hover{{transform:scale(1.2);z-index:10;position:relative;box-shadow:0 0 0 2px #e2e8f0}}
{HM_NOW_CSS}
.split{{display:flex;gap:12px;flex-wrap:wrap}}
.split>div{{flex:1;min-width:280px}}
.pager{{display:flex;align-items:center;gap:8px;margin-top:8px;font-size:12px;color:#64748b}}
.pager button{{background:#1e293b;border:1px solid #334155;color:#e2e8f0;padding:4px 10px;border-radius:5px;cursor:pointer}}
.tooltip-box{{position:fixed;background:#1e293b;border:1px solid #475569;border-radius:6px;padding:10px 12px;font-size:12px;pointer-events:none;z-index:99;display:none;line-height:1.75;max-width:240px}}
.meta-bar{{background:#1e293b;padding:8px 20px;font-size:12px;color:#94a3b8;border-bottom:1px solid #334155}}
</style>
</head>
<body>
<h1>SweepHunter — แดชบอร์ดเทรด <span id="gen-time"></span></h1>
<div class="meta-bar">
  📊 ข้อมูล: <b>{era_label}</b>{date_range} &nbsp;·&nbsp; {n} ไม้ที่ปิดแล้ว
  {' &nbsp;·&nbsp; <span class="yellow">' + str(len(open_list)) + ' ไม้เปิดอยู่</span>' if open_list else ''}
  &nbsp;·&nbsp; สร้างเมื่อ <b>{gen_time}</b> &nbsp;·&nbsp; รัน <code>view_trades.bat</code> เพื่ออัปเดต
</div>
<div class="tabs">
  <div class="tab active" onclick="show('overview',this)">📈 ภาพรวม</div>
  <div class="tab" onclick="show('trades',this)">📋 รายการไม้</div>
  <div class="tab" onclick="show('heatmap',this)">🗓️ ช่วงเวลา</div>
  <div class="tab" onclick="show('series',this)">🔄 Recovery</div>
  <div class="tab" onclick="show('retrains',this)">🤖 ฝึก Model</div>
</div>

<!-- ภาพรวม -->
<div id="page-overview" class="page active">
  <div class="help-box">
    <b>อ่านอย่างไร?</b> แดชบอร์ดนี้สรุปผลบอทจาก DB — นับเฉพาะไม้ที่<b>ปิดแล้ว</b> (ชนะ/แพ้).
    ใช้ยุด config ปัจจุบัน (snapshot ≥ 14) เหมือน <code>analyze_slots_current.bat</code>.
    ไม้ที่<b>ปิดแล้ว</b>อยู่ด้านล่าง · ไม้ที่<b>ยังเปิด</b>แสดงในกล่องสีเหลือง (ถ้ามี).
  </div>
  {'<div class="panel" style="border-color:#854d0e"><h2>⚡ ไม้ที่เปิดอยู่ตอนนี้ (' + str(len(open_list)) + ')</h2><p class="desc">ยังไม่ปิด — ไม่นับในสถิติ WR/PnL จนกว่าจะ WIN/LOSS</p><div class="tbl-wrap"><table><thead><tr><th>เวลาเปิด</th><th>ทิศทาง</th><th>Lot</th><th>Conf%</th><th>เข้า</th><th>SL</th><th>TP</th><th>Step</th><th>Series</th><th>สถานะ</th></tr></thead><tbody id="open-body"></tbody></table></div></div>' if open_list else ''}
  <div class="cards">
    <div class="card"><div class="card-label">ไม้ทั้งหมด</div>
      <div class="card-value blue">{n}</div>
      <div class="card-sub">ชนะ {total_win} / แพ้ {total_loss} ไม้</div></div>
    <div class="card"><div class="card-label">อัตราชนะ (WR)</div>
      <div class="card-value {'green' if wr_pct>=45 else 'yellow' if wr_pct>=38 else 'red'}">{wr_pct:.1f}%</div>
      <div class="card-sub">% ไม้ที่ชนะ — ระบบ RR 2:1 breakeven ~33%</div></div>
    <div class="card"><div class="card-label">กำไร/ขาดทุนสุทธิ</div>
      <div class="card-value {'green' if net_pnl>=0 else 'red'}">${net_pnl:+.2f}</div>
      <div class="card-sub">รวมทุกไม้ในยุดนี้</div></div>
    <div class="card"><div class="card-label">Profit Factor (PF)</div>
      <div class="card-value {'green' if pf_total>=1.2 else 'yellow' if pf_total>=1 else 'red'}">{pf_total:.2f}</div>
      <div class="card-sub">กำไรรวม ÷ ขาดทุนรวม — PF&gt;1 = ทำกำไรสุทธิ</div></div>
    <div class="card"><div class="card-label">Risk/Reward เฉลี่ย</div>
      <div class="card-value {'green' if rr>=1.5 else 'yellow'}">{rr:.2f}x</div>
      <div class="card-sub">กำไรเฉลี่ยต่อไม้ ÷ ขาดทุนเฉลี่ย</div></div>
    <div class="card"><div class="card-label">หนี้ Recovery ปัจจุบัน</div>
      <div class="card-value {'red' if cum_loss_active>0 else 'green'}">${cum_loss_active:.2f}</div>
      <div class="card-sub">หนี้ใน series ที่กำลังกู้ · แพ้ติด {consec} ไม้</div></div>
    <div class="card"><div class="card-label">หนี้สะสม (Global)</div>
      <div class="card-value {'red' if global_debt>0 else 'green'}">${global_debt:.2f}</div>
      <div class="card-sub">หนี้ข้าม series เมื่อ recovery ครบ 4 ไม้แล้วยังไม่จบ</div></div>
  </div>

  <div class="split">
    <div class="panel">
      <h2>📈 กราฟ Equity (กำไรสะสม)</h2>
      <p class="desc">เส้นกราฟ = กำไร/ขาดทุนสะสมตามลำดับเวลา · เขียว = บวก · แดง = ลบ</p>
      <div class="chart-wrap"><canvas id="equityChart"></canvas></div>
    </div>
    <div class="panel" style="min-width:260px;max-width:320px">
      <h2>🎯 แยก BUY / SELL</h2>
      <p class="desc">สัดส่วนไม้ซื้อ vs ขาย แยกชนะ/แพ้</p>
      <div class="chart-wrap" style="height:200px"><canvas id="dirChart"></canvas></div>
      <div style="margin-top:14px;font-size:12px">
        <div style="display:flex;justify-content:space-between;margin-bottom:6px">
          <span class="blue">BUY ซื้อ</span>
          <span>{buy_win+buy_loss} ไม้ · WR {buy_win/(buy_win+buy_loss)*100 if (buy_win+buy_loss) else 0:.1f}%</span>
        </div>
        <div style="display:flex;justify-content:space-between">
          <span class="purple">SELL ขาย</span>
          <span>{sell_win+sell_loss} ไม้ · WR {sell_win/(sell_win+sell_loss)*100 if (sell_win+sell_loss) else 0:.1f}%</span>
        </div>
      </div>
    </div>
  </div>

  <div class="panel">
    <h2>🕐 20 ไม้ล่าสุด</h2>
    <div class="tbl-wrap"><table id="recent-tbl">
      <thead><tr>
        <th>เวลาเปิด</th><th>ทิศทาง</th><th>Lot</th><th>Conf%</th>
        <th>เข้า</th><th>ออก</th><th>PnL</th><th>ผล</th><th>Step</th>
      </tr></thead>
      <tbody id="recent-body"></tbody>
    </table></div>
  </div>
</div>

<!-- รายการไม้ -->
<div id="page-trades" class="page">
  <div class="help-box">
    <b>รายการไม้ทั้งหมด</b> — คลิกหัวคอลัมน์เพื่อเรียง · ค้นหาด้วยเลข series หรือ id
  </div>
  <div class="panel">
    <div class="filters">
      <input type="text" id="trade-search" placeholder="ค้นหา series / id..." oninput="filterTrades()">
      <select id="trade-side" onchange="filterTrades()">
        <option value="">ทุกทิศทาง</option>
        <option>BUY</option><option>SELL</option>
      </select>
      <select id="trade-status" onchange="filterTrades()">
        <option value="">ทุกผล</option>
        <option>WIN</option><option>LOSS</option>
      </select>
      <span id="trade-count" style="color:#64748b;margin-left:auto"></span>
    </div>
    <div class="tbl-wrap"><table>
      <thead><tr>
        <th onclick="sortTrades('ts')">เวลาเปิด</th>
        <th onclick="sortTrades('closed')">เวลาปิด</th>
        <th onclick="sortTrades('side')">ทิศทาง</th>
        <th onclick="sortTrades('vol')">Lot</th>
        <th onclick="sortTrades('conf')">Conf%</th>
        <th onclick="sortTrades('atr')">ATR</th>
        <th onclick="sortTrades('sprd')">Spread</th>
        <th onclick="sortTrades('entry')">ราคาเข้า</th>
        <th onclick="sortTrades('close')">ราคาออก</th>
        <th onclick="sortTrades('pnl')">PnL</th>
        <th onclick="sortTrades('status')">ผล</th>
        <th onclick="sortTrades('step')">Step</th>
        <th onclick="sortTrades('series')">Series</th>
      </tr></thead>
      <tbody id="trades-body"></tbody>
    </table></div>
    <div class="pager">
      <button onclick="tradePage(-1)">&#8592; ก่อน</button>
      <span id="trade-page-info"></span>
      <button onclick="tradePage(1)">ถัดไป &#8594;</button>
    </div>
  </div>
</div>

<!-- HEATMAP -->
<div id="page-heatmap" class="page">
  <div class="help-box">
    <b>Heatmap วัน × ชั่วโมง</b> — แสดงว่าช่วงไหนทำกำไร/ขาดทุน (เวลา broker UTC+{BROKER_OFFSET})
    <ul>
      <li><b>ตัวเลขในช่อง</b> = PnL สุทธิ ($) ของไม้ที่<b>ปิดแล้ว</b>ในช่วงนั้น · บรรทัดเล็ก = จำนวนไม้ + WR%</li>
      <li><b>สำคัญ</b> นับจากเวลา<b>ปิด</b> (closed_at) ไม่ใช่เวลาที่ดูหน้าจอ — ไม้ +82 ที่ปิด UTC 11:26 = broker <b>14:xx</b> → คอลัมน์ <b>14</b></li>
      <li><b>กรอบเหลือง</b> = ไม้ปิดล่าสุด · <b>คอลัมน์ฟ้า</b> = ชั่วโมง broker ตอนนี้</li>
      <li><b>สี</b> = Profit Factor (PF) — เขียว=ดี · แดง=ขาดทุนสุทธิ · กรอบแดง=ช่วงที่ block ใน config</li>
      <li><b>เกณฑ์ block</b> (analyze_slots): ≥8 ไม้ + PF&lt;0.9 — ดูรายละเอียดที่ <code>analyze_slots_current.bat</code></li>
    </ul>
  </div>
  <div class="panel">
    <h2>🗓️ Heatmap วัน × ชั่วโมง</h2>
    <p class="desc">{era_label}{date_range} · ตอนนี้ broker = <b>{now_lbl}</b> · รัน view_trades.bat แล้ว Ctrl+F5 ถ้าข้อมูลเก่า</p>
    <p class="desc">{HM_NOW_LEGEND} — กรอบฟ้า = ชั่วโมงปัจจุบัน · กรอบทอง NOW = ช่องวัน×ชม. ที่บอทอยู่ตอนนี้</p>
    <div style="margin-bottom:12px;display:flex;gap:8px;flex-wrap:wrap;align-items:center;font-size:11px">
      <span style="color:#94a3b8">สี = PF:</span>
      <span style="background:#14532d;padding:3px 10px;border-radius:4px;color:#4ade80;font-weight:600">PF &gt; 1.5 ดี</span>
      <span style="background:#713f12;padding:3px 10px;border-radius:4px;color:#fde68a;font-weight:600">PF 1.0–1.5 ปานกลาง</span>
      <span style="background:#7f1d1d;padding:3px 10px;border-radius:4px;color:#fca5a5;font-weight:600">PF &lt; 1.0 แย่</span>
      <span style="background:#1e293b;border:1px solid #334155;padding:3px 10px;border-radius:4px;color:#475569">ไม่มีข้อมูล</span>
      <span style="background:#1e293b;border:2px solid #ef4444;padding:3px 10px;border-radius:4px;color:#f87171">BLOCK ใน config</span>
    </div>
    <div style="overflow-x:auto">
      <table id="heatmap-tbl" style="border-collapse:separate;border-spacing:2px;table-layout:fixed;font-size:11px">
        <thead><tr id="hm-head"></tr></thead>
        <tbody id="hm-body"></tbody>
      </table>
    </div>
  </div>
  <div id="tooltip-box" class="tooltip-box"></div>
  <div class="panel">
    <h2>📍 ไม้ล่าสุด → ช่อง heatmap</h2>
    <p class="desc">ดูว่าไม้แต่ละไม้ไปอยู่คอลัมน์ไหน (เวลาปิด → broker slot)</p>
    <div class="tbl-wrap" style="max-height:220px"><table>
      <thead><tr><th>ID</th><th>ปิด (UTC)</th><th>PnL</th><th>ช่อง heatmap</th></tr></thead>
      <tbody id="slot-map-body"></tbody>
    </table></div>
  </div>
</div>

<!-- SERIES -->
<div id="page-series" class="page">
  <div class="help-box">
    <b>Recovery Series</b> — 1 series = 1 รอบเทรดที่อาจมีหลาย step (ไม้หลัก + ไม้กู้)
    · Step 1 = ไม้แรก · Step 2–4 = recovery · จบ series เมื่อชนะหรือครบ max_steps
  </div>
  <div class="panel">
    <h2>🔄 Recovery Series (200 ล่าสุด)</h2>
    <div class="tbl-wrap"><table>
      <thead><tr>
        <th>ID</th><th>เปิด</th><th>ปิด</th><th>ทิศทาง</th>
        <th>Steps</th><th>PnL</th><th>สถานะ</th>
      </tr></thead>
      <tbody id="series-body"></tbody>
    </table></div>
  </div>
</div>

<!-- RETRAINS -->
<div id="page-retrains" class="page">
  <div class="help-box">
    <b>ประวัติฝึก Model</b> — บันทึกทุกครั้งที่รัน retrain · CV/OOS = ความแม่นบน train/test
    · Accepted = ใช้ model ใหม่แทนของเดิม
  </div>
  <div class="panel">
    <h2>🤖 ประวัติฝึก Model</h2>
    <div class="tbl-wrap"><table>
      <thead><tr>
        <th>เวลา</th><th>แถวข้อมูล</th><th>CV Acc%</th><th>OOS Acc%</th><th>รับใช้</th><th>หมายเหตุ</th>
      </tr></thead>
      <tbody id="retrain-body"></tbody>
    </table></div>
  </div>
</div>

<script>
const TRADES = {json.dumps(trades)};
const OPEN_TRADES = {json.dumps(open_list)};
const SERIES = {json.dumps(series_list)};
const RETRAINS = {json.dumps(retrain_list)};
const EQUITY = {json.dumps(equity_curve)};
const HM = {json.dumps(hm_data)};
const DOW = {json.dumps(list(DOW_TH_SHORT))};
const DOW_FULL = {json.dumps(list(DOW_TH))};
const BLOCKED_SLOTS_DATA = {json.dumps(blocked_slots)};
const NOW_SLOT = {json.dumps(now_slot)};
const RECENT_SLOTS = {json.dumps(recent_slot_trades)};
const DOW_EN = {json.dumps(list(DOW))};
const LATEST_TRADE = RECENT_SLOTS.length ? RECENT_SLOTS[0] : null;

document.getElementById('gen-time').textContent = 'สร้างเมื่อ {gen_time}';

function show(name, el) {{
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.getElementById('page-'+name).classList.add('active');
  if (el) el.classList.add('active');
  if (name === 'heatmap') buildHeatmap();
}}

// ── open trades ───────────────────────────────────────────────────────────────
const openBody = document.getElementById('open-body');
if (openBody) {{
  OPEN_TRADES.forEach(t => {{
    const sideCls = (t.side || 'hold').toLowerCase();
    openBody.innerHTML += `<tr>
      <td>${{t.ts}}</td>
      <td><span class="badge badge-${{sideCls}}">${{t.side}}</span></td>
      <td>${{t.vol}}</td><td>${{t.conf}}%</td>
      <td>${{t.entry}}</td><td>${{t.sl}}</td><td>${{t.tp}}</td>
      <td>${{t.step}}</td><td>#${{t.series}}</td>
      <td><span class="badge badge-ok">${{t.status}}</span></td>
    </tr>`;
  }});
}}

// ── recent 20 ─────────────────────────────────────────────────────────────────
const recBody = document.getElementById('recent-body');
TRADES.slice(-20).reverse().forEach(t => {{
  const sideCls = (t.side || 'hold').toLowerCase();
  recBody.innerHTML += `<tr>
    <td>${{t.ts}}</td>
    <td><span class="badge badge-${{sideCls}}">${{t.side}}</span></td>
    <td>${{t.vol}}</td>
    <td>${{t.conf}}%</td>
    <td>${{t.entry}}</td>
    <td>${{t.close}}</td>
    <td style="color:${{t.pnl>=0?'#22c55e':'#ef4444'}};font-weight:600">${{t.pnl>=0?'+':''}}${{t.pnl}}</td>
    <td><span class="badge badge-${{t.status.toLowerCase()}}">${{t.status}}</span></td>
    <td>${{t.step}}</td>
  </tr>`;
}});

// ── trades table ───────────────────────────────────────────────────────────────
let filteredTrades = [...TRADES];
let tradeSortKey = 'ts';
let tradeSortAsc = false;
let tradePg = 0;
const PAGE_SIZE = 50;

function filterTrades() {{
  const q = document.getElementById('trade-search').value.toLowerCase();
  const side = document.getElementById('trade-side').value;
  const status = document.getElementById('trade-status').value;
  filteredTrades = TRADES.filter(t => {{
    if (q && !String(t.series).includes(q) && !String(t.id).includes(q)) return false;
    if (side && t.side !== side) return false;
    if (status && t.status !== status) return false;
    return true;
  }});
  tradePg = 0;
  renderTradesTable();
}}

function sortTrades(key) {{
  if (tradeSortKey === key) tradeSortAsc = !tradeSortAsc;
  else {{ tradeSortKey = key; tradeSortAsc = true; }}
  filteredTrades.sort((a,b) => {{
    const av = a[key], bv = b[key];
    return tradeSortAsc ? (av > bv ? 1 : -1) : (av < bv ? 1 : -1);
  }});
  renderTradesTable();
}}

function tradePage(d) {{
  const maxPg = Math.ceil(filteredTrades.length / PAGE_SIZE) - 1;
  tradePg = Math.max(0, Math.min(maxPg, tradePg + d));
  renderTradesTable();
}}

function renderTradesTable() {{
  const body = document.getElementById('trades-body');
  const start = tradePg * PAGE_SIZE;
  const page = filteredTrades.slice(start, start + PAGE_SIZE);
  body.innerHTML = '';
  page.forEach(t => {{
    const sideCls = (t.side || 'hold').toLowerCase();
    body.innerHTML += `<tr>
      <td>${{t.ts}}</td><td>${{t.closed}}</td>
      <td><span class="badge badge-${{sideCls}}">${{t.side}}</span></td>
      <td>${{t.vol}}</td><td>${{t.conf}}%</td><td>${{t.atr}}</td><td>${{t.sprd}}</td>
      <td>${{t.entry}}</td><td>${{t.close}}</td>
      <td style="color:${{t.pnl>=0?'#22c55e':'#ef4444'}};font-weight:600">${{t.pnl>=0?'+':''}}${{t.pnl}}</td>
      <td><span class="badge badge-${{t.status.toLowerCase()}}">${{t.status}}</span></td>
      <td>${{t.step}}</td><td>#${{t.series}}</td>
    </tr>`;
  }});
  document.getElementById('trade-count').textContent =
    `${{filteredTrades.length}} ไม้`;
  document.getElementById('trade-page-info').textContent =
    `หน้า ${{tradePg+1}} / ${{Math.max(1, Math.ceil(filteredTrades.length/PAGE_SIZE))}}`;
}}
filteredTrades.sort((a,b) => a.ts > b.ts ? -1 : 1);
renderTradesTable();

// ── heatmap ────────────────────────────────────────────────────────────────────
// blocked slots from config (written by Python below)
const BLOCKED_SLOTS = BLOCKED_SLOTS_DATA;

function isBlocked(dow, h) {{
  return BLOCKED_SLOTS.some(s => s.dow === dow && s.hours.includes(h));
}}

function pfColor(pf) {{
  if (pf >= 1.5) {{
    const intensity = Math.min((pf - 1.5) / 2.0 * 0.5 + 0.5, 1.0);
    return {{ bg: `rgba(20,83,45,${{intensity}})`, tc: '#4ade80' }};
  }} else if (pf >= 1.0) {{
    const intensity = Math.min((pf - 1.0) / 0.5 * 0.5 + 0.35, 1.0);
    return {{ bg: `rgba(113,63,18,${{intensity}})`, tc: '#fde68a' }};
  }} else {{
    const intensity = Math.min((1.0 - pf) / 1.0 * 0.6 + 0.35, 1.0);
    return {{ bg: `rgba(127,29,29,${{intensity}})`, tc: '#fca5a5' }};
  }}
}}

function fmtPnl(v) {{
  const sign = v >= 0 ? '+' : '';
  return sign + '$' + Math.abs(v).toFixed(2);
}}

function buildHeatmap() {{
  const head = document.getElementById('hm-head');
  const body = document.getElementById('hm-body');
  head.innerHTML = '';
  body.innerHTML = '';

  head.innerHTML = '<th style="width:52px;background:#0f172a;padding:4px 2px;color:#475569;font-size:10px;text-align:center">วัน/ชม</th>';
  for (let h = 0; h < 24; h++) {{
    const nowCol = h === NOW_SLOT.hour ? ' hm-now-col' : '';
    head.innerHTML += `<th class="${{nowCol}}" style="width:42px;background:#0f172a;padding:4px 2px;color:${{nowCol?'#60a5fa':'#64748b'}};font-size:10px;text-align:center;font-weight:${{nowCol?'700':'normal'}}">${{h.toString().padStart(2,'0')}}</th>`;
  }}

  for (let dow = 0; dow < 7; dow++) {{
    const rowCls = dow === NOW_SLOT.dow ? ' hm-now-row' : '';
    let row = `<tr class="${{rowCls}}"><td style="background:#1e293b;padding:6px 8px;font-weight:700;color:#94a3b8;text-align:center;border-radius:4px;white-space:nowrap">${{DOW_EN[dow]}}</td>`;
    for (let h = 0; h < 24; h++) {{
      const key = `${{dow}}_${{h}}`;
      const d = HM[key];
      const blocked = isBlocked(dow, h);
      const isLatest = LATEST_TRADE && LATEST_TRADE.dow === dow && LATEST_TRADE.hour === h;
      const isNow = dow === NOW_SLOT.dow && h === NOW_SLOT.hour;
      const nowCol = h === NOW_SLOT.hour ? ' hm-now-col' : '';
      let style = 'width:42px;height:36px;text-align:center;vertical-align:middle;border-radius:3px;cursor:default;';
      let cls = nowCol + (isLatest ? ' hm-latest' : '') + (isNow ? ' hm-now-cell' : '');

      if (d && d.t > 0) {{
        const c = d.pnl >= 0 ? pfColor(Math.max(d.pf, 1.01)) : pfColor(Math.min(d.pf, 0.99));
        style += `background:${{c.bg}};color:${{d.pnl>=0?'#4ade80':'#fca5a5'}};font-weight:700;font-size:9px;line-height:1.2;`;
        if (blocked) style += 'outline:2px solid #ef4444;outline-offset:-2px;';
        const pnlLbl = fmtPnl(d.pnl);
        const subLbl = d.t + 't · ' + d.wr + '%';
        row += `<td class="${{cls}}" style="${{style}}" title="${{DOW_EN[dow]}} ${{h.toString().padStart(2,'0')}}:xx | ${{d.t}} ไม้ | WR=${{d.wr}}% | PF=${{d.pf}} | PnL=${{fmtPnl(d.pnl)}}"
          onmouseenter="showTooltip(event,{{'t':${{d.t}},'wr':${{d.wr}},'pf':${{d.pf}},'pnl':${{d.pnl}}}},${{dow}},${{h}})"
          onmouseleave="document.getElementById('tooltip-box').style.display='none'"
          ><div>${{pnlLbl}}</div><div style="font-size:8px;font-weight:400;opacity:.85">${{subLbl}}</div></td>`;
      }} else {{
        style += 'background:#0a0f1a;color:#1e293b;';
        if (blocked) style += 'outline:2px solid #ef444455;outline-offset:-2px;';
        row += `<td class="${{cls}}" style="${{style}}">${{blocked?'BLOCK':'—'}}</td>`;
      }}
    }}
    row += '</tr>';
    body.innerHTML += row;
  }}

  const mapBody = document.getElementById('slot-map-body');
  if (mapBody) {{
    mapBody.innerHTML = '';
    RECENT_SLOTS.forEach((r, i) => {{
      const hi = i === 0 ? ' style="background:#422006"' : '';
      mapBody.innerHTML += `<tr${{hi}}><td>#${{r.id}}</td><td>${{r.closed}}</td>
        <td style="color:${{r.pnl>=0?'#22c55e':'#ef4444'}};font-weight:600">${{fmtPnl(r.pnl)}}</td>
        <td><b>${{r.slot}}</b>${{i===0?' ← ล่าสุด':''}}</td></tr>`;
    }});
  }}
}}

function showTooltip(e, d, dow, h) {{
  const box = document.getElementById('tooltip-box');
  box.innerHTML = `<b>${{DOW_FULL[dow]}} ${{h.toString().padStart(2,'0')}}:xx</b> (broker UTC+{BROKER_OFFSET})<br>
    ไม้: ${{d.t}}<br>
    อัตราชนะ: ${{d.wr}}%<br>
    PF: ${{d.pf}}<br>
    PnL: ${{d.pnl>=0?'+':''}}$${{d.pnl}}`;
  box.style.display = 'block';
  box.style.left = (e.clientX + 12) + 'px';
  box.style.top  = (e.clientY - 10) + 'px';
}}

// ── series ────────────────────────────────────────────────────────────────────
const sBody = document.getElementById('series-body');
SERIES.forEach(s => {{
  const statusColor = s.status.includes('TP') ? 'badge-win' :
                      s.status.includes('MAX') ? 'badge-loss' :
                      s.pnl >= 0 ? 'badge-win' : 'badge-loss';
  sBody.innerHTML += `<tr>
    <td>#${{s.id}}</td><td>${{s.opened}}</td><td>${{s.closed || '—'}}</td>
    <td><span class="badge badge-${{s.side.toLowerCase()}}">${{s.side||'—'}}</span></td>
    <td>${{s.steps}}</td>
    <td style="color:${{s.pnl>=0?'#22c55e':'#ef4444'}};font-weight:600">${{s.pnl>=0?'+':''}}${{s.pnl}}</td>
    <td><span class="badge ${{statusColor}}" style="font-size:10px">${{s.status}}</span></td>
  </tr>`;
}});

// ── retrains ──────────────────────────────────────────────────────────────────
const rBody = document.getElementById('retrain-body');
RETRAINS.forEach(r => {{
  rBody.innerHTML += `<tr>
    <td>${{r.ts}}</td><td>${{r.rows.toLocaleString()}}</td>
    <td>${{r.cv}}%</td><td>${{r.oos}}%</td>
    <td><span class="badge ${{r.ok?'badge-ok':'badge-no'}}">${{r.ok?'ใช่':'ไม่'}}</span></td>
    <td style="color:#64748b">${{r.notes}}</td>
  </tr>`;
}});

// ── charts (optional — tables above always render even if CDN fails) ────────────
function initCharts() {{
  if (typeof Chart === 'undefined') {{
    document.querySelectorAll('.chart-wrap').forEach(el => {{
      el.innerHTML = '<p style="color:#64748b;padding:20px;text-align:center">กราฟไม่โหลด (Chart.js CDN) — ตารางด้านล่างใช้ได้ปกติ</p>';
    }});
    return;
  }}
  try {{
    const eqCtx = document.getElementById('equityChart').getContext('2d');
    new Chart(eqCtx, {{
      type: 'line',
      data: {{
        labels: EQUITY.map(e => e.t),
        datasets: [{{
          data: EQUITY.map(e => e.pnl),
          borderColor: '#22c55e',
          segment: {{ borderColor: ctx => (ctx.p1.parsed.y >= 0 ? '#22c55e' : '#ef4444') }},
          borderWidth: 1.5, fill: false, pointRadius: 0, tension: 0.2
        }}]
      }},
      options: {{
        responsive: true, maintainAspectRatio: false, animation: false,
        plugins: {{ legend: {{ display: false }} }},
        scales: {{
          x: {{ display: false }},
          y: {{ grid: {{ color: '#1e293b' }}, ticks: {{ color: '#64748b', callback: v => '$'+v }} }}
        }}
      }}
    }});
    const dirCtx = document.getElementById('dirChart').getContext('2d');
    new Chart(dirCtx, {{
      type: 'doughnut',
      data: {{
        labels: ['ซื้อชนะ','ซื้อแพ้','ขายชนะ','ขายแพ้'],
        datasets: [{{ data: [{buy_win},{buy_loss},{sell_win},{sell_loss}],
          backgroundColor: ['#1d4ed8','#1e3a5f','#7c3aed','#3b1a40'],
          borderColor: '#0f172a', borderWidth: 2 }}]
      }},
      options: {{
        responsive: true, maintainAspectRatio: false, animation: false,
        plugins: {{ legend: {{ position: 'right', labels: {{ color: '#94a3b8', font: {{ size: 11 }} }} }} }}
      }}
    }});
  }} catch (err) {{
    console.warn('Chart init failed:', err);
  }}
}}
function onReady() {{
  buildHeatmap();
  initCharts();
}}
if (document.readyState === 'loading') {{
  document.addEventListener('DOMContentLoaded', onReady);
}} else {{
  onReady();
}}
</script>
</body>
</html>"""

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    f.write(HTML)

abs_out = os.path.abspath(OUT)
print(f"แดชบอร์ด: {abs_out}")
print(f"  Era: {era_label}")
print(f"  Trades: {n}  Open: {len(open_list)}  WR: {wr_pct:.1f}%  PnL: ${net_pnl:+.2f}")
webbrowser.open(f"file:///{abs_out.replace(chr(92),'/')}")
