"""
generate_report.py
==================
Generate a beautiful HTML report from hyper_trades.sqlite

Usage: python generate_report.py
Output: data/reports/report_<timestamp>.html  (auto-opens in browser)
"""
from __future__ import annotations
import sys, os, sqlite3, webbrowser
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

DB_PATH = Path("data/db/hyper_trades.sqlite")
OUT_DIR = Path("data/reports")
OUT_DIR.mkdir(parents=True, exist_ok=True)

if not DB_PATH.exists():
    print(f"[ERR] DB not found: {DB_PATH}")
    sys.exit(1)

con = sqlite3.connect(str(DB_PATH))
con.row_factory = sqlite3.Row
cur = con.cursor()

# ---------- Aggregations ----------
def fetchall(sql, params=()):
    return cur.execute(sql, params).fetchall()

# Overall
overall = fetchall("""SELECT
    COUNT(*) total,
    SUM(CASE WHEN status='WIN'  THEN 1 ELSE 0 END) wins,
    SUM(CASE WHEN status='LOSS' THEN 1 ELSE 0 END) losses,
    COALESCE(SUM(pnl),0) net,
    COALESCE(AVG(CASE WHEN status='WIN'  THEN pnl END),0) avg_win,
    COALESCE(AVG(CASE WHEN status='LOSS' THEN pnl END),0) avg_loss,
    COALESCE(MAX(CASE WHEN status='WIN'  THEN pnl END),0) max_win,
    COALESCE(MIN(CASE WHEN status='LOSS' THEN pnl END),0) max_loss
    FROM decisions WHERE status IN ('WIN','LOSS')""")[0]

total = overall["total"] or 0
wins = overall["wins"] or 0
losses = overall["losses"] or 0
wr = (wins/total*100) if total else 0
net = overall["net"] or 0
avg_win = overall["avg_win"] or 0
avg_loss = overall["avg_loss"] or 0
profit_factor = abs((wins*avg_win)/(losses*avg_loss)) if losses and avg_loss else 0
real_rr = abs(avg_win/avg_loss) if avg_loss else 0

# Direction breakdown
direction = fetchall("""SELECT prediction,
    COUNT(*) n,
    SUM(CASE WHEN status='WIN' THEN 1 ELSE 0 END) wins,
    COALESCE(SUM(pnl),0) net
    FROM decisions WHERE status IN ('WIN','LOSS')
    GROUP BY prediction""")

# Hour breakdown
hour = fetchall("""SELECT
    CAST(SUBSTR(ts_utc,12,2) AS INTEGER) hr,
    COUNT(*) n,
    SUM(CASE WHEN status='WIN' THEN 1 ELSE 0 END) wins,
    COALESCE(SUM(pnl),0) net
    FROM decisions WHERE status IN ('WIN','LOSS')
    GROUP BY hr ORDER BY hr""")

# ATR regime
atr_q = fetchall("""SELECT atr FROM decisions WHERE atr>0 AND status IN ('WIN','LOSS') ORDER BY atr""")
atrs = [r["atr"] for r in atr_q]
if atrs:
    q1, q2 = atrs[len(atrs)//3], atrs[2*len(atrs)//3]
else:
    q1 = q2 = 0
atr_regime = fetchall(f"""SELECT
    CASE
      WHEN atr<{q1} THEN 'LOW'
      WHEN atr<{q2} THEN 'MID'
      ELSE 'HIGH'
    END regime,
    COUNT(*) n,
    SUM(CASE WHEN status='WIN' THEN 1 ELSE 0 END) wins,
    COALESCE(SUM(pnl),0) net
    FROM decisions WHERE atr>0 AND status IN ('WIN','LOSS')
    GROUP BY regime""")

# Recovery series outcomes
series_q = fetchall("""SELECT status, COUNT(*) n, COALESCE(SUM(final_pnl),0) net
    FROM series WHERE status != 'OPEN' GROUP BY status""")

# Confidence buckets
conf_q = fetchall("""SELECT
    CASE
      WHEN confidence<0.45 THEN '<45%'
      WHEN confidence<0.50 THEN '45-50%'
      WHEN confidence<0.55 THEN '50-55%'
      WHEN confidence<0.60 THEN '55-60%'
      ELSE '>=60%'
    END bucket,
    COUNT(*) n,
    SUM(CASE WHEN status='WIN' THEN 1 ELSE 0 END) wins,
    COALESCE(SUM(pnl),0) net
    FROM decisions WHERE status IN ('WIN','LOSS') AND confidence IS NOT NULL
    GROUP BY bucket ORDER BY bucket""")

# Daily breakdown (last 14 days)
daily = fetchall("""SELECT
    SUBSTR(ts_utc,1,10) d,
    COUNT(*) n,
    SUM(CASE WHEN status='WIN' THEN 1 ELSE 0 END) wins,
    COALESCE(SUM(pnl),0) net
    FROM decisions WHERE status IN ('WIN','LOSS')
    GROUP BY d ORDER BY d DESC LIMIT 14""")

# Equity curve
curve = fetchall("""SELECT ts_utc, pnl FROM decisions WHERE status IN ('WIN','LOSS') ORDER BY ts_utc""")
equity = []
running = 0.0
for r in curve:
    running += (r["pnl"] or 0)
    equity.append((r["ts_utc"], running))

# Recent 25 trades
recent = fetchall("""SELECT ts_utc, prediction, confidence, atr, spread_points,
    entry_price, close_price, volume, pnl, status
    FROM decisions WHERE status IN ('WIN','LOSS')
    ORDER BY ts_utc DESC LIMIT 25""")

# ---------- HTML build ----------
def fmt_money(v): return f"<span class='{'pos' if v>=0 else 'neg'}'>${v:+.2f}</span>"
def fmt_wr(w,n): return f"{(w/n*100):.1f}%" if n else "—"

side_name = {0:"SELL", 1:"HOLD", 2:"BUY"}

# direction rows
dir_rows = ""
for r in direction:
    sn = side_name.get(r["prediction"], str(r["prediction"]))
    color = "#10b981" if sn=="BUY" else ("#ef4444" if sn=="SELL" else "#6b7280")
    dir_rows += f"""<tr>
        <td><span style='background:{color};color:white;padding:3px 10px;border-radius:6px;font-weight:600'>{sn}</span></td>
        <td>{r['n']}</td><td>{r['wins']}</td>
        <td>{fmt_wr(r['wins'],r['n'])}</td>
        <td>{fmt_money(r['net'])}</td>
    </tr>"""

# hour rows
hour_rows = ""
for r in hour:
    bar_w = (r["wins"]/r["n"]*100) if r["n"] else 0
    bar_color = "#10b981" if bar_w>=50 else ("#f59e0b" if bar_w>=40 else "#ef4444")
    hour_rows += f"""<tr>
        <td><b>{r['hr']:02d}:00 UTC</b></td>
        <td>{r['n']}</td>
        <td>{fmt_wr(r['wins'],r['n'])}</td>
        <td><div class='bar'><div style='width:{bar_w}%;background:{bar_color}'>&nbsp;</div></div></td>
        <td>{fmt_money(r['net'])}</td>
    </tr>"""

# atr regime rows
atr_rows = ""
regime_color = {"LOW":"#3b82f6", "MID":"#f59e0b", "HIGH":"#ef4444"}
for r in atr_regime:
    color = regime_color.get(r["regime"], "#6b7280")
    atr_rows += f"""<tr>
        <td><span style='background:{color};color:white;padding:3px 10px;border-radius:6px'>{r['regime']}</span></td>
        <td>{r['n']}</td><td>{fmt_wr(r['wins'],r['n'])}</td>
        <td>{fmt_money(r['net'])}</td>
    </tr>"""

# series rows
series_rows = ""
for r in series_q:
    series_rows += f"""<tr><td><b>{r['status']}</b></td><td>{r['n']}</td><td>{fmt_money(r['net'])}</td></tr>"""

# conf rows
conf_rows = ""
for r in conf_q:
    bar_w = (r["wins"]/r["n"]*100) if r["n"] else 0
    bar_color = "#10b981" if bar_w>=50 else ("#f59e0b" if bar_w>=40 else "#ef4444")
    conf_rows += f"""<tr>
        <td><b>{r['bucket']}</b></td><td>{r['n']}</td>
        <td>{fmt_wr(r['wins'],r['n'])}</td>
        <td><div class='bar'><div style='width:{bar_w}%;background:{bar_color}'>&nbsp;</div></div></td>
        <td>{fmt_money(r['net'])}</td>
    </tr>"""

# daily rows
daily_rows = ""
for r in reversed(daily):
    daily_rows += f"""<tr>
        <td><b>{r['d']}</b></td><td>{r['n']}</td>
        <td>{fmt_wr(r['wins'],r['n'])}</td>
        <td>{fmt_money(r['net'])}</td>
    </tr>"""

# recent rows
rec_rows = ""
for r in recent:
    sn = side_name.get(r["prediction"], "?")
    side_color = "#10b981" if sn=="BUY" else "#ef4444"
    status_color = "#10b981" if r["status"]=="WIN" else "#ef4444"
    rec_rows += f"""<tr>
        <td>{r['ts_utc'][5:16].replace('T',' ')}</td>
        <td><span style='color:{side_color};font-weight:600'>{sn}</span></td>
        <td>{(r['confidence'] or 0)*100:.0f}%</td>
        <td>{r['atr'] or 0:.2f}</td>
        <td>{r['spread_points'] or 0:.0f}</td>
        <td>{r['volume'] or 0:.2f}</td>
        <td>{r['entry_price'] or 0:.2f}</td>
        <td>{r['close_price'] or 0:.2f}</td>
        <td><span style='color:{status_color};font-weight:700'>{r['status']}</span></td>
        <td>{fmt_money(r['pnl'] or 0)}</td>
    </tr>"""

# equity curve as inline SVG
svg = ""
if equity:
    pnls = [v for _,v in equity]
    n = len(pnls)
    minv, maxv = min(pnls), max(pnls)
    rng = maxv - minv if maxv != minv else 1
    W, H = 900, 250
    pad = 20
    pts = []
    for i, v in enumerate(pnls):
        x = pad + i * (W - 2*pad) / max(1, n-1)
        y = H - pad - (v - minv) / rng * (H - 2*pad)
        pts.append(f"{x:.1f},{y:.1f}")
    poly = " ".join(pts)
    zero_y = H - pad - (0 - minv) / rng * (H - 2*pad) if minv <= 0 <= maxv else None
    zero_line = f"<line x1='{pad}' y1='{zero_y:.1f}' x2='{W-pad}' y2='{zero_y:.1f}' stroke='#999' stroke-dasharray='3,3'/>" if zero_y else ""
    final_color = "#10b981" if pnls[-1] >= 0 else "#ef4444"
    svg = f"""<svg viewBox='0 0 {W} {H}' style='width:100%;max-width:900px;background:#0f172a;border-radius:8px'>
        {zero_line}
        <polyline points='{poly}' fill='none' stroke='{final_color}' stroke-width='2'/>
        <text x='{pad}' y='15' fill='#94a3b8' font-size='11'>Equity Curve (cumulative P/L)</text>
        <text x='{W-pad}' y='15' fill='{final_color}' font-size='11' text-anchor='end'>End: ${pnls[-1]:+.2f}</text>
        <text x='{pad}' y='{H-3}' fill='#94a3b8' font-size='10'>Min: ${minv:+.2f}</text>
        <text x='{W-pad}' y='{H-3}' fill='#94a3b8' font-size='10' text-anchor='end'>Max: ${maxv:+.2f}</text>
    </svg>"""

ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
generated = datetime.now().strftime("%Y%m%d_%H%M%S")
net_color = "#10b981" if net >= 0 else "#ef4444"
wr_color = "#10b981" if wr >= 50 else ("#f59e0b" if wr >= 45 else "#ef4444")

html = f"""<!DOCTYPE html>
<html lang="th">
<head>
<meta charset="utf-8">
<title>SweepHunter Report — {ts}</title>
<style>
* {{ box-sizing: border-box; margin:0; padding:0; }}
body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #f1f5f9; color: #1e293b; padding: 20px; }}
.container {{ max-width: 1200px; margin: 0 auto; }}
header {{ background: linear-gradient(135deg, #1e3a8a, #3b82f6); color: white; padding: 30px; border-radius: 12px; margin-bottom: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }}
header h1 {{ font-size: 28px; margin-bottom: 8px; }}
header .sub {{ opacity: 0.85; font-size: 14px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px,1fr)); gap: 15px; margin-bottom: 20px; }}
.card {{ background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }}
.card .label {{ font-size: 12px; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; }}
.card .value {{ font-size: 28px; font-weight: 700; }}
.card .sub {{ font-size: 12px; color: #94a3b8; margin-top: 4px; }}
.section {{ background: white; padding: 25px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); margin-bottom: 20px; }}
.section h2 {{ font-size: 18px; margin-bottom: 15px; color: #334155; padding-bottom: 10px; border-bottom: 2px solid #e2e8f0; }}
table {{ width: 100%; border-collapse: collapse; }}
th {{ background: #f8fafc; padding: 10px; text-align: left; font-size: 12px; color: #64748b; text-transform: uppercase; border-bottom: 2px solid #e2e8f0; }}
td {{ padding: 10px; border-bottom: 1px solid #f1f5f9; font-size: 14px; }}
tr:hover {{ background: #f8fafc; }}
.pos {{ color: #10b981; font-weight: 600; }}
.neg {{ color: #ef4444; font-weight: 600; }}
.bar {{ width: 100px; height: 14px; background: #e2e8f0; border-radius: 4px; overflow: hidden; display: inline-block; vertical-align: middle; }}
.bar > div {{ height: 100%; }}
.two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
@media (max-width: 768px) {{ .two-col {{ grid-template-columns: 1fr; }} }}
footer {{ text-align: center; color: #94a3b8; font-size: 12px; margin-top: 30px; padding: 15px; }}
</style>
</head>
<body>
<div class="container">
  <header>
    <h1>🏆 SweepHunter Trading Report</h1>
    <div class="sub">Generated: {ts}  •  Source: hyper_trades.sqlite</div>
  </header>

  <div class="grid">
    <div class="card">
      <div class="label">Total Trades</div>
      <div class="value">{total}</div>
      <div class="sub">{wins} wins / {losses} losses</div>
    </div>
    <div class="card">
      <div class="label">Win Rate</div>
      <div class="value" style="color:{wr_color}">{wr:.1f}%</div>
      <div class="sub">break-even ~50%</div>
    </div>
    <div class="card">
      <div class="label">Net P/L</div>
      <div class="value" style="color:{net_color}">${net:+.2f}</div>
      <div class="sub">across all trades</div>
    </div>
    <div class="card">
      <div class="label">Real RR</div>
      <div class="value">{real_rr:.2f}</div>
      <div class="sub">avg_win / avg_loss</div>
    </div>
    <div class="card">
      <div class="label">Profit Factor</div>
      <div class="value">{profit_factor:.2f}</div>
      <div class="sub">≥1 = profitable</div>
    </div>
    <div class="card">
      <div class="label">Avg Win / Loss</div>
      <div class="value" style="font-size:18px"><span class='pos'>${avg_win:+.2f}</span> / <span class='neg'>${avg_loss:+.2f}</span></div>
      <div class="sub">max ${overall['max_win']:+.2f} / ${overall['max_loss']:+.2f}</div>
    </div>
  </div>

  <div class="section">
    <h2>📈 Equity Curve</h2>
    {svg or '<p>No data</p>'}
  </div>

  <div class="two-col">
    <div class="section">
      <h2>🎯 Direction Breakdown</h2>
      <table><thead><tr><th>Side</th><th>N</th><th>Wins</th><th>WR</th><th>Net P/L</th></tr></thead>
      <tbody>{dir_rows}</tbody></table>
    </div>
    <div class="section">
      <h2>⚡ ATR Regime</h2>
      <table><thead><tr><th>Regime</th><th>N</th><th>WR</th><th>Net P/L</th></tr></thead>
      <tbody>{atr_rows}</tbody></table>
    </div>
  </div>

  <div class="section">
    <h2>🕐 Win Rate per Hour (UTC)</h2>
    <table><thead><tr><th>Hour</th><th>Trades</th><th>WR</th><th>Distribution</th><th>Net P/L</th></tr></thead>
    <tbody>{hour_rows}</tbody></table>
  </div>

  <div class="section">
    <h2>🎲 Win Rate by Confidence</h2>
    <table><thead><tr><th>Bucket</th><th>N</th><th>WR</th><th>Distribution</th><th>Net P/L</th></tr></thead>
    <tbody>{conf_rows}</tbody></table>
  </div>

  <div class="two-col">
    <div class="section">
      <h2>📅 Daily P/L (last 14)</h2>
      <table><thead><tr><th>Date</th><th>Trades</th><th>WR</th><th>Net</th></tr></thead>
      <tbody>{daily_rows}</tbody></table>
    </div>
    <div class="section">
      <h2>♻️ Recovery Series Outcomes</h2>
      <table><thead><tr><th>Status</th><th>N</th><th>Net P/L</th></tr></thead>
      <tbody>{series_rows or '<tr><td colspan=3>No closed series</td></tr>'}</tbody></table>
    </div>
  </div>

  <div class="section">
    <h2>📋 Recent 25 Trades</h2>
    <table><thead><tr>
      <th>Time</th><th>Side</th><th>Conf</th><th>ATR</th><th>Sprd</th>
      <th>Vol</th><th>Entry</th><th>Close</th><th>Result</th><th>P/L</th>
    </tr></thead><tbody>{rec_rows}</tbody></table>
  </div>

  <footer>
    🤖 SweepHunter AI • XAUUSD Hyper-Frequency Bot • Report generated by generate_report.py
  </footer>
</div>
</body>
</html>"""

out = OUT_DIR / f"report_{generated}.html"
out.write_text(html, encoding="utf-8")
print(f"[OK] Report saved: {out}")
print(f"     Total: {total} trades  |  WR: {wr:.1f}%  |  Net: ${net:+.2f}")

# Auto-open in browser
try:
    webbrowser.open(out.resolve().as_uri())
    print("[OK] Opening in browser...")
except Exception as e:
    print(f"[WARN] Could not auto-open: {e}")
