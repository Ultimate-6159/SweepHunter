"""
simulate_recovery.py — จำลอง recovery ทีละไม้ (เคสแย่: โดน SL ทุกไม้)
Usage: python simulate_recovery.py
"""
from __future__ import annotations

import json
import math
import webbrowser
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.json"
OUT_DIR = ROOT / "data/reports"


@dataclass
class SimParams:
    name: str
    balance: float
    atr: float
    sl_mult: float
    tp_mult: float
    sl_mult_recovery: float
    risk_pct: float
    max_lot_pct: float
    lot_mult: float
    vol_mult: float
    max_rec_mult: float
    max_steps: int
    commission: float
    usd_per_point_per_lot: float = 100.0
    volume_step: float = 0.01
    primary_loss_usd: float = 25.0


def load_cfg() -> dict:
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return {}


def base_lot_and_cap(p: SimParams) -> tuple[float, float]:
    sl_dist = p.sl_mult * p.atr
    loss_per_lot = sl_dist * p.usd_per_point_per_lot
    risk_usd = p.balance * p.risk_pct / 100.0
    base = risk_usd / max(loss_per_lot, 1e-9)
    cap = (p.balance * p.max_lot_pct / 100.0) / max(loss_per_lot, 1e-9)
    base = max(0.01, round(base, 3))
    cap = max(base, round(cap, 2))
    return base, cap


def net_profit_per_lot(p: SimParams) -> float:
    tp_dist = p.tp_mult * p.atr
    gross = tp_dist * p.usd_per_point_per_lot
    return max(gross - p.commission, 1e-9)


def calc_lot(
    p: SimParams,
    consec_losses: int,
    cum_loss: float,
    cum_vol: float,
    old_style: bool = False,
) -> dict:
    base, abs_cap = base_lot_and_cap(p)
    net = net_profit_per_lot(p)
    mult = p.lot_mult
    vol_m = p.vol_mult
    max_rec = 0.0 if old_style else p.max_rec_mult

    steps_used = (
        consec_losses
        if p.max_steps <= 0
        else min(consec_losses, p.max_steps)
    )
    geo = base * (mult ** steps_used)
    recover = (cum_loss + 0.5) / net
    vol_floor = cum_vol * vol_m

    rec_cap = base * max_rec if max_rec > 0 else abs_cap
    if max_rec > 0:
        geo = min(geo, rec_cap)
        recover = min(recover, rec_cap)
        vol_floor = min(vol_floor, rec_cap)

    raw = max(geo, recover, vol_floor)
    step = p.volume_step
    final = math.ceil(raw / step - 1e-9) * step
    if not old_style:
        final = min(final, abs_cap, rec_cap)
    else:
        final = min(final, abs_cap)

    driver = "geo"
    if abs(raw - recover) < 1e-6 or recover >= geo >= vol_floor:
        driver = "recover"
    if vol_floor >= geo and vol_floor >= recover:
        driver = "vol"
    if max_rec > 0 and final < raw - 1e-6:
        driver += "+cap"

    sl_m = p.sl_mult_recovery if consec_losses > 0 and p.sl_mult_recovery > 0 else p.sl_mult
    sl_dist = sl_m * p.atr
    risk_usd = sl_dist * final * p.usd_per_point_per_lot

    return {
        "step": consec_losses + 1,
        "base": base,
        "abs_cap": abs_cap,
        "rec_cap": rec_cap,
        "geo": geo,
        "recover": recover,
        "vol_floor": vol_floor,
        "lot": final,
        "driver": driver,
        "sl_mult": sl_m,
        "risk_usd": risk_usd,
        "net_per_lot": net,
    }


def run_series(p: SimParams, n_steps: int, old_style: bool = False) -> list[dict]:
    rows = []
    cum_loss = 0.0
    cum_vol = 0.0
    total_loss = 0.0

    # ไม้แรก (PRIMARY) — สมมติเสีย fixed หรือตาม SL
    sl_dist0 = p.sl_mult * p.atr
    base, _ = base_lot_and_cap(p)
    lot0 = base
    loss0 = p.primary_loss_usd if p.primary_loss_usd > 0 else sl_dist0 * lot0 * p.usd_per_point_per_lot
    cum_loss = loss0
    cum_vol = lot0
    total_loss = loss0
    rows.append({
        "step": 1,
        "role": "PRIMARY",
        "lot": lot0,
        "result": "LOSS",
        "pnl": -loss0,
        "cum_loss": cum_loss,
        "cum_vol": cum_vol,
        "total_loss": total_loss,
        "risk_usd": sl_dist0 * lot0 * p.usd_per_point_per_lot,
        "note": "ไม้แรก",
    })

    for i in range(1, n_steps):
        info = calc_lot(p, i, cum_loss, cum_vol, old_style=old_style)
        lot = info["lot"]
        sl_dist = info["sl_mult"] * p.atr
        loss = sl_dist * lot * p.usd_per_point_per_lot
        cum_loss += loss
        cum_vol += lot
        total_loss += loss
        halted = p.max_steps > 0 and (i + 1) >= p.max_steps
        rows.append({
            "step": i + 1,
            "role": "RECOVERY",
            "lot": lot,
            "result": "LOSS",
            "pnl": -loss,
            "cum_loss": cum_loss,
            "cum_vol": cum_vol,
            "total_loss": total_loss,
            "risk_usd": info["risk_usd"],
            "driver": info["driver"],
            "geo": info["geo"],
            "recover": info["recover"],
            "vol_floor": info["vol_floor"],
            "rec_cap": info["rec_cap"],
            "note": "HALT 30min" if halted else "",
        })
        if halted:
            break
    return rows


def fmt_rows_html(rows: list[dict], title: str) -> str:
    hdr = """<tr>
      <th>#</th><th>Role</th><th>Lot</th><th>SL risk $</th><th>P/L</th>
      <th>สะสมขาดทุน</th><th>Driver</th><th>หมายเหตุ</th>
    </tr>"""
    body = ""
    for r in rows:
        body += f"""<tr>
          <td>{r['step']}</td><td>{r.get('role','')}</td>
          <td><b>{r['lot']:.2f}</b></td>
          <td>${r.get('risk_usd',0):.0f}</td>
          <td class='neg'>${r['pnl']:+.0f}</td>
          <td class='neg'>${r['total_loss']:.0f}</td>
          <td>{r.get('driver','')}</td>
          <td>{r.get('note','')}</td>
        </tr>"""
    return f"<h3>{title}</h3><table>{hdr}{body}</table>"


def main() -> None:
    cfg = load_cfg()
    t = cfg.get("trading", {})
    r = cfg.get("recovery", {})
    a = cfg.get("account_scaling", {})
    c = cfg.get("commission", {})

    # เคสจริงจาก log ผู้ใช้ (~$1079, ATR~6.1)
    user_case = SimParams(
        name="เคสจริง (log 20 พ.ค.)",
        balance=1079.0,
        atr=6.14,
        sl_mult=float(t.get("sl_atr_mult", 0.8)),
        tp_mult=float(t.get("tp_atr_mult", 1.6)),
        sl_mult_recovery=float(r.get("sl_atr_mult_recovery", 1.25)),
        risk_pct=float(a.get("risk_per_trade_pct", 1.5)),
        max_lot_pct=float(a.get("max_lot_pct_of_balance", 25.0)),
        lot_mult=float(r.get("lot_multiplier", 1.25)),
        vol_mult=float(r.get("profit_volume_multiplier", 1.05)),
        max_rec_mult=float(r.get("max_recovery_lot_multiplier", 2.0)),
        max_steps=int(r.get("max_steps", 4)),
        commission=float(c.get("per_lot_round_trip_usd", 6.0)),
        primary_loss_usd=80.40,
    )

    current = SimParams(
        name="Config ปัจจุบัน",
        balance=1000.0,
        atr=6.0,
        sl_mult=float(t.get("sl_atr_mult", 0.8)),
        tp_mult=float(t.get("tp_atr_mult", 1.6)),
        sl_mult_recovery=float(r.get("sl_atr_mult_recovery", 1.25)),
        risk_pct=float(a.get("risk_per_trade_pct", 1.5)),
        max_lot_pct=float(a.get("max_lot_pct_of_balance", 25.0)),
        lot_mult=float(r.get("lot_multiplier", 1.25)),
        vol_mult=float(r.get("profit_volume_multiplier", 1.05)),
        max_rec_mult=float(r.get("max_recovery_lot_multiplier", 2.0)),
        max_steps=int(r.get("max_steps", 4)),
        commission=float(c.get("per_lot_round_trip_usd", 6.0)),
        primary_loss_usd=0.0,
    )

    old_bad = SimParams(
        name="แบบเก่า (อันตราย)",
        balance=1079.0,
        atr=6.14,
        sl_mult=0.8,
        tp_mult=1.6,
        sl_mult_recovery=0.8,
        risk_pct=1.5,
        max_lot_pct=25.0,
        lot_mult=1.5,
        vol_mult=1.2,
        max_rec_mult=0.0,
        max_steps=0,
        commission=6.0,
        primary_loss_usd=80.40,
    )

    scenarios = [
        (user_case, 6, False, "เคสจริง — config ใหม่ (max 6 ไม้ถ้าไม่ halt)"),
        (user_case, 6, True, "เคสจริง — แบบเก่า (ไม่มี cap, SL 0.8)"),
        (current, 5, False, "Balance $1000 ATR 6 — config ปัจจุบัน"),
        (old_bad, 8, True, "Balance $1079 — แบบเก่า 8 ไม้"),
    ]

    print("=" * 72)
    print("  RECOVERY SIMULATION — เคสแย่: โดน SL ทุกไม้")
    print("=" * 72)

    html_sections = ""
    for params, n, old, title in scenarios:
        rows = run_series(params, n, old_style=old)
        total = rows[-1]["total_loss"]
        pct = total / params.balance * 100
        print(f"\n--- {title} ---")
        print(f"  Balance ${params.balance:.0f} | ATR {params.atr} | "
              f"max_steps={params.max_steps} | rec_cap={params.max_rec_mult}x")
        for row in rows:
            extra = ""
            if row.get("driver"):
                extra = f" [{row['driver']}]"
            print(f"  ไม้ {row['step']:2d} {row.get('role',''):8s} "
                  f"lot={row['lot']:.2f}  SL~${row.get('risk_usd',0):6.0f}  "
                  f"PnL=${row['pnl']:+7.0f}  สะสม=${row['total_loss']:7.0f}{extra}  {row.get('note','')}")
        print(f"  >>> ขาดทุนรวม ${total:.0f} ({pct:.1f}% ของ balance)")
        if params.max_steps > 0 and len(rows) >= params.max_steps:
            print(f"  >>> หยุดที่ max_steps={params.max_steps} (halt 30 นาที)")
        html_sections += fmt_rows_html(rows, f"{title} — ขาดทุนรวม ${total:.0f} ({pct:.1f}%)")

    # Win ไม้สุดท้ายหลังเสีย 3 ไม้
    print("\n--- ถ้าไม้ที่ 4 ชนะ (config ปัจจุบัน) ---")
    cum_loss = 0.0
    cum_vol = 0.0
    for i in range(3):
        info = calc_lot(current, i, cum_loss, cum_vol, old_style=False)
        loss = info["sl_mult"] * current.atr * info["lot"] * current.usd_per_point_per_lot
        cum_loss += loss
        cum_vol += info["lot"]
        print(f"  ไม้ {i+1} LOSS lot={info['lot']:.2f} → ค้าง ${cum_loss:.0f}")
    info = calc_lot(current, 3, cum_loss, cum_vol, old_style=False)
    win = info["net_per_lot"] * info["lot"]
    print(f"  ไม้ 4 WIN  lot={info['lot']:.2f} → กำไร ~${win:.0f} | สุทธิ ${win - cum_loss:+.0f}")

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"recovery_sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    html = f"""<!DOCTYPE html>
<html lang="th"><head><meta charset="utf-8">
<title>Recovery Simulation</title>
<style>
body{{font-family:Segoe UI,sans-serif;background:#0f172a;color:#e2e8f0;padding:20px}}
h1{{font-size:20px}} h3{{margin-top:24px;color:#94a3b8}}
table{{width:100%;border-collapse:collapse;margin:8px 0 20px;font-size:13px}}
th,td{{padding:8px;border-bottom:1px solid #334155;text-align:left}}
th{{color:#64748b;font-size:11px;text-transform:uppercase}}
.neg{{color:#f87171;font-weight:600}}
.note{{color:#64748b;font-size:12px}}
</style></head><body>
<h1>Recovery Simulation — เคสแย่ (SL ทุกไม้)</h1>
<p class="note">Generated {ts} · XAUUSD ~$100/point/lot · สมมติโดน SL ทุกไม้</p>
{html_sections}
<p class="note">รันใหม่: python simulate_recovery.py</p>
</body></html>"""
    out.write_text(html, encoding="utf-8")
    print(f"\n[OK] HTML: {out}")
    try:
        webbrowser.open(out.resolve().as_uri())
    except Exception:
        pass


if __name__ == "__main__":
    main()
