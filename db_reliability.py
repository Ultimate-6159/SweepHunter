# -*- coding: utf-8 -*-
"""
db_reliability.py — รายงานความน่าเชื่อถือข้อมูลใน hyper_trades.sqlite

Usage:
  python db_reliability.py
  python db_reliability.py --snapshot 26   # โฟกัส snapshot เดียว
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DB = ROOT / "data" / "db" / "hyper_trades.sqlite"
MIN_TRADES_SLOT = 8
FOCUS_SNAP_MIN = 14  # ยุด SL=0.8 risk=1.5% conf=0.4 (config ปัจจุบันโดยประมาณ)


def pf(gw: float, gl: float) -> float:
    if gl > 0:
        return gw / gl
    return 999.0 if gw > 0 else 0.0


def stars(n: int, thresholds: tuple[int, int, int]) -> str:
    """n trades → ★ rating"""
    if n >= thresholds[2]:
        return "★★★★☆"
    if n >= thresholds[1]:
        return "★★★☆☆"
    if n >= thresholds[0]:
        return "★★☆☆☆"
    return "★☆☆☆☆"


def main() -> int:
    parser = argparse.ArgumentParser(description="DB reliability report")
    parser.add_argument("--snapshot", type=int, default=0,
                        help="โฟกัส snapshot id (0 = แสดงทั้งหมด)")
    args = parser.parse_args()

    if not DB.exists():
        print(f"ERROR: ไม่พบ {DB}")
        return 1

    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row

    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()]

    total = conn.execute(
        "SELECT COUNT(*) FROM decisions WHERE status IN ('WIN','LOSS')"
    ).fetchone()[0]
    with_snap = conn.execute(
        "SELECT COUNT(*) FROM decisions WHERE status IN ('WIN','LOSS') "
        "AND config_snapshot_id IS NOT NULL"
    ).fetchone()[0]
    no_snap = total - with_snap
    open_n = conn.execute(
        "SELECT COUNT(*) FROM decisions WHERE status IN ('OPEN','PENDING')"
    ).fetchone()[0]
    series_n = conn.execute("SELECT COUNT(*) FROM series").fetchone()[0]
    snap_count = conn.execute("SELECT COUNT(*) FROM config_snapshots").fetchone()[0]

    SEP = "=" * 72
    print(SEP)
    print("  SweepHunter — ความน่าเชื่อถือข้อมูลใน hyper_trades.sqlite")
    print(SEP)
    print(f"  ไฟล์     : {DB}")
    print(f"  Tables   : {', '.join(tables)}")
    print(f"  เทรดปิด  : {total} ไม้ (WIN/LOSS)")
    print(f"    ├─ มี config_snapshot : {with_snap} ({100 * with_snap / max(total, 1):.1f}%)")
    print(f"    └─ ไม่มี snapshot (เก่า): {no_snap} ({100 * no_snap / max(total, 1):.1f}%)")
    print(f"  เปิดค้าง  : {open_n}  |  series: {series_n}  |  config versions: {snap_count}")

    # ── stats per snapshot ───────────────────────────────────────────────
    stats: dict[int, dict] = {}
    for row in conn.execute("""
        SELECT config_snapshot_id sid,
               COUNT(*) n,
               SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) wins,
               SUM(pnl) pnl,
               SUM(CASE WHEN pnl > 0 THEN pnl ELSE 0 END) gw,
               SUM(CASE WHEN pnl < 0 THEN ABS(pnl) ELSE 0 END) gl,
               MIN(ts_utc) first_ts,
               MAX(ts_utc) last_ts
        FROM decisions
        WHERE status IN ('WIN','LOSS') AND config_snapshot_id IS NOT NULL
        GROUP BY config_snapshot_id
    """):
        stats[row["sid"]] = dict(row)

    snaps = conn.execute("""
        SELECT id, ts_utc, label, config_hash,
               risk_per_trade_pct, sl_atr_mult, tp_atr_mult,
               smart_trailing_enabled, hourly_lot_mult_enabled,
               full_config_json
        FROM config_snapshots ORDER BY id
    """).fetchall()

    if args.snapshot:
        snaps = [s for s in snaps if s["id"] == args.snapshot]
        if not snaps:
            print(f"\n  ERROR: ไม่พบ snapshot #{args.snapshot}")
            conn.close()
            return 1

    print("\n" + "-" * 72)
    print("  เวอร์ชัน config — แต่ละ snapshot ต่างกันอย่างไร")
    print("-" * 72)
    print("  คอลัมน์: risk=%%balance/ไม้  SL/TP=×ATR  trail=smart_trailing  hmul=hourly lot")
    print("-" * 72)

    prev_fields: dict | None = None
    prev_id: int | None = None

    for s in snaps:
        sid = s["id"]
        st = stats.get(sid, {})
        n = int(st.get("n") or 0)
        wr = 100 * st["wins"] / n if n else 0.0
        pnl = float(st.get("pnl") or 0)
        gpf = pf(float(st.get("gw") or 0), float(st.get("gl") or 0)) if n else 0.0

        cur = {
            "risk": s["risk_per_trade_pct"],
            "sl": s["sl_atr_mult"],
            "tp": s["tp_atr_mult"],
            "trail": s["smart_trailing_enabled"],
            "hmul": s["hourly_lot_mult_enabled"],
        }
        diff = ""
        if prev_fields is not None:
            parts = []
            for k, label in [("risk", "risk"), ("sl", "SL"), ("tp", "TP"),
                             ("trail", "trail"), ("hmul", "hmul")]:
                if cur[k] != prev_fields[k]:
                    parts.append(f"{label}:{prev_fields[k]}→{cur[k]}")
            diff = " | ".join(parts) if parts else "(hash/label เปลี่ยน — key fields เท่าเดิม)"

        label = (s["label"] or "")[:58]
        ts = (s["ts_utc"] or "")[:19]
        rel = stars(n, (1, MIN_TRADES_SLOT, 30))

        print(f"\n  [#{sid:2d}] {ts}  {rel}")
        print(f"       trades={n:3d}  PnL=${pnl:+9.2f}  WR={wr:4.0f}%  PF={gpf:.2f}")
        print(f"       risk={cur['risk']}%  SL={cur['sl']}×  TP={cur['tp']}×  "
              f"trail={cur['trail']}  hmul={cur['hmul']}")
        print(f"       hash={s['config_hash']}  |  {label}")
        if diff:
            print(f"       Δ vs #{prev_id}: {diff}")

        prev_fields = cur
        prev_id = sid

    # NULL era
    if not args.snapshot:
        ns = conn.execute("""
            SELECT COUNT(*) n,
                   SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) wins,
                   SUM(pnl) pnl,
                   SUM(CASE WHEN pnl > 0 THEN pnl ELSE 0 END) gw,
                   SUM(CASE WHEN pnl < 0 THEN ABS(pnl) ELSE 0 END) gl,
                   MIN(ts_utc) first_ts, MAX(ts_utc) last_ts
            FROM decisions
            WHERE status IN ('WIN','LOSS') AND config_snapshot_id IS NULL
        """).fetchone()
        if ns["n"]:
            wr = 100 * ns["wins"] / ns["n"]
            gpf = pf(ns["gw"] or 0, ns["gl"] or 0)
            print(f"\n  [#NULL] ★☆☆☆☆  ยุคก่อนมี snapshot")
            print(f"       trades={ns['n']:3d}  PnL=${ns['pnl']:+9.2f}  WR={wr:.0f}%  PF={gpf:.2f}")
            print(f"       ช่วง {str(ns['first_ts'])[:19]} → {str(ns['last_ts'])[:19]}")
            print("       ⚠ ไม่ทราบ config — อย่าใช้สรุปยุดปัจจุบัน")

    # ── 5 ยุดหลัก ────────────────────────────────────────────────────────
    if not args.snapshot:
        print("\n" + "-" * 72)
        print("  สรุป 5 ยุดหลัก (รวม trades)")
        print("-" * 72)
        eras = [
            ("NULL (ไม่มี snapshot)", "config_snapshot_id IS NULL"),
            ("SL=1.2× ยุดแรก", "config_snapshot_id IN (2,3,4,5,6,7,8)"),
            ("SL=0.8 risk=2%", "config_snapshot_id IN (9,10,11,12)"),
            ("SL=0.8 risk=0.3%", "config_snapshot_id = 13"),
            (f"SL=0.8 risk=1.5% (#{FOCUS_SNAP_MIN}+ ปัจจุบัน)",
             f"config_snapshot_id >= {FOCUS_SNAP_MIN}"),
        ]
        print(f"  {'ยุด':<28} | {'ไม้':>5} | {'PnL':>10} | {'PF':>6} | {'WR':>5}")
        print(f"  {'-'*28}-+-{'-'*5}-+-{'-'*10}-+-{'-'*6}-+-{'-'*5}")
        for name, where in eras:
            row = conn.execute(
                "SELECT COUNT(*) n, SUM(pnl) p, "
                "SUM(CASE WHEN pnl>0 THEN 1 ELSE 0 END) w, "
                "SUM(CASE WHEN pnl>0 THEN pnl ELSE 0 END) gw, "
                "SUM(CASE WHEN pnl<0 THEN ABS(pnl) ELSE 0 END) gl "
                "FROM decisions WHERE status IN ('WIN','LOSS') AND (" + where + ")"
            ).fetchone()
            n = row["n"] or 0
            pnl = row["p"] or 0
            wr = 100 * row["w"] / n if n else 0
            gpf = pf(row["gw"] or 0, row["gl"] or 0)
            print(f"  {name:<28} | {n:5d} | ${pnl:+9.2f} | {gpf:6.2f} | {wr:4.0f}%")

    # ── retrain ──────────────────────────────────────────────────────────
    print("\n" + "-" * 72)
    print("  ประวัติ retrain โมเดล (5 ล่าสุด)")
    print("-" * 72)
    try:
        retrains = conn.execute("""
            SELECT ts_utc, accepted, cv_acc, oos_acc, rows_trained, notes
            FROM model_retrains ORDER BY id DESC LIMIT 5
        """).fetchall()
        tot_r = conn.execute(
            "SELECT COUNT(*) n, SUM(accepted) acc FROM model_retrains"
        ).fetchone()
        print(f"  รวม {tot_r['n']} ครั้ง (accepted {tot_r['acc'] or 0})")
        for r in retrains:
            mark = "✓" if r["accepted"] else "✗"
            oos = r["oos_acc"] or 0
            cv = r["cv_acc"] or 0
            notes = (r["notes"] or "")[:45]
            print(f"  {mark} {(r['ts_utc'] or '')[:19]}  rows={r['rows_trained']}  "
                  f"CV={cv:.3f} OOS={oos:.3f}  {notes}")
    except sqlite3.OperationalError:
        print("  (ไม่มีตาราง model_retrains)")

    # ── config ปัจจุบัน (snapshot ล่าสุด) ───────────────────────────────
    latest = conn.execute("SELECT MAX(id) FROM config_snapshots").fetchone()[0]
    if latest:
        row = conn.execute(
            "SELECT full_config_json FROM config_snapshots WHERE id=?", (latest,)
        ).fetchone()
        st = stats.get(latest, {})
        n_cur = int(st.get("n") or 0)
        try:
            fc = json.loads(row["full_config_json"] or "{}")
            hf = fc.get("hyper_frequency", {})
            rec = fc.get("recovery", {})
            sw = fc.get("session_weighting", {})
            print("\n" + "-" * 72)
            print(f"  Config ปัจจุบัน (snapshot #{latest}, {n_cur} ไม้ใน DB)")
            print("-" * 72)
            print(f"    min_confidence     = {hf.get('min_confidence')}")
            print(f"    recovery max_steps = {rec.get('max_steps')}")
            print(f"    recovery spread    = {rec.get('recovery_spread_trades')}")
            print(f"    risk_per_trade     = {fc.get('account_scaling', {}).get('risk_per_trade_pct')}%")
            print(f"    blocked_slots      = {len(sw.get('blocked_slots') or [])} วัน×ชม.")
        except (json.JSONDecodeError, TypeError):
            pass

    # ── คำแนะนำ ──────────────────────────────────────────────────────────
    print("\n" + "-" * 72)
    print("  ความน่าเชื่อถือตามการใช้งาน")
    print("-" * 72)
    focus_row = conn.execute(
        f"SELECT COUNT(*) n FROM decisions WHERE status IN ('WIN','LOSS') "
        f"AND config_snapshot_id >= {FOCUS_SNAP_MIN}"
    ).fetchone()[0]
    tips = [
        ("analyze_slots (PF block ชั่วโมง)", "★★★★☆",
         f"ใช้ข้อมูลรวม {total} ไม้ได้ — PF ไม่ขึ้นกับ lot"),
        (f"วิเคราะห์ยุด config ปัจจุบัน (#{FOCUS_SNAP_MIN}+)",
         stars(focus_row, (30, 100, 300)),
         f"มี {focus_row} ไม้ — แนะนำ ≥300 ก่อนตัดสิน block ใหม่"),
        ("เปรียบ WR ข้าม snapshot", "★★☆☆☆",
         "lot/risk ต่างกัน — ใช้ PF/PnL แทน WR"),
        ("ยุค NULL (ไม่มี snapshot)", "★☆☆☆☆",
         f"{no_snap} ไม้ — ไม่ทราบ config"),
    ]
    for use, rating, note in tips:
        print(f"  {rating}  {use}")
        print(f"         → {note}")

    print("\n" + SEP)
    print("  รันซ้ำ: db_reliability.bat  |  โฟกัส snapshot: python db_reliability.py --snapshot 26")
    print(SEP)

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
