"""
📊 compare_strategies.py — เปรียบเทียบผลงานของแต่ละ config snapshot

ใช้: python compare_strategies.py
จะแสดงตารางเปรียบเทียบทุก snapshot:
  - วันที่ใช้
  - จำนวน trade
  - WR / Net PnL / Avg WIN/LOSS / RR
  - ผลลัพธ์ (ดีกว่า/แย่กว่า config ก่อนหน้า)
"""
from __future__ import annotations
import sqlite3
from pathlib import Path

DB = Path(__file__).resolve().parent / "data" / "db" / "hyper_trades.sqlite"


def main() -> int:
    if not DB.exists():
        print(f"❌ {DB} not found")
        return 1
    conn = sqlite3.connect(str(DB))
    conn.row_factory = sqlite3.Row

    snaps = conn.execute("""
        SELECT id, ts_utc, label, risk_per_trade_pct, sl_atr_mult, tp_atr_mult,
               smart_trailing_enabled, be_trigger_atr,
               hourly_lot_mult_enabled
        FROM config_snapshots ORDER BY ts_utc
    """).fetchall()

    if not snaps:
        print("⚠️  ยังไม่มี snapshot — รัน 'python snapshot_config.py \"label\"' ก่อน")
        return 1

    # คำนวณช่วง timestamp ของแต่ละ snapshot (ts จากตัวนี้ → ts ของตัวถัดไป)
    print("=" * 100)
    print(f"  📊 STRATEGY COMPARISON — {len(snaps)} snapshots")
    print("=" * 100)
    print(f"  {'ID':<3} {'Date':<11} {'Label':<28} {'risk':<5} {'SL/TP':<8} {'Trail':<6} {'HMul':<5} "
          f"{'N':<4} {'WR%':<5} {'Win$':<6} {'Loss$':<7} {'RR':<5} {'Net':<8} {'$/tr':<6}")
    print("  " + "-" * 98)

    snap_list = list(snaps)
    for i, s in enumerate(snap_list):
        ts_from = s["ts_utc"]
        ts_to = snap_list[i+1]["ts_utc"] if i+1 < len(snap_list) else "9999-12-31"

        # ดึงสถิติของ trades ในช่วงนั้น
        row = conn.execute("""
            SELECT COUNT(*) n,
                   SUM(CASE WHEN status='WIN' THEN 1 ELSE 0 END) w_n,
                   AVG(CASE WHEN status='WIN' AND pnl > 0 THEN pnl END) avg_w,
                   AVG(CASE WHEN status='LOSS' AND pnl < 0 THEN pnl END) avg_l,
                   SUM(pnl) net
            FROM decisions
            WHERE status IN ('WIN','LOSS')
              AND ts_utc >= ? AND ts_utc < ?
        """, (ts_from, ts_to)).fetchone()
        n = row["n"] or 0
        w_n = row["w_n"] or 0
        wr = w_n/n*100 if n else 0
        avg_w = row["avg_w"] or 0
        avg_l = row["avg_l"] or 0
        rr = avg_w/abs(avg_l) if avg_l else 0
        net = row["net"] or 0
        per_tr = net/n if n else 0

        date = s["ts_utc"][:10]
        label = (s["label"] or "")[:27]
        sltp = f"{s['sl_atr_mult'] or 0}/{s['tp_atr_mult'] or 0}"
        trail = "ON" if s["smart_trailing_enabled"] else "OFF"
        hmul = "Y" if s["hourly_lot_mult_enabled"] else "N"
        risk = f"{s['risk_per_trade_pct'] or 0}%"

        flag = ""
        if i > 0 and n > 10:
            prev_per_tr = float(snap_list[i-1].get("_per_tr", per_tr) if hasattr(snap_list[i-1], "get") else per_tr)
            if per_tr > prev_per_tr * 1.2: flag = " 🚀"
            elif per_tr < prev_per_tr * 0.8: flag = " 📉"

        print(f"  {s['id']:<3} {date:<11} {label:<28} {risk:<5} {sltp:<8} {trail:<6} {hmul:<5} "
              f"{n:<4} {wr:<5.1f} ${avg_w:<+5.2f} ${avg_l:<+6.2f} {rr:<5.2f} ${net:<+7.2f} ${per_tr:<+5.2f}{flag}")

    print()
    print("  💡 อ่านยังไง:")
    print("     - $/tr (ต่อไม้) สูงสุด = strategy ดีสุด")
    print("     - RR สูง + WR > 50% = healthy strategy")
    print("     - 🚀 = ดีกว่า snapshot ก่อนหน้า > 20%")
    print("     - 📉 = แย่กว่า snapshot ก่อนหน้า > 20%")
    return 0


if __name__ == "__main__":
    main()
