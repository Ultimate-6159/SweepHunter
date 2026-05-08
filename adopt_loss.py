"""
♻️ adopt_loss.py — รับขาดทุนเก่าเข้ามาเป็น "series ค้าง" ให้ bot กู้คืน

ใช้กรณี:
  - ขาดทุนสะสมจากออเดอร์ manual หรือ bot ตัวเก่า
  - ต้องการให้ bot "พยายามกู้" ขาดทุนนี้ผ่าน Smart Recovery
  - มี floating loss ค้างใน MT5 ที่อยากให้ระบบรู้จัก

ทำอะไร:
  1. ดึงข้อมูล MT5: balance, equity, open positions ของ symbol
  2. ดึงขาดทุนสะสม (realized) จาก DB (ถ้ามี)
  3. แสดงตัวเลือก: รับขาดทุนเท่าไหร่ + เป็นทิศ BUY หรือ SELL
  4. ตรวจความปลอดภัย — ปฏิเสธถ้าเกิน max_adopt_pct (default 20% balance)
  5. สร้าง series ใหม่ใน DB + สร้าง "synthetic decision" บันทึกขาดทุน
  6. อัพเดท recovery_state.json
  7. พอ start bot → จะเห็นเป็น series ค้าง + พยายามกู้ตามปกติ

⚠️  คำเตือน:
  - การกู้ทุนก้อนใหญ่ = lot ที่ต้องเปิดจะใหญ่ตาม → ระเบิดได้
  - เราคำนวณให้ดูก่อนว่า lot recovery จะใหญ่แค่ไหน
  - ถ้าใหญ่เกิน max_lot_cap → จะกู้ไม่หมดในครั้งเดียว
"""
from __future__ import annotations
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import MetaTrader5 as mt5

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core.config import Config  # noqa: E402
from core.paths import db_path, data_path  # noqa: E402

# ============================================================ Safety limits
MAX_ADOPT_PCT_OF_BALANCE = 20.0   # ปฏิเสธถ้ารับขาดทุนเกิน 20% balance
RECOMMEND_ADOPT_PCT = 8.0         # แนะนำไม่เกิน 8%


def banner():
    print("=" * 70)
    print("♻️  SweepHunter — Adopt Existing Loss as Recovery Series")
    print("=" * 70)


def connect_mt5() -> bool:
    cfg = Config.section("mt5")
    init_kwargs = {}
    if cfg.get("terminal_path"):
        init_kwargs["path"] = cfg["terminal_path"]
    if cfg.get("login"):
        init_kwargs.update({
            "login": int(cfg["login"]),
            "password": cfg.get("password", ""),
            "server": cfg.get("server", ""),
        })
    if not mt5.initialize(**init_kwargs):
        print(f"❌ MT5 connect failed: {mt5.last_error()}")
        return False
    return True


def show_account_info(symbol: str) -> dict:
    acc = mt5.account_info()
    if not acc:
        print("❌ ไม่ได้ข้อมูลบัญชี")
        return {}
    print(f"\n💼 Account: {acc.login} @ {acc.server}")
    print(f"   Balance: ${acc.balance:.2f}")
    print(f"   Equity:  ${acc.equity:.2f}")
    print(f"   Floating P/L: ${acc.equity - acc.balance:+.2f}")

    # Open positions on this symbol
    positions = mt5.positions_get(symbol=symbol) or []
    floating_loss_buy = 0.0
    floating_loss_sell = 0.0
    if positions:
        print(f"\n📊 Open positions on {symbol}: {len(positions)} ไม้")
        for p in positions:
            side = "BUY" if p.type == mt5.POSITION_TYPE_BUY else "SELL"
            print(f"   #{p.ticket}  {side} {p.volume:.2f} lot @ {p.price_open:.3f}  "
                  f"P/L=${p.profit:+.2f}  magic={p.magic}")
            if p.profit < 0:
                if side == "BUY":
                    floating_loss_buy += abs(p.profit)
                else:
                    floating_loss_sell += abs(p.profit)
    else:
        print(f"\n📊 ไม่มี open position บน {symbol}")
    return {
        "balance": acc.balance,
        "equity": acc.equity,
        "floating_loss_buy": floating_loss_buy,
        "floating_loss_sell": floating_loss_sell,
        "positions": list(positions),
    }


def get_realized_loss_from_db(symbol: str) -> float:
    fname = Config.section("database").get("filename", "hyper_trades.sqlite")
    path = db_path(fname)
    if not path.exists():
        return 0.0
    with sqlite3.connect(str(path)) as conn:
        row = conn.execute(
            "SELECT COALESCE(SUM(pnl), 0) FROM decisions "
            "WHERE symbol=? AND status IN ('WIN','LOSS')",
            (symbol,)).fetchone()
    return float(row[0] or 0.0)


def estimate_recovery_lot(loss_usd: float, balance: float, base_lot: float = 0.01) -> dict:
    """ประมาณ lot ที่ต้องเปิดเพื่อกู้ขาดทุน (TP=$2 ต่อ 0.01 lot โดยเฉลี่ย)."""
    # ทอง: 1 lot ≈ $1 ต่อ pip; TP = ~26 pips (1.6×ATR ปกติ ~16) → ~$26 ต่อ lot
    # 0.01 lot → ~$0.26 ต่อ pip → TP ≈ ~$2.5 ต่อไม้
    # หา lot ที่จะกู้ได้ใน 1 ไม้
    profit_per_001_lot = 2.5
    lot_needed = (loss_usd / profit_per_001_lot) * 0.01
    return {
        "lot_to_recover_in_one_trade": lot_needed,
        "is_realistic": lot_needed <= 0.30,  # > 0.30 lot = risky บนพอร์ตเล็ก
        "scenarios": [
            {"steps": 1, "lot": lot_needed,        "comment": "กู้ใน 1 ไม้ (lot ใหญ่)"},
            {"steps": 3, "lot": lot_needed / 2.5,  "comment": "กู้ใน 3 ไม้ recovery"},
            {"steps": 5, "lot": lot_needed / 4.5,  "comment": "กู้ใน 5 ไม้ (ปลอดภัยสุด)"},
        ],
    }


def adopt(symbol: str, side: str, loss_usd: float, balance: float,
          dry_run: bool = False) -> bool:
    """สร้าง series + decision (synthetic LOSS) ใน DB + อัพเดท recovery_state.json."""
    fname = Config.section("database").get("filename", "hyper_trades.sqlite")
    path = db_path(fname)
    state_path = data_path("recovery_state.json")

    if dry_run:
        print("\n🧪 DRY RUN — ไม่บันทึกอะไร")
        return True

    now_iso = datetime.now(timezone.utc).isoformat()
    side_upper = side.upper()

    # 1. Insert synthetic series
    with sqlite3.connect(str(path)) as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO series (opened_at_utc, symbol, side, steps, status, notes) "
            "VALUES (?, ?, ?, 1, 'OPEN', 'ADOPTED_LOSS')",
            (now_iso, symbol, side_upper))
        sid = cur.lastrowid
        # 2. Insert synthetic LOSS decision (placeholder ticket id = -sid เพื่อกัน clash)
        cur.execute(
            "INSERT INTO decisions (ts_utc, series_id, step, symbol, timeframe, "
            "prediction, confidence, role, ticket, volume, status, pnl, "
            "closed_at_utc, notes) VALUES (?, ?, 1, ?, 'M5', ?, 0.0, 'PRIMARY', ?, "
            "0.01, 'LOSS', ?, ?, 'adopted: pre-existing loss imported')",
            (now_iso, sid, symbol, 2 if side_upper == "BUY" else 0,
             -int(sid), -float(loss_usd), now_iso))
        conn.commit()

    # 3. Update recovery_state.json
    state = {
        "cumulative_loss_usd": float(loss_usd),
        "cumulative_losing_volume": 0.01,  # nominal
        "consecutive_losses": 1,
        "series_id": sid,
        "last_side": side_upper,
        "processed_tickets": [-int(sid)],
        "halted_until_ts": 0.0,
        "last_loss_side": side_upper,
        "consec_same_dir_losses": 1,
        "blocked_side": None,
        "block_side_until_ts": 0.0,
        "pause_resume_count": 0,
    }
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    print(f"\n✅ บันทึกเรียบร้อย:")
    print(f"   📌 New series #{sid} status=OPEN")
    print(f"   💔 Synthetic LOSS decision: ${loss_usd:.2f} ({side_upper})")
    print(f"   ♻️ recovery_state.json updated → cumulative_loss=${loss_usd:.2f}")
    return True


def main() -> int:
    banner()
    cfg_t = Config.section("trading")
    symbol = cfg_t["symbol"]
    print(f"\n🎯 Symbol: {symbol}")

    if not connect_mt5():
        return 1
    try:
        info = show_account_info(symbol)
        if not info:
            return 1

        realized = get_realized_loss_from_db(symbol)
        print(f"\n📚 Realized P/L from DB: ${realized:+.2f}")
        print(f"   (หากเป็นลบ = ขาดทุนสะสมจากการเทรดของ bot ที่ผ่านมา)")

        # ============================================================ ASK USER
        print("\n" + "─" * 70)
        print("💡 จำนวนที่จะรับเข้า series:")
        suggested = max(0, -realized)
        if suggested > 0:
            print(f"   - แนะนำ (= |realized loss จาก DB|): ${suggested:.2f}")
        if info['floating_loss_buy'] > 0:
            print(f"   - Floating loss BUY ค้าง: ${info['floating_loss_buy']:.2f}")
        if info['floating_loss_sell'] > 0:
            print(f"   - Floating loss SELL ค้าง: ${info['floating_loss_sell']:.2f}")
        print(f"   - แนะนำสูงสุด ({RECOMMEND_ADOPT_PCT}% balance): "
              f"${info['balance'] * RECOMMEND_ADOPT_PCT / 100:.2f}")
        print(f"   - LIMIT สูงสุด ({MAX_ADOPT_PCT_OF_BALANCE}% balance): "
              f"${info['balance'] * MAX_ADOPT_PCT_OF_BALANCE / 100:.2f}")

        amt_str = input("\n💰 จะรับขาดทุนเข้า series เท่าไหร่ ($) [0=ยกเลิก]: ").strip()
        try:
            loss_usd = float(amt_str)
        except ValueError:
            print("❌ ตัวเลขไม่ถูกต้อง")
            return 1
        if loss_usd <= 0:
            print("❌ ยกเลิก")
            return 1

        # Safety check
        pct = (loss_usd / info['balance']) * 100
        max_allowed = info['balance'] * MAX_ADOPT_PCT_OF_BALANCE / 100
        if loss_usd > max_allowed:
            print(f"\n⛔ REJECTED: ${loss_usd:.2f} = {pct:.1f}% balance "
                  f"เกินขีดจำกัด {MAX_ADOPT_PCT_OF_BALANCE}% (${max_allowed:.2f})")
            print("   เหตุผล: lot ที่ต้องเปิดเพื่อกู้จะใหญ่เกินไป → ระเบิดได้")
            return 1
        if pct > RECOMMEND_ADOPT_PCT:
            print(f"\n⚠️  คำเตือน: ${loss_usd:.2f} = {pct:.1f}% balance "
                  f"(สูงกว่าแนะนำ {RECOMMEND_ADOPT_PCT}%)")

        side = input("\n📊 ทิศทางขาดทุน BUY หรือ SELL? [B/S]: ").strip().upper()
        if side in ("B", "BUY"):
            side = "BUY"
        elif side in ("S", "SELL"):
            side = "SELL"
        else:
            print("❌ ทิศไม่ถูกต้อง")
            return 1

        # Show recovery scenarios
        scenarios = estimate_recovery_lot(loss_usd, info['balance'])
        cap = float(Config.section("recovery").get("max_lot_cap", 0.15))
        print(f"\n📐 ประมาณ lot ที่ต้องใช้กู้ (max_lot_cap={cap}):")
        for s in scenarios['scenarios']:
            warn = "❌ เกิน cap!" if s['lot'] > cap else "✅"
            print(f"   {warn} {s['comment']:<40s} lot ≈ {s['lot']:.3f}")

        if not scenarios['is_realistic']:
            print(f"\n⚠️  Lot ที่จะกู้ใน 1 ไม้ ({scenarios['lot_to_recover_in_one_trade']:.3f}) "
                  f"ใหญ่กว่า 0.30 — ต้องอาศัย max_steps สูงเพื่อกู้คืน")

        # Final confirm
        print("\n" + "═" * 70)
        print(f"📋 สรุป: จะสร้าง series ใหม่กับ:")
        print(f"   - cumulative_loss = ${loss_usd:.2f}")
        print(f"   - last_side = {side}")
        print(f"   - bot จะพยายามกู้ใน max_steps = "
              f"{Config.section('recovery').get('max_steps', 5)} ไม้")
        ans = input("\nพิมพ์ 'YES' เพื่อยืนยัน: ").strip()
        if ans != "YES":
            print("❌ ยกเลิก")
            return 1

        adopt(symbol, side, loss_usd, info['balance'])

        print("\n" + "═" * 70)
        print("🚀 ขั้นตอนต่อไป:")
        print("═" * 70)
        print("   1. หยุด bot ที่รันอยู่ (ถ้ามี) — Ctrl+C")
        print("   2. python run.py bot")
        print("   3. ดู log: 'series #N | RECOVERY | lot=...' = ระบบเริ่มกู้แล้ว")
        print("\n💡 ถ้าอยาก undo: ลบ data/recovery_state.json แล้ว restart bot")
        return 0
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    sys.exit(main())
