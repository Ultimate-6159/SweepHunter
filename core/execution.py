"""
execution.py
=============
Robust MT5 execution สำหรับ Hyper-Frequency Micro-Scalping.
- IOC fill, deviation 30 points (SRD)
- Aggressive retry: สูงสุด 5 ครั้ง × 0.2s
- Dynamic Spread Allowance (SRD): ยอมให้เทรดถ้า spread <= 2.0 × MA(60min) หรือ hard cap 45p
- รองรับ SL=0 / TP=0 (martingale steps ไม่ตั้ง SL/TP เอง — จัดการที่ series level)
"""
from __future__ import annotations
import time
from collections import deque
from dataclasses import dataclass
from threading import RLock
from typing import Deque, Optional, Tuple

import MetaTrader5 as mt5

from .config import Config
from .logger import get_logger
from .mt5_connector import MT5Connector, SymbolSpec

log = get_logger("exec")


# ===================================================== Dynamic Spread Tracker
class DynamicSpreadTracker:
    """
    Rolling MA ของ spread ใน N นาทีล่าสุด (SRD: 60 นาที).
    เก็บ (timestamp, spread_points). ใช้สำหรับ spread_guard().
    """

    def __init__(self) -> None:
        cfg = Config.section("dynamic_spread")
        self.window_sec = float(cfg.get("rolling_window_minutes", 60)) * 60
        self._buf: Deque[Tuple[float, float]] = deque()
        self._lock = RLock()

    def update(self, spread_points: float) -> None:
        now = time.time()
        with self._lock:
            self._buf.append((now, float(spread_points)))
            cutoff = now - self.window_sec
            while self._buf and self._buf[0][0] < cutoff:
                self._buf.popleft()

    def average(self) -> Optional[float]:
        with self._lock:
            if not self._buf:
                return None
            return sum(s for _, s in self._buf) / len(self._buf)

    def samples(self) -> int:
        with self._lock:
            return len(self._buf)


def spread_guard(spec: SymbolSpec, current_spread_points: float,
                 tracker: Optional[DynamicSpreadTracker] = None) -> Tuple[bool, str]:
    """
    SRD Dynamic Spread Allowance:
      - hard_max_points (default 45): กรณีไหนก็ห้ามเกิน
      - ถ้ามี tracker + samples >= 30 -> ใช้ avg * multiplier
      - ถ้าไม่มี baseline -> fallback_max_points (default 30)

    Auto-scale: โบรก 3-digit / 5-digit (fractional pip) จะคูณ 10 อัตโนมัติ
    เพราะ 1 pip = 10 broker-points ในโบรกประเภทนี้
    """
    cfg = Config.section("dynamic_spread")
    pip_factor = 10 if spec.digits in (3, 5) else 1
    hard = float(cfg.get("hard_max_points", 45)) * pip_factor
    if current_spread_points > hard:
        return False, f"spread {current_spread_points:.1f} > hard cap {hard:.0f}"

    if tracker is not None and tracker.samples() >= 30:
        avg = tracker.average()
        mult = float(cfg.get("max_multiplier_of_avg", 2.0))
        limit = max(avg * mult, 5.0 * pip_factor)
        if current_spread_points > limit:
            return False, f"spread {current_spread_points:.1f} > {mult}× avg ({avg:.1f}) = {limit:.1f}"
        return True, f"OK ({current_spread_points:.1f} <= {limit:.1f}, avg={avg:.1f})"

    fallback = float(cfg.get("fallback_max_points", 30)) * pip_factor
    if current_spread_points > fallback:
        return False, f"spread {current_spread_points:.1f} > fallback {fallback:.0f} (no baseline yet)"
    return True, f"OK ({current_spread_points:.1f} <= fallback {fallback:.0f})"


# ===================================================== Order types
@dataclass
class ExecutionResult:
    ok: bool
    ticket: Optional[int]
    retcode: int
    comment: str
    price: float
    sl: float
    tp: float
    spread_points: float


# ===================================================== Commission helper
def commission_price_offset(spec: SymbolSpec) -> float:
    """
    คำนวณ "ส่วนต่างราคา" ที่ต้องเพิ่มเพื่อชดเชย commission round-trip ต่อ 1 lot.
    คืนค่าเป็น price units (เช่น 0.07 สำหรับ XAUUSD ที่ commission $7 ต่อ lot).

    Formula:
      profit_per_price_per_lot = tick_value / tick_size   (USD per 1.0 of price, per 1 lot)
      offset                   = commission_usd / profit_per_price_per_lot
    """
    cfg = Config.section("commission") or {}
    com_usd = float(cfg.get("per_lot_round_trip_usd", 0.0))
    if com_usd <= 0 or spec.trade_tick_size <= 0 or spec.trade_tick_value <= 0:
        return 0.0
    profit_per_price = spec.trade_tick_value / spec.trade_tick_size
    if profit_per_price <= 0:
        return 0.0
    return com_usd / profit_per_price


# ===================================================== open_market_order
def open_market_order(
    symbol: str,
    side: str,
    volume: float,
    atr_value: float,
    magic: int,
    comment: str = "Hyper",
    spread_tracker: Optional[DynamicSpreadTracker] = None,
    sl_atr_mult: float = 0.0,
    tp_atr_mult: float = 0.0,
) -> ExecutionResult:
    """
    SRD:
      - ORDER_FILLING_IOC, deviation 30 points
      - Retry 5 ครั้ง × 0.2s เมื่อ requote/price changed/off
      - sl_atr_mult/tp_atr_mult = 0 -> ไม่ตั้งที่ฝั่ง MT5 (Martingale series จัดเอง)
    """
    cfg_v = Config.section("vulnerability_patches")
    spec = MT5Connector.get_symbol_spec(symbol)
    volume = spec.normalize_volume(volume)

    max_dev = int(cfg_v.get("max_deviation_points", 30))
    retries = int(cfg_v.get("requote_retry_count", 5))
    delay = float(cfg_v.get("requote_retry_delay_sec", 0.2))

    side_u = side.upper()
    if side_u not in ("BUY", "SELL"):
        return ExecutionResult(False, None, -1, f"invalid side {side}", 0, 0, 0, 0)

    min_stop_price = max(spec.stops_level, 1) * spec.point
    sl_distance = max(sl_atr_mult * atr_value, min_stop_price) if sl_atr_mult > 0 else 0.0
    tp_distance = max(tp_atr_mult * atr_value, min_stop_price) if tp_atr_mult > 0 else 0.0

    last = ExecutionResult(False, None, -1, "no attempt", 0, 0, 0, 0)
    for attempt in range(1, retries + 1):
        tick = MT5Connector.get_tick(symbol)
        if tick is None or tick.ask <= 0 or tick.bid <= 0:
            time.sleep(delay)
            continue
        cur_spread = (tick.ask - tick.bid) / spec.point
        if spread_tracker is not None:
            spread_tracker.update(cur_spread)

        ok, reason = spread_guard(spec, cur_spread, spread_tracker)
        if not ok:
            log.warning("[ABORT-SPREAD] %s", reason)
            return ExecutionResult(False, None, -2, reason, 0, 0, 0, cur_spread)

        if side_u == "BUY":
            price = tick.ask
            sl = spec.normalize_price(price - sl_distance) if sl_distance > 0 else 0.0
            tp = spec.normalize_price(price + tp_distance) if tp_distance > 0 else 0.0
            order_type = mt5.ORDER_TYPE_BUY
        else:
            price = tick.bid
            sl = spec.normalize_price(price + sl_distance) if sl_distance > 0 else 0.0
            tp = spec.normalize_price(price - tp_distance) if tp_distance > 0 else 0.0
            order_type = mt5.ORDER_TYPE_SELL

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": spec.name,
            "volume": volume,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": max_dev,
            "magic": magic,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": spec.preferred_filling(),
        }
        result = mt5.order_send(request)
        if result is None:
            err = mt5.last_error()
            log.error("order_send None err=%s attempt=%d", err, attempt)
            time.sleep(delay)
            continue

        last = ExecutionResult(
            ok=(result.retcode == mt5.TRADE_RETCODE_DONE),
            ticket=getattr(result, "order", None) or getattr(result, "deal", None),
            retcode=result.retcode, comment=result.comment,
            price=result.price, sl=sl, tp=tp, spread_points=cur_spread,
        )
        if last.ok:
            log.info("[FILLED] %s %s vol=%.2f @%.5f sl=%.5f tp=%.5f spread=%.1fp (attempt %d)",
                     side_u, symbol, volume, result.price, sl, tp, cur_spread, attempt)
            return last

        if result.retcode in (
            mt5.TRADE_RETCODE_REQUOTE,
            mt5.TRADE_RETCODE_PRICE_CHANGED,
            mt5.TRADE_RETCODE_PRICE_OFF,
            10004,
        ):
            log.warning("[REQUOTE] retcode=%d attempt=%d/%d", result.retcode, attempt, retries)
            time.sleep(delay)
            continue
        log.error("[FAIL] retcode=%d %s", result.retcode, result.comment)
        return last
    return last


# ===================================================== modify SL/TP
def modify_position_sl(ticket: int, new_sl: Optional[float] = None,
                       new_tp: Optional[float] = None) -> bool:
    pos_list = mt5.positions_get(ticket=ticket)
    if not pos_list:
        return False
    pos = pos_list[0]
    sl = float(new_sl) if new_sl is not None else float(pos.sl)
    tp = float(new_tp) if new_tp is not None else float(pos.tp)
    req = {
        "action": mt5.TRADE_ACTION_SLTP,
        "position": ticket,
        "symbol": pos.symbol,
        "sl": sl,
        "tp": tp,
        "magic": pos.magic,
    }
    res = mt5.order_send(req)
    if res is None or res.retcode != mt5.TRADE_RETCODE_DONE:
        return False
    return True


# ===================================================== close
def close_position(ticket: int) -> bool:
    pos_list = mt5.positions_get(ticket=ticket)
    if not pos_list:
        return False
    pos = pos_list[0]
    spec = MT5Connector.get_symbol_spec(pos.symbol)
    tick = MT5Connector.get_tick(pos.symbol)
    if tick is None:
        return False
    is_buy = (pos.type == 0)
    price = tick.bid if is_buy else tick.ask
    cfg_v = Config.section("vulnerability_patches")
    req = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": pos.symbol,
        "volume": pos.volume,
        "type": mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY,
        "position": ticket,
        "price": price,
        "deviation": int(cfg_v.get("max_deviation_points", 30)),
        "magic": pos.magic,
        "comment": "close",
        "type_filling": spec.preferred_filling(),
    }
    res = mt5.order_send(req)
    if res is None or res.retcode != mt5.TRADE_RETCODE_DONE:
        log.warning("close_position ticket=%d retcode=%s",
                    ticket, getattr(res, "retcode", None))
        return False
    log.info("[CLOSED] ticket=%d @%.5f vol=%.2f", ticket, price, pos.volume)
    return True
