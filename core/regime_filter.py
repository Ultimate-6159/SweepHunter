"""
🌊 regime_filter.py
====================
Trend-Aware Regime Filter — ป้องกันบอทเทรดสวนทาง trending market.

3 Layers:
  Layer 1 (HTF Trend):
    - คำนวณ trend_strength = |price - EMA50_HTF| / ATR_HTF
    - STRONG (> threshold)  → block counter-trend
    - WEAK   (in band)      → ลด lot
    - RANGE  (< threshold)  → ปกติ

  Layer 2 (Daily Drawdown):
    - ติดตาม PnL วันนี้ (จาก DB) เทียบ balance
    - ทะลุ max_dd_pct → halt ทั้งวัน (reset เที่ยงคืน UTC)

  Layer 3 (Disable Recovery in Trend):
    - ถ้า strong trend + อยู่ใน recovery → ปิด recovery escalation
      → ใช้ base lot เท่านั้น (ไม่ double-down ในทิศที่ตลาดวิ่งหนัก)

ใช้:
    from core.regime_filter import RegimeFilter
    rf = RegimeFilter()
    state = rf.evaluate(side="BUY", symbol="XAUUSD", balance=1000)
    if not state.allow_entry: return
    lot_factor = state.lot_factor          # 0.5 ถึง 1.0
    skip_recovery = state.skip_recovery_escalation
"""
from __future__ import annotations
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import MetaTrader5 as mt5

from .config import Config
from .logger import get_logger
from .paths import db_path

log = get_logger("regime")

TF_MAP = {
    "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
}


@dataclass
class RegimeState:
    allow_entry: bool = True
    reason: str = "OK"
    lot_factor: float = 1.0                 # 0.5-1.0 multiplier
    skip_recovery_escalation: bool = False  # use base lot in recovery
    trend_strength: float = 0.0
    trend_state: str = "RANGE"              # RANGE / WEAK / STRONG_UP / STRONG_DOWN
    daily_dd_pct: float = 0.0


class RegimeFilter:
    def __init__(self) -> None:
        self.cfg = Config.section("regime_filter") or {}
        self._cache_trend: Optional[tuple] = None    # (ts, state)
        self._cache_dd: Optional[tuple] = None       # (ts, dd_pct, halted)
        self._cache_ttl = 60.0  # re-eval ทุก 60s

    # ----------------------------------------------------------------- HTF trend
    def _compute_htf_trend(self, symbol: str) -> tuple[str, float]:
        """คืน (trend_state, strength) — strength = |price-EMA50_HTF| / ATR_HTF"""
        cfg_t = self.cfg.get("htf_trend", {}) or {}
        if not cfg_t.get("enabled", True):
            return "RANGE", 0.0
        tf_str = cfg_t.get("timeframe", "H1")
        tf = TF_MAP.get(tf_str, mt5.TIMEFRAME_H1)
        ema_p = int(cfg_t.get("ema_period", 50))
        atr_p = int(cfg_t.get("atr_period", 14))
        strong_thr = float(cfg_t.get("strong_trend_threshold", 1.5))
        weak_thr = float(cfg_t.get("weak_trend_threshold", 0.5))

        try:
            need = max(ema_p, atr_p) + 5
            rates = mt5.copy_rates_from_pos(symbol, tf, 0, need + 50)
            if rates is None or len(rates) < need:
                return "RANGE", 0.0
            close = rates["close"]
            high = rates["high"]
            low = rates["low"]
            # EMA
            alpha = 2.0 / (ema_p + 1.0)
            ema = close[0]
            for c in close[1:]:
                ema = alpha * c + (1 - alpha) * ema
            # ATR (simple)
            tr = np.maximum.reduce([
                high[1:] - low[1:],
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1]),
            ])
            atr = float(np.mean(tr[-atr_p:]))
            if atr <= 0:
                return "RANGE", 0.0
            price = float(close[-1])
            dist = price - ema
            strength = abs(dist) / atr
            if strength >= strong_thr:
                state = "STRONG_UP" if dist > 0 else "STRONG_DOWN"
            elif strength >= weak_thr:
                state = "WEAK_UP" if dist > 0 else "WEAK_DOWN"
            else:
                state = "RANGE"
            return state, strength
        except Exception as e:
            log.debug("HTF trend calc failed: %s", e)
            return "RANGE", 0.0

    # ----------------------------------------------------------------- daily DD
    def _compute_daily_dd_pct(self, balance: float) -> float:
        """คืน drawdown % ของวันนี้ (UTC) เทียบ balance"""
        if balance <= 0:
            return 0.0
        try:
            cfg_db = Config.section("database") or {}
            path = db_path(cfg_db.get("filename", "hyper_trades.sqlite"))
            if not path.exists():
                return 0.0
            today = datetime.now(timezone.utc).date().isoformat()
            with sqlite3.connect(str(path)) as c:
                row = c.execute(
                    "SELECT COALESCE(SUM(pnl),0) FROM decisions "
                    "WHERE status IN ('WIN','LOSS') AND date(ts_utc)=?",
                    (today,)).fetchone()
            net = float(row[0] or 0.0)
            if net >= 0:
                return 0.0
            return abs(net) / balance * 100.0
        except Exception as e:
            log.debug("daily DD calc failed: %s", e)
            return 0.0

    # ----------------------------------------------------------------- main API
    def evaluate(self, side: str, symbol: str, balance: float,
                 in_recovery: bool = False) -> RegimeState:
        """ตรวจทุก layer แล้วคืน RegimeState"""
        st = RegimeState()
        if not self.cfg.get("enabled", True):
            return st

        now = time.time()

        # Layer 2: Daily DD (cached) — check ก่อนเพื่อ short-circuit
        cfg_dd = self.cfg.get("daily_drawdown", {}) or {}
        if cfg_dd.get("enabled", True):
            if not self._cache_dd or (now - self._cache_dd[0]) > self._cache_ttl:
                dd = self._compute_daily_dd_pct(balance)
                self._cache_dd = (now, dd, False)
            dd = self._cache_dd[1]
            st.daily_dd_pct = dd
            max_dd = float(cfg_dd.get("max_dd_pct_of_balance", 5.0))
            if dd >= max_dd:
                st.allow_entry = False
                st.reason = f"DAILY_DD_HALT: -{dd:.2f}% ≥ -{max_dd:.1f}% (รอเที่ยงคืน UTC)"
                return st

        # Layer 1: HTF Trend (cached)
        if not self._cache_trend or (now - self._cache_trend[0]) > self._cache_ttl:
            trend_state, strength = self._compute_htf_trend(symbol)
            self._cache_trend = (now, (trend_state, strength))
        trend_state, strength = self._cache_trend[1]
        st.trend_state = trend_state
        st.trend_strength = strength

        # Block counter-trend in STRONG trend
        if trend_state == "STRONG_UP" and side == "SELL":
            st.allow_entry = False
            st.reason = f"COUNTER_TREND: SELL ใน STRONG UPTREND (strength {strength:.2f})"
            return st
        if trend_state == "STRONG_DOWN" and side == "BUY":
            st.allow_entry = False
            st.reason = f"COUNTER_TREND: BUY ใน STRONG DOWNTREND (strength {strength:.2f})"
            return st

        # Layer 3: Disable recovery escalation in strong trend
        if (in_recovery and trend_state in ("STRONG_UP", "STRONG_DOWN")
                and self.cfg.get("disable_recovery_in_strong_trend", True)):
            st.skip_recovery_escalation = True
            st.reason = f"RECOVERY-NO-ESCALATION (trend {trend_state})"

        # WEAK trend — reduce lot
        cfg_t = self.cfg.get("htf_trend", {}) or {}
        weak_factor = float(cfg_t.get("weak_trend_lot_factor", 0.5))
        if trend_state.startswith("WEAK"):
            # weak counter-trend → reduce lot, weak with-trend → normal
            with_trend = ((trend_state == "WEAK_UP" and side == "BUY") or
                          (trend_state == "WEAK_DOWN" and side == "SELL"))
            if not with_trend:
                st.lot_factor = weak_factor
                st.reason = f"WEAK_COUNTER_TREND: lot × {weak_factor}"

        return st

    def to_log_str(self, s: RegimeState) -> str:
        return (f"trend={s.trend_state}({s.trend_strength:.2f}) "
                f"dd={s.daily_dd_pct:.1f}% lot×{s.lot_factor:.2f}")
