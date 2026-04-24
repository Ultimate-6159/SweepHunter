"""
Tick-Level Confirmation Engine
==============================
วิเคราะห์ tick data ก่อนเปิดออเดอร์ เพื่อยืนยัน/ยกเลิก signal จาก M5 model.

Detect:
- Tick imbalance (buy vs sell pressure ที่ระดับ tick)
- Spread spike (อาจมี broker manipulation/news leak)
- Stop hunt pattern (ราคาแหลมแล้วกลับ = smart money ไล่ stop)
- Tick velocity burst (ราคาวิ่งผิดปกติ = news/whale)
- Best Bid/Ask imbalance
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np

from .logger_setup import get_logger
from .mt5_connector import MT5Connector

log = get_logger("tick")


@dataclass
class TickAnalysis:
    """ผลการวิเคราะห์ tick — แต่ละ field 0..1 (ยิ่งใกล้ 0 = สัญญาณแย่)"""
    n_ticks: int
    buy_tick_ratio: float           # 0..1 (>0.5 = up-tick มากกว่า)
    spread_zscore: float            # spread เทียบ baseline (สูง = abnormal)
    stop_hunt_score: float          # 0..1 (สูง = pattern หลอก)
    velocity_burst: float           # 0..1 (สูง = sudden move)
    micro_trend: int                # +1 ขึ้น, -1 ลง, 0 sideways
    confirmed: bool
    reason: str

    def to_log(self) -> str:
        return (f"ticks={self.n_ticks} buy_ratio={self.buy_tick_ratio:.2f} "
                f"spread_z={self.spread_zscore:.1f} stop_hunt={self.stop_hunt_score:.2f} "
                f"burst={self.velocity_burst:.2f} micro={self.micro_trend:+d}")


def analyze_ticks_for_signal(
    symbol: str,
    side: str,                    # "BUY" or "SELL"
    seconds_back: int = 60,
    min_ticks: int = 20,
    max_spread_zscore: float = 3.0,
    min_buy_ratio_for_buy: float = 0.55,
    max_stop_hunt_score: float = 0.7,
    max_velocity_burst: float = 0.85,
    require_micro_trend_aligned: bool = True,
) -> TickAnalysis:
    """
    Pull last N seconds of ticks → analyze → return TickAnalysis.

    Logic:
    1. ถ้า tick น้อยเกิน → unconfirmed (ตลาดเงียบ ไม่เทรดดีกว่า)
    2. spread spike abnormal → unconfirmed (อาจมี news leak)
    3. ถ้า BUY แต่ buy_tick_ratio < 0.55 → unconfirmed (ทวนทาง tick)
    4. stop hunt pattern detected → unconfirmed (อาจถูกหลอก)
    5. velocity burst สูง → unconfirmed (ตามแล้วช้า = ตอนสุดเทรนด์)
    6. micro trend ไม่ align → unconfirmed
    """
    ticks = MT5Connector.copy_ticks_recent(symbol, seconds_back=seconds_back)
    if ticks is None or len(ticks) < min_ticks:
        return TickAnalysis(
            n_ticks=len(ticks) if ticks is not None else 0,
            buy_tick_ratio=0.5, spread_zscore=0.0, stop_hunt_score=0.0,
            velocity_burst=0.0, micro_trend=0, confirmed=False,
            reason=f"tick น้อยเกิน ({len(ticks) if ticks is not None else 0}<{min_ticks}) — ตลาดเงียบ",
        )

    bids = ticks["bid"].astype(float)
    asks = ticks["ask"].astype(float)
    mids = (bids + asks) / 2.0
    spreads = asks - bids

    # ── 1. Buy/Sell tick imbalance (uptick rule)
    mid_diff = np.diff(mids)
    up_ticks = int(np.sum(mid_diff > 0))
    down_ticks = int(np.sum(mid_diff < 0))
    total_dir = up_ticks + down_ticks
    buy_ratio = up_ticks / total_dir if total_dir > 0 else 0.5

    # ── 2. Spread z-score (compare last 10% vs full window)
    n_recent = max(5, len(spreads) // 10)
    recent_spread = float(np.mean(spreads[-n_recent:]))
    base_spread = float(np.median(spreads[:-n_recent])) if len(spreads) > n_recent else float(np.median(spreads))
    spread_std = float(np.std(spreads[:-n_recent])) if len(spreads) > n_recent else 0.001
    spread_z = (recent_spread - base_spread) / max(spread_std, 0.001)

    # ── 3. Stop hunt detection: ราคาแหลมไปทางหนึ่ง > 1.5σ แล้วกลับมาในเวลาสั้น
    rng = float(np.max(mids) - np.min(mids))
    final_pos = float(mids[-1])
    if rng > 0:
        # แหลมขึ้นแล้วกลับ → stop hunt ของ BUY (ไล่ SL ของ short-seller)
        peak_high = float(np.max(mids))
        peak_low = float(np.min(mids))
        retrace_from_high = (peak_high - final_pos) / rng
        retrace_from_low = (final_pos - peak_low) / rng
        # ถ้าราคาวิ่งสุดทางหนึ่งแล้วกลับเกือบครึ่ง = stop hunt
        stop_hunt = max(retrace_from_high, retrace_from_low)
    else:
        stop_hunt = 0.0

    # ── 4. Velocity burst: max single-tick move เทียบ avg
    if len(mid_diff) > 5:
        abs_moves = np.abs(mid_diff)
        burst = float(np.max(abs_moves) / (np.mean(abs_moves) + 1e-9))
        # normalize 0..1 (burst > 10 = severe)
        velocity_burst = min(burst / 10.0, 1.0)
    else:
        velocity_burst = 0.0

    # ── 5. Micro trend: linear regression slope ของ mid prices
    x = np.arange(len(mids), dtype=float)
    if len(mids) > 5:
        slope = float(np.polyfit(x, mids, 1)[0])
        # threshold by avg tick movement
        avg_move = float(np.mean(np.abs(mid_diff))) if len(mid_diff) > 0 else 0.001
        if slope > avg_move * 0.3:
            micro_trend = 1
        elif slope < -avg_move * 0.3:
            micro_trend = -1
        else:
            micro_trend = 0
    else:
        micro_trend = 0

    # ── Decide
    reasons = []
    if abs(spread_z) > max_spread_zscore:
        reasons.append(f"spread spike z={spread_z:.1f}")
    if side == "BUY":
        if buy_ratio < min_buy_ratio_for_buy:
            reasons.append(f"buy_ratio {buy_ratio:.2f}<{min_buy_ratio_for_buy} (tick ทวน BUY)")
        if require_micro_trend_aligned and micro_trend < 0:
            reasons.append("micro trend ลง แต่จะ BUY")
    elif side == "SELL":
        if (1 - buy_ratio) < min_buy_ratio_for_buy:
            reasons.append(f"sell_ratio {1-buy_ratio:.2f}<{min_buy_ratio_for_buy} (tick ทวน SELL)")
        if require_micro_trend_aligned and micro_trend > 0:
            reasons.append("micro trend ขึ้น แต่จะ SELL")
    if stop_hunt > max_stop_hunt_score:
        reasons.append(f"stop_hunt={stop_hunt:.2f} (ราคาแหลมแล้วกลับ)")
    if velocity_burst > max_velocity_burst:
        reasons.append(f"burst={velocity_burst:.2f} (เข้าตอนสุดเทรนด์)")

    confirmed = len(reasons) == 0
    reason = "✅ tick ยืนยัน" if confirmed else " | ".join(reasons)

    return TickAnalysis(
        n_ticks=len(ticks),
        buy_tick_ratio=buy_ratio,
        spread_zscore=spread_z,
        stop_hunt_score=stop_hunt,
        velocity_burst=velocity_burst,
        micro_trend=micro_trend,
        confirmed=confirmed,
        reason=reason,
    )
