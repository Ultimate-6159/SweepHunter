"""
m1_hyper_pipeline.py
=====================
Feature Extraction + Aggressive Labeling สำหรับ Hyper-Frequency Micro-Scalping (M1).

เป้าหมาย: ผลิตสัญญาณ Train ที่ "ครอบคลุมการขยับเล็กๆ" ของราคา
เพื่อให้ AI กล้าเข้าเทรดบ่อย (40+ ออเดอร์/วัน) ที่ confidence threshold ต่ำ (0.35)

Features ตาม SRD:
  - Micro_Wick_Ratios       : สัดส่วนไส้เทียน vs body
  - Fast_ATR_Normalization  : wick / ATR(5)
  - Price_Velocity          : delta close 2 แท่ง
  - Volume_Acceleration     : tick_vol / mean(tick_vol[-3:])
  - Micro_Breakout_Signal   : ราคาทะลุ high/low ของ 5 แท่งล่าสุด

Aggressive Labeling:
  TP = 0.6 * ATR  (เล็ก -> ครอบคลุม win ได้ง่าย)
  SL = 1.2 * ATR  (กว้างกว่า TP -> ลดการชน SL ก่อน)
  lookahead = 30 แท่ง (M1 -> 30 นาที)
"""
from __future__ import annotations
from typing import List, Tuple
import numpy as np
import pandas as pd

from .config import Config
from .logger import get_logger

log = get_logger("pipeline")

FEATURE_COLUMNS: List[str] = [
    # ---- Micro candle anatomy ----
    "body_atr",
    "upper_wick_body_ratio",
    "lower_wick_body_ratio",
    "candle_direction",
    # ---- Fast ATR normalisation ----
    "upper_wick_fast_atr",
    "lower_wick_fast_atr",
    "body_fast_atr",
    # ---- Price velocity / momentum ----
    "price_velocity_2",
    "price_velocity_5",
    "ret_1",
    "rsi_7",
    "ema_dist_atr",
    # ---- Volume acceleration ----
    "vol_accel_3",
    "vol_spike_10",
    # ---- Micro breakout (last 5 bars high/low) ----
    "breakout_up_5",
    "breakout_dn_5",
    "near_high_5",
    "near_low_5",
    # ---- Volatility regime ----
    "atr_ratio",
    # ---- Time / Session weighting ----
    "time_sin",
    "time_cos",
    "session_score",
    # ---- 🆕 Multi-Timeframe context (HTF) ----
    "htf1_trend_dir",       # +1 above EMA, -1 below, 0 near
    "htf1_ema_dist_atr",
    "htf1_rsi_norm",         # rsi/100 - 0.5 (centered around 0)
    "htf2_trend_dir",
    "htf2_ema_dist_atr",
    "htf2_rsi_norm",
    # ---- 🆕 Symmetric Candle Patterns (3-5 bar lookback) ----
    "pat_engulf_bull",       # bullish engulfing
    "pat_engulf_bear",       # bearish engulfing
    "pat_pinbar_top_5",      # rejection from top (max upper_wick/range in 5)
    "pat_pinbar_bot_5",      # rejection from bottom
    "pat_inside_bar",        # consolidation
    "pat_outside_bar",       # volatility expansion
    "pat_range_expansion_3", # current range vs avg(3)
    "pat_swept_high_5",      # liquidity grab at high (sweep + reverse)
    "pat_swept_low_5",       # liquidity grab at low
    "pat_consec_streak",     # signed: + bull streak / - bear streak
    # ---- 🆕 Support/Resistance + Multi-Pattern Aggregation (V3) ----
    "sr_dist_pivot_high_50",   # ATRs to 50-bar resistance
    "sr_dist_pivot_low_50",    # ATRs to 50-bar support
    "sr_dist_round_50",        # ATRs to nearest 50-multiple price
    "sr_dist_round_100",       # ATRs to nearest 100-multiple price
    "sr_range_position_20",    # 0..1 position within 20-bar range
    "mom_velocity_accel",      # velocity_5 acceleration
    "mom_body_accel",          # body size acceleration
    "mom_wick_imbalance",      # (upper - lower) / (upper + lower)
    "mom_pattern_bull_5",      # count of bullish patterns in last 5 bars
    "mom_pattern_bear_5",      # count of bearish patterns in last 5 bars
    # ---- 🆕 V4: Smart Market Structure + Volume Analysis ----
    "liq_sweep_bull",          # bullish liquidity sweep score (bear SL hunt → price recovers)
    "liq_sweep_bear",          # bearish liquidity sweep score (bull SL hunt → price drops)
    "mom_divergence",          # RSI vs price divergence: +1=bull div, 0=none, -1=bear div
    "order_block_bull",        # bullish order block: bearish candle precedes strong up impulse
    "order_block_bear",        # bearish order block: bullish candle precedes strong down impulse
    "fvg_bull",                # bullish fair value gap (gap up — mean-reversion from below)
    "fvg_bear",                # bearish fair value gap (gap down — mean-reversion from above)
    "cum_delta_norm",          # normalized cumulative volume delta (+= buying pressure)
    "vol_poc_dist",            # distance from VWAP/POC proxy to close price (ATRs)
    "supply_demand_imbal",     # (bull_vol − bear_vol) / total_vol in 20-bar window
]


# --------------------------------------------------------------------- helpers
def _rates_to_df(rates) -> pd.DataFrame:
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    high, low = df["high"], df["low"]
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def _rsi(close: pd.Series, period: int = 7) -> pd.Series:
    d = close.diff()
    gain = d.clip(lower=0)
    loss = (-d).clip(lower=0)
    ag = gain.ewm(alpha=1 / period, adjust=False).mean()
    al = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = ag / al.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def _session_score(hour_utc: pd.Series) -> pd.Series:
    """0=Asia/Off (low), 1=London, 2=NY, 3=Overlap (best)."""
    h = hour_utc
    s = pd.Series(0.0, index=h.index)
    s[(h >= 7) & (h < 13)] = 1.0
    s[(h >= 13) & (h < 17)] = 3.0
    s[(h >= 17) & (h < 22)] = 2.0
    return s


def _mtf_features(df: pd.DataFrame, multiplier: int, prefix: str,
                   ema_period: int = 20, atr_period: int = 14, rsi_period: int = 14
                   ) -> pd.DataFrame:
    """
    Compute Higher-Timeframe context features by resampling the base df.
    multiplier = 3 means: if base=M5 → HTF=M15 (3×); if base=M5 → HTF=H1 (12×).
    Each base bar inherits the latest closed HTF bar's features (forward fill).
    Returns DataFrame with columns: {prefix}_trend_dir, {prefix}_ema_dist_atr, {prefix}_rsi_norm
    """
    if multiplier <= 1:
        out = pd.DataFrame(index=df.index)
        out[f"{prefix}_trend_dir"] = 0.0
        out[f"{prefix}_ema_dist_atr"] = 0.0
        out[f"{prefix}_rsi_norm"] = 0.0
        return out
    # Build HTF bars: every Nth row's OHLC aggregated
    htf = pd.DataFrame(index=df.index)
    htf["bucket"] = (np.arange(len(df)) // multiplier)
    grp = df.groupby(htf["bucket"])
    htf_o = grp["open"].first()
    htf_h = grp["high"].max()
    htf_l = grp["low"].min()
    htf_c = grp["close"].last()
    htf_df = pd.DataFrame({"open": htf_o, "high": htf_h, "low": htf_l, "close": htf_c})
    # ATR + EMA + RSI on HTF
    htf_atr = _atr(htf_df, atr_period)
    htf_ema = htf_c.ewm(span=ema_period, adjust=False).mean()
    htf_rsi = _rsi(htf_c, rsi_period)
    atr_safe = htf_atr.replace(0, np.nan)
    htf_dist = (htf_c - htf_ema) / atr_safe
    htf_dir = np.sign(htf_dist).fillna(0.0)
    htf_rsi_norm = (htf_rsi / 100.0) - 0.5
    # Map back to base df rows (use bucket-1 = last CLOSED higher-TF bar)
    bucket_for_row = htf["bucket"].astype(int) - 1
    bucket_for_row = bucket_for_row.clip(lower=0)
    out = pd.DataFrame(index=df.index)
    out[f"{prefix}_trend_dir"] = bucket_for_row.map(htf_dir).fillna(0.0).astype(float).values
    out[f"{prefix}_ema_dist_atr"] = bucket_for_row.map(htf_dist).fillna(0.0).astype(float).values
    out[f"{prefix}_rsi_norm"] = bucket_for_row.map(htf_rsi_norm).fillna(0.0).astype(float).values
    return out


# --------------------------------------------------------------------- features
def build_features(rates) -> pd.DataFrame:
    """
    คำนวณ Features Hyper-Frequency บน M1.
    คืน DataFrame เต็ม (รวมคอลัมน์ดิบ) - dropna ภายนอก
    """
    cfg = Config.section("trading")
    atr_p = int(cfg.get("atr_period", 14))
    fatr_p = int(cfg.get("fast_atr_period", 5))
    ema_p = int(cfg.get("ema_period", 20))

    df = _rates_to_df(rates) if not isinstance(rates, pd.DataFrame) else rates.copy()

    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    body_signed = c - o
    body = body_signed.abs()
    upper_wick = h - np.maximum(o, c)
    lower_wick = np.minimum(o, c) - l

    # ATR (slow + fast)
    df["atr"] = _atr(df, atr_p)
    df["atr_fast"] = _atr(df, fatr_p)
    atr_safe = df["atr"].replace(0, np.nan)
    fatr_safe = df["atr_fast"].replace(0, np.nan)
    body_safe = body + 1e-6

    # ----- Micro candle anatomy -----
    df["body_atr"] = body / atr_safe
    df["upper_wick_body_ratio"] = (upper_wick / body_safe).clip(0, 20)
    df["lower_wick_body_ratio"] = (lower_wick / body_safe).clip(0, 20)
    df["candle_direction"] = np.sign(body_signed).astype(float)

    # ----- Fast ATR normalisation (SRD: wick / ATR5) -----
    df["upper_wick_fast_atr"] = upper_wick / fatr_safe
    df["lower_wick_fast_atr"] = lower_wick / fatr_safe
    df["body_fast_atr"] = body / fatr_safe

    # ----- Price velocity (SRD) -----
    df["price_velocity_2"] = (c - c.shift(2)) / fatr_safe
    df["price_velocity_5"] = (c - c.shift(5)) / fatr_safe
    df["ret_1"] = c.pct_change(1) * 100.0

    # ----- Momentum -----
    df["rsi_7"] = _rsi(c, 7)
    ema = c.ewm(span=ema_p, adjust=False).mean()
    df["ema_dist_atr"] = (c - ema) / atr_safe

    # ----- Volume Acceleration (SRD: vol vs avg(last 3)) -----
    vol = df["tick_volume"].astype(float)
    vol_recent = vol.shift(1).rolling(3, min_periods=3).mean()
    df["vol_accel_3"] = (vol / vol_recent.replace(0, np.nan)).fillna(1.0)
    df["vol_spike_10"] = (vol / vol.rolling(10, min_periods=10).mean().replace(0, np.nan)).fillna(1.0)

    # ----- Micro Breakout Signal (SRD: ทะลุ High/Low ของ 5 แท่งล่าสุด) -----
    rolling_high_5 = h.shift(1).rolling(5, min_periods=5).max()
    rolling_low_5 = l.shift(1).rolling(5, min_periods=5).min()
    df["breakout_up_5"] = (c > rolling_high_5).astype(float)
    df["breakout_dn_5"] = (c < rolling_low_5).astype(float)
    range_5 = (rolling_high_5 - rolling_low_5).replace(0, np.nan)
    df["near_high_5"] = ((c - rolling_low_5) / range_5).clip(0, 1).fillna(0.5)
    df["near_low_5"] = 1.0 - df["near_high_5"]

    # ----- Volatility regime -----
    df["atr_ratio"] = df["atr"] / df["atr"].rolling(50, min_periods=50).mean().replace(0, np.nan)

    # ----- Cyclical time + session -----
    minutes = df["time"].dt.hour * 60 + df["time"].dt.minute
    angle = 2.0 * np.pi * minutes / (24 * 60)
    df["time_sin"] = np.sin(angle)
    df["time_cos"] = np.cos(angle)
    df["session_score"] = _session_score(df["time"].dt.hour)

    # ----- 🆕 Multi-Timeframe context -----
    # base TF → infer multipliers (M1: htf1=15, htf2=60 | M5: htf1=3, htf2=12 | M15: htf1=2, htf2=4)
    base_tf = str(cfg.get("timeframe", "M5")).upper()
    if base_tf == "M1":
        m1, m2 = 15, 60
    elif base_tf == "M5":
        m1, m2 = 3, 12   # M15, H1
    elif base_tf == "M15":
        m1, m2 = 4, 16   # H1, H4
    else:
        m1, m2 = 4, 12
    htf1 = _mtf_features(df, m1, "htf1", ema_p, atr_p)
    htf2 = _mtf_features(df, m2, "htf2", ema_p, atr_p)
    for col in htf1.columns:
        df[col] = htf1[col].values
    for col in htf2.columns:
        df[col] = htf2[col].values

    # ----- 🆕 Symmetric Candle Patterns (3-5 bar lookback) -----
    # symmetric features: ทำงานทั้ง BUY/SELL ได้เท่ากัน → กัน directional bias
    o1, h1, l1, c1 = o.shift(1), h.shift(1), l.shift(1), c.shift(1)
    body_now = (c - o)
    body_prev = (c1 - o1)
    bull_now = (body_now > 0)
    bear_now = (body_now < 0)
    bull_prev = (body_prev > 0)
    bear_prev = (body_prev < 0)

    # Engulfing: bull engulfs prev bear, body > prev body
    df["pat_engulf_bull"] = (
        bear_prev & bull_now &
        (c > o1) & (o < c1) &
        (body_now.abs() > body_prev.abs())
    ).astype(float)
    df["pat_engulf_bear"] = (
        bull_prev & bear_now &
        (c < o1) & (o > c1) &
        (body_now.abs() > body_prev.abs())
    ).astype(float)

    # Pin bar: max upper/lower wick / range over last 5 bars
    rng = (h - l).replace(0, np.nan)
    upper_pct = (upper_wick / rng).clip(0, 1).fillna(0)
    lower_pct = (lower_wick / rng).clip(0, 1).fillna(0)
    df["pat_pinbar_top_5"] = upper_pct.rolling(5, min_periods=3).max().fillna(0)
    df["pat_pinbar_bot_5"] = lower_pct.rolling(5, min_periods=3).max().fillna(0)

    # Inside bar: current range fully inside prev range
    df["pat_inside_bar"] = ((h < h1) & (l > l1)).astype(float)
    # Outside bar: current range fully engulfs prev
    df["pat_outside_bar"] = ((h > h1) & (l < l1)).astype(float)

    # Range expansion: current range / avg(last 3)
    range_now = (h - l)
    avg_range_3 = range_now.shift(1).rolling(3, min_periods=2).mean().replace(0, np.nan)
    df["pat_range_expansion_3"] = (range_now / avg_range_3).clip(0, 5).fillna(1.0)

    # Liquidity sweep: high/low pierces 5-bar extreme then closes back inside
    prev_high_5 = h.shift(1).rolling(5, min_periods=3).max()
    prev_low_5 = l.shift(1).rolling(5, min_periods=3).min()
    df["pat_swept_high_5"] = ((h > prev_high_5) & (c < prev_high_5)).astype(float)
    df["pat_swept_low_5"] = ((l < prev_low_5) & (c > prev_low_5)).astype(float)

    # Consecutive streak (signed): +k = k bull bars, -k = k bear bars
    direction = np.sign(body_now)
    streak_id = (direction != direction.shift(1)).cumsum()
    streak_len = direction.groupby(streak_id).cumcount() + 1
    df["pat_consec_streak"] = (direction * streak_len).clip(-7, 7).fillna(0)

    # ----- 🆕 V3: Support/Resistance + Multi-Pattern Aggregation -----
    # ดูแท่งเทียนเยอะขึ้น (50 bars) + แนวรับแนวต้าน + รวม pattern หลายตัว

    atr_safe2 = df["atr"].replace(0, np.nan)

    # S/R Pivot levels (50-bar lookback) — กลายเป็น "ATR distance"
    pivot_high_50 = h.shift(1).rolling(50, min_periods=20).max()
    pivot_low_50 = l.shift(1).rolling(50, min_periods=20).min()
    df["sr_dist_pivot_high_50"] = ((pivot_high_50 - c) / atr_safe2).clip(-10, 10).fillna(0)
    df["sr_dist_pivot_low_50"] = ((c - pivot_low_50) / atr_safe2).clip(-10, 10).fillna(0)

    # Round number distance (psychological levels) — gold uses 50/100 multiples
    round_50 = (c / 50.0).round() * 50.0
    round_100 = (c / 100.0).round() * 100.0
    df["sr_dist_round_50"] = ((c - round_50).abs() / atr_safe2).clip(0, 10).fillna(1.0)
    df["sr_dist_round_100"] = ((c - round_100).abs() / atr_safe2).clip(0, 10).fillna(1.0)

    # Range position 0..1 (where in 20-bar range) — 0=at low (supp) 1=at high (res)
    rh20 = h.rolling(20, min_periods=10).max()
    rl20 = l.rolling(20, min_periods=10).min()
    rng20 = (rh20 - rl20).replace(0, np.nan)
    df["sr_range_position_20"] = ((c - rl20) / rng20).clip(0, 1).fillna(0.5)

    # Momentum dynamics (acceleration = derivative of velocity)
    velocity_5_safe = df["price_velocity_5"]
    df["mom_velocity_accel"] = (velocity_5_safe - velocity_5_safe.shift(1)).fillna(0)
    body_atr_safe = df["body_atr"]
    df["mom_body_accel"] = (body_atr_safe - body_atr_safe.shift(1)).fillna(0)

    # Wick imbalance: +1 = upper wick dominant (sellers up there)  -1 = lower wick (buyers down)
    wick_total = (upper_wick + lower_wick + 1e-6)
    df["mom_wick_imbalance"] = ((upper_wick - lower_wick) / wick_total).fillna(0)

    # Multi-pattern aggregation in last 5 bars (sum of bullish vs bearish patterns)
    bull_patterns = df["pat_engulf_bull"] + df["pat_swept_low_5"]
    bear_patterns = df["pat_engulf_bear"] + df["pat_swept_high_5"]
    df["mom_pattern_bull_5"] = bull_patterns.rolling(5, min_periods=1).sum().fillna(0)
    df["mom_pattern_bear_5"] = bear_patterns.rolling(5, min_periods=1).sum().fillna(0)

    # ----- 🆕 V4: Smart Market Structure + Volume Analysis -----

    # --- Liquidity Sweep Score ---
    # bull sweep: price briefly dipped below 10-bar low then closed above it
    # score = how far below it went (sweep depth) / ATR → bigger spike = stronger sweep
    prev_low_10 = l.shift(1).rolling(10, min_periods=5).min()
    prev_high_10 = h.shift(1).rolling(10, min_periods=5).max()
    bull_swept = (l < prev_low_10) & (c > prev_low_10)
    bear_swept = (h > prev_high_10) & (c < prev_high_10)
    df["liq_sweep_bull"] = (bull_swept * ((prev_low_10 - l).clip(lower=0) / atr_safe)).fillna(0).clip(0, 3)
    df["liq_sweep_bear"] = (bear_swept * ((h - prev_high_10).clip(lower=0) / atr_safe)).fillna(0).clip(0, 3)

    # --- Momentum Divergence (RSI vs Price) ---
    # +1 = bullish divergence (price falling, RSI rising → reversal up)
    # -1 = bearish divergence (price rising, RSI falling → reversal down)
    # 0  = no divergence / aligned
    price_dir_3 = np.sign(c - c.shift(3))
    rsi_dir_3 = np.sign(df["rsi_7"] - df["rsi_7"].shift(3))
    # divergence when price and RSI move in opposite directions
    mom_div_raw = (rsi_dir_3 - price_dir_3)  # +2=bull div, -2=bear div, 0=aligned, ±1=partial
    df["mom_divergence"] = (mom_div_raw / 2.0).fillna(0).clip(-1, 1)

    # --- Order Block Detection ---
    # Bullish OB: a strong bearish candle (body > 0.5 ATR) followed within 1-3 bars by a strong bullish candle
    # Indicates institutional demand at that price level
    strong_bear_body = (body_signed < -0.5 * atr_safe)
    strong_bull_body = (body_signed > 0.5 * atr_safe)
    # OB bull: was there a strong bear candle 1 or 2 bars ago, and current is strong bull?
    ob_bull_trigger = (strong_bear_body.shift(1) | strong_bear_body.shift(2)) & strong_bull_body
    # OB bear: strong bull candle 1-2 bars ago, now strong bearish move
    ob_bear_trigger = (strong_bull_body.shift(1) | strong_bull_body.shift(2)) & strong_bear_body
    df["order_block_bull"] = ob_bull_trigger.astype(float).fillna(0)
    df["order_block_bear"] = ob_bear_trigger.astype(float).fillna(0)

    # --- Fair Value Gap (FVG) ---
    # Bullish FVG: current bar's low > 2-bars-ago high → gap up (institutional imbalance, price may retrace up)
    # Bearish FVG: current bar's high < 2-bars-ago low → gap down
    df["fvg_bull"] = (l > h.shift(2)).astype(float).fillna(0)
    df["fvg_bear"] = (h < l.shift(2)).astype(float).fillna(0)

    # --- Cumulative Volume Delta (buying vs selling pressure) ---
    # Approximate buy/sell volume from candle structure (no raw tick data available)
    # buy_vol = tick_vol × (close−low)/(high−low)  — fraction that went up
    # sell_vol = tick_vol × (high−close)/(high−low) — fraction that went down
    candle_range_safe = (h - l).replace(0, np.nan)
    buy_frac = ((c - l) / candle_range_safe).fillna(0.5).clip(0, 1)
    buy_vol_20 = (vol * buy_frac).rolling(20, min_periods=5).sum()
    sell_vol_20 = (vol * (1.0 - buy_frac)).rolling(20, min_periods=5).sum()
    vol_mean_20 = vol.rolling(20, min_periods=10).mean().replace(0, np.nan)
    # Normalize by vol_mean × 20 bars so it's scale-independent
    cum_delta = buy_vol_20 - sell_vol_20
    df["cum_delta_norm"] = (cum_delta / (vol_mean_20 * 20.0)).fillna(0).clip(-1, 1)

    # --- Volume POC Distance (VWAP proxy) ---
    # VWAP over 20 bars = volume-weighted average price → proxy for point of control
    # Price below VWAP → discount zone (potential support / demand)
    # Price above VWAP → premium zone (potential resistance / supply)
    vwap_20 = (c * vol).rolling(20, min_periods=10).sum() / vol.rolling(20, min_periods=10).sum().replace(0, np.nan)
    df["vol_poc_dist"] = ((c - vwap_20) / atr_safe).fillna(0).clip(-5, 5)

    # --- Supply/Demand Zone Imbalance ---
    # Bull pressure volume vs bear pressure volume in 20-bar lookback
    # +1 = all demand (buying), -1 = all supply (selling)
    bear_vol_sum = (vol * (body_signed < 0)).rolling(20, min_periods=10).sum()
    bull_vol_sum = (vol * (body_signed >= 0)).rolling(20, min_periods=10).sum()
    total_vol_sum = (bear_vol_sum + bull_vol_sum).replace(0, np.nan)
    df["supply_demand_imbal"] = ((bull_vol_sum - bear_vol_sum) / total_vol_sum).fillna(0).clip(-1, 1)

    return df


# --------------------------------------------------------------------- labeling
def aggressive_label(
    df: pd.DataFrame,
    tp_atr: float = 1.6,
    sl_atr: float = 0.8,
    lookahead: int = 12,
) -> pd.Series:
    """
    Profit-Driven Triple-Barrier Labeling (v3 — "ชนะให้ใหญ่ เสียให้เล็ก"):

      Defaults: TP = 1.6 ATR, SL = 0.8 ATR  → RR = 2:1
      - lookahead สั้น (12 แท่ง) → ไม่ hold นาน, สอน model ให้เห็นโอกาสที่
        "วิ่งเร็วและไกล" ภายในหน้าต่างสั้น
      - Tie-break (TP+SL ในแท่งเดียวกัน) = SL ก่อน → conservative,
        ป้องกัน label ให้สัญญาณ WIN ทั้งที่จริงๆอาจชน SL ก่อน
      - 2 = BUY win, 0 = SELL win, 1 = HOLD

    NOTE: aggressive_label ชื่อนี้ legacy — พฤติกรรมเปลี่ยนเป็น profit-driven แล้ว
    """
    n = len(df)
    labels = np.full(n, 1, dtype=np.int8)

    close = df["close"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    atr = df["atr"].to_numpy()

    for i in range(n - 1):
        a = atr[i]
        if not np.isfinite(a) or a <= 0:
            continue
        entry = close[i]
        tp_up = entry + tp_atr * a
        sl_dn = entry - sl_atr * a
        tp_dn = entry - tp_atr * a
        sl_up = entry + sl_atr * a

        end = min(n, i + 1 + lookahead)
        buy_resolved = sell_resolved = False
        buy_win = sell_win = False

        for j in range(i + 1, end):
            hj, lj = high[j], low[j]
            if not buy_resolved:
                tp_hit = hj >= tp_up
                sl_hit = lj <= sl_dn
                # 🛡️ tie-break: ถ้าทั้ง TP และ SL hit ในแท่งเดียวกัน → ถือว่า SL ก่อน
                #    (conservative — กัน label พูดเกินจริง)
                if sl_hit:
                    buy_resolved = True  # SL ก่อน → loss
                elif tp_hit:
                    buy_resolved = True; buy_win = True
            if not sell_resolved:
                tp_hit = lj <= tp_dn
                sl_hit = hj >= sl_up
                if sl_hit:
                    sell_resolved = True
                elif tp_hit:
                    sell_resolved = True; sell_win = True
            if buy_resolved and sell_resolved:
                break

        # ฝั่งใดฝั่งหนึ่งชัดเจน → label ฝั่งนั้น
        if buy_win and not sell_win:
            labels[i] = 2
        elif sell_win and not buy_win:
            labels[i] = 0

    return pd.Series(labels, index=df.index, name="label")


def regime_label(
    df: pd.DataFrame,
    lookahead: int = 12,
) -> pd.Series:
    """
    🆕 Regime-Aware Asymmetric Labeling (v4):
    ใช้ market regime ที่ตรวจจับได้ เพื่อตั้ง TP/SL แบบ asymmetric ตามทิศทาง

    Regimes detected from `df` columns (computed by build_features):
      TRENDING_UP   (htf1_trend_dir > 0 AND htf1_ema_dist_atr > threshold)
        → BUY:  TP=1.8×ATR, SL=0.6×ATR  (trend follower, tight SL)
        → SELL: TP=1.2×ATR, SL=1.2×ATR  (counter-trend, balanced — rare signal)

      TRENDING_DOWN (htf1_trend_dir < 0 AND htf1_ema_dist_atr < -threshold)
        → SELL: TP=1.4×ATR, SL=1.0×ATR
        → BUY:  TP=1.2×ATR, SL=1.2×ATR

      HIGH_VOLATILITY (atr_ratio > 1.5)
        → Skip labeling → HOLD (too risky)

      BREAKOUT (breakout_up_5 OR breakout_dn_5 AND atr_ratio > 1.2)
        → Aggressive: TP=2.0×ATR, SL=0.8×ATR

      CHOPPY / DEFAULT
        → TP=0.8×ATR, SL=1.2×ATR  (mean-reversion targets)

    Falls back to aggressive_label defaults if required columns missing.
    """
    cfg = Config.section("trading")
    n = len(df)
    labels = np.full(n, 1, dtype=np.int8)

    close = df["close"].to_numpy()
    high  = df["high"].to_numpy()
    low   = df["low"].to_numpy()
    atr   = df["atr"].to_numpy()

    # pull regime columns (may not exist in legacy dfs)
    htf_trend = df.get("htf1_trend_dir", pd.Series(0.0, index=df.index)).to_numpy()
    htf_dist  = df.get("htf1_ema_dist_atr", pd.Series(0.0, index=df.index)).to_numpy()
    atr_ratio = df.get("atr_ratio", pd.Series(1.0, index=df.index)).to_numpy()
    brk_up    = df.get("breakout_up_5", pd.Series(0.0, index=df.index)).to_numpy()
    brk_dn    = df.get("breakout_dn_5", pd.Series(0.0, index=df.index)).to_numpy()

    trend_thr = float(cfg.get("regime_trend_threshold", 0.5))

    for i in range(n - 1):
        a = atr[i]
        if not np.isfinite(a) or a <= 0:
            continue
        entry = close[i]

        # --- Detect regime at bar i ---
        ar = float(atr_ratio[i]) if np.isfinite(atr_ratio[i]) else 1.0
        td = float(htf_trend[i]) if np.isfinite(htf_trend[i]) else 0.0
        hd = float(htf_dist[i]) if np.isfinite(htf_dist[i]) else 0.0
        bu = bool(brk_up[i])
        bd = bool(brk_dn[i])

        # HIGH_VOLATILITY → skip (label stays HOLD)
        if ar > 1.5:
            continue

        # BREAKOUT
        if ar > 1.2 and (bu or bd):
            buy_tp,  buy_sl  = 2.0, 0.8
            sell_tp, sell_sl = 2.0, 0.8

        # TRENDING_UP
        elif td > 0 and hd > trend_thr:
            buy_tp,  buy_sl  = 1.8, 0.6
            sell_tp, sell_sl = 1.2, 1.2

        # TRENDING_DOWN
        elif td < 0 and hd < -trend_thr:
            buy_tp,  buy_sl  = 1.2, 1.2
            sell_tp, sell_sl = 1.4, 1.0

        # CHOPPY / RANGE
        else:
            buy_tp,  buy_sl  = 0.8, 1.2
            sell_tp, sell_sl = 0.8, 1.2

        # triple-barrier forward scan
        tp_up  = entry + buy_tp  * a
        sl_dn  = entry - buy_sl  * a
        tp_dn  = entry - sell_tp * a
        sl_up  = entry + sell_sl * a

        end = min(n, i + 1 + lookahead)
        buy_resolved = sell_resolved = False
        buy_win = sell_win = False

        for j in range(i + 1, end):
            hj, lj = high[j], low[j]
            if not buy_resolved:
                if lj <= sl_dn:
                    buy_resolved = True
                elif hj >= tp_up:
                    buy_resolved = True; buy_win = True
            if not sell_resolved:
                if hj >= sl_up:
                    sell_resolved = True
                elif lj <= tp_dn:
                    sell_resolved = True; sell_win = True
            if buy_resolved and sell_resolved:
                break

        if buy_win and not sell_win:
            labels[i] = 2
        elif sell_win and not buy_win:
            labels[i] = 0

    return pd.Series(labels, index=df.index, name="label")


def build_training_dataset(rates) -> Tuple[pd.DataFrame, pd.Series]:
    cfg = Config.section("trading")
    df = build_features(rates)

    use_regime = bool(cfg.get("asymmetric_labels_enabled", False))
    if use_regime:
        log.info("Using regime-aware asymmetric labeling (V4)")
        labels = regime_label(df, lookahead=int(cfg.get("label_lookahead", 12)))
    else:
        labels = aggressive_label(
            df,
            tp_atr=float(cfg.get("label_tp_atr", 1.6)),
            sl_atr=float(cfg.get("label_sl_atr", 0.8)),
            lookahead=int(cfg.get("label_lookahead", 12)),
        )
    df["label"] = labels
    df = df.dropna(subset=FEATURE_COLUMNS + ["label"])

    # Balance: downsample HOLD
    hold_ratio = float(cfg.get("label_hold_to_event_ratio", 2.0))
    df_buy = df[df["label"] == 2]
    df_sell = df[df["label"] == 0]
    df_hold = df[df["label"] == 1]
    n_events = len(df_buy) + len(df_sell)
    keep_hold = min(len(df_hold), int(n_events * hold_ratio))
    if 0 < keep_hold < len(df_hold):
        df_hold = df_hold.sample(n=keep_hold, random_state=42).sort_index()
    df_bal = pd.concat([df_buy, df_sell, df_hold]).sort_index()

    log.info("Hyper Dataset: BUY=%d SELL=%d HOLD=%d (hold_ratio=%.1fx)",
             len(df_buy), len(df_sell), len(df_hold), hold_ratio)
    X = df_bal[FEATURE_COLUMNS].astype(float)
    y = df_bal["label"].astype(int)
    # Set DatetimeIndex for downstream trade-augmentation matching
    if "time" in df_bal.columns:
        idx = pd.to_datetime(df_bal["time"], utc=True)
        X.index = idx
        y.index = idx
        df_bal.index = idx
    # 🆕 attach raw OHLC+ATR to X for profit simulation downstream
    # (เก็บไว้ใน X.attrs เพื่อไม่ทำลาย shape ของ features)
    try:
        X.attrs["df_raw"] = df_bal[["open", "high", "low", "close", "atr"]].copy()
    except Exception:
        pass
    return X, y


def detect_reversal(df: pd.DataFrame, idx: int, side_to_recover: str,
                    wick_atr_min: float = 0.5) -> bool:
    """
    Price-Action confirmation สำหรับ Martingale (SRD):
      ถ้า series เป็น BUY (ติดลบลง)  -> ต้องการ BULLISH reversal:
        แท่งปิด -> lower_wick / ATR >= wick_atr_min  AND  lower_wick > upper_wick
      ถ้า series เป็น SELL (ติดลบขึ้น) -> ต้องการ BEARISH reversal:
        upper_wick / ATR >= wick_atr_min AND upper_wick > lower_wick
    """
    if idx >= len(df) or idx < -len(df):
        return False
    row = df.iloc[idx]
    a = float(row.get("atr", 0) or 0)
    if a <= 0:
        return False
    o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    if side_to_recover.upper() == "BUY":
        return (lower_wick / a) >= wick_atr_min and lower_wick > upper_wick
    if side_to_recover.upper() == "SELL":
        return (upper_wick / a) >= wick_atr_min and upper_wick > lower_wick
    return False
