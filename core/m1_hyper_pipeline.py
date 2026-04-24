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
    df["vol_accel_3"] = vol / vol_recent.replace(0, np.nan)
    df["vol_spike_10"] = vol / vol.rolling(10, min_periods=10).mean().replace(0, np.nan)

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

    return df


# --------------------------------------------------------------------- labeling
def aggressive_label(
    df: pd.DataFrame,
    tp_atr: float = 0.6,
    sl_atr: float = 1.2,
    lookahead: int = 30,
) -> pd.Series:
    """
    Hyper-Aggressive Labeling (SRD):
      - TP = 0.6 * ATR (เล็ก) | SL = 1.2 * ATR (ใหญ่กว่า) -> WIN ครอบคลุม
      - สแกน lookahead 30 แท่ง M1 (~30 นาที)
      - 2 = BUY win, 0 = SELL win, 1 = HOLD (ทั้งคู่ไม่ชัดเจน หรือชน SL ก่อน)
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
                if tp_hit and sl_hit:
                    buy_resolved = True  # worst case
                elif tp_hit:
                    buy_resolved = True; buy_win = True
                elif sl_hit:
                    buy_resolved = True
            if not sell_resolved:
                tp_hit = lj <= tp_dn
                sl_hit = hj >= sl_up
                if tp_hit and sl_hit:
                    sell_resolved = True
                elif tp_hit:
                    sell_resolved = True; sell_win = True
                elif sl_hit:
                    sell_resolved = True
            if buy_resolved and sell_resolved:
                break

        # ทั้งสองฝั่ง win -> ambiguous -> hold
        if buy_win and not sell_win:
            labels[i] = 2
        elif sell_win and not buy_win:
            labels[i] = 0

    return pd.Series(labels, index=df.index, name="label")


def build_training_dataset(rates) -> Tuple[pd.DataFrame, pd.Series]:
    cfg = Config.section("trading")
    df = build_features(rates)
    labels = aggressive_label(
        df,
        tp_atr=float(cfg.get("label_tp_atr", 0.6)),
        sl_atr=float(cfg.get("label_sl_atr", 1.2)),
        lookahead=int(cfg.get("label_lookahead", 30)),
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
    return X, y


def latest_feature_row(rates, closed_index: int = -2) -> pd.Series:
    """ดึง feature ของแท่งปิดล่าสุด (default -2) สำหรับ inference."""
    df = build_features(rates)
    if len(df) < abs(closed_index) + 1:
        raise ValueError("Not enough bars for inference")
    row = df.iloc[closed_index]
    if row[FEATURE_COLUMNS].isna().any():
        raise ValueError("Feature row has NaN — need more warmup bars")
    return row[FEATURE_COLUMNS].astype(float)


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
