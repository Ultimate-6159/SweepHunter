# 🧬 Feature Engineering V2 → V4 Roadmap

> **สถานะจริง (audit จาก `core/m1_hyper_pipeline.py`):** ระบบมี **48 features** ใน **11 มิติ** (ไม่ใช่ 38 ตามเอกสารเก่า)
> **เป้าหมาย V4:** เพิ่มอีก **15 features** (Order Flow + Regime + RSI Divergence) → **63 features** | accuracy 43% → **52-58%**

<p align="center">
  <img src="https://img.shields.io/badge/Current-48_features_(V3)-brightgreen" />
  <img src="https://img.shields.io/badge/Target_V4-63_features-blue" />
  <img src="https://img.shields.io/badge/Implemented-2%2F4_dimensions-orange" />
  <img src="https://img.shields.io/badge/Status-Audit_Synced-success" />
</p>

---

## 📑 สารบัญ

1. [📊 Audit สถานะปัจจุบัน (48 features)](#-audit-สถานะปัจจุบัน-48-features)
2. [✅ มิติ 3+4: ที่ทำเสร็จแล้วใน V3](#-มิติ-34-ที่ทำเสร็จแล้วใน-v3)
3. [🚧 มิติ 1: Order Flow Proxy (ยังไม่ทำ)](#-มิติ-1-order-flow-proxy-ยังไม่ทำ)
4. [🚧 มิติ 2: Market Regime (ยังไม่ทำ)](#-มิติ-2-market-regime-ยังไม่ทำ)
5. [🚧 ส่วนเสริม Momentum (RSI Div + Streak)](#-ส่วนเสริม-momentum-rsi-div--streak)
6. [🎓 Better Labels + Meta-Labeling](#-better-labels--meta-labeling)
7. [📋 Roadmap V4 (4 สัปดาห์)](#-roadmap-v4-4-สัปดาห์)
8. [⚠️ Risk Management ระหว่างเก็บ Data](#️-risk-management-ระหว่างเก็บ-data)

---

# 📊 Audit สถานะปัจจุบัน (48 features)

> 🔍 **เช็คตรงๆ จากโค้ด:**
> ```bash
> python -c "from core.m1_hyper_pipeline import FEATURE_COLUMNS; print(len(FEATURE_COLUMNS))"
> # → 48
> ```

## 🗂️ 11 มิติ × 48 Features

| # | มิติ | จำนวน | Prefix | ทำเมื่อ |
|:-:|---|:-:|---|:-:|
| 1 | Candle Anatomy | 4 | `body_*`, `*_wick_*` | V1 |
| 2 | Fast ATR Norm | 3 | `*_fast_atr` | V1 |
| 3 | Momentum (basic) | 5 | `price_velocity_*`, `rsi_7`, `ret_1`, `ema_dist_atr` | V1 |
| 4 | Volume | 2 | `vol_accel_3`, `vol_spike_10` | V1 |
| 5 | Micro Breakout | 4 | `breakout_*`, `near_*` | V1 |
| 6 | Volatility Regime | 1 | `atr_ratio` | V1 |
| 7 | Time Encoding | 3 | `time_sin/cos`, `session_score` | V1 |
| 8 | Multi-Timeframe | 6 | `htf1_*`, `htf2_*` | V2 |
| 9 | Patterns | 10 | `pat_*` | V2 |
| 10 | **S/R Levels** ✅ | 5 | `sr_*` | **V3** |
| 11 | **Momentum Dynamics** ✅ | 5 | `mom_*` | **V3** |
|  | **รวม** | **48** | | |

## 🎯 4 มิติใน Blueprint เดิม → สถานะจริง

```
                    🤖 AI ปัจจุบัน V3 (48 features)
                            │
        ┌───────────┬───────┴───────┬───────────┐
        ▼           ▼               ▼           ▼
   Order Flow    Regime         Levels      Momentum
       ❌           ❌            ✅           ✅
   (ยังไม่ทำ)  (ยังไม่ทำ)    (sr_* 5 ตัว) (mom_* 5 ตัว)
```

| มิติใน Blueprint V2 | สถานะ | เหตุผล/ที่อยู่ในโค้ด |
|---|:-:|---|
| 📊 Order Flow Proxy (5) | ❌ ยังไม่ทำ | ต้องเขียน function ใหม่ |
| 🌊 Market Regime (5) | ❌ ยังไม่ทำ | ต้องเขียน function ใหม่ |
| 📍 S/R Levels (5) | ✅ ทำแล้ว | `sr_dist_pivot_high/low_50`, `sr_dist_round_50/100`, `sr_range_position_20` |
| ⚡ Momentum Dynamics (5) | ✅ ทำแล้ว 4/5 | `mom_velocity_accel`, `mom_body_accel`, `mom_wick_imbalance`, `mom_pattern_bull/bear_5` (ยังขาด `rsi_divergence`, `consec_same_dir`) |

> ⚠️ **เอกสารรุ่นก่อนหน้า** บอกว่า "baseline 38 features → ต้องเพิ่ม 20" — **ผิด!** baseline จริง = 48 และ Levels+Momentum **ทำไปแล้ว** ในเวอร์ชัน V3

---

# ✅ มิติ 3+4: ที่ทำเสร็จแล้วใน V3

## 📍 มิติ 10: S/R Levels (5 features) — ใช้งานจริงในโค้ด

**ที่อยู่:** `core/m1_hyper_pipeline.py` บรรทัด ~318-338

```python
# ----- 🆕 V3: Support/Resistance + Multi-Pattern Aggregation -----
atr_safe2 = df["atr"].replace(0, np.nan)

pivot_high_50 = h.shift(1).rolling(50, min_periods=20).max()
pivot_low_50  = l.shift(1).rolling(50, min_periods=20).min()
df["sr_dist_pivot_high_50"] = ((pivot_high_50 - c) / atr_safe2).clip(-10, 10).fillna(0)
df["sr_dist_pivot_low_50"]  = ((c - pivot_low_50) / atr_safe2).clip(-10, 10).fillna(0)

round_50  = (c / 50.0 ).round() * 50.0
round_100 = (c / 100.0).round() * 100.0
df["sr_dist_round_50"]  = ((c - round_50 ).abs() / atr_safe2).clip(0, 10).fillna(1.0)
df["sr_dist_round_100"] = ((c - round_100).abs() / atr_safe2).clip(0, 10).fillna(1.0)

rh20 = h.rolling(20, min_periods=10).max()
rl20 = l.rolling(20, min_periods=10).min()
rng20 = (rh20 - rl20).replace(0, np.nan)
df["sr_range_position_20"] = ((c - rl20) / rng20).clip(0, 1).fillna(0.5)
```

| Feature | ความหมาย |
|---|---|
| `sr_dist_pivot_high_50` | ห่าง resistance 50-bar กี่ ATR |
| `sr_dist_pivot_low_50` | ห่าง support 50-bar กี่ ATR |
| `sr_dist_round_50` | ห่างเลขกลม 50 (เช่น 4750, 4800) |
| `sr_dist_round_100` | ห่างเลขกลม 100 (สำคัญกว่า) |
| `sr_range_position_20` | ตำแหน่งใน range 20-bar (0=ก้น, 1=บน) |

## ⚡ มิติ 11: Momentum Dynamics (5 features) — ใช้งานจริง

**ที่อยู่:** `core/m1_hyper_pipeline.py` บรรทัด ~340-356

```python
df["mom_velocity_accel"] = (velocity_5_safe - velocity_5_safe.shift(1)).fillna(0)
df["mom_body_accel"]     = (body_atr_safe - body_atr_safe.shift(1)).fillna(0)

wick_total = (upper_wick + lower_wick + 1e-6)
df["mom_wick_imbalance"] = ((upper_wick - lower_wick) / wick_total).fillna(0)

bull_patterns = df["pat_engulf_bull"] + df["pat_swept_low_5"]
bear_patterns = df["pat_engulf_bear"] + df["pat_swept_high_5"]
df["mom_pattern_bull_5"] = bull_patterns.rolling(5, min_periods=1).sum().fillna(0)
df["mom_pattern_bear_5"] = bear_patterns.rolling(5, min_periods=1).sum().fillna(0)
```

| Feature | ความหมาย |
|---|---|
| `mom_velocity_accel` | velocity เร่ง/ชะลอ (continuation vs reversal) |
| `mom_body_accel` | candle body กำลังโตหรือเล็กลง |
| `mom_wick_imbalance` | seller บน vs buyer ล่าง |
| `mom_pattern_bull_5` | นับ pattern bullish ใน 5 bars |
| `mom_pattern_bear_5` | นับ pattern bearish ใน 5 bars |

> ✅ **2 ใน 4 มิติของ V2 blueprint ลงโค้ดแล้ว** — ยังเหลือ Order Flow + Regime ในแผน V4

---

# 🚧 มิติ 1: Order Flow Proxy (ยังไม่ทำ)

## 🎬 อุปมา: ดูสมรภูมิรบจากภาพถ่าย

> Forex retail ไม่มี order book จริง — แต่ใช้ **tick volume + price action** ประมาณการณ์ buy/sell pressure ได้

## 🧩 5 Features (เป้าหมาย V4)

| Feature | สูตร | ตีความ |
|---|---|---|
| `of_buy_pressure_3` | `Σ(volume × bullish_dir) / Σ(volume)` ใน 3 bars | > 0.7 = institutional buying |
| `of_sell_pressure_3` | `1 - of_buy_pressure_3` | symmetric |
| `of_vol_at_high_ratio` | volume of bars closing near high / total vol(10) | > 0.4 = real buying (ไม่ใช่ wick fake) |
| `of_vol_at_low_ratio` | volume of bars closing near low / total vol(10) | symmetric |
| `of_delta_proxy` | `((c - midpoint)/hl_range) × vol_normalized` | direction × intensity |

## 💻 โค้ดที่ต้องเพิ่ม (ใส่ใน `m1_hyper_pipeline.py` ก่อน `return df`)

```python
def _add_order_flow(df: pd.DataFrame) -> None:
    direction = np.sign(df["close"] - df["open"])
    vol = df["tick_volume"].astype(float)

    df["of_buy_pressure_3"] = (
        (vol * (direction > 0)).rolling(3).sum()
        / vol.rolling(3).sum().replace(0, 1)
    ).fillna(0.5)
    df["of_sell_pressure_3"] = 1.0 - df["of_buy_pressure_3"]

    hl = (df["high"] - df["low"]).replace(0, np.nan)
    near_high = (df["close"] >= df["high"] - 0.2 * hl).astype(float)
    near_low  = (df["close"] <= df["low"]  + 0.2 * hl).astype(float)
    df["of_vol_at_high_ratio"] = (vol * near_high).rolling(10).sum() / vol.rolling(10).sum().replace(0, 1)
    df["of_vol_at_low_ratio"]  = (vol * near_low ).rolling(10).sum() / vol.rolling(10).sum().replace(0, 1)

    bias = (df["close"] - (df["high"] + df["low"]) / 2) / hl
    vol_norm = vol / vol.rolling(20).mean().replace(0, 1)
    df["of_delta_proxy"] = (bias * vol_norm).clip(-5, 5).fillna(0)
```

**อย่าลืม:** เพิ่ม 5 ชื่อนี้ใน `FEATURE_COLUMNS` ด้วย!

---

# 🚧 มิติ 2: Market Regime (ยังไม่ทำ)

## 🎬 อุปมา: ตกปลาทะเล ≠ ตกปลาบ่อ — ใช้เบ็ดต่างกัน

> 🎣 trending → breakout strategy
> 🌊 ranging → mean reversion
> 🌪️ choppy → อย่าเทรด!

## 🧩 5 Features (เป้าหมาย V4)

| Feature | สูตร | ตีความ |
|---|---|---|
| `rg_trend_strength` | `\|EMA20 − EMA50\| / ATR` | > 2.0 = trending แรง |
| `rg_chop_index` | `100·log10(ΣTR14 / (max-min)) / log10(14)` | > 61.8 = choppy |
| `rg_efficiency` | `\|c_now − c_10ago\| / Σ\|return\|` | 1.0 = trend perfect, 0 = noise |
| `rg_range_width_atr` | `(max_high_20 − min_low_20) / ATR` | < 3 = squeeze (จะ breakout!) |
| `rg_breakout_potential` | 1 if range_pos > 0.85 หรือ < 0.15 | ใกล้ขอบ range |

## 💻 โค้ดที่ต้องเพิ่ม

```python
def _add_regime(df: pd.DataFrame) -> None:
    c = df["close"]
    atr = df["atr"].replace(0, np.nan)

    ema20 = c.ewm(span=20, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()
    df["rg_trend_strength"] = ((ema20 - ema50).abs() / atr).clip(0, 10).fillna(0)

    sum_tr = atr.rolling(14).sum()
    hi_max = df["high"].rolling(14).max()
    lo_min = df["low"].rolling(14).min()
    df["rg_chop_index"] = (
        100 * np.log10(sum_tr / (hi_max - lo_min).replace(0, np.nan)) / np.log10(14)
    ).clip(0, 100).fillna(50)

    direction  = (c - c.shift(10)).abs()
    volatility = c.diff().abs().rolling(10).sum()
    df["rg_efficiency"] = (direction / volatility.replace(0, np.nan)).clip(0, 1).fillna(0.3)

    df["rg_range_width_atr"] = (
        (df["high"].rolling(20).max() - df["low"].rolling(20).min()) / atr
    ).clip(0, 30).fillna(5)

    rt = df["high"].rolling(20).max()
    rb = df["low"].rolling(20).min()
    range_pos = (c - rb) / (rt - rb).replace(0, np.nan)
    df["rg_breakout_potential"] = ((range_pos > 0.85) | (range_pos < 0.15)).astype(float).fillna(0)
```

---

# 🚧 ส่วนเสริม Momentum (RSI Div + Streak)

ขาด 2 features สุดท้ายจาก Momentum Dynamics blueprint:

```python
def _add_momentum_extra(df: pd.DataFrame) -> None:
    c = df["close"]
    rsi = _rsi(c, 7)
    price_up = (c   > c.shift(5)).astype(int)
    rsi_up   = (rsi > rsi.shift(5)).astype(int)
    df["mom_rsi_divergence"] = (price_up != rsi_up).astype(float)

    direction = np.sign(c - df["open"])
    same = (direction == direction.shift(1)).astype(int)
    streak = same.groupby((same != same.shift(1)).cumsum()).cumsum()
    df["mom_consec_same_dir"] = streak.fillna(0).clip(0, 10)
```

> 💡 **`pat_consec_streak`** มีอยู่แล้วในโค้ด (signed) — ต่างกับ `mom_consec_same_dir` ที่ไม่มีเครื่องหมาย แต่ measure decay ได้ดีกว่า

---

# 🎓 Better Labels + Meta-Labeling

## 🅰️ Adaptive Triple-Barrier (ปรับตาม volatility)

**ที่อยู่ปัจจุบัน:** `aggressive_label()` ใช้ TP/SL **คงที่** = 0.6/1.2 × ATR

```python
def adaptive_label(df, lookahead=24, base_atr_mult=1.5):
    n = len(df)
    labels = np.full(n, 1, dtype=np.int8)
    atr_norm = df["atr"] / df["atr"].rolling(50).mean()  # 1.0=normal, 2.0=double vol

    for i in range(n - lookahead):
        a = df["atr"].iloc[i]
        if not np.isfinite(a) or a <= 0: continue

        vol_factor = atr_norm.iloc[i] if np.isfinite(atr_norm.iloc[i]) else 1.0
        tp = base_atr_mult * vol_factor * a
        sl = base_atr_mult * vol_factor * a

        entry = df["close"].iloc[i]
        upper, lower = entry + tp, entry - sl
        for j in range(i+1, i+lookahead+1):
            if df["high"].iloc[j] >= upper: labels[i] = 2; break  # BUY win
            if df["low"].iloc[j]  <= lower: labels[i] = 0; break  # SELL win
    return labels
```

## 🅱️ Meta-Labeling (López de Prado)

```
Primary Model → ทาย ทิศทาง (BUY/SELL/HOLD)
       ↓
Meta Model    → ทาย "ควรเทรดไหม?" (yes/no)
       ↓
ทั้งคู่ผ่าน → เทรด ✅
```

ผลลัพธ์: WR เพิ่ม 5-15%

---

# 📋 Roadmap V4 (4 สัปดาห์)

```
Week 1     Week 2      Week 3        Week 4
  📥         🔍          🛠️            🧪
Collect → Analyze → Implement → Validate
```

| สัปดาห์ | Task | Output |
|---|---|---|
| 1️⃣ | Data collection (`data_collection_mode: true`, lot 0.01) | 500-1000 trades ใน `hyper_trades.sqlite` |
| 2️⃣ | Analyze: WR by hour/spread/atr/regime | รู้ว่า model **เสียตอนไหน + ทำไม** |
| 3️⃣ | เพิ่ม `_add_order_flow` + `_add_regime` + `_add_momentum_extra` ใน `m1_hyper_pipeline.py` → 48 → **63 features** | retrain + check feature importance |
| 4️⃣ | Walk-forward backtest 3 เดือน → A/B test V3 vs V4 | ถ้า V4 > V3 ≥ 5pp → roll out |

## 🎯 Expected Progression

| Phase | Features | Accuracy เป้า | Drawdown |
|---|---:|---:|---:|
| 🟡 V1 | 33 | 38% | -18% |
| 🟢 V2 (HTF + Patterns) | 49 | 41% | -16% |
| 🟢 **V3 (S/R + MomDyn)** ← **ปัจจุบัน** | **48** | **43%** | **-15%** |
| 🟢 V4a (+ Order Flow) | 53 | 46% | -12% |
| 🟢 V4b (+ Regime) | 58 | 49% | -10% |
| 🌟 **V4c (+ Mom Extra + Adaptive Label)** | **63** | **52-58%** | **-5 ถึง -8%** |

---

# ⚠️ Risk Management ระหว่างเก็บ Data

## 💸 Math พื้นฐาน

```
balance $500 × 0.10% risk = $0.50/trade
Worst case: เสีย 100 trades ติด = -$50 (10%)
+ commission $7/lot × 100 = $7
รวม worst case: -$57 (~11.4%)

📊 Realistic case (random signal):
WR ~ 33-40% → net loss/trade ~ $0.30
1000 trades × -$0.30 = -$300 → ❌ blow account 60%!
```

## 🛡️ Safeguards (ใน `config.json` แล้ว)

| Layer | Setting | ป้องกัน |
|---|---|---|
| 🎯 Risk per trade | `0.10%` | ความเสียหายต่อไม้ |
| 📏 Max lot % | `2.0%` | exposure |
| 🔒 Max lot cap | `0.05` | ฮาร์ดเพดาน |
| ♻️ Recovery steps | `2` | ขุดหลุมตัวเอง |
| 🛑 Equity stop | `15%` | safety net |
| ⏸️ Halt cooldown | `5 min` | อย่ารีบ |

## 💡 คำแนะนำ

```
1. ใช้ Demo เท่านั้น 🚨
2. Topup ทุกสัปดาห์ (back to $500)
3. ดู review_trades.py ทุกวัน
4. ถ้า DD > 10% → หยุด, debug
```

---

<div align="center">

## 🌟 คำคมส่งท้าย

> **"เอกสารต้องตรงกับโค้ด — ถ้าไม่ตรง คือเอกสารโกหก** 🧬
> **V3 = 48 features (ปัจจุบัน) → V4 = 63 features (เป้าหมาย)"**

---

### 🎓 Next Steps

1. ✅ อ่านเอกสารนี้จบ → เข้าใจว่าอะไร "ทำแล้ว" vs "ยังไม่ทำ"
2. 🛠️ ลอง implement `_add_order_flow()` ก่อน (ง่ายสุด, impact ชัด)
3. 🧪 A/B test V3 (48) vs V3 + OrderFlow (53)
4. 📊 ดู feature importance — ถ้า of_* ติด top-15 → success
5. 🔄 ทำ Regime → Adaptive Label → Meta ตามลำดับ

---

**🧬 Audit-synced with `core/m1_hyper_pipeline.py` — Last verified: V3 (48 features)**

<sub>Blueprint v4.0 — Order Flow + Regime stack ready to implement</sub>

</div>
