# 🧬 Feature Engineering — สถานะปัจจุบัน (58 features) + Roadmap

> **สถานะจริง (audit จาก `core/m1_hyper_pipeline.py`):** ระบบมี **58 features** ใน **12 มิติ**
> ชุด **Smart-Money / Order-Flow (10 features)** ลงโค้ดแล้ว — เหลือ **Market Regime (rg_*)** + RSI-Divergence/Streak แบบ explicit ใน roadmap

<p align="center">
  <img src="https://img.shields.io/badge/Current-58_features-brightgreen" />
  <img src="https://img.shields.io/badge/Done-SMC%2FOrderFlow-success" />
  <img src="https://img.shields.io/badge/Roadmap-Regime_(rg_*)-orange" />
  <img src="https://img.shields.io/badge/Status-Audit_Synced-success" />
</p>

---

## 📑 สารบัญ

1. [📊 Audit สถานะปัจจุบัน (58 features)](#-audit-สถานะปัจจุบัน-58-features)
2. [✅ มิติที่ทำเสร็จแล้ว (S/R + MomDyn + SMC)](#-มิติที่ทำเสร็จแล้ว-sr--momdyn--smc)
3. [🚧 Market Regime (ยังไม่ทำ — roadmap)](#-market-regime-ยังไม่ทำ--roadmap)
4. [🎓 Better Labels + Meta-Labeling](#-better-labels--meta-labeling)
5. [📋 Roadmap ถัดไป](#-roadmap-ถัดไป)

---

# 📊 Audit สถานะปัจจุบัน (58 features)

> 🔍 **เช็คตรงๆ จากโค้ด:**
> ```bash
> python -c "from core.m1_hyper_pipeline import FEATURE_COLUMNS; print(len(FEATURE_COLUMNS))"
> # → 58
> ```

## 🗂️ 12 มิติ × 58 Features

| # | มิติ | จำนวน | Prefix | ทำเมื่อ |
|:-:|---|:-:|---|:-:|
| 1 | Candle Anatomy | 4 | `body_*`, `*_wick_body_ratio`, `candle_direction` | V1 |
| 2 | Fast ATR Norm | 3 | `*_fast_atr` | V1 |
| 3 | Momentum (basic) | 5 | `price_velocity_*`, `rsi_7`, `ret_1`, `ema_dist_atr` | V1 |
| 4 | Volume | 2 | `vol_accel_3`, `vol_spike_10` | V1 |
| 5 | Micro Breakout | 4 | `breakout_*`, `near_*` | V1 |
| 6 | Volatility Regime | 1 | `atr_ratio` | V1 |
| 7 | Time Encoding | 3 | `time_sin/cos`, `session_score` | V1 |
| 8 | Multi-Timeframe | 6 | `htf1_*` (M15), `htf2_*` (H1) | V2 |
| 9 | Patterns | 10 | `pat_*` | V2 |
| 10 | **S/R Levels** ✅ | 5 | `sr_*` | V3 |
| 11 | **Momentum Dynamics** ✅ | 5 | `mom_velocity/body/wick/pattern_*` | V3 |
| 12 | **Smart-Money / Order-Flow** ✅ | 10 | `liq_sweep_*`, `mom_divergence`, `order_block_*`, `fvg_*`, `cum_delta_norm`, `vol_poc_dist`, `supply_demand_imbal` | **V4** |
|  | **รวม** | **58** | | |

## 🎯 Blueprint เดิม → สถานะจริง

| มิติใน Blueprint | สถานะ | หมายเหตุ |
|---|:-:|---|
| 📊 Order Flow Proxy | ✅ ทำแล้ว | ลงเป็นชุด SMC: `cum_delta_norm`, `vol_poc_dist`, `supply_demand_imbal`, `order_block_*`, `fvg_*`, `liq_sweep_*` |
| ⚡ Momentum Dynamics | ✅ ทำแล้ว | `mom_*` 5 ตัว + `mom_divergence` (RSI divergence แบบ explicit) |
| 📍 S/R Levels | ✅ ทำแล้ว | `sr_dist_pivot/round_*`, `sr_range_position_20` |
| 🌊 Market Regime (`rg_*`) | ❌ ยังไม่ทำ | trend_strength / chop_index / efficiency ฯลฯ — ดูส่วน roadmap |

> ⚠️ **เอกสารรุ่นก่อน** บอก "48 features, Order Flow ยังไม่ทำ" — **ล้าสมัยแล้ว** baseline จริง = 58 และชุด SMC/Order-Flow ลงโค้ดเรียบร้อย

---

# ✅ มิติที่ทำเสร็จแล้ว (S/R + MomDyn + SMC)

## 📍 มิติ 10: S/R Levels (5 features)

| Feature | ความหมาย |
|---|---|
| `sr_dist_pivot_high_50` | ห่าง resistance 50-bar กี่ ATR |
| `sr_dist_pivot_low_50` | ห่าง support 50-bar กี่ ATR |
| `sr_dist_round_50` | ห่างเลขกลม 50 (เช่น 4750, 4800) |
| `sr_dist_round_100` | ห่างเลขกลม 100 (สำคัญกว่า) |
| `sr_range_position_20` | ตำแหน่งใน range 20-bar (0=ก้น, 1=บน) |

## ⚡ มิติ 11: Momentum Dynamics (5 features)

| Feature | ความหมาย |
|---|---|
| `mom_velocity_accel` | velocity เร่ง/ชะลอ (continuation vs reversal) |
| `mom_body_accel` | candle body กำลังโตหรือเล็กลง |
| `mom_wick_imbalance` | seller บน vs buyer ล่าง |
| `mom_pattern_bull_5` / `mom_pattern_bear_5` | นับ pattern bull/bear ใน 5 bars |

## 🏦 มิติ 12: Smart-Money / Order-Flow (10 features) — 🆕 ใช้งานจริง

**ที่อยู่:** `core/m1_hyper_pipeline.py` บรรทัด ~370-430

| Feature | สูตร / ที่มา | ตีความ |
|---|---|---|
| `liq_sweep_bull` | swept prev-low 10 แล้วเด้ง × (ระยะลึก/ATR) | bear SL hunt → ราคาฟื้น = สัญญาณขึ้น |
| `liq_sweep_bear` | swept prev-high 10 แล้วร่วง × (ระยะ/ATR) | bull SL hunt → ราคาลง |
| `mom_divergence` | RSI vs price divergence (−1/0/+1) | +1 = bull div, −1 = bear div |
| `order_block_bull` | แท่งแดงก่อน impulse ขึ้นแรง | โซน demand institutional |
| `order_block_bear` | แท่งเขียวก่อน impulse ลงแรง | โซน supply |
| `fvg_bull` / `fvg_bear` | fair-value gap (low > high[2] / high < low[2]) | ช่องว่างที่ราคามักย้อนเติม |
| `cum_delta_norm` | (buy_vol − sell_vol) 20-bar normalize | + = แรงซื้อสะสม |
| `vol_poc_dist` | ระยะจาก VWAP/POC proxy (ATRs) | ห่าง fair value แค่ไหน |
| `supply_demand_imbal` | (bull_vol − bear_vol) / total_vol 20-bar | ความไม่สมดุล demand/supply |

> ✅ Forex retail ไม่มี order book จริง — ชุดนี้ใช้ **tick volume + price action** ประมาณ buy/sell pressure (proxy)

---

# 🚧 Market Regime (ยังไม่ทำ — roadmap)

## 🎬 อุปมา: ตกปลาทะเล ≠ ตกปลาบ่อ — ใช้เบ็ดต่างกัน

> 🎣 trending → breakout | 🌊 ranging → mean reversion | 🌪️ choppy → อย่าเทรด!

> 💡 **หมายเหตุ:** ปัจจุบัน "ความเป็น regime" ถูกใช้ใน **runtime filter** แล้ว (`core/regime_filter.py` — HTF trend block counter-trend + ห้าม recovery ตอน strong trend) แต่ยัง **ไม่ได้เป็น feature ป้อนเข้าโมเดล** ส่วนนี้คือการเพิ่ม `rg_*` เป็น input features

## 🧩 5 Features (เป้าหมาย)

| Feature | สูตร | ตีความ |
|---|---|---|
| `rg_trend_strength` | `\|EMA20 − EMA50\| / ATR` | > 2.0 = trending แรง |
| `rg_chop_index` | `100·log10(ΣTR14 / (max−min)) / log10(14)` | > 61.8 = choppy |
| `rg_efficiency` | `\|c_now − c_10ago\| / Σ\|return\|` | 1.0 = trend perfect, 0 = noise |
| `rg_range_width_atr` | `(max_high_20 − min_low_20) / ATR` | < 3 = squeeze (จะ breakout) |
| `rg_breakout_potential` | 1 ถ้า range_pos > 0.85 หรือ < 0.15 | ใกล้ขอบ range |

## 💻 โค้ดที่ต้องเพิ่ม (ใน `m1_hyper_pipeline.py` ก่อน `return df`)

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

**อย่าลืม:** เพิ่ม 5 ชื่อนี้ใน `FEATURE_COLUMNS` ด้วย!

---

# 🎓 Better Labels + Meta-Labeling

## 🅰️ สถานะ Label ปัจจุบัน

**ที่อยู่:** `core/m1_hyper_pipeline.py` — ใช้ Triple-Barrier **คงที่**: TP=1.6×ATR, SL=0.8×ATR, lookahead=12 แท่ง (ตรงกับ execution จริง)

`config.trading.asymmetric_labels_enabled` = `false` (regime-aware labeling เตรียมไว้แต่ยังปิด)

## 🅱️ Adaptive Triple-Barrier (ปรับตาม volatility — roadmap)

```python
def adaptive_label(df, lookahead=12, base_atr_mult=1.2):
    atr_norm = df["atr"] / df["atr"].rolling(50).mean()  # 1.0=normal, 2.0=double vol
    # tp/sl = base_atr_mult × vol_factor × atr  (ขยาย barrier ตอนผันผวน)
    ...
```

## 🅲 Meta-Labeling (López de Prado)

```
Primary Model → ทาย ทิศทาง (BUY/SELL/HOLD)
       ↓
Meta Model    → ทาย "ควรเทรดไหม?" (yes/no)
       ↓
ทั้งคู่ผ่าน → เทรด ✅
```

ผลลัพธ์ที่คาดหวัง: WR เพิ่ม 5-15%

---

# 📋 Roadmap ถัดไป

| ลำดับ | Task | Output |
|:-:|---|---|
| 1️⃣ | เพิ่ม `_add_regime()` → 58 → **63 features** | retrain + check feature importance |
| 2️⃣ | `mom_consec_same_dir` (unsigned streak) เสริม `mom_divergence` | decay-aware momentum |
| 3️⃣ | Adaptive Triple-Barrier (เปิด `asymmetric_labels`) | label ตรง regime |
| 4️⃣ | Meta-Labeling layer | กรอง false signal |
| 5️⃣ | Walk-forward A/B test เทียบ baseline 58 | roll out ถ้า ≥ +5pp |

## 🎯 Expected Progression

| Phase | Features | สถานะ |
|---|---:|:-:|
| V1 (Candle+Momentum+Time) | 33 | ✅ |
| V2 (+ HTF + Patterns) | 49 | ✅ |
| V3 (+ S/R + MomDyn) | 48-53 | ✅ |
| **V4 (+ SMC/Order-Flow)** ← **ปัจจุบัน** | **58** | ✅ |
| V5 (+ Regime `rg_*`) | 63 | 🚧 |
| V5+ (+ Adaptive Label + Meta) | 63+ | 🚧 |

---

<div align="center">

> **"เอกสารต้องตรงกับโค้ด — ถ้าไม่ตรง คือเอกสารโกหก** 🧬
> **ปัจจุบัน = 58 features (SMC done) → ถัดไป = Regime + Adaptive Label"**

**🧬 Audit-synced with `core/m1_hyper_pipeline.py` — verified: 58 features**

</div>
