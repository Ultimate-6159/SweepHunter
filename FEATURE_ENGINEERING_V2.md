# 🧬 Feature Engineering V2 — ยกระดับ AI ให้ฉลาดขึ้น

> **เป้าหมาย:** เพิ่ม accuracy จาก **43% → 55-58%** โดยการ "เปิดตา" ใหม่ให้ AI 20 คู่
> **เวลาอ่าน:** 30-45 นาที | **ระดับ:** Beginner-Friendly + Deep Dive

<p align="center">
  <img src="https://img.shields.io/badge/Target-+15%25_Accuracy-brightgreen" />
  <img src="https://img.shields.io/badge/New_Features-20-blue" />
  <img src="https://img.shields.io/badge/Dimensions-4-orange" />
  <img src="https://img.shields.io/badge/Status-Blueprint-yellow" />
</p>

---

## 📑 สารบัญ

1. [🤔 ทำไม Model ปัจจุบันแม่นแค่ 43%?](#-ทำไม-model-ปัจจุบันแม่นแค่-43)
2. [🎯 4 มิติใหม่ที่จะเพิ่ม](#-4-มิติใหม่ที่จะเพิ่ม)
3. [📊 มิติ 1: Order Flow Proxy](#-มิติ-1-order-flow-proxy-เห็นแรงซื้อ-แรงขาย)
4. [📊 มิติ 2: Market Regime](#-มิติ-2-market-regime-รู้ว่าตลาดอารมณ์ไหน)
5. [📊 มิติ 3: Support/Resistance](#-มิติ-3-supportresistance-รู้แนวต้าน-แนวรับ)
6. [📊 มิติ 4: Momentum Dynamics](#-มิติ-4-momentum-dynamics-รู้ว่าโมเมนตัมเร่งหรือชะลอ)
7. [🎓 Bonus: Better Labels](#-bonus-better-labels)
8. [📋 Roadmap 4 สัปดาห์](#-roadmap-4-สัปดาห์)
9. [⚠️ Risk Management](#️-risk-management-ระหว่างเก็บ-data)

---

# 🤔 ทำไม Model ปัจจุบันแม่นแค่ 43%?

## 🎬 เปรียบเทียบ: AI = นักเรียนที่ตาบอดสีบางสี

> ลองนึกภาพ: คุณให้นักเรียน**ตาบอดสีเขียว-แดง** มาวาดรูปต้นไม้
> เขาวาดได้ — แต่ใบไม้กับลำต้นจะดูเหมือนกัน → ผลงานออกมา**ผิด**

**AI ของเราตอนนี้ก็เป็นแบบนั้น** — มี 38 features แต่ยัง "มองไม่เห็น" บางอย่างที่สำคัญ

## 🔍 ช่องโหว่ที่เจอจาก Trade History

| 🚨 ปัญหา | 😕 อาการ | 💡 สาเหตุ |
|---|---|---|
| **เข้าตอนเทรนด์จบ** | BUY ที่ top, SELL ที่ bottom | ❌ ไม่มี mean-reversion features |
| **ไม่รู้ market regime** | strategy เดิมใช้ทั้ง trending+ranging | ❌ ไม่มี regime classifier |
| **ไม่เห็น orderflow** | ไม่รู้ว่า big money ทำอะไร | ❌ มีแค่ tick volume ดิบ |
| **ไม่รู้ S/R levels** | ชน level แล้วเด้ง = SL hit | ❌ ไม่มี pivot levels |
| **เมิน round numbers** | ราคาเด้งที่ 4700, 4750, 4800 | ❌ ไม่มี psychological levels |
| **ไม่ track momentum decay** | momentum กำลังจะหมด | ❌ มีแค่ velocity ปัจจุบัน |

> 💡 **Insight:** AI ตอนนี้ **"เห็นภาพ"** แต่ **"ไม่เข้าใจบริบท"** → V2 จะเติมบริบทให้

---

# 🎯 4 มิติใหม่ที่จะเพิ่ม

```
                    🤖 AI V2
                       │
     ┌─────────┬───────┼───────┬─────────┐
     ▼         ▼       ▼       ▼         ▼
   38 เก่า  📊 OFlow 🌊 Regime 📍 Levels ⚡ Momentum
            (5 ตัว)  (5 ตัว)  (5 ตัว)   (5 ตัว)
                  ↓
              รวม 58 features
                  ↓
              accuracy 55-58% 🎯
```

| มิติ | Features | ตอบคำถามว่า... |
|---|:---:|---|
| 📊 **Order Flow** | 5 | "ใครซื้อ ใครขาย แรงแค่ไหน?" |
| 🌊 **Regime** | 5 | "ตลาดเทรนด์ ranging หรือ choppy?" |
| 📍 **Levels** | 5 | "ราคาใกล้ S/R ไหม?" |
| ⚡ **Momentum** | 5 | "โมเมนตัมเร่งขึ้นหรือชะลอ?" |

---

# 📊 มิติ 1: Order Flow Proxy (เห็นแรงซื้อ-แรงขาย)

## 🎬 อุปมา: ดูสมรภูมิรบจากภาพถ่าย

> Forex retail ไม่มี order book จริง (ต่างจาก crypto)
> แต่เราใช้ **tick volume + price action** ประมาณการณ์ได้ — เหมือนนักสืบดูร่องรอย

## 🧩 5 Features

### 1️⃣ `buy_pressure_3` — แรงซื้อใน 3 บาร์ล่าสุด

```python
buy_pressure = Σ(volume × ทิศทางขาขึ้น) / Σ(volume)
# 1.0 = ทุกบาร์ขาขึ้น volume สูง
# 0.5 = balanced
# 0.0 = ทุกบาร์ขาลง
```

**ตีความ:** > 0.7 = institutional buying, < 0.3 = institutional selling

### 2️⃣ `sell_pressure_3` — แรงขาย (สมมาตร)
```python
sell_pressure = 1 - buy_pressure
```

### 3️⃣ `vol_at_high_ratio` — สะสมที่ high

```python
# Volume ของบาร์ที่ปิดใกล้ high / total volume 10 บาร์
```
> 💡 **ตีความ:** > 0.4 = แรงซื้อจริง (ไม่ใช่แค่ wick fake)

### 4️⃣ `vol_at_low_ratio` — สะสมที่ low (สมมาตร)

### 5️⃣ `delta_proxy` — bullish/bearish bias × intensity

```python
bias = (close - midpoint) / hl_range   # -1 ถึง +1
delta = bias × volume_normalized       # มี bias + แรงเยอะ
```

## 💻 โค้ดจริง

```python
def order_flow_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    direction = np.sign(df["close"] - df["open"])
    vol = df["tick_volume"].astype(float)

    # 1+2: buy/sell pressure
    out["buy_pressure_3"] = (
        (vol * (direction > 0)).rolling(3).sum()
        / vol.rolling(3).sum().replace(0, 1)
    )
    out["sell_pressure_3"] = 1.0 - out["buy_pressure_3"]

    # 3+4: volume at high/low
    near_high = (df["close"] >= df["high"] - 0.2 * (df["high"] - df["low"])).astype(float)
    near_low  = (df["close"] <= df["low"]  + 0.2 * (df["high"] - df["low"])).astype(float)
    out["vol_at_high_ratio"] = (vol * near_high).rolling(10).sum() / vol.rolling(10).sum().replace(0, 1)
    out["vol_at_low_ratio"]  = (vol * near_low ).rolling(10).sum() / vol.rolling(10).sum().replace(0, 1)

    # 5: delta proxy
    hl_range = (df["high"] - df["low"]).replace(0, np.nan)
    bias = (df["close"] - (df["high"] + df["low"]) / 2) / hl_range
    out["delta_proxy"] = (bias * vol / vol.rolling(20).mean().replace(0, 1)).fillna(0)

    return out
```

> ✨ **ทำไมสำคัญ:** จับ "smart money footprints" ที่ price action เพียวๆ มองไม่เห็น

---

# 📊 มิติ 2: Market Regime (รู้ว่าตลาดอารมณ์ไหน)

## 🎬 อุปมา: ตกปลาทะเลกับตกปลาบ่อ — ใช้เบ็ดต่างกัน

> 🎣 ตลาด **trending** = วิ่งทางเดียวยาวๆ → strategy = breakout
> 🌊 ตลาด **ranging** = วิ่งกรอบ → strategy = mean reversion
> 🌪️ ตลาด **choppy** = noise มาก → strategy = อย่าเทรด!

**ปัจจุบัน AI ใช้ strategy เดียวทุก regime → ผิดทางครึ่งหนึ่ง**

## 🧩 5 Features

### 1️⃣ `regime_trend_strength` — เทรนด์แรงไหม

```python
strength = |EMA20 - EMA50| / ATR
# > 2.0 = trending แรงมาก
# < 0.5 = ranging
```

### 2️⃣ `regime_chop_index` — Choppiness (สูตรของ Bill Dreiss)

```python
chop = 100 × log10(Σ_ATR_14 / (max_high - min_low)) / log10(14)
# > 61.8 = choppy (อย่าเทรด trend)
# < 38.2 = trending ชัด
```

### 3️⃣ `regime_efficiency` — Kaufman Efficiency Ratio

```python
efficiency = |close_now - close_10_ago| / Σ|return|
# 1.0 = trend สมบูรณ์
# 0.0 = noise ล้วน
```

> 🎓 **Insight:** ถ้า efficiency < 0.3 → ตลาดเดินมั่ว → AI confidence ต่ำลง

### 4️⃣ `regime_range_width_atr` — กว้างกี่ ATR

```python
width = (high_max_20 - low_min_20) / ATR
# > 8 = range กว้าง (โอกาสกำไรเยอะ)
# < 3 = range แคบ (squeeze → breakout!)
```

### 5️⃣ `regime_breakout_potential` — จะทะลุไหม

```python
range_pos = (close - range_bottom) / (range_top - range_bottom)
# 1 ถ้า range_pos > 0.85 หรือ < 0.15 (ใกล้ขอบ)
```

## 💻 โค้ดจริง

```python
def regime_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    c = df["close"]
    atr = df["atr"]

    # 1: trend strength
    ema20 = c.ewm(span=20, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()
    out["regime_trend_strength"] = (ema20 - ema50).abs() / atr.replace(0, np.nan)

    # 2: Choppiness Index
    sum_tr = atr.rolling(14).sum()
    high_max = df["high"].rolling(14).max()
    low_min  = df["low"].rolling(14).min()
    out["regime_chop_index"] = 100 * np.log10(sum_tr / (high_max - low_min).replace(0, np.nan)) / np.log10(14)

    # 3: Kaufman Efficiency
    direction  = (c - c.shift(10)).abs()
    volatility = c.diff().abs().rolling(10).sum()
    out["regime_efficiency"] = direction / volatility.replace(0, np.nan)

    # 4: range width
    out["regime_range_width_atr"] = (df["high"].rolling(20).max() - df["low"].rolling(20).min()) / atr.replace(0, np.nan)

    # 5: breakout potential
    range_top = df["high"].rolling(20).max()
    range_bot = df["low"].rolling(20).min()
    range_pos = (c - range_bot) / (range_top - range_bot).replace(0, np.nan)
    out["regime_breakout_potential"] = ((range_pos > 0.85) | (range_pos < 0.15)).astype(float)

    return out
```

> ✨ **ทำไมสำคัญ:** AI เรียนได้ว่า "feature X ใช้ได้ใน trending แต่ไม่ใช้ใน ranging"

---

# 📊 มิติ 3: Support/Resistance (รู้แนวต้าน-แนวรับ)

## 🎬 อุปมา: เพดาน-พื้นบ้าน — ลูกบอลเด้งเสมอ

> 🏠 ทุกห้องมี **เพดาน** (resistance) และ **พื้น** (support)
> 🏀 ลูกบอลเด้งจากพื้น พุ่งไปเพดาน → เด้งกลับ
> นักเทรดรู้ตำแหน่งนี้ → ตั้ง order เด้ง/ทะลุ

**AI ปัจจุบันไม่รู้ "พื้น/เพดาน" → ยิงตอนชน level → SL hit ทันที!**

## 🧩 5 Features

### 1️⃣ `dist_to_pivot_high` — ห่าง resistance กี่ ATR

```python
pivot_high = df["high"].rolling(50).max()
dist = (pivot_high - close) / ATR
# 0 = ชนพอดี (อันตราย!)
# 3+ = ห่างมาก (ปลอดภัย)
```

### 2️⃣ `dist_to_pivot_low` — ห่าง support กี่ ATR (สมมาตร)

### 3️⃣ `dist_to_round_50` — ห่างเลขกลม 50

```python
# ทอง: 4700, 4750, 4800 = magnetic levels
round_50 = round(close / 50) × 50
dist = |close - round_50| / ATR
```

> 💡 **ทำไมสำคัญ:** retail/algo ตั้ง stop ที่เลขกลม → ราคาเด้ง/ทะลุที่นี่

### 4️⃣ `dist_to_round_100` — เลขกลม 100 (สำคัญกว่า)

### 5️⃣ `pivot_test_count` — level โดน test กี่ครั้ง

```python
# ยิ่ง test มาก → level ยิ่งสำคัญ → ยิ่งโอกาสเด้ง/ทะลุแรง
proximity = 0.3 × ATR
near_pivot = (high ≥ pivot - proximity) & (high ≤ pivot + proximity)
count = near_pivot.rolling(50).sum()
```

## 💻 โค้ดจริง

```python
def level_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    c = df["close"]
    atr = df["atr"].replace(0, np.nan)

    # 1+2: pivot distances
    pivot_high = df["high"].rolling(50).max()
    pivot_low  = df["low"].rolling(50).min()
    out["dist_to_pivot_high"] = (pivot_high - c) / atr
    out["dist_to_pivot_low"]  = (c - pivot_low) / atr

    # 3+4: round number distances
    round_50  = (c / 50 ).round() * 50
    round_100 = (c / 100).round() * 100
    out["dist_to_round_50"]  = (c - round_50 ).abs() / atr
    out["dist_to_round_100"] = (c - round_100).abs() / atr

    # 5: pivot test count
    proximity = 0.3 * atr
    near_high = (
        (df["high"] >= pivot_high - proximity) &
        (df["high"] <= pivot_high + proximity)
    ).astype(int)
    out["pivot_test_count"] = near_high.rolling(50).sum().fillna(0)

    return out
```

> ✨ **ทำไมสำคัญ:** AI เลิกยิงที่ resistance/support → ลด stop hunt loss ~30%

---

# 📊 มิติ 4: Momentum Dynamics (รู้ว่าโมเมนตัมเร่งหรือชะลอ)

## 🎬 อุปมา: รถวิ่งแรง vs รถกำลังหยุด

> 🏎️ รถวิ่ง 120 km/h **+ เร่ง** → ยังไปได้อีกไกล (continuation)
> 🚗 รถวิ่ง 120 km/h **+ ชะลอ** → กำลังจะหยุด/กลับ (reversal)

**AI ปัจจุบันรู้แค่ "ตอนนี้เร็วเท่าไร" — ไม่รู้ว่า "เร่งหรือชะลอ"!**

## 🧩 5 Features

### 1️⃣ `velocity_acceleration` — เร่งหรือชะลอ?

```python
velocity_5 = (close - close_5_ago) / ATR
accel = velocity_5 - velocity_5.shift(1)
# > 0 = เร่ง (continuation)
# < 0 = ชะลอ (reversal coming)
```

### 2️⃣ `rsi_divergence` — RSI Divergence (Classic Signal!)

```python
# Bearish divergence: ราคาขึ้น แต่ RSI ลง → จะลง!
price_up = close > close_5_ago
rsi_up   = rsi   > rsi_5_ago
divergence = (price_up != rsi_up)   # 1 if diverge
```

> 🎓 **Classic technical analysis** — ใช้ได้ผลมา 50+ ปี

### 3️⃣ `consec_same_dir_bars` — บาร์ทิศเดียวกันติดกี่บาร์

```python
# > 5 = exhausted, จะกลับตัว (mean reversion)
streak = consecutive_same_direction_count
```

### 4️⃣ `body_acceleration` — แท่งใหญ่ขึ้นไหม

```python
body = |close - open| / ATR
accel = body - body.shift(1)
# > 0 = momentum กำลังมา
```

### 5️⃣ `wick_imbalance` — ใส้บน vs ใส้ล่าง

```python
imbalance = (upper_wick - lower_wick) / (upper + lower + ε)
# > 0 = seller บน (rejection ขาขึ้น)
# < 0 = buyer ล่าง (rejection ขาลง)
```

## 💻 โค้ดจริง

```python
def momentum_dynamics(df: pd.DataFrame, rsi_period: int = 7) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    c = df["close"]
    atr_safe = df["atr"].replace(0, np.nan)

    # 1: velocity acceleration
    velocity_5 = (c - c.shift(5)) / atr_safe
    out["velocity_acceleration"] = velocity_5 - velocity_5.shift(1)

    # 2: RSI divergence
    rsi = _rsi(c, rsi_period)
    price_up = (c   > c.shift(5)).astype(int)
    rsi_up   = (rsi > rsi.shift(5)).astype(int)
    out["rsi_divergence"] = (price_up != rsi_up).astype(float)

    # 3: consecutive same direction
    direction = np.sign(c - df["open"])
    same = (direction == direction.shift(1)).astype(int)
    streak = same.groupby((same != same.shift(1)).cumsum()).cumsum()
    out["consec_same_dir_bars"] = streak.fillna(0).clip(0, 10)

    # 4: body acceleration
    body = (c - df["open"]).abs() / atr_safe
    out["body_acceleration"] = body - body.shift(1)

    # 5: wick imbalance
    upper_wick = df["high"] - np.maximum(df["open"], c)
    lower_wick = np.minimum(df["open"], c) - df["low"]
    total = upper_wick + lower_wick + 1e-6
    out["wick_imbalance"] = (upper_wick - lower_wick) / total

    return out
```

> ✨ **ทำไมสำคัญ:** จับ "turning points" ก่อนที่จะเกิด → entry ดีกว่า

---

# 🎓 Bonus: Better Labels

> 💡 **Insight ลับ:** บางครั้ง model แม่นต่ำไม่ใช่เพราะ feature แย่ — เพราะ **label ผิด!**

## 🅰️ Triple-Barrier with Volatility Targeting

### ❌ ปัญหาแบบเดิม
```python
tp = 1.5 × atr   # ใช้ atr ปัจจุบันเสมอ
```
→ ตลาด **low vol** TP ไกลเกิน (ไม่แตะ) | ตลาด **high vol** TP ใกล้เกิน (โดน noise)

### ✅ Solution: ปรับ TP/SL ตาม volatility regime

```python
def adaptive_label(df, lookahead=24, base_atr_mult=1.5):
    """
    TP/SL ปรับตาม volatility regime:
    - Low vol  → tight TP  (0.8 ATR)
    - High vol → wider TP  (2.0 ATR)
    """
    n = len(df)
    labels = np.full(n, 1, dtype=np.int8)
    atr_norm = df["atr"] / df["atr"].rolling(50).mean()

    for i in range(n - lookahead):
        a = df["atr"].iloc[i]
        if not np.isfinite(a) or a <= 0:
            continue

        vol_factor = atr_norm.iloc[i] if np.isfinite(atr_norm.iloc[i]) else 1.0
        tp = base_atr_mult * vol_factor * a
        sl = base_atr_mult * vol_factor * a

        entry = df["close"].iloc[i]
        upper = entry + tp
        lower = entry - sl

        for j in range(i+1, i+lookahead+1):
            if df["high"].iloc[j] >= upper: labels[i] = 2; break  # BUY win
            if df["low"].iloc[j]  <= lower: labels[i] = 0; break  # SELL win
    return labels
```

## 🅱️ Meta-Labeling (Marcos López de Prado) ⭐

> 🎓 **เทคนิคขั้นสูง** จากหนังสือ *Advances in Financial ML*

### แนวคิด: AI 2 ตัวซ้อนกัน

```
Step 1: Primary Model ทาย "ทิศทาง" (BUY/SELL/HOLD)
            ↓
Step 2: Meta Model ทาย "ควรเทรดไหม?" (yes/no)
            ↓
ถ้าทั้งคู่ผ่าน → เทรด ✅ ถ้าไม่ → skip ❌
```

### ผลลัพธ์
- ✅ Primary model recall สูง (จับ signal ได้ครบ)
- ✅ Meta model precision สูง (กรองสัญญาณห่วยออก)
- ✅ Win rate เพิ่มขึ้น 5-15%

### โค้ดตัวอย่าง

```python
# Step 1: Primary model (ของเดิม)
primary_pred = model_v1.predict(X)
primary_proba = model_v1.predict_proba(X)

# Step 2: Meta model
meta_features = pd.concat([X, primary_proba, market_regime], axis=1)
meta_label = (trade_was_profitable).astype(int)
meta_model = XGBClassifier().fit(meta_features, meta_label)

# Inference
if primary_proba[BUY] > 0.50 AND meta_model.predict_proba > 0.60:
    take_buy_trade()  # ✅
```

---

# 📋 Roadmap 4 สัปดาห์

```
Week 1     Week 2      Week 3        Week 4
  📥         🔍          🛠️            🧪
Collect → Analyze → Implement → Validate
```

## 📅 Week 1: Data Collection (ตอนนี้!)

- ✅ Config: `data_collection_mode: true`
- ✅ Trade ถี่ๆ เก็บ **500-1000 trades**
- ✅ Lot 0.01 (max loss/trade ~$2)
- 🎯 **Output:** `hyper_trades.sqlite` ที่มี trade เพียงพอ

## 🔍 Week 2: Analysis

```python
# ดู confusion matrix แยก regime
import sqlite3, pandas as pd
df = pd.read_sql("SELECT * FROM decisions WHERE status IN ('WIN','LOSS')", conn)

# Win rate by hour
print(df.groupby(df['ts'].dt.hour)['status'].value_counts(normalize=True))

# Win rate by spread tier
df['spread_tier'] = pd.cut(df['spread'], bins=[0, 15, 30, 100])
print(df.groupby('spread_tier')['status'].value_counts(normalize=True))
```

🎯 **Output:** รู้ว่า model **เสียตอนไหน** + **ทำไม**

## 🛠️ Week 3: Implementation

- เพิ่ม 4 functions ใหม่ใน `core/m1_hyper_pipeline.py`
- รวม features ทั้งหมด: `pd.concat([base, of, regime, levels, momentum])`
- Update `FEATURE_COLUMNS` list
- รัน `python run.py train`
- เช็ค feature importance: ตัวไหนได้ใช้จริง?

## 🧪 Week 4: Validation

```python
# Walk-forward backtest 3 เดือน
for month in last_3_months:
    train = data[before month]
    test  = data[month]
    model.fit(train)
    score = model.score(test)
    print(f"Month {month}: accuracy = {score:.2%}")
```

🎯 **Decision:** ถ้า V2 > V1 ≥ **5pp** → roll out, ไม่ใช่ → กลับไป engineer ใหม่

---

# 🎯 Expected Outcome

| Phase | Features | Accuracy | Drawdown |
|---|---:|---:|---:|
| 🟡 V1 (ตอนนี้) | 38 | 43% | -15% |
| 🟢 V1 + Order Flow | 43 | 46% | -12% |
| 🟢 V1 + Order Flow + Regime | 48 | 49% | -10% |
| 🟢 V1 + Order Flow + Regime + Levels | 53 | 52% | -8% |
| 🌟 **V2 ครบ + Meta-labeling** | **58+** | **55-58%** | **-5%** |

> 💎 **Insight:** ทุก +5 features = +3% accuracy (diminishing returns)
> หลัง 60 features → improvement น้อยลง → ต้อง engineering ใหม่ทั้ง pipeline

---

# ⚠️ Risk Management ระหว่างเก็บ Data

## 💸 Math พื้นฐาน

```
balance $500 × 0.10% risk = $0.50/trade
Worst case: เสีย 100 trades ติด = -$50 (10%)
+ commission $7/lot × 100 = $7
รวม worst case: -$57 (~11.4%)

📊 Realistic case (random signal):
WR ~ 33-40%
Net loss/trade ~ $0.30
1000 trades × -$0.30 = -$300 → ❌ blow account 60%!
```

## 🛡️ Safeguards ที่ใส่ไว้แล้ว

| Layer | Setting | ป้องกันอะไร |
|---|---|---|
| 🎯 Risk per trade | `0.10%` | จำกัดความเสียหายต่อไม้ |
| 📏 Max lot % | `2.0%` | จำกัด exposure |
| 🔒 Max lot cap | `0.05` | ฮาร์ดเพดาน |
| ♻️ Recovery steps | `2` | ไม่ให้ขุดหลุมตัวเอง |
| 🛑 Equity stop | `15%` | safety net สุดท้าย |
| ⏸️ Halt cooldown | `5 นาที` | ใจเย็นๆ |

## 💡 คำแนะนำ

```
1. ใช้ Demo account เท่านั้น 🚨
2. Topup ทุกสัปดาห์ (back to $500)
3. ดู review_trades.py ทุกวัน
4. ถ้า DD > 10% → หยุดทันที, debug
```

---

<div align="center">

## 🌟 คำคมส่งท้าย

> **"Engineering ที่ดี = AI ที่ฉลาด**
> **ไม่ใช่ data เยอะอย่างเดียว — แต่เป็น data ที่ มีความหมาย"** 🧬

---

### 🎓 Next Steps

1. ✅ อ่านเอกสารนี้จบ
2. 🛠️ ลอง implement 1 มิติก่อน (แนะนำ Order Flow)
3. 🧪 A/B test กับ V1
4. 📊 ดู feature importance ตัวไหนได้ใช้จริง
5. 🔄 Iterate

---

**🧬 Made with 🧠 + 📊 + ☕ for serious quants**

<sub>Blueprint v2.0 — Order Flow + Regime + Levels + Momentum Stack</sub>

</div>
