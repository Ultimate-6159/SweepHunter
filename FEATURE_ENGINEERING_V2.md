# 🧬 Feature Engineering V2 — Blueprint สำหรับเพิ่ม Accuracy

> **เป้าหมาย:** ยกระดับ accuracy จาก 43% → **52-58%**
>
> **วิธี:** เพิ่ม 20 features ใหม่จาก 4 มิติที่ระบบปัจจุบันยัง "มองไม่เห็น"

---

## 🔍 ทำไม Model ปัจจุบันแม่นแค่ 43%?

### ช่องโหว่ที่พบจาก Trade History

| ปัญหา | อาการ | สาเหตุ |
|---|---|---|
| **เข้าตอนเทรนด์จบ** | BUY ที่ top, SELL ที่ bottom | ไม่มี mean-reversion features |
| **ไม่รู้ market regime** | strategy ใช้ใน trending ≠ ranging | ไม่มี regime classifier |
| **ไม่เห็น orderflow** | ไม่รู้ว่า big money ทำอะไร | มีแค่ tick volume ดิบ |
| **ไม่รู้ Support/Resistance** | ชน level แล้วเด้ง = SL hit | ไม่มี pivot levels |
| **เมิน psychological levels** | ราคาเด้งที่ 4700, 4750, 4800 | ไม่มี round number distance |
| **ไม่ track momentum decay** | momentum กำลังจะหมด | มีแค่ velocity ตอนนี้ |

---

## 🎯 4 มิติ Features ใหม่

### 📊 Dimension 1: Order Flow Proxy (5 features)

**แนวคิด:** ใช้ tick_volume + price action ประมาณการณ์ buy/sell pressure

| Feature | สูตร | ตีความ |
|---|---|---|
| `buy_pressure_3` | `Σ(volume × (close > open ? 1 : -1)) ใน 3 bars / Σ(volume)` | +1=ทุก bar ขาขึ้น volume สูง |
| `sell_pressure_3` | `1 - buy_pressure_3` | สมมาตร |
| `vol_at_high_ratio` | `volume_when_close_near_high / total_volume(10)` | สะสมที่ high = แรงซื้อ |
| `vol_at_low_ratio` | `volume_when_close_near_low / total_volume(10)` | สะสมที่ low = แรงขาย |
| `delta_proxy` | `(close - (high+low)/2) / (high - low) × volume` | bullish/bearish bias × intensity |

**Code skeleton:**
```python
def order_flow_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    direction = np.sign(df["close"] - df["open"])
    vol = df["tick_volume"].astype(float)
    
    out["buy_pressure_3"] = (vol * (direction > 0)).rolling(3).sum() / vol.rolling(3).sum().replace(0, 1)
    out["sell_pressure_3"] = 1.0 - out["buy_pressure_3"]
    
    near_high = (df["close"] >= df["high"] - 0.2 * (df["high"] - df["low"])).astype(float)
    near_low = (df["close"] <= df["low"] + 0.2 * (df["high"] - df["low"])).astype(float)
    
    out["vol_at_high_ratio"] = (vol * near_high).rolling(10).sum() / vol.rolling(10).sum().replace(0, 1)
    out["vol_at_low_ratio"] = (vol * near_low).rolling(10).sum() / vol.rolling(10).sum().replace(0, 1)
    
    hl_range = (df["high"] - df["low"]).replace(0, np.nan)
    bias = (df["close"] - (df["high"] + df["low"]) / 2) / hl_range
    out["delta_proxy"] = (bias * vol / vol.rolling(20).mean().replace(0, 1)).fillna(0)
    
    return out
```

---

### 📊 Dimension 2: Market Regime Classifier (5 features)

**แนวคิด:** ตลาด trending vs ranging vs choppy → strategy ใช้ได้ผลต่างกัน

| Feature | สูตร | ตีความ |
|---|---|---|
| `regime_trend_strength` | `\|EMA20 - EMA50\| / ATR` | สูง = trending ชัด |
| `regime_chop_index` | `Σ\|close-prev_close\| / (max-min) × 100 ใน 14 bars` | สูง=ranging, ต่ำ=trending |
| `regime_efficiency` | `\|close - close.shift(10)\| / Σ\|return\| ใน 10 bars` | 1.0=trend perfect, 0=noise |
| `regime_range_width_atr` | `(max(high,20) - min(low,20)) / ATR` | width ของ range |
| `regime_breakout_potential` | `1 if close ใกล้ขอบ range > 0.85, else 0` | จะ breakout ไหม |

**Code skeleton:**
```python
def regime_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    c = df["close"]
    atr = df["atr"]
    
    ema20 = c.ewm(span=20, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()
    out["regime_trend_strength"] = (ema20 - ema50).abs() / atr.replace(0, np.nan)
    
    # Choppiness Index (Bill Dreiss)
    sum_tr = atr.rolling(14).sum()
    high_max = df["high"].rolling(14).max()
    low_min = df["low"].rolling(14).min()
    out["regime_chop_index"] = 100 * np.log10(sum_tr / (high_max - low_min).replace(0, np.nan)) / np.log10(14)
    
    # Kaufman Efficiency Ratio
    direction = (c - c.shift(10)).abs()
    volatility = c.diff().abs().rolling(10).sum()
    out["regime_efficiency"] = direction / volatility.replace(0, np.nan)
    
    out["regime_range_width_atr"] = (df["high"].rolling(20).max() - df["low"].rolling(20).min()) / atr.replace(0, np.nan)
    
    range_top = df["high"].rolling(20).max()
    range_bot = df["low"].rolling(20).min()
    range_pos = (c - range_bot) / (range_top - range_bot).replace(0, np.nan)
    out["regime_breakout_potential"] = ((range_pos > 0.85) | (range_pos < 0.15)).astype(float)
    
    return out
```

---

### 📊 Dimension 3: Support/Resistance & Levels (5 features)

**แนวคิด:** ราคามักจะเด้งหรือทะลุ "level" ที่นักเทรดจับตามอง

| Feature | สูตร | ตีความ |
|---|---|---|
| `dist_to_pivot_high` | `(pivot_high_50 - close) / ATR` | ห่าง resistance กี่ ATR |
| `dist_to_pivot_low` | `(close - pivot_low_50) / ATR` | ห่าง support กี่ ATR |
| `dist_to_round_50` | `min(\|close - round_50\|) / ATR` | ห่างเลขกลม (4700/4750) กี่ ATR |
| `dist_to_round_100` | `min(\|close - round_100\|) / ATR` | ห่าง 4700/4800 |
| `pivot_test_count` | จำนวนครั้งที่ราคา test pivot ใน 50 bars | level ยิ่งโดน test = ยิ่งสำคัญ |

**Code skeleton:**
```python
def level_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    c = df["close"]
    atr = df["atr"].replace(0, np.nan)
    
    # Pivot levels (rolling max/min)
    pivot_high = df["high"].rolling(50).max()
    pivot_low = df["low"].rolling(50).min()
    out["dist_to_pivot_high"] = (pivot_high - c) / atr
    out["dist_to_pivot_low"] = (c - pivot_low) / atr
    
    # Round number distance
    round_50 = (c / 50).round() * 50
    round_100 = (c / 100).round() * 100
    out["dist_to_round_50"] = (c - round_50).abs() / atr
    out["dist_to_round_100"] = (c - round_100).abs() / atr
    
    # Pivot test count: how many bars touched the recent pivot ±0.3 ATR
    proximity = 0.3 * atr
    near_high = ((df["high"] >= pivot_high - proximity) & 
                 (df["high"] <= pivot_high + proximity)).astype(int)
    out["pivot_test_count"] = near_high.rolling(50).sum().fillna(0)
    
    return out
```

---

### 📊 Dimension 4: Momentum Dynamics (5 features)

**แนวคิด:** momentum กำลังเร่งขึ้น (continuation) หรือ ชะลอ (reversal)?

| Feature | สูตร | ตีความ |
|---|---|---|
| `velocity_acceleration` | `velocity_5 - velocity_5.shift(1)` | บวก=เร่ง, ลบ=ชะลอ |
| `rsi_divergence` | `1 if (price↑ & rsi↓) or (price↓ & rsi↑)` | classic reversal signal |
| `consec_same_dir_bars` | จำนวน bar ที่ทิศเดียวกันต่อเนื่อง | สูงเกิน 5 = exhausted |
| `body_acceleration` | `body_atr - body_atr.shift(1)` | candle ใหญ่ขึ้น = momentum |
| `wick_imbalance` | `(upper_wick - lower_wick) / (upper + lower + ε)` | บวก=มี seller บน, ลบ=มี buyer ล่าง |

**Code skeleton:**
```python
def momentum_dynamics(df: pd.DataFrame, rsi_period: int = 7) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    c = df["close"]
    atr_safe = df["atr"].replace(0, np.nan)
    
    # Velocity acceleration
    velocity_5 = (c - c.shift(5)) / atr_safe
    out["velocity_acceleration"] = velocity_5 - velocity_5.shift(1)
    
    # RSI divergence (price up but RSI down → bearish)
    rsi = _rsi(c, rsi_period)
    price_up = (c > c.shift(5)).astype(int)
    rsi_up = (rsi > rsi.shift(5)).astype(int)
    out["rsi_divergence"] = (price_up != rsi_up).astype(float)
    
    # Consecutive same-direction bars
    direction = np.sign(c - df["open"])
    same = (direction == direction.shift(1)).astype(int)
    # cumulative streak (resets when direction changes)
    streak = same.groupby((same != same.shift(1)).cumsum()).cumsum()
    out["consec_same_dir_bars"] = streak.fillna(0).clip(0, 10)
    
    # Body acceleration
    body = (c - df["open"]).abs() / atr_safe
    out["body_acceleration"] = body - body.shift(1)
    
    # Wick imbalance
    upper_wick = df["high"] - np.maximum(df["open"], c)
    lower_wick = np.minimum(df["open"], c) - df["low"]
    total = upper_wick + lower_wick + 1e-6
    out["wick_imbalance"] = (upper_wick - lower_wick) / total
    
    return out
```

---

## 🎓 Bonus: Better Labels (เปลี่ยนวิธี label)

### Triple-Barrier with Volatility Targeting

```python
def adaptive_label(df, lookahead=24, base_atr_mult=1.5):
    """
    TP/SL ปรับตาม volatility regime:
    - Low vol → tight TP (0.8 ATR)
    - High vol → wider TP (2.0 ATR)
    """
    n = len(df)
    labels = np.full(n, 1, dtype=np.int8)
    atr_norm = df["atr"] / df["atr"].rolling(50).mean()  # 1.0 = normal, 2.0 = double vol
    
    for i in range(n - lookahead):
        a = df["atr"].iloc[i]
        if not np.isfinite(a) or a <= 0: continue
        
        # Adaptive TP/SL
        vol_factor = atr_norm.iloc[i] if np.isfinite(atr_norm.iloc[i]) else 1.0
        tp = base_atr_mult * vol_factor * a
        sl = base_atr_mult * vol_factor * a
        
        entry = df["close"].iloc[i]
        upper = entry + tp
        lower = entry - sl
        
        for j in range(i+1, i+lookahead+1):
            if df["high"].iloc[j] >= upper: labels[i] = 2; break
            if df["low"].iloc[j] <= lower:  labels[i] = 0; break
    return labels
```

### Meta-Labeling (Marcos López de Prado)

```python
# Step 1: Primary model predicts direction
# Step 2: Meta-model predicts "should I take this trade?"
# Result: filter low-quality signals from primary model

primary_pred = model_v1.predict(X)
meta_features = X + [primary_pred_proba, market_regime, ...]
meta_label = (trade_was_profitable).astype(int)
meta_model.fit(meta_features, meta_label)

# At inference:
if primary.predict_proba > 0.5 AND meta.predict_proba > 0.6:
    take_trade()
```

---

## 📋 Implementation Roadmap

### Week 1: Data Collection (ตอนนี้)
- ✅ Config: data_collection_mode = true
- ✅ Trade ถี่ๆ เก็บ 500-1000 trades
- ✅ Lot 0.01 (max loss/trade ~$2)

### Week 2: Analysis
- ดู `data/db/hyper_trades.sqlite`
- หา pattern: เสียตอนไหน, accuracy เปลี่ยนยังไงตาม session/spread/atr
- Plot confusion matrix แยกตาม regime

### Week 3: Feature V2 Implementation
- Implement 20 features ตาม blueprint นี้
- Add to `m1_hyper_pipeline.py`
- Retrain → เช็ค accuracy ขึ้นไหม

### Week 4: Validation
- Walk-forward backtest 3 เดือน
- A/B test: V1 vs V2 บน demo
- ถ้า V2 > V1 ≥ 5pp → roll out

---

## 🎯 Expected Outcome

| Phase | Features | Accuracy เป้า | Drawdown |
|---|---:|---:|---:|
| V1 (ตอนนี้) | 28 | 43% | -15% |
| V1 + Order Flow | 33 | 46% | -12% |
| V1 + Order Flow + Regime | 38 | 49% | -10% |
| V1 + Order Flow + Regime + Levels | 43 | 52% | -8% |
| **V2 ครบ + Meta-labeling** | **48+** | **55-58%** | **-5%** |

---

## ⚠️ Risk ระหว่าง Data Collection

```
balance $500 × 0.10% risk = $0.50/trade
ถ้าเสีย 100 trades ติด (worst case) = -$50 (10%)
+ commission $7/lot × 100 = $7
รวม worst case: -$57 (~11.4%)

📊 Realistic case:
WR ที่ random/poor signal = 33-40%
Net loss/trade ~ $0.30
1000 trades × -$0.30 = -$300 → ❌ blow account!
```

### 🛡️ Safeguards ที่ใส่ไว้แล้ว

- ✅ `risk_per_trade_pct` = 0.10% (เล็กสุด)
- ✅ `max_lot_pct_of_balance` = 2.0%
- ✅ `max_lot_cap` = 0.05 (สูงสุด 5 cents/trade)
- ✅ `max_steps` = 2 (recovery สั้น)
- ✅ `global_equity_stop_pct` = 15.0
- ✅ `halt_after_max_steps_minutes` = 5

**คำแนะนำ:** หยุดทุกสัปดาห์ → topup demo ใหม่ → continue

---

<div align="center">

**🧬 Engineering ที่ดี = AI ที่ฉลาด — ไม่ใช่ data เยอะอย่างเดียว**

</div>
