# 📚 SweepHunter AI — คู่มือการเรียนรู้ฉบับนักเรียน

> **เอกสารนี้สำหรับ:** นักเรียน/นักศึกษาที่อยากเข้าใจการสร้าง **AI Trading Bot ระดับ Production** ตั้งแต่ทฤษฎีจนถึงการ deploy จริง
>
> **ความรู้ที่ต้องมี:** Python พื้นฐาน, รู้จัก pandas/numpy, เข้าใจ Forex/CFD เบื้องต้น
>
> **เวลาที่ใช้:** 4-8 ชั่วโมง (อ่าน) + 10-20 ชั่วโมง (ลงมือทำ)

---

## 🗺️ Roadmap การเรียนรู้

```
┌─────────────────────────────────────────────────────────────────┐
│  Module 1: Big Picture       (เข้าใจระบบทั้งหมดใน 15 นาที)         │
│  Module 2: Feature Engineering (สร้าง 22 features จากแท่งเทียน)    │
│  Module 3: Machine Learning   (XGBoost classification 3 class)    │
│  Module 4: Trading Logic      (เปิด/ปิด/trail/recovery)           │
│  Module 5: Risk Management    (7 layers + math เบื้องหลัง)         │
│  Module 6: Production Patterns (async, error handling, scaling)   │
│  Module 7: Hands-on Lab       (แบบฝึกหัด 10 ข้อ)                  │
└─────────────────────────────────────────────────────────────────┘
```

---

# 📖 Module 1: Big Picture

## 1.1 ระบบนี้แก้ปัญหาอะไร?

**ปัญหา:** เทรดทอง (XAUUSD) บน M1 (1 นาที) ด้วยมือ → เร็วไม่ทัน, อารมณ์เข้าครอบงำ, ไม่มีวินัย

**Solution:** สร้าง bot ที่:
1. ดึงข้อมูลราคา real-time จาก MetaTrader 5
2. สกัด features จากแท่งเทียน (22 ตัว)
3. ใช้ AI (XGBoost) ทำนายว่าราคาจะ **ขึ้น/ลง/นิ่ง** ใน 30 นาทีข้างหน้า
4. ตัดสินใจเปิดออเดอร์ถ้ามั่นใจ ≥ 50%
5. มีระบบ **กู้ทุนอัจฉริยะ** ถ้าเสียติดกัน
6. ป้องกัน blow account ด้วย hard limits

## 1.2 Tech Stack — ทำไมเลือกแต่ละตัว?

| เทคโนโลยี | ใช้ทำอะไร | ทำไมเลือกตัวนี้ |
|---|---|---|
| **Python 3.10+** | ภาษาหลัก | ML libraries เยอะ, MT5 มี API |
| **MetaTrader 5** | broker platform | Standard ของวงการ Forex |
| **XGBoost** | ML model | เร็ว, แม่น, รับ tabular data ดี |
| **pandas/numpy** | data wrangling | de-facto standard |
| **SQLite** | บันทึก trade | embedded, ไม่ต้องตั้ง server |
| **threading.Thread** | async DB writes | ไม่ block main loop |
| **joblib** | save/load model | จัดการ pickle ได้ดีกว่า built-in |

## 1.3 Data Flow — ภาพรวม 1 minute

```
   MT5 (broker)
        ↓ copy_rates()
   200 แท่ง M1 ล่าสุด
        ↓ build_features()
   22 features (numpy array)
        ↓ model.predict_proba()
   [P(SELL), P(HOLD), P(BUY)]
        ↓ argmax + threshold
   pred=BUY conf=0.58
        ↓ filters: trend, spread, news, cooldown
   ผ่านทุก filter
        ↓ open_market_order() → MT5
   เปิด position
        ↓ (รอ TP/SL)
   ปิดออเดอร์
        ↓ update DB + RecoveryState
   เริ่ม cycle ใหม่
```

## 1.4 อ่าน Code ตามลำดับนี้

> 🎯 **คำแนะนำ:** อย่าอ่าน file-by-file แบบ alphabetical — อ่านตาม **call graph** จะเข้าใจเร็วกว่า

```
1. run.py                    ← entry point (เล็กมาก ~50 บรรทัด)
2. core/config.py            ← โหลด JSON
3. core/paths.py             ← portable paths
4. core/mt5_connector.py     ← เชื่อม broker
5. core/m1_hyper_pipeline.py ← feature engineering ★
6. core/model_trainer.py     ← train XGBoost
7. core/execution.py         ← place order safely
8. core/xauusd_hyper_core.py ← main loop ★★★ (หัวใจ)
9. core/async_db_manager.py  ← log trades
10. core/news_filter.py      ← filter ข่าว
```

⭐ = อ่านสำคัญ, ⭐⭐⭐ = ต้องอ่านละเอียด

---

# 🔬 Module 2: Feature Engineering

## 2.1 ทฤษฎี — ทำไมต้อง Feature Engineering?

ML model **ไม่เข้าใจ** "แท่งเทียน" หรือ "ราคา" — เข้าใจแค่ **ตัวเลข**

หน้าที่เรา: **แปลงข้อมูลดิบ (OHLCV) → ตัวเลขที่มีความหมาย** ที่ model จะเรียนรู้ pattern ได้

### ตัวอย่างเปรียบเทียบ

❌ **Bad feature:** ราคา close = 2050.30
- Model จำไม่ได้ว่า 2050 vs 1800 ต่างกันยังไงในแง่ pattern

✅ **Good feature:** body_atr = (close - open) / ATR
- เป็นค่า **normalized** บอกว่าแท่งนี้ใหญ่/เล็กแค่ไหน relative กับ volatility ปัจจุบัน
- ใช้ได้ที่ราคา 2050 หรือ 1800 ก็ได้

## 2.2 22 Features ในระบบ — แยกเป็น 7 กลุ่ม

### Group 1: Candle Anatomy (รูปร่างแท่ง)

```python
body_atr               = (close - open) / atr
upper_wick_body_ratio  = (high - max(open, close)) / abs(body)
lower_wick_body_ratio  = (min(open, close) - low) / abs(body)
candle_direction       = sign(close - open)  # +1, 0, -1
```

**ทำไม:** แท่ง doji ที่มีไส้ยาว = อาจเป็น reversal signal

### Group 2: Fast ATR Normalization

```python
upper_wick_fast_atr = (high - max(open, close)) / fast_atr_5
body_fast_atr       = abs(close - open) / fast_atr_5
```

**ทำไม:** ใช้ ATR เร็ว (5 bars) ตรวจ volatility สั้นๆ ที่กำลังเปลี่ยน

### Group 3: Momentum

```python
price_velocity_5 = (close - close.shift(5)) / atr   # ราคาเดิน 5 บาร์ = กี่ ATR
ret_1            = (close / close.shift(1)) - 1     # return 1 bar
rsi_7            = 100 - (100 / (1 + rs_7))         # RSI 7 period
ema_dist_atr     = (close - ema_20) / atr           # ห่าง EMA กี่ ATR
```

**ทำไม:** เทรนด์แรง vs sideways → ต้องเทรดต่างกัน

### Group 4: Volume

```python
vol_accel_3   = volume / volume.rolling(3).mean()
vol_spike_10  = volume / volume.rolling(10).max()
```

**ทำไม:** Volume spike = institutional money เข้า → high probability move

### Group 5: Micro Breakout

```python
breakout_up_5  = (high > rolling_high_5.shift(1)).astype(int)
near_high_5    = (high - rolling_low_5) / (rolling_high_5 - rolling_low_5)
```

**ทำไม:** ทะลุ resistance/support 5 bars → momentum continuation

### Group 6: Volatility Regime

```python
atr_ratio = atr_14 / atr_50   # ตลาดเงียบ vs ผันผวน
```

**ทำไม:** strategy ใช้ได้ผลใน high-vol ≠ low-vol

### Group 7: Time Encoding

```python
import math
hour = bar_time.hour
time_sin = math.sin(2 * math.pi * hour / 24)
time_cos = math.cos(2 * math.pi * hour / 24)
session_score = 1.0 if london_open or ny_open else 0.5
```

**ทำไม:** Asia session ≠ London session — pattern ต่างกัน  
**Sin/Cos encoding:** เพราะ hour=23 และ hour=0 ใกล้กันในเวลาจริง แต่ตัวเลขห่างกัน 23

## 2.3 Lab — ลองสร้าง feature ใหม่

```python
# ตัวอย่าง: bullish engulfing ratio
def bullish_engulfing(df):
    prev_bear = df['close'].shift(1) < df['open'].shift(1)
    cur_bull = df['close'] > df['open']
    bigger = df['close'] > df['open'].shift(1)
    return (prev_bear & cur_bull & bigger).astype(int)
```

**คำถาม:** feature นี้ดีไหม? ลอง backtest ดู correlation กับ label

---

# 🤖 Module 3: Machine Learning

## 3.1 ปัญหา = Multi-class Classification (3 class)

```
input:  X = [22 features]   →   output: y ∈ {0=SELL, 1=HOLD, 2=BUY}
```

## 3.2 Label สร้างยังไง? — Triple Barrier Method

```python
def make_label(df, idx, tp_atr=2.0, sl_atr=1.0, lookahead=30):
    entry = df.loc[idx, 'close']
    atr = df.loc[idx, 'atr']
    
    upper = entry + tp_atr * atr   # BUY's TP
    lower = entry - sl_atr * atr   # BUY's SL
    
    for i in range(idx+1, idx+lookahead+1):
        if df.loc[i, 'high'] >= upper: return 2  # BUY win
        if df.loc[i, 'low']  <= lower: return 0  # BUY loss = SELL win
    return 1  # HOLD (ไม่แตะอะไรใน lookahead)
```

> 💡 **Triple Barrier** = idea จาก *Marcos López de Prado* (Advances in Financial ML)
> — labels การเทรดที่ realistic กว่า "ราคาขึ้น/ลง" เฉยๆ

## 3.3 ทำไมเลือก XGBoost?

| Algorithm | Tabular data | Speed | Tuning ง่าย | Production-ready |
|---|---|---|---|---|
| Linear/Logistic | ⚠️ จำกัด | ⚡⚡⚡ | ✅ | ✅ |
| Random Forest | ✅ | ⚡ | ✅ | ✅ |
| **XGBoost** | ✅✅ | ⚡⚡ | ✅✅ | ✅✅ |
| Neural Network | ⚠️ overkill | ⚡ | ❌ | ⚠️ |
| LSTM | ❌ ไม่จำเป็น | ⚡ | ❌ | ❌ |

## 3.4 Training Pipeline

```python
# ขั้นตอนที่ใช้ใน model_trainer.py
1. ดึงข้อมูล MT5: 200,000 bars ของ XAUUSD M1
2. สร้าง features: build_features(rates) → DataFrame
3. สร้าง labels: make_label() → 0/1/2
4. Drop NaN: df.dropna()
5. Balance classes: ใช้ "label_hold_to_event_ratio" จำกัด HOLD ไม่ให้เยอะเกิน
6. Train/Test split: 80/20 (time-based, ไม่ใช่ random!)
7. fit XGBoost: max_depth=6, n_estimators=500, learning_rate=0.05
8. Evaluate: out-of-sample accuracy
9. Save: joblib.dump({"model": model, "features": cols}, "xgb_hyper_model.pkl")
```

> ⚠️ **สำคัญมาก:** Time-series ห้ามใช้ random split! ต้องเรียงตามเวลาเสมอ
> มิฉะนั้น = data leakage (รู้อนาคตตอน train)

## 3.5 Confidence จากไหน?

```python
# XGBoost มี method นี้
proba = model.predict_proba(X)   # shape: (n_samples, 3)
# ตัวอย่าง: proba = [[0.20, 0.30, 0.50]]
#                    SELL  HOLD  BUY

pred = np.argmax(proba)          # = 2 (BUY)
conf = proba[0][pred]            # = 0.50
```

### conf หมายถึงอะไร?

`conf = "ตามข้อมูล historical, รูปแบบนี้ตามด้วย BUY ในอัตรา 50%"`

| conf | ตีความ |
|---|---|
| 0.33 | สุ่ม (3 class baseline) |
| 0.50 | ดีกว่าสุ่ม 50% |
| 0.65 | มั่นใจมาก |
| 0.85+ | สงสัย overfitting หรือ dataset แปลก |

---

# ⚙️ Module 4: Trading Logic

## 4.1 Main Loop Pattern

```python
while True:
    try:
        self._tick()                    # 1 รอบงาน
    except Exception as e:
        log.exception("loop error: %s", e)   # ห้ามล้ม! log แล้วทำต่อ
    time.sleep(0.5)                      # หายใจ
```

> 🎓 **บทเรียน:** Production loop **ห้ามตาย** เพราะ exception ใดๆ
> ถ้า MT5 disconnect, ถ้า model error → log แล้วลองใหม่รอบหน้า

## 4.2 _tick() — หัวใจของระบบ

```
┌──────────────────────────────────┐
│ A. Periodic tasks (every 15s)     │   ← update closed trades, trail SL
├──────────────────────────────────┤
│ B. Heartbeat (every 30s)          │   ← log สถานะ
├──────────────────────────────────┤
│ C. Self-retrain check             │
├──────────────────────────────────┤
│ D. Halt cooldown check?           │
├──────────────────────────────────┤
│ E. New bar closed? → ใช่ ค่อยทำต่อ  │   ← ⭐ idempotency
├──────────────────────────────────┤
│ F. Build 22 features              │
├──────────────────────────────────┤
│ G. Already in trade? → return     │   ← single-position by design
├──────────────────────────────────┤
│ H. News block?                    │
├──────────────────────────────────┤
│ I. AI inference + filters         │
│    (confidence, trend, cooldown)   │
├──────────────────────────────────┤
│ J. Calculate lot + open order     │
└──────────────────────────────────┘
```

## 4.3 Idempotency Pattern (สำคัญ!)

```python
# E. New bar closed?
bar_time = int(rates[closed_idx]["time"])
if self._last_processed_bar == bar_time:
    return                                  # ⭐ ไม่ทำซ้ำ
self._last_processed_bar = bar_time
```

> 🎓 **Idempotency:** เรียกฟังก์ชันซ้ำ 100 ครั้ง = ผลเหมือนเรียกครั้งเดียว
> สำคัญมากใน loop ที่รัน 2 ครั้ง/วินาที — ถ้าไม่ระวัง = เปิด order ซ้ำ!

## 4.4 Order Execution — Defensive Programming

```python
# core/execution.py
def open_market_order(...):
    for attempt in range(retry_count):           # ลอง 5 ครั้ง
        result = mt5.order_send(request)
        if result.retcode == TRADE_RETCODE_DONE:
            return Result(ok=True, ...)
        if result.retcode == TRADE_RETCODE_REQUOTE:
            time.sleep(0.2)                      # รอ requote
            continue
        if result.retcode in PERMANENT_ERRORS:
            return Result(ok=False, ...)         # ยอมแพ้
    return Result(ok=False, comment="exhausted retries")
```

> 🎓 **Lesson:** Trading API → flaky network เป็นเรื่องปกติ — **ต้อง retry**

---

# 🛡️ Module 5: Risk Management

## 5.1 ทำไม Risk Management สำคัญที่สุด?

> "AI แม่นแค่ไหนก็ blow account ได้ ถ้าไม่มี risk management"
> — Tom Basso, Market Wizard

**สูตรพื้นฐาน:** Kelly Criterion (simplified)
```
optimal_fraction = (win_rate × avg_win - loss_rate × avg_loss) / avg_win
```

ถ้า win=55%, RR=2:1 → optimal_fraction ≈ 32% ของ capital  
แต่ใน practice → ใช้ **Fractional Kelly** (เช่น 1/4 ของ optimal = 8%) เพื่อลด drawdown

## 5.2 ระบบ 7 ชั้นในระบบนี้

### Layer 1-3: ก่อนเปิดออเดอร์ (filters)

```python
if conf < 0.50:                           # Layer 1
    return  # AI ไม่มั่นใจพอ
if not trend_aligned(side, ema_dist):     # Layer 2
    return  # ทวนเทรนด์
if time_since_last_entry < 180:           # Layer 3
    return  # cooldown
```

### Layer 4-5: market condition

```python
if spread > max_allowed_spread:           # Layer 4
    return  # broker ขยับ spread
if news_filter.is_blocked():              # Layer 5
    return  # ใกล้ข่าว
```

### Layer 6-7: ป้องกันบัญชี

```python
if consecutive_losses >= max_steps:       # Layer 6
    halt_until = now + 10 * 60   # หยุด 10 นาที
if cum_loss >= balance * 0.07:            # Layer 7
    close_all()                  # EQUITY STOP!
```

## 5.3 Recovery Engine — Math เบื้องหลัง

### ปัญหา: เสีย $1.62 → ไม้ถัดไปต้องการเท่าไรเพื่อ recover?

```
TP เป้าหมาย = 2 × ATR = 2 × 1.6 = 3.2 USD/oz
Profit per lot = 3.2 × 100 = $320 (ขั้นต่อ 1 lot)
Commission = $7
Net profit per lot = $313

ต้องการ recover $1.62 + กำไร $1 = $2.62
→ lot ที่ต้องใช้ = $2.62 / $313 = 0.0084 lot
```

แต่ระบบใช้ **3 floors** เลือกตัวสูงสุด:

```python
floor_geo     = 0.01 × 1.7^1 = 0.017     # ← ตัวนี้ชนะ
floor_recover = 2.62 / 313 = 0.008
floor_volume  = 0.01 × 1.3 = 0.013

next_lot = max(0.017, 0.008, 0.013) = 0.017
```

**ทำไมต้อง 3 floors?**
- `floor_geo` — บังคับให้ lot โต **เป็นเรขาคณิต** → ไม่ underfit
- `floor_recover` — กรณีขาดทุนเยอะ → lot ต้องพอ recover เลย
- `floor_volume` — กำไรไม้นี้ต้องมากกว่ารวม lot ที่เสียไป

## 5.4 Hard Limits Math

### ทำไม `max_steps = 4`?

ถ้า lot_multiplier = 1.7:

| Step | lot | สูญสะสม (ถ้าเสียทั้งหมด) |
|---|---|---|
| 1 | 0.01 (1×) | $1.62 |
| 2 | 0.017 (1.7×) | $4.37 |
| 3 | 0.029 (2.89×) | $9.07 |
| 4 | 0.049 (4.91×) | $16.91 |
| 5 | 0.083 (8.35×) | $30.45 ⚠️ |
| 6 | 0.142 | $52.41 ⚠️⚠️ |

→ step 4-5 ขาดทุน ~7-15% ของ $200 balance = **เพดาน Equity Stop**

## 5.5 Dynamic Account Scaling

```python
# คำนวณ base_lot อัตโนมัติ
risk_usd_per_trade = balance × risk_pct / 100
base_lot = risk_usd_per_trade / (sl_distance × $/price/lot)

# ตัวอย่าง: balance=$10,000, risk=0.3%, sl=$160/lot
# → risk = $30
# → base_lot = $30 / $160 = 0.1875 lot
```

**Insight:** สูตรเดียว → ทำงานได้ทั้ง $500 และ $10M

---

# 🏗️ Module 6: Production Patterns

## 6.1 Async DB Writes

**ปัญหา:** SQLite write = blocking → หยุด trading loop ตอนเขียน DB

**Solution:** worker thread + queue

```python
# core/async_db_manager.py
class AsyncDBManager:
    def __init__(self):
        self.queue = Queue()
        self.thread = Thread(target=self._worker, daemon=True)
        self.thread.start()
    
    def insert_decision(self, **kwargs):
        self.queue.put(("insert", kwargs))   # ไม่ block!
    
    def _worker(self):
        while True:
            op, data = self.queue.get()
            # ทำงานใน background
            self._execute(op, data)
```

> 🎓 **Lesson:** I/O = ทำใน thread แยก, computation = main thread

## 6.2 Configuration Management

```python
# core/config.py
class Config:
    _data = None
    
    @classmethod
    def load(cls):
        if cls._data is None:
            cls._data = json.load(open("config.json"))
        return cls._data
    
    @classmethod
    def section(cls, name):
        return cls.load().get(name, {})
```

> 🎓 **Singleton pattern:** load 1 ครั้ง, ใช้ทั่วระบบ
> แต่ระวัง testing — อาจต้อง reset

## 6.3 Logging Best Practices

```python
# ผิด:
print("trading happened")

# ถูก:
log.info("🆕 เปิดไม้ #%d | %s %.2f lot | AI %.1f%%",
         ticket, side, lot, conf*100)
```

**ทำไม:**
- ✅ มี timestamp อัตโนมัติ
- ✅ Filter level ได้ (DEBUG/INFO/WARNING/ERROR)
- ✅ เขียนลงไฟล์ + console พร้อมกัน
- ✅ Lazy formatting (`%s` vs f-string) — ไม่ format ถ้าไม่ log

## 6.4 Error Handling Tiers

```python
# Tier 1: Critical (ต้องหยุด)
if not MT5Connector.initialise():
    raise SystemExit("MT5 init failed")

# Tier 2: Recoverable (log แล้ว retry รอบหน้า)
try:
    rates = MT5Connector.copy_rates(...)
except Exception as e:
    log.debug("rates fetch failed: %s", e)
    return

# Tier 3: Fatal-but-isolated (catch-all in loop)
try:
    self._tick()
except Exception as e:
    log.exception("loop error")   # log full traceback
```

## 6.5 State Persistence

```python
# Recovery state ต้องรอด restart
def _save_recovery(self):
    RECOVERY_STATE_FILE.write_text(
        json.dumps(self.recovery.to_dict(), indent=2)
    )

def _restore_recovery(self):
    if RECOVERY_STATE_FILE.exists():
        d = json.loads(RECOVERY_STATE_FILE.read_text())
        self.recovery = RecoveryState.from_dict(d)
```

> 🎓 **Lesson:** ทุก mutable state ที่ "ห้ามหาย" ต้อง persist ลง disk
> ถ้า bot crash หรือ restart → resume ได้

## 6.6 Self-Healing — Auto Retrain

```python
def _maybe_retrain(self):
    if time_since_last < 240 * 60:        # ทุก 4 ชม.
        return
    if has_open_position():               # ไม่ retrain ระหว่างถือ
        return
    if new_trades_count < 500:            # ต้องมีข้อมูลใหม่พอ
        return
    train_from_mt5()                       # retrain
    self._load_model()                     # reload
```

> 🎓 **Lesson:** Production ML = **ตลาดเปลี่ยน → model เปลี่ยน**
> ระบบที่ดีต้อง **adapt อัตโนมัติ**

---

# 🧪 Module 7: Hands-on Lab

## 🎯 แบบฝึกหัด 10 ข้อ (เรียงจากง่าย → ยาก)

### Lab 1 ⭐ — เปลี่ยน symbol จาก XAUUSD เป็น BTCUSD
```
แก้: config.json "symbol": "BTCUSD"
รัน: python run.py train
สังเกต: features ทำงานได้ไหม? accuracy เปลี่ยนยังไง?
```

### Lab 2 ⭐ — ปรับ risk_per_trade_pct
```
ลอง: 0.10, 0.50, 1.00
สังเกต: lot ที่บอทเปิดเปลี่ยนยังไง?
คำถาม: ที่ %ไหนพอดีกับ DD ที่คุณรับได้?
```

### Lab 3 ⭐⭐ — เพิ่ม feature ใหม่ "MACD signal"
```python
# ใน m1_hyper_pipeline.py
df["macd_signal"] = ...  # คำนวณ MACD
FEATURE_COLUMNS.append("macd_signal")
```
รัน train ใหม่ → accuracy ดีขึ้นไหม?

### Lab 4 ⭐⭐ — เพิ่ม timeframe เป็น M5
```
แก้ config: "timeframe": "M5"
สังเกต: spread กิน profit % น้อยลงไหม?
```

### Lab 5 ⭐⭐ — เขียน unit test
```python
# tests/test_recovery.py
def test_geometric_floor():
    rs = RecoveryState(consecutive_losses=2)
    # คำนวณว่า floor_geo = 0.01 × 1.7^2 = 0.029
    assert ...
```

### Lab 6 ⭐⭐⭐ — สร้าง dashboard
```
ใช้ Streamlit หรือ Flask
แสดง: balance over time, win rate, today's PnL
ดึงจาก: data/db/hyper_trades.sqlite
```

### Lab 7 ⭐⭐⭐ — Walk-forward backtest
```python
# split data เป็น 12 เดือน
# train 11 เดือน → test เดือนที่ 12 → walk forward
# วัด: monthly Sharpe, max DD, win rate
```

### Lab 8 ⭐⭐⭐⭐ — Hyperparameter tuning
```python
from sklearn.model_selection import GridSearchCV
params = {
    'max_depth': [4, 6, 8],
    'n_estimators': [300, 500, 800],
    'learning_rate': [0.03, 0.05, 0.1]
}
# หา combo ที่ดีที่สุด
```

### Lab 9 ⭐⭐⭐⭐ — Multi-symbol portfolio
```
รัน 3 instances: XAUUSD, EURUSD, GBPUSD
แต่ละตัวมี config + DB แยก
รวม risk: max ใช้ 50% ของ balance / symbol
```

### Lab 10 ⭐⭐⭐⭐⭐ — แทน XGBoost ด้วย LSTM/Transformer
```python
import torch
class TradingLSTM(nn.Module):
    def __init__(self):
        self.lstm = nn.LSTM(input_size=22, hidden_size=64, num_layers=2)
        self.fc = nn.Linear(64, 3)
    ...
```
**คำถาม:** ดีขึ้นไหม? trade-off คืออะไร?

---

# 📚 Module 8: บทเรียนสำคัญ (Key Takeaways)

## 8.1 ML Lessons

✅ **Feature engineering > model selection** — features ดี + model ธรรมดา > features แย่ + model ซับซ้อน

✅ **Time-series → ห้าม random split** — ต้อง chronological

✅ **Class imbalance ใน trading** — HOLD เยอะกว่า BUY/SELL → ต้อง balance

✅ **Regime change** — model ที่ดีปีที่แล้ว อาจแย่ปีนี้ → ต้อง retrain

✅ **Confidence calibration** — `predict_proba` อาจไม่ใช่ probability จริงๆ

## 8.2 Trading Lessons

✅ **Risk Management > Strategy** — บัญชีเสีย 50% ต้องกำไร 100% ถึงจะกลับมา

✅ **Win Rate ≠ Profit** — 60% WR กับ RR 1:0.5 ขาดทุน, 40% WR กับ RR 1:3 กำไร

✅ **Costs matter** — spread + commission กิน scalping ก่อนเริ่ม

✅ **Cooldown after losses** — เสียติด = market regime แปลก, อย่ารีบเข้า

## 8.3 Software Engineering Lessons

✅ **Defensive coding** — สมมติทุก API call จะ fail

✅ **State persistence** — ทุก mutable state สำคัญ ต้อง save

✅ **Idempotency** — design ให้เรียกซ้ำได้

✅ **Async I/O** — ห้ามให้ DB write block trading

✅ **Logging is gold** — debug ได้, audit ได้, sell ได้ (UX!)

✅ **Config-driven** — ทุกค่า magic อยู่ใน JSON ไม่ใช่ใน code

## 8.4 Production Lessons

✅ **Monitor everything** — heartbeat log, DB metrics, model accuracy

✅ **Fail-safe defaults** — ถ้า config หาย → ค่า default ที่ปลอดภัย

✅ **Hard limits everywhere** — `max_lot`, `max_steps`, `equity_stop`

✅ **Dry-run first** — Demo account 2-4 weeks ก่อน live

---

# 🎓 ทรัพยากรเพิ่มเติม

## หนังสือแนะนำ
- 📘 *Advances in Financial Machine Learning* — Marcos López de Prado
- 📘 *Machine Trading* — Ernest Chan
- 📘 *Trading Systems and Methods* — Perry Kaufman
- 📘 *Hands-On Machine Learning* — Aurélien Géron

## คอร์สออนไลน์
- 🎥 [Coursera: ML for Trading (Tucker Balch)](https://www.coursera.org/learn/machine-learning-trading)
- 🎥 [QuantConnect: Tutorial](https://www.quantconnect.com/learning)

## Libraries ที่ควรรู้
- `pandas`, `numpy`, `scikit-learn` — basic
- `xgboost`, `lightgbm`, `catboost` — gradient boosting
- `MetaTrader5`, `ccxt` — broker APIs
- `vectorbt`, `backtesting.py`, `zipline` — backtesting
- `optuna` — hyperparameter tuning

## Communities
- r/algotrading (Reddit)
- QuantConnect Forum
- Discord: Algorithmic Trading

---

# 🏁 สิ่งที่ควรรู้ก่อนใช้จริง

## ✅ Checklist ก่อน Live

- [ ] ทดสอบบน Demo อย่างน้อย 2-4 สัปดาห์
- [ ] เข้าใจทุก config ใน `config.json`
- [ ] รู้วิธี emergency stop (Ctrl+C, kill process, MT5 disable)
- [ ] Backup `data/db/` รายวัน
- [ ] ตั้ง alert (Discord/Telegram) เมื่อ EQUITY STOP
- [ ] อ่าน source code อย่างน้อย 1 รอบ
- [ ] เข้าใจ recovery math (อย่าใช้ถ้ายังคำนวณไม่ถูก)
- [ ] ตั้ง risk_per_trade_pct ตาม risk tolerance ของตัวเอง
- [ ] มี exit plan (รวมถึง psychological — เลิกเมื่อไร?)

## ⚠️ Red Flags ในตลาด → หยุดบอททันที
- 🚨 ข่าว Black Swan (war, central bank emergency)
- 🚨 Spread กว้างผิดปกติ (broker liquidity issue)
- 🚨 บอท trade ผิดทาง 5 ไม้ติด แม้ filter ผ่านหมด
- 🚨 Model self-test fail (`mean_max < 0.40`)

---

# 🎯 Final Words

> **"การสร้าง trading bot ไม่ใช่เกี่ยวกับการทำให้ AI แม่น 100%
> แต่เกี่ยวกับการสร้างระบบที่ **อยู่รอด** ในตลาดได้ในทุกสภาพ"**

ถ้าคุณอ่านเอกสารนี้จบ + ทำ Lab ครบ 10 ข้อ → คุณได้:
- ✅ ความเข้าใจ ML pipeline สำหรับ time-series
- ✅ Production patterns (async, error handling, state)
- ✅ Risk management math
- ✅ Trading domain knowledge

ทักษะเหล่านี้ใช้ได้ทั้งกับ:
- 💼 งาน Quant Developer
- 🎓 Research papers
- 🚀 Startup ของตัวเอง
- 💰 Personal investing

---

<div align="center">

**📚 Happy Learning! 🚀**

*"แค่อ่านไม่พอ — ต้องลงมือทำ ผิดพลาด แล้วเรียนรู้"*

— SweepHunter Team

</div>
