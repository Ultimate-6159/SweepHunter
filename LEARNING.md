# 📚 SweepHunter AI — คู่มือเรียนรู้ฉบับเข้าใจง่าย

> **เขียนเพื่อ:** นักเรียนมัธยมปลาย / มือใหม่ที่อยากเข้าใจ AI Trading Bot ตั้งแต่ 0 → ใช้งานเป็น
> **ใช้เวลา:** อ่าน 2-3 ชม. + ลงมือทำ 5-10 ชม.
> **พื้นฐาน:** Python นิดหน่อย (ตัวแปร, if-else, function) — ไม่ต้องรู้ AI/Forex มาก่อน

<p align="center">
  <img src="https://img.shields.io/badge/Level-Beginner_Friendly-brightgreen" />
  <img src="https://img.shields.io/badge/Style-Story_Mode-blueviolet" />
  <img src="https://img.shields.io/badge/Lab-10_Exercises-orange" />
</p>

---

## 🗺️ แผนที่การเรียน (อ่านตามลำดับ)

```
🌱 บทที่ 1: ทอง XAUUSD คืออะไร? ทำไมเทรดได้?     (10 นาที)
🌿 บทที่ 2: AI ทำงานยังไง? อธิบายแบบเด็กม.ปลาย   (20 นาที)
🌳 บทที่ 3: Feature Engineering — ป้อน "ตา" ให้ AI   (30 นาที)
🌲 บทที่ 4: XGBoost — สมองของบอท                  (25 นาที)
🌴 บทที่ 5: ตัดสินใจเทรด — Pipeline 11 ชั้น         (30 นาที)
🎄 บทที่ 6: Recovery Engine — กู้ทุนแบบมีสติ         (25 นาที)
🏔️ บทที่ 7: Risk Management — กันพอร์ตระเบิด        (20 นาที)
🚀 บทที่ 8: Production Patterns — โค้ดแบบมือโปร     (30 นาที)
🧪 บทที่ 9: ห้องแล็บ — แบบฝึก 10 ข้อ              (5+ ชม.)
🎓 บทที่ 10: บทเรียนชีวิตจากการเทรด                  (10 นาที)
```

---

# 🌱 บทที่ 1: ทอง XAUUSD คืออะไร?

## 🎬 เรื่องเล่า

> ลองนึกภาพ: คุณซื้อทอง 1 บาท ที่ราคา 35,000 บาท วันนี้
> พรุ่งนี้ราคาขึ้นเป็น 36,000 → ขาย ได้กำไร **1,000 บาท** 💰

**XAUUSD** = ทองคำ (XAU) เทียบกับ ดอลลาร์สหรัฐ (USD)
- 1 ออนซ์ ≈ 31.1 กรัม
- ราคาตอนนี้ประมาณ $2,050 ต่อออนซ์

## 💡 ทำไมต้องเทรดทอง? (ไม่ใช่หุ้น)

| 🥇 ทอง | 📈 หุ้น |
|---|---|
| เปิด 24 ชั่วโมง (5 วัน/สัปดาห์) | เปิดแค่ 9:30-16:30 |
| Volatility สูง = โอกาสกำไรเยอะ | ผันผวนน้อยกว่า |
| ใช้ Leverage ได้ (1:100, 1:500) | ใช้ leverage ยาก |
| Spread ต่ำ (~15 cents) | Spread + commission แพง |
| ไม่ต้องวิเคราะห์ "บริษัท" | ต้องอ่านงบการเงิน |

## ⚠️ แต่ระวัง!

```
Leverage 1:100 = เหรียญ 2 ด้าน
✅ กำไร 1% = เพิ่มทุน 100%
❌ ขาดทุน 1% = หายทุน 100% (margin call!)
```

> 🎓 **บทเรียนแรก:** Trading = พายเรือในพายุ — เก่งแค่ไหนก็จมได้ถ้าไม่มีเสื้อชูชีพ (Risk Management)

---

# 🌿 บทที่ 2: AI ทำงานยังไง?

## 🧠 อธิบาย AI ใน 60 วินาที

**AI** ก็เหมือน **เด็กที่ดูตัวอย่างเยอะๆ** แล้วจำ pattern ได้

### ตัวอย่าง: สอน AI ดูว่าหมาหรือแมว

```
Step 1: ป้อนรูปหมา 1,000 รูป + ป้ายว่า "หมา"
Step 2: ป้อนรูปแมว 1,000 รูป + ป้ายว่า "แมว"
Step 3: AI เรียนรู้ pattern (หูตั้ง=แมว, ลิ้นห้อย=หมา)
Step 4: ให้รูปใหม่ → AI ทาย → "หมา 87%"
```

## 🎯 AI เทรดของเรา ทำเหมือนกัน

```
Step 1: ป้อนแท่งเทียน 300,000 แท่ง + ป้ายว่า "ขึ้น/ลง/นิ่ง"
Step 2: AI เรียนรู้ pattern (เช่น "wick ยาวล่าง + RSI ต่ำ = ขึ้น 60%")
Step 3: ทุก 5 นาที → ป้อนแท่งล่าสุด → AI ทาย
Step 4: ถ้ามั่นใจ ≥ 55% → เปิดออเดอร์
```

## 📊 แต่... AI ไม่เห็น "ภาพ" — เห็นแค่ "ตัวเลข"

```
❌ AI ไม่เห็นแท่งเทียนสีเขียว/แดง
✅ AI เห็น: open=2050, close=2052, high=2053, low=2049, volume=1500
```

**หน้าที่เรา:** แปลง OHLC → ตัวเลขที่มีความหมาย ให้ AI เข้าใจ
นี่คือ → **Feature Engineering** (บทถัดไป!)

---

# 🌳 บทที่ 3: Feature Engineering — ป้อน "ตา" ให้ AI

## 🔍 Feature คืออะไร?

**Feature** = ตัวเลขหนึ่งตัวที่บอกอะไรเกี่ยวกับตลาด

### ❌ Feature แย่
```python
close = 2050.30  # ราคาตอนปิด
```
**ปัญหา:** ปีหน้าราคาอาจเป็น 3,000 → AI งง

### ✅ Feature ดี
```python
body_atr = (close - open) / atr   # = 0.5 (ค่าคงที่ relative)
```
**ดีตรงไหน:** ค่าจะอยู่ในช่วง -3 ถึง +3 เสมอ ไม่ว่าราคาเท่าไร

> 💡 **กฎทอง:** Feature ที่ดี = **Normalized** (ทำให้เปรียบเทียบได้)

## 🎨 11 มิติ × 48 Features = ตา 48 ตา ของ AI

> 📌 ในบทนี้จะอธิบาย **9 มิติพื้นฐาน** ก่อน — ส่วน **มิติ S/R Levels** และ **Momentum Dynamics** (เพิ่มใน V3) ดูรายละเอียดที่ [`FEATURE_ENGINEERING_V2.md`](FEATURE_ENGINEERING_V2.md)

### 📊 มิติ 1: รูปร่างแท่งเทียน

```python
body_atr   = (close - open) / atr
upper_wick = (high - close) / |body|
lower_wick = (open - low) / |body|
```

**ตีความ:** body ใหญ่ = แรงเยอะ, ไส้ยาว = มีแรงต้าน

### ⚡ มิติ 2: Momentum

```python
velocity_5 = (close_now - close_5_ago) / atr
rsi_7      = 100 - 100/(1 + avg_gain/avg_loss)
```

**ตีความ:** RSI > 70 = overbought, < 30 = oversold

### 📈 มิติ 3: Volume

```python
vol_spike = current_volume / max_volume_10_bars
```

### 💥 มิติ 4: Breakout

```python
breakout_up = high > max(high_5_bars_ago)
```

### 🌊 มิติ 5: Volatility Regime

```python
atr_ratio = atr_14 / atr_50
# < 0.8 = ตลาดสงบ
# > 1.2 = ตลาดพายุ
```

### ⏰ มิติ 6: เวลา (Sin/Cos Encoding 🌀)

> **ปัญหา:** ชั่วโมง 23 และ 0 ตัวเลขห่างกัน 23 → AI งง

```python
hour_sin = sin(2π × hour / 24)
hour_cos = cos(2π × hour / 24)
# → 23:00 และ 00:00 จะใกล้กันในวงกลม ✅
```

### 🔭 มิติ 7: Multi-Timeframe

```python
htf_trend_dir = sign(close - ema20_h1)  # +1, 0, -1
```

> 💡 "เทรด M5 แต่มองด้วยตา H1" → win rate +5-10%

### 🎭 มิติ 8: Patterns

```python
pinbar_bot = (lower_wick > 2 × body) AND (upper_wick < 0.3 × body)
# pinbar ก้น = สัญญาณกลับตัวขาขึ้น 🔨
```

### 🛡️ มิติ 9: Symmetric (กัน AI bias)

> **ปัญหา:** มี `pinbar_bullish` แต่ไม่มี `pinbar_bearish` → AI bias ไป BUY

**Solution:** ทุก pattern มีคู่ Bull/Bear เสมอ ✅

---

# 🌲 บทที่ 4: XGBoost — สมองของบอท

## 🤔 XGBoost คืออะไร?

**X**treme **G**radient **Boost**ing = "ต้นไม้ตัดสินใจหลายต้นรวมกัน"

### 🌱 อธิบายแบบ Story

```
ต้นที่ 1 ทาย: BUY (ผิด!)
   ↓ ดูว่าผิดเพราะอะไร
ต้นที่ 2 ทาย: SELL (ถูก แต่ confidence ต่ำ)
   ↓ ดูว่ายังขาดอะไร
ต้นที่ 3 ทาย: SELL (มั่นใจมากขึ้น)
   ↓
... (รวม 500 ต้น)
   ↓
รวมเสียง → SELL 65% / HOLD 25% / BUY 10%
```

> 🎓 **Boosting:** ต้นถัดไปเรียนจาก**ความผิดพลาด**ของต้นก่อน → ฉลาดขึ้นเรื่อยๆ

## ⚖️ ทำไมไม่ใช้ Deep Learning?

| Algorithm | Tabular | Speed | Data น้อย | ตีความได้ |
|---|:---:|:---:|:---:|:---:|
| Linear | ⭐⭐ | ⚡⚡⚡ | ✅ | ✅ |
| Random Forest | ⭐⭐⭐ | ⚡⚡ | ✅ | ⚠️ |
| **XGBoost** ⭐ | ⭐⭐⭐⭐⭐ | ⚡⚡ | ✅ | ✅ |
| Neural Net | ⭐⭐ | ⚡ | ❌ | ❌ |
| LSTM | ⭐⭐ | 🐢 | ❌ | ❌ |

**สรุป:** XGBoost = **เร็ว + แม่น + เบา + ตีความได้** ✨

## 🎯 Training Pipeline

### Step 1: สร้าง Label (Triple Barrier)

```python
def make_label(idx):
    entry = close[idx]
    upper = entry + 1.5 × atr   # แตะบน → BUY ชนะ
    lower = entry - 1.5 × atr   # แตะล่าง → SELL ชนะ
    for i in range(idx+1, idx+24):
        if high[i] >= upper: return 2  # BUY 🟢
        if low[i] <= lower: return 0   # SELL 🔴
    return 1  # HOLD 🟡
```

### Step 2: Split (ห้าม Random!)

```python
# ❌ ผิด:
train, test = random_split(data)  # leak อนาคต!

# ✅ ถูก:
train = data[:80%]   # อดีต
test  = data[80%:]   # อนาคต
```

### Step 3: Train

```python
model = XGBClassifier(
    max_depth=6,           # กัน overfitting
    n_estimators=500,
    learning_rate=0.05,    # เรียนช้าๆ
    objective='multi:softprob'
)
model.fit(X_train, y_train, sample_weight=class_weights)
```

## 🔮 Inference

```python
proba = model.predict_proba(X)[0]   # [0.20, 0.30, 0.50]
                                     #  SELL HOLD  BUY
pred = np.argmax(proba)             # = 2 (BUY)
conf = proba[pred]                  # = 0.50

if conf >= 0.55:
    open_buy_order()
```

### conf หมายถึงอะไร?

| conf | แปลว่า |
|---|---|
| 0.33 | สุ่ม (3 class baseline) |
| 0.50 | ดีกว่าสุ่ม 50% |
| 0.65 | มั่นใจมาก |
| 0.85+ | ⚠️ สงสัย overfitting |

---

# 🌴 บทที่ 5: Pipeline 11 ชั้น

## 🚪 เปรียบเทียบ: เข้างานต้องผ่าน 11 ด่าน

```
สัญญาณ AI 🤖
    ↓
🚪 ด่าน 1: AI มั่นใจพอ? (≥ threshold)
🚪 ด่าน 2: BUY/SELL threshold ต่างกัน?
🚪 ด่าน 3: ทวนเทรนด์ใหญ่?
🚪 ด่าน 4: signal ติดกัน N แท่ง?
🚪 ด่าน 5: FOMO? (วิ่งแรงเกินไป)
🚪 ด่าน 6: tick 60s ผ่าน? (กัน fake)
🚪 ด่าน 7: Cooldown หมด?
🚪 ด่าน 8: Spread ปกติ?
🚪 ด่าน 9: ข่าว High Impact?
🚪 ด่าน 10: Max steps?
🚪 ด่าน 11: Equity stop?
    ↓
✅ ทุกด่านผ่าน → เปิดออเดอร์ 🚀
```

## 🎬 ตัวอย่างจริง

```
🎯 14:33 | AI=BUY 64.2% ≥ 62.0%      ✅ ด่าน 1+2
✅ Trend OK, EMA dist=+0.5            ✅ ด่าน 3
✅ Multi-bar OK                       ✅ ด่าน 4
✅ ไม่ FOMO (velocity=0.8)            ✅ ด่าน 5
✅ Tick confirm: buy_ratio=0.61       ✅ ด่าน 6
✅ Cooldown หมด                       ✅ ด่าน 7
✅ Spread=14p ปกติ                    ✅ ด่าน 8
✅ ไม่มีข่าว                          ✅ ด่าน 9
✅ ไม่ใน HALT                         ✅ ด่าน 10+11

🆕 เปิดไม้ #12345 BUY 0.01 lot @2050.30
```

> 💡 **Insight:** บอทดี ≠ ทาย AI แม่น แต่ = **กรองสัญญาณดี**

---

# 🎄 บทที่ 6: Recovery Engine

## 💔 ปัญหา: ถ้าเสียติด ทำยังไง?

### ❌ Martingale มือใหม่

```
ไม้ 1: 0.01 → เสีย $1
ไม้ 2: 0.02 → เสีย $2
ไม้ 3: 0.04 → เสีย $4
...
ไม้ 10: 5.12 → 💀 พอร์ตระเบิด
```

### ✅ Smart Recovery (SweepHunter)

```python
floor_geo     = 0.01 × 1.7^step       # โตช้ากว่า Martingale
floor_recover = (cum_loss + 3) ÷ 250  # พอกู้ทุน
floor_volume  = sum_losing × 1.5      # safety net

next_lot = max(geo, recover, volume)
next_lot = min(next_lot, 0.08)        # CAP! 🛑
```

## 🧮 ตัวอย่าง

```
สถานการณ์: เสียไม้ 1 ที่ 0.01 lot → ขาดทุน $1.92

floor_geo     = 0.01 × 1.7^1 = 0.017
floor_recover = ($1.92 + $3) / $250 = 0.020 ← ใหญ่สุด
floor_volume  = 0.01 × 1.5 = 0.015

next_lot = 0.020

ถ้าชนะ: +$5.00
- ขาดทุนสะสม: $1.92
- กำไรสุทธิ: $5.00 - $1.92 = $3.08 ✅
```

## ⏱️ Lifecycle

```
🟢 Series #25 เปิด
├─ ไม้ 1: BUY 0.01 → 💔 LOSS -$1.92
├─ ไม้ 2: BUY 0.02 → 💚 WIN +$5.00
└─ 🎉 ปิด — กำไรสุทธิ +$3.08 → กลับ lot ปกติ
```

---

# 🏔️ บทที่ 7: Risk Management

## 📊 Math พื้นฐาน

### กฎ 1: Kelly Criterion

```
optimal_fraction = (WR × avg_win - LR × avg_loss) / avg_win

WR=55%, RR=2:1
= (0.55 × 2 - 0.45 × 1) / 2 = 0.325 (32.5%)
```

⚠️ Kelly เต็ม → drawdown สูง
**ใช้ 1/4 Kelly:** ~8% ปลอดภัยกว่า

### กฎ 2: Asymmetric Loss (กฎโหด!)

```
ขาดทุน 10% → ต้องกำไร 11% เพื่อกลับมา
ขาดทุน 50% → ต้องกำไร 100%! 😱
ขาดทุน 90% → ต้องกำไร 900%!! 💀
```

> 🎓 **บทเรียน:** ป้องกันขาดทุนใหญ่ > เพิ่มกำไรเล็ก

## 🛡️ เกราะหลายชั้น

```
ชั้น 1: SL ทันที — ไม่ปล่อยขาดทุนลอย
ชั้น 2: max_steps — จำกัดไม้แพ้ติด
ชั้น 3: max_lot_cap — จำกัด lot สูงสุด
ชั้น 4: equity_stop — ปิดทุกอย่างถ้าเสียถึง %
ชั้น 5: news_filter — หยุดก่อนข่าว
```

## 💰 Dynamic Account Scaling

```python
base_lot = (balance × risk%) / (sl_distance × $/lot)

# Balance $10k, risk 0.3%, SL=$160/lot
# → risk_usd = $30
# → base_lot = $30/$160 = 0.1875 lot
```

✨ **สูตรเดียว → ทำงานกับพอร์ต $500 ถึง $10M ได้!**

---

# 🚀 บทที่ 8: Production Patterns

## 🏗️ Pattern 1: Idempotency

```python
# ❌ บอทขี้แพ้:
while True:
    if signal: open_order()  # อาจเปิด 100 ออเดอร์/วินาที!

# ✅ บอทมือโปร:
while True:
    if last_processed == bar_time: continue
    last_processed = bar_time
    if signal: open_order()
```

## 🏗️ Pattern 2: Defensive Programming

```python
for attempt in range(5):
    try:
        result = mt5.order_send(request)
        if result and result.retcode == DONE:
            return result
        if result.retcode == REQUOTE:
            time.sleep(0.2)
            continue
    except Exception as e:
        log.warning("attempt %d: %s", attempt, e)
return None  # ยอมแพ้แบบ graceful
```

## 🏗️ Pattern 3: Async I/O

```python
class AsyncDB:
    def __init__(self):
        self.queue = Queue()
        Thread(target=self._worker, daemon=True).start()

    def write(self, data):
        self.queue.put(data)  # ไม่ block!

    def _worker(self):
        while True:
            data = self.queue.get()
            self._actual_write(data)
```

## 🏗️ Pattern 4: State Persistence

```python
def _save_recovery(self):
    FILE.write_text(json.dumps(self.state))

def _restore_recovery(self):
    if FILE.exists():
        self.state = json.loads(FILE.read_text())
    else:
        self.state = self.db.reconstruct()  # 🆕 fallback
```

## 🏗️ Pattern 5: Self-Healing

```python
def _maybe_retrain(self):
    if no_position and 500_new_trades and 4hr_passed:
        train_from_mt5()
        self._load_model()  # auto adapt!
```

---

# 🧪 บทที่ 9: ห้องแล็บ

> 💪 **อ่านเฉยๆไม่พอ — ลงมือทำถึงจะเข้าใจ!**

### 🟢 Lab 1: เปลี่ยน Symbol (10 นาที)
```json
"trading": { "symbol": "EURUSD" }
```

### 🟢 Lab 2: ปรับความเสี่ยง (5 นาที)
```json
"account_scaling": { "risk_per_trade_pct": 0.50 }
```

### 🟢 Lab 3: ดู Database (15 นาที)
```python
import sqlite3
conn = sqlite3.connect("data/db/hyper_trades.sqlite")
for row in conn.execute("SELECT * FROM decisions LIMIT 10"):
    print(row)
```

### 🟡 Lab 4: เพิ่ม Feature ใหม่ (30 นาที)
```python
df["my_feature"] = df["close"].pct_change(3)
FEATURE_COLUMNS.append("my_feature")
```

### 🟡 Lab 5: เขียน Unit Test (45 นาที)
```python
def test_lot_calculation():
    bot = HyperBot()
    bot.recovery.consecutive_losses = 2
    bot.recovery.cumulative_loss_usd = 5.0
    lot = bot._compute_lot_for_recovery(spec, atr=1.6)
    assert 0.02 < lot < 0.10
```

### 🟡 Lab 6: สร้าง Dashboard (2 ชม.)
```python
import streamlit as st
import pandas as pd
import sqlite3

conn = sqlite3.connect("data/db/hyper_trades.sqlite")
df = pd.read_sql("SELECT * FROM decisions WHERE status IN ('WIN','LOSS')", conn)

st.title("📊 SweepHunter Dashboard")
st.metric("Total Trades", len(df))
st.metric("Win Rate", f"{(df['status']=='WIN').mean()*100:.1f}%")
st.line_chart(df['pnl'].cumsum())
```

### 🟠 Lab 7: Walk-Forward Backtest (3 ชม.)
แบ่ง 12 เดือน → train 11 → test 1 → เลื่อน

### 🟠 Lab 8: Hyperparameter Tuning (2 ชม.)
```python
from sklearn.model_selection import GridSearchCV
params = {'max_depth':[4,6,8], 'n_estimators':[300,500,800]}
grid = GridSearchCV(XGBClassifier(), params, cv=5)
grid.fit(X, y)
print(grid.best_params_)
```

### 🔴 Lab 9: Multi-Symbol Bot (4 ชม.)
รัน 3 instances: XAUUSD + EURUSD + GBPUSD

### 🔴 Lab 10: แทน XGBoost ด้วย Neural Network (5 ชม.)
```python
import torch.nn as nn

class TradingNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(48, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=1)
```

---

# 🎓 บทที่ 10: บทเรียนชีวิต

## 💡 บทเรียน ML

| # | บทเรียน |
|:-:|---|
| 1 | **Feature ดี > Model ซับซ้อน** |
| 2 | **Time-series ห้าม random split** — leak อนาคต = backtest หลอก |
| 3 | **Regime change** → ต้อง retrain |
| 4 | **Class imbalance** → balance ก่อน train |
| 5 | **Confidence ≠ Probability** — 90% อาจคือ overfit |

## 💰 บทเรียน Trading

| # | บทเรียน |
|:-:|---|
| 1 | **Risk Management > Strategy** |
| 2 | **Win Rate ≠ Profit** — 60% WR + RR 0.5 = ขาดทุน |
| 3 | **Costs matter** — spread กิน scalping |
| 4 | **Cooldown หลังเสีย** — regime แปลก อย่ารีบ |
| 5 | **อยู่รอด > กำไรมาก** |

## 💻 บทเรียน Software

| # | บทเรียน |
|:-:|---|
| 1 | **Defensive coding** — สมมติทุก API จะ fail |
| 2 | **State persistence** — restart ต้อง resume ได้ |
| 3 | **Idempotency** — เรียกซ้ำต้องไม่พัง |
| 4 | **Async I/O** — DB ห้าม block trading |
| 5 | **Logging is gold** |
| 6 | **Config-driven** — magic number อยู่ใน JSON |

## 🌟 บทเรียนชีวิต

> **"Trading Bot สอนคุณมากกว่าวิธีหาเงิน:**
> มันสอน **วินัย** — ตลาดลงโทษคนไม่มีวินัยทันที
> มันสอน **ความถ่อมตน** — คุณคิดว่าฉลาด แต่ตลาดฉลาดกว่า
> มันสอน **ความอดทน** — edge เล็ก × เวลายาว = ความรวย
> มันสอน **การยอมรับ** — แพ้บ่อย ต้อง move on ให้ไว"

---

## 📚 อ่านต่อ

### 📖 หนังสือ
- 📕 *Advances in Financial Machine Learning* — Marcos López de Prado ⭐⭐⭐⭐⭐
- 📕 *Machine Trading* — Ernest Chan
- 📕 *Hands-On Machine Learning* — Aurélien Géron

### 🎥 คอร์ส
- Coursera: *ML for Trading* (Tucker Balch)
- QuantConnect Learning Center (ฟรี)

### 🛠️ Tools
| Level | Library |
|---|---|
| 🟢 พื้นฐาน | `pandas`, `numpy`, `scikit-learn`, `matplotlib` |
| 🟡 กลาง | `xgboost`, `lightgbm`, `MetaTrader5`, `optuna` |
| 🔴 ขั้นสูง | `pytorch`, `vectorbt`, `qlib` |

---

## ✅ Checklist ก่อนใช้งานจริง

- [ ] ทดสอบ Demo อย่างน้อย **2-4 สัปดาห์** 📅
- [ ] เข้าใจทุก config ใน `config.json` ⚙️
- [ ] รู้วิธี Emergency Stop 🛑
- [ ] Backup `data/db/` รายวัน 💾
- [ ] เข้าใจสูตร Recovery Math 🧮
- [ ] ตั้ง risk ตามที่ตัวเองรับได้ 🛡️
- [ ] อ่าน source code อย่างน้อย 1 รอบ 📖
- [ ] มี Exit Plan 🎯

---

## 🚨 Red Flags → หยุดบอททันที!

- 🚨 ข่าว Black Swan (สงคราม, central bank emergency)
- 🚨 Spread กว้างผิดปกติ
- 🚨 บอทเทรดผิดทาง 5 ไม้ติด
- 🚨 Model self-test fail
- 🚨 Equity Stop trigger บ่อยกว่า 1 ครั้ง/สัปดาห์

---

<div align="center">

## 🎯 คำคมส่งท้าย

> **"AI ไม่ใช่หมอดู — มันแค่จำ pattern จากอดีต**
> **Trading bot ที่ดีไม่ใช่บอทที่ทำกำไรมากที่สุด**
> **แต่เป็นบอทที่ อยู่รอด ได้ทุกสภาพตลาด"** 🛡️

---

### 📚 Happy Learning! 🚀

*"แค่อ่านไม่พอ — ต้องลงมือทำ ผิดพลาด แล้วเรียนรู้"*

**— SweepHunter Team —**

<sub>Made with 🧠 + ☕ + ❤️ for curious minds</sub>

</div>
