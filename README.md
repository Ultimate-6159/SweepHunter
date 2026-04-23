# 🏆 SweepHunter AI — XAUUSD Hyper-Frequency Trading Bot

> **AI-Driven Micro-Scalping + Sequential Loss Recovery Engine สำหรับทอง XAUUSD บน MetaTrader 5**
>
> ระบบเทรดอัตโนมัติระดับโปร — XGBoost ทำนายทิศทาง + Recovery Engine กู้ทุนอัจฉริยะ + ป้องกันความเสี่ยงหลายชั้น

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)]()
[![MT5](https://img.shields.io/badge/MetaTrader-5-orange)]()
[![ML](https://img.shields.io/badge/AI-XGBoost-green)]()
[![License](https://img.shields.io/badge/License-Commercial-red)]()

---

## 💡 ทำไมต้อง SweepHunter?

| ปัญหาที่นักเทรดเจอ | SweepHunter แก้ยังไง |
|---|---|
| ❌ มือเทรดช้า ตามตลาด M1 ไม่ทัน | ✅ AI ทำงาน 24/5 ตัดสินใจใน 0.5 วินาที |
| ❌ เสีย streak ติดแล้วล้างพอร์ต | ✅ **3-Floor Recovery Engine** กู้ทุน + กำไรเพิ่ม |
| ❌ Martingale ทั่วไป = lot โตจน blow account | ✅ **Hard Limits 3 ชั้น** (max_steps, max_lot_cap, equity_stop) |
| ❌ AI ทาย "ขึ้น" แต่ครั้งเดียวก็เสียยับ | ✅ **Trend Filter + Confidence Threshold + Cooldown** |
| ❌ Trail SL ปิดเร็ว กำไรไม่คุ้ม | ✅ **Smart Trailing** ปิดอัตโนมัติระหว่าง recovery |
| ❌ ไม่รู้บอททำอะไร log อ่านไม่รู้เรื่อง | ✅ **Heartbeat ภาษาไทย** + Emoji เล่าเรื่อง |
| ❌ พอร์ตโตแล้วต้องแก้ config ใหม่ทุกครั้ง | ✅ **Dynamic Account Scaling** $500 → $10M+ อัตโนมัติ |

---

## 🧠 AI Engine — XGBoost 22 Features

ระบบเรียนรู้จากแท่งเทียน M1 ย้อนหลัง **200,000 แท่ง** (ราว 5 เดือน) ครอบคลุม 7 มิติ:

```
📊 Candle Anatomy         body_atr, wick ratios, candle direction
⚡ Fast ATR Normalisation  upper/lower wicks, body relative to fast ATR
🚀 Momentum               velocity_2/5, ret_1, RSI_7, EMA distance
📈 Volume Surge           vol_accel_3, vol_spike_10
💥 Micro Breakout         breakout up/down 5 bars, near high/low
🌊 Volatility Regime      ATR ratio (calm vs volatile market)
⏰ Time Encoding          time_sin, time_cos, session_score
```

**Output:** `[P(SELL), P(HOLD), P(BUY)]` → เทรดเฉพาะตอนที่ AI มั่นใจ ≥ 50%

---

## 🛡️ Sequential Loss Recovery Engine (เทคโนโลยีหลัก)

ไม่ใช่ Martingale ธรรมดา — เป็น **Smart Recovery** ที่คำนวณ lot ใหม่จาก **3 Floors** แล้วเลือกตัวที่ใหญ่ที่สุด:

### 🧮 สูตรคำนวณ Lot ไม้ Recovery

```python
floor_geo     = base_lot × 1.7^step                       # โตแบบเรขาคณิต
floor_recover = (cum_loss + min_profit) ÷ profit_per_lot  # พอกู้ทุน
floor_volume  = sum(losing_lots) × 1.3                    # > รวม lot ที่เสียไป

next_lot = max(floor_geo, floor_recover, floor_volume)
next_lot = min(next_lot, max_lot_cap)                     # capped
```

### 📊 ตัวอย่างจริง (ATR=1.6, RR 2:1)

| Step | เสีย $ | lot ใหม่ | ถ้า WIN | กำไรสุทธิหลัง recover |
|---:|---:|---:|---:|---:|
| 1 (Primary) | — | 0.01 | +$3.13 | +$3.13 |
| 2 (Recovery) | $1.67 | **0.02** | +$6.26 | **+$4.59** |
| 3 (Recovery) | $5.01 | **0.03** | +$9.39 | +$4.38 |
| 4 (Recovery) | $10.02 | **0.05** | +$15.65 | +$5.63 |
| 5 (HALT) | $18.37 | — | หยุดเทรด 10 นาที | — |

**ทุก step ที่ชนะ → กำไรสุทธิ ≥ $4** เสมอ

---

## 🚦 ระบบป้องกันความเสี่ยง 7 ชั้น

| ชั้น | กลไก | ป้องกันอะไร |
|---|---|---|
| 1️⃣ | **Confidence ≥ 50%** | กรองสัญญาณอ่อน (random baseline = 33%) |
| 2️⃣ | **Trend Filter** (`ema_dist_atr`) | ห้ามเทรดทวนเทรนด์ |
| 3️⃣ | **Inter-trade Cooldown** (180s) | ป้องกัน over-trading จาก noise |
| 4️⃣ | **Dynamic Spread Guard** | ไม่เทรดตอน spread กว้างผิดปกติ |
| 5️⃣ | **News Filter** (Forex Factory) | หยุดก่อน-หลังข่าวสำคัญ ±10 นาที |
| 6️⃣ | **Max Steps + Halt Cooldown** | เสีย 4 ไม้ติด → หยุดเทรด 10 นาที reset |
| 7️⃣ | **Global Equity Stop** | ขาดทุนรวม ≥ 7% ของ balance → ปิดทุกไม้ + HALT |

---

## ⚡ Smart Trailing Stop

- ✅ **Trail เฉพาะตอนไม่มีหนี้** (recovery → ปิด trail ปล่อยถึง TP เต็ม)
- ✅ **Trigger ที่ 1.0×ATR** (ไม่ trail เร็วเกิน)
- ✅ **Lock กำไร + commission offset** (ไม่ถูก wick กิน)

---

## 📺 Log แบบเล่าเรื่อง (ภาษาไทย + Emoji)

```
🚀 บอทเริ่มทำงาน | XAUUSD tf=M1 magic=990077
💼 เริ่มสด: ไม่มีขาดทุนค้าง

💓 14:32:00 | bid=2050.30 ask=2050.45 spread=15p | ✅ ไม่มีหนี้ | 👀 รอสัญญาณ (อีก 28s จะมีแท่งใหม่)

🎯 14:33 | AI=BUY 58.2% ≥ 50.0% → จะเทรด | ✅ ไม่ค้าง | atr=1.625 spread=14p
🆕 เปิดไม้ #12345 (series 25, ไม้ที่ 1) | BUY 0.01 lot @2050.30
   SL=2048.68 (เสี่ยง $1.62) TP=2053.55 (เป้า $3.25) | AI 58.2%

💔 ไม้ #12345 ปิดแล้ว → LOSS -$1.62
   ↳ เสียติด 1 ไม้ ขาดทุนสะสม $1.62 → ไม้ถัดไป lot จะโต

🧮 คำนวณ lot: เสียติด 1 ไม้ ค้าง $1.62 → ใช้ 0.02 lot
   [เลือก แบบเรขาคณิต ×1.7^1 | options: geo=0.017, ต้อง=0.008, รวม×=0.013]
♻️ เปิดไม้ #12346 (series 25, ไม้ที่ 2) | BUY 0.02 lot @2049.10

💚 ไม้ #12346 ปิดแล้ว → WIN +$6.34
🎉 RECOVER สำเร็จ! series #25 จบ กำไรสุทธิ +$4.72 → กลับ lot ปกติ
```

---

## 🏗️ สถาปัตยกรรมระบบ

```
SweepHunter/
├── run.py                        🎯 Entry: train | bot | status
├── config.json                   ⚙️  All-in-one config
├── core/
│   ├── xauusd_hyper_core.py      🧠 Main loop + Recovery Engine
│   ├── m1_hyper_pipeline.py      📊 22 Feature engineering
│   ├── model_trainer.py          🎓 XGBoost training pipeline
│   ├── execution.py              ⚡ IOC + Spread + Retry (5×0.2s)
│   ├── mt5_connector.py          🔌 MT5 auto-detect spec
│   ├── news_filter.py            📰 Forex Factory XML
│   ├── async_db_manager.py       💾 Async SQLite + Webhook
│   ├── config.py / logger.py / paths.py
│   └── __init__.py
└── data/                         📁 auto: models/ db/ logs/ cache/
```

**100% Portable** — path resolve อัตโนมัติจาก `Path(__file__).resolve().parent.parent`

---

## 📦 Installation

### Requirements
- Windows 10/11
- Python 3.10+
- MetaTrader 5 (แนะนำ Vantage / Exness / IC Markets — broker ที่ allow EA)
- บัญชี Hedge Account

### Setup (3 ขั้นตอน)

```powershell
# 1. ติดตั้ง dependencies
pip install -r requirements.txt

# 2. แก้ config.json — ใส่ MT5 login
{
  "mt5": { "login": YOUR_LOGIN, "password": "YOUR_PASS", "server": "YOUR_SERVER" }
}

# 3. Train model + รันบอท
python run.py train      # ~5-15 นาที (XGBoost on 200k bars)
python run.py bot        # เริ่มเทรด!
```

---

## ⚙️ Configuration ที่ปรับได้

### Recovery Engine
```json
"recovery": {
  "enabled": true,
  "max_steps": 4,
  "lot_multiplier": 1.7,
  "min_profit_target_usd": 1.0,
  "profit_volume_multiplier": 1.3,
  "max_lot_cap": 0.5,
  "halt_after_max_steps_minutes": 10,
  "global_equity_stop_pct": 7.0
}
```

### AI Decision Filters
```json
"hyper_frequency": {
  "min_confidence": 0.50,
  "trend_filter_enabled": true,
  "trend_min_ema_dist_atr": 0.10,
  "min_seconds_between_entries": 180,
  "cooldown_seconds_after_series_close": 60
}
```

### Smart Trailing
```json
"smart_trailing": {
  "enabled": true,
  "disable_during_recovery": true,
  "be_trigger_atr": 1.0,
  "trail_distance_atr": 0.6,
  "trail_step_atr": 0.3
}
```

---

## 📈 Performance Expectations

> ⚠️ **Disclaimer:** ผลลัพธ์ขึ้นอยู่กับสภาพตลาด, broker, และการตั้งค่า — ทดสอบบน Demo ก่อนเสมอ

| Metric | ค่าคาดหวัง |
|---|---|
| Trades / วัน | **5-10** ไม้ (เน้นคุณภาพ ไม่ปริมาณ) |
| Win Rate | **55-65%** (ขึ้นกับ market regime) |
| Avg Win / Loss Ratio | **1.8 - 2.2** (TP=2×ATR, SL=1×ATR) |
| Max Concurrent Position | 1 (single position by design) |
| Max Drawdown ต่อ session | ≤ 7% (Equity Stop) |

---

## 💼 Dynamic Account Scaling — รองรับ $500 ถึง $10M+

ระบบ **คำนวณ lot อัตโนมัติ** ตามขนาดบัญชีแบบ real-time ไม่ต้องแก้ config มือเมื่อทุนเปลี่ยน:

```
base_lot     = (balance × risk_per_trade_pct%) ÷ (SL_distance × USD/lot)
max_lot_cap  = (balance × max_lot_pct%)        ÷ (SL_distance × USD/lot)
```

### 📊 ตาราง Scaling อัตโนมัติ (ATR=1.6, SL=1×ATR ≈ $160 loss/lot, risk=0.3%, cap=5%)

| Balance | Risk/trade | base_lot | max_lot_cap | Loss สูงสุด/series | Profit เป้าหมาย/series |
|---:|---:|---:|---:|---:|---:|
| **$500** | $1.50 | 0.01 | 0.16 | ~$25 (5%) | ~$5-15 |
| **$1,000** | $3.00 | 0.02 | 0.31 | ~$50 (5%) | ~$10-30 |
| **$5,000** | $15 | 0.09 | 1.56 | ~$250 (5%) | ~$50-150 |
| **$10,000** | $30 | 0.19 | 3.13 | ~$500 (5%) | ~$100-300 |
| **$50,000** | $150 | 0.94 | 15.6 | ~$2,500 (5%) | ~$500-1,500 |
| **$100,000** | $300 | 1.88 | 31.3 | ~$5,000 (5%) | ~$1,000-3,000 |
| **$500,000** | $1,500 | 9.38 | 100 ⁂ | ~$25,000 (5%) | ~$5,000-15,000 |
| **$1,000,000** | $3,000 | 18.75 | 100 ⁂ | ~$50,000 (5%) | ~$10,000-30,000 |
| **$10,000,000** | $30,000 | 50 † | 100 ⁂ | ~$500,000 (5%) | ~$100,000-300,000 |

⁂ capped by `max_lot_cap_absolute=100.0` (เกิน 100 lot = liquidity issue)
† capped by `max_base_lot=50.0`

### ⚙️ Config `account_scaling`

```json
"account_scaling": {
  "enabled": true,
  "risk_per_trade_pct": 0.30,
  "max_lot_pct_of_balance": 5.0,
  "min_base_lot": 0.01,
  "max_base_lot": 50.0,
  "min_lot_cap": 0.05,
  "max_lot_cap_absolute": 100.0,
  "balance_refresh_seconds": 300
}
```

### 🎯 ปรับ Risk Profile

| Profile | risk_per_trade_pct | max_lot_pct | เหมาะกับ |
|---|---:|---:|---|
| **Ultra-Conservative** | 0.15 | 2.0 | $10M+, hedge fund style |
| **Conservative** *(default)* | 0.30 | 5.0 | $1k-$1M, mainstream |
| **Balanced** | 0.50 | 7.5 | คนรับ DD ได้ปานกลาง |
| **Aggressive** | 1.00 | 12.0 | small account ($500-$5k) |

> 💡 **สูตรง่ายๆ:** ระบบ scale lot ให้ **ความเสี่ยง $ = % ของพอร์ต** ไม่ว่าพอร์ตเล็กหรือใหญ่ → ผลตอบแทน % เท่ากันโดยประมาณ

### ⚠️ ข้อควรระวังพอร์ตใหญ่ ($1M+)

- ✅ ใช้ broker **ECN/Raw spread** (เช่น IC Markets, Pepperstone Razor, Vantage Pro)
- ✅ เปิด **VPS ใกล้ broker server** (ลด latency เหลือ < 5ms)
- ✅ ลด `max_lot_pct_of_balance` ลงเหลือ 2-3% (slippage ที่ size ใหญ่จะเพิ่ม)
- ✅ พิจารณา **split orders** หาก lot > 50 (บาง broker จำกัด max single lot)
- ⚠️ ทดสอบ Demo ขนาดเท่าจริงอย่างน้อย 1 เดือน

---

## 🎓 Self-Retraining

ระบบ **retrain ตัวเองอัตโนมัติ** ทุกครั้งที่:
- มี closed trades ใหม่ ≥ 500 ไม้
- ผ่านไป ≥ 4 ชั่วโมงจาก train ครั้งก่อน
- ไม่มี position เปิด + ไม่มี recovery ค้าง

→ AI ปรับตัวตามสภาพตลาดที่เปลี่ยน โดยไม่ต้อง manual

---

## 💾 ข้อมูลที่บันทึก (SQLite + Webhook)

ทุก trade บันทึกลง `data/db/hyper_trades.sqlite`:
- `decisions` — ทุกการตัดสินใจ (proba, conf, spread, atr, lot, sl, tp, pnl, status)
- `series` — ทุก recovery series (start, close reason, total pnl)

**Optional Webhook** → ส่งสรุปไป Discord/Telegram/Slack ทุก 12 ชั่วโมง

---

## 🛠️ คำสั่งที่ใช้บ่อย

```powershell
python run.py status     # ตรวจ MT5 + symbol spec
python run.py train      # train model ใหม่
python run.py bot        # รันบอท (โหมด live)

# ลบ recovery state ค้าง
Remove-Item data\recovery_state.json

# ดู log แบบ live
Get-Content data\logs\hyper.log -Wait -Tail 30
```

---

## ⚠️ Risk Warning

> 🛑 การเทรด Forex/CFD มีความเสี่ยงสูง อาจสูญเงินทั้งหมด
>
> - ⚠️ **เริ่มจาก Demo** อย่างน้อย 2-4 สัปดาห์
> - ⚠️ **ทุนเริ่มต้นแนะนำ** ≥ $500 (สำหรับ base_lot=0.01, max_lot_cap=0.5)
> - ⚠️ **ใช้บัญชี Hedge** ที่ broker อนุญาต EA
> - ⚠️ ผลตอบแทนในอดีต **ไม่รับประกัน** ผลตอบแทนในอนาคต

---

## 🤝 Support & License

- 📧 **Support:** ติดต่อผู้พัฒนาเพื่อความช่วยเหลือทางเทคนิค
- 📜 **License:** Commercial — ใช้ส่วนตัวเท่านั้น ห้ามแจกจ่ายต่อ
- 🔄 **Updates:** อัพเดท model + features ฟรี ตามรอบที่กำหนด

---

## 🌟 ทำไมต้องเลือก SweepHunter?

> ✅ **AI ที่ Train จากข้อมูลจริง 200k bars** — ไม่ใช่ rule-based ธรรมดา
>
> ✅ **Recovery Engine 3-Floor** — โตอย่างมีหลักการ ไม่บ้าคลั่งแบบ Martingale
>
> ✅ **7 Layer Risk Control** — ป้องกันบัญชีระเบิด
>
> ✅ **Self-Retraining** — ปรับตัวตามตลาด ไม่ต้องดูแลรายวัน
>
> ✅ **Log ภาษาไทย** — เข้าใจทุกการตัดสินใจของบอท
>
> ✅ **100% Portable** — ลาก folder ไปไหนก็รันได้ทันที

---

<div align="center">

**🏆 SweepHunter AI — เทรด XAUUSD แบบมือโปร โดยไม่ต้องนั่งเฝ้า 🏆**

*Built with Python • XGBoost • MetaTrader 5 • SQLite*

</div>
