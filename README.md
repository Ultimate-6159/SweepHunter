# 🏆 SweepHunter AI

> **บอทเทรดทอง XAUUSD อัตโนมัติบน MetaTrader 5**
> AI ทำนายทิศทาง (XGBoost) + Recovery Engine กู้ทุนแบบมีสติ + ระบบป้องกันความเสี่ยงหลายชั้น

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/MetaTrader-5-FF6F00?logo=metatrader&logoColor=white" />
  <img src="https://img.shields.io/badge/AI-XGBoost-00A86B" />
  <img src="https://img.shields.io/badge/DB-SQLite-003B57?logo=sqlite&logoColor=white" />
  <img src="https://img.shields.io/badge/Symbol-XAUUSD-FFD700" />
  <img src="https://img.shields.io/badge/Status-Production-success" />
</p>

---

## 🎯 ระบบทำอะไรได้บ้าง? (อธิบายแบบง่ายๆ)

ลองนึกภาพว่าคุณมี "เทรดเดอร์มืออาชีพ" ทำงานให้ตลอด 24 ชั่วโมง โดย:

1. 🧠 **มองตลาด** ทุก 0.5 วินาที — พอแท่งเทียน M5 ปิด ให้ AI ทำนายว่า **ขึ้น / ลง / นิ่ง**
2. 🎯 **เปิดออเดอร์** เฉพาะตอนที่ AI มั่นใจสูง (≥ 55%) และผ่านตัวกรอง **14 ชั้น**
3. 🛡️ **ตั้ง SL/TP** อัตโนมัติจาก ATR (TP = 1.6×ATR, SL = 1.2×ATR)
4. ♻️ **ถ้าเสีย** — คำนวณ lot ใหม่ให้ "ไม้ถัดไปชนะแล้วกู้ทุนคืนได้" (ไม่ใช่ Martingale บ้าๆ)
5. ✋ **ถ้าเสียติดเกิน max_steps** หรือ **ขาดทุนเกิน 8% balance** → ปิด series นั้น เริ่มใหม่
6. 📊 **บันทึกทุกการเทรด** ลง SQLite + retrain โมเดลทุก 1000 ออเดอร์ใหม่ (มี **Acceptance Gate** กันรีเทรนแล้วแย่ลง)

---

## ✨ จุดเด่นที่ต่างจากบอททั่วไป

| ❌ บอททั่วไป | ✅ SweepHunter |
|---|---|
| Martingale ×2 ทุกครั้ง → ระเบิดพอร์ต | **3-Floor Recovery** เลือก lot จาก max(geo, recover-target, volume) |
| ใช้ AI ทาย แต่เปิดทันที | **กรอง 11 ชั้น** (Confidence + Trend + Tick + Spread + News + ...) |
| AI bias ไป BUY อย่างเดียว | **Per-Direction Threshold** — ตั้ง threshold BUY/SELL คนละค่า |
| ต้องแก้ lot เองทุกครั้งพอร์ตโต | **Dynamic Account Scaling** — รองรับ $500 → $10M+ อัตโนมัติ |
| เสีย state ตอน restart | **Recovery State Persist** + DB Auto-Reconstruct |
| Log งงๆ อ่านยาก | **Heartbeat ภาษาไทย + Emoji** เล่าเรื่องเหมือนคนคุย |
| Fake signal / stop hunt → ติดกับ | **Tick-Level Confirmation** — ดู tick 60s ก่อนยิง |

---

## 🧠 AI Engine — XGBoost + 48 Features (V3)

โมเดลเรียนรู้จากแท่ง **M5** ย้อนหลัง **300,000 แท่ง** ครอบคลุม **11 มิติ**:

| มิติ | # | ตัวอย่าง Features |
|---|:-:|---|
| 📊 **Candle Anatomy** | 4 | body_atr, upper/lower wick ratios, direction |
| ⚡ **Fast ATR** | 3 | wicks / fast_ATR (วัด volatility ระยะสั้น) |
| 🚀 **Momentum** | 5 | velocity_2/5, RSI_7, EMA-distance, ret_1 |
| 📈 **Volume Surge** | 2 | vol_accel_3, vol_spike_10 |
| 💥 **Micro Breakout** | 4 | breakout up/down 5 bars, near high/low |
| 🌊 **Volatility Regime** | 1 | ATR ratio (calm vs storm) |
| ⏰ **Time Encoding** | 3 | sin/cos hour, session score |
| 🔭 **Multi-Timeframe** | 6 | M15 + H1 trend, ema_dist, rsi_norm |
| 🎭 **Patterns** | 10 | engulfing, pinbar, inside/outside, sweep, streak |
| 📍 **S/R Levels** (V3) | 5 | dist_pivot_high/low_50, dist_round_50/100, range_position_20 |
| ⚙️ **Momentum Dynamics** (V3) | 5 | velocity_accel, body_accel, wick_imbalance, pattern_bull/bear_5 |

**Output:** `[P(SELL), P(HOLD), P(BUY)]` → เทรดเฉพาะ confidence ≥ threshold

---

## 🛡️ Recovery Engine — หัวใจของระบบ

> ❗ **ไม่ใช่ Martingale แบบมั่วๆ** — เป็น **Smart Recovery** ที่คำนวณ lot จาก 3 สูตร แล้วเลือกตัวที่ใหญ่ที่สุด

### 🧮 สูตรคำนวณ Lot ไม้ Recovery

```python
floor_geo     = base_lot × 1.5^step                       # โตแบบเรขาคณิต (เดิม 1.7 → 1.5 ปลอดภัยกว่า)
floor_recover = (cum_loss + min_profit) ÷ profit_per_lot  # พอกู้ทุน + กำไร
floor_volume  = sum(losing_lots) × 1.3                    # > รวม lot ที่เสียไป (เดิม 1.5 → 1.3)

next_lot = max(floor_geo, floor_recover, floor_volume)
next_lot = min(next_lot, max_lot_cap)                     # capped (default 0.15)
```

### ⏱️ Lifecycle ของ Series

```
🟢 PRIMARY (ไม้ที่ 1, lot=base)
        │
   ┌────┴────┐
  WIN       LOSS
   │         │
   ✅ Reset  ♻️ RECOVERY (ไม้ที่ 2-5, lot โตขึ้น)
              │
         ┌────┴────┐
        WIN       LOSS
         │         │
   ✅ กู้ทุน    🛑 ครบ max_steps (=5) หรือ ขาดทุน > 8% balance → ปิด series
```

### 📊 ตัวอย่างจริง (config ปัจจุบัน: max_steps=5, mult=1.5, balance $500)

| Step | Role | เสียสะสม | lot ที่ใช้ | ถ้า WIN | กำไรสุทธิ |
|:---:|:---:|---:|---:|---:|---:|
| 1 | 🟢 PRIMARY | — | 0.01 | +$2.40 | +$2.40 |
| 2 | ♻️ RECOVERY | $2.40 | 0.022 | +$5.28 | +$2.88 |
| 3 | ♻️ RECOVERY | $7.68 | 0.041 | +$9.84 | +$2.16 |
| 4 | ♻️ RECOVERY | $17.52 | 0.10 | +$24.00 | +$6.48 |
| 5 | ♻️ RECOVERY | $41.52 | 0.15 *(capped)* | +$36.00 | -$5.52 ❌ |
| 6 | 🛑 HALT/Cap | $77.52 | — | (ปิด series → halt 30 นาที) | — |

> 💡 **ระบบเลือก lot ที่ "ใหญ่ที่สุดใน 3 floors"** — มักเป็น `floor_volume` ตอนเสียติด 3-4 ไม้  
> 🛡️ **Per-Series Loss Cap** จะปิด series ก่อนถึงจุดระเบิด (ถ้าขาดทุน > 8% balance)

---

## 🚦 ระบบป้องกันความเสี่ยง 14 ชั้น

ก่อนยิงแต่ละออเดอร์ ต้องผ่านด่านทั้งหมดนี้ — ไม่ผ่านด่านเดียว = ไม่เทรด

| # | กลไก | ป้องกัน |
|:-:|---|---|
| 1️⃣ | **Confidence ≥ threshold** | กรองสัญญาณอ่อน |
| 2️⃣ | **Per-Direction Threshold** | แก้ AI bias (BUY 0.55 / SELL 0.62) |
| 3️⃣ | **Trend Filter** (ema_dist_atr) | ห้ามเทรดทวนเทรนด์ (ปิดได้) |
| 4️⃣ | **Multi-bar Confirmation** | signal ต้องติดกัน N แท่ง |
| 5️⃣ | **FOMO Exhaustion Guard** | ไม่เข้าตอนสุดเทรนด์ (ปิดได้) |
| 6️⃣ | **Tick Confirmation** | ดู tick 60s — กัน fake/stop hunt |
| 7️⃣ | **Inter-trade Cooldown** | ป้องกัน over-trading |
| 8️⃣ | **Dynamic Spread Guard** | ไม่เทรดตอน spread ผิดปกติ |
| 9️⃣ | **News Filter** (Forex Factory) | หยุด ±10 นาทีรอบข่าว High |
| 🔟 | **Max Steps + Halt** | เสียติดเกิน → หยุดพัก/reset |
| 1️⃣1️⃣ | **Global Equity Stop** | ขาดทุนรวม ≥ 18% balance → ปิดทุกไม้ |
| 1️⃣2️⃣ | 🆕 **ATR-Spike Filter** | ATR > 2× ค่าเฉลี่ย 50 แท่ง → ข้าม (กันเทรดตอน volatile abnormal) |
| 1️⃣3️⃣ | 🆕 **Direction-Flip Lock** | เสียทิศเดียวติด 2 ไม้ → block ทิศนั้น 30 นาที (กัน catching rocket) |
| 1️⃣4️⃣ | 🆕 **Per-Series Loss Cap** | series เดียวขาดทุน > 8% balance → ปิด + halt 30 นาที (กันระเบิดก่อน max_steps) |

---

## 💼 Dynamic Account Scaling — รองรับ $500 → $10M+

ระบบคำนวณ lot **อัตโนมัติตามขนาดบัญชี** — ไม่ต้องแก้ config มือเมื่อทุนโต

```
base_lot     = (balance × risk_per_trade_pct%) ÷ (SL_distance × USD/lot)
max_lot_cap  = (balance × max_lot_pct%)        ÷ (SL_distance × USD/lot)
```

### 📊 ตาราง Scaling อัตโนมัติ (risk=0.10%, cap=2%)

| Balance | base_lot | max_lot_cap | Loss สูงสุด/series |
|---:|---:|---:|---:|
| **$500** | 0.01 | 0.05 | ~$10 (2%) |
| **$1,000** | 0.01 | 0.10 | ~$20 (2%) |
| **$5,000** | 0.03 | 0.52 | ~$100 (2%) |
| **$10,000** | 0.06 | 1.04 | ~$200 (2%) |
| **$100,000** | 0.62 | 10.4 | ~$2,000 (2%) |
| **$1,000,000** | 6.25 | 100 ⁂ | ~$20,000 (2%) |

⁂ capped by `max_lot_cap_absolute=100`

### 🎯 Risk Profile (ปรับได้ใน `config.json`)

| Profile | risk_per_trade_pct | max_lot_pct | เหมาะกับ |
|---|---:|---:|---|
| 🛡️ **Ultra-Safe** *(default ตอนนี้)* | 0.10 | 2.0 | Long-term, capital preservation |
| 🟢 **Conservative** | 0.30 | 5.0 | $1k-$1M, mainstream |
| 🟡 **Balanced** | 0.50 | 7.5 | รับ DD ปานกลาง |
| 🔴 **Aggressive** | 1.00 | 12.0 | Small account growth |

---

## 🏗️ สถาปัตยกรรมระบบ

```
SweepHunter/
├── 🚀 run.py                       Entry: train | bot | status
├── ⚙️  config.json                  คอนฟิกทั้งหมด (ที่เดียว)
├── 📚 README.md / LEARNING.md       เอกสาร
├── 📊 generate_report.py            สร้างรายงาน HTML
├── 🔄 retrain.bat / watchdog.bat    Automation scripts
│
├── core/
│   ├── 🧠 xauusd_hyper_core.py     Main loop + Recovery + Filters
│   ├── 📈 m1_hyper_pipeline.py     48 Features V3 (Micro+MTF+Patterns+S/R+MomDyn)
│   ├── 🎓 model_trainer.py         XGBoost training + class weights
│   ├── 🔬 tick_analyzer.py         Tick-level confirmation
│   ├── ⚡ execution.py              IOC orders + Spread + Retry
│   ├── 🔌 mt5_connector.py         MT5 wrapper + auto-reconnect
│   ├── 📰 news_filter.py           Forex Factory XML parser
│   ├── 💾 async_db_manager.py      Async SQLite + state recovery
│   ├── 🛠️  adaptive.py              Adaptive threshold tuning
│   └── config.py / logger.py / paths.py
│
└── data/                           (auto-created)
    ├── models/    🎯 trained .pkl
    ├── db/        💾 hyper_trades.sqlite
    ├── logs/      📝 daily logs
    └── cache/     ⚡ news/spread cache
```

✅ **100% Portable** — paths resolve อัตโนมัติจาก `Path(__file__).resolve().parent.parent`

---

## 📦 การติดตั้ง

### Requirements
- 🪟 Windows 10/11
- 🐍 Python 3.10+
- 📊 MetaTrader 5 (Vantage / Exness / IC Markets แนะนำ)
- 💼 บัญชี Hedge Account

### 3 ขั้นตอน Setup

```powershell
# 1️⃣ ติดตั้ง dependencies
pip install -r requirements.txt

# 2️⃣ แก้ config.json — ใส่ MT5 login
{
  "mt5": {
    "login": YOUR_LOGIN,
    "password": "YOUR_PASS",
    "server": "YOUR_SERVER"
  }
}

# 3️⃣ Train model + รันบอท
python run.py train      # ~5-15 นาที (XGBoost on 300k bars)
python run.py bot        # 🚀 เริ่มเทรด!
```

---

## ⚙️ Config สำคัญ (ค่าปัจจุบัน)

### 🎯 Trading Core
```json
"trading": {
  "symbol": "XAUUSD",
  "timeframe": "M5",
  "magic_number": 990077,
  "base_lot": 0.01,
  "sl_atr_mult": 1.2,
  "tp_atr_mult": 1.6
}
```

### 🛡️ Recovery Engine
```json
"recovery": {
  "enabled": true,
  "max_steps": 5,
  "lot_multiplier": 1.5,
  "min_profit_target_usd": 3.0,
  "profit_volume_multiplier": 1.3,
  "max_lot_cap": 0.15,
  "halt_after_max_steps_minutes": 0,
  "global_equity_stop_pct": 18.0
}
```

> 💡 **อยากให้นับ 3 ไม้แพ้ติดอยู่ series เดียว?** เปลี่ยน `max_steps: 3` (ปลอดภัยกว่า แต่กู้ทุนยากขึ้น)

### 🆕 Risk Filters (3 ชั้นป้องกันใหม่)
```json
"risk_filters": {
  "atr_spike":       { "enabled": true, "rolling_bars": 50, "max_atr_ratio": 2.0 },
  "direction_flip":  { "enabled": true, "min_consec_same_dir_losses": 2, "cooldown_minutes": 30 },
  "series_loss_cap": { "enabled": true, "max_loss_pct_of_balance": 8.0, "halt_minutes": 30 }
}
```

### 🧠 AI & Filters
```json
"hyper_frequency": {
  "min_confidence": 0.55,
  "directional_threshold": {
    "enabled": true,
    "buy": 0.55,
    "sell": 0.62
  },
  "tick_confirmation": {
    "enabled": true,
    "seconds_back": 60,
    "min_directional_ratio": 0.55,
    "max_stop_hunt_score": 0.70
  }
}
```

### 🆕 Acceptance Gate (กันรีเทรนแล้วแย่ลง)
```json
"ai": {
  "retrain_min_new_trades": 1000,
  "retrain_check_interval_min": 1440,
  "trade_augmentation": {
    "enabled": true,
    "min_db_trades": 500,
    "win_weight": 1.5,
    "loss_weight": 1.5,
    "loss_mode": "none"
  },
  "acceptance_gate": {
    "enabled": true,
    "max_test_acc_drop_pct": 2.0,
    "min_oos_test_acc": 0.42,
    "min_dir_balance_ratio": 0.55,
    "max_dir_balance_ratio": 1.8
  }
}
```

> 🚦 หลังรีเทรน → เปรียบเทียบกับ model เก่าบน OOS test set:
> - ถ้า acc ลดลง > 2% → **REJECT** เก็บ model เก่า
> - ถ้า BUY/SELL ratio < 0.55 หรือ > 1.8 → **REJECT** (model เอียงข้างเดียว)

### ⚡ Smart Trailing
```json
"smart_trailing": {
  "enabled": true,
  "disable_during_recovery": false,
  "be_trigger_atr": 0.5,
  "be_buffer_pips": 2.0,
  "trail_mode": "be_only"
}
```

> 💡 **Mode `be_only`** = ขยับ SL ไป break-even เท่านั้น ไม่ trail ต่อ → กัน early stop ตอน price retrace

---

## 📺 Log แบบเล่าเรื่อง (ภาษาไทย + Emoji)

```
🚀 บอทเริ่มทำงาน | XAUUSD tf=M5 magic=990077
💼 เริ่มสด: ไม่มีขาดทุนค้าง

💓 14:32:00 | bid=2050.30 ask=2050.45 spread=15p | ✅ ไม่มีหนี้ | 👀 รอสัญญาณ (อีก 28s)

🎯 14:33 | AI=BUY 64.2% ≥ 62.0% → จะเทรด | atr=1.625 spread=14p
✅ Tick confirm: buy_ratio=0.61 stop_hunt=0.32 micro_trend=+0.5
🆕 เปิดไม้ #12345 (series 25, ไม้ที่ 1) | BUY 0.01 lot @2050.30
   SL=2048.36 (เสี่ยง $1.94) TP=2052.90 (เป้า $2.60) | AI 64.2%

💔 ไม้ #12345 ปิดแล้ว → LOSS -$1.94
   ↳ เสียติด 1 ไม้ ขาดทุนสะสม $1.94

🧮 คำนวณ lot: เสียติด 1 ไม้ ค้าง $1.94 → ใช้ 0.02 lot
   [เลือก แบบเรขาคณิต ×1.7^1 | options: geo=0.017, ต้อง=0.019, รวม×=0.015]
♻️ เปิดไม้ #12346 (series 25, ไม้ที่ 2) | BUY 0.02 lot @2049.10

💚 ไม้ #12346 ปิดแล้ว → WIN +$5.20
🎉 RECOVER สำเร็จ! series #25 จบ กำไรสุทธิ +$3.26 → กลับ lot ปกติ
```

---

## 🧬 State Persistence — กัน restart ทำลายระบบ

| ไฟล์/กลไก | หน้าที่ |
|---|---|
| `data/recovery_state.json` | snapshot real-time ของ recovery state |
| **DB Auto-Reconstruct** | ถ้าไฟล์ json หาย → สร้างใหม่จาก SQLite |
| **Processed tickets cache** | กัน double-count win/loss ของ ticket เดียวกัน |
| **Model timestamp baseline** | retrain counter ไม่ reset ตอน restart |

---

## 📈 Performance ที่คาดหวัง

> ⚠️ **Disclaimer:** ผลลัพธ์ขึ้นอยู่กับ market regime, broker spread, และการตั้งค่า — **ทดสอบบน Demo ก่อนเสมอ**

| Metric | ค่าที่คาดหวัง |
|---|---|
| 📊 Trades / วัน | **5-15 ไม้** (เน้นคุณภาพ) |
| 🎯 Win Rate | **55-65%** |
| ⚖️ Avg RR | **~1.3 : 1** (TP=1.6×ATR / SL=1.2×ATR) |
| 🔢 Concurrent Position | **1** (single position by design) |
| 📉 Max DD ต่อ session | **≤ 18%** (Equity Stop) + **≤ 8% ต่อ series** (Loss Cap) |
| 🔄 Auto-Retrain | ทุก **1000 trades ใหม่** หรือ 1440 นาที (24 ชม.) + Acceptance Gate |

---

## 🛠️ Commands ใช้บ่อย

```powershell
python run.py train            # train โมเดลใหม่
python run.py bot              # รันบอท production
python run.py status           # ดูสถานะปัจจุบัน
python generate_report.py      # สร้าง HTML report
python _inspect_db.py          # debug: ดู series/decisions
.\watchdog.bat                 # auto-restart ถ้า crash
.\retrain.bat                  # retrain แบบ manual
```

---

## 🚨 ข้อควรระวัง

- ❌ **อย่าใช้กับเงินที่เสียไม่ได้** — ทดสอบ Demo ขั้นต่ำ 2 สัปดาห์
- ❌ **อย่าเทรดมือบนบัญชีเดียวกันตอนบอททำงาน** — magic_number กันได้แค่ระดับโค้ด
- ⚠️ **News High Impact** — บอทหยุดให้ ±10 นาที แต่ slippage ยังเกิดได้
- ⚠️ **Broker spread แตกต่าง** — ปรับ `dynamic_spread.hard_max_points` ตาม broker
- ⚠️ **VPS แนะนำ** — latency ต่ำ + uptime 24/5

---

## 📜 License

Commercial — Private use only. ห้ามเผยแพร่ source code โดยไม่ได้รับอนุญาต

---

<p align="center">
  <b>Made with 🧠 + ☕ for serious traders</b><br>
  <sub>v2.5 — Hyper-Frequency + Smart Recovery + Risk Filters + Acceptance Gate</sub>
</p>
