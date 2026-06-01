# 🏆 SweepHunter AI

> **บอทเทรดทอง XAUUSD อัตโนมัติบน MetaTrader 5**
> AI ทำนายทิศทาง (XGBoost + LGBM ensemble) + Recovery Engine กู้ทุนแบบมีสติ + ระบบป้องกันความเสี่ยงหลายชั้น

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/MetaTrader-5-FF6F00?logo=metatrader&logoColor=white" />
  <img src="https://img.shields.io/badge/AI-XGBoost%2BLGBM-00A86B" />
  <img src="https://img.shields.io/badge/DB-SQLite-003B57?logo=sqlite&logoColor=white" />
  <img src="https://img.shields.io/badge/Symbol-XAUUSD-FFD700" />
  <img src="https://img.shields.io/badge/Features-58-blue" />
</p>

> ⚠️ **เอกสารนี้ sync กับ `config.json` + โค้ดจริง** — ค่าทั้งหมดดึงจากระบบที่รันอยู่ ถ้าแก้ config แล้วค่าในนี้ไม่ตรง ให้ยึด `config.json` เป็นหลัก

---

## 🎯 ระบบทำอะไรได้บ้าง? (อธิบายแบบง่ายๆ)

ลองนึกภาพว่าคุณมี "เทรดเดอร์มืออาชีพ" ทำงานให้ตลอด 24 ชั่วโมง โดย:

1. 🧠 **มองตลาด** ทุก 0.5 วินาที — พอแท่งเทียน M5 ปิด ให้ AI ทำนายว่า **ขึ้น / ลง / นิ่ง**
2. 🎯 **เปิดออเดอร์** เฉพาะตอนที่ AI ผ่าน threshold (จาก `model_meta.json`, ปัจจุบัน **0.58**) และผ่านตัวกรองความเสี่ยงทุกชั้น
3. 🛡️ **ตั้ง SL/TP** อัตโนมัติจาก ATR — **TP = 1.6×ATR, SL = 0.8×ATR** (RR 2:1, breakeven WR ~33%)
4. ♻️ **ถ้าเสีย** — คำนวณ lot ไม้ถัดไปจาก 3 floors ให้ "ชนะแล้วกู้ทุนคืนได้" (ไม่ใช่ Martingale ×2 บ้าๆ)
5. ✋ **ถ้าเสียติดครบ `max_steps` (=4)** → ปิด series ยอมรับหนี้ไป `global_debt` แล้วค่อยๆ กู้คืน
6. 📊 **บันทึกทุกการเทรด** ลง SQLite + retrain โมเดลอัตโนมัติ (มี **Acceptance Gate** กันรีเทรนแล้วแย่ลง)

---

## ✨ จุดเด่นที่ต่างจากบอททั่วไป

| ❌ บอททั่วไป | ✅ SweepHunter |
|---|---|
| Martingale ×2 ทุกครั้ง → ระเบิดพอร์ต | **3-Floor Recovery** เลือก lot จาก max(geo, recover-target, volume) |
| ใช้ AI ทาย แล้วเปิดทันที | **กรองหลายชั้น** (Confidence + Tick + Spread + News + Regime + ATR-Spike + Flip-Lock) |
| โมเดลตัวเดียว | **Ensemble** XGBoost (0.6) + LightGBM (0.4) |
| ต้องแก้ lot เองทุกครั้งพอร์ตโต | **Dynamic Account Scaling** — คำนวณ lot ตาม balance อัตโนมัติ |
| เสีย state ตอน restart | **Recovery State Persist** + DB Auto-Reconstruct |
| Log งงๆ อ่านยาก | **Heartbeat ภาษาไทย + Emoji** เล่าเรื่องเหมือนคนคุย |
| Fake signal / stop hunt → ติดกับ | **Tick-Level Confirmation** — ดู tick 45s ก่อนยิง |
| คูณ lot ตามชั่วโมงเดียวทุกวัน | **Per-Slot Lot Multiplier** — แยกตัวคูณราย วัน×ชั่วโมง (broker TZ) |

---

## 🧠 AI Engine — XGBoost + LGBM Ensemble + 58 Features

โมเดลเรียนรู้จากแท่ง **M5** ย้อนหลัง **~100,000 แท่ง** ครอบคลุม **12 มิติ / 58 features**:

| # | มิติ | จำนวน | Prefix / ตัวอย่าง |
|:-:|---|:-:|---|
| 1 | **Candle Anatomy** | 4 | `body_atr`, `upper/lower_wick_body_ratio`, `candle_direction` |
| 2 | **Fast ATR** | 3 | `*_fast_atr` (volatility ระยะสั้น) |
| 3 | **Momentum** | 5 | `price_velocity_2/5`, `ret_1`, `rsi_7`, `ema_dist_atr` |
| 4 | **Volume** | 2 | `vol_accel_3`, `vol_spike_10` |
| 5 | **Micro Breakout** | 4 | `breakout_up/dn_5`, `near_high/low_5` |
| 6 | **Volatility Regime** | 1 | `atr_ratio` |
| 7 | **Time Encoding** | 3 | `time_sin/cos`, `session_score` |
| 8 | **Multi-Timeframe** | 6 | `htf1_*` (M15), `htf2_*` (H1) |
| 9 | **Patterns** | 10 | `pat_engulf/pinbar/inside/outside/swept/consec_*` |
| 10 | **S/R Levels** | 5 | `sr_dist_pivot/round_*`, `sr_range_position_20` |
| 11 | **Momentum Dynamics** | 5 | `mom_velocity_accel`, `mom_body_accel`, `mom_wick_imbalance`, `mom_pattern_bull/bear_5` |
| 12 | **Smart-Money / Order-Flow** 🆕 | 10 | `liq_sweep_*`, `mom_divergence`, `order_block_*`, `fvg_*`, `cum_delta_norm`, `vol_poc_dist`, `supply_demand_imbal` |

**Output:** `[P(SELL), P(HOLD), P(BUY)]` → เทรดเฉพาะ confidence ≥ `best_threshold`

> 📌 จำนวน feature ตรวจได้จริง: `python -c "from core.m1_hyper_pipeline import FEATURE_COLUMNS; print(len(FEATURE_COLUMNS))"` → **58**
> ดูรายละเอียดมิติทั้งหมด + roadmap ที่ [`FEATURE_ENGINEERING_V2.md`](FEATURE_ENGINEERING_V2.md)

### 📊 สถานะโมเดลปัจจุบัน (`data/models/model_meta.json`)

| Metric | ค่า |
|---|---|
| Rows trained | ~99,900 (train/val/test = 70k/15k/15k) |
| OOS test accuracy | ~32% (3-class, baseline 33%) |
| **best_threshold** | **0.58** (เลือกจาก profit sweep — PF≈2.9, WR≈61% บน OOS) |
| Label | TP=1.6×ATR / SL=0.8×ATR / lookahead=12 แท่ง |
| Ensemble | XGBoost 0.6 + LightGBM 0.4 |

> 💡 accuracy ต่ำ ≠ ขาดทุน — ระบบ RR 2:1 + threshold filter ทำให้ **PF > 1** ได้แม้ accuracy ~32% โดยเลือกเทรดเฉพาะสัญญาณมั่นใจสูง

---

## 🛡️ Recovery Engine — หัวใจของระบบ

> ❗ **ไม่ใช่ Martingale แบบมั่วๆ** — คำนวณ lot จาก 3 สูตร แล้วเลือกตัวที่ใหญ่ที่สุด (capped หลายชั้น)

### 🧮 สูตรคำนวณ Lot ไม้ Recovery

```python
floor_geo     = base_lot × 1.25^step                      # โตแบบเรขาคณิต (ช้ากว่า Martingale)
floor_recover = (cum_loss + min_profit) ÷ net_per_lot      # พอกู้ทุน + กำไรเล็กน้อย
floor_volume  = sum(losing_lots) × 1.05                    # > รวม lot ที่เสียไป

next_lot = max(floor_geo, floor_recover, floor_volume)
# cap หลายชั้น:
#   geo/volume floor → cap ที่ base × max_recovery_lot_multiplier (8×)
#   recover floor    → cap ที่ balance-based absolute_cap (account_scaling)
```

### ⏱️ Lifecycle ของ Series

```
🟢 PRIMARY (ไม้ที่ 1, lot=base)
        │
   ┌────┴────┐
  WIN       LOSS
   │         │
   ✅ Reset  ♻️ RECOVERY (ไม้ที่ 2-4, lot โตขึ้น)
              │
         ┌────┴────┐
        WIN       LOSS
         │         │
   ✅ กู้ทุน    🛑 ครบ max_steps (=4) → ปิด series, หนี้ที่เหลือ → global_debt
```

### 💳 Gentle Slope (กู้หนี้ค้างข้าม series)

ถ้าปิด series แล้วยังมี `global_debt` ค้าง → ไม้ต่อไป (ที่ไม่ใช่ recovery) จะใช้ lot สูงกว่า base **เล็กน้อย** เพื่อทยอยกู้คืนใน ~50 ไม้ (สูงสุด `base × 3`) — ปลอดภัยกว่าการรีบ

> 🛡️ ตอน recovery (มีหนี้ค้างใน series) **ระบบไม่คูณ hourly lot multiplier และ confidence multiplier** — กัน lot ระเบิดซ้ำ

---

## 🚦 ระบบป้องกันความเสี่ยง (ค่าปัจจุบัน)

ก่อนยิงแต่ละออเดอร์ ต้องผ่านด่านที่เปิดใช้งานทั้งหมด — ไม่ผ่านด่านเดียว = ไม่เทรด

| กลไก | สถานะ | ค่าปัจจุบัน |
|---|:-:|---|
| **Confidence ≥ best_threshold** | ✅ | 0.58 (จาก `model_meta.json`) |
| **Tick Confirmation** | ✅ | ดู tick 45s — buy/sell ratio, stop-hunt, velocity burst |
| **Dynamic Spread Guard** | ✅ | hard_max 45p, ≤ 2× ค่าเฉลี่ย 60 นาที |
| **News Filter** (Forex Factory) | ✅ | หยุด ±10 นาทีรอบข่าว USD High |
| **Exhaustion / FOMO Guard** | ✅ | velocity_5 ≤ 1.8×ATR, ไม่เข้าใกล้ extreme > 0.92 |
| **Regime Filter** (HTF trend) | ✅ | block counter-trend + ห้าม recovery ตอน strong trend (>2.5) |
| **ATR-Spike Filter** | ✅ | ATR > 1.4× ค่าเฉลี่ย 50 แท่ง → ข้าม |
| **Direction-Flip Lock** | ✅ | เสียทิศเดิมติด 2 ไม้ → block ทิศนั้น 30 นาที |
| **Session / Slot Block** | ✅ | block ราย วัน×ชั่วโมง (broker TZ) — ปัจจุบัน พุธ 19:00 |
| **Inter-trade Cooldown** | ✅ | ≥ 90s ระหว่างออเดอร์, 30s หลังปิด series |
| **Weekend Close** | ✅ | บังคับปิดก่อนตลาดปิดศุกร์, หยุดเปิดใหม่ 60 นาทีก่อนปิด |
| **Per-Direction Threshold** | ⛔ ปิด | ใช้ best_threshold ตัวเดียวแทน |
| **Trend Filter (ema_dist)** | ⛔ ปิด | ใช้ regime_filter แทน |
| **Global Equity Stop** | ⛔ ปิด | ใช้ slot filter + recovery cap แทน |
| **Series Loss Cap** | ⛔ ปิด | ใช้ max_steps=4 จำกัดแทน |

---

## 💼 Dynamic Account Scaling

ระบบคำนวณ lot **อัตโนมัติตามขนาดบัญชี** (อ่าน balance จาก MT5 ทุก 5 นาที) — ไม่ต้องแก้ config มือเมื่อทุนโต

```
base_lot     = (balance × risk_per_trade_pct%) ÷ (SL_distance × USD/lot)
max_lot_cap  = (balance × max_lot_pct_of_balance%) ÷ (SL_distance × USD/lot)
ทั้งคู่ clamp ระหว่าง min/max ใน config
```

### 🎯 ค่าปัจจุบัน (`account_scaling`)

| Param | ค่า | ความหมาย |
|---|---:|---|
| `risk_per_trade_pct` | **1.5%** | เสี่ยงต่อไม้ถ้าโดน SL |
| `max_lot_pct_of_balance` | **10%** | เพดาน exposure |
| `min_base_lot` / `max_base_lot` | 0.01 / 50 | ขอบเขต base lot |
| `min_lot_cap` / `max_lot_cap_absolute` | 0.05 / **5.0** | ขอบเขต cap |

> ⚠️ **เพดานปัจจุบัน 5 lot** — เหมาะกับพอร์ตเล็ก-กลาง ถ้าจะ scale ถึงระดับ $1M+ ต้องขยับ `max_lot_cap_absolute` + ทดสอบ margin กับ broker (โดยเฉพาะถ้า leverage ลดจาก 1:500)

---

## ⚖️ Per-Slot Lot Multiplier (วัน × ชั่วโมง)

ปรับ lot ตามผลย้อนหลังของแต่ละช่วงเวลา (broker UTC+3) — บูสต์ช่วงดี ลดช่วงแย่

- **`slot_multipliers`** — แยก วัน×ชั่วโมง (ตรง heatmap) มาก่อน
- **`multipliers`** — ชั่วโมงเดียวทุกวัน (fallback)
- **`default`** = 1.0 | ไม่คูณตอน recovery

อัปเดตอัตโนมัติจากข้อมูลจริง: `analyze_lot_multipliers.bat` (มีรายงาน HTML + ถามก่อนแก้ config)

---

## 🏗️ สถาปัตยกรรมระบบ

```
SweepHunter/
├── 🚀 run.py                       Entry: train | train5 | bot | status
├── ⚙️  config.json                  คอนฟิกทั้งหมด (ที่เดียว)
├── 🎮 menu.bat                      Control Panel (เมนูรวมทุกเครื่องมือ)
│
├── core/
│   ├── 🧠 xauusd_hyper_core.py     Main loop + Recovery + Filters + Lot calc
│   ├── 📈 m1_hyper_pipeline.py     58 Features (Candle→MTF→Patterns→S/R→MomDyn→SMC)
│   ├── 🎓 model_trainer.py         XGBoost+LGBM ensemble + Acceptance Gate
│   ├── 🔬 tick_analyzer.py         Tick-level confirmation (45s)
│   ├── ⚡ execution.py              IOC orders + Spread guard + Retry
│   ├── 🔌 mt5_connector.py         MT5 wrapper + symbol spec
│   ├── 📰 news_filter.py           Forex Factory XML parser
│   ├── 🌊 regime_filter.py         HTF trend + counter-trend block
│   ├── 💾 async_db_manager.py      Async SQLite + state recovery
│   ├── 📸 config_snapshot.py       Config versioning (snapshot per change)
│   ├── ⚖️  strategy_weights.py      Performance-weighted training
│   └── config.py / logger.py / paths.py / retrain_log.py
│
└── data/                           (auto-created)
    ├── models/    🎯 xgb_hyper_model.pkl + model_meta.json
    ├── db/        💾 hyper_trades.sqlite
    ├── logs/      📝 hyper.log / mt5.log / db.log
    ├── reports/   📊 strategy_*.html, lot_multiplier_*.html
    └── cache/     ⚡ news cache
```

✅ **Portable** — paths resolve อัตโนมัติจาก `core/paths.py`

---

## 📦 การติดตั้ง

### Requirements
- 🪟 Windows 10/11 + 🐍 Python 3.10+
- 📊 MetaTrader 5 (broker ปัจจุบัน: **Vantage Markets**)
- 💼 บัญชี Hedging

### Setup

```powershell
# 1) ติดตั้ง dependencies (หรือดับเบิลคลิก install.bat / SETUP.bat)
pip install -r requirements.txt

# 2) แก้ config.json — ใส่ MT5 login/password/server
# 3) Train model แล้วรันบอท
python run.py train      # train XGBoost+LGBM (~100k bars)
python run.py bot        # 🚀 เริ่มเทรด!  (หรือ 1_start_bot.bat)
```

---

## ⚙️ Config สำคัญ (ค่าปัจจุบันจริง)

### 🎯 Trading Core
```json
"trading": {
  "symbol": "XAUUSD",
  "timeframe": "M5",
  "magic_number": 990077,
  "base_lot": 0.01,
  "sl_atr_mult": 0.8,
  "tp_atr_mult": 1.6,
  "label_lookahead": 12,
  "history_bars_for_training": 100000
}
```

### 🛡️ Recovery Engine
```json
"recovery": {
  "enabled": true,
  "max_steps": 4,
  "lot_multiplier": 1.25,
  "min_profit_target_usd": 0.5,
  "profit_volume_multiplier": 1.05,
  "max_lot_cap": 5.0,
  "max_recovery_lot_multiplier": 8.0,
  "recovery_spread_trades": 1,
  "debt_recovery_target_trades": 50,
  "debt_recovery_max_lot_mult": 3.0,
  "global_equity_stop_pct": 0
}
```

### 💼 Account Scaling
```json
"account_scaling": {
  "enabled": true,
  "risk_per_trade_pct": 1.5,
  "max_lot_pct_of_balance": 10.0,
  "min_lot_cap": 0.05,
  "max_lot_cap_absolute": 5.0
}
```

### 🧠 AI / Filters
```json
"hyper_frequency": {
  "min_confidence": 0.4,
  "min_seconds_between_entries": 90,
  "tick_confirmation": { "enabled": true, "seconds_back": 45, "min_ticks": 15 },
  "directional_threshold": { "enabled": false }
}
```

### 🆕 AI Engine (Ensemble + Acceptance Gate)
```json
"ai": {
  "retrain_min_new_trades": 30,
  "retrain_check_interval_min": 120,
  "ensemble": { "enabled": true, "xgb_weight": 0.6, "lgbm_weight": 0.4 },
  "trade_augmentation": { "enabled": true, "min_db_trades": 300,
                          "win_weight": 2.5, "loss_weight": 2.0, "loss_mode": "flip" },
  "acceptance_gate": { "enabled": true, "min_pf_floor": 1.02, "min_oos_test_acc": 0.3 }
}
```

> 🚦 **Acceptance Gate** — โมเดลใหม่ต้อง beat all-time best "quality score" (exp × PF × √net) บน OOS
> ถ้าแย่ลง → **REJECT** เก็บ model เก่าทำงานต่อ (zero-regression)

### 🚫 Session Blocking + ⚖️ Lot Multiplier
```json
"session_weighting": {
  "broker_offset_hours": 3,
  "blocked_slots": [ { "dow": 2, "hours": [19] } ]
},
"hourly_lot_multiplier": {
  "enabled": true,
  "disable_during_recovery": true,
  "slot_multipliers": [ ... ],   // แยก วัน×ชั่วโมง
  "multipliers": { ... }          // ชั่วโมงเดียวทุกวัน (fallback)
}
```

---

## 🛠️ เครื่องมือ & Commands

### Control Panel
```powershell
menu.bat        # เมนูรวม: start bot, status, retrain, รายงาน, วิเคราะห์
```

### รันบอท / เทรน
| Command | หน้าที่ |
|---|---|
| `python run.py bot` / `1_start_bot.bat` | รันบอท production |
| `python run.py train` / `3_retrain.bat` | retrain โมเดล (+ Acceptance Gate) |
| `python run.py train5` | multi-seed retrain (เลือก seed ดีสุด) |
| `python run.py status` / `2_status.bat` | health check |
| `watchdog.bat` | auto-restart ถ้า crash |

### รายงาน & วิเคราะห์
| เครื่องมือ | หน้าที่ |
|---|---|
| `view_trades.bat` | แดชบอร์ดเทรด HTML (ภาพรวม, รายไม้, heatmap, recovery) |
| `strategy_report.bat` | รายงานกลยุทธ์ + heatmap (snapshot ≥ 14) |
| `analyze_slots_current.bat` | วิเคราะห์ช่วงเวลาที่ควร block (PF<0.9, ≥8 ไม้) |
| `analyze_lot_multipliers.bat` | วิเคราะห์ + แนะนำ lot multiplier ราย วัน×ชั่วโมง |
| `simulate_recovery.bat` | จำลอง recovery worst-case |
| `db_reliability.bat` | ตรวจความสมบูรณ์ของ DB |

> 💡 รายงานที่อิง "ยุด config ปัจจุบัน" ใช้ `config_snapshot_id >= 14` เพื่อไม่ปนข้อมูลเก่า

---

## 🧬 State Persistence — กัน restart ทำลายระบบ

| ไฟล์/กลไก | หน้าที่ |
|---|---|
| `data/recovery_state.json` | snapshot real-time ของ recovery state |
| **DB Auto-Reconstruct** | ถ้าไฟล์ json หาย → สร้างใหม่จาก SQLite |
| **Processed tickets cache** | กัน double-count win/loss ของ ticket เดียวกัน |
| **bot.lock (PID)** | กันรัน 2 instance พร้อมกัน |

---

## 📈 Performance ที่คาดหวัง

> ⚠️ **Disclaimer:** ผลลัพธ์ขึ้นกับ market regime, broker spread, การตั้งค่า — **ทดสอบบน Demo ก่อนเสมอ**

| Metric | ค่าที่คาดหวัง |
|---|---|
| 📊 Trades / วัน | **20-40 ไม้** (hyper-frequency) |
| 🎯 Win Rate | **35-47%** (RR 2:1 breakeven ~33%) |
| ⚖️ RR | **2:1** (TP=1.6×ATR / SL=0.8×ATR) |
| 🔢 Concurrent Position | **1** (single series at a time) |
| 🛡️ Recovery | สูงสุด **4 ไม้/series** แล้วยอมปิด |
| 🔄 Auto-Retrain | ทุก **30 trades ใหม่** หรือ 120 นาที + Acceptance Gate |

---

## 🚨 ข้อควรระวัง

- ❌ **อย่าใช้กับเงินที่เสียไม่ได้** — ทดสอบ Demo ขั้นต่ำ 2 สัปดาห์
- ❌ **อย่าเทรดมือบนบัญชีเดียวกันตอนบอททำงาน** — magic_number กันได้แค่ระดับโค้ด
- ⚠️ **Leverage ลด (1:500→1:100/1:50)** → กระทบเมื่อ lot ใหญ่ margin ไม่พอ; บอทไม่ได้คำนวณ margin ล่วงหน้า (broker ตอบ retcode)
- ⚠️ **Broker spread แตกต่าง** — ปรับ `dynamic_spread.hard_max_points` ตาม broker
- ⚠️ **VPS แนะนำ** — latency ต่ำ + uptime 24/5

---

## 📜 License

Commercial — Private use only. ห้ามเผยแพร่ source code โดยไม่ได้รับอนุญาต

---

<p align="center">
  <b>Made with 🧠 + ☕ for serious traders</b><br>
  <sub>Hyper-Frequency + 58-Feature Ensemble + Smart Recovery + Per-Slot Lot Multiplier</sub>
</p>
