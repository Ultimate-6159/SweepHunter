"""
xauusd_hyper_core.py
=====================
Main loop: Hyper-Frequency Micro-Scalping + **Sequential Loss Recovery Engine**

?????????? (?????????? in-position averaging ????):
  1. ???????? SL+TP ??????? MT5 (cut loss ???????????????? — ??????)
  2. ??????? ? ??????????????? ? ????????????? lot ??? recover ???
  3. ?????????????? **??? AI ??????????** (????????????????)
  4. WIN ? reset, lot ???? base; LOSS ? step++, lot ????????
  5. Hard limits:
       - max_steps (default 4)
       - max_lot_cap (default 0.5 lot)
       - global_equity_stop_pct
  6. TP=2×ATR / SL=1×ATR ? RR=2:1 (????????)

Pipeline ??? tick:
  A. Periodic: ???? trade ???????? ? update RecoveryState + DB
  B. Smart Trailing ?? position ??????????
  C. ????? position ???? ? return
  D. ???????? ? AI predict ? ????? lot ??? recovery ? ???????????
"""
from __future__ import annotations
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Set

import joblib
import numpy as np
import MetaTrader5 as mt5

from .config import Config
from .logger import get_logger
from .mt5_connector import MT5Connector
from .news_filter import NewsFilter
from .regime_filter import RegimeFilter
from .async_db_manager import AsyncDBManager, WebhookBroadcaster
from .m1_hyper_pipeline import FEATURE_COLUMNS, build_features
from .execution import (
    open_market_order, modify_position_sl, close_position,
    spread_guard, DynamicSpreadTracker, commission_price_offset,
)
from .paths import model_path, DATA_DIR

log = get_logger("hyper")
RECOVERY_STATE_FILE = DATA_DIR / "recovery_state.json"
CLASS_NAMES = {0: "SELL", 1: "HOLD", 2: "BUY"}


# ============================================================ Recovery State
@dataclass
class RecoveryState:
    cumulative_loss_usd: float = 0.0
    cumulative_losing_volume: float = 0.0   # sum ของ lot ทุกไม้ที่เสีย (สำหรับ volume floor)
    consecutive_losses: int = 0
    series_id: Optional[int] = None
    last_side: Optional[str] = None
    processed_tickets: Set[int] = field(default_factory=set)
    halted_until_ts: float = 0.0
    # 🆕 Direction-flip lock: ติดตามว่าเสียทิศไหนติดต่อกันกี่ไม้
    last_loss_side: Optional[str] = None
    consec_same_dir_losses: int = 0
    # 🆕 Direction-block: ห้ามเทรดทิศนี้จนกว่าจะถึง ts (cooldown)
    blocked_side: Optional[str] = None
    block_side_until_ts: float = 0.0
    # 🆕 Loss-cap pause-resume: นับว่า series นี้ถูก pause จาก loss cap มาแล้วกี่ครั้ง
    pause_resume_count: int = 0
    last_loss_ts: float = 0.0
    # 🆕 Global debt: หนี้ที่ carry over ข้าม series resets — ไม่หายแม้ max_steps/halt
    global_debt_usd: float = 0.0

    def to_dict(self) -> dict:
        return {
            "cumulative_loss_usd": self.cumulative_loss_usd,
            "cumulative_losing_volume": self.cumulative_losing_volume,
            "consecutive_losses": self.consecutive_losses,
            "series_id": self.series_id,
            "last_side": self.last_side,
            "processed_tickets": sorted(self.processed_tickets)[-1000:],  # เก็บแค่ 1000 ล่าสุด ป้องกันไฟล์บวม
            "halted_until_ts": self.halted_until_ts,
            "last_loss_side": self.last_loss_side,
            "consec_same_dir_losses": self.consec_same_dir_losses,
            "blocked_side": self.blocked_side,
            "block_side_until_ts": self.block_side_until_ts,
            "pause_resume_count": self.pause_resume_count,
            "last_loss_ts": self.last_loss_ts,
            "global_debt_usd": self.global_debt_usd,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RecoveryState":
        s = cls()
        s.cumulative_loss_usd = float(d.get("cumulative_loss_usd", 0.0))
        s.cumulative_losing_volume = float(d.get("cumulative_losing_volume", 0.0))
        s.consecutive_losses = int(d.get("consecutive_losses", 0))
        s.series_id = d.get("series_id")
        s.last_side = d.get("last_side")
        s.processed_tickets = set(int(t) for t in d.get("processed_tickets", []))
        s.halted_until_ts = float(d.get("halted_until_ts", 0.0))
        s.last_loss_side = d.get("last_loss_side")
        s.consec_same_dir_losses = int(d.get("consec_same_dir_losses", 0))
        s.blocked_side = d.get("blocked_side")
        s.block_side_until_ts = float(d.get("block_side_until_ts", 0.0))
        s.pause_resume_count = int(d.get("pause_resume_count", 0))
        s.last_loss_ts = float(d.get("last_loss_ts", 0.0))
        s.global_debt_usd = float(d.get("global_debt_usd", 0.0))
        return s


# ============================================================ Bot
class HyperBot:
    def __init__(self) -> None:
        self.cfg_t = Config.section("trading")
        self.cfg_a = Config.section("ai")
        self.cfg_v = Config.section("vulnerability_patches")
        self.cfg_l = Config.section("loop")
        self.cfg_h = Config.section("hyper_frequency")
        self.cfg_sw = Config.section("session_weighting")
        self.cfg_st = Config.section("smart_trailing")
        self.cfg_r = Config.section("recovery") or {}
        self.cfg_as = Config.section("account_scaling") or {}
        self.cfg_rf = Config.section("risk_filters") or {}
        self.cfg_hl = Config.section("hourly_lot_multiplier") or {}
        self.cfg_wc = Config.section("weekend_close") or {}

        self.symbol = self.cfg_t["symbol"]
        self.timeframe = self.cfg_t.get("timeframe", "M1")
        self.magic = int(self.cfg_t["magic_number"])

        self.db = AsyncDBManager()
        self.webhook = WebhookBroadcaster()
        self.news = NewsFilter()
        self.regime = RegimeFilter()
        self.spread_tracker = DynamicSpreadTracker()

        self.model = None
        self.model_features = FEATURE_COLUMNS
        self._best_threshold: float | None = None  # set by _load_model from model_meta
        self._load_model()

        self.recovery = RecoveryState()
        self._restore_recovery()

        self._last_processed_bar: Optional[int] = None
        self._last_result_check = 0.0
        self._last_retrain_check = time.time()
        self._last_heartbeat = 0.0
        self._last_entry_ts = 0.0
        self._cached_balance: float = 0.0
        self._cached_balance_ts: float = 0.0
        self._recent_predictions: list = []   # multi-bar confirmation history
        self._last_weekend_warn_ts: float = 0.0  # rate-limit weekend log (ทุก 5 นาที)
        # Initialise retrain counter from model timestamp so restarts don't reset it
        self._trades_at_last_train = self._count_trades_before_model()

    def _count_trades_before_model(self) -> int:
        """Count settled trades that occurred BEFORE the current model was trained.
        This becomes the baseline for retrain triggering — so restart won't reset it."""
        try:
            from datetime import datetime, timezone
            p = model_path(self.cfg_a["model_filename"])
            if not p.exists():
                return 0
            model_ts = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
            return int(self.db.count_settled_before(model_ts))
        except Exception:
            return self.db.count_settled()

    # ============================================================ model
    def _load_model(self) -> bool:
        p = model_path(self.cfg_a["model_filename"])
        if not p.exists():
            log.warning("Model not found at %s — run `python run.py train` first.", p)
            self.model = None
            return False
        bundle = joblib.load(p)
        self.model = bundle["model"]
        self.model_features = bundle.get("features", FEATURE_COLUMNS)
        try:
            test = np.random.randn(10, len(self.model_features))
            proba = self.model.predict_proba(test)
            mean_max = float(proba.max(axis=1).mean())
            if mean_max < 0.40:
                log.error("[DEGENERATE MODEL] mean_max=%.3f -- bot will NOT trade", mean_max)
                self.model = None
                return False
        except Exception as e:
            log.error("Model self-test failed: %s", e)
            self.model = None
            return False
        # 🔑 Read best_threshold — config min_confidence overrides model_meta if set lower (user intentional)
        meta_p = model_path(self.cfg_a.get("metadata_filename", "model_meta.json"))
        cfg_min_conf = float(self.cfg_a.get("min_confidence", self.cfg_h.get("min_confidence", 0.58)))
        _meta_thr = None
        if meta_p.exists():
            try:
                meta = json.loads(meta_p.read_text(encoding="utf-8"))
                saved_thr = meta.get("best_threshold")
                if saved_thr is not None:
                    _meta_thr = float(saved_thr)
                    pm = meta.get("profit_metrics_oos") or {}
                    if pm:
                        log.info("📊 OOS metrics: net=%.2f ATRs | PF=%.2f | WR=%.1f%% | trades=%d",
                                 pm.get("net_pnl_atrs", 0), pm.get("profit_factor", 0),
                                 pm.get("win_rate", 0)*100, pm.get("n_trades", 0))
            except Exception as e:
                log.warning("Could not read best_threshold from meta: %s", e)
        if _meta_thr is not None and cfg_min_conf < _meta_thr:
            # Config explicitly set lower → user override
            self._best_threshold = cfg_min_conf
            log.info("⚙️  Threshold OVERRIDDEN by config: %.2f (model_meta=%.2f)", cfg_min_conf, _meta_thr)
        elif _meta_thr is not None:
            self._best_threshold = _meta_thr
            log.info("✅ Threshold loaded from model_meta: %.2f", _meta_thr)
        else:
            self._best_threshold = cfg_min_conf
            log.info("⚙️  Threshold from config: %.2f", cfg_min_conf)
        log.info("Model loaded -> %s (features=%d)", p, len(self.model_features))
        return True

    # ============================================================ recovery I/O
    def _save_recovery(self) -> None:
        try:
            data = json.dumps(self.recovery.to_dict(), indent=2)
            tmp = RECOVERY_STATE_FILE.with_suffix(".tmp")
            tmp.write_text(data, encoding="utf-8")
            tmp.replace(RECOVERY_STATE_FILE)   # atomic rename — ป้องกัน corruption จาก kill/power-loss
        except Exception as e:
            log.warning("Could not save recovery state: %s", e)

    def _restore_recovery(self) -> None:
        if RECOVERY_STATE_FILE.exists():
            try:
                d = json.loads(RECOVERY_STATE_FILE.read_text(encoding="utf-8"))
                self.recovery = RecoveryState.from_dict(d)
                log.info("📂 Recovery state restored (file): losses=%d cum_loss=$%.2f series=%s",
                         self.recovery.consecutive_losses,
                         self.recovery.cumulative_loss_usd,
                         self.recovery.series_id)
                return
            except Exception as e:
                log.warning("Could not restore recovery state file: %s — fallback to DB", e)
        # 🆕 fallback: reconstruct from DB if file missing/corrupt
        try:
            db_state = self.db.find_open_series_state(self.symbol)
        except Exception as e:
            log.warning("DB reconstruct failed: %s", e)
            db_state = None
        if db_state and db_state["consecutive_losses"] > 0:
            self.recovery.series_id = db_state["series_id"]
            self.recovery.consecutive_losses = db_state["consecutive_losses"]
            self.recovery.cumulative_loss_usd = db_state["cumulative_loss_usd"]
            self.recovery.cumulative_losing_volume = db_state["cumulative_losing_volume"]
            self.recovery.last_side = db_state["side"]
            log.warning("🔄 Recovery state RECONSTRUCTED จาก DB: series #%d เสียติด %d ไม้ ค้าง $%.2f → ต่อจากเดิม",
                        self.recovery.series_id,
                        self.recovery.consecutive_losses,
                        self.recovery.cumulative_loss_usd)
            self._save_recovery()
        else:
            log.info("📂 ไม่มี recovery state เก่า — เริ่มสด")

    # ============================================================ heartbeat
    def _log_heartbeat(self) -> None:
        """Log สถานะแบบสรุปเพื่อบอกว่าบอทยังทำงาน"""
        try:
            tick = MT5Connector.get_tick(self.symbol)
            spec = MT5Connector.get_symbol_spec(self.symbol)
        except Exception:
            log.info("💓 บอทยังทำงาน (เชื่อม MT5 ไม่ได้ชั่วขณะ)")
            return
        if tick is None:
            log.info("💓 บอทยังทำงาน (รอ tick %s)", self.symbol)
            return

        # countdown ถึงแท่งถัดไป (M1 = 60s, M5 = 300s, etc.)
        tf_sec = {"M1": 60, "M5": 300, "M15": 900, "M30": 1800, "H1": 3600}.get(self.timeframe, 60)
        next_bar = tf_sec - (int(time.time()) % tf_sec)

        live = MT5Connector.positions(symbol=self.symbol, magic=self.magic)
        spread = (tick.ask - tick.bid) / spec.point if spec.point > 0 else 0.0

        # สถานะ recovery
        if self.recovery.consecutive_losses > 0:
            rec = "💳 ค้าง $%.2f (เสีย %d ไม้, lot รวม %.3f)" % (
                self.recovery.cumulative_loss_usd,
                self.recovery.consecutive_losses,
                self.recovery.cumulative_losing_volume)
        else:
            rec = "✅ ไม่มีหนี้"

        # สถานะ position
        if live:
            p = live[0]
            side_txt = "BUY" if p.type == 0 else "SELL"
            pnl_txt = ("+$%.2f" % p.profit) if p.profit >= 0 else ("-$%.2f" % abs(p.profit))
            pos_txt = "📌 ถือ %s %.2f lot @%.3f (PnL %s)" % (side_txt, p.volume, p.price_open, pnl_txt)
        elif time.time() < self.recovery.halted_until_ts:
            wait = int(self.recovery.halted_until_ts - time.time())
            pos_txt = "🛑 หยุดเทรด อีก %d วินาที" % wait
        else:
            pos_txt = "👀 รอสัญญาณ (อีก %ds จะมีแท่งใหม่)" % next_bar

        log.info("💓 %s | bid=%.3f ask=%.3f spread=%.0fp | %s | %s",
                 datetime.now().strftime("%H:%M:%S"), tick.bid, tick.ask, spread,
                 rec, pos_txt)

    # ============================================================ session weighting
    _DOW_NAMES = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")

    def _broker_offset_hours(self) -> int:
        return int(self.cfg_sw.get("broker_offset_hours", 3))

    def _utc_to_broker_hour(self, dt: datetime) -> int:
        """แปลง datetime (UTC-aware หรือ naive=UTC) → ชั่วโมง broker 0–23."""
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return (dt.hour + self._broker_offset_hours()) % 24

    def _current_broker_hour(self) -> int:
        return self._utc_to_broker_hour(datetime.now(timezone.utc))

    def _current_broker_datetime(self) -> datetime:
        """เวลา broker ปัจจุบัน (UTC + broker_offset) — วัน+ชั่วโมงตรง heatmap."""
        return datetime.now(timezone.utc) + timedelta(hours=self._broker_offset_hours())

    def _bar_broker_datetime(self, bar_time: int) -> datetime:
        """เวลาเปิดแท่ง MT5 (UTC epoch) → เวลา broker (offset จาก config)."""
        return datetime.fromtimestamp(bar_time, tz=timezone.utc) + timedelta(
            hours=self._broker_offset_hours())

    def _is_slot_blocked(self, bar_time: int) -> tuple[bool, str]:
        """บล็อกตามวัน+ชั่วโมง broker (จากสถิติ DB) และ/หรือชั่วโมงทุกวัน."""
        if not self.cfg_sw.get("enabled", True):
            return False, ""
        bd = self._bar_broker_datetime(bar_time)
        dow, hour = bd.weekday(), bd.hour
        dow_name = self._DOW_NAMES[dow]

        for slot in self.cfg_sw.get("blocked_slots", []) or []:
            try:
                if int(slot.get("dow", -1)) != dow:
                    continue
                hours = [int(h) for h in slot.get("hours", [])]
                if hour in hours:
                    return True, f"{dow_name} {hour:02d}:xx (broker)"
            except (TypeError, ValueError):
                continue

        for h in self.cfg_sw.get("blocked_hours_broker", []) or []:
            try:
                if hour == int(h):
                    return True, f"hour {hour:02d} every day (broker)"
            except (TypeError, ValueError):
                continue

        # legacy key (ค่าใน list = broker hour ถ้ามี broker_offset_hours)
        legacy = self.cfg_sw.get("blocked_hours_utc", []) or []
        if legacy and "blocked_slots" not in self.cfg_sw and "blocked_hours_broker" not in self.cfg_sw:
            if hour in [int(x) for x in legacy]:
                return True, f"hour {hour:02d} (legacy list)"
        return False, ""

    def _session_threshold(self, base: float, hour_broker: int) -> float:
        if not self.cfg_sw.get("enabled", True):
            return base

        def in_range(rng):
            try:
                return int(rng[0]) <= hour_broker < int(rng[1])
            except Exception:
                return False

        add = 0.0
        if in_range(self.cfg_sw.get("overlap_hours_utc", [13, 16])):
            add = float(self.cfg_sw.get("overlap_threshold_add", -0.04))
        elif in_range(self.cfg_sw.get("london_hours_utc", [7, 12])):
            add = float(self.cfg_sw.get("london_threshold_add", -0.02))
        elif in_range(self.cfg_sw.get("ny_hours_utc", [17, 21])):
            add = float(self.cfg_sw.get("ny_threshold_add", -0.02))
        elif in_range(self.cfg_sw.get("asia_hours_utc", [0, 6])):
            add = float(self.cfg_sw.get("asia_threshold_add", 0.05))
        return max(0.30, min(0.95, base + add))

    def _recovery_entry_block_reason(self, side: str, feat_row) -> Optional[str]:
        """กัน recovery ซ้ำๆ โดนล่า SL ก่อนเด้ง (stop-sweep แล้วกลับ)."""
        if self.recovery.consecutive_losses <= 0:
            return None

        min_after = float(self.cfg_r.get("min_seconds_after_loss", 0))
        if min_after > 0 and self.recovery.last_loss_ts > 0:
            elapsed = time.time() - self.recovery.last_loss_ts
            if elapsed < min_after:
                return f"รอหลังไม้เสีย {int(min_after - elapsed)}s (recovery cooldown)"

        if bool(self.cfg_r.get("require_with_trend_in_recovery", False)):
            ema_dist = float(feat_row.get("ema_dist_atr", 0.0))
            if side == "BUY" and ema_dist < 0.05:
                return f"BUY recovery แต่ราคายังใต้ EMA (dist={ema_dist:.2f})"
            if side == "SELL" and ema_dist > -0.05:
                return f"SELL recovery แต่ราคายังเหนือ EMA (dist={ema_dist:.2f})"

        sweep = self.cfg_r.get("anti_sweep") or {}
        if not sweep.get("enabled", False):
            return None
        vel5 = float(feat_row.get("price_velocity_5", 0.0))
        vel2 = float(feat_row.get("price_velocity_2", 0.0))
        if side == "BUY":
            near_low = float(feat_row.get("near_low_5", 0.0))
            drop_thr = float(sweep.get("min_drop_velocity_atr", 0.8))
            low_thr = float(sweep.get("near_low_threshold", 0.85))
            rebound = float(sweep.get("min_rebound_velocity_atr", 0.12))
            if near_low >= low_thr and vel5 <= -drop_thr and vel2 < rebound:
                return (f"anti-sweep BUY: ราคาแหลมลง (near_low={near_low:.0%}, vel5={vel5:.2f}) "
                        "→ รอ rebound ก่อนเปิด")
        elif side == "SELL":
            near_high = float(feat_row.get("near_high_5", 0.0))
            rise_thr = float(sweep.get("min_rise_velocity_atr", 0.8))
            high_thr = float(sweep.get("near_high_threshold", 0.85))
            pullback = float(sweep.get("min_pullback_velocity_atr", 0.12))
            if near_high >= high_thr and vel5 >= rise_thr and vel2 > -pullback:
                return (f"anti-sweep SELL: ราคาแหลมขึ้น (near_high={near_high:.0%}, vel5={vel5:.2f}) "
                        "→ รอ pullback ก่อนเปิด")

        return None

    # ============================================================ main loop
    def run(self) -> None:
        if not MT5Connector.initialise():
            raise SystemExit("MT5 init failed")
        spec = MT5Connector.get_symbol_spec(self.symbol)
        log.info("🚀 บอทเริ่มทำงาน | %s tf=%s magic=%d (digits=%d, point=%g, min_stop=%dp)",
                 spec.name, self.timeframe, self.magic, spec.digits, spec.point, spec.stops_level)
        if self.recovery.consecutive_losses > 0:
            log.info("💼 มีหนี้ค้าง %d ไม้ ขาดทุนสะสม $%.2f (กำลัง recovery)",
                     self.recovery.consecutive_losses, self.recovery.cumulative_loss_usd)
        else:
            log.info("💼 เริ่มสด: ไม่มีขาดทุนค้าง")
        eq_pct = float(self.cfg_r.get("global_equity_stop_pct", 0.0))
        rec_cap = float(self.cfg_r.get("max_recovery_lot_multiplier", 0.0))
        if eq_pct <= 0:
            log.info("🛡️  Equity stop: ปิด (global_equity_stop_pct=0)")
        else:
            log.info("🛡️  Equity stop: เปิดที่ %.1f%% ของ balance", eq_pct)
        if rec_cap > 0:
            log.info("🛡️  Recovery lot cap: สูงสุด %.1f× base lot", rec_cap)

        try:
            while True:
                try:
                    self._tick()
                except Exception as e:
                    log.exception("loop error: %s", e)
                # 🗓️ Sleep นานขึ้นเมื่อตลาดปิด weekend เพื่อประหยัด CPU
                fc = self._friday_close_status()
                if fc["zone"] == "weekend":
                    time.sleep(float(self.cfg_wc.get("weekend_sleep_seconds", 60.0)))
                else:
                    time.sleep(float(self.cfg_l.get("poll_seconds", 0.5)))
        except KeyboardInterrupt:
            log.info("Stop requested")
        finally:
            self.db.shutdown()
            MT5Connector.shutdown()

    # ============================================================ per-tick
    def _tick(self) -> None:
        now = time.time()

        # A. Periodic background tasks
        if now - self._last_result_check >= float(self.cfg_l.get("result_update_seconds", 15)):
            self._update_closed_trades()
            self._manage_trailing()
            self._maybe_global_equity_stop()
            self._maybe_close_friday_positions()   # 🗓️ ปิด position ก่อนตลาดปิดวันศุกร์
            self.webhook.maybe_send_summary(self.db)
            self._last_result_check = now

        # A2. Heartbeat — log สถานะทุก 30s ให้รู้ว่ายังมีชีวิต
        hb_interval = float(self.cfg_l.get("heartbeat_seconds", 30))
        if hb_interval > 0 and now - self._last_heartbeat >= hb_interval:
            self._last_heartbeat = now
            self._log_heartbeat()

        # B. Self-retrain
        self._maybe_retrain()
        if self.model is None:
            return

        # C. Halt cooldown
        if now < self.recovery.halted_until_ts:
            return

        # C2. 🗓️ Friday / weekend gate (ก่อน fetch bars เพื่อประหยัด CPU)
        _fc = self._friday_close_status()
        if _fc["zone"] in ("no_new_entry", "force_close", "weekend"):
            mins_left = _fc["minutes_to_close"]
            if _fc["zone"] == "weekend":
                # Log ทุก 5 นาทีเท่านั้น (ไม่ spam)
                if now - self._last_weekend_warn_ts >= 300:
                    self._last_weekend_warn_ts = now
                    cfg_wc = Config.section("weekend_close")
                    close_h = int(cfg_wc.get("market_close_hour_utc", 21))
                    reopen_h = int(cfg_wc.get("market_reopen_hour_utc_monday", 22))
                    log.info("🗓️ ตลาดปิด Weekend — บอทรอจนกว่าจะเปิดวันจันทร์ ~%02d:00 UTC "
                             "(close %02d:00 UTC ศุกร์) | หยุด 60s/rnd", reopen_h, close_h)
            else:
                log.info("🗓️ Friday close zone [%s]: เหลือ %.1f นาทีก่อนตลาดปิด → หยุดเปิดออเดอร์ใหม่",
                         _fc["zone"], mins_left)
            return

        # D. New closed bar?
        bars_needed = max(int(self.cfg_t.get("history_bars_for_inference", 200)), 100)
        try:
            rates = MT5Connector.copy_rates(self.symbol, self.timeframe, bars_needed)
        except Exception as e:
            log.info("⚠️  rates fetch failed: %s", e)
            return
        if len(rates) < 50:
            log.info("⚠️  ข้อมูล rates น้อยเกิน (%d bars) → ข้ามแท่ง", len(rates))
            return
        closed_idx = int(self.cfg_v.get("use_only_closed_candle_index", -2))
        bar_time = int(rates[closed_idx]["time"])
        if self._last_processed_bar == bar_time:
            return
        self._last_processed_bar = bar_time
        log.debug("📊 แท่งใหม่ bar_time=%d → ประมวลผล", bar_time)

        # E. Build features
        try:
            df = build_features(rates)
            atr_value = float(df.iloc[closed_idx]["atr"])
            if not np.isfinite(atr_value) or atr_value <= 0:
                log.info("⚠️  ATR invalid (%.4f) → ข้ามแท่ง", atr_value)
                return
            feat_row = df.iloc[closed_idx][FEATURE_COLUMNS]
            if feat_row.isna().any():
                nan_cols = feat_row[feat_row.isna()].index.tolist()
                log.info("⚠️  Feature NaN → ข้ามแท่ง | cols=%s", nan_cols)
                return
        except Exception as e:
            log.info("⚠️  Feature build error → ข้ามแท่ง: %s", e)
            return

        # 🆕 🅑 ATR-Spike Filter — กันเทรดตอนตลาด volatile ผิดปกติ
        # ถ้า ATR ปัจจุบันสูงกว่าค่าเฉลี่ย rolling × max_ratio → skip (loss/ไม้จะใหญ่เกิน)
        atr_cfg = self.cfg_rf.get("atr_spike", {})
        if atr_cfg.get("enabled", True):
            try:
                lookback = int(atr_cfg.get("rolling_bars", 50))
                max_ratio = float(atr_cfg.get("max_atr_ratio", 2.0))
                atr_series = df["atr"].tail(lookback + 1).iloc[:-1]
                atr_mean = float(atr_series.mean()) if len(atr_series) > 5 else 0.0
                if atr_mean > 0:
                    ratio = atr_value / atr_mean
                    if ratio > max_ratio:
                        log.info("🚫 ATR spike: ATR=%.2f สูงกว่าค่าเฉลี่ย %d แท่ง (%.2f) %.1f× (เกิน %.1f×) → ข้าม",
                                 atr_value, lookback, atr_mean, ratio, max_ratio)
                        return
            except Exception as e:
                log.debug("atr-spike check skip: %s", e)

        # F. spread tracker
        spec = MT5Connector.get_symbol_spec(self.symbol)
        tick = MT5Connector.get_tick(self.symbol)
        if tick is None or tick.ask <= 0 or tick.bid <= 0:
            log.info("⚠️  ไม่มี tick data → ข้ามแท่ง (tick=%s)", tick)
            return
        cur_spread = (tick.ask - tick.bid) / spec.point
        self.spread_tracker.update(cur_spread)

        # G. Already in trade? -> skip new entry
        live = MT5Connector.positions(symbol=self.symbol, magic=self.magic)
        if live:
            log.info("📌 มี position %d ไม้อยู่แล้ว (ticket=%s) → ข้ามการเปิดใหม่",
                     len(live), [p.ticket for p in live])
            return

        # H. News block
        blocked, reason = self.news.is_blocked()
        if blocked:
            log.info("📰 หยุดเทรด: ใกล้ข่าวสำคัญ — %s", reason)
            return

        # I. Inference
        X = feat_row.to_numpy().reshape(1, -1)
        proba = self.model.predict_proba(X)[0]
        pred = int(np.argmax(proba))
        conf = float(proba[pred])
        side = "BUY" if pred == 2 else ("SELL" if pred == 0 else "HOLD")

        # 🆕 Same-Direction Recovery — ระหว่าง recovery บังคับเทรดทิศเดิมกับที่เสีย
        # เหตุผล: ถ้าเสีย BUY → แสดงว่า setup ยังไม่หมด → กู้ด้วย BUY ต่อ ไม่ใช่ SELL
        in_recovery = self.recovery.consecutive_losses > 0 and self.recovery.last_side is not None
        if in_recovery and bool(self.cfg_r.get("same_direction_enabled", False)):
            forced_side = self.recovery.last_side
            forced_pred = 2 if forced_side == "BUY" else 0
            forced_conf = float(proba[forced_pred])
            if pred != forced_pred:
                log.info("♻️  Same-dir recovery: AI=%s %.1f%% แต่บังคับ %s %.1f%% (เสียติด %d ไม้ทิศ %s)",
                         side, conf*100, forced_side, forced_conf*100,
                         self.recovery.consecutive_losses, forced_side)
            pred = forced_pred
            conf = forced_conf
            side = forced_side

        # Always use best_threshold from model_meta (set at load time) — exact match to OOS simulation
        base_thr = self._best_threshold if self._best_threshold is not None else \
                   float(self.cfg_h.get("min_confidence", self.cfg_a.get("min_confidence", 0.45)))
        bar_broker = self._bar_broker_datetime(bar_time)
        hour_broker = bar_broker.hour

        blocked, block_reason = self._is_slot_blocked(bar_time)
        if blocked:
            log.info("🕐 Blocked slot %s → ข้ามแท่ง", block_reason)
            return

        thr = self._session_threshold(base_thr, hour_broker)

        # 🆕 Recovery threshold reduction — ลด threshold ระหว่าง recovery เพื่อให้เทรดถี่ขึ้น
        thr_reduce = float(self.cfg_r.get("recovery_threshold_reduce", 0.0))
        if in_recovery and thr_reduce > 0:
            old_thr = thr
            thr = max(0.35, thr - thr_reduce)
            log.debug("♻️  Recovery threshold: %.2f → %.2f (ลด %.2f)", old_thr, thr, thr_reduce)

        # 🆕 Per-direction confidence override (anti-bias)
        # ถ้า model มี directional bias (เช่น BUY แม่นน้อยกว่า SELL) → ตั้ง threshold คนละค่า
        dir_thr_cfg = self.cfg_h.get("directional_threshold", {})
        max_thr = None  # 🆕 upper bound (skip overfit zone)
        if dir_thr_cfg.get("enabled", False):
            if side == "BUY":
                thr = float(dir_thr_cfg.get("buy", thr))
                if "buy_max" in dir_thr_cfg:
                    max_thr = float(dir_thr_cfg["buy_max"])
            elif side == "SELL":
                thr = float(dir_thr_cfg.get("sell", thr))
                if "sell_max" in dir_thr_cfg:
                    max_thr = float(dir_thr_cfg["sell_max"])

        bar_dt = datetime.fromtimestamp(bar_time, tz=timezone.utc)
        debt_str = ("💳 ค้าง $%.2f (เสียติด %d ไม้)" %
                    (self.recovery.cumulative_loss_usd, self.recovery.consecutive_losses)
                    if self.recovery.consecutive_losses > 0 else "✅ ไม่ค้าง")
        # 🆕 max threshold check — skip overconfident overfit zone
        if pred != 1 and max_thr is not None and conf > max_thr:
            log.info("⏸️  %s | AI=%s %.1f%% > เพดาน %.1f%% (overfit zone) → ข้าม | %s | atr=%.3f spread=%.0fp",
                     bar_dt.strftime("%H:%M"), CLASS_NAMES[pred], conf*100, max_thr*100,
                     debt_str, atr_value, cur_spread)
            return
        if pred == 1 or conf < thr:
            if pred == 1:
                # 🤚 AI ตัดสินใจ HOLD — ตลาดไม่มีสัญญาณชัด (อาจ confidence สูงก็ได้)
                log.info("⏸️  %s | AI=HOLD %.1f%% (รอจังหวะ ไม่มีสัญญาณชัด) → ข้าม | %s | atr=%.3f spread=%.0fp",
                         bar_dt.strftime("%H:%M"), conf*100, debt_str, atr_value, cur_spread)
            else:
                # 📉 AI predict BUY/SELL แต่มั่นใจไม่พอ
                log.info("⏸️  %s | AI=%s %.1f%% < ขั้นต่ำ %.1f%% → ข้าม | %s | atr=%.3f spread=%.0fp",
                         bar_dt.strftime("%H:%M"), CLASS_NAMES[pred], conf*100, thr*100,
                         debt_str, atr_value, cur_spread)
            return
        log.info("🎯 %s | AI=%s %.1f%% ≥ %.1f%% → จะเทรด | %s | atr=%.3f spread=%.0fp",
                 bar_dt.strftime("%H:%M"), CLASS_NAMES[pred], conf*100, thr*100,
                 debt_str, atr_value, cur_spread)

        # 🆕 🅐 Direction-Flip Lock — กันเคส "AI ส่งทิศเดียวซ้ำตอน trend แรง" (catching rocket)
        # ถ้าเสีย N ไม้ติดทิศเดียว → block ทิศนั้น cooldown_minutes
        df_cfg = self.cfg_rf.get("direction_flip", {})
        if df_cfg.get("enabled", True):
            # Cooldown active?
            if (self.recovery.blocked_side == side
                    and time.time() < self.recovery.block_side_until_ts):
                wait = int(self.recovery.block_side_until_ts - time.time())
                log.info("🚫 Direction lock: %s ยัง block อีก %ds (เสียติด %d ไม้ทาง %s) → ข้าม",
                         side, wait, self.recovery.consec_same_dir_losses,
                         self.recovery.last_loss_side)
                return
            # Trigger fresh block?
            min_consec = int(df_cfg.get("min_consec_same_dir_losses", 2))
            if (self.recovery.last_loss_side == side
                    and self.recovery.consec_same_dir_losses >= min_consec
                    and self.recovery.blocked_side != side):
                cooldown_min = float(df_cfg.get("cooldown_minutes", 30))
                self.recovery.blocked_side = side
                self.recovery.block_side_until_ts = time.time() + cooldown_min * 60
                self._save_recovery()
                log.warning("🛑 Direction-flip lock activated: เสีย %s ติด %d ไม้ → block %s %d นาที",
                            side, self.recovery.consec_same_dir_losses, side, int(cooldown_min))
                return

        # Trend filter: ใช้ ema_dist_atr (price - EMA20 หาร ATR)
        # BUY ต้องมี price เหนือ EMA, SELL ต้องอยู่ใต้
        if bool(self.cfg_h.get("trend_filter_enabled", True)):
            min_dist = float(self.cfg_h.get("trend_min_ema_dist_atr", 0.1))
            ema_dist = float(feat_row.get("ema_dist_atr", 0.0))
            if side == "BUY" and ema_dist < min_dist:
                log.info("🚫 ทวนเทรนด์: BUY แต่ราคายังอยู่ใต้/ใกล้ EMA (dist=%.2f ATR < %.2f) → ข้าม",
                         ema_dist, min_dist)
                return
            if side == "SELL" and ema_dist > -min_dist:
                log.info("🚫 ทวนเทรนด์: SELL แต่ราคายังอยู่เหนือ/ใกล้ EMA (dist=%.2f ATR > -%.2f) → ข้าม",
                         ema_dist, min_dist)
                return

        # 🆕 Exhaustion Filter — ป้องกัน FOMO entry (เข้าตอนสุดเทรนด์)
        # ถ้าราคาวิ่งไปทาง signal มากเกินไปแล้ว → มีแนวโน้มกลับตัว → skip
        ex_cfg = self.cfg_h.get("exhaustion_filter", {})
        if ex_cfg.get("enabled", True):
            max_velocity = float(ex_cfg.get("max_velocity_5_atr", 1.5))
            max_near_extreme = float(ex_cfg.get("max_near_extreme", 0.85))
            velocity_5 = float(feat_row.get("price_velocity_5", 0.0))
            near_high_5 = float(feat_row.get("near_high_5", 0.5))
            near_low_5 = float(feat_row.get("near_low_5", 0.5))
            if side == "BUY":
                if velocity_5 > max_velocity:
                    log.info("🚫 FOMO guard: BUY แต่ราคาวิ่งขึ้นแล้ว %.2f ATR ใน 5 bars (เกิน %.2f) → ข้าม (กลัวกลับตัว)",
                             velocity_5, max_velocity)
                    return
                if near_high_5 > max_near_extreme:
                    log.info("🚫 FOMO guard: BUY แต่ราคาอยู่ใกล้ high 5 bars แล้ว (%.0f%% > %.0f%%) → ข้าม",
                             near_high_5*100, max_near_extreme*100)
                    return
            elif side == "SELL":
                if velocity_5 < -max_velocity:
                    log.info("🚫 FOMO guard: SELL แต่ราคาวิ่งลงแล้ว %.2f ATR ใน 5 bars (เกิน -%.2f) → ข้าม (กลัวกลับตัว)",
                             velocity_5, max_velocity)
                    return
                if near_low_5 > max_near_extreme:
                    log.info("🚫 FOMO guard: SELL แต่ราคาอยู่ใกล้ low 5 bars แล้ว (%.0f%% > %.0f%%) → ข้าม",
                             near_low_5*100, max_near_extreme*100)
                    return

        ok, reason = spread_guard(spec, cur_spread, self.spread_tracker)
        if not ok:
            log.info("🚫 ยกเลิก: spread กว้างเกิน — %s", reason)
            return

        rec_block = self._recovery_entry_block_reason(side, feat_row)
        if rec_block:
            log.info("🛡️  Recovery guard: %s → ข้าม", rec_block)
            return

        # 🆕 Tick-Level Confirmation — ดู tick ล่าสุดก่อนยิง (กัน fake signal/stop hunt)
        tick_cfg = self.cfg_h.get("tick_confirmation", {})
        if tick_cfg.get("enabled", False):
            try:
                from .tick_analyzer import analyze_ticks_for_signal
                in_rec = self.recovery.consecutive_losses > 0
                strict = bool(self.cfg_r.get("strict_tick_in_recovery", True)) and in_rec
                max_hunt = float(tick_cfg.get("max_stop_hunt_score", 0.7))
                max_burst = float(tick_cfg.get("max_velocity_burst", 0.85))
                micro_align = bool(tick_cfg.get("require_micro_trend_aligned", False))
                if strict:
                    max_hunt = min(max_hunt, float(self.cfg_r.get("max_stop_hunt_score_recovery", 0.55)))
                    max_burst = min(max_burst, float(self.cfg_r.get("max_velocity_burst_recovery", 0.7)))
                    micro_align = True
                ta = analyze_ticks_for_signal(
                    symbol=self.symbol, side=side,
                    seconds_back=int(tick_cfg.get("seconds_back", 60)),
                    min_ticks=int(tick_cfg.get("min_ticks", 20)),
                    max_spread_zscore=float(tick_cfg.get("max_spread_zscore", 3.0)),
                    min_buy_ratio_for_buy=float(tick_cfg.get("min_directional_ratio", 0.55)),
                    max_stop_hunt_score=max_hunt,
                    max_velocity_burst=max_burst,
                    require_micro_trend_aligned=micro_align,
                )
                if not ta.confirmed:
                    log.info("🚫 Tick reject: %s | %s", ta.reason, ta.to_log())
                    return
                log.info("✅ Tick confirm: %s", ta.to_log())
            except Exception as e:
                log.debug("tick analysis failed (skip): %s", e)

        # 🆕 Multi-bar confirmation — ป้องกัน whipsaw
        # เก็บ prediction ของแท่งล่าสุด ถ้า require_consecutive_bars=2 → ต้อง 2 แท่งติดเห็นทิศเดียวกัน
        require_n = int(self.cfg_h.get("require_consecutive_bars", 1))
        self._recent_predictions.append(pred)
        if len(self._recent_predictions) > 5:
            self._recent_predictions = self._recent_predictions[-5:]
        if require_n > 1 and len(self._recent_predictions) >= require_n:
            last_n = self._recent_predictions[-require_n:]
            if not all(p == pred for p in last_n):
                log.info("🚫 รอยืนยัน: ต้องการ %d แท่งติด predict %s แต่ล่าสุด %s → ข้าม",
                         require_n, CLASS_NAMES[pred],
                         "/".join(CLASS_NAMES[p] for p in last_n))
                return

        # cooldown ระหว่างไม้ (กัน over-trading จาก noise)
        cooldown = float(self.cfg_h.get("min_seconds_between_entries", 0))
        if cooldown > 0 and (time.time() - self._last_entry_ts) < cooldown:
            wait = int(cooldown - (time.time() - self._last_entry_ts))
            log.info("⏱️  รอ cooldown อีก %ds (ป้องกันเทรดถี่เกิน)", wait)
            return

        # 🌊 Regime Filter — ตรวจ HTF trend + daily DD + recovery-in-trend
        balance_for_regime = self._get_balance()
        rgm = self.regime.evaluate(side=side, symbol=self.symbol,
                                   balance=balance_for_regime,
                                   in_recovery=(self.recovery.consecutive_losses > 0))
        if not rgm.allow_entry:
            log.warning("🌊 Regime block: %s | %s", rgm.reason, self.regime.to_log_str(rgm))
            # ถ้าเป็น DAILY_DD → halt ตาม config (ดีฟอลต์ 4 ชม.) ไม่ใช่ถึงเที่ยงคืนอีกต่อไป
            if rgm.reason.startswith("DAILY_DD_HALT"):
                cfg_dd = (Config.section("regime_filter") or {}).get("daily_drawdown", {})
                halt_hours = float(cfg_dd.get("halt_hours_after_trip", 4.0))
                halt_until = time.time() + halt_hours * 3600
                self.recovery.halted_until_ts = max(
                    self.recovery.halted_until_ts, halt_until)
                self._save_recovery()
                log.warning("⏸️  DD halt เปิด %.1f ชม. (ไม่รอถึงเที่ยงคืน) — ปรับ halt_hours_after_trip ได้",
                            halt_hours)
            return
        if rgm.lot_factor < 1.0 or rgm.skip_recovery_escalation:
            log.info("🌊 Regime modifier: %s | %s",
                     rgm.reason, self.regime.to_log_str(rgm))

        # J. lot calc + open
        lot = self._compute_lot_for_recovery(
            spec, atr_value, skip_recovery_escalation=rgm.skip_recovery_escalation)
        if lot <= 0:
            log.warning("Computed lot <= 0, skip")
            return
        # apply regime lot factor (weak counter-trend = 0.5×)
        if rgm.lot_factor < 1.0:
            lot = round(lot * rgm.lot_factor, 2)
        base_lot = lot
        # 🆕 Hourly lot multiplier — เพิ่ม lot ในชั่วโมงดี ลดในชั่วโมงแย่
        lot = self._apply_hourly_lot_multiplier(lot, spec)
        # 🆕 Confidence-tier lot multiplier — บูสต์ตอน AI มั่นใจสูง
        lot = self._apply_confidence_lot_multiplier(lot, spec, conf)
        # safety: clamp อย่าให้เกิน max_total_multiplier × base_lot
        cfg_cm = Config.section("confidence_lot_multiplier") or {}
        max_total = float(cfg_cm.get("max_total_multiplier", 2.0))
        lot_cap = round(base_lot * max_total, 2)
        if lot > lot_cap and base_lot > 0:
            log.info("🛡️  Lot cap: %.2f → %.2f (× %.2f base %.2f)",
                     lot, lot_cap, max_total, base_lot)
            lot = lot_cap
        if lot <= 0:
            return
        self._open_trade(side, conf, atr_value, cur_spread, lot)
        self._last_entry_ts = time.time()

    def _resolve_hourly_lot_multiplier(self, dow: int, hour: int) -> tuple[float, str]:
        """
        ลำดับ lookup:
          1) slot_multipliers — แยกวัน×ชั่วโมง (ตรง heatmap / blocked_slots)
          2) multipliers — ชั่วโมงเดียวทุกวัน (legacy)
          3) default
        """
        default = float(self.cfg_hl.get("default", 1.0))
        hour_mults = self.cfg_hl.get("multipliers", {}) or {}
        slot_cfg = self.cfg_hl.get("slot_multipliers")

        if slot_cfg:
            if isinstance(slot_cfg, dict):
                v = slot_cfg.get(f"{dow}_{hour}")
                if v is not None:
                    return float(v), "slot"
            elif isinstance(slot_cfg, list):
                for entry in slot_cfg:
                    try:
                        if int(entry.get("dow", -1)) == dow and int(entry.get("hour", -1)) == hour:
                            return float(entry.get("mult", default)), "slot"
                    except (TypeError, ValueError):
                        continue

        if str(hour) in hour_mults:
            return float(hour_mults[str(hour)]), "hour"
        return default, "default"

    # 🆕 Hourly Lot Multiplier — วัน×ชั่วโมง broker (ตรง heatmap / blocked_slots)
    def _apply_hourly_lot_multiplier(self, lot: float, spec) -> float:
        """
        ปรับ lot ตามช่วง broker ปัจจุบัน:
          - slot_multipliers: แยกวัน+ชั่วโมง (Mon=0 … Sun=6, ชม. 0–23)
          - multipliers: ชั่วโมงเดียวทุกวัน (fallback ถ้าไม่มี slot)
        """
        if not self.cfg_hl.get("enabled", False):
            return lot
        if self.cfg_hl.get("disable_during_recovery", True) and self.recovery.consecutive_losses > 0:
            return lot
        bd = self._current_broker_datetime()
        dow, h = bd.weekday(), bd.hour
        mult, src = self._resolve_hourly_lot_multiplier(dow, h)
        if abs(mult - 1.0) < 1e-6:
            return lot
        new_lot = lot * mult
        min_lot = float(self.cfg_hl.get("min_lot", spec.volume_min if hasattr(spec, "volume_min") else 0.01))
        if new_lot < min_lot:
            new_lot = min_lot
        new_lot = round(new_lot, 2)
        emoji = "🚀" if mult > 1.2 else ("🐢" if mult < 0.8 else "➖")
        off = self._broker_offset_hours()
        dow_name = self._DOW_NAMES[dow]
        log.info(
            "%s Lot mult [%s]: %s %02d:xx broker (UTC%+d, %s) × %.2f → lot %.2f → %.2f",
            emoji, src, dow_name, h, off, f"{dow}_{h}", mult, lot, new_lot,
        )
        return new_lot

    # 🆕 Confidence-Tier Lot Multiplier — บูสต์ lot ตอน AI มั่นใจสูง
    def _apply_confidence_lot_multiplier(self, lot: float, spec, conf: float) -> float:
        """
        ปรับ lot ตามระดับความมั่นใจ AI:
          - Tiered: หา min_conf สูงสุดที่ conf ผ่าน → ใช้ mult ของ tier นั้น
          - Disable ตอน recovery (เพื่อไม่ให้ recovery lot ระเบิดซ้ำ)
        Config: confidence_lot_multiplier.{enabled, disable_during_recovery, tiers}
        """
        cfg = Config.section("confidence_lot_multiplier") or {}
        if not cfg.get("enabled", False):
            return lot
        # 🛡️ ตอน recovery — ใช้ lot ตาม Recovery Engine ตรงๆ
        if cfg.get("disable_during_recovery", True) and self.recovery.consecutive_losses > 0:
            return lot
        tiers = cfg.get("tiers") or []
        if not tiers:
            return lot
        # หา tier สูงสุดที่ conf ผ่าน
        mult = 1.0
        matched = None
        for t in sorted(tiers, key=lambda x: float(x.get("min_conf", 0))):
            if conf >= float(t.get("min_conf", 0)):
                mult = float(t.get("mult", 1.0))
                matched = t
        if abs(mult - 1.0) < 1e-6:
            return lot
        new_lot = round(lot * mult, 2)
        emoji = "🔥" if mult >= 1.5 else "⚡"
        log.info("%s Conf-tier mult: conf=%.1f%% × %.2f → lot %.2f → %.2f",
                 emoji, conf*100, mult, lot, new_lot)
        return new_lot

    # ============================================================ lot calc
    def _dynamic_lot_caps(self, spec, atr_value: float) -> tuple:
        """
        คำนวณ base_lot + max_lot_cap แบบ dynamic ตาม balance ปัจจุบัน
        เพื่อรองรับพอร์ตขนาด $500 - $10M+ โดยไม่ต้องแก้ config มือ.

        Returns: (base_lot, max_lot_cap, balance_used)

        สูตร:
          - base_lot = (balance × risk_pct/100) ÷ sl_distance_usd_per_lot
          - max_lot_cap = balance × max_lot_pct/100 ÷ price_per_lot_proxy
          ทั้งคู่ clamp ระหว่าง min/max ของ config
        """
        cfg = self.cfg_as
        # ถ้าไม่เปิด -> ใช้ค่า static เดิม
        if not cfg.get("enabled", True):
            base = float(self.cfg_t.get("base_lot", 0.01))
            cap = float(self.cfg_r.get("max_lot_cap", 0.5))
            return base, cap, 0.0

        # cache balance (ไม่ต้อง query MT5 ทุก tick)
        now = time.time()
        refresh = float(cfg.get("balance_refresh_seconds", 300))
        if now - self._cached_balance_ts > refresh or self._cached_balance <= 0:
            try:
                acc = mt5.account_info()
                if acc and acc.balance > 0:
                    self._cached_balance = float(acc.balance)
                    self._cached_balance_ts = now
            except Exception:
                pass
        balance = self._cached_balance

        # fallback ถ้าไม่ได้ balance
        if balance <= 0:
            base = float(self.cfg_t.get("base_lot", 0.01))
            cap = float(self.cfg_r.get("max_lot_cap", 0.5))
            return base, cap, 0.0

        # คำนวณ base_lot: เสี่ยง X% ของ balance ต่อ SL hit
        risk_pct = float(cfg.get("risk_per_trade_pct", 0.30))
        risk_usd = balance * risk_pct / 100.0

        sl_mult = float(self.cfg_t.get("sl_atr_mult", 1.0))
        sl_distance = sl_mult * atr_value
        if spec.trade_tick_size <= 0 or spec.trade_tick_value <= 0:
            base = float(self.cfg_t.get("base_lot", 0.01))
            cap = float(self.cfg_r.get("max_lot_cap", 0.5))
            return base, cap, balance

        usd_per_price_per_lot = spec.trade_tick_value / spec.trade_tick_size
        loss_per_lot = sl_distance * usd_per_price_per_lot
        if loss_per_lot <= 0:
            base = float(self.cfg_t.get("base_lot", 0.01))
            cap = float(self.cfg_r.get("max_lot_cap", 0.5))
            return base, cap, balance

        base_lot_dyn = risk_usd / loss_per_lot

        # คำนวณ max_lot_cap: ใช้ % notional ของ balance
        # notional ~ price × lot × contract_size; ใช้ proxy ผ่าน loss_per_lot × 100 (โดยประมาณ leverage)
        max_pct = float(cfg.get("max_lot_pct_of_balance", 5.0))
        # cap = (balance × max_pct/100) ÷ loss_per_lot × multiplier เพื่อให้ความเสี่ยงสูงสุด ~ max_pct%
        max_cap_dyn = (balance * max_pct / 100.0) / max(loss_per_lot, 1e-9)

        # clamp ทั้งคู่
        base_lot = max(float(cfg.get("min_base_lot", 0.01)),
                       min(base_lot_dyn, float(cfg.get("max_base_lot", 50.0))))
        max_cap = max(float(cfg.get("min_lot_cap", 0.05)),
                      min(max_cap_dyn, float(cfg.get("max_lot_cap_absolute", 100.0))))

        # max_cap ต้อง >= base_lot (กัน edge case)
        if max_cap < base_lot:
            max_cap = base_lot * 5.0

        return base_lot, max_cap, balance

    # ============================================================ lot calc
    def _compute_lot_for_recovery(self, spec, atr_value: float,
                                   skip_recovery_escalation: bool = False) -> float:
        """
        คำนวณ lot สำหรับไม้ถัดไป + รองรับการ scale ตามขนาดบัญชี ($500 - $10M).

        skip_recovery_escalation=True → ใช้ base lot (ไม่ใช่ geometric/recover/volume floors)
        ใช้โดย Regime Filter ตอนเจอ strong trend + recovery (เพื่อไม่ double-down).

        ขั้นตอน:
          1. หา base_lot และ cap แบบ dynamic จาก balance (account_scaling)
          2. ถ้าไม่มีหนี้ -> base_lot
          3. ถ้ามีหนี้ -> max ของ 3 floors (geo, recover, volume) capped
        """
        # 1) หา dynamic base + cap ตามขนาดบัญชี
        base, absolute_cap, balance = self._dynamic_lot_caps(spec, atr_value)

        # ── Gentle Slope Recovery: ไม่มี active series แต่มี global_debt ──────────────
        # ใช้ lot สูงกว่า base นิดหน่อย เพื่อกู้หนี้ carry-over ทีละน้อย (safe, ไม่ martingale)
        global_debt = self.recovery.global_debt_usd
        if (not self.cfg_r.get("enabled", True) or self.recovery.consecutive_losses <= 0):
            if global_debt > 0.5:
                tp_mult_g = float(self.cfg_t.get("tp_atr_mult", 2.0))
                tp_dist_g = tp_mult_g * atr_value
                if spec.trade_tick_size > 0 and spec.trade_tick_value > 0 and tp_dist_g > 0:
                    com_g = float(Config.section("commission").get("per_lot_round_trip_usd", 0.0))
                    net_g = (tp_dist_g * spec.trade_tick_value / spec.trade_tick_size) - com_g
                    if net_g > 0:
                        target_trades = float(self.cfg_r.get("debt_recovery_target_trades", 50))
                        needed = global_debt / (target_trades * net_g)
                        max_mult = float(self.cfg_r.get("debt_recovery_max_lot_mult", 3.0))
                        elevated = min(base + needed, base * max_mult, absolute_cap)
                        elevated = spec.normalize_volume(elevated)
                        if elevated > base:
                            log.info("💳 Gentle slope: global_debt=$%.2f → lot %.2f (base=%.2f, ≈%d ไม้จะคืนหนี้)",
                                     global_debt, elevated, base, int(global_debt / max(net_g * (elevated - base), 1e-9)))
                            return elevated
            return spec.normalize_volume(base)

        # 🌊 Regime override: ใน strong trend → ใช้ base lot เท่านั้น (no escalation)
        if skip_recovery_escalation:
            log.info("🌊 Recovery escalation SKIPPED (strong trend) → base lot %.2f", base)
            return spec.normalize_volume(base)

        tp_mult = float(self.cfg_t.get("tp_atr_mult", 2.0))
        tp_distance = tp_mult * atr_value
        if spec.trade_tick_size <= 0 or spec.trade_tick_value <= 0:
            return spec.normalize_volume(base)
        profit_per_price_per_lot = spec.trade_tick_value / spec.trade_tick_size
        profit_per_lot = tp_distance * profit_per_price_per_lot

        com_per_lot = float(Config.section("commission").get("per_lot_round_trip_usd", 0.0))
        net_per_lot = profit_per_lot - com_per_lot
        if net_per_lot <= 0:
            log.error("commission > expected profit at TP. Increase tp_atr_mult.")
            return spec.normalize_volume(base)

        # 1) Floor: geometric growth - lot ต้องโตแบบ exponentially หลังเสียทุกครั้ง
        mult = float(self.cfg_r.get("lot_multiplier", 1.7))
        max_steps = int(self.cfg_r.get("max_steps", 4))
        steps_used = (self.recovery.consecutive_losses if max_steps <= 0
                      else min(self.recovery.consecutive_losses, max_steps))
        geometric_floor = base * (mult ** steps_used)

        # 2) Recovery target: กู้ทีเดียว (spread=1) ถ้าหนี้ไม่ใหญ่เกินไป
        #    one_shot_max_debt_pct_of_balance: หนี้ > X% balance → กระจายอัตโนมัติให้ lot ไม่เกิน cap
        import math
        min_profit = float(self.cfg_r.get("min_profit_target_usd", 1.0))
        spread = max(1, int(self.cfg_r.get("recovery_spread_trades", 1)))
        target_usd = self.recovery.cumulative_loss_usd + min_profit
        max_debt_pct = float(self.cfg_r.get("one_shot_max_debt_pct_of_balance", 0))
        if max_debt_pct > 0 and balance > 0 and target_usd > 0:
            debt_limit = balance * max_debt_pct / 100.0
            max_win_at_cap = absolute_cap * net_per_lot
            if target_usd > debt_limit and max_win_at_cap > 0:
                spread = max(spread, int(math.ceil(target_usd / max_win_at_cap)))
                log.info(
                    "🛡️ กู้ทีเดียวปลอดภัย: หนี้ $%.2f > %.0f%% balance ($%.0f) → "
                    "กระจาย %d ไม้ (lot สูงสุด %.2f, win/ไม้≈$%.0f)",
                    target_usd, max_debt_pct, debt_limit, spread, absolute_cap, max_win_at_cap)
        recovery_lot = target_usd / (net_per_lot * spread)

        # 3) Volume floor: lot ต้องใหญ่กว่า "ผลรวม lot ทุกไม้ที่เสีย" × profit_multiplier
        #    เพื่อให้กำไรของไม้นี้มากกว่ายอด lot ที่เสียไปรวมกัน
        prof_vol_mult = float(self.cfg_r.get("profit_volume_multiplier", 1.3))
        volume_floor = self.recovery.cumulative_losing_volume * prof_vol_mult

        # 4) cap strategy:
        #   geometric_floor: capped ที่ rec_lot_cap (base × max_recovery_lot_multiplier)
        #   volume_floor:    capped ที่ rec_lot_cap — กัน exponential runaway
        #   recovery_lot:    capped ที่ absolute_cap (balance-based) เท่านั้น
        #                    ต้องไม่ถูก cap ด้วย rec_lot_cap เพราะ formula นี้คำนวณ exact lot
        #                    ที่พอ cover debt ใน 1 win — ถ้า cap ต่ำกว่า debt ก็ไม่มีวันหมด
        max_rec_mult = float(self.cfg_r.get("max_recovery_lot_multiplier", 2.5))
        rec_lot_cap = (base * max_rec_mult) if max_rec_mult > 0 else absolute_cap
        geometric_floor = min(geometric_floor, rec_lot_cap)
        volume_floor    = min(volume_floor, rec_lot_cap)
        recovery_lot    = min(recovery_lot, absolute_cap)   # ← balance cap เท่านั้น
        # Final = max ของทั้ง 3 floors แล้ว cap ด้วย absolute_cap เท่านั้น
        # (ไม่ cap ด้วย rec_lot_cap เพราะ recovery_lot อาจสูงกว่า rec_lot_cap และถูกต้อง)
        candidate = max(geometric_floor, recovery_lot, volume_floor)
        step = spec.volume_step if spec.volume_step > 0 else 0.01
        candidate = math.ceil(candidate / step - 1e-9) * step
        final = min(candidate, absolute_cap)   # ← แก้แล้ว: cap ด้วย balance เท่านั้น
        final = spec.normalize_volume(final)

        # log แบบเล่าเรื่อง: ทำไมเลือก lot นี้
        which = ("geo" if candidate == geometric_floor else
                 "recover" if candidate == recovery_lot else "vol")
        why = {
            "geo":     "แบบเรขาคณิต (×%.1f^%d)" % (mult, steps_used),
            "recover": "พอ recover ขาดทุน",
            "vol":     "ใหญ่กว่ารวม lot ที่เสีย ×%.1f" % prof_vol_mult,
        }[which]
        attempt = self.recovery.consecutive_losses + 1
        max_s = int(self.cfg_r.get("max_steps", 4)) or 4
        log.info(
            "🧮 ไม้กู้ #%d/%d: ค้าง $%.2f → lot %.2f [ถ้าชนะปิดหนี้ทั้งก้อน | %s | geo=%.3f ต้อง=%.3f cap=%.2f]",
            attempt, max_s, self.recovery.cumulative_loss_usd, final, why,
            geometric_floor, recovery_lot, absolute_cap)
        return final

    # ============================================================ open trade
    def _open_trade(self, side: str, conf: float, atr_value: float,
                    cur_spread: float, lot: float) -> None:
        spec = MT5Connector.get_symbol_spec(self.symbol)

        if self.recovery.series_id is None:
            self.recovery.series_id = self.db.open_series(self.symbol, side)
        sid = self.recovery.series_id

        step = self.recovery.consecutive_losses + 1
        role = "PRIMARY" if self.recovery.consecutive_losses == 0 else "RECOVERY"

        # 🆕 Tag with current config snapshot — for performance-weighted retrain later
        try:
            from .config_snapshot import get_current_snapshot_id
            snap_id = get_current_snapshot_id()
        except Exception:
            snap_id = None

        decision_id = self.db.insert_decision(
            series_id=sid, step=step, role=role,
            symbol=self.symbol, timeframe=self.timeframe,
            prediction=(2 if side == "BUY" else 0), confidence=conf,
            spread_points=cur_spread, atr=atr_value, volume=lot, status="PENDING",
            config_snapshot_id=snap_id,
        )

        sl_mult = float(self.cfg_t.get("sl_atr_mult", 1.0))
        tp_mult = float(self.cfg_t.get("tp_atr_mult", 2.0))
        if self.recovery.consecutive_losses > 0:
            sl_rec = float(self.cfg_r.get("sl_atr_mult_recovery", 0))
            tp_rec = float(self.cfg_r.get("tp_atr_mult_recovery", 0))
            if sl_rec > 0:
                sl_mult = sl_rec
            if tp_rec > 0:
                tp_mult = tp_rec

        result = open_market_order(
            symbol=self.symbol, side=side, volume=lot, atr_value=atr_value,
            magic=self.magic, comment=f"HY#{sid}.{step}",
            spread_tracker=self.spread_tracker,
            sl_atr_mult=sl_mult, tp_atr_mult=tp_mult,
        )
        if not result.ok:
            self.db.update_decision(decision_id, status="ERROR",
                                    notes=f"retcode={result.retcode} {result.comment}")
            log.warning("❌ เปิดออเดอร์ไม่ได้: %s %.2f lot (retcode=%d %s)",
                        side, lot, result.retcode, result.comment)
            return

        self.db.update_decision(decision_id, ticket=result.ticket, entry_price=result.price,
                                sl=result.sl, tp=result.tp, status="OPEN")
        self.recovery.last_side = side
        self._save_recovery()
        role_emoji = "🆕" if role == "PRIMARY" else "♻️"
        ppp = (spec.trade_tick_value / spec.trade_tick_size) if spec.trade_tick_size > 0 else 0.0
        risk_usd = abs(result.price - result.sl) * lot * ppp
        reward_usd = abs(result.tp - result.price) * lot * ppp
        log.info("%s เปิดไม้ #%d (series %d, ไม้ที่ %d) | %s %.2f lot @%.3f | "
                 "SL=%.3f (เสี่ยง $%.2f) TP=%.3f (เป้า $%.2f) | AI %.1f%%",
                 role_emoji, result.ticket, sid, step, side, lot, result.price,
                 result.sl, risk_usd, result.tp, reward_usd, conf*100)

    # ============================================================ closed-trade processing
    def _update_closed_trades(self) -> None:
        """
        Robust detection ของไม้ที่ปิดไป:
          1. ดึง decisions ที่ status=OPEN จาก DB (ของ symbol นี้)
          2. เช็คว่ายังอยู่ใน live positions หรือไม่ — ถ้าไม่ = ปิดไปแล้ว
          3. หา deals ของ position นั้นด้วย mt5.history_deals_get(position=ticket)
             (ไม่ filter magic เพราะ broker SL close deal อาจ magic=0)
          4. คำนวณ pnl + update DB + update RecoveryState
        """
        opens = self.db.open_decisions()
        if not opens:
            return
        # filter เฉพาะ symbol ปัจจุบัน
        opens = [r for r in opens if r["symbol"] == self.symbol]
        open_by_ticket = {int(row["ticket"]): row for row in opens if row["ticket"]}
        if not open_by_ticket:
            return

        try:
            live_pos = MT5Connector.positions(symbol=self.symbol, magic=self.magic)
        except ConnectionError as e:
            log.warning("⚠️ MT5 disconnect ใน _update_closed_trades → ข้าม (ไม่นับว่าปิด): %s", e)
            return
        live_tickets = {p.ticket for p in live_pos}
        closed_tickets = [t for t in open_by_ticket if t not in live_tickets]
        if not closed_tickets:
            return

        log.info("🔍 ตรวจไม้ที่เพิ่งปิด: เปิดใน DB %d ไม้ / live %d ไม้ → ต้องเก็บผล %d ไม้",
                 len(open_by_ticket), len(live_tickets), len(closed_tickets))

        for ticket in closed_tickets:
            if ticket in self.recovery.processed_tickets:
                # อาจติดค้างจาก rounce ก่อนหน้า — ไม่ต้องประมวลซ้ำ
                continue

            # Query deals ของ position นี้โดยตรง (ไม่ filter magic)
            try:
                ds_raw = mt5.history_deals_get(position=ticket)
                ds = list(ds_raw) if ds_raw else []
            except Exception as e:
                log.warning("history_deals_get(position=%d) failed: %s", ticket, e)
                ds = []

            if not ds:
                # ลอง fallback ด้วย time-range search
                date_to = datetime.now(timezone.utc) + timedelta(minutes=5)
                date_from = datetime.now(timezone.utc) - timedelta(hours=48)
                all_deals = MT5Connector.history_deals(date_from, date_to)
                ds = [d for d in all_deals if d.position_id == ticket]

            if not ds:
                log.warning("⚠️  ไม้ #%d ปิดไปแล้วแต่ยังหา deal ไม่เจอ — จะลองใหม่รอบหน้า", ticket)
                # ไม่ mark processed เพื่อให้ retry รอบถัดไป
                continue

            row = open_by_ticket[ticket]
            pnl = float(sum(d.profit + d.swap + d.commission for d in ds))
            close_deal = max(ds, key=lambda d: d.time)
            status = "WIN" if pnl > 0 else "LOSS"
            self.db.update_decision(
                row["id"], status=status, pnl=pnl,
                close_price=float(close_deal.price),
                closed_at_utc=datetime.fromtimestamp(close_deal.time, tz=timezone.utc).isoformat(),
            )
            emoji = "💚" if pnl > 0 else "💔"
            sign = "+" if pnl > 0 else ""
            log.info("%s ไม้ #%d ปิดแล้ว → %s %s$%.2f",
                     emoji, ticket, status, sign, pnl)
            self.recovery.processed_tickets.add(ticket)

            if pnl > 0:
                self.recovery.cumulative_loss_usd -= pnl
                # 🆕 reset direction-flip counter on WIN
                self.recovery.consec_same_dir_losses = 0
                self.recovery.last_loss_side = None
                # ปลด block ทันที (WIN = market อาจเปลี่ยน regime)
                self.recovery.blocked_side = None
                self.recovery.block_side_until_ts = 0.0
                if self.recovery.cumulative_loss_usd <= 0:
                    self._on_recovery_complete()
                else:
                    log.info(
                        "   ↳ ชนะแต่ยังไม่ครบหนี้ — เหลือ $%.2f → ไม้กู้ถัดไปจะลองปิดทั้งก้อนอีก",
                        self.recovery.cumulative_loss_usd)
                # หักหนี้ global ด้วย (ไม่ว่าจะอยู่ใน series หรือ gentle slope)
                if self.recovery.global_debt_usd > 0:
                    before = self.recovery.global_debt_usd
                    self.recovery.global_debt_usd = max(0.0, self.recovery.global_debt_usd - pnl)
                    if self.recovery.global_debt_usd <= 0:
                        log.info("🎊 Global debt cleared! หนี้สะสม $%.2f คืนหมดแล้ว", before)
                    else:
                        log.info("   ↳ global_debt เหลือ $%.2f (หักได้ $%.2f)",
                                 self.recovery.global_debt_usd, pnl)
            else:
                self.recovery.cumulative_loss_usd += abs(pnl)
                self.recovery.cumulative_losing_volume += float(row["volume"] or 0.0)
                self.recovery.consecutive_losses += 1
                self.recovery.last_loss_ts = time.time()
                # 🆕 update direction-flip counter
                trade_side = "BUY" if row["prediction"] == 2 else "SELL"
                if self.recovery.last_loss_side == trade_side:
                    self.recovery.consec_same_dir_losses += 1
                else:
                    self.recovery.consec_same_dir_losses = 1
                    self.recovery.last_loss_side = trade_side
                log.info("   ↳ เสียติด %d ไม้แล้ว ขาดทุนสะสม $%.2f (lot รวม %.3f) → ไม้ถัดไป lot จะโต",
                         self.recovery.consecutive_losses,
                         self.recovery.cumulative_loss_usd,
                         self.recovery.cumulative_losing_volume)
                # 🆕 🅒 Per-Series Loss Cap — ปิด series ก่อนที่ max_steps จะระเบิด
                cap_cfg = self.cfg_rf.get("series_loss_cap", {})
                if cap_cfg.get("enabled", False):  # default False — ต้องเปิดใน config ถึงจะทำงาน
                    cap_pct = float(cap_cfg.get("max_loss_pct_of_balance", 8.0))
                    bal = self._get_balance()
                    if bal > 0:
                        loss_pct = self.recovery.cumulative_loss_usd / bal * 100.0
                        if loss_pct >= cap_pct:
                            log.error("🚨 Series Loss Cap! ขาดทุน $%.2f = %.1f%% ของ balance $%.0f (เกิน %.1f%%) → ปิด series + halt",
                                      self.recovery.cumulative_loss_usd, loss_pct, bal, cap_pct)
                            self._on_series_loss_cap()
                            self._save_recovery()
                            continue
                max_steps = int(self.cfg_r.get("max_steps", 4))
                if max_steps > 0 and self.recovery.consecutive_losses >= max_steps:
                    self._on_max_steps_exceeded()

            self._save_recovery()

        if len(self.recovery.processed_tickets) > 5000:
            self.recovery.processed_tickets = set(list(self.recovery.processed_tickets)[-2000:])

    def _on_recovery_complete(self) -> None:
        sid = self.recovery.series_id
        net_recovered = -self.recovery.cumulative_loss_usd  # positive = profit on top
        log.info("🎉 RECOVER สำเร็จ! series #%s จบ (เสียก่อนหน้า %d ไม้) กำไรสุทธิ +$%.2f → กลับ lot ปกติ",
                 sid, self.recovery.consecutive_losses, net_recovered)
        if sid is not None:
            self.db.close_series(sid, status="CLOSED_TP", final_pnl=net_recovered,
                                 total_volume=0.0, avg_entry_price=0.0,
                                 notes=f"steps={self.recovery.consecutive_losses + 1}")
            self.webhook.series_closed({
                "series_id": sid, "symbol": self.symbol,
                "consecutive_losses": self.recovery.consecutive_losses,
                "final_pnl": net_recovered, "reason": "RECOVERED",
            })
        self.recovery.cumulative_loss_usd = 0.0
        self.recovery.cumulative_losing_volume = 0.0
        self.recovery.consecutive_losses = 0
        self.recovery.series_id = None
        self.recovery.last_side = None
        self.recovery.pause_resume_count = 0
        # หนี้ global (carry-over จาก series ก่อน) ยังคงอยู่ → จะค่อยๆ หักผ่าน gentle slope

    def _on_max_steps_exceeded(self) -> None:
        sid = self.recovery.series_id
        halt_min = float(self.cfg_r.get("halt_after_max_steps_minutes", 10))
        remaining_debt = self.recovery.cumulative_loss_usd
        max_steps = int(self.cfg_r.get("max_steps", 4))
        if halt_min > 0:
            log.error("🛑 ครบ max_steps (%d)! เสียติด %d ไม้ ขาดทุน $%.2f → หยุดเทรด %d นาที",
                      max_steps, self.recovery.consecutive_losses, remaining_debt, int(halt_min))
        else:
            log.warning(
                "🛑 ครบ %d ไม้ — ไม่บังคับกู้ต่อ (เอาเท่าที่ไหว) | หนี้ค้าง $%.2f → global_debt | "
                "เทรดต่อปกติ — ช่วงแพ้ยาวจะ block จาก analyze_slots ทีหลัง",
                max_steps, remaining_debt)
        if sid is not None:
            self.db.close_series(sid, status="CLOSED_MAX_STEPS",
                                 final_pnl=-remaining_debt,
                                 total_volume=0.0, avg_entry_price=0.0,
                                 notes=f"max_steps={max_steps} after {self.recovery.consecutive_losses} losses")
            self.webhook.series_closed({
                "series_id": sid, "symbol": self.symbol,
                "consecutive_losses": self.recovery.consecutive_losses,
                "final_pnl": -remaining_debt,
                "reason": "MAX_STEPS",
            })
        # ย้ายหนี้ไปยัง global_debt — กู้ทีละน้อยผ่าน gentle slope ไม่ไล่ martingale
        self.recovery.global_debt_usd += remaining_debt
        log.warning("💳 หนี้ carry-over: $%.2f → global_debt รวม $%.2f (gentle slope ~%d ไม้)",
                    remaining_debt, self.recovery.global_debt_usd,
                    int(self.cfg_r.get("debt_recovery_target_trades", 50)))
        if halt_min > 0:
            self.recovery.halted_until_ts = time.time() + halt_min * 60
        self.recovery.cumulative_loss_usd = 0.0
        self.recovery.cumulative_losing_volume = 0.0
        self.recovery.consecutive_losses = 0
        self.recovery.series_id = None
        self.recovery.last_side = None
        self.recovery.pause_resume_count = 0

    # 🆕 helper: ดึง balance
    def _get_balance(self) -> float:
        try:
            now = time.time()
            if now - self._cached_balance_ts > 30 or self._cached_balance <= 0:
                acc = mt5.account_info()
                if acc and acc.balance > 0:
                    self._cached_balance = float(acc.balance)
                    self._cached_balance_ts = now
            return float(self._cached_balance)
        except Exception:
            return 0.0

    # 🆕 🅒 Series Loss Cap — Pause & Resume mode (default) หรือ Close mode
    def _on_series_loss_cap(self) -> None:
        sid = self.recovery.series_id
        cap_cfg = self.cfg_rf.get("series_loss_cap", {})
        halt_min = float(cap_cfg.get("halt_minutes", 30))
        max_resumes = int(cap_cfg.get("max_pause_resumes", 2))
        action = str(cap_cfg.get("action", "pause_resume")).lower()

        # === PAUSE & RESUME mode: ให้โอกาสกู้คืน ===
        if action == "pause_resume" and self.recovery.pause_resume_count < max_resumes:
            self.recovery.pause_resume_count += 1
            self.recovery.halted_until_ts = time.time() + halt_min * 60
            log.warning("⏸️ Series PAUSED (#%d) — ขาดทุน $%.2f | resume %d/%d → halt %d นาที แล้วกลับมา recover ต่อ",
                        sid or 0, self.recovery.cumulative_loss_usd,
                        self.recovery.pause_resume_count, max_resumes, int(halt_min))
            log.warning("   ↳ series ยังมีชีวิต ไม่ปิดทิ้ง — รอจังหวะดีแล้วลุยต่อ 💪")
            # อย่า reset cumulative_loss / consecutive_losses → carry over เพื่อ recover ต่อ
            return

        # === CLOSE mode (หรือ pause-resume หมดโอกาสแล้ว) ===
        reason_label = "LOSS_CAP_EXHAUSTED" if action == "pause_resume" else "LOSS_CAP"
        remaining_debt = self.recovery.cumulative_loss_usd
        if sid is not None:
            self.db.close_series(sid, status=f"CLOSED_{reason_label}",
                                 final_pnl=-remaining_debt,
                                 total_volume=0.0, avg_entry_price=0.0,
                                 notes=f"loss cap final close after {self.recovery.pause_resume_count} pauses")
            self.webhook.series_closed({
                "series_id": sid, "symbol": self.symbol,
                "consecutive_losses": self.recovery.consecutive_losses,
                "final_pnl": -remaining_debt,
                "reason": reason_label,
            })
        # ย้ายหนี้ไปยัง global_debt แทนที่จะลืม
        self.recovery.global_debt_usd += remaining_debt
        log.warning("💳 หนี้ carry-over (loss cap): $%.2f → global_debt รวม $%.2f",
                    remaining_debt, self.recovery.global_debt_usd)
        self.recovery.halted_until_ts = time.time() + halt_min * 60
        self.recovery.cumulative_loss_usd = 0.0
        self.recovery.cumulative_losing_volume = 0.0
        self.recovery.consecutive_losses = 0
        self.recovery.series_id = None
        self.recovery.last_side = None
        self.recovery.pause_resume_count = 0
        log.error("🛑 Series ปิดถาวรแล้ว (ใช้โอกาส resume หมด) → halt %d นาที", int(halt_min))

    # ============================================================ Friday / weekend close
    def _friday_close_status(self) -> dict:
        """คืน dict ที่บอกว่าตอนนี้อยู่ใน zone ไหนของ Friday close
        Returns:
            zone: "normal" | "no_new_entry" | "force_close" | "weekend"
            minutes_to_close: นาทีที่เหลือก่อน market_close (ลบ = ผ่านไปแล้ว)
        """
        cfg_wc = Config.section("weekend_close")
        if not cfg_wc.get("enabled", False):
            return {"zone": "normal", "minutes_to_close": 9999}

        now_utc = datetime.now(timezone.utc)
        weekday = now_utc.weekday()  # Mon=0 … Fri=4, Sat=5, Sun=6

        close_hour = int(cfg_wc.get("market_close_hour_utc", 21))
        close_min = int(cfg_wc.get("market_close_minute_utc", 0))
        no_entry_before = int(cfg_wc.get("no_new_entry_minutes_before_close", 60))
        force_close_before = int(cfg_wc.get("force_close_minutes_before_close", 15))
        reopen_hour_mon = int(cfg_wc.get("market_reopen_hour_utc_monday", 22))

        # ===== วันเสาร์หรือวันอาทิตย์ก่อนเวลา reopen Monday =====
        if weekday == 5:  # Saturday
            return {"zone": "weekend", "minutes_to_close": -9999}
        if weekday == 6:  # Sunday ก่อน reopen
            if now_utc.hour < reopen_hour_mon:
                return {"zone": "weekend", "minutes_to_close": -9999}
            return {"zone": "normal", "minutes_to_close": 9999}

        # ===== วันศุกร์ =====
        if weekday == 4:  # Friday
            close_dt = now_utc.replace(hour=close_hour, minute=close_min, second=0, microsecond=0)
            minutes_to_close = (close_dt - now_utc).total_seconds() / 60.0

            if minutes_to_close <= 0:
                # ผ่านเวลาปิดแล้ว — ถือว่า weekend
                return {"zone": "weekend", "minutes_to_close": minutes_to_close}
            if minutes_to_close <= force_close_before:
                return {"zone": "force_close", "minutes_to_close": minutes_to_close}
            if minutes_to_close <= no_entry_before:
                return {"zone": "no_new_entry", "minutes_to_close": minutes_to_close}

        return {"zone": "normal", "minutes_to_close": 9999}

    def _maybe_close_friday_positions(self) -> bool:
        """ปิด position ทั้งหมดเมื่ออยู่ใน force_close / weekend zone
        คืน True ถ้าปิดไปแล้ว (caller ควร return ทันที)
        """
        status = self._friday_close_status()
        zone = status["zone"]

        if zone not in ("force_close", "weekend"):
            return False

        live = MT5Connector.positions(symbol=self.symbol, magic=self.magic)
        if not live:
            return False

        mins = status["minutes_to_close"]
        if zone == "force_close":
            log.warning("⏰ Friday Force-Close: เหลือ %.1f นาทีก่อนตลาดปิด → ปิด %d position ทันที",
                        mins, len(live))
        else:
            log.warning("⏰ Weekend zone: ตลาดปิดแล้ว → ปิด %d position ที่ค้างอยู่", len(live))

        for p in live:
            ok = close_position(p.ticket)
            side_txt = "BUY" if p.type == 0 else "SELL"
            if ok:
                log.info("✅ Friday-close ปิดสำเร็จ: #%d %s %.2f lot @entry %.3f | PnL ≈ $%.2f",
                         p.ticket, side_txt, p.volume, p.price_open, p.profit)
            else:
                log.error("❌ Friday-close ปิดไม่สำเร็จ: #%d %s — จะลองใหม่รอบถัดไป",
                          p.ticket, side_txt)
        return True

    # ============================================================ smart trailing
    def _manage_trailing(self) -> None:
        if not self.cfg_st.get("enabled", True):
            return
        # ระหว่าง recovery → ปล่อยให้ TP จริงทำงาน ไม่ trail (กัน SL ปิดก่อนถึง TP)
        if (self.cfg_st.get("disable_during_recovery", True)
                and self.recovery.consecutive_losses > 0):
            return
        live = MT5Connector.positions(symbol=self.symbol, magic=self.magic)
        if not live:
            return
        spec = MT5Connector.get_symbol_spec(self.symbol)
        tick = MT5Connector.get_tick(self.symbol)
        if tick is None:
            return

        try:
            rates = MT5Connector.copy_rates(self.symbol, self.timeframe, 100)
            df = build_features(rates)
            closed_idx = int(self.cfg_v.get("use_only_closed_candle_index", -2))
            atr_value = float(df.iloc[closed_idx]["atr"])
        except Exception:
            return
        if not np.isfinite(atr_value) or atr_value <= 0:
            return

        be_trigger = float(self.cfg_st.get("be_trigger_atr", 0.4)) * atr_value
        be_buf_pips = float(self.cfg_st.get("be_buffer_pips", 2.0))
        pip_factor = 10 if spec.digits in (3, 5) else 1
        be_buf = be_buf_pips * 10 * pip_factor * spec.point
        com_offset = commission_price_offset(spec)
        be_lock = be_buf + com_offset

        # 🆕 Trail mode:
        #   "be_only"    → ขยับ SL ไปที่ entry+buffer ครั้งเดียว แล้วปล่อยวิ่งไป TP (RECOMMENDED)
        #   "percentage" → trail ตาม X% ของกำไรที่วิ่งไป (lock (100-X)%) — เก็บกำไรเร็วแต่ตัด upside
        #   "atr"        → trail แบบ fixed distance × ATR (legacy)
        #   "off"        → ไม่ทำ trailing เลย (ให้ TP เดิมทำงาน)
        trail_mode = str(self.cfg_st.get("trail_mode", "be_only")).lower()
        if trail_mode == "off":
            return

        trail_pct = max(0.05, min(0.95, float(self.cfg_st.get("trail_pct_of_excursion", 0.30))))
        trail_dist_atr_val = float(self.cfg_st.get("trail_distance_atr", 0.3)) * atr_value
        trail_step = float(self.cfg_st.get("trail_step_atr", 0.1)) * atr_value

        for p in live:
            entry = float(p.price_open)
            is_buy = (p.type == 0)
            cur = float(tick.bid if is_buy else tick.ask)
            profit_price = (cur - entry) if is_buy else (entry - cur)
            if profit_price < be_trigger:
                continue

            # คำนวณ sl_target ตาม mode
            if trail_mode == "be_only":
                # BE only: SL = entry + buffer (กันขาดทุน) — ไม่ trail ตามต่อ
                sl_target = (entry + be_lock) if is_buy else (entry - be_lock)
            elif trail_mode == "percentage":
                trail_dist = max(profit_price * trail_pct, 0.10 * atr_value)
                sl_target = (max(entry + be_lock, cur - trail_dist) if is_buy
                             else min(entry - be_lock, cur + trail_dist))
            else:  # "atr"
                sl_target = (max(entry + be_lock, cur - trail_dist_atr_val) if is_buy
                             else min(entry - be_lock, cur + trail_dist_atr_val))
            sl_target = spec.normalize_price(sl_target)

            cur_sl = float(p.sl) if p.sl else 0.0

            # be_only: ขยับครั้งเดียวแล้วหยุด (ถ้า SL อยู่ที่ entry หรือดีกว่าแล้ว → skip)
            if trail_mode == "be_only":
                if is_buy and cur_sl >= entry - 1e-9:
                    continue
                if (not is_buy) and 0 < cur_sl <= entry + 1e-9:
                    continue
            else:
                if is_buy and cur_sl > 0 and sl_target <= cur_sl + trail_step:
                    continue
                if (not is_buy) and cur_sl > 0 and sl_target >= cur_sl - trail_step:
                    continue

            if modify_position_sl(p.ticket, new_sl=sl_target, new_tp=p.tp):
                tag = {
                    "be_only": "BE locked",
                    "percentage": "pct lock %.0f%%" % ((1.0 - trail_pct) * 100),
                    "atr": "atr trail",
                }.get(trail_mode, trail_mode)
                log.info("📈 เลื่อน SL ไม้ #%d [%s] | กำไรวิ่ง %.3f → SL: %.3f → %.3f",
                         p.ticket, tag, profit_price, cur_sl, sl_target)

    # ============================================================ global equity stop
    def _maybe_global_equity_stop(self) -> None:
        pct = float(self.cfg_r.get("global_equity_stop_pct", 7.0))
        if pct <= 0 or self.recovery.cumulative_loss_usd <= 0:
            return
        acc = mt5.account_info()
        if acc is None or acc.balance <= 0:
            return
        max_loss = float(acc.balance) * pct / 100.0
        try:
            live = MT5Connector.positions(symbol=self.symbol, magic=self.magic)
        except ConnectionError:
            return  # ไม่รู้ floating → ข้าม equity stop รอบนี้
        floating = sum(float(p.profit) + float(p.swap) for p in live)
        effective = self.recovery.cumulative_loss_usd + max(0.0, -floating)
        if effective < max_loss:
            return
        log.error("🚨 EQUITY STOP! ขาดทุนรวม $%.2f ≥ เพดาน $%.2f (%.1f%% ของ balance) → ปิดทุกไม้ + หยุดเทรด",
                  effective, max_loss, pct)
        for p in live:
            close_position(p.ticket)
        # ไม่รีเซ็ต cumulative_loss — ให้ _update_closed_trades บันทึก PnL จริงทีละไม้
        # (รีเซ็ตก่อนหน้านี้ทำให้หนี้ $80 หาย เหลือแค่ -$162 ใน state)
        sid = self.recovery.series_id
        halt_min = float(self.cfg_r.get("halt_after_max_steps_minutes", 10))
        if sid is not None:
            self.db.close_series(sid, status="CLOSED_EQUITY_STOP",
                                 final_pnl=-effective,
                                 total_volume=0.0, avg_entry_price=0.0,
                                 notes=f"equity stop effective=${effective:.2f}")
            self.webhook.series_closed({
                "series_id": sid, "symbol": self.symbol,
                "consecutive_losses": self.recovery.consecutive_losses,
                "final_pnl": -effective,
                "reason": "EQUITY_STOP",
            })
        self.recovery.halted_until_ts = time.time() + halt_min * 60
        self.recovery.series_id = None
        self.recovery.last_side = None
        self._save_recovery()

    # ============================================================ self-retrain
    def _maybe_retrain(self) -> None:
        interval_min = float(self.cfg_a.get("retrain_check_interval_min", 240))
        if time.time() - self._last_retrain_check < interval_min * 60:
            return
        self._last_retrain_check = time.time()
        live = MT5Connector.positions(symbol=self.symbol, magic=self.magic)
        if live or self.recovery.consecutive_losses > 0:
            return
        settled = self.db.count_settled()
        if (settled - self._trades_at_last_train) < int(self.cfg_a.get("retrain_min_new_trades", 500)):
            return
        log.info("[RETRAIN] %d new closed trades -> retrain",
                 settled - self._trades_at_last_train)
        try:
            from .model_trainer import train_from_mt5
            train_from_mt5()
            MT5Connector.initialise()
            self._load_model()
            self._trades_at_last_train = settled
        except Exception as e:
            log.exception("retrain failed: %s", e)
            MT5Connector.initialise()


def main() -> None:
    HyperBot().run()


if __name__ == "__main__":
    main()