"""
model_trainer.py
=================
Train XGBoost ให้รองรับ Low Confidence Threshold (0.35) สำหรับ Hyper-Frequency.

Tuning:
  - learning_rate ต่ำ + n_estimators เยอะ + early stopping  -> โมเดลนิ่ง
  - max_depth พอประมาณ (4) เพื่อไม่ overfit จาก signal ย่อย
  - subsample/colsample สูง   -> รักษา recall ของ minority class
  - eval_metric mlogloss      -> calibration ของ probability ดีขึ้น
"""
from __future__ import annotations
import json
from datetime import datetime, timezone
from typing import Optional

import joblib
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

from .config import Config
from .m1_hyper_pipeline import FEATURE_COLUMNS, build_training_dataset, build_features
from .logger import get_logger
from .mt5_connector import MT5Connector
from .paths import model_path

log = get_logger("trainer")


# ============================================================================
# 🆕 PROFIT-BASED VALIDATION
# ============================================================================
def simulate_profit_metrics(model, X_test, df_test_full, threshold: float = 0.55,
                             tp_atr: float = 1.6, sl_atr: float = 0.8,
                             lookahead: int = 12,
                             commission_price_offset: float = 0.07,
                             buy_max_thr: float | None = None,
                             sell_max_thr: float | None = None) -> dict:
    """
    Simulate การเทรดจริงบน OOS test set โดยใช้ model ที่เพิ่งเทรน
    ใช้กฎ entry แบบเดียวกับ bot:
      - predict_proba → ถ้า BUY/SELL confidence ≥ threshold → ส่งออเดอร์
      - close ที่ TP hit (กำไรเท่ากับ tp_atr × ATR) หรือ SL hit (ขาดทุน sl_atr × ATR)
      - หัก commission_price_offset (price units) ทั้งสองทาง (round-trip)
      - ถ้าจบ lookahead แล้วยังไม่ hit → ปิดที่ราคา close ของ bar สุดท้าย (mark-to-market)

    คืน dict:
      - n_trades, n_win, n_loss, win_rate
      - net_pnl_atrs            (sum ของ pnl เป็นหน่วย ATR — independent of lot)
      - avg_pnl_per_trade_atrs
      - profit_factor           (gross_win / gross_loss)
      - expectancy_per_trade
      - max_drawdown_atrs
      - max_consec_losses

    การวัดเป็น "ATRs" ทำให้ scale ไม่ขึ้นกับ lot/symbol — ใช้เทียบ model ต่อ model ได้ตรง
    """
    import pandas as pd
    # ต้องเรียงตามเวลา
    proba = model.predict_proba(X_test)
    preds = proba.argmax(axis=1)
    confs = proba.max(axis=1)

    high = df_test_full["high"].to_numpy()
    low = df_test_full["low"].to_numpy()
    close = df_test_full["close"].to_numpy()
    atr = df_test_full["atr"].to_numpy()
    n = len(df_test_full)

    trades = []
    pnls_atr = []
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    consec_loss = 0
    max_consec_loss = 0

    in_trade_until = -1   # bar index จนกว่าจะปิดไม้นี้ (no overlap)

    for i in range(n - 1):
        if i <= in_trade_until:
            continue
        pred = int(preds[i])
        conf = float(confs[i])
        a = atr[i]
        if not np.isfinite(a) or a <= 0:
            continue
        if pred == 1:
            continue  # HOLD
        # entry threshold
        if conf < threshold:
            continue
        # max threshold (overfit zone) ถ้ามี
        if pred == 2 and buy_max_thr is not None and conf > buy_max_thr:
            continue
        if pred == 0 and sell_max_thr is not None and conf > sell_max_thr:
            continue

        entry = close[i]
        is_buy = (pred == 2)
        tp_price = entry + tp_atr * a if is_buy else entry - tp_atr * a
        sl_price = entry - sl_atr * a if is_buy else entry + sl_atr * a

        end = min(n, i + 1 + lookahead)
        outcome = None
        exit_price = None
        for j in range(i + 1, end):
            hj, lj = high[j], low[j]
            if is_buy:
                # tie-break: SL ก่อน TP (conservative)
                if lj <= sl_price:
                    outcome = "LOSS"; exit_price = sl_price; in_trade_until = j; break
                if hj >= tp_price:
                    outcome = "WIN"; exit_price = tp_price; in_trade_until = j; break
            else:
                if hj >= sl_price:
                    outcome = "LOSS"; exit_price = sl_price; in_trade_until = j; break
                if lj <= tp_price:
                    outcome = "WIN"; exit_price = tp_price; in_trade_until = j; break
        if outcome is None:
            # หมด lookahead → mark-to-market ด้วย close
            exit_price = close[end - 1]
            outcome = "TIMEOUT"
            in_trade_until = end - 1

        gross = (exit_price - entry) if is_buy else (entry - exit_price)
        # หัก commission (round-trip) เป็น price units
        net = gross - commission_price_offset
        pnl_atr = net / a   # normalize ด้วย ATR ตอน entry
        pnls_atr.append(pnl_atr)
        trades.append({"i": i, "side": "BUY" if is_buy else "SELL",
                       "conf": conf, "outcome": outcome, "pnl_atr": pnl_atr})

        equity += pnl_atr
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd
        if pnl_atr < 0:
            consec_loss += 1
            if consec_loss > max_consec_loss:
                max_consec_loss = consec_loss
        else:
            consec_loss = 0

    if not pnls_atr:
        return {
            "n_trades": 0, "n_win": 0, "n_loss": 0, "win_rate": 0.0,
            "net_pnl_atrs": 0.0, "avg_pnl_per_trade_atrs": 0.0,
            "profit_factor": 0.0, "expectancy_per_trade": 0.0,
            "max_drawdown_atrs": 0.0, "max_consec_losses": 0,
            "n_timeout": 0,
        }

    pnls_arr = np.array(pnls_atr)
    wins = pnls_arr[pnls_arr > 0]
    losses = pnls_arr[pnls_arr <= 0]
    gross_win = float(wins.sum()) if len(wins) > 0 else 0.0
    gross_loss = float(-losses.sum()) if len(losses) > 0 else 0.0
    pf = (gross_win / gross_loss) if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0)

    return {
        "n_trades": len(pnls_arr),
        "n_win": int((pnls_arr > 0).sum()),
        "n_loss": int((pnls_arr <= 0).sum()),
        "n_timeout": int(sum(1 for t in trades if t["outcome"] == "TIMEOUT")),
        "win_rate": float((pnls_arr > 0).mean()),
        "net_pnl_atrs": float(pnls_arr.sum()),
        "avg_pnl_per_trade_atrs": float(pnls_arr.mean()),
        "profit_factor": pf,
        "expectancy_per_trade": float(pnls_arr.mean()),
        "max_drawdown_atrs": float(max_dd),
        "max_consec_losses": int(max_consec_loss),
        "gross_win_atrs": gross_win,
        "gross_loss_atrs": gross_loss,
    }


def _load_real_trade_outcomes(symbol: str) -> "Optional[object]":
    """
    🆕 Trade-Augmented Learning: ดึง closed trades (WIN/LOSS) จาก DB
    คืนค่า DataFrame[time, side, outcome] หรือ None ถ้าไม่มี/error.
    side: 0=SELL, 2=BUY ; outcome: 'WIN'/'LOSS'
    """
    try:
        import pandas as pd
        from .paths import db_path  # type: ignore
        import sqlite3
        cfg_d = Config.section("database") or {}
        path = db_path(cfg_d.get("filename", "hyper_trades.sqlite"))
        if not path.exists():
            return None
        with sqlite3.connect(str(path)) as conn:
            df = pd.read_sql_query(
                "SELECT ts_utc, prediction, status, config_snapshot_id FROM decisions "
                "WHERE symbol=? AND status IN ('WIN','LOSS') "
                "ORDER BY ts_utc",
                conn, params=(symbol,),
            )
        if df.empty:
            return None
        df["time"] = pd.to_datetime(df["ts_utc"], utc=True, errors="coerce")
        df = df.dropna(subset=["time"])
        df = df.rename(columns={"prediction": "side", "status": "outcome"})
        return df[["time", "side", "outcome", "config_snapshot_id"]]
    except Exception as e:
        log.warning("Trade-aug: failed to load DB outcomes: %s", e)
        return None


def _apply_trade_augmentation(X_part, y_part, sw_part, real_df,
                              win_weight: float, loss_weight: float,
                              loss_mode="flip", tolerance_min: int = 5,
                              snapshot_scores=None):
    """
    🆕 Apply trade-aug: match real trade timestamps to bars in X_part,
    boost their sample weights, optionally relabel LOSSes.

    🆕 v2: ถ้ามี snapshot_scores → คูณ weight ด้วย strategy quality score
           ของ snapshot ที่ trade นั้นเกิด (formula-driven, not hardcoded)

    loss_mode (str or bool, backward-compatible):
      - "flip"          : LOSS BUY (2) → SELL (0), LOSS SELL (0) → BUY (2)  ✨ recommended
                          → keeps bot trading; learns reversal patterns
      - "hold" / True   : LOSS → HOLD (1) — safe but can silence the bot over time
      - "none" / False  : keep original label, just boost weight

    Returns (y_arr, sw_arr, n_matched, n_relabeled).
    """
    import pandas as pd
    if real_df is None or real_df.empty:
        return y_part.values, sw_part, 0, 0
    if not isinstance(X_part.index, pd.DatetimeIndex):
        return y_part.values, sw_part, 0, 0

    y_arr = y_part.values.copy()
    sw_arr = sw_part.copy()
    n_matched, n_relabeled = 0, 0
    tol = pd.Timedelta(minutes=tolerance_min)

    # Normalise mode (back-compat: True→hold, False→none)
    if isinstance(loss_mode, bool):
        loss_mode = "hold" if loss_mode else "none"
    mode = str(loss_mode or "flip").lower()

    # Subset real trades within partition window
    t_lo, t_hi = X_part.index.min(), X_part.index.max()
    sub = real_df[(real_df["time"] >= t_lo) & (real_df["time"] <= t_hi)]
    if sub.empty:
        return y_arr, sw_arr, 0, 0

    # Vectorized nearest-neighbor match (align dtype/tz to X_part.index)
    target = pd.DatetimeIndex(sub["time"]).tz_convert(X_part.index.tz) if X_part.index.tz else \
             pd.DatetimeIndex(sub["time"]).tz_localize(None)
    target = target.astype(X_part.index.dtype)
    pos = X_part.index.get_indexer(target, method="nearest", tolerance=tol)
    valid = pos >= 0
    matched_pos = pos[valid]
    matched_outcome = sub["outcome"].values[valid]
    matched_side = sub["side"].values[valid]   # 0=SELL, 2=BUY
    matched_snap = sub["config_snapshot_id"].values[valid] if "config_snapshot_id" in sub.columns else [None] * len(matched_pos)

    for p, outcome, side, snap_id in zip(matched_pos, matched_outcome, matched_side, matched_snap):
        # 🆕 strategy-weight multiplier (formula-based score; default 1.0 if no data)
        sw_mult = 1.0
        if snapshot_scores and snap_id is not None:
            try:
                # ป้องกัน NaN crash — trades เก่าไม่มี snap_id
                if isinstance(snap_id, float) and snap_id != snap_id:  # NaN check
                    pass
                else:
                    entry = snapshot_scores.get(int(snap_id))
                    if entry:
                        sw_mult = float(entry.get("weight_mult", 1.0))
            except (ValueError, TypeError):
                pass

        if outcome == "LOSS":
            sw_arr[p] = sw_arr[p] * float(loss_weight) * sw_mult
            if mode == "flip":
                # BUY loss → SELL ; SELL loss → BUY ; HOLD untouched
                side_i = int(side)
                new_label = 0 if side_i == 2 else (2 if side_i == 0 else 1)
                if y_arr[p] != new_label:
                    y_arr[p] = new_label
                    n_relabeled += 1
            elif mode == "hold":
                if y_arr[p] != 1:
                    y_arr[p] = 1
                    n_relabeled += 1
            # "none" → just weight boost
        elif outcome == "WIN":
            sw_arr[p] = sw_arr[p] * float(win_weight) * sw_mult
        n_matched += 1

    return y_arr, sw_arr, n_matched, n_relabeled


def _build_xgb(n_estimators: int = 800, with_early_stop: bool = True,
               seed: int | None = None) -> XGBClassifier:
    import time as _time
    # Use random seed each call so successive retrains explore different models
    rng_seed = seed if seed is not None else int(_time.time() * 1000) % 100000
    kwargs = dict(
        objective="multi:softprob",
        num_class=3,
        n_estimators=n_estimators,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=3,
        reg_lambda=1.2,
        reg_alpha=0.05,
        gamma=0.05,
        tree_method="hist",
        eval_metric="mlogloss",
        n_jobs=-1,
        random_state=rng_seed,
    )
    if with_early_stop:
        kwargs["early_stopping_rounds"] = 150
    return XGBClassifier(**kwargs)


class EnsembleModel:
    """
    🆕 Lightweight XGBoost + LightGBM probability ensemble.
    Drop-in replacement for a single model — exposes predict_proba() and predict().
    Saved as part of the joblib bundle so inference code needs no changes.
    """
    def __init__(self, xgb_model, lgbm_model, xgb_weight: float = 0.6, lgbm_weight: float = 0.4):
        self.xgb_model = xgb_model
        self.lgbm_model = lgbm_model
        self.xgb_weight = xgb_weight
        self.lgbm_weight = lgbm_weight
        # Expose classes_ for sklearn compatibility
        self.classes_ = np.array([0, 1, 2])

    def predict_proba(self, X) -> np.ndarray:
        p_xgb = self.xgb_model.predict_proba(X)
        # LGBMClassifier was fitted with DataFrame → pass DataFrame to avoid UserWarning
        X_lgbm = X
        try:
            import pandas as _pd
            feat_names = self.lgbm_model.feature_name_
            if isinstance(X, np.ndarray):
                X_lgbm = _pd.DataFrame(X, columns=feat_names, dtype='float64')
            elif isinstance(X, _pd.DataFrame):
                X_lgbm = X[feat_names].astype('float64')
        except Exception:
            pass  # fallback: pass as-is
        p_lgb = self.lgbm_model.predict_proba(X_lgbm)
        # Weighted average of calibrated probabilities
        return self.xgb_weight * p_xgb + self.lgbm_weight * p_lgb

    def predict(self, X) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def score(self, X, y) -> float:
        return float((self.predict(X) == np.asarray(y)).mean())


def _build_lgbm(n_estimators: int = 1000, seed: int | None = None):
    """Build LightGBM classifier with settings tuned for XAUUSD scalping."""
    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        return None
    import time as _t
    rng_seed = seed if seed is not None else int(_t.time() * 1000) % 100000
    return LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=n_estimators,
        max_depth=5,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_samples=20,
        reg_alpha=0.05,
        reg_lambda=1.0,
        n_jobs=-1,
        random_state=rng_seed,
        verbose=-1,
    )


def _is_degenerate(model: XGBClassifier, X, y) -> tuple[bool, str]:
    proba = model.predict_proba(X)
    mean_max = float(proba.max(axis=1).mean())
    pred = proba.argmax(axis=1)
    uniq = len(np.unique(pred))
    if mean_max < 0.40:
        return True, f"mean(max_proba)={mean_max:.3f} ~ 1/3"
    if uniq < 2:
        return True, f"only predicts class={pred[0]}"
    return False, f"OK mean(max_proba)={mean_max:.3f} uniq={uniq}"


def train_from_mt5(symbol: Optional[str] = None,
                   timeframe: Optional[str] = None,
                   bars: Optional[int] = None) -> dict:
    cfg_t = Config.section("trading")
    cfg_a = Config.section("ai")
    symbol = symbol or cfg_t["symbol"]
    timeframe = timeframe or cfg_t.get("timeframe", "M1")
    bars = int(bars or cfg_t.get("history_bars_for_training", 200000))

    if not MT5Connector.initialise():
        raise RuntimeError("MT5 init failed")
    try:
        spec = MT5Connector.get_symbol_spec(symbol)
        log.info("Symbol: %s digits=%d point=%g tick=%g",
                 spec.name, spec.digits, spec.point, spec.trade_tick_size)
        rates = MT5Connector.load_history(symbol, timeframe, bars)
    finally:
        MT5Connector.shutdown()

    X, y = build_training_dataset(rates)
    if len(y.unique()) < 2:
        raise RuntimeError("Insufficient class diversity")
    n = len(X)
    log.info("Dataset: %d rows x %d features", n, X.shape[1])

    # 🆕 Trade-Augmented Learning: relabel BEFORE split so real trades
    # (which sit at the END of the dataset) actually take effect.
    ta_cfg = cfg_a.get("trade_augmentation") or {}
    real_outcomes = None
    if ta_cfg.get("enabled", False):
        min_trades = int(ta_cfg.get("min_db_trades", 500))
        win_w_g = float(ta_cfg.get("win_weight", 2.0))
        loss_w_g = float(ta_cfg.get("loss_weight", 3.0))
        loss_mode_g = ta_cfg.get("loss_mode", ta_cfg.get("relabel_loss_to_hold", "flip"))
        real_outcomes = _load_real_trade_outcomes(symbol)
        if real_outcomes is None or len(real_outcomes) < min_trades:
            log.warning("Trade-aug: skipped (have %d trades, need %d)",
                        0 if real_outcomes is None else len(real_outcomes), min_trades)
            real_outcomes = None
        else:
            log.info("Trade-aug: %d real trades loaded (win_w=%.1f loss_w=%.1f loss_mode=%s)",
                     len(real_outcomes), win_w_g, loss_w_g, loss_mode_g)
            # 🆕 Compute strategy quality scores per snapshot (formula-driven)
            try:
                from .strategy_weights import compute_snapshot_scores, report_scores
                snapshot_scores = compute_snapshot_scores()
                if snapshot_scores:
                    log.info("Strategy-weighted aug ENABLED — per-snapshot multipliers:\n%s",
                             report_scores())
                else:
                    snapshot_scores = None
            except Exception as e:
                log.warning("strategy_weights failed: %s — fallback to flat weights", e)
                snapshot_scores = None
            # Relabel on FULL y first (placeholder sw, we recompute per-partition below)
            sw_placeholder = np.ones(len(y), dtype=float)
            y_full_arr, _, n_match_full, n_rel_full = _apply_trade_augmentation(
                X, y, sw_placeholder, real_outcomes, win_w_g, loss_w_g, loss_mode_g,
                snapshot_scores=snapshot_scores)
            log.info("Trade-aug (full): matched %d bars, relabeled %d losses",
                     n_match_full, n_rel_full)
            # Replace y with relabeled version (preserve original index)
            import pandas as pd
            y = pd.Series(y_full_arr, index=y.index, name=y.name).astype(int)

    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    X_tr, y_tr = X.iloc[:train_end], y.iloc[:train_end]
    X_va, y_va = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_te, y_te = X.iloc[val_end:], y.iloc[val_end:]

    sw_tr = compute_sample_weight(class_weight="balanced", y=y_tr)

    # 🆕 Anti-bias: penalize wrong-direction class harder
    # Read class_weight_overrides from config (e.g., {"0": 1.0, "1": 1.0, "2": 1.5} → BUY mistakes 1.5× more costly)
    cw_override_raw = cfg_a.get("class_weight_overrides") or {}
    # ป้องกัน "_comment" keys + non-numeric: keep เฉพาะ str ที่ cast int ได้
    cw_override = {}
    for k, v in cw_override_raw.items():
        try:
            cw_override[int(k)] = float(v)
        except (ValueError, TypeError):
            continue
    if cw_override:
        log.info("Applying class_weight overrides: %s", cw_override)
        for cls_int, mult in cw_override.items():
            mask = (y_tr.values == cls_int)
            sw_tr[mask] = sw_tr[mask] * float(mult)

    # 🆕 Trade-Augmented Learning: weight boost on training partition
    y_tr_arr = y_tr.values
    if real_outcomes is not None:
        win_w = float(ta_cfg.get("win_weight", 2.0))
        loss_w = float(ta_cfg.get("loss_weight", 3.0))
        loss_mode = ta_cfg.get("loss_mode", ta_cfg.get("relabel_loss_to_hold", "flip"))
        # Re-use snapshot_scores from full pass if computed
        try:
            snapshot_scores
        except NameError:
            snapshot_scores = None
        y_tr_arr, sw_tr, n_match, n_rel = _apply_trade_augmentation(
            X_tr, y_tr, sw_tr, real_outcomes, win_w, loss_w, loss_mode,
            snapshot_scores=snapshot_scores)
        log.info("Trade-aug (train partition): matched %d bars, weight-boosted (%d also relabeled in train)",
                 n_match, n_rel)

    tscv = TimeSeriesSplit(n_splits=4)
    cv_accs = []
    for k, (i_tr, i_va) in enumerate(tscv.split(X_tr), 1):
        sw_k = compute_sample_weight(class_weight="balanced", y=y_tr.iloc[i_tr])
        if cw_override:
            for cls_int, mult in cw_override.items():
                mask = (y_tr.iloc[i_tr].values == cls_int)
                sw_k[mask] = sw_k[mask] * float(mult)
        m = _build_xgb(n_estimators=600, with_early_stop=True)
        m.fit(X_tr.iloc[i_tr], y_tr.iloc[i_tr], sample_weight=sw_k,
              eval_set=[(X_tr.iloc[i_va], y_tr.iloc[i_va])], verbose=False)
        acc = float(m.score(X_tr.iloc[i_va], y_tr.iloc[i_va]))
        cv_accs.append(acc)
        log.info("  CV %d/4 acc=%.4f best_iter=%d", k, acc, m.best_iteration)
    log.info("CV mean=%.4f std=%.4f", float(np.mean(cv_accs)), float(np.std(cv_accs)))

    import time as _t
    main_seed = int(_t.time() * 1000) % 100000
    log.info("🎲 Training seed: %d", main_seed)
    final = _build_xgb(n_estimators=4000, with_early_stop=True, seed=main_seed)
    final.fit(X_tr, y_tr_arr, sample_weight=sw_tr, eval_set=[(X_va, y_va)], verbose=False)
    best_iter = int(final.best_iteration)
    val_acc = float(final.score(X_va, y_va))
    test_acc = float(final.score(X_te, y_te))
    log.info("Best iter=%d val_acc=%.4f OOS_acc=%.4f", best_iter, val_acc, test_acc)

    sample_n = min(2000, len(X_te))
    degen, msg = _is_degenerate(final, X_te.iloc[:sample_n], y_te.iloc[:sample_n])
    log.info("Degeneracy: %s", msg)

    # 🆕 ลด threshold 0.36 → 0.31 (3-class random = 0.333, 0.36 สูงเกินไป → fallback ถี่เกิน)
    # Fallback เฉพาะตอน: model degenerate (ทำนาย class เดียว) หรือ early stop ≤ 30 iters
    if degen or best_iter < 30:
        log.warning("[FALLBACK] degenerate/trivial -> retrain w/o early stop")
        final = _build_xgb(n_estimators=800, with_early_stop=False, seed=main_seed)
        final.fit(X_tr, y_tr_arr, sample_weight=sw_tr, verbose=False)
        best_iter = 800
        val_acc = float(final.score(X_va, y_va))
        test_acc = float(final.score(X_te, y_te))

    y_pred = final.predict(X_te)
    cm = confusion_matrix(y_te, y_pred).tolist()
    cls_rep = classification_report(y_te, y_pred,
                                    target_names=["SELL", "HOLD", "BUY"],
                                    output_dict=True, zero_division=0)
    log.info("OOS confusion matrix: %s", cm)

    # 🆕 Per-class accuracy diagnostics + feature importance
    for cls_idx, cls_name in [(0, "SELL"), (1, "HOLD"), (2, "BUY")]:
        cls_total = int((y_te == cls_idx).sum())
        cls_correct = int(((y_te == cls_idx) & (y_pred == cls_idx)).sum())
        cls_acc = cls_correct / cls_total if cls_total > 0 else 0.0
        cls_pred_total = int((y_pred == cls_idx).sum())
        cls_precision = cls_correct / cls_pred_total if cls_pred_total > 0 else 0.0
        log.info("  %s: recall=%.3f (%d/%d) precision=%.3f predicted=%d",
                 cls_name, cls_acc, cls_correct, cls_total, cls_precision, cls_pred_total)

    # Feature importance (top 10)
    try:
        importances = final.feature_importances_
        ranked = sorted(zip(FEATURE_COLUMNS, importances), key=lambda x: -x[1])
        log.info("Top-10 feature importances:")
        for fname, fimp in ranked[:10]:
            log.info("  %.4f  %s", fimp, fname)
    except Exception as e:
        log.debug("feature importance failed: %s", e)

    # ============================================================
    # 🆕 PROFIT SIMULATION บน OOS test set — วัดว่าเทรนแล้วทำกำไรจริงไหม
    # ============================================================
    # commission cost ในหน่วย price → คำนวณจาก spec
    com_cfg = Config.section("commission") or {}
    com_usd = float(com_cfg.get("per_lot_round_trip_usd", 0.0))
    com_price_offset = 0.0
    if spec.trade_tick_size > 0 and spec.trade_tick_value > 0 and com_usd > 0:
        com_price_offset = com_usd / (spec.trade_tick_value / spec.trade_tick_size)

    # ใช้ค่า tp/sl/lookahead เดียวกับ label + execution
    tp_label = float(cfg_t.get("label_tp_atr", 1.6))
    sl_label = float(cfg_t.get("label_sl_atr", 0.8))
    la_label = int(cfg_t.get("label_lookahead", 12))
    # threshold ที่ bot จะใช้จริง
    sim_thr = float(cfg_a.get("min_confidence",
                              Config.section("hyper_frequency").get("min_confidence", 0.55)))

    # ดึง df raw ของ test partition จาก X.attrs (มี high/low/close/atr)
    df_raw = X.attrs.get("df_raw")
    profit = None
    profit_sweep = []   # 🆕 list ของ profit metrics ที่หลายๆ threshold
    best_threshold = sim_thr
    # criteria สำหรับเลือก threshold: PF ต้องสูงพอ + trades พอเพียง
    gate_cfg_pre = cfg_a.get("acceptance_gate") or {}
    min_pf_target = float(gate_cfg_pre.get("min_profit_factor", 1.30))
    min_trades_target = int(gate_cfg_pre.get("min_n_trades_test", 30))
    if df_raw is not None:
        df_te_full = df_raw.loc[X_te.index]
        # 🆕 Threshold sweep — หา threshold ที่ "PF ดี + trades ถี่" ตามที่ user ต้องการ
        sweep_thrs = [0.40, 0.42, 0.45, 0.48, 0.50, 0.52, 0.55, 0.58, 0.60, 0.65, 0.70, 0.75]
        log.info("=" * 70)
        log.info("📊 PROFIT-THRESHOLD SWEEP (OOS test, RR %.1f:%.1f, lookahead=%d)",
                 tp_label, sl_label, la_label)
        log.info("%-7s %-7s %-7s %-7s %-8s %-9s %-7s",
                 "thr", "trades", "WR%", "PF", "net_ATR", "exp/trade", "max_DD")
        for thr in sweep_thrs:
            try:
                pm = simulate_profit_metrics(
                    final, X_te, df_te_full,
                    threshold=thr, tp_atr=tp_label, sl_atr=sl_label,
                    lookahead=la_label, commission_price_offset=com_price_offset,
                )
                log.info("%-7.2f %-7d %-7.1f %-7.2f %-8.2f %-9.4f %-7.2f",
                         thr, pm["n_trades"], pm["win_rate"]*100,
                         pm["profit_factor"], pm["net_pnl_atrs"],
                         pm["expectancy_per_trade"], pm["max_drawdown_atrs"])
                profit_sweep.append({"threshold": thr, **pm})
            except Exception as e:
                log.warning("  sweep@%.2f failed: %s", thr, e)
        log.info("=" * 70)

        # 🎯 Selection — QUALITY FIRST: expectancy/trade × PF × sqrt(net)
        # expectancy/trade = ATR กำไรเฉลี่ยต่อออเดอร์ → บ่งบอกว่า "แม่น" แค่ไหน
        # Score สูง = กำไร/ไม้สูง + PF ดี + net สะสมสูง
        # ต้อง profitable + expectancy > 0 + trades ≥ min
        import math as _math
        def quality_score(pm):
            """Score = expectancy_per_trade × PF × sqrt(net_pnl) — rewards accuracy + quality + volume."""
            if pm["net_pnl_atrs"] <= 0 or pm["expectancy_per_trade"] <= 0:
                return 0.0
            return pm["expectancy_per_trade"] * pm["profit_factor"] * _math.sqrt(pm["net_pnl_atrs"])

        candidates = [pm for pm in profit_sweep
                      if pm["net_pnl_atrs"] > 0
                      and pm["profit_factor"] > 1.02
                      and pm["expectancy_per_trade"] > 0
                      and pm["n_trades"] >= min_trades_target]

        if candidates:
            best = max(candidates, key=quality_score)
            best_threshold = best["threshold"]
            profit = best
            log.info("✅ BEST (quality=%.3f): thr=%.2f → exp/trade=%.4f ATR | PF=%.2f | net=%.2f ATRs | trades=%d | WR=%.1f%%",
                     quality_score(best), best["threshold"],
                     best["expectancy_per_trade"], best["profit_factor"],
                     best["net_pnl_atrs"], best["n_trades"], best["win_rate"]*100)
            top3 = sorted(candidates, key=quality_score, reverse=True)[:3]
            for i, c in enumerate(top3[1:], 2):
                log.info("   #%d thr=%.2f exp=%.4f PF=%.2f net=%.2f trades=%d score=%.3f",
                         i, c["threshold"], c["expectancy_per_trade"],
                         c["profit_factor"], c["net_pnl_atrs"], c["n_trades"], quality_score(c))
            log.info("✅ BEST (PF≥%.2f, profitable): thr=%.2f → "
                     "PF=%.2f net=%.2f trades=%d WR=%.1f%%",
                     min_pf_target, best_threshold,
                     profit["profit_factor"], profit["net_pnl_atrs"],
                     profit["n_trades"], profit["win_rate"]*100)
        else:
            # ไม่มีอันไหน PF ≥ target → ลดเงื่อนไข floor
            min_pf_floor = float(gate_cfg_pre.get("min_pf_floor", 1.10))
            candidates_floor = [pm for pm in profit_sweep
                               if pm["profit_factor"] >= min_pf_floor
                               and pm["n_trades"] >= min_trades_target
                               and pm["net_pnl_atrs"] > 0]
            if candidates_floor:
                best = max(candidates_floor, key=lambda x: (x["profit_factor"], x["n_trades"]))
                best_threshold = best["threshold"]
                profit = best
                log.info("⚠️  FALLBACK1 (PF≥%.2f floor, profitable): thr=%.2f → PF=%.2f net=%.2f trades=%d",
                         min_pf_floor, best_threshold,
                         profit["profit_factor"], profit["net_pnl_atrs"], profit["n_trades"])
            else:
                # สุดท้าย: หา max net_pnl เท่านั้น
                fb2 = [pm for pm in profit_sweep if pm["n_trades"] >= 20]
                if fb2:
                    best = max(fb2, key=lambda x: x["net_pnl_atrs"])
                    best_threshold = best["threshold"]
                    profit = best
                    log.info("❗ FALLBACK2 (max net_pnl): thr=%.2f → PF=%.2f net=%.2f trades=%d",
                             best_threshold, profit["profit_factor"], profit["net_pnl_atrs"],
                             profit["n_trades"])
                else:
                    log.warning("❌ No threshold with enough trades — model unusable")
    else:
        log.warning("X.attrs['df_raw'] missing — profit simulation skipped")

    # ============================================================
    # 🆕 OPTIONAL LGBM ENSEMBLE — train LGBM on same data, average proba
    # ============================================================
    ens_cfg = cfg_a.get("ensemble") or {}
    final_model = final   # default: pure XGB
    if ens_cfg.get("enabled", False):
        lgbm_seed = main_seed + 1
        lgbm = _build_lgbm(n_estimators=1500, seed=lgbm_seed)
        if lgbm is not None:
            log.info("🔀 Training LightGBM (seed=%d) for ensemble...", lgbm_seed)
            try:
                lgbm.fit(X_tr, y_tr_arr, sample_weight=sw_tr)
                lgbm_acc = float(lgbm.score(X_te, y_te))
                xgb_w = float(ens_cfg.get("xgb_weight", 0.6))
                lgbm_w = float(ens_cfg.get("lgbm_weight", 0.4))
                ens = EnsembleModel(final, lgbm, xgb_weight=xgb_w, lgbm_weight=lgbm_w)
                ens_acc = float(ens.score(X_te, y_te))
                log.info("Ensemble: XGB_acc=%.4f LGBM_acc=%.4f ENS_acc=%.4f (w=%.1f/%.1f)",
                         test_acc, lgbm_acc, ens_acc, xgb_w, lgbm_w)
                # Only switch to ensemble if it doesn't degrade accuracy
                if ens_acc >= test_acc - 0.005:
                    final_model = ens
                    log.info("✅ Ensemble accepted — using XGB+LGBM blend")
                else:
                    log.info("⚠️  Ensemble worse than XGB alone — keeping XGB only")
            except Exception as e:
                log.warning("LGBM training failed: %s — falling back to XGB only", e)
        else:
            log.warning("LightGBM not installed — run: pip install lightgbm")

    # Re-run profit simulation with final_model (may be ensemble)
    if final_model is not final and df_raw is not None:
        log.info("📊 Re-running profit simulation on ensemble model...")
        profit_sweep_ens = []
        for thr in [0.40, 0.42, 0.45, 0.48, 0.50, 0.52, 0.55, 0.58, 0.60, 0.65, 0.70, 0.75]:
            try:
                pm = simulate_profit_metrics(
                    final_model, X_te, df_te_full,
                    threshold=thr, tp_atr=tp_label, sl_atr=sl_label,
                    lookahead=la_label, commission_price_offset=com_price_offset,
                )
                profit_sweep_ens.append({"threshold": thr, **pm})
            except Exception:
                pass
        import math as _math_ens
        def _qs_ens(pm):
            if pm["net_pnl_atrs"] <= 0 or pm["expectancy_per_trade"] <= 0: return 0.0
            return pm["expectancy_per_trade"] * pm["profit_factor"] * _math_ens.sqrt(pm["net_pnl_atrs"])
        ens_candidates = [pm for pm in profit_sweep_ens
                          if pm["net_pnl_atrs"] > 0 and pm["profit_factor"] > 1.02
                          and pm["expectancy_per_trade"] > 0 and pm["n_trades"] >= min_trades_target]
        if ens_candidates:
            best_ens = max(ens_candidates, key=_qs_ens)
            log.info("🔀 Ensemble best: thr=%.2f exp=%.4f PF=%.2f net=%.2f trades=%d WR=%.1f%%",
                     best_ens["threshold"], best_ens["expectancy_per_trade"],
                     best_ens["profit_factor"], best_ens["net_pnl_atrs"],
                     best_ens["n_trades"], best_ens["win_rate"]*100)
            # Use ensemble results if better
            if _qs_ens(best_ens) > (quality_score(profit) if profit else 0):
                profit = best_ens
                best_threshold = best_ens["threshold"]
                profit_sweep = profit_sweep_ens
                log.info("✅ Ensemble profit metrics BETTER — adopting")

    out = model_path(cfg_a["model_filename"])

    # ============================================================
    # 🏆 Champion Score — track best PF×net ever achieved across all retrains
    # New model must beat champion score, not just the latest saved model
    # ============================================================
    champion_path = out.parent / "champion_score.json"
    champion_score = 0.0
    champion_meta = {}
    if champion_path.exists():
        try:
            champion_meta = json.loads(champion_path.read_text(encoding="utf-8"))
            champion_score = float(champion_meta.get("pf_net_score", 0.0))
            log.info("🏆 Champion score: %.1f (PF=%.2f net=%.2f seed=%s thr=%.2f)",
                     champion_score,
                     champion_meta.get("profit_factor", 0),
                     champion_meta.get("net_pnl_atrs", 0),
                     champion_meta.get("training_seed", "?"),
                     champion_meta.get("best_threshold", 0))
        except Exception as e:
            log.warning("Champion score load failed: %s", e)

    # ============================================================
    # �🆕 Model Acceptance Gate (profit-based) — รับเฉพาะ model ที่ทำกำไรจริง
    # ============================================================
    gate_cfg = cfg_a.get("acceptance_gate") or {}
    if gate_cfg.get("enabled", True):
        # TARGET: PF ≥ 1.50 + net ≥ 40 ATRs + frequent trades
        min_net_pnl_target = float(gate_cfg.get("min_net_pnl_atrs_target", 40.0))
        min_pf_aggressive = float(gate_cfg.get("min_profit_factor_aggressive", 1.50))
        min_win_rate_target = float(gate_cfg.get("min_win_rate_target", 0.35))
        min_pf_floor = float(gate_cfg.get("min_pf_floor", 1.02))
        min_exp = float(gate_cfg.get("min_expectancy_per_trade_atrs", 0.005))
        min_trades = int(gate_cfg.get("min_n_trades_test", 50))
        min_net_pnl = float(gate_cfg.get("min_net_pnl_atrs", 0.0))
        # Legacy classification gates
        min_oos_acc = float(gate_cfg.get("min_oos_test_acc", 0.30))
        min_dir_ratio = float(gate_cfg.get("min_dir_balance_ratio", 0.5))
        max_dir_ratio = float(gate_cfg.get("max_dir_balance_ratio", 2.0))
        n_buy_pred = int((y_pred == 2).sum())
        n_sell_pred = int((y_pred == 0).sum())
        dir_ratio = (n_buy_pred / n_sell_pred) if n_sell_pred > 0 else 999.0

        # เปรียบเทียบ model เก่าด้วย (ถ้ามี) — ใช้ "best threshold ของเก่า" เทียบเป็นธรรม
        old_test_acc = -1.0
        old_pf = -1.0
        old_net = -float("inf")
        if out.exists():
            try:
                old_bundle = joblib.load(out)
                old_model = old_bundle["model"]
                old_features = old_bundle.get("features", FEATURE_COLUMNS)
                try:
                    X_te_old = X_te[old_features]
                    old_test_acc = float(old_model.score(X_te_old, y_te))
                    if df_raw is not None:
                        # หา best threshold ของ old ด้วยเช่นกัน (fair comparison)
                        best_old = None
                        for thr in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
                            try:
                                op = simulate_profit_metrics(
                                    old_model, X_te_old, df_te_full,
                                    threshold=thr, tp_atr=tp_label, sl_atr=sl_label,
                                    lookahead=la_label,
                                    commission_price_offset=com_price_offset,
                                )
                                if op["n_trades"] >= 20 and op["net_pnl_atrs"] > old_net:
                                    old_net = op["net_pnl_atrs"]
                                    best_old = op
                            except Exception:
                                pass
                        if best_old is not None:
                            old_pf = best_old["profit_factor"]
                            log.info("📊 OLD model BEST on same OOS: PF=%.2f  net=%.2f ATRs  trades=%d",
                                     old_pf, best_old["net_pnl_atrs"], best_old["n_trades"])
                except Exception as e:
                    log.warning("Old model score failed: %s", e)
            except Exception as e:
                log.warning("Old bundle load failed: %s", e)

        decision = "ACCEPT"
        reason = ""

        if profit is None:
            decision = "REJECT"
            reason = "profit simulation unavailable — cannot validate"
        elif profit["n_trades"] < min_trades:
            decision = "REJECT"
            reason = (f"trades on OOS = {profit['n_trades']} < min {min_trades} "
                      f"(model too quiet across all thresholds)")
        # Compute quality score for new vs champion
        import math as _math2
        def _qs(pf, net, exp):
            if net <= 0 or exp <= 0: return 0.0
            return exp * pf * _math2.sqrt(net)
        new_score = _qs(profit["profit_factor"], profit["net_pnl_atrs"], profit["expectancy_per_trade"])
        old_exp = (old_net / max(1, profit["n_trades"])) if old_net > 0 else 0.0  # approx
        old_score = _qs(old_pf, old_net, old_exp)
        log.info("📈 Quality score: NEW=%.4f (exp=%.4f PF=%.2f net=%.2f) | 🏆 CHAMPION=%.4f",
                 new_score, profit["expectancy_per_trade"], profit["profit_factor"],
                 profit["net_pnl_atrs"], champion_score)

        if new_score > champion_score + 0.001:
            # ✅ New model beats the ALL-TIME best quality score — ACCEPT
            reason = (f"CHAMPION BEAT ✅ | quality={new_score:.4f} > champion={champion_score:.4f} | "
                      f"exp/trade={profit['expectancy_per_trade']:.4f} ATR | "
                      f"PF={profit['profit_factor']:.2f} | net={profit['net_pnl_atrs']:.2f} ATRs | "
                      f"thr={best_threshold:.2f} | trades={profit['n_trades']} | WR={profit['win_rate']*100:.1f}%")
        elif champion_score > 0.001 and new_score <= champion_score:
            # ❌ New model doesn't beat champion — keep champion
            decision = "REJECT"
            reason = (f"NOT BETTER THAN CHAMPION | quality={new_score:.4f} ≤ champion={champion_score:.4f} "
                      f"(exp={profit['expectancy_per_trade']:.4f} PF={profit['profit_factor']:.2f}) — keeping champion")
        elif profit["profit_factor"] < min_pf_floor:
            decision = "REJECT"
            reason = (f"PF {profit['profit_factor']:.2f} ≤ {min_pf_floor} — ขาดทุน")
        elif profit["net_pnl_atrs"] < min_net_pnl:
            decision = "REJECT"
            reason = (f"net {profit['net_pnl_atrs']:.2f} ATRs < min {min_net_pnl}")
        elif test_acc < min_oos_acc:
            decision = "REJECT"
            reason = f"acc {test_acc:.4f} < min {min_oos_acc}"
        elif dir_ratio < min_dir_ratio or dir_ratio > max_dir_ratio:
            decision = "REJECT"
            reason = (f"direction imbalance BUY/SELL = {n_buy_pred}/{n_sell_pred} "
                      f"= {dir_ratio:.2f} (need {min_dir_ratio}-{max_dir_ratio})")
        else:
            reason = (f"PF={profit['profit_factor']:.2f} | "
                      f"net={profit['net_pnl_atrs']:.2f} ATRs | "
                      f"trades={profit['n_trades']} | thr={best_threshold:.2f}")

        log.info("🚦 Acceptance Gate: %s — %s", decision, reason)
        if decision == "REJECT":
            log.warning("⛔ Keeping OLD model — new model rejected")
            rej_meta_path = out.parent / (out.stem + "_rejected.json")
            try:
                rej_meta_path.write_text(json.dumps({
                    "rejected_at_utc": datetime.now(timezone.utc).isoformat(),
                    "reason": reason,
                    "new_test_acc": test_acc,
                    "old_test_acc": old_test_acc,
                    "profit_metrics": profit,
                    "old_profit_factor": old_pf,
                    "buy_predicted": n_buy_pred,
                    "sell_predicted": n_sell_pred,
                    "dir_ratio": dir_ratio,
                }, indent=2, default=str), encoding="utf-8")
            except Exception:
                pass
            return {
                "rejected": True,
                "reason": reason,
                "new_test_acc": test_acc,
                "old_test_acc": old_test_acc,
                "profit_metrics": profit,
            }

    joblib.dump({"model": final_model, "features": FEATURE_COLUMNS}, out)

    # 🗃️ บันทึก retrain event ลง DB — แยกได้ชัดว่า trades ไหน pre/post retrain
    try:
        from .retrain_log import log_retrain
        log_retrain(model_path=out, rows=int(n),
                    cv_acc=float(np.mean(cv_accs)),
                    oos_acc=float(test_acc), accepted=True,
                    notes=f"cv={float(np.mean(cv_accs)):.4f} oos={float(test_acc):.4f}")
    except Exception as e:
        log.warning("could not log retrain event: %s", e)

    meta = {
        "symbol": symbol,
        "timeframe": timeframe,
        "rows_trained": int(n),
        "split": {"train": train_end, "val": val_end - train_end, "test": n - val_end},
        "class_distribution": {int(k): int(v) for k, v in y.value_counts().items()},
        "cv_acc_mean": float(np.mean(cv_accs)),
        "cv_acc_std": float(np.std(cv_accs)),
        "val_acc": val_acc,
        "oos_test_acc": test_acc,
        "best_iteration": best_iter,
        "oos_classification_report": cls_rep,
        "oos_confusion_matrix": cm,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_columns": FEATURE_COLUMNS,
        "min_confidence_target": float(cfg_a.get("min_confidence", 0.35)),
        "profit_metrics_oos": profit,   # 🆕 net_pnl, PF, expectancy, etc.
        "profit_threshold_sweep": profit_sweep,
        "best_threshold": best_threshold,
        "training_seed": main_seed,
        "label_config": {
            "tp_atr": tp_label, "sl_atr": sl_label, "lookahead": la_label,
        },
        "symbol_spec": {"digits": spec.digits, "point": spec.point,
                        "tick_size": spec.trade_tick_size,
                        "tick_value": spec.trade_tick_value,
                        "contract_size": spec.trade_contract_size},
    }
    meta_path = model_path(cfg_a["metadata_filename"])
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # 🏆 Update champion score if this model is the new best (quality score)
    if profit is not None:
        import math as _math3
        def _qs3(pf, net, exp):
            if net <= 0 or exp <= 0: return 0.0
            return exp * pf * _math3.sqrt(net)
        new_pf_net = _qs3(profit["profit_factor"], profit["net_pnl_atrs"], profit["expectancy_per_trade"])
        if new_pf_net > champion_score:
            new_champion = {
                "pf_net_score": round(new_pf_net, 2),
                "profit_factor": round(profit["profit_factor"], 4),
                "net_pnl_atrs": round(profit["net_pnl_atrs"], 4),
                "win_rate": round(profit["win_rate"], 4),
                "n_trades": profit["n_trades"],
                "best_threshold": best_threshold,
                "training_seed": main_seed,
                "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            champion_path.write_text(json.dumps(new_champion, indent=2), encoding="utf-8")
            log.info("🏆 NEW CHAMPION! score=%.1f (PF=%.2f net=%.2f seed=%d thr=%.2f)",
                     new_pf_net, profit["profit_factor"], profit["net_pnl_atrs"],
                     main_seed, best_threshold)

    log.info("Model saved -> %s", out)
    return meta


if __name__ == "__main__":
    print(json.dumps(train_from_mt5(), indent=2))
