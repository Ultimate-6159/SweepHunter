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
from .m1_hyper_pipeline import FEATURE_COLUMNS, build_training_dataset
from .logger import get_logger
from .mt5_connector import MT5Connector
from .paths import model_path

log = get_logger("trainer")


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
                "SELECT ts_utc, prediction, status FROM decisions "
                "WHERE symbol=? AND status IN ('WIN','LOSS') "
                "ORDER BY ts_utc",
                conn, params=(symbol,),
            )
        if df.empty:
            return None
        df["time"] = pd.to_datetime(df["ts_utc"], utc=True, errors="coerce")
        df = df.dropna(subset=["time"])
        df = df.rename(columns={"prediction": "side", "status": "outcome"})
        return df[["time", "side", "outcome"]]
    except Exception as e:
        log.warning("Trade-aug: failed to load DB outcomes: %s", e)
        return None


def _apply_trade_augmentation(X_part, y_part, sw_part, real_df,
                              win_weight: float, loss_weight: float,
                              loss_mode="flip", tolerance_min: int = 5):
    """
    🆕 Apply trade-aug: match real trade timestamps to bars in X_part,
    boost their sample weights, optionally relabel LOSSes.

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

    for p, outcome, side in zip(matched_pos, matched_outcome, matched_side):
        if outcome == "LOSS":
            sw_arr[p] = sw_arr[p] * float(loss_weight)
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
            sw_arr[p] = sw_arr[p] * float(win_weight)
        n_matched += 1

    return y_arr, sw_arr, n_matched, n_relabeled


def _build_xgb(n_estimators: int = 800, with_early_stop: bool = True) -> XGBClassifier:
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
        random_state=42,
    )
    if with_early_stop:
        kwargs["early_stopping_rounds"] = 150
    return XGBClassifier(**kwargs)


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
            # Relabel on FULL y first (placeholder sw, we recompute per-partition below)
            sw_placeholder = np.ones(len(y), dtype=float)
            y_full_arr, _, n_match_full, n_rel_full = _apply_trade_augmentation(
                X, y, sw_placeholder, real_outcomes, win_w_g, loss_w_g, loss_mode_g)
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
    cw_override = cfg_a.get("class_weight_overrides")
    if cw_override:
        log.info("Applying class_weight overrides: %s", cw_override)
        for cls_str, mult in cw_override.items():
            cls_int = int(cls_str)
            mask = (y_tr.values == cls_int)
            sw_tr[mask] = sw_tr[mask] * float(mult)

    # 🆕 Trade-Augmented Learning: weight boost on training partition
    y_tr_arr = y_tr.values
    if real_outcomes is not None:
        win_w = float(ta_cfg.get("win_weight", 2.0))
        loss_w = float(ta_cfg.get("loss_weight", 3.0))
        loss_mode = ta_cfg.get("loss_mode", ta_cfg.get("relabel_loss_to_hold", "flip"))
        y_tr_arr, sw_tr, n_match, n_rel = _apply_trade_augmentation(
            X_tr, y_tr, sw_tr, real_outcomes, win_w, loss_w, loss_mode)
        log.info("Trade-aug (train partition): matched %d bars, weight-boosted (%d also relabeled in train)",
                 n_match, n_rel)

    tscv = TimeSeriesSplit(n_splits=4)
    cv_accs = []
    for k, (i_tr, i_va) in enumerate(tscv.split(X_tr), 1):
        sw_k = compute_sample_weight(class_weight="balanced", y=y_tr.iloc[i_tr])
        if cw_override:
            for cls_str, mult in cw_override.items():
                cls_int = int(cls_str)
                mask = (y_tr.iloc[i_tr].values == cls_int)
                sw_k[mask] = sw_k[mask] * float(mult)
        m = _build_xgb(n_estimators=600, with_early_stop=True)
        m.fit(X_tr.iloc[i_tr], y_tr.iloc[i_tr], sample_weight=sw_k,
              eval_set=[(X_tr.iloc[i_va], y_tr.iloc[i_va])], verbose=False)
        acc = float(m.score(X_tr.iloc[i_va], y_tr.iloc[i_va]))
        cv_accs.append(acc)
        log.info("  CV %d/4 acc=%.4f best_iter=%d", k, acc, m.best_iteration)
    log.info("CV mean=%.4f std=%.4f", float(np.mean(cv_accs)), float(np.std(cv_accs)))

    final = _build_xgb(n_estimators=4000, with_early_stop=True)
    final.fit(X_tr, y_tr_arr, sample_weight=sw_tr, eval_set=[(X_va, y_va)], verbose=False)
    best_iter = int(final.best_iteration)
    val_acc = float(final.score(X_va, y_va))
    test_acc = float(final.score(X_te, y_te))
    log.info("Best iter=%d val_acc=%.4f OOS_acc=%.4f", best_iter, val_acc, test_acc)

    sample_n = min(2000, len(X_te))
    degen, msg = _is_degenerate(final, X_te.iloc[:sample_n], y_te.iloc[:sample_n])
    log.info("Degeneracy: %s", msg)

    if degen or best_iter < 30 or test_acc < 0.36:
        log.warning("[FALLBACK] degenerate -> retrain w/o early stop")
        final = _build_xgb(n_estimators=500, with_early_stop=False)
        final.fit(X_tr, y_tr, sample_weight=sw_tr, verbose=False)
        best_iter = 500
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

    out = model_path(cfg_a["model_filename"])

    # 🆕 Model Acceptance Gate — กันรีเทรนแล้วแย่ลง
    # เปรียบเทียบ test_acc ของ model เก่า (ถ้ามี) บน same OOS set
    # ถ้า new < old - max_drop_pct → REJECT, keep old model
    gate_cfg = cfg_a.get("acceptance_gate") or {}
    if gate_cfg.get("enabled", True) and out.exists():
        try:
            max_drop = float(gate_cfg.get("max_test_acc_drop_pct", 3.0)) / 100.0
            min_oos_acc = float(gate_cfg.get("min_oos_test_acc", 0.40))
            old_bundle = joblib.load(out)
            old_model = old_bundle["model"]
            old_features = old_bundle.get("features", FEATURE_COLUMNS)
            # ใช้ X_te ปัจจุบัน (subset features ให้ตรงกับ old model)
            try:
                X_te_old = X_te[old_features]
                old_test_acc = float(old_model.score(X_te_old, y_te))
            except Exception as e:
                log.warning("Acceptance gate: cannot score old model (%s) → accept new",
                            e)
                old_test_acc = -1.0

            decision = "ACCEPT"
            reason = ""
            # 🆕 Direction Balance Check — กัน model เอียงทำนายข้างเดียว
            min_dir_ratio = float(gate_cfg.get("min_dir_balance_ratio", 0.5))
            max_dir_ratio = float(gate_cfg.get("max_dir_balance_ratio", 2.0))
            n_buy_pred = int((y_pred == 2).sum())
            n_sell_pred = int((y_pred == 0).sum())
            dir_ratio = (n_buy_pred / n_sell_pred) if n_sell_pred > 0 else 999.0

            if test_acc < min_oos_acc:
                decision = "REJECT"
                reason = f"new acc {test_acc:.4f} < min_oos_acc {min_oos_acc}"
            elif old_test_acc > 0 and test_acc < (old_test_acc - max_drop):
                decision = "REJECT"
                reason = (f"new {test_acc:.4f} drops > {max_drop*100:.1f}% "
                          f"vs old {old_test_acc:.4f}")
            elif dir_ratio < min_dir_ratio or dir_ratio > max_dir_ratio:
                decision = "REJECT"
                reason = (f"direction imbalance: BUY/SELL = {n_buy_pred}/{n_sell_pred} "
                          f"= {dir_ratio:.2f} (need {min_dir_ratio}-{max_dir_ratio})")
            else:
                old_str = f"{old_test_acc:.4f}" if old_test_acc > 0 else "n/a"
                reason = (f"new {test_acc:.4f} vs old {old_str} | "
                          f"BUY/SELL ratio={dir_ratio:.2f}")

            log.info("🚦 Acceptance Gate: %s — %s", decision, reason)
            if decision == "REJECT":
                log.warning("⛔ Keeping OLD model — new model rejected")
                # อัพเดท meta เพื่อบันทึก attempt ที่ถูก reject
                rej_meta_path = out.parent / (out.stem + "_rejected.json")
                rej_meta_path.write_text(json.dumps({
                    "rejected_at_utc": datetime.now(timezone.utc).isoformat(),
                    "reason": reason,
                    "new_test_acc": test_acc,
                    "old_test_acc": old_test_acc,
                    "buy_predicted": n_buy_pred,
                    "sell_predicted": n_sell_pred,
                    "dir_ratio": dir_ratio,
                    "min_required": min_oos_acc,
                }, indent=2), encoding="utf-8")
                # คืน meta แต่ไม่ overwrite model
                return {
                    "rejected": True,
                    "reason": reason,
                    "new_test_acc": test_acc,
                    "old_test_acc": old_test_acc,
                }
        except Exception as e:
            log.warning("Acceptance gate failed (will accept new): %s", e)

    joblib.dump({"model": final, "features": FEATURE_COLUMNS}, out)

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
        "symbol_spec": {"digits": spec.digits, "point": spec.point,
                        "tick_size": spec.trade_tick_size,
                        "tick_value": spec.trade_tick_value,
                        "contract_size": spec.trade_contract_size},
    }
    meta_path = model_path(cfg_a["metadata_filename"])
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("Model saved -> %s", out)
    return meta


if __name__ == "__main__":
    print(json.dumps(train_from_mt5(), indent=2))
