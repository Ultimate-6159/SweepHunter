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

    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    X_tr, y_tr = X.iloc[:train_end], y.iloc[:train_end]
    X_va, y_va = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_te, y_te = X.iloc[val_end:], y.iloc[val_end:]

    sw_tr = compute_sample_weight(class_weight="balanced", y=y_tr)

    tscv = TimeSeriesSplit(n_splits=4)
    cv_accs = []
    for k, (i_tr, i_va) in enumerate(tscv.split(X_tr), 1):
        sw_k = compute_sample_weight(class_weight="balanced", y=y_tr.iloc[i_tr])
        m = _build_xgb(n_estimators=600, with_early_stop=True)
        m.fit(X_tr.iloc[i_tr], y_tr.iloc[i_tr], sample_weight=sw_k,
              eval_set=[(X_tr.iloc[i_va], y_tr.iloc[i_va])], verbose=False)
        acc = float(m.score(X_tr.iloc[i_va], y_tr.iloc[i_va]))
        cv_accs.append(acc)
        log.info("  CV %d/4 acc=%.4f best_iter=%d", k, acc, m.best_iteration)
    log.info("CV mean=%.4f std=%.4f", float(np.mean(cv_accs)), float(np.std(cv_accs)))

    final = _build_xgb(n_estimators=2500, with_early_stop=True)
    final.fit(X_tr, y_tr, sample_weight=sw_tr, eval_set=[(X_va, y_va)], verbose=False)
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

    out = model_path(cfg_a["model_filename"])
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
