"""
⚖️  strategy_weights.py
========================
Performance-Weighted Trade Augmentation
----------------------------------------
แทนที่จะให้ทุก WIN/LOSS น้ำหนัก × 1.5 เท่ากัน → คำนวณ "strategy quality score"
ของ snapshot ที่ trade นั้นเกิด แล้วปรับ weight ให้สอดคล้อง:

    weight_multiplier = base_weight × (W_FLOOR + (W_CEIL - W_FLOOR) × score)

โดย score ∈ (0, 1) คำนวณจากสูตร (sigmoid blend):

    score = σ(α·EV_norm + β·RR_norm + γ·CommHealth_norm + δ·Stability_norm)

ตัวแปร (ทั้งหมดอ่านจาก config — ไม่มีค่า hardcode):

  α = profit_weight       (น้ำหนักของ EV/trade)
  β = rr_weight           (น้ำหนักของ Risk-Reward Ratio)
  γ = commission_weight   (penalty ถ้า commission กิน PnL เยอะ)
  δ = stability_weight    (penalty ถ้า sample เล็กเกินไป)

ผลลัพธ์:
  - Snapshot ดี (high EV, high RR, low comm) → weight × ~2.0
  - Snapshot กลาง                            → weight × ~1.0
  - Snapshot แย่ (negative EV)               → weight × ~0.5  (ลด)
"""
from __future__ import annotations
import math
import sqlite3
from pathlib import Path
from typing import Dict, Optional

from .config import Config
from .logger import get_logger
from .paths import db_path

log = get_logger("sw")


def _sigmoid(x: float) -> float:
    if x >= 50:
        return 1.0
    if x <= -50:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def _get_db_path() -> Path:
    fname = Config.section("database").get("filename", "hyper_trades.sqlite")
    return db_path(fname)


def _load_weights_config() -> dict:
    """Defaults ถ้าไม่ได้กำหนดใน config.json — ผู้ใช้ override ได้ทุกค่า."""
    cfg = Config.section("ai") or {}
    sw = cfg.get("strategy_weighted_aug") or {}
    return {
        "enabled": bool(sw.get("enabled", False)),
        # Sample size guards
        "min_trades_per_snapshot": int(sw.get("min_trades_per_snapshot", 30)),
        "min_trades_use_default": int(sw.get("min_trades_use_default", 10)),
        # Sigmoid weights for score blending (formula coefficients)
        "alpha_profit": float(sw.get("alpha_profit", 0.30)),     # per $/trade
        "beta_rr": float(sw.get("beta_rr", 1.50)),               # per RR unit (above 1.0)
        "gamma_commission": float(sw.get("gamma_commission", 4.00)),  # penalty for high comm%
        "delta_stability": float(sw.get("delta_stability", 0.50)),    # bonus for large sample
        # Weight scaling envelope
        "weight_floor": float(sw.get("weight_floor", 0.50)),     # min multiplier (worst strategies)
        "weight_ceil": float(sw.get("weight_ceil", 2.00)),       # max multiplier (best strategies)
        # Targets used for normalization (where score=0.5 baseline)
        "target_ev_per_trade_usd": float(sw.get("target_ev_per_trade_usd", 1.50)),
        "target_rr": float(sw.get("target_rr", 1.30)),
        "target_commission_ratio": float(sw.get("target_commission_ratio", 0.30)),  # 30% baseline
        # Default for snapshots with too few trades
        "default_score": float(sw.get("default_score", 0.50)),
    }


def _commission_per_trade_usd() -> float:
    cfg_c = Config.section("commission") or {}
    return float(cfg_c.get("per_lot_round_trip_usd", 0.0))


def compute_snapshot_scores() -> Dict[int, dict]:
    """
    คำนวณ score ของทุก snapshot — คืน {snapshot_id: {score, weight_mult, stats}}

    สูตร (formula-driven, NOT hardcoded — ทุกค่าปรับใน config.json):
        ev_norm     = (EV - target_ev) / target_ev          # > 0 if profitable
        rr_norm     = (RR - target_rr) / target_rr           # > 0 if RR > target
        comm_norm   = (target_comm - actual_comm) / target_comm
                                                             # > 0 if commission กิน PnL น้อยกว่า target
        stab_norm   = log(1 + n / min_trades) - 1            # > 0 if sample เกิน min
        z = α·ev_norm + β·rr_norm + γ·comm_norm + δ·stab_norm
        score = sigmoid(z)
        weight_mult = floor + (ceil - floor) × score
    """
    cfg = _load_weights_config()
    if not cfg["enabled"]:
        return {}

    path = _get_db_path()
    if not path.exists():
        return {}

    com_per_lot = _commission_per_trade_usd()
    out: Dict[int, dict] = {}

    with sqlite3.connect(str(path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT
                config_snapshot_id AS sid,
                COUNT(*)                                                AS n,
                SUM(CASE WHEN status='WIN'  THEN 1 ELSE 0 END)         AS w_n,
                AVG(CASE WHEN status='WIN'  AND pnl > 0 THEN pnl END)  AS avg_w,
                AVG(CASE WHEN status='LOSS' AND pnl < 0 THEN pnl END)  AS avg_l,
                SUM(pnl)                                                AS net_pnl,
                SUM(volume)                                             AS total_volume
            FROM decisions
            WHERE status IN ('WIN','LOSS') AND config_snapshot_id IS NOT NULL
            GROUP BY config_snapshot_id
        """).fetchall()

    for r in rows:
        sid = int(r["sid"])
        n = int(r["n"] or 0)
        if n < cfg["min_trades_use_default"]:
            # very few samples — neutral weight
            score = cfg["default_score"]
            weight_mult = cfg["weight_floor"] + (cfg["weight_ceil"] - cfg["weight_floor"]) * score
            out[sid] = {"score": score, "weight_mult": weight_mult, "n": n,
                        "ev": 0.0, "rr": 0.0, "comm_ratio": 0.0,
                        "reason": "too_few_trades"}
            continue

        avg_w = float(r["avg_w"] or 0.0)
        avg_l = float(r["avg_l"] or 0.0)
        net = float(r["net_pnl"] or 0.0)
        vol = float(r["total_volume"] or 0.0)

        ev = net / n if n else 0.0
        rr = (avg_w / abs(avg_l)) if avg_l < 0 else 0.0
        gross = sum([abs(avg_w) * (r["w_n"] or 0), abs(avg_l) * (n - (r["w_n"] or 0))])
        commission = vol * com_per_lot
        comm_ratio = commission / gross if gross > 0 else 1.0

        # ----- formula-driven normalization -----
        ev_norm = (ev - cfg["target_ev_per_trade_usd"]) / max(cfg["target_ev_per_trade_usd"], 1e-6)
        rr_norm = (rr - cfg["target_rr"]) / max(cfg["target_rr"], 1e-6)
        comm_norm = (cfg["target_commission_ratio"] - comm_ratio) / max(cfg["target_commission_ratio"], 1e-6)
        # stability bonus: log scale → diminishing returns above min_trades
        stab_norm = math.log(1 + n / max(cfg["min_trades_per_snapshot"], 1)) - 1.0

        z = (cfg["alpha_profit"] * ev_norm +
             cfg["beta_rr"] * rr_norm +
             cfg["gamma_commission"] * comm_norm +
             cfg["delta_stability"] * stab_norm)
        score = _sigmoid(z)
        weight_mult = cfg["weight_floor"] + (cfg["weight_ceil"] - cfg["weight_floor"]) * score

        out[sid] = {
            "score": round(score, 4),
            "weight_mult": round(weight_mult, 4),
            "n": n,
            "ev": round(ev, 4),
            "rr": round(rr, 4),
            "comm_ratio": round(comm_ratio, 4),
            "z": round(z, 4),
        }
    return out


def report_scores() -> str:
    """ทำตารางอ่านง่ายของทุก snapshot — ใช้ debug + print หลัง training."""
    scores = compute_snapshot_scores()
    if not scores:
        return "[strategy-weighted aug DISABLED or no data]"
    lines = [
        f"  {'sid':<4} {'n':<5} {'EV/tr':<8} {'RR':<6} {'comm%':<7} {'score':<7} {'×weight':<7}",
        "  " + "-" * 52,
    ]
    for sid, s in sorted(scores.items()):
        lines.append(
            f"  {sid:<4} {s['n']:<5} ${s.get('ev',0):<+6.2f}  "
            f"{s.get('rr',0):<5.2f} {s.get('comm_ratio',0)*100:<6.1f} "
            f"{s['score']:<6.3f} {s['weight_mult']:<6.3f}"
        )
    return "\n".join(lines)
