"""
🗂️  config_snapshot.py
======================
Auto-detect config changes → upsert config_snapshots row → return current id.

หลักการ:
  - Hash full config.json → ถ้า hash เปลี่ยน = เกิด snapshot ใหม่
  - ใช้ in-process cache ลด DB hit (refresh ทุก N วินาที)
  - ทุก decision ที่ insert จะมี config_snapshot_id ผูกไว้
    → ใช้ภายหลังคำนวณ "strategy quality score"
"""
from __future__ import annotations
import hashlib
import json
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .logger import get_logger
from .paths import db_path

log = get_logger("snap")

_CACHE_LOCK = threading.Lock()
_CACHED_ID: Optional[int] = None
_CACHED_HASH: Optional[str] = None
_LAST_CHECK_TS: float = 0.0
_CHECK_INTERVAL_SEC: float = 60.0  # re-hash config ทุก 60s (cheap)

_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_FILE = _ROOT / "config.json"


def _hash_config(cfg: dict) -> str:
    raw = json.dumps(cfg, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _extract_key_fields(cfg: dict) -> dict:
    t = cfg.get("trading", {})
    a = cfg.get("account_scaling", {})
    s = cfg.get("smart_trailing", {})
    rf = cfg.get("risk_filters", {}).get("series_loss_cap", {})
    hl = cfg.get("hourly_lot_multiplier", {})
    return {
        "risk_per_trade_pct": a.get("risk_per_trade_pct"),
        "sl_atr_mult": t.get("sl_atr_mult"),
        "tp_atr_mult": t.get("tp_atr_mult"),
        "smart_trailing_enabled": int(bool(s.get("enabled", False))),
        "be_trigger_atr": s.get("be_trigger_atr"),
        "series_loss_cap_pct": rf.get("max_loss_pct_of_balance"),
        "series_loss_cap_action": rf.get("action", "close"),
        "hourly_lot_mult_enabled": int(bool(hl.get("enabled", False))),
    }


def _build_diff_label(prev_fields: Optional[dict], new_fields: dict, full_cfg: dict) -> str:
    """
    สร้าง label ที่บอกชัดว่า "อะไรเปลี่ยนจากเก่า" — มีประโยชน์มากกว่า "auto-detected".
    Example: "auto: trail 1→0 | risk 1.0→1.25 | conf 0.55→0.65"
    """
    if not prev_fields:
        return "auto: initial config snapshot"
    changes = []
    label_map = {
        "risk_per_trade_pct": "risk",
        "sl_atr_mult": "sl",
        "tp_atr_mult": "tp",
        "smart_trailing_enabled": "trail",
        "be_trigger_atr": "trigger",
        "series_loss_cap_pct": "loss_cap",
        "series_loss_cap_action": "cap_act",
        "hourly_lot_mult_enabled": "hmul",
    }
    for key, short in label_map.items():
        old = prev_fields.get(key)
        new = new_fields.get(key)
        if old != new:
            changes.append(f"{short} {old}→{new}")
    # Also detect deeper changes that aren't in the key fields
    extras = _detect_extra_changes(full_cfg)
    if extras:
        changes.extend(extras)
    if not changes:
        return "auto: config saved (no key changes)"
    return "auto: " + " | ".join(changes[:5])  # cap at 5 items


def _detect_extra_changes(cfg: dict) -> list:
    """ตรวจ key fields นอกเหนือจาก _extract_key_fields ที่สำคัญ"""
    out = []
    h = cfg.get("hyper_frequency", {})
    if "min_confidence" in h:
        out.append(f"min_conf={h['min_confidence']}")
    dt = h.get("directional_threshold", {})
    if dt.get("enabled"):
        b = dt.get("buy"); s_ = dt.get("sell")
        bm = dt.get("buy_max"); sm = dt.get("sell_max")
        out.append(f"BUY[{b}-{bm or '?'}] SELL[{s_}-{sm or '?'}]")
    rec = cfg.get("recovery", {})
    if "enabled" in rec:
        out.append(f"recov={'ON' if rec['enabled'] else 'OFF'}")
    rgm = cfg.get("regime_filter", {})
    if rgm.get("enabled"):
        out.append("regime=ON")
    return out


def _db_path() -> Path:
    from .config import Config
    fname = Config.section("database").get("filename", "hyper_trades.sqlite")
    return db_path(fname)


def get_current_snapshot_id(force_refresh: bool = False) -> Optional[int]:
    """
    คืน id ของ snapshot ปัจจุบัน — auto-create ถ้า config hash ใหม่.
    Cache ผลลัพธ์ 60s ลด overhead.
    """
    global _CACHED_ID, _CACHED_HASH, _LAST_CHECK_TS
    now = time.time()
    if not force_refresh and _CACHED_ID is not None and (now - _LAST_CHECK_TS) < _CHECK_INTERVAL_SEC:
        return _CACHED_ID

    with _CACHE_LOCK:
        try:
            if not _CONFIG_FILE.exists():
                return _CACHED_ID
            cfg = json.loads(_CONFIG_FILE.read_text(encoding="utf-8"))
            cur_hash = _hash_config(cfg)
            if cur_hash == _CACHED_HASH and _CACHED_ID is not None:
                _LAST_CHECK_TS = now
                return _CACHED_ID

            path = _db_path()
            with sqlite3.connect(str(path)) as conn:
                # check if hash already in DB
                row = conn.execute(
                    "SELECT id FROM config_snapshots WHERE config_hash=? "
                    "ORDER BY id DESC LIMIT 1", (cur_hash,)).fetchone()
                if row:
                    sid = int(row[0])
                else:
                    fields = _extract_key_fields(cfg)
                    # ดึง snapshot ก่อนหน้า → สร้าง diff label
                    prev = conn.execute("""
                        SELECT risk_per_trade_pct, sl_atr_mult, tp_atr_mult,
                               smart_trailing_enabled, be_trigger_atr,
                               series_loss_cap_pct, series_loss_cap_action,
                               hourly_lot_mult_enabled
                        FROM config_snapshots ORDER BY id DESC LIMIT 1
                    """).fetchone()
                    prev_dict = {k: prev[i] for i, k in enumerate([
                        "risk_per_trade_pct","sl_atr_mult","tp_atr_mult",
                        "smart_trailing_enabled","be_trigger_atr",
                        "series_loss_cap_pct","series_loss_cap_action",
                        "hourly_lot_mult_enabled"])} if prev else None
                    auto_label = _build_diff_label(prev_dict, fields, cfg)
                    cur = conn.execute("""
                        INSERT INTO config_snapshots
                          (ts_utc, label, config_hash,
                           risk_per_trade_pct, sl_atr_mult, tp_atr_mult,
                           smart_trailing_enabled, be_trigger_atr,
                           series_loss_cap_pct, series_loss_cap_action,
                           hourly_lot_mult_enabled, full_config_json)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        datetime.now(timezone.utc).isoformat(),
                        auto_label, cur_hash,
                        fields["risk_per_trade_pct"], fields["sl_atr_mult"],
                        fields["tp_atr_mult"], fields["smart_trailing_enabled"],
                        fields["be_trigger_atr"], fields["series_loss_cap_pct"],
                        fields["series_loss_cap_action"], fields["hourly_lot_mult_enabled"],
                        json.dumps(cfg, sort_keys=True),
                    ))
                    sid = int(cur.lastrowid)
                    conn.commit()
                    log.info("[snapshot] 🆕 NEW snapshot id=%d hash=%s | %s",
                             sid, cur_hash, auto_label)

            _CACHED_ID = sid
            _CACHED_HASH = cur_hash
            _LAST_CHECK_TS = now
            return sid
        except Exception as e:
            log.warning("[snapshot] failed to detect: %s", e)
            return _CACHED_ID
