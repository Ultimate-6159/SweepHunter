"""Broker time helpers for HTML reports (heatmap «NOW» marker)."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

DOW_EN = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
DOW_TH = ("จ.", "อ.", "พ.", "พฤ.", "ศ.", "ส.", "อา.")


def broker_offset_from_config(config_path: str | Path = "config.json") -> int:
    try:
        p = Path(config_path)
        if p.exists():
            cfg = json.loads(p.read_text(encoding="utf-8"))
            return int((cfg.get("session_weighting") or {}).get("broker_offset_hours", 3))
    except Exception:
        pass
    return 3


def broker_now(offset_hours: int | None = None) -> datetime:
    off = offset_hours if offset_hours is not None else broker_offset_from_config()
    return datetime.now(timezone.utc) + timedelta(hours=off)


def now_slot(offset_hours: int | None = None) -> tuple[int, int]:
    bd = broker_now(offset_hours)
    return bd.weekday(), bd.hour


def now_slot_label(offset_hours: int | None = None) -> str:
    off = offset_hours if offset_hours is not None else broker_offset_from_config()
    dow, h = now_slot(off)
    return f"{DOW_EN[dow]} {h:02d}:xx broker (UTC+{off})"


def broker_slot_from_closed_at(
    closed_at: str,
    offset_hours: int | None = None,
) -> tuple[int, int] | None:
    """closed_at_utc (UTC จริง) → (weekday Mon=0, broker hour 0–23)."""
    if not closed_at:
        return None
    off = offset_hours if offset_hours is not None else broker_offset_from_config()
    s = str(closed_at).replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    bd = dt.astimezone(timezone.utc) + timedelta(hours=off)
    return bd.weekday(), bd.hour


def broker_slot_label(closed_at: str, offset_hours: int | None = None) -> str:
    slot = broker_slot_from_closed_at(closed_at, offset_hours)
    if slot is None:
        return "?"
    dow, h = slot
    return f"{DOW_EN[dow]} {h:02d}:xx"


def hm_cell_classes(
    dow: int | None,
    hour: int,
    cur_dow: int,
    cur_h: int,
    *,
    col_only: bool = False,
) -> str:
    """CSS classes for heatmap cell/column (col_only = hour-row or header)."""
    parts: list[str] = []
    if hour == cur_h:
        parts.append("hm-now-col")
    if not col_only and dow is not None and dow == cur_dow and hour == cur_h:
        parts.append("hm-now-cell")
    return " ".join(parts)


HM_NOW_CSS = """
.hm th.hm-now-col, .hm td.hm-now-col { outline: 2px solid #60a5fa; outline-offset: -2px; }
.hm tr.hm-now-row > th.dow { color: #60a5fa !important; font-weight: 700; }
.hm td.hm-now-cell { box-shadow: inset 0 0 0 3px #fbbf24 !important; position: relative; }
.hm td.hm-now-cell::before {
  content: 'NOW'; position: absolute; top: 0; left: 0; font-size: 6px;
  background: #fbbf24; color: #1c1917; padding: 0 3px; border-radius: 0 0 4px 0;
  font-weight: 700; line-height: 1.3; z-index: 1;
}
#heatmap-tbl .hm-now-col { outline: 2px solid #60a5fa; outline-offset: -1px; }
#heatmap-tbl tr.hm-now-row > td:first-child { color: #60a5fa !important; font-weight: 700; }
#heatmap-tbl .hm-now-cell { box-shadow: inset 0 0 0 3px #fbbf24 !important; position: relative; }
#heatmap-tbl .hm-now-cell::before {
  content: 'NOW'; position: absolute; top: 0; left: 0; font-size: 6px;
  background: #fbbf24; color: #1c1917; padding: 0 2px; font-weight: 700; z-index: 2;
}
.lm-hm th.hm-now-col, .lm-hm td.hm-now-col { outline: 2px solid #60a5fa; outline-offset: -1px; }
.lm-hm tr.lm-now-row > th.dow { color: #60a5fa !important; font-weight: 700; }
.lm-hm td.hm-now-cell { box-shadow: inset 0 0 0 2px #fbbf24 !important; position: relative; }
.lm-hm td.hm-now-cell::before {
  content: '▶'; position: absolute; top: 1px; right: 2px; font-size: 8px; color: #fbbf24;
}
"""

HM_NOW_LEGEND = (
    '<span><i class="swatch" style="outline:2px solid #60a5fa;background:#1e293b"></i> '
    "NOW = ชั่วโมง broker ปัจจุบัน</span>"
)
