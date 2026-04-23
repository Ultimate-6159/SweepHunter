"""
News Filter Engine - ดึง XML จาก Forex Factory และบล็อกการเทรดรอบๆ ข่าว High Impact.
- Cache ไว้ใน data/cache/news.xml
- Refresh ทุกๆ N ชั่วโมงตาม config
- ตรวจสอบ block window: now in [event - X min, event + Y min]
"""
from __future__ import annotations
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from typing import List, Tuple, Optional

import requests
from dateutil import parser as dtparser

from .config import Config
from .logger import get_logger
from .paths import cache_path

log = get_logger("news")


class NewsFilter:
    def __init__(self) -> None:
        self.events: List[Tuple[datetime, str, str, str]] = []  # (utc_time, currency, impact, title)
        self.last_refresh: Optional[datetime] = None
        self.cache_file = cache_path("news.xml")

    def _refresh_needed(self) -> bool:
        cfg = Config.section("news_filter")
        hrs = float(cfg.get("refresh_hours", 6))
        if self.last_refresh is None:
            return True
        return (datetime.now(timezone.utc) - self.last_refresh) > timedelta(hours=hrs)

    def refresh(self, force: bool = False) -> None:
        cfg = Config.section("news_filter")
        if not cfg.get("enabled", True):
            return
        if not force and not self._refresh_needed():
            return

        url = cfg.get("url")
        try:
            resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            self.cache_file.write_bytes(resp.content)
            log.info("News XML refreshed (%d bytes)", len(resp.content))
        except Exception as e:
            log.warning("News refresh failed: %s (using cached if exists)", e)
            if not self.cache_file.exists():
                self.events = []
                return

        try:
            self._parse(self.cache_file.read_bytes())
            self.last_refresh = datetime.now(timezone.utc)
        except Exception as e:
            log.error("News parse error: %s", e)
            self.events = []

    def _parse(self, content: bytes) -> None:
        events: List[Tuple[datetime, str, str, str]] = []
        root = ET.fromstring(content)
        for ev in root.findall("event"):
            try:
                title = (ev.findtext("title") or "").strip()
                country = (ev.findtext("country") or "").strip().upper()
                impact = (ev.findtext("impact") or "").strip()
                date_s = (ev.findtext("date") or "").strip()
                time_s = (ev.findtext("time") or "").strip()
                if not country or not impact or not date_s:
                    continue
                # FF XML มี date+time แยก, time อาจเป็น "All Day" -> ข้าม
                if "day" in time_s.lower() or not time_s:
                    continue
                dt = dtparser.parse(f"{date_s} {time_s}")
                # FF XML เป็น Eastern Time (US) -> assume UTC ถ้าไม่มี tz; safer: treat as UTC offset from header
                # เนื่องจาก FF feed default เป็น EST/EDT; แต่ feed nfs.faireconomy ส่วนใหญ่จะใส่ UTC offset แล้ว
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                else:
                    dt = dt.astimezone(timezone.utc)
                events.append((dt, country, impact, title))
            except Exception:
                continue
        self.events = events
        log.info("Parsed %d news events", len(events))

    def is_blocked(self, now: Optional[datetime] = None) -> Tuple[bool, str]:
        cfg = Config.section("news_filter")
        if not cfg.get("enabled", True):
            return False, ""
        self.refresh()
        if not self.events:
            return False, ""

        now = now or datetime.now(timezone.utc)
        before = timedelta(minutes=int(cfg.get("block_minutes_before", 15)))
        after = timedelta(minutes=int(cfg.get("block_minutes_after", 15)))
        currencies = {c.upper() for c in cfg.get("currencies", ["USD"])}
        impacts = {i.lower() for i in cfg.get("impact_levels", ["High"])}

        for ev_time, ccy, impact, title in self.events:
            if ccy not in currencies:
                continue
            if impact.lower() not in impacts:
                continue
            if ev_time - before <= now <= ev_time + after:
                return True, f"{ccy} {impact}: {title} @ {ev_time.isoformat()}"
        return False, ""
