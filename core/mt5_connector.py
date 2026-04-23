"""
MT5 Connector - เชื่อมต่อ MT5 และ auto-detect คุณสมบัติของ symbol
(digits, point, trade_tick_size, contract_size, stops_level, filling_mode, ฯลฯ)
ไม่มี hardcoded ค่าใดๆ ที่ specific กับโบรกเกอร์.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import time

import MetaTrader5 as mt5

from .config import Config
from .logger import get_logger

log = get_logger("mt5")


# Map MT5 timeframe strings -> constants
TIMEFRAME_MAP = {
    "M1": mt5.TIMEFRAME_M1, "M2": mt5.TIMEFRAME_M2, "M3": mt5.TIMEFRAME_M3,
    "M4": mt5.TIMEFRAME_M4, "M5": mt5.TIMEFRAME_M5, "M6": mt5.TIMEFRAME_M6,
    "M10": mt5.TIMEFRAME_M10, "M12": mt5.TIMEFRAME_M12, "M15": mt5.TIMEFRAME_M15,
    "M20": mt5.TIMEFRAME_M20, "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1, "H2": mt5.TIMEFRAME_H2, "H3": mt5.TIMEFRAME_H3,
    "H4": mt5.TIMEFRAME_H4, "H6": mt5.TIMEFRAME_H6, "H8": mt5.TIMEFRAME_H8,
    "H12": mt5.TIMEFRAME_H12,
    "D1": mt5.TIMEFRAME_D1, "W1": mt5.TIMEFRAME_W1, "MN1": mt5.TIMEFRAME_MN1,
}


@dataclass
class SymbolSpec:
    """ค่าทุกอย่างของ symbol ที่ระบบต้องใช้คำนวณ - ดึงจาก MT5 ทั้งหมด."""
    name: str
    digits: int
    point: float
    trade_tick_size: float
    trade_tick_value: float
    trade_contract_size: float
    volume_min: float
    volume_max: float
    volume_step: float
    stops_level: int          # min stop distance in points
    freeze_level: int
    spread: int               # current spread in points
    filling_mode: int         # bitmask
    raw: Dict[str, Any] = field(default_factory=dict)

    def points_to_price(self, points: float) -> float:
        return points * self.point

    def price_to_points(self, price_diff: float) -> float:
        return price_diff / self.point if self.point else 0.0

    def normalize_price(self, price: float) -> float:
        return round(price, self.digits)

    def normalize_volume(self, volume: float) -> float:
        if self.volume_step <= 0:
            return max(self.volume_min, volume)
        steps = round((volume - self.volume_min) / self.volume_step)
        v = self.volume_min + steps * self.volume_step
        v = max(self.volume_min, min(self.volume_max, v))
        return round(v, 2)

    def preferred_filling(self) -> int:
        """เลือก filling mode ที่โบรกเกอร์รองรับ (ให้ความสำคัญ IOC ตาม SRD)."""
        # SYMBOL_FILLING_FOK = 1, SYMBOL_FILLING_IOC = 2
        if self.filling_mode & 2:
            return mt5.ORDER_FILLING_IOC
        if self.filling_mode & 1:
            return mt5.ORDER_FILLING_FOK
        return mt5.ORDER_FILLING_RETURN


class MT5Connector:
    """Singleton wrapper เชื่อม MT5 + cache SymbolSpec."""
    _initialised: bool = False
    _spec_cache: Dict[str, SymbolSpec] = {}

    @classmethod
    def initialise(cls) -> bool:
        if cls._initialised:
            return True
        cfg = Config.section("mt5")
        kwargs: Dict[str, Any] = {}
        if cfg.get("terminal_path"):
            kwargs["path"] = cfg["terminal_path"]
        if cfg.get("login"):
            kwargs["login"] = int(cfg["login"])
            kwargs["password"] = cfg.get("password", "")
            kwargs["server"] = cfg.get("server", "")

        if not mt5.initialize(**kwargs):
            log.error("MT5 initialise failed: %s", mt5.last_error())
            return False
        info = mt5.account_info()
        if info is None:
            log.error("MT5 account_info None: %s", mt5.last_error())
            mt5.shutdown()
            return False
        log.info("MT5 connected. account=%s server=%s balance=%.2f",
                 info.login, info.server, info.balance)
        cls._initialised = True
        return True

    @classmethod
    def shutdown(cls) -> None:
        if cls._initialised:
            mt5.shutdown()
            cls._initialised = False

    # cache จาก base name -> resolved broker name (เช่น "XAUUSD" -> "XAUUSDm")
    _symbol_alias: Dict[str, str] = {}

    @classmethod
    def resolve_symbol(cls, base: str) -> str:
        """
        แปลงชื่อ symbol ของผู้ใช้ -> ชื่อจริงของโบรกเกอร์ (รองรับ suffix เช่น m,z,c,.,#)
        ใช้ mt5.symbols_get() ค้นหาทั้งหมดแล้วเลือกตัวที่ match ดีที่สุด.
        """
        if base in cls._symbol_alias:
            return cls._symbol_alias[base]

        # 1) ลองตรงๆ ก่อน
        info = mt5.symbol_info(base)
        if info is not None:
            cls._symbol_alias[base] = info.name
            return info.name

        # 2) สแกน symbols ทั้งหมด หา candidate ที่ขึ้นต้นด้วย base
        all_syms = mt5.symbols_get() or []
        base_u = base.upper()
        candidates: list[tuple[int, str]] = []
        for s in all_syms:
            n_u = s.name.upper()
            if n_u == base_u:
                candidates.append((0, s.name))
            elif n_u.startswith(base_u):
                # priority by suffix length (สั้น = ใกล้เคียงสุด)
                candidates.append((len(s.name) - len(base), s.name))
            elif base_u in n_u:
                candidates.append((100 + len(s.name), s.name))

        if not candidates:
            raise RuntimeError(
                f"No symbol matches '{base}' on this broker. "
                f"Open MarketWatch in MT5 and check the exact symbol name."
            )

        candidates.sort(key=lambda x: x[0])
        resolved = candidates[0][1]
        if resolved != base:
            log.info("Symbol auto-resolved: '%s' -> '%s'", base, resolved)
        cls._symbol_alias[base] = resolved
        return resolved

    @classmethod
    def get_symbol_spec(cls, symbol: str, refresh_spread: bool = True) -> SymbolSpec:
        resolved = cls.resolve_symbol(symbol)
        if resolved in cls._spec_cache and not refresh_spread:
            return cls._spec_cache[resolved]

        if not mt5.symbol_select(resolved, True):
            raise RuntimeError(f"Cannot select symbol {resolved}: {mt5.last_error()}")

        info = mt5.symbol_info(resolved)
        if info is None:
            raise RuntimeError(f"symbol_info({resolved}) is None: {mt5.last_error()}")

        spec = SymbolSpec(
            name=info.name,
            digits=info.digits,
            point=info.point,
            trade_tick_size=info.trade_tick_size,
            trade_tick_value=info.trade_tick_value,
            trade_contract_size=info.trade_contract_size,
            volume_min=info.volume_min,
            volume_max=info.volume_max,
            volume_step=info.volume_step,
            stops_level=info.trade_stops_level,
            freeze_level=info.trade_freeze_level,
            spread=info.spread,
            filling_mode=info.filling_mode,
            raw={k: getattr(info, k) for k in dir(info) if not k.startswith("_")
                 and not callable(getattr(info, k, None))},
        )
        cls._spec_cache[resolved] = spec
        return spec

    @classmethod
    def get_tick(cls, symbol: str):
        return mt5.symbol_info_tick(cls.resolve_symbol(symbol))

    @classmethod
    def copy_rates(cls, symbol: str, timeframe: str, count: int):
        """
        ดึงแท่งย้อนหลังจาก position 0.
        - ใช้สำหรับ inference loop (count เล็ก เช่น 200)
        - ถ้า count ใหญ่มาก (>50000) ให้ใช้ load_history แทน
        """
        tf = TIMEFRAME_MAP.get(timeframe.upper())
        if tf is None:
            raise ValueError(f"Unknown timeframe: {timeframe}")
        resolved = cls.resolve_symbol(symbol)
        for attempt in range(3):
            rates = mt5.copy_rates_from_pos(resolved, tf, 0, count)
            if rates is not None and len(rates) >= max(2, count - 1):
                return rates
            time.sleep(0.2)
        raise RuntimeError(f"copy_rates_from_pos failed for {resolved}/{timeframe}: {mt5.last_error()}")

    @classmethod
    def load_history(cls, symbol: str, timeframe: str, bars: int):
        """
        โหลดประวัติย้อนหลังจำนวนมากแบบ chunked ผ่าน copy_rates_from_pos.
        - เลี่ยง datetime arg ที่มี edge case "Invalid params" บนบางโบรกเกอร์
        - ไล่ start_pos เพิ่มทีละ chunk_size จนกว่าจะครบ หรือไม่มีข้อมูลแล้ว
        - คืน numpy structured array เรียงตามเวลาเก่า -> ใหม่
        """
        import numpy as np
        tf = TIMEFRAME_MAP.get(timeframe.upper())
        if tf is None:
            raise ValueError(f"Unknown timeframe: {timeframe}")
        resolved = cls.resolve_symbol(symbol)
        bars = int(bars)

        log.info("Loading history: %s %s bars=%d (chunked from_pos)...",
                 resolved, timeframe, bars)

        chunk_size = 10_000
        all_chunks = []
        start_pos = 0
        loaded = 0
        empty_streak = 0
        max_empty = 3

        while loaded < bars and empty_streak < max_empty:
            need = min(chunk_size, bars - loaded)
            r = mt5.copy_rates_from_pos(resolved, tf, start_pos, need)

            if r is None or len(r) == 0:
                empty_streak += 1
                err = mt5.last_error()
                # ถ้าเริ่มต้นเลย fail = ปัญหาจริง; ถ้าระหว่างทาง = หมด history
                if start_pos == 0:
                    log.warning("  initial chunk empty err=%s — retrying smaller...", err)
                    chunk_size = max(1000, chunk_size // 2)
                    if chunk_size < 1000:
                        break
                else:
                    log.info("  no more history at start_pos=%d (loaded %d bars)", start_pos, loaded)
                    break
                time.sleep(0.3)
                continue

            empty_streak = 0
            all_chunks.append(r)
            loaded += len(r)
            start_pos += len(r)
            if loaded % 50_000 < chunk_size:
                log.info("  ... loaded %d / %d bars", loaded, bars)

            # ถ้าได้น้อยกว่าที่ขอ = ใกล้หมด history แล้ว
            if len(r) < need:
                log.info("  broker returned %d < requested %d — end of history", len(r), need)
                break

        if not all_chunks:
            raise RuntimeError(
                f"load_history returned 0 bars: {mt5.last_error()}.\n"
                f"แก้: เปิด chart {resolved}/{timeframe} ใน MT5 terminal "
                f"แล้วกด End + PageUp หลายครั้ง เพื่อบังคับ download history เพิ่ม."
            )

        # chunks ถูกดึงจากใหม่ -> เก่า, ต้อง reverse แล้ว concat
        merged = np.concatenate(all_chunks[::-1])
        # ลบแถวซ้ำ (กันกรณี chunk overlap) + เรียงเวลา
        _, uniq_idx = np.unique(merged["time"], return_index=True)
        merged = merged[np.sort(uniq_idx)]
        if len(merged) > bars:
            merged = merged[-bars:]
        log.info("Total history loaded: %d bars (requested %d)", len(merged), bars)
        return merged




    @classmethod
    def positions(cls, symbol: Optional[str] = None, magic: Optional[int] = None):
        resolved = cls.resolve_symbol(symbol) if symbol else None
        pos = mt5.positions_get(symbol=resolved) if resolved else mt5.positions_get()
        if pos is None:
            return []
        if magic is not None:
            pos = [p for p in pos if p.magic == magic]
        return list(pos)

    @classmethod
    def history_deals(cls, date_from, date_to):
        deals = mt5.history_deals_get(date_from, date_to)
        return list(deals) if deals else []
