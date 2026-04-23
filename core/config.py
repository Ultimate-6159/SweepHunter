"""Config loader - อ่าน config.json แบบ portable."""
from __future__ import annotations
import json
from typing import Any, Dict
from .paths import CONFIG_FILE


class Config:
    _data: Dict[str, Any] = {}
    _loaded: bool = False

    @classmethod
    def load(cls, force: bool = False) -> Dict[str, Any]:
        if cls._loaded and not force:
            return cls._data
        if not CONFIG_FILE.exists():
            raise FileNotFoundError(f"Config file not found: {CONFIG_FILE}")
        with CONFIG_FILE.open("r", encoding="utf-8") as f:
            cls._data = json.load(f)
        cls._loaded = True
        return cls._data

    @classmethod
    def get(cls, *keys: str, default: Any = None) -> Any:
        d = cls.load()
        for k in keys:
            if not isinstance(d, dict) or k not in d:
                return default
            d = d[k]
        return d

    @classmethod
    def section(cls, name: str) -> Dict[str, Any]:
        return cls.load().get(name, {})


def get_config() -> Dict[str, Any]:
    return Config.load()
