"""
Portable path resolver.
ทุก path ในระบบ resolve จากตำแหน่งของไฟล์นี้เอง -> ไม่ต้อง hardcode path
สามารถ pack zip ไปวางที่ไหนก็รันได้ทันที.
"""
from __future__ import annotations
from pathlib import Path

# Project root = parent ของ /core
ROOT: Path = Path(__file__).resolve().parent.parent

CONFIG_FILE: Path = ROOT / "config.json"
DATA_DIR: Path = ROOT / "data"
MODELS_DIR: Path = DATA_DIR / "models"
LOGS_DIR: Path = DATA_DIR / "logs"
DB_DIR: Path = DATA_DIR / "db"
CACHE_DIR: Path = DATA_DIR / "cache"


def ensure_dirs() -> None:
    """สร้างทุก directory ที่จำเป็น (idempotent)."""
    for d in (DATA_DIR, MODELS_DIR, LOGS_DIR, DB_DIR, CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)


def model_path(filename: str) -> Path:
    return MODELS_DIR / filename


def db_path(filename: str) -> Path:
    return DB_DIR / filename


def log_path(filename: str) -> Path:
    return LOGS_DIR / filename


def cache_path(filename: str) -> Path:
    return CACHE_DIR / filename


# auto-create on import เพื่อให้ทุก module ใช้ได้ทันที
ensure_dirs()
