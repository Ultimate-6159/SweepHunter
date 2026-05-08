"""Smoke test trade-augmentation imports + DB query"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

from core.model_trainer import _load_real_trade_outcomes, _apply_trade_augmentation
print("[OK] Imports successful")

# Test DB query
df = _load_real_trade_outcomes("XAUUSD")
if df is None:
    print("[INFO] No DB data found")
else:
    print(f"[OK] Loaded {len(df)} closed trades from DB")
    print(f"     time range: {df['time'].min()}  ->  {df['time'].max()}")
    print(f"     outcomes: WIN={sum(df['outcome']=='WIN')} LOSS={sum(df['outcome']=='LOSS')}")
    print(f"     sides:    BUY(2)={sum(df['side']==2)} SELL(0)={sum(df['side']==0)}")
print("\n[DONE] Trade-augmentation ready (currently DISABLED in config)")
print("       To enable after 500 trades:")
print('       config.json -> "ai.trade_augmentation.enabled": true')
