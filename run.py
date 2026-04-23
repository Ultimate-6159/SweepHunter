"""
Single entry point - portable.
Usage:
    python run.py train    # train hyper M1 model from MT5 history
    python run.py bot      # run live Hyper-Frequency + Martingale loop
    python run.py status   # quick health check
"""
from __future__ import annotations
import sys
from core.config import Config
from core.logger import get_logger
from core.paths import ROOT, MODELS_DIR, DB_DIR, LOGS_DIR

log = get_logger("entry")


def cmd_train() -> None:
    from core.model_trainer import train_from_mt5
    meta = train_from_mt5()
    log.info("TRAINED: oos_acc=%.4f rows=%d", meta["oos_test_acc"], meta["rows_trained"])


def cmd_bot() -> None:
    from core.xauusd_hyper_core import main
    main()


def cmd_status() -> None:
    from core.mt5_connector import MT5Connector
    Config.load()
    print(f"ROOT      : {ROOT}")
    print(f"MODELS    : {MODELS_DIR}")
    print(f"DB        : {DB_DIR}")
    print(f"LOGS      : {LOGS_DIR}")
    if MT5Connector.initialise():
        spec = MT5Connector.get_symbol_spec(Config.section("trading")["symbol"])
        print(f"SYMBOL    : {spec.name} digits={spec.digits} point={spec.point} "
              f"tick_size={spec.trade_tick_size} stops_level={spec.stops_level}p "
              f"filling_mask={spec.filling_mode} spread_now={spec.spread}p")
        MT5Connector.shutdown()


COMMANDS = {"train": cmd_train, "bot": cmd_bot, "status": cmd_status}


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "bot"
    if cmd not in COMMANDS:
        print(f"Unknown command: {cmd}\nUsage: python run.py [{'|'.join(COMMANDS)}]")
        sys.exit(1)
    COMMANDS[cmd]()


if __name__ == "__main__":
    main()
