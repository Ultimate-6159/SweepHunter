"""
Single entry point - portable.
Usage:
    python run.py train         # train hyper M1 model from MT5 history
    python run.py train5        # multi-seed retrain (5 seeds, keep best)
    python run.py bot           # run live Hyper-Frequency + Martingale loop
    python run.py status        # quick health check
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
    if meta.get("rejected"):
        log.warning("RETRAIN REJECTED: %s | new_acc=%.4f old_acc=%.4f → keep old model",
                    meta.get("reason", "?"),
                    meta.get("new_test_acc", 0.0),
                    meta.get("old_test_acc", 0.0))
        return  # ไม่ rollback — model เก่ายังอยู่
    log.info("TRAINED: oos_acc=%.4f rows=%d", meta["oos_test_acc"], meta["rows_trained"])


def cmd_train5() -> None:
    """Multi-seed retrain: รัน N seeds, โค้ดภายในเลือก seed ที่ดีที่สุดผ่าน champion gate."""
    import time
    from core.model_trainer import train_from_mt5
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    log.info("Multi-seed retrain: %d seeds", n_seeds)
    accepted = 0
    rejected = 0
    for i in range(1, n_seeds + 1):
        log.info("=" * 60)
        log.info("SEED RUN %d/%d", i, n_seeds)
        log.info("=" * 60)
        meta = train_from_mt5()
        if meta.get("rejected"):
            rejected += 1
            log.info("Seed %d: REJECTED (%s)", i, meta.get("reason", "?")[:80])
        else:
            accepted += 1
            pm = meta.get("profit_metrics_oos") or {}
            log.info("Seed %d: ACCEPTED — PF=%.2f WR=%.1f%% net=%.2f ATRs thr=%.2f",
                     i, pm.get("profit_factor", 0), pm.get("win_rate", 0) * 100,
                     pm.get("net_pnl_atrs", 0), meta.get("best_threshold", 0))
        if i < n_seeds:
            time.sleep(1)   # ให้ timestamp seed ต่างกัน
    log.info("Multi-seed done: %d accepted / %d rejected / %d total",
             accepted, rejected, n_seeds)


def cmd_bot() -> None:
    import os
    import atexit
    from core.paths import ROOT

    lock_file = ROOT / "bot.lock"

    # ป้องกัน instance ซ้ำ — ถ้ามี lock file และ PID ยังรันอยู่ → หยุดทันที
    if lock_file.exists():
        try:
            old_pid = int(lock_file.read_text().strip())
            # ตรวจสอบว่า process นั้นยังรันอยู่หรือเปล่า
            import psutil
            if psutil.pid_exists(old_pid):
                proc = psutil.Process(old_pid)
                if proc.is_running() and "python" in proc.name().lower():
                    log.error("🚫 Bot instance already running (PID=%d)! ยกเลิก — ห้ามรัน 2 instance พร้อมกัน", old_pid)
                    sys.exit(1)
        except Exception:
            pass  # stale lock → proceed

    # เขียน PID ปัจจุบัน
    lock_file.write_text(str(os.getpid()))

    def _remove_lock():
        try:
            lock_file.unlink(missing_ok=True)
        except Exception:
            pass

    atexit.register(_remove_lock)

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


COMMANDS = {"train": cmd_train, "train5": cmd_train5, "bot": cmd_bot, "status": cmd_status}


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "bot"
    if cmd not in COMMANDS:
        print(f"Unknown command: {cmd}\nUsage: python run.py [{'|'.join(COMMANDS)}]")
        sys.exit(1)
    COMMANDS[cmd]()


if __name__ == "__main__":
    main()
