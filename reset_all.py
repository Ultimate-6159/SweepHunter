"""
🧹 reset_all.py — ล้างข้อมูลทั้งหมด + เตรียมรีเทรนใหม่จากศูนย์

ทำสิ่งเหล่านี้:
  1. Backup DB (hyper_trades.sqlite → hyper_trades_<ts>.sqlite.bak)
  2. Backup model (xgb_hyper_model.pkl → archive/<ts>/)
  3. Backup recovery_state.json
  4. ลบ DB / model / recovery_state / cache
  5. แสดงคำสั่งรันต่อ

⚠️ ใช้เมื่อต้องการเริ่มใหม่หมดเท่านั้น — ข้อมูลเทรดทั้งหมดจะหายไป!
"""
from __future__ import annotations
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DB_DIR = DATA / "db"
MODEL_DIR = DATA / "models"
LOG_DIR = DATA / "logs"
CACHE_DIR = DATA / "cache"

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
BACKUP = DATA / "archive" / f"reset_{ts}"


def confirm() -> bool:
    print("=" * 70)
    print("🧹 SweepHunter — RESET ALL (ลบข้อมูล + รีเทรนใหม่ทั้งหมด)")
    print("=" * 70)
    print(f"\n📦 จะ backup ไปที่: {BACKUP}\n")
    print("จะลบสิ่งต่อไปนี้:")
    if (DB_DIR / "hyper_trades.sqlite").exists():
        sz = (DB_DIR / "hyper_trades.sqlite").stat().st_size / 1024
        print(f"  ❌ DB: hyper_trades.sqlite ({sz:.0f} KB)")
    if (MODEL_DIR / "xgb_hyper_model.pkl").exists():
        sz = (MODEL_DIR / "xgb_hyper_model.pkl").stat().st_size / 1024
        print(f"  ❌ Model: xgb_hyper_model.pkl ({sz:.0f} KB)")
    rec = DATA / "recovery_state.json"
    if rec.exists():
        print(f"  ❌ Recovery state: recovery_state.json")
    if CACHE_DIR.exists():
        n = len(list(CACHE_DIR.glob("*")))
        if n:
            print(f"  ❌ Cache: {n} ไฟล์")
    print("\n⚠️  คำเตือน: ข้อมูลเทรดทั้งหมด (621+ ไม้) จะหายไป!")
    print("   หลังจากนี้ต้อง:")
    print("     1. python run.py train     # train model ใหม่จาก historical")
    print("     2. python run.py bot       # เริ่มเทรดสด")
    ans = input("\nพิมพ์ 'YES' เพื่อยืนยัน (อย่างอื่น = ยกเลิก): ").strip()
    return ans == "YES"


def backup_then_delete():
    BACKUP.mkdir(parents=True, exist_ok=True)
    actions = []

    # 1. DB
    db = DB_DIR / "hyper_trades.sqlite"
    if db.exists():
        shutil.copy2(db, BACKUP / "hyper_trades.sqlite")
        db.unlink()
        actions.append(f"✅ DB backed up + deleted")

    # 2. Model + meta
    for fname in ["xgb_hyper_model.pkl", "model_meta.json", "xgb_hyper_model_rejected.json"]:
        f = MODEL_DIR / fname
        if f.exists():
            shutil.copy2(f, BACKUP / fname)
            f.unlink()
            actions.append(f"✅ {fname} backed up + deleted")

    # 3. recovery_state
    rec = DATA / "recovery_state.json"
    if rec.exists():
        shutil.copy2(rec, BACKUP / "recovery_state.json")
        rec.unlink()
        actions.append("✅ recovery_state.json backed up + deleted")

    # 4. cache
    if CACHE_DIR.exists():
        cache_bk = BACKUP / "cache"
        cache_bk.mkdir(exist_ok=True)
        n = 0
        for f in CACHE_DIR.glob("*"):
            if f.is_file():
                shutil.copy2(f, cache_bk / f.name)
                f.unlink()
                n += 1
        if n:
            actions.append(f"✅ cache cleared ({n} ไฟล์)")

    return actions


def main() -> int:
    if not confirm():
        print("\n❌ ยกเลิก — ไม่มีอะไรเปลี่ยนแปลง")
        return 1
    print("\n🧹 กำลังลบ...")
    actions = backup_then_delete()
    print()
    for a in actions:
        print(f"  {a}")
    print(f"\n📦 Backup เก็บที่: {BACKUP}")
    print("\n" + "=" * 70)
    print("✨ Reset สำเร็จ! ขั้นตอนต่อไป:")
    print("=" * 70)
    print("  1. python run.py train     # train model ใหม่ (~5-15 นาที)")
    print("  2. python run.py bot       # เริ่มเทรด")
    print("\n💡 หลัง train เสร็จ ให้เช็ค model_meta.json ว่า oos_test_acc > 0.45")
    return 0


if __name__ == "__main__":
    sys.exit(main())
