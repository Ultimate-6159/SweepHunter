@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8

if exist .python_path.txt (
  set /p PYEXE=<.python_path.txt
) else (
  set PYEXE=python
)

echo.
echo === วิเคราะห์ Lot Multiplier วัน x ชั่วโมง (snapshot ^>= 14) ===
echo.
echo   ดูรายงานอย่างเดียว (ไม่ถามแก้ config):
echo     analyze_lot_multipliers.bat --dry-run
echo.
echo   สร้างรายงาน + ถามว่าจะอัปเดต config:
echo     analyze_lot_multipliers.bat
echo.
"%PYEXE%" analyze_lot_multipliers.py --since-snapshot=14 %*
echo.
pause
