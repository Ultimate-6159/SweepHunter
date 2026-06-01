@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
echo.
echo === วิเคราะห์ block ช่วงเวลา (ยุด config ปัจจุบัน snapshot ^>= 14) ===
echo.
echo   ดูอย่างเดียว (ไม่แก้ config):  analyze_slots_current.bat --dry-run
echo   อัปเดต config (ถามยืนยัน):    analyze_slots_current.bat
echo.
python analyze_slots.py --since-snapshot=14 %*
echo.
pause
