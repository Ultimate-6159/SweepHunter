@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
echo.
echo === analyze_slots (ข้อมูลทั้งหมดใน DB — รวมยุดเก่า) ===
echo     แนะนำใช้ analyze_slots_current.bat แทน (ยุด config ปัจจุบัน)
echo.
python analyze_slots.py %*
echo.
pause
