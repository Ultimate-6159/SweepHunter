@echo off
REM 🧹 Reset SweepHunter — ลบ DB + model + state แล้วพร้อมรีเทรนใหม่
chcp 65001 >nul
cd /d "%~dp0"
python reset_all.py
pause
