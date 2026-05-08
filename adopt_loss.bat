@echo off
REM ♻️ Adopt Loss — รับขาดทุนเก่าเข้าให้ bot กู้คืน
chcp 65001 >nul
cd /d "%~dp0"
python adopt_loss.py
pause
