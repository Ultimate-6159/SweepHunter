@echo off
cd /d "%~dp0"
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
echo.
echo === แดชบอร์ดเทรด SweepHunter (snapshot ^>= 14) ===
echo.
python view_trades.py --since-snapshot=14 %*
if errorlevel 1 (
    echo.
    echo ERROR: Failed to generate dashboard
    pause
)
