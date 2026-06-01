@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
echo.
echo === SweepHunter DB Reliability Report ===
echo.
python db_reliability.py %*
echo.
pause
