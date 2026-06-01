@echo off
cd /d "%~dp0"
chcp 65001 > nul
set PYTHONIOENCODING=utf-8
if exist "%~dp0.python_path.txt" (set /p PYEXE=<"%~dp0.python_path.txt") else (set PYEXE=python)
title SweepHunter - Strategy Report (current era)
echo.
echo === Block Slots + Recovery Lot Report ===
echo === ยุด config ปัจจุบัน snapshot ^>= 14 ===
echo.
"%PYEXE%" generate_strategy_report.py --since-snapshot=14 %*
echo.
pause
