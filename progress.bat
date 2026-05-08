@echo off
REM ================================================
REM  SweepHunter — Quick Progress Check
REM ================================================
cd /d "%~dp0"
chcp 65001 > nul
set PYTHONIOENCODING=utf-8
python progress.py
echo.
pause
