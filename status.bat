@echo off
REM ============================================================
REM  SweepHunter - Quick status / health check
REM ============================================================
setlocal
cd /d "%~dp0"

python run.py status
pause
