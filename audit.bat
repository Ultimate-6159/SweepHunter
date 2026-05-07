@echo off
REM ============================================================
REM  audit.bat - SweepHunter Trading Audit (Pretty Edition)
REM ============================================================
REM  Usage:
REM    audit            -> ทุก trade
REM    audit 100        -> ล่าสุด 100 trades
REM    audit 550        -> ล่าสุด 550 trades
REM ============================================================

setlocal
chcp 65001 >nul 2>&1
cd /d "%~dp0"

if "%~1"=="" (
    python audit.py
) else (
    python audit.py %~1
)

echo.
pause
endlocal
