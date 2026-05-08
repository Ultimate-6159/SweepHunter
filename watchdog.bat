@echo off
REM ========================================================
REM  SweepHunter Watchdog - Auto-restart bot on crash
REM  Usage: double-click หรือ .\watchdog.bat
REM  Stop:  กด Ctrl+C สองครั้ง
REM ========================================================
setlocal
cd /d "%~dp0"
chcp 65001 > nul
set PYTHONIOENCODING=utf-8
set RESTART_DELAY=10
if not exist data\logs mkdir data\logs
set LOG=data\logs\watchdog.log

echo === SweepHunter Watchdog started ===
echo Bot will auto-restart on crash (delay %RESTART_DELAY%s)
echo Press Ctrl+C twice to stop permanently
echo.

:loop
echo [%date% %time%] === Starting bot ===
echo [%date% %time%] Starting bot >> "%LOG%"
python run.py bot
echo [%date% %time%] Bot exited code=%ERRORLEVEL%, restart in %RESTART_DELAY%s
echo [%date% %time%] Bot exited code=%ERRORLEVEL% >> "%LOG%"
timeout /t %RESTART_DELAY% /nobreak > nul
goto loop
