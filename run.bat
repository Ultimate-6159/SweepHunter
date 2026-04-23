@echo off
REM ============================================================
REM  SweepHunter - Run live trading bot (zero-touch loop)
REM  Auto-restart ถ้า process crash (เพื่อ Windows VPS deployment)
REM ============================================================
setlocal
cd /d "%~dp0"

title SweepHunter Bot - %CD%

:loop
echo.
echo === [SweepHunter] Starting bot at %DATE% %TIME% ===
echo.
python run.py bot
set EC=%ERRORLEVEL%

echo.
echo === Bot exited with code %EC% at %DATE% %TIME% ===
if "%EC%"=="0" (
    echo Clean shutdown. Exiting.
    exit /b 0
)

echo Restarting in 10 seconds... (Ctrl+C to abort)
timeout /t 10 /nobreak >nul
goto loop
