@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo.
echo ================================================================
echo   🤖 SweepHunter Bot - START
echo ================================================================
echo.
python run.py bot
echo.
echo Bot stopped. Press any key to close...
pause >nul
