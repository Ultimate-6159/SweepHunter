@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo.
echo ================================================================
echo   📊 Data Status Dashboard
echo ================================================================
echo.
python data_status.py
echo.
pause
