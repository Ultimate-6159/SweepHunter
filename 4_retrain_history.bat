@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo.
echo ================================================================
echo   🗃️  Model Retrain History
echo ================================================================
echo.
python data_status.py retrains
echo.
pause
