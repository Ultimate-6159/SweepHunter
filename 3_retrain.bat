@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo.
echo ================================================================
echo   🧠 Train AI Model (Retrain)
echo ================================================================
echo.
echo ⚠️  WARNING: ใช้เวลา 2-5 นาที — อย่าปิดหน้าต่างนี้
echo.
echo Press any key to start...
pause >nul
echo.
python run.py train
echo.
echo ✅ Training done. Retrain event logged to DB.
echo.
pause
