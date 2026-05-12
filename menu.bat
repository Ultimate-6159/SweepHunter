@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo.
echo ================================================================
echo   🎮 SweepHunter Control Panel
echo ================================================================
echo.
echo   1. ▶️  Start Bot              (run.py bot)
echo   2. 📊 Data Status            (overview dashboard)
echo   3. 🧠 Retrain AI Model       (~3 min)
echo   4. 🗃️  Retrain History
echo   5. 📝 Add Snapshot Note
echo   6. 📈 Compare Strategies     (A/B snapshot test)
echo   7. ⚡ Status (last 24h only)
echo   8. 🛠️  Open config.json
echo   9. 🚪 Exit
echo.
set /p CHOICE="Choose [1-9]: "

if "%CHOICE%"=="1" (call 1_start_bot.bat & goto :eof)
if "%CHOICE%"=="2" (call 2_status.bat & goto :menu)
if "%CHOICE%"=="3" (call 3_retrain.bat & goto :menu)
if "%CHOICE%"=="4" (call 4_retrain_history.bat & goto :menu)
if "%CHOICE%"=="5" (call 5_add_note.bat & goto :menu)
if "%CHOICE%"=="6" (python compare_strategies.py & pause & goto :menu)
if "%CHOICE%"=="7" (python data_status.py status --since 24h & pause & goto :menu)
if "%CHOICE%"=="8" (start notepad config.json & goto :menu)
if "%CHOICE%"=="9" (exit /b)

echo Invalid choice.
pause

:menu
%~f0
