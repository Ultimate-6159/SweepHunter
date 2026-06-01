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
echo   6. ⏯️  Clear Halt             (ปลดล็อก bot)
echo   7. 📈 Compare Strategies     (A/B snapshot test)
echo   8. ⚡ Status (last 24h only)
echo   9. 🛠️  Open config.json
echo   A. 📋 View Trades Dashboard
echo   B. 📊 Strategy Report
echo   C. 🚫 Analyze Block Slots
echo   D. ⚖️  Lot Multiplier (วัน x ชม.)
echo   0. 🚪 Exit
echo.
set /p CHOICE="Choose [0-9 A-D]: "

if "%CHOICE%"=="1" (call 1_start_bot.bat & goto :eof)
if "%CHOICE%"=="2" (call 2_status.bat & goto :menu)
if "%CHOICE%"=="3" (call 3_retrain.bat & goto :menu)
if "%CHOICE%"=="4" (call 4_retrain_history.bat & goto :menu)
if "%CHOICE%"=="5" (call 5_add_note.bat & goto :menu)
if "%CHOICE%"=="6" (call 6_clear_halt.bat & goto :menu)
if "%CHOICE%"=="7" (python compare_strategies.py & pause & goto :menu)
if "%CHOICE%"=="8" (python data_status.py status --since 24h & pause & goto :menu)
if "%CHOICE%"=="9" (start notepad config.json & goto :menu)
if /I "%CHOICE%"=="A" (call view_trades.bat & goto :menu)
if /I "%CHOICE%"=="B" (call strategy_report.bat & goto :menu)
if /I "%CHOICE%"=="C" (call analyze_slots_current.bat & goto :menu)
if /I "%CHOICE%"=="D" (call analyze_lot_multipliers.bat & goto :menu)
if "%CHOICE%"=="0" (exit /b)

echo Invalid choice.
pause

:menu
%~f0
