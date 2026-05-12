@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo.
echo ================================================================
echo   📝 Add Note to Snapshot
echo ================================================================
echo.
set /p SID="Enter snapshot ID: "
set /p TEXT="Enter note text: "
echo.
python data_status.py note %SID% "%TEXT%"
echo.
pause
