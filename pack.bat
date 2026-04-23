@echo off
REM ============================================================
REM  SweepHunter - Pack project to portable ZIP
REM  Excludes: data/, __pycache__, .vs, .git, *.pyc, *.log, *.sqlite, *.zip
REM  Output: ..\SweepHunter_<yyyyMMdd_HHmmss>.zip
REM  Pure PowerShell (no wmic; works on Windows 11 24H2+)
REM ============================================================
setlocal
cd /d "%~dp0"

echo.
echo === [SweepHunter] Packing portable bundle ===
echo Source : %CD%
echo.

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0pack.ps1"
set EC=%ERRORLEVEL%

echo.
if "%EC%"=="0" (
    echo === Pack completed ===
    echo Copy the ZIP to any Windows VPS, unzip, run install.bat then run.bat.
) else (
    echo [ERROR] Pack failed with exit code %EC%.
)
pause
exit /b %EC%
