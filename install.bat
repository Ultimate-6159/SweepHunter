@echo off
REM ============================================================
REM  SweepHunter - Install dependencies
REM  Portable: ใช้ path ของไฟล์นี้เป็นฐาน ไม่ขึ้นกับ working dir
REM ============================================================
setlocal
cd /d "%~dp0"

echo.
echo === [SweepHunter] Installing dependencies ===
echo Working dir: %CD%
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found in PATH. Please install Python 3.10+.
    pause
    exit /b 1
)

python -m pip install --upgrade pip
if errorlevel 1 goto :fail

python -m pip install -r requirements.txt
if errorlevel 1 goto :fail

echo.
echo === Install completed successfully ===
pause
exit /b 0

:fail
echo.
echo [ERROR] Install failed.
pause
exit /b 1
