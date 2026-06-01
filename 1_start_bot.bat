@echo off
chcp 65001 >nul
cd /d "%~dp0"
if exist "%~dp0.python_path.txt" (set /p PYEXE=<"%~dp0.python_path.txt") else (set PYEXE=python)
title SweepHunter Bot
echo. & echo ================================ & echo   SweepHunter Bot [%date% %time%] & echo ================================ & echo.
:loop
"%PYEXE%" run.py bot
echo [%date% %time%] Bot exited - restart in 10s... (ปิดหน้าต่างนี้เพื่อหยุด)
timeout /t 10 /nobreak >nul & goto loop
