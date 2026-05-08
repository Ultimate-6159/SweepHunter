@echo off
cd /d "%~dp0"
chcp 65001 > nul
set PYTHONIOENCODING=utf-8
python generate_report.py
echo.
pause
