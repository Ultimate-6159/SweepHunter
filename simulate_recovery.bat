@echo off
cd /d "%~dp0"
chcp 65001 > nul
if exist "%~dp0.python_path.txt" (set /p PYEXE=<"%~dp0.python_path.txt") else (set PYEXE=python)
"%PYEXE%" simulate_recovery.py
pause
