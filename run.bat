@echo off
chcp 65001 >nul
cd /d "%~dp0"
if exist "%~dp0.python_path.txt" (set /p PYEXE=<"%~dp0.python_path.txt") else (set PYEXE=python)
"%PYEXE%" run.py %*
