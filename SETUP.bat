@echo off
chcp 65001 >nul 2>nul
setlocal EnableDelayedExpansion
cd /d "%~dp0"
title SweepHunter - Setup

echo.
echo ================================================================
echo   SweepHunter - ONE-CLICK SETUP
echo   Works on any Windows - just copy the folder and run this
echo ================================================================
echo   Folder: %CD%
echo.

echo [1/4] Checking Python...
set PYTHON_EXE=

for %%C in (python python3 py) do (
    if not defined PYTHON_EXE (
        %%C --version >nul 2>nul
        if not errorlevel 1 set PYTHON_EXE=%%C
    )
)

if not defined PYTHON_EXE (
    for %%P in (
        "%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
        "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
        "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
        "%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
        "C:\Python312\python.exe"
        "C:\Python311\python.exe"
        "C:\Program Files\Python312\python.exe"
    ) do (
        if not defined PYTHON_EXE if exist %%P set PYTHON_EXE=%%P
    )
)

if not defined PYTHON_EXE (
    echo.
    echo [ERROR] Python not found!
    echo Please install Python 3.10+ from: https://www.python.org/downloads/
    echo IMPORTANT: Check "Add Python to PATH" during install
    echo.
    pause & exit /b 1
)

for /f "tokens=2" %%V in ('"!PYTHON_EXE!" --version 2^>^&1') do set PY_VER=%%V
echo     Found Python !PY_VER! at: !PYTHON_EXE!
echo !PYTHON_EXE!> .python_path.txt
echo     OK

echo.
echo [2/4] Upgrading pip...
"!PYTHON_EXE!" -m pip install --upgrade pip --quiet 2>nul
echo     OK

echo.
echo [3/4] Installing packages (2-5 minutes)...
echo.
"!PYTHON_EXE!" -m pip install --upgrade "MetaTrader5>=5.0.45" "pandas>=2.0.0" "numpy>=1.24.0" "scikit-learn>=1.3.0" "xgboost>=2.0.0" "lightgbm>=4.0.0" "joblib>=1.3.0" "requests>=2.31.0" "python-dateutil>=2.8.2" "pytz>=2023.3"
if errorlevel 1 (
    echo [ERROR] Install failed - check internet, disable antivirus, try again
    pause & exit /b 1
)
echo     OK - All packages installed

echo.
echo [4/4] Writing launchers...

>"1_start_bot.bat" (
    echo @echo off
    echo chcp 65001 ^>nul
    echo cd /d "%%~dp0"
    echo if exist "%%~dp0.python_path.txt" ^(set /p PYEXE=^<"%%~dp0.python_path.txt"^) else ^(set PYEXE=python^)
    echo title SweepHunter Bot
    echo echo. ^& echo ================================ ^& echo   SweepHunter Bot [%%date%% %%time%%] ^& echo ================================ ^& echo.
    echo :loop
    echo "%%PYEXE%%" run.py bot
    echo set EC=%%ERRORLEVEL%%
    echo if "%%EC%%"=="0" ^(echo Clean shutdown. ^& pause ^& exit /b 0^)
    echo echo Crash ^(%%EC%%^) - restart in 10s... ^(Ctrl+C to stop^)
    echo timeout /t 10 /nobreak ^>nul ^& goto loop
)

>"3_retrain.bat" (
    echo @echo off
    echo chcp 65001 ^>nul
    echo cd /d "%%~dp0"
    echo if exist "%%~dp0.python_path.txt" ^(set /p PYEXE=^<"%%~dp0.python_path.txt"^) else ^(set PYEXE=python^)
    echo title Retrain
    echo echo === Retrain 5 seeds === ^& echo.
    echo "%%PYEXE%%" run.py train5 5
    echo echo. ^& pause
)

>"run.bat" (
    echo @echo off
    echo chcp 65001 ^>nul
    echo cd /d "%%~dp0"
    echo if exist "%%~dp0.python_path.txt" ^(set /p PYEXE=^<"%%~dp0.python_path.txt"^) else ^(set PYEXE=python^)
    echo "%%PYEXE%%" run.py %%*
)

echo     OK

echo.
echo [TEST] Testing imports...
"!PYTHON_EXE!" -c "import xgboost, lightgbm, pandas, numpy, sklearn, MetaTrader5; print('    OK - all imports pass')"
if errorlevel 1 (echo [WARNING] Some package issue - run SETUP.bat again & pause & exit /b 1)

echo.
echo ================================================================
echo   SETUP COMPLETE!
echo.
echo   1. Open MetaTrader 5 and login your account
echo   2. Edit config.json: fill in login / password / server
echo   3. Double-click 1_start_bot.bat to START the bot
echo   4. Double-click 3_retrain.bat to RETRAIN the model
echo ================================================================
echo.
pause
exit /b 0