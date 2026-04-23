@echo off
REM ============================================================
REM  SweepHunter - Train XGBoost model from MT5 history
REM  - Auto-backup ?????????????? data\models\archive\<timestamp>\
REM    ????? rollback ??? ???????????????????????
REM  - ?????? DB / adaptive state / logs (???????????????)
REM ============================================================
setlocal
cd /d "%~dp0"

echo.
echo === [SweepHunter] Training model ===
echo Working dir: %CD%
echo.

REM ---- Backup ????????? (?????) ----
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$src = Join-Path (Get-Location) 'data\models';" ^
  "if (Test-Path $src) {" ^
  "    $files = @('xgb_model.pkl','model_meta.json') | ForEach-Object { Join-Path $src $_ } | Where-Object { Test-Path $_ };" ^
  "    if ($files.Count -gt 0) {" ^
  "        $ts  = Get-Date -Format 'yyyyMMdd_HHmmss';" ^
  "        $arc = Join-Path $src ('archive\' + $ts);" ^
  "        New-Item -ItemType Directory -Path $arc -Force | Out-Null;" ^
  "        foreach ($f in $files) { Move-Item $f $arc -Force };" ^
  "        Write-Host ('  [backup] old model -> data\models\archive\' + $ts) -ForegroundColor Cyan" ^
  "    } else {" ^
  "        Write-Host '  [backup] no existing model to backup'" ^
  "    }" ^
  "}"

echo.
python run.py train
set EC=%ERRORLEVEL%

echo.
if "%EC%"=="0" (
    echo === Training completed ===
    echo Old model backed up in data\models\archive\ (rollback available).
) else (
    echo [ERROR] Training failed with exit code %EC%
    echo You can rollback by copying files from data\models\archive\^<timestamp^>\ back to data\models\
)
pause
exit /b %EC%
