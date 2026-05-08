@echo off
REM ============================================================
REM  SweepHunter - One-Click Retrain
REM  - Auto-detects model filenames from config.json
REM  - Backs up current model + metadata to data\models\archive\<ts>\
REM  - Runs training (python run.py train)
REM  - On failure: auto-rollback from latest archive
REM ============================================================
setlocal EnableDelayedExpansion
cd /d "%~dp0"

echo.
echo ============================================================
echo   [SweepHunter] One-Click Retrain
echo   Working dir: %CD%
echo ============================================================
echo.

REM ---- 1) Backup current model (read filenames from config.json) ----
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference='Stop';" ^
  "$cfg = Get-Content -Raw -Encoding UTF8 'config.json' | ConvertFrom-Json;" ^
  "$mdl = $cfg.ai.model_filename;    if (-not $mdl) { $mdl = 'xgb_hyper_model.pkl' };" ^
  "$mta = $cfg.ai.metadata_filename; if (-not $mta) { $mta = 'model_meta.json' };" ^
  "$src = Join-Path (Get-Location) 'data\models';" ^
  "if (-not (Test-Path $src)) { New-Item -ItemType Directory -Path $src -Force | Out-Null };" ^
  "$files = @($mdl,$mta) | ForEach-Object { Join-Path $src $_ } | Where-Object { Test-Path $_ };" ^
  "if ($files.Count -gt 0) {" ^
  "  $ts  = Get-Date -Format 'yyyyMMdd_HHmmss';" ^
  "  $arc = Join-Path $src ('archive\' + $ts);" ^
  "  New-Item -ItemType Directory -Path $arc -Force | Out-Null;" ^
  "  foreach ($f in $files) { Copy-Item $f $arc -Force };" ^
  "  $ts | Set-Content -Encoding ASCII (Join-Path $src '.last_backup');" ^
  "  Write-Host ('  [backup] -> data\models\archive\' + $ts) -ForegroundColor Cyan" ^
  "} else {" ^
  "  Write-Host '  [backup] no existing model to backup (first run)' -ForegroundColor Yellow" ^
  "}"

if errorlevel 1 (
    echo [ERROR] Backup step failed. Aborting.
    pause
    exit /b 1
)

REM ---- 2) Train ----
echo.
echo --- Running trainer ---
python run.py train
set EC=%ERRORLEVEL%
echo.

if "%EC%"=="0" goto :ok

REM ---- 3a) On failure: auto-rollback ----
echo [ERROR] Training failed (exit %EC%). Rolling back...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$src = Join-Path (Get-Location) 'data\models';" ^
  "$marker = Join-Path $src '.last_backup';" ^
  "if (Test-Path $marker) {" ^
  "  $ts  = (Get-Content $marker).Trim();" ^
  "  $arc = Join-Path $src ('archive\' + $ts);" ^
  "  if (Test-Path $arc) {" ^
  "    Get-ChildItem $arc | ForEach-Object { Copy-Item $_.FullName $src -Force };" ^
  "    Write-Host ('  [rollback] restored from archive\' + $ts) -ForegroundColor Yellow" ^
  "  } else { Write-Host '  [rollback] archive folder missing' -ForegroundColor Red }" ^
  "} else { Write-Host '  [rollback] no backup marker found' -ForegroundColor Red }"
echo.
pause
exit /b %EC%

:ok
REM ---- 3b) On success: print summary from model_meta.json ----
echo === Training completed ===
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$cfg = Get-Content -Raw -Encoding UTF8 'config.json' | ConvertFrom-Json;" ^
  "$mta = $cfg.ai.metadata_filename; if (-not $mta) { $mta = 'model_meta.json' };" ^
  "$path = Join-Path (Get-Location) ('data\models\' + $mta);" ^
  "if (Test-Path $path) {" ^
  "  $m = Get-Content -Raw $path | ConvertFrom-Json;" ^
  "  Write-Host '';" ^
  "  Write-Host '--- Model Summary ---' -ForegroundColor Green;" ^
  "  Write-Host ('  Symbol      : {0} {1}' -f $m.symbol, $m.timeframe);" ^
  "  Write-Host ('  Rows trained: {0}' -f $m.rows_trained);" ^
  "  Write-Host ('  CV acc      : {0:N4} +/- {1:N4}' -f $m.cv_acc_mean, $m.cv_acc_std);" ^
  "  Write-Host ('  Val acc     : {0:N4}' -f $m.val_acc);" ^
  "  Write-Host ('  OOS test acc: {0:N4}' -f $m.oos_test_acc);" ^
  "  Write-Host ('  Best iter   : {0}' -f $m.best_iteration);" ^
  "  Write-Host ('  Trained at  : {0}' -f $m.trained_at_utc);" ^
  "  Write-Host '';" ^
  "  Write-Host 'Rollback (if needed):' -ForegroundColor DarkGray;" ^
  "  Write-Host '  copy data\models\archive\<timestamp>\* data\models\' -ForegroundColor DarkGray" ^
  "}"
echo.
pause
exit /b 0
