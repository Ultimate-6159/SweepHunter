@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo.
echo ================================================================
echo   ⏯️  Clear Halt — ปลดล็อก bot ให้เทรดต่อทันที
echo ================================================================
echo.
python -c "import json,time; p=r'data\recovery_state.json'; d=json.load(open(p)); ts=d.get('halted_until_ts',0); now=time.time(); wait=ts-now; print(f'   ก่อน: halted_until_ts = {ts:.0f}'); print(f'           wait = {wait:.0f}s = {wait/3600:.1f}h'); d['halted_until_ts']=0.0; json.dump(d,open(p,'w'),indent=2); print(f'   หลัง: halted_until_ts = 0  ✅ ปลดล็อก')"
echo.
echo ⚠️  ถ้า bot กำลังรัน — จะเทรดต่อในรอบถัดไป (ภายใน 1 นาที)
echo.
pause
