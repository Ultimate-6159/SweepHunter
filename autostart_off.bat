@echo off
chcp 65001 >nul
title SweepHunter - ปิด Auto-Start

net session >nul 2>&1
if errorlevel 1 (
    echo [ERROR] กรุณา Right-click แล้วเลือก "Run as administrator"
    pause & exit /b 1
)

echo.
echo ================================================================
echo   ปิด Auto-Start ทั้งหมด
echo ================================================================
echo.

schtasks /change /tn "SweepHunterBot"      /disable >nul 2>&1 && echo   [OK] SweepHunterBot     - ปิดแล้ว || echo   [SKIP] SweepHunterBot ไม่พบ
schtasks /change /tn "MetaTrader5AutoStart" /disable >nul 2>&1 && echo   [OK] MetaTrader5         - ปิดแล้ว || echo   [SKIP] MetaTrader5 ไม่พบ
schtasks /change /tn "SweepHunterReboot"   /disable >nul 2>&1 && echo   [OK] SweepHunterReboot  - ปิดแล้ว || echo   [SKIP] SweepHunterReboot ไม่พบ

echo.
echo   ปิด Auto-Login (Windows จะถาม password หลัง reboot)
reg add "HKLM\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Winlogon" /v AutoAdminLogon /t REG_SZ /d "0" /f >nul
echo   [OK] Auto-Login - ปิดแล้ว

echo.
echo   ทุกอย่างปิดแล้ว บอทและ MT5 จะไม่เปิดเองหลัง reboot
echo ================================================================
echo.
pause
