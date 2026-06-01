@echo off
chcp 65001 >nul
title SweepHunter - เปิด Auto-Start

net session >nul 2>&1
if errorlevel 1 (
    echo [ERROR] กรุณา Right-click แล้วเลือก "Run as administrator"
    pause & exit /b 1
)

echo.
echo ================================================================
echo   เปิด Auto-Start ทั้งหมด
echo ================================================================
echo.

schtasks /change /tn "SweepHunterBot"      /enable >nul 2>&1 && echo   [OK] SweepHunterBot     - เปิดแล้ว || echo   [SKIP] SweepHunterBot ไม่พบ (รัน setup_autostart.bat ก่อน)
schtasks /change /tn "MetaTrader5AutoStart" /enable >nul 2>&1 && echo   [OK] MetaTrader5         - เปิดแล้ว || echo   [SKIP] MetaTrader5 ไม่พบ
schtasks /change /tn "SweepHunterReboot"   /enable >nul 2>&1 && echo   [OK] SweepHunterReboot  - เปิดแล้ว || echo   [SKIP] SweepHunterReboot ไม่พบ

echo.
echo   เปิด Auto-Login (Windows จะ login อัตโนมัติหลัง reboot)
echo.
echo   กรุณาใส่ Password ของ Administrator
set /p WIN_PASS=  Password: 
reg add "HKLM\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Winlogon" /v AutoAdminLogon /t REG_SZ /d "1" /f >nul
reg add "HKLM\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Winlogon" /v DefaultUserName /t REG_SZ /d "Administrator" /f >nul
reg add "HKLM\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Winlogon" /v DefaultDomainName /t REG_SZ /d "." /f >nul
reg add "HKLM\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Winlogon" /v DefaultPassword /t REG_SZ /d "%WIN_PASS%" /f >nul
echo   [OK] Auto-Login - เปิดแล้ว

echo.
echo   ทุกอย่างเปิดแล้ว บอทและ MT5 จะเปิดเองหลัง reboot
echo   Reboot ครั้งถัดไป: ทุกเสาร์ 04:00 UTC
echo ================================================================
echo.
pause
