@echo off
chcp 65001 >nul
cd /d "%~dp0"
title SweepHunter - AutoStart Setup

echo.
echo ================================================================
echo   SweepHunter - Auto-Start Setup
echo   - Bot เปิดอัตโนมัติหลัง reboot
echo   - Windows reboot ทุกเสาร์ 04:00 UTC (ตลาดปิด)
echo ================================================================
echo.

:: ต้องรันด้วย Administrator
net session >nul 2>&1
if errorlevel 1 (
    echo [ERROR] กรุณา Right-click แล้วเลือก "Run as administrator"
    pause & exit /b 1
)

:: หา Python path
if exist "%~dp0.python_path.txt" (
    set /p PYEXE=<"%~dp0.python_path.txt"
) else (
    set PYEXE=python
)

set BOT_DIR=%~dp0
:: ตัด trailing backslash
if "%BOT_DIR:~-1%"=="\" set BOT_DIR=%BOT_DIR:~0,-1%
set BOT_BAT=%BOT_DIR%\1_start_bot.bat

echo [1/4] ตั้งค่า Windows Auto-Login...
echo.
echo   กรุณาใส่ Password ของ Administrator
echo   (เพื่อให้ Windows login อัตโนมัติหลัง reboot)
echo   ถ้าไม่มี password กด Enter ปล่าว
echo.
set /p WIN_PASS=  Password: 

:: ตั้ง auto-login registry
reg add "HKLM\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Winlogon" /v AutoAdminLogon /t REG_SZ /d "1" /f >nul
reg add "HKLM\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Winlogon" /v DefaultUserName /t REG_SZ /d "Administrator" /f >nul
reg add "HKLM\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Winlogon" /v DefaultDomainName /t REG_SZ /d "%COMPUTERNAME%" /f >nul
reg add "HKLM\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Winlogon" /v DefaultPassword /t REG_SZ /d "%WIN_PASS%" /f >nul
echo   OK - Auto-login ตั้งค่าแล้ว

echo.
echo [2/4] ลบ Task เก่า (ถ้ามี)...
schtasks /delete /tn "SweepHunterBot" /f >nul 2>&1
schtasks /delete /tn "SweepHunterReboot" /f >nul 2>&1
echo   OK

echo.
echo [3/4] สร้าง Task: เปิดบอทอัตโนมัติเมื่อ Login...
schtasks /create /tn "SweepHunterBot" ^
    /tr "cmd.exe /c \"%BOT_BAT%\"" ^
    /sc onlogon ^
    /ru "Administrator" ^
    /rl highest ^
    /delay 0000:30 ^
    /f
if errorlevel 1 (
    echo [ERROR] สร้าง task ไม่สำเร็จ
    pause & exit /b 1
)
echo   OK - บอทจะเปิดใน 30 วินาทีหลัง login

echo.
echo [4/4] สร้าง Task: Reboot ทุกเสาร์ 04:00 UTC...
schtasks /create /tn "SweepHunterReboot" ^
    /tr "shutdown.exe /r /f /t 30" ^
    /sc weekly ^
    /d SAT ^
    /st 04:00 ^
    /ru SYSTEM ^
    /rl highest ^
    /f
if errorlevel 1 (
    echo [ERROR] สร้าง task reboot ไม่สำเร็จ
    pause & exit /b 1
)
echo   OK - Reboot ทุกเสาร์ 04:00 UTC

echo.
echo [ตรวจสอบ MT5 Auto-start]
set MT5_PATH=C:\Program Files\MetaTrader 5\terminal64.exe
if exist "%MT5_PATH%" (
    schtasks /delete /tn "MetaTrader5AutoStart" /f >nul 2>&1
    schtasks /create /tn "MetaTrader5AutoStart" ^
        /tr "\"%MT5_PATH%\"" ^
        /sc onlogon ^
        /ru "Administrator" ^
        /rl highest ^
        /delay 0000:15 ^
        /f >nul 2>&1
    echo   OK - MT5 จะเปิดใน 15 วินาทีหลัง login (ก่อนบอท)
) else (
    echo   [SKIP] ไม่พบ MT5 ที่ %MT5_PATH%
    echo   กรุณาเพิ่ม MT5 ใน Windows Startup ด้วยตัวเอง
)

echo.
echo ================================================================
echo   SETUP เสร็จแล้ว!
echo.
echo   ทดสอบ: รีบูท Windows แล้วรอ ~30 วินาที
echo          บอทควรเปิดขึ้นมาเอง
echo.
echo   Scheduled Tasks:
schtasks /query /tn "SweepHunterBot" /fo list 2>&1 | findstr "Task Name\|Next Run"
schtasks /query /tn "SweepHunterReboot" /fo list 2>&1 | findstr "Task Name\|Next Run"
echo ================================================================
echo.
pause
exit /b 0
