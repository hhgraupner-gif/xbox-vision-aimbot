@echo off
title Xbox Vision AI - Aimbot Launcher
color 0A

echo.
echo  ╔═══════════════════════════════════════════════════════════╗
echo  ║           XBOX VISION AI - AIMBOT LAUNCHER                ║
echo  ║              Scuf Valor Pro Edition                       ║
echo  ╚═══════════════════════════════════════════════════════════╝
echo.

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python nicht gefunden! Bitte Python installieren.
    echo Download: https://www.python.org/downloads/
    pause
    exit /b 1
)

:: Check if Node.js is installed
node --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js nicht gefunden! Bitte Node.js installieren.
    echo Download: https://nodejs.org/
    pause
    exit /b 1
)

echo [OK] Python gefunden
echo [OK] Node.js gefunden
echo.

:: Get the directory where this script is located
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

echo ══════════════════════════════════════════════════════════════
echo  SCHRITT 1: Backend starten...
echo ══════════════════════════════════════════════════════════════
echo.

:: Start Backend in new window
start "Xbox Vision AI - Backend" cmd /k "cd /d "%SCRIPT_DIR%backend" && echo Starting Backend Server... && python -m uvicorn server:app --host 0.0.0.0 --port 8001 --reload"

:: Wait for backend to start
echo Warte auf Backend-Start...
timeout /t 5 /nobreak >nul

echo.
echo ══════════════════════════════════════════════════════════════
echo  SCHRITT 2: Frontend starten...
echo ══════════════════════════════════════════════════════════════
echo.

:: Start Frontend in new window
start "Xbox Vision AI - Frontend" cmd /k "cd /d "%SCRIPT_DIR%frontend" && echo Starting Frontend... && npm start"

:: Wait for frontend to start
echo Warte auf Frontend-Start...
timeout /t 8 /nobreak >nul

echo.
echo ══════════════════════════════════════════════════════════════
echo  SCHRITT 3: Browser oeffnen...
echo ══════════════════════════════════════════════════════════════
echo.

:: Open browser
start http://localhost:3000

echo.
echo ╔═══════════════════════════════════════════════════════════╗
echo ║                    APP GESTARTET!                         ║
echo ╠═══════════════════════════════════════════════════════════╣
echo ║                                                           ║
echo ║  1. Xbox Remote Play im VOLLBILD starten                  ║
echo ║  2. In der App: "Demo Mode" AUS schalten                  ║
echo ║  3. "START" druecken                                      ║
echo ║  4. Controller: LB = Aim Assist an/aus                    ║
echo ║                                                           ║
echo ║  Controller-Tasten:                                       ║
echo ║  - LB      = Aim Assist Toggle                            ║
echo ║  - LT      = ADS (Aim aktiv beim Zielen)                  ║
echo ║  - RT      = Schiessen                                    ║
echo ║  - RS Klick = Snap zum Ziel                               ║
echo ║  - RB      = Naechstes Ziel                               ║
echo ║  - LB + RB = NOTAUS (alles deaktivieren)                  ║
echo ║                                                           ║
echo ╚═══════════════════════════════════════════════════════════╝
echo.
echo Druecke eine Taste zum Beenden dieses Fensters...
echo (Backend und Frontend laufen weiter!)
echo.
pause >nul
