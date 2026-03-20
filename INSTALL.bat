@echo off
title Xbox Vision AI - Installer
color 0B

echo.
echo  ╔═══════════════════════════════════════════════════════════╗
echo  ║         XBOX VISION AI - INSTALLATION                     ║
echo  ║            Einmalig ausfuehren!                           ║
echo  ╚═══════════════════════════════════════════════════════════╝
echo.

:: Get the directory where this script is located
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

echo ══════════════════════════════════════════════════════════════
echo  SCHRITT 1: Python Pakete installieren...
echo ══════════════════════════════════════════════════════════════
echo.

cd backend
pip install -r requirements.txt
if errorlevel 1 (
    echo [FEHLER] Backend Installation fehlgeschlagen!
    pause
    exit /b 1
)
echo [OK] Backend Pakete installiert

echo.
echo ══════════════════════════════════════════════════════════════
echo  SCHRITT 2: YOLO Modell vorladen...
echo ══════════════════════════════════════════════════════════════
echo.

python -c "from ultralytics import YOLO; YOLO('yolov8n.pt'); print('[OK] YOLO Modell geladen')"

echo.
echo ══════════════════════════════════════════════════════════════
echo  SCHRITT 3: Frontend Pakete installieren...
echo ══════════════════════════════════════════════════════════════
echo.

cd ..\frontend
call npm install
if errorlevel 1 (
    echo [FEHLER] Frontend Installation fehlgeschlagen!
    pause
    exit /b 1
)
echo [OK] Frontend Pakete installiert

echo.
echo ╔═══════════════════════════════════════════════════════════╗
echo ║              INSTALLATION ABGESCHLOSSEN!                  ║
echo ╠═══════════════════════════════════════════════════════════╣
echo ║                                                           ║
echo ║  Starte jetzt die App mit:                                ║
echo ║                                                           ║
echo ║     START_AIMBOT.bat                                      ║
echo ║                                                           ║
echo ╚═══════════════════════════════════════════════════════════╝
echo.
pause
