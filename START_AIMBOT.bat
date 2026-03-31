@echo off
title Xbox Vision AI - Aimbot v3 (FPS-KI)
color 0A

echo.
echo  ========================================================
echo              XBOX VISION AI - AIMBOT v3
echo            FPS-KI Modell (Black Ops / Warzone)
echo  ========================================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python nicht gefunden!
    echo Download: https://www.python.org/downloads/
    pause
    exit /b 1
)
echo [OK] Python gefunden

:: Get script directory
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

:: Check for FPS model files
if exist "backend\sunxds_nano_320.onnx" (
    echo [OK] FPS-Modell nano gefunden (schnell, 320px)
) else (
    echo [!!] FPS-Modell nano NICHT gefunden
)
if exist "backend\sunxds_640.onnx" (
    echo [OK] FPS-Modell standard gefunden (genau, 640px)
) else (
    echo [!!] FPS-Modell standard NICHT gefunden
)
echo.

:: Install dependencies if needed
echo Pruefe Abhaengigkeiten...
pip show onnxruntime-directml >nul 2>&1
if errorlevel 1 (
    echo Installiere onnxruntime-directml...
    pip install onnxruntime-directml
)
pip show opencv-python >nul 2>&1
if errorlevel 1 (
    echo Installiere opencv-python...
    pip install opencv-python
)
pip show numpy >nul 2>&1
if errorlevel 1 (
    echo Installiere numpy...
    pip install numpy
)
echo [OK] Abhaengigkeiten OK
echo.

echo  ========================================================
echo  Starte Aimbot v3...
echo  ========================================================
echo.
echo  Tasten:
echo    S = Screenshot-Sammlung an/aus
echo    M = Modell wechseln (nano/standard)
echo    Q = Beenden
echo.
echo  Der Aimbot aktiviert sich automatisch bei ADS!
echo  ========================================================
echo.

python aimbot_direct.py

echo.
echo  Aimbot beendet.
pause
