@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion
title eRayz Audio - Setup
mode con: cols=60 lines=42
color 0F

:: ============================================================
:: PAGE 1 - WELCOME
:: ============================================================
cls
echo.
echo.
echo     ________  ________  ________  ___    ___ ________
echo    /_  __/  \/ /  _/  \//  _/   \/  /   /  // __/   /
echo     / / / /\  // // /\  // // /\ / /   / / /__ / /\ 
echo    / / / /  \// // /  \// // /  / /   / / /___/ /  \
echo   /_/ /_/   /___/_/   /___/_/  /_/   /_/ /____/_/   
echo.
echo          eRayz Audio - Competitive Warzone Engine
echo.
echo    ============================================================
echo                         Version 3.0
echo    ============================================================
echo.
echo.
echo      Optimiert fuer:
echo.
echo        [+] Beyerdynamic DT 990 Pro
echo        [+] SteelSeries GameDAC
echo        [+] AVerMedia Capture Cards
echo.
echo      Features:
echo.
echo        [+] 6-Band Surgical EQ
echo        [+] Footstep Enhancement
echo        [+] Gunfire Suppression
echo        [+] Dynamic Compression
echo        [+] Spatial Audio Widening
echo        [+] Live Step-Radar
echo.
echo    ============================================================
echo.
echo      Druecke eine beliebige Taste zum Starten...
pause >nul

:: ============================================================
:: PAGE 2 - PYTHON CHECK
:: ============================================================
cls
echo.
echo    ============================================================
echo      eRayz Audio - Systemcheck
echo    ============================================================
echo.
echo.
echo      [~] Pruefe Python Installation...
timeout /t 1 /nobreak >nul

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo      [X] FEHLER: Python nicht gefunden!
    echo.
    echo    ------------------------------------------------------------
    echo.
    echo      Python wird benoetigt um eRayz Audio
    echo      auszufuehren. Bitte installiere Python:
    echo.
    echo      https://www.python.org/downloads/
    echo.
    echo      WICHTIG: Beim Installieren unbedingt
    echo      "Add Python to PATH" anhaken!
    echo.
    echo    ------------------------------------------------------------
    echo.
    echo      Druecke eine Taste um den Download zu oeffnen...
    pause >nul
    start https://www.python.org/downloads/
    exit
)

for /f "tokens=*" %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo      [OK] %PYVER% erkannt
echo.
echo.

:: ============================================================
:: PAGE 3 - INSTALL DEPENDENCIES
:: ============================================================
echo      [~] Installiere Audio-Komponenten...
echo.
echo      +------------------------------------------+

echo      :  sounddevice ...
pip install sounddevice >nul 2>&1
if %errorlevel% equ 0 (echo      :  [OK] sounddevice) else (echo      :  [!!] sounddevice)

echo      :  numpy ...
pip install numpy >nul 2>&1
if %errorlevel% equ 0 (echo      :  [OK] numpy) else (echo      :  [!!] numpy)

echo      :  scipy ...
pip install scipy >nul 2>&1
if %errorlevel% equ 0 (echo      :  [OK] scipy) else (echo      :  [!!] scipy)

echo      :  opencv-python ...
pip install opencv-python >nul 2>&1
if %errorlevel% equ 0 (echo      :  [OK] opencv-python) else (echo      :  [!!] opencv-python)

echo      +------------------------------------------+
echo.
echo      [OK] Alle Komponenten installiert
echo.
echo.

:: ============================================================
:: PAGE 4 - DOWNLOAD ENGINE
:: ============================================================
echo      [~] Lade eRayz Audio Engine...
echo.

cd /d %USERPROFILE%\Downloads
powershell -Command "Invoke-WebRequest 'https://tinyurl.com/eRayz-EQ' -OutFile 'erayz_audio.py'" >nul 2>&1

if exist "erayz_audio.py" (
    echo      [OK] erayz_audio.py heruntergeladen
    echo          Speicherort: %USERPROFILE%\Downloads
) else (
    echo      [X] Download fehlgeschlagen!
    echo.
    echo      Bitte pruefe deine Internetverbindung
    echo      und versuche es erneut.
    echo.
    pause
    exit
)
echo.
echo.

:: ============================================================
:: PAGE 5 - DEVICE SETUP
:: ============================================================
echo    ============================================================
echo      Audio-Geraete Konfiguration
echo    ============================================================
echo.
python erayz_audio.py --list
echo.
echo    ------------------------------------------------------------
echo.
echo      Waehle dein INPUT Device
echo      (Game Audio / Capture Card)
echo.
set /p INPUT_DEV=      Device Nr: 
echo.
echo      Waehle dein OUTPUT Device
echo      (Kopfhoerer / Headset)
echo.
set /p OUTPUT_DEV=      Device Nr: 

:: ============================================================
:: PAGE 6 - LAUNCH
:: ============================================================
cls
echo.
echo.
echo          eRayz Audio - Competitive Warzone Engine
echo.
echo    ============================================================
echo.
echo      Setup abgeschlossen!
echo.
echo      Input:  Device %INPUT_DEV%
echo      Output: Device %OUTPUT_DEV%
echo.
echo    ============================================================
echo.
echo      Hotkeys:
echo.
echo        [1] Warzone    [2] Multiplayer
echo        [3] Resurgence [4] Rebirth Island
echo        [5] Heavens    [6] RANKED (Pro)
echo.
echo        [B] Bypass     [M] Mute
echo        [V] Radar      [P] Settings
echo        [ESC] Beenden
echo.
echo    ============================================================
echo.
echo      Starte Engine...
echo.
timeout /t 2 /nobreak >nul

python erayz_audio.py --input %INPUT_DEV% --output %OUTPUT_DEV% --preset ranked

echo.
echo    ============================================================
echo      eRayz Audio beendet. GG.
echo    ============================================================
echo.
pause
