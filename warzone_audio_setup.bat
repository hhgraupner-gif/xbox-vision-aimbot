@echo off
chcp 65001 >nul
title eRayz Audio - Setup
color 0A

echo.
echo  ============================================
echo   eRayz Audio - INSTALLER
echo   Ranked Resurgence Pro Edition
echo  ============================================
echo.

:: Python Check
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo  [!] Python nicht gefunden!
    echo  [!] Bitte installiere Python von: https://www.python.org/downloads/
    echo  [!] WICHTIG: "Add to PATH" anhaken beim Installieren!
    echo.
    pause
    start https://www.python.org/downloads/
    exit
)

echo  [OK] Python gefunden
echo.

:: Dependencies installieren
echo  Installiere Pakete...
pip install sounddevice numpy scipy opencv-python >nul 2>&1
echo  [OK] Pakete installiert
echo.

:: Script downloaden
echo  Lade eRayz Audio Tool...
cd /d %USERPROFILE%\Downloads
powershell -Command "Invoke-WebRequest 'https://tinyurl.com/eRayz-EQ' -OutFile 'competitive_audio.py'" >nul 2>&1
echo  [OK] Download fertig
echo.

:: Geraete anzeigen
echo  ============================================
echo   DEINE AUDIO-GERAETE:
echo  ============================================
echo.
python competitive_audio.py --list
echo.
echo  ============================================
echo.
echo  Welches Geraet ist dein GAME AUDIO INPUT?
echo  (Stereo Mix, Capture Card, oder Headset)
set /p INPUT_DEV=  Input Nummer eingeben: 

echo.
echo  Welches Geraet ist dein KOPFHOERER OUTPUT?
echo  (Headset, Lautsprecher, DAC)
set /p OUTPUT_DEV=  Output Nummer eingeben: 

echo.
echo  Starte mit Ranked Preset...
echo.

:: Starten
python competitive_audio.py --input %INPUT_DEV% --output %OUTPUT_DEV% --preset ranked

pause
