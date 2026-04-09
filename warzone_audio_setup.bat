@echo off
chcp 65001 >nul
title eRayz Audio
mode con: cols=62 lines=40
color 0F

echo.
echo   ╔══════════════════════════════════════════════════════╗
echo   ║                                                      ║
echo   ║          ███████╗██████╗  █████╗ ██╗   ██╗███████╗  ║
echo   ║          ██╔════╝██╔══██╗██╔══██╗╚██╗ ██╔╝╚══███╔╝  ║
echo   ║          █████╗  ██████╔╝███████║ ╚████╔╝   ███╔╝   ║
echo   ║          ██╔══╝  ██╔══██╗██╔══██║  ╚██╔╝   ███╔╝    ║
echo   ║          ███████╗██║  ██║██║  ██║   ██║   ███████╗   ║
echo   ║          ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚══════╝   ║
echo   ║                   A U D I O                          ║
echo   ║                                                      ║
echo   ║          Competitive Warzone Audio Engine            ║
echo   ║                    v3.0                              ║
echo   ║                                                      ║
echo   ╚══════════════════════════════════════════════════════╝
echo.

:: Python Check
python --version >nul 2>&1
if %errorlevel% neq 0 (
    color 0C
    echo   [X] Python wurde nicht gefunden.
    echo.
    echo   Bitte installiere Python:
    echo   https://www.python.org/downloads/
    echo.
    echo   WICHTIG: "Add Python to PATH" anhaken!
    echo.
    pause
    start https://www.python.org/downloads/
    exit
)

echo   [+] Python erkannt
echo.
echo   Installiere Komponenten...
pip install sounddevice numpy scipy opencv-python >nul 2>&1
echo   [+] Alle Komponenten installiert
echo.
echo   Lade eRayz Audio Engine...
cd /d %USERPROFILE%\Downloads
powershell -Command "Invoke-WebRequest 'https://tinyurl.com/eRayz-EQ' -OutFile 'erayz_audio.py'" >nul 2>&1
echo   [+] Download abgeschlossen
echo.
echo   ══════════════════════════════════════════════════════
echo                    AUDIO DEVICES
echo   ══════════════════════════════════════════════════════
echo.
python erayz_audio.py --list
echo.
echo   ══════════════════════════════════════════════════════
echo.
echo   Waehle dein INPUT Device (Game Audio / Capture Card)
set /p INPUT_DEV=   Device Nr: 

echo.
echo   Waehle dein OUTPUT Device (Kopfhoerer / Headset)
set /p OUTPUT_DEV=   Device Nr: 

echo.
echo   ══════════════════════════════════════════════════════
echo            Starte eRayz Audio Engine...
echo   ══════════════════════════════════════════════════════
echo.

python erayz_audio.py --input %INPUT_DEV% --output %OUTPUT_DEV% --preset ranked

pause
