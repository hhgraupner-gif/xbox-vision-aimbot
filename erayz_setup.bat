@echo off
title eRayz Audio ULTIMATE - Installer v1.0
color 0A
cls

echo.
echo  ============================================================
echo       eRayz Audio ULTIMATE - Installer
echo  ============================================================
echo       Competitive Audio Tool fuer Warzone / BO7
echo       11-Band EQ ^| Gunfire Ducker ^| HRTF Spatial
echo  ============================================================
echo.
echo  [1/5] Pruefe Python...

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo  FEHLER: Python nicht gefunden!
    echo  Bitte installiere Python von: https://www.python.org/downloads/
    echo  WICHTIG: Haekchen bei "Add Python to PATH" setzen!
    echo.
    pause
    exit /b 1
)

echo        Python gefunden!
echo.
echo  [2/5] Installiere Abhaengigkeiten...
echo.

pip install numpy scipy sounddevice --quiet --disable-pip-version-check
if %errorlevel% neq 0 (
    echo.
    echo  WARNUNG: Einige Pakete konnten nicht installiert werden.
    echo  Versuche es manuell: pip install numpy scipy sounddevice
    echo.
)

echo.
echo  [3/5] Erstelle Programmordner...

set "INSTALL_DIR=%USERPROFILE%\eRayz Audio"
if not exist "%INSTALL_DIR%" mkdir "%INSTALL_DIR%"

echo        Ordner: %INSTALL_DIR%
echo.
echo  [4/5] Lade eRayz Audio ULTIMATE herunter...

powershell -Command "Invoke-WebRequest -Uri 'https://bo7-aimbot-vision.preview.emergentagent.com/api/download/erayz_audio_ultimate.py' -OutFile '%INSTALL_DIR%\erayz_audio_ultimate.py'"
if %errorlevel% neq 0 (
    echo.
    echo  FEHLER: Download fehlgeschlagen!
    echo  Pruefe deine Internetverbindung.
    echo.
    pause
    exit /b 1
)

echo        Download OK!
echo.
echo  [5/5] Erstelle Desktop-Verknuepfung...

REM Launcher im Install-Ordner
(
echo @echo off
echo title eRayz Audio ULTIMATE
echo cd /d "%INSTALL_DIR%"
echo python erayz_audio_ultimate.py
echo if %%errorlevel%% neq 0 ^(
echo     echo.
echo     echo FEHLER beim Starten! Druecke eine Taste...
echo     pause
echo ^)
) > "%INSTALL_DIR%\eRayz Audio ULTIMATE.bat"

REM Desktop Shortcut via PowerShell
powershell -Command "$ws = New-Object -ComObject WScript.Shell; $sc = $ws.CreateShortcut('%USERPROFILE%\Desktop\eRayz Audio ULTIMATE.lnk'); $sc.TargetPath = '%INSTALL_DIR%\eRayz Audio ULTIMATE.bat'; $sc.WorkingDirectory = '%INSTALL_DIR%'; $sc.Description = 'eRayz Audio ULTIMATE - Competitive Audio Tool'; $sc.IconLocation = 'shell32.dll,168'; $sc.Save()"

echo        Desktop-Verknuepfung erstellt!
echo.
echo  ============================================================
echo.
echo       INSTALLATION ERFOLGREICH!
echo.
echo       Installiert in: %INSTALL_DIR%
echo       Desktop-Icon:   eRayz Audio ULTIMATE
echo.
echo       Starten: Doppelklick auf das Desktop-Icon
echo.
echo       Einstellungen:
echo         1. INPUT = deine Capture Card (HDMI/AVerMedia)
echo         2. OUTPUT = dein Headset
echo         3. START druecken
echo         4. Slider nach Geschmack anpassen
echo.
echo  ============================================================
echo.

set /p START_NOW="  Jetzt starten? (j/n): "
if /i "%START_NOW%"=="j" (
    cd /d "%INSTALL_DIR%"
    start "" python erayz_audio_ultimate.py
)

echo.
echo  Viel Spass! - eRayz
echo.
pause
