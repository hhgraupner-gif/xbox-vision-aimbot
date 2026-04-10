@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion
title eRayz Audio
mode con: cols=60 lines=42
color 0F

cls
echo.
echo.
echo          eRayz Audio - Competitive Warzone Engine
echo.
echo    ============================================================
echo              SURGICAL PRO v6  |  Version 6.0
echo    ============================================================
echo.
echo.

cd /d %USERPROFILE%\Downloads
if not exist "erayz_audio.py" (
    echo      [X] erayz_audio.py nicht gefunden!
    echo.
    echo      Bitte fuehre zuerst das Setup aus:
    echo      warzone_audio_setup.bat
    echo.
    echo    ------------------------------------------------------------
    echo.
    pause
    exit
)

echo      [OK] Engine gefunden
echo.
echo      Starte mit gespeicherten Einstellungen...
echo.
echo    ------------------------------------------------------------
echo.
echo      Steuerung:
echo.
echo        [Q/W] Step Body 250Hz     -/+
echo        [E/R] Step Texture 2kHz   -/+
echo        [A/S] Step Direction 3kHz -/+
echo        [D/F] Step Clarity 4.5kHz -/+
echo        [G/H] Output Gain         -/+
echo        [X/C] Spatial             -/+
echo        [U/I] Gunfire Cut         -/+
echo.
echo        [B] Bypass    [M] Mute
echo        [V] Radar     [P] Settings
echo        [ESC] Beenden
echo.
echo    ------------------------------------------------------------
echo.
timeout /t 2 /nobreak >nul

python erayz_audio.py

echo.
echo    ============================================================
echo      eRayz Audio beendet. GG.
echo    ============================================================
echo.
pause
