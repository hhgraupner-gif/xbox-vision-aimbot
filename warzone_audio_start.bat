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
echo              MULTIBAND PRO  |  Version 4.0
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
echo        [Q/W] Band A (Steps Low)  -/+
echo        [E/R] Band B (Mud Zone)   -/+
echo        [A/S] Band C (Step Detail)-/+
echo        [D/F] Band D (Gunfire)    -/+
echo        [G/H] Output Gain         -/+
echo        [X/C] Spatial             -/+
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
