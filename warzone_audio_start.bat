@echo off
chcp 65001 >nul
title eRayz Audio
color 0F
cd /d %USERPROFILE%\Downloads

echo.
echo   ╔══════════════════════════════════════════╗
echo   ║         eRayz Audio Engine v3.0          ║
echo   ╚══════════════════════════════════════════╝
echo.

python erayz_audio.py
