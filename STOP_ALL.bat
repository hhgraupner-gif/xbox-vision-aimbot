@echo off
title Xbox Vision AI - Stop All
color 0C

echo.
echo  Stoppe alle Xbox Vision AI Prozesse...
echo.

:: Kill Python (Backend)
taskkill /F /IM python.exe /T 2>nul
taskkill /F /IM uvicorn.exe /T 2>nul

:: Kill Node (Frontend)
taskkill /F /IM node.exe /T 2>nul

echo.
echo  [OK] Alle Prozesse gestoppt!
echo.
timeout /t 2 >nul
