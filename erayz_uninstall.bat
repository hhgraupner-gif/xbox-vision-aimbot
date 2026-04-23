@echo off
title eRayz Audio ULTIMATE - Deinstallation
color 0C
cls

echo.
echo  ============================================================
echo       eRayz Audio ULTIMATE - Deinstallation
echo  ============================================================
echo.

set "INSTALL_DIR=%USERPROFILE%\eRayz Audio"

echo  Folgendes wird geloescht:
echo    - %INSTALL_DIR%
echo    - Desktop-Verknuepfung
echo    - Einstellungen
echo.

set /p CONFIRM="  Wirklich deinstallieren? (j/n): "
if /i not "%CONFIRM%"=="j" (
    echo  Abgebrochen.
    pause
    exit /b 0
)

echo.
echo  Loesche Dateien...

if exist "%INSTALL_DIR%" rmdir /s /q "%INSTALL_DIR%"
if exist "%USERPROFILE%\Desktop\eRayz Audio ULTIMATE.lnk" del "%USERPROFILE%\Desktop\eRayz Audio ULTIMATE.lnk"
if exist "%USERPROFILE%\Desktop\eRayz Audio ULTIMATE.bat" del "%USERPROFILE%\Desktop\eRayz Audio ULTIMATE.bat"
if exist "%USERPROFILE%\erayz_ultimate.json" del "%USERPROFILE%\erayz_ultimate.json"

echo.
echo  Deinstallation abgeschlossen!
echo.
pause
