@echo off
title eRayz Aimbot — Build .exe
color 0A
cls

echo.
echo  ============================================================
echo       eRayz Aimbot — EXE Builder
echo  ============================================================
echo.
echo  [1/4] Pruefe Python + PyInstaller...

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo  FEHLER: Python nicht gefunden!
    pause
    exit /b 1
)

pip install pyinstaller --quiet --disable-pip-version-check
echo        PyInstaller OK!

echo.
echo  [2/4] Pruefe Abhaengigkeiten...

pip install opencv-python numpy onnxruntime-directml --quiet --disable-pip-version-check
echo        Abhaengigkeiten OK!

echo.
echo  [3/4] Baue .exe (dauert 2-5 Minuten)...
echo.

cd /d "%USERPROFILE%\Downloads"

pyinstaller --noconfirm --onedir --console ^
    --name "eRayz Aimbot" ^
    --add-data "backend;backend" ^
    --add-data "config.json;." ^
    --hidden-import=onnxruntime ^
    --hidden-import=cv2 ^
    --hidden-import=numpy ^
    --collect-all onnxruntime ^
    aimbot_direct.py

if %errorlevel% neq 0 (
    echo.
    echo  FEHLER beim Bauen! Siehe Fehlermeldung oben.
    pause
    exit /b 1
)

echo.
echo  [4/4] Erstelle Desktop-Verknuepfung...

powershell -Command "$ws = New-Object -ComObject WScript.Shell; $sc = $ws.CreateShortcut('%USERPROFILE%\Desktop\eRayz Aimbot.lnk'); $sc.TargetPath = '%USERPROFILE%\Downloads\dist\eRayz Aimbot\eRayz Aimbot.exe'; $sc.WorkingDirectory = '%USERPROFILE%\Downloads\dist\eRayz Aimbot'; $sc.Description = 'eRayz Aimbot — Computer Vision AI'; $sc.IconLocation = 'shell32.dll,176'; $sc.Save()"

echo.
echo  ============================================================
echo.
echo       BUILD ERFOLGREICH!
echo.
echo       EXE:  %USERPROFILE%\Downloads\dist\eRayz Aimbot\eRayz Aimbot.exe
echo       Desktop-Icon: eRayz Aimbot
echo.
echo       WICHTIG: Der "dist\eRayz Aimbot" Ordner muss komplett
echo       bleiben (nicht nur die .exe verschieben!)
echo.
echo       Die ONNX Modelle muessen in:
echo       dist\eRayz Aimbot\backend\  liegen
echo.
echo  ============================================================
echo.

set /p START_NOW="  Jetzt starten? (j/n): "
if /i "%START_NOW%"=="j" (
    cd /d "%USERPROFILE%\Downloads\dist\eRayz Aimbot"
    start "" "eRayz Aimbot.exe"
)

pause
