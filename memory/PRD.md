# Xbox Vision AI - Computer Vision Aimbot System

## Original Problem Statement
Baue mir ein Computer Vision KI fuer Xbox - mit Gegnererkennung (YOLO) und Zielhilfe/Aimbot Funktionalitaet.
Spiele: Black Ops 7 / Warzone (selbe Engine).
Hardware: AVerMedia GC571 Capture Card, XIM Matrix, KMBox Net, Scuf Valor Pro Controller, AMD RX 7800 XT.
App laeuft LOKAL auf dem Windows PC des Users.

## Architecture
- **Core Script**: `aimbot_direct.py` (Standalone OpenCV + ONNX + KMBox loop)
- **AI Models**: SunOner FPS YOLOv8 ONNX (sunxds_0.2.1) - 10 FPS-spezifische Klassen
- **Hardware Input**: KMBox Net (UDP) -> XIM Matrix -> Xbox Controller
- **Video Input**: AVerMedia GC571 Capture Card -> OpenCV
- **GPU Acceleration**: ONNX Runtime DirectML (AMD RX 7800 XT)
- **Legacy UI**: FastAPI + AIMBOT.html (fuer Konfiguration)

## Core Requirements
- [x] Real-time enemy detection using YOLO
- [x] FPS-spezifisches KI-Modell (nicht generisches COCO)
- [x] Headshot-Priorisierung (Kopf-Erkennung)
- [x] Klassen-Filterung: Nur Spieler/Bots/Koepfe, NICHT Waffen/Tote/Rauch
- [x] ADS-Erkennung (Aimbot nur bei Aim Down Sights)
- [x] KMBox Net Hardware-Integration (UDP)
- [x] Zwei Modell-Varianten (nano 320px / standard 640px)
- [x] Hot-Switch zwischen Modellen (Taste M)
- [x] Screenshot-Sammlung fuer Custom-Training (Taste S)
- [x] DirectML GPU-Beschleunigung (AMD)
- [ ] Custom YOLO-Modell auf Black Ops 7 Gameplay trainiert

## What's Been Implemented

### March 2026 - Initial Build
- FastAPI backend with YOLO detection
- Standalone AIMBOT.html dashboard
- Demo frame generation, settings persistence, game profiles

### March 2026 - Hardware Integration
- KMBox Net UDP protocol (`kmbox_net.py`)
- Eliminated PyTorch/Ultralytics dependency
- Migrated to ONNX Runtime with DirectML
- Created `aimbot_direct.py` for high-performance game loop
- Visual ADS detection

### March 2026 - FPS Model Integration (v3)
- **Replaced generic COCO model with FPS-specific SunOner model**
  - 10 specialized classes: player, bot, weapon, outline, dead_body, hideout_targets, head, smoke, fire
  - Trained on 17,000+ FPS game images (Warface, Destiny 2, Battlefield, CS:GO, CS2)
  - Auto-detects model type (FPS vs COCO) based on output dimensions
  - fp16 inference support for maximum GPU performance
- **Two model variants**: nano (320px, fast) and standard (640px, accurate)
- **Headshot prioritization**: Head detections are preferred over body
- **Smart class filtering**: Only aims at player/bot/head, ignores weapons/dead_bodies/smoke/fire
- **Updated yolo_onnx.py**: Supports both FPS and COCO models with auto-detection
- **Model hot-switch**: Press M to toggle between nano and standard in-game

## Key Files
- `/app/aimbot_direct.py` - Core aimbot script v3 (FPS model)
- `/app/backend/yolo_onnx.py` - YOLO ONNX inference engine (FPS + COCO)
- `/app/backend/kmbox_net.py` - KMBox Net UDP driver
- `/app/backend/sunxds_nano_320.onnx` - FPS model nano (6 MB, 320px)
- `/app/backend/sunxds_640.onnx` - FPS model standard (22 MB, 640px)
- `/app/backend/server.py` - Legacy FastAPI backend
- `/app/AIMBOT.html` - Legacy UI
- `/app/START_AIMBOT.bat` - Windows launcher
- `/app/ANLEITUNG_FPS_MODELL.md` - Deutsche Anleitung fuer FPS-Modell

## Prioritized Backlog

### P0 (User muss testen)
- [ ] User: FPS-Modell im Spiel testen (git pull -> START_AIMBOT.bat)
- [ ] User: ADS-Erkennung verifizieren
- [ ] User: nano vs standard Performance vergleichen

### P1 (Wichtig)
- [ ] Sensitivity/Smoothing fuer Black Ops 7 feintunen
- [ ] DirectML Performance-Tuning falls FPS sinkt

### P2 (Nice to Have)
- [ ] Custom YOLO-Modell auf Black Ops 7 Screenshots trainieren (Roboflow)
- [ ] BO6 Enemy Detection Roboflow-Modell integrieren (80.6% mAP, 4 Klassen: head/body/enemy/friendly)
- [ ] AVerMedia Capture Card FPS-Drop ohne proprietaere Software

## Hardware Config
- KMBox Net IP: 192.168.2.188
- KMBox Net Port: 32778
- KMBox Net UUID: C14AE466
- Capture Device: 0
