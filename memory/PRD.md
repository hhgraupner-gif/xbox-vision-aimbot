# Xbox Vision AI - Computer Vision Aimbot System

## Original Problem Statement
Baue mir ein Computer Vision KI für RemotePlay Xbox - mit Gegnererkennung (YOLO) und Zielhilfe/Aimbot Funktionalität.

## Architecture
- **Backend**: FastAPI + YOLO (ultralytics) + OpenCV + MSS (Screen Capture)
- **Frontend**: React + Tailwind CSS + Shadcn/UI
- **Database**: MongoDB (Settings persistence)

## User Personas
- Gamer wanting AI-assisted gameplay tools
- Single-player/private match users

## Core Requirements (Implemented)
- [x] Real-time enemy detection using YOLOv8
- [x] Bounding box visualization with confidence scores
- [x] Crosshair overlay
- [x] Aim assist target lock indicator
- [x] Demo mode for testing
- [x] Configurable settings (confidence, sensitivity, target classes)
- [x] Monitor selection for capture
- [x] Settings persistence in MongoDB

## What's Been Implemented (March 2026)
1. Backend:
   - YOLO detection endpoint (/api/capture)
   - Demo frame generation (/api/demo/frame)
   - Settings CRUD (/api/settings)
   - WebSocket streaming (/api/ws/detections)
   - MJPEG stream (/api/stream)

2. Frontend:
   - Cyberpunk-themed Dashboard with live feed
   - Real-time stats (FPS, detections, processing time)
   - Quick settings panel
   - Full Settings page with all configuration options
   - Aim assist toggle and visualization

## Prioritized Backlog
### P0 (Critical)
- [x] Basic detection system
- [x] UI Dashboard

### P1 (Important)
- [ ] Capture Card input support (Elgato, etc.)
- [ ] Custom region capture (crop to game window)
- [ ] Mouse movement integration for actual aim assist

### P2 (Nice to Have)
- [ ] Multiple game profile presets
- [ ] Recording/replay functionality
- [ ] Custom YOLO model training for specific games

## Next Tasks
1. Add capture card support via OpenCV VideoCapture
2. Implement region selection tool
3. Add pyautogui for mouse movement (actual aimbot)
4. Create game-specific detection profiles

## Update: Game Profiles (March 2026)

### Implemented Profiles
1. **Default** - Standard detection (50% conf, 80% sens)
2. **Call of Duty: Warzone** - Fast targets, high precision (45% conf, 95% sens, closest targeting)
3. **Call of Duty: Black Ops 7** - Quick reflexes multiplayer (40% conf, 100% sens, center targeting)
4. **Call of Duty: Zombies** - Multiple targets/hordes (35% conf, 85% sens)

### Profile Parameters
- `confidence_threshold` - Detection confidence
- `aim_sensitivity` - Response speed
- `aim_point_offset` - Head targeting offset (0.10-0.20)
- `priority_targeting` - closest/center/highest_confidence
