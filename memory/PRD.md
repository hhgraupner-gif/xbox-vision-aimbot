# Xbox Vision AI - Computer Vision Aimbot System

## Original Problem Statement
Baue mir ein Computer Vision KI fuer RemotePlay Xbox - mit Gegnererkennung (YOLO) und Zielhilfe/Aimbot Funktionalitaet.
Hardware: AVerMedia GC571 Capture Card, XIM Matrix, Scuf Valor Pro Controller.
App laeuft LOKAL auf dem Windows PC des Users.

## Architecture
- **Backend**: FastAPI + YOLOv8 (ultralytics) + OpenCV + MSS/Capture Card
- **Frontend**: Standalone AIMBOT.html (React wurde verworfen)
- **Database**: MongoDB optional (in-memory fallback)
- **Hardware**: AVerMedia GC571 -> YOLO -> PyAutoGUI -> XIM Matrix -> Xbox

## Core Requirements
- [x] Real-time enemy detection using YOLOv8
- [x] Bounding box visualization with confidence scores
- [x] Crosshair overlay
- [x] Aim assist target lock indicator
- [x] Demo mode for testing
- [x] Configurable settings (confidence, sensitivity, target classes)
- [x] Game Profiles (Warzone, BO7, Zombies)
- [x] Capture Card input support (AVerMedia GC571)
- [x] Stabilized aimbot with EMA tracking (anti-twitching)
- [x] Fine-tuning controls (deadzone, max speed, lock frames)
- [ ] XIM Matrix integration (user must configure hardware)
- [ ] Custom YOLO model for Call of Duty

## What's Been Implemented

### March 2026 - Initial Build
- FastAPI backend with YOLO detection
- Standalone AIMBOT.html dashboard
- Demo frame generation
- Settings persistence in MongoDB
- Game profiles

### March 2026 - Stabilization Update
- **Target Tracker**: EMA-based tracking across frames (anti-twitching)
- **Velocity Capping**: max_move_px limits mouse speed per frame
- **Lock Frames**: Target must be seen N frames before aimbot engages
- **Deadzone**: Configurable pixel deadzone around crosshair
- **Capture Card**: cv2.VideoCapture with CAP_DSHOW for Windows
- **Capture Card Test**: API endpoint + UI button to test device
- **UI Tuning Controls**: Deadzone, Max Speed, Lock Frames sliders
- **Tracker Reset**: Button to reset stuck tracker
- **XIM Matrix Anleitung**: Detailed German setup guide
- **Capture Card Anleitung**: Detailed German setup guide
- **Removed vgamepad** dependency (caused conflicts with Scuf)

## Prioritized Backlog
### P0 (Critical - User Must Do)
- [ ] User: Configure XIM Matrix (see ANLEITUNG_XIM_MATRIX.md)
- [ ] User: Test Capture Card locally (see ANLEITUNG_CAPTURE_CARD.md)
- [ ] User: Test aimbot with real gameplay

### P1 (Important)
- [ ] Fine-tune sensitivity/smoothing for actual CoD gameplay
- [ ] Persistent capture card connection (keep cap open, not re-open each frame)

### P2 (Nice to Have)
- [ ] Custom YOLO model trained on Call of Duty assets
- [ ] Recording/replay functionality
- [ ] Hotkey to toggle aimbot without UI

## Key Files
- `/app/backend/server.py` - Core backend
- `/app/AIMBOT.html` - Standalone UI dashboard
- `/app/ANLEITUNG_XIM_MATRIX.md` - XIM Matrix setup guide (German)
- `/app/ANLEITUNG_CAPTURE_CARD.md` - Capture Card setup guide (German)
- `/app/test_capture.py` - Capture Card device finder script
- `/app/backend/controller.py` - Controller input handler

## Key API Endpoints
- `GET /api/` - Health check
- `GET /api/capture` - Frame with detections
- `GET /api/demo/frame` - Demo frame
- `POST /api/aimbot/enable` / `disable` - Toggle aimbot
- `PUT /api/aimbot/sensitivity` - Set sensitivity + smoothing
- `PUT /api/aimbot/tuning` - Set deadzone, max_move, lock_frames
- `POST /api/aimbot/reset-tracker` - Reset target tracker
- `GET /api/aimbot/status` - Full aimbot status with tracker info
- `PUT /api/capture-card/config` - Enable/disable capture card
- `GET /api/capture-card/test` - Test capture card connection
- `GET /api/profiles` - List game profiles
- `POST /api/profiles/{id}/activate` - Activate profile
