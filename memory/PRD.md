# Xbox Vision AI — PRD (Product Requirements Document)

## Ziel
Computer Vision AI Aimbot fuer Xbox (RemotePlay) mit:
- Feind-Erkennung via YOLO/ONNX
- Hardware-Input via KMBox Net (aktuell) → Titan Two (ab 08.04)
- Spiel: Call of Duty Black Ops 7 / Warzone

## Hardware-Setup
- **Capture:** AVerMedia GC571 (1440p@60FPS Capture, 1080p intern)
- **GPU:** AMD RX 7800 XT (DirectML)
- **Input aktuell:** KMBox Net → XIM Matrix → Xbox
- **Input neu (08.04):** Titan Two → Xbox (direkt, 1:1 praezise)
- **Controller:** Scuf Envision Pro (PC, XInput)

## Code-Architektur
```
/app/
├── aimbot_direct.py       # v14 Hauptscript (Capture + AI + Tracking + Input)
├── minimap_reader.py      # Minimap HSV-Farberkennung (Teammates/Feinde)
├── titan_two.py           # Titan Two Kommunikation (GCV Protokoll)
├── aimbot_gpc.gpc         # GPC Script fuer Titan Two (Macros, Aim, Anti-Recoil)
├── TITAN_TWO_SETUP.md     # Setup-Anleitung Deutsch
├── backend/
│   ├── yolo_onnx.py       # ONNX Runtime Inference
│   ├── kmbox_net.py       # KMBox UDP Client
│   └── sunxds_0.5.6.onnx  # SunOner FPS Modell (30k+ Bilder)
```

## Abgeschlossene Features
- [x] YOLO Feind-Erkennung (SunOner FPS Modell)
- [x] KMBox Net Integration
- [x] Threaded Capture + Inference (70-90 FPS, <1ms AI)
- [x] ROI-Cropping (640x640 Mitte)
- [x] Display-Throttle + Performance-Optimierungen
- [x] Minimap-Reader (Teammate blau/gruen, Feinde rot)
- [x] Teammate-Schutz (filtert Ziele in Teammate-Richtung)
- [x] Live Minimap-Kalibrierung (IJKL + Groesse)
- [x] Config-System (auto-save/load config.json)
- [x] Velocity Prediction (2 Frames voraus)
- [x] Dynamic Strength (anpassbar nach Distanz)
- [x] Sticky Target (bleibt auf aktuellem Ziel)
- [x] Dead-Body Filter, Sky-Filter, Ground-Loot Filter
- [x] Titan Two Code vorbereitet (Python + GPC + Anleitung)
- [x] Macro-Definitionen (Dropshot, Snaking, Slide-Cancel, Bunny Hop, Auto-Fire, YY)
- [x] Anti-Recoil Profile (11 Waffen-Profile)
- [x] Nano/Full Modell-Switch

## In Arbeit / Wartend
- [ ] Titan Two Hardware-Integration (Geraet kommt 08.04.2026)
- [ ] Scuf Envision Pro + Titan Two Merger
- [ ] Anti-Recoil Feintuning (Werte pro Waffe im Spiel testen)
- [ ] Macro Timing Feintuning (Werte im Spiel anpassen)

## Backlog
- [ ] Custom BO7 YOLO Modell trainieren (bessere Erkennung)
- [ ] Triggerbot (erst nach Teammate-Erkennung zuverlaessig)
- [ ] Aim-Humanisierung (Bezier-Kurven, Random-Delays)
- [ ] Waffen-Profile in Config (verschiedene Settings pro Waffe)
- [ ] Kill-Feed Reader (Ziel wechseln nach Kill)
