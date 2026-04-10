# Xbox Vision AI + eRayz Audio — PRD

## Ziel
1. **Aimbot**: Computer Vision AI fuer RemotePlay Xbox (BO7/Warzone). Erkennt Gegner via YOLO, gibt sanfte Aim-Hilfe ("Soft Assist"), laeuft lokal auf Windows PC mit Capture Card.
2. **eRayz Audio**: Competitive Audio Tool fuer Warzone/BO7. Real-time DSP mit Footstep Enhancement, Gunfire Suppression, Dynamic Compression. Kommerzielles Produkt.

## Hardware-Kette
AVerMedia GC571 -> OpenCV Capture -> YOLO ONNX (DirectML) -> KMBox Net (aktuell) / Titan Two (pending)

## Erledigte Features — Aimbot
- [x] Threaded Capture + Inference (70-90 FPS)
- [x] YOLOv8 ONNX mit DirectML GPU
- [x] Custom BO7 V4/V5 Modell
- [x] Dead-Body Filter (Breite > Hoehe)
- [x] Sky-Filter (obere 10%) + Ground-Filter (untere 12%)
- [x] Loot-Box Filter (min_box_height)
- [x] Teammate-Schutz via Namensschild-Farberkennung
- [x] Soft Assist Aiming (sanftes Nudging statt Hard-Lock)
- [x] ROI Cropping (640x640 Mitte)
- [x] Config Hotkeys (STR, FOV, DZ, CD, Conf)
- [x] Overlay Toggle (V-Taste fuer FPS Boost)
- [x] Sticky Aim Close-Fight (Abgestufte Ziel-Klebrigkeit)
- [x] Titan Two Scripts vorbereitet (titan_two.py, aimbot_gpc.gpc)
- [x] Anti-Oszillation Logik (quadratische Daempfung)
- [x] Velocity Prediction (deaktiviert — verursacht Pendeln mit XIM)

## Erledigte Features — eRayz Audio
- [x] Real-time Audio DSP Engine (sounddevice + scipy)
- [x] 4-Band Multiband Pro Preset (User-getestet, bestes Ergebnis)
- [x] Per-Band Compression mit konfigurierbaren Thresholds/Ratios
- [x] Transient Enhancer fuer Step Detail
- [x] DT 990 Pro Treble Taming (8kHz Notch)
- [x] Spatial Audio Widening
- [x] Live Step-Radar (OpenCV)
- [x] Duplex WASAPI Stream + Ring-Buffer Fallback
- [x] Hotkeys fuer alle Parameter (Band A-D, Comp, Gain, Spatial)
- [x] Geraete-Config wird gespeichert (JSON)
- [x] Professionelle Distribution Scripts (warzone_audio_setup.bat, warzone_audio_start.bat)
- [x] TinyURL-Maskierung fuer kommerzielle Verteilung
- [x] Multi-Page Setup Flow mit ASCII-Logo, Fortschrittsanzeige, Fehlerbehandlung

## Offene Issues
- [ ] Audio Stuttering im Warzone Main Menu (P1 — wartet auf Sound Blaster Z / ASIO)
- [ ] KMBox/XIM Deadzone Problem (P0 — wird durch Titan Two geloest)
- [ ] Aimbot Feintuning Strength 1.1 / max_move 38 — USER VERIFICATION PENDING

## Naechste Schritte
1. **User Verification** — Audio Multiband Pro Preset und Aimbot Tracking testen
2. **Titan Two Setup** — Hardware anschliessen, Gtuner IV, GPC Script laden (wartet auf Hardware)
3. **Sound Blaster Z ASIO** — ASIO Backend fuer stotterfreies Audio (wartet auf Hardware)
4. Sensitivity-Tuning mit Titan Two
5. Anti-Recoil Profile testen

## Backlog (P2)
- [ ] Besseres YOLO-Modell (mAP50 > 0.65)
- [ ] Triggerbot (erst bei 100% Teammate-Schutz)
- [ ] Aim-Humanisierung (Bezier-Kurven)
- [ ] Kill-Feed Reader
- [ ] Scuf Envision Pro Full Gameplay Passthrough (scuf_passthrough.py)

## Architektur
```
/app/
  aimbot_direct.py           # Core aimbot (v14, Anti-Oszillation, Sticky Aim)
  competitive_audio.py       # eRayz Audio (Multiband Pro, 48kHz)
  warzone_audio_setup.bat    # Installer fuer Kunden
  warzone_audio_start.bat    # Launcher fuer Kunden
  titan_two.py               # (Pending) Titan Two API
  aimbot_gpc.gpc             # (Pending) Titan Two GPC Script
  backend/
    server.py                # FastAPI (Download-API, Settings, YOLO)
    yolo_onnx.py             # ONNX Runtime Inference
    kmbox_net.py             # KMBox Net UDP Client
    bo7_v5_640.onnx          # Custom BO7 V5 Modell
    sunxds_0.5.6.onnx        # FPS Modell (Primary)
```

## Stand: Feb 2026
- Projekt-Gesundheit: Stabil (Audio + Aimbot funktional)
- Blocker: Hardware-Lieferung (Titan Two, Sound Blaster Z)
