# Xbox Vision AI + eRayz Audio — PRD

## Ziel
1. **Aimbot**: Computer Vision AI fuer RemotePlay Xbox (BO7/Warzone). Erkennt Gegner via YOLO, gibt sanfte Aim-Hilfe, laeuft lokal auf Windows PC mit Capture Card.
2. **eRayz Audio**: Competitive Audio Tool fuer Warzone/BO7. Real-time DSP mit Footstep Enhancement, Gunfire Suppression, Dynamic Compression. Kommerzielles Produkt.

## Hardware-Kette
AVerMedia GC571 -> OpenCV Capture -> YOLO ONNX (DirectML) -> KMBox Net (aktuell) / Titan Two (pending)

## Erledigte Features — Aimbot
- [x] Threaded Capture + Inference (70-90 FPS)
- [x] YOLOv8 ONNX mit DirectML GPU
- [x] Custom BO7 V4/V5 Modell
- [x] Dead-Body Filter, Sky-Filter, Ground-Filter, Loot-Box Filter
- [x] Teammate-Schutz via Namensschild-Farberkennung
- [x] Soft Assist Aiming (sanftes Nudging)
- [x] ROI Cropping (640x640 Mitte)
- [x] Config Hotkeys + Overlay Toggle
- [x] Sticky Aim Close-Fight
- [x] Anti-Oszillation Logik (quadratische Daempfung)
- [x] Titan Two Scripts vorbereitet (titan_two.py, aimbot_gpc.gpc)

## Erledigte Features — eRayz Audio
- [x] SURGICAL PRO v6 — 9-Band Parametric EQ (aktiv seit Apr 2026)
  - 3x DT 990 Pro Harman-Korrektur (Mid Fill 1.2kHz, Spike 8.2kHz, Sibilance 6kHz)
  - 4x Chirurgische Step-Boosts (Body 250Hz, Texture 2kHz, Direction 3.17kHz, Clarity 4.5kHz)
  - 2x Cuts (Mud 500Hz, Gunfire 5.5kHz)
  - HP 80Hz + LP 10kHz
- [x] Dynamic Compression (3.5:1 @ -22dB) + Noise Gate
- [x] Spatial Audio Widening (1.5x)
- [x] Live Step-Radar (OpenCV)
- [x] Duplex WASAPI + Ring-Buffer Fallback
- [x] Hotkeys fuer alle Parameter
- [x] Geraete-Config wird gespeichert (JSON)
- [x] Professionelle Distribution Scripts v6 (warzone_audio_setup.bat, warzone_audio_start.bat)
- [x] TinyURL-Maskierung fuer kommerzielle Verteilung

## Offene Issues
- [ ] Audio Stuttering im Warzone Main Menu (P1 — wartet auf Sound Blaster Z / ASIO)
- [ ] KMBox/XIM Deadzone Problem (P0 — wird durch Titan Two geloest)
- [ ] Aimbot Feintuning Strength 1.1 / max_move 38 — USER VERIFICATION PENDING

## Naechste Schritte
1. **User Verification** — SURGICAL PRO v6 Audio-Preset testen
2. **Titan Two Setup** — Hardware anschliessen, GPC Script laden (wartet auf Hardware)
3. **Sound Blaster Z ASIO** — ASIO Backend (wartet auf Hardware)
4. Sensitivity-Tuning mit Titan Two

## Backlog (P2)
- [ ] Besseres YOLO-Modell (mAP50 > 0.65)
- [ ] Triggerbot (erst bei 100% Teammate-Schutz)
- [ ] Aim-Humanisierung (Bezier-Kurven)
- [ ] Kill-Feed Reader
- [ ] Scuf Envision Pro Full Gameplay Passthrough

## Architektur
```
/app/
  aimbot_direct.py           # Core aimbot (v14)
  competitive_audio.py       # eRayz Audio (SURGICAL PRO v6, 9-Band EQ, 48kHz/256)
  warzone_audio_setup.bat    # Installer v6
  warzone_audio_start.bat    # Launcher v6
  titan_two.py               # (Pending) Titan Two API
  aimbot_gpc.gpc             # (Pending) Titan Two GPC Script
  backend/
    server.py                # FastAPI (Download-API)
    yolo_onnx.py             # ONNX Runtime Inference
    kmbox_net.py             # KMBox Net UDP Client
```

## Stand: Apr 2026
- Audio: SURGICAL PRO v6 aktiv (User-Wahl)
- Aimbot: Stabil, wartet auf Titan Two
- Blocker: Hardware-Lieferung (Titan Two, Sound Blaster Z)
