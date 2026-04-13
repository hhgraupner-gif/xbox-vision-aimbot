# Xbox Vision AI + eRayz Audio — PRD

## Ziel
1. **Aimbot**: Computer Vision AI fuer RemotePlay Xbox (BO7/Warzone). Erkennt Gegner via YOLO, gibt sanfte Aim-Hilfe, laeuft lokal auf Windows PC mit Capture Card.
2. **eRayz Audio**: Competitive Audio Tool fuer Warzone/BO7. Real-time DSP mit Footstep Enhancement, Gunfire Suppression, Dynamic Compression. Kommerzielles Produkt.

## Hardware-Kette (AKTUELL)
```
AVerMedia GC571 -> OpenCV Capture -> YOLO ONNX (DirectML) -> KMBox Net -> Titan Two (Input Translator) -> Xbox
```
XIM Matrix wurde entfernt. Titan Two ersetzt die Mouse-to-Stick Translation mit 0-Deadzone.

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
- [x] EMA Tracker v14 (Velocity Prediction, Dynamic Strength)

## Erledigte Features — eRayz Audio
- [x] SURGICAL PRO v6 — 9-Band Parametric EQ (aktiv seit Apr 2026)
  - 3x DT 990 Pro Harman-Korrektur (Mid Fill 1.2kHz, Spike 8.2kHz, Sibilance 6kHz)
  - 4x Chirurgische Step-Boosts (Body 250Hz, Texture 2kHz, Direction 3.17kHz, Clarity 4.5kHz)
  - 2x Cuts (Mud 500Hz, Gunfire 5.5kHz)
  - HP 80Hz + LP 10kHz
- [x] Dynamic Compression (5.0:1 @ -28dB) + Noise Gate
- [x] Spatial Audio Widening (2.0x)
- [x] Live Step-Radar (OpenCV)
- [x] Duplex WASAPI + Ring-Buffer Fallback
- [x] Hotkeys fuer alle Parameter
- [x] Geraete-Config wird gespeichert (JSON)
- [x] Professionelle Distribution Scripts v6 (warzone_audio_setup.bat, warzone_audio_start.bat)
- [x] TinyURL-Maskierung fuer kommerzielle Verteilung

## Offene Issues
- [ ] Titan Two Input Translator konfigurieren (P0 — IN PROGRESS)
- [ ] Aimbot Feintuning fuer Titan Two (P1 — wartet auf Input Translator)
- [ ] Audio Stuttering im Warzone Main Menu (P2 — wartet auf Sound Blaster Z / ASIO)

## Naechste Schritte
1. **Titan Two Input Translator** — Mouse X/Y -> STICK_1_X/Y, Deadzone=0, Memory Slot laden
2. **Aimbot Retuning** — Strength, Smoothing, Deadzone anpassen fuer 0-Deadzone Hardware
3. **Sound Blaster Z ASIO** — ASIO Backend (wartet auf Hardware)

## Backlog (P2)
- [ ] Besseres YOLO-Modell (mAP50 > 0.65)
- [ ] Triggerbot (erst bei 100% Teammate-Schutz)
- [ ] Aim-Humanisierung (Bezier-Kurven)
- [ ] Kill-Feed Reader

## Architektur
```
/app/
  aimbot_direct.py           # Core aimbot v14 (EMA Tracker, KMBox)
  competitive_audio.py       # eRayz Audio (SURGICAL PRO v6, 9-Band EQ, 48kHz/256)
  warzone_audio_setup.bat    # Installer v6
  warzone_audio_start.bat    # Launcher v6
  titan_two.py               # Titan Two Utilities (pixels_to_stick, Macros, Recoil)
  TITAN_TWO_SETUP.md         # Schritt-fuer-Schritt Anleitung Input Translator
  backend/
    server.py                # FastAPI (Download-API)
    yolo_onnx.py             # ONNX Runtime Inference
    kmbox_net.py             # KMBox Net UDP Client
```

## Deprecated (geloescht Apr 2026)
- aimbot_gpc.gpc (GPC Script funktionierte nicht, Input Translator ersetzt es)
- titan_bridge.py (Gtuner Python Bridge, nicht noetig mit Input Translator)

## Stand: Apr 2026
- Audio: SURGICAL PRO v6 aktiv (User-Wahl)
- Aimbot: Stabil v14, wartet auf Titan Two Input Translator Konfiguration
- Hardware: Titan Two angeschlossen, Firmware aktuell, Input Translator wird konfiguriert
