# Xbox Vision AI + eRayz Audio — PRD

## Ziel
1. **Aimbot**: Computer Vision AI fuer RemotePlay Xbox (BO7/Warzone). Erkennt Gegner via YOLO, gibt sanfte Aim-Hilfe, laeuft lokal auf Windows PC mit Capture Card.
2. **eRayz Audio**: Competitive Audio Tool fuer Warzone/BO7. Real-time DSP mit Footstep Enhancement, Gunfire Suppression, Dynamic Compression. Kommerzielles Produkt.

## Hardware-Kette (ENDGUELTIG)
```
Xbox HDMI → GC571 HDMI In → GC571 HDMI Out → Monitor
Xbox USB  ← Titan Two OUTPUT
             Titan Two INPUT-A ← KMBox Net USB (Maus vom Aimbot)
             Titan Two INPUT-B ← Xbox Controller (Auth + Gameplay)
             Titan Two PROG   → PC USB (Gtuner IV Setup)
PC Ethernet → KMBox Net (UDP Aimbot-Befehle)
```

## XIM Matrix: ENTFERNT (Titan Two Input Translator ersetzt ihn)

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
- [x] SURGICAL PRO v6 — 9-Band Parametric EQ
- [x] Dynamic Compression (5.0:1 @ -28dB) + Noise Gate
- [x] Spatial Audio Widening (2.0x)
- [x] Live Step-Radar (OpenCV)
- [x] Duplex WASAPI + Ring-Buffer Fallback
- [x] Professionelle Distribution Scripts v6

## Erledigte Features — Titan Two
- [x] SCHLACHTPLAN.txt (komplettes Hardware-Setup A bis Z)
- [x] titan_antirecoil.gpc (Standalone Anti-Recoil + ADS Slowdown)
- [x] TITAN_TWO_SETUP.md (Input Translator Anleitung)
- [x] Deprecated Dateien geloescht (aimbot_gpc.gpc, titan_bridge.py)
- [x] Download-API aktualisiert fuer neue Dateien

## Offene Issues
- [ ] Titan Two Input Translator auf Memory Slot laden (P0 — User muss testen)
- [ ] Aimbot Feintuning fuer Titan Two (P1 — wartet auf Input Translator)
- [ ] Audio Stuttering im Warzone Main Menu (P2 — wartet auf Sound Blaster Z)

## Naechste Schritte
1. User laed Input Translator auf Memory Slot (Drag&Drop oder Ctrl+1)
2. ODER: User laed titan_antirecoil.gpc als Plan B (standalone)
3. Aimbot Retuning fuer 0-Deadzone Hardware
4. Sound Blaster Z ASIO Backend

## Backlog (P2)
- [ ] Besseres YOLO-Modell (mAP50 > 0.65)
- [ ] Aim-Humanisierung (Bezier-Kurven)

## Architektur
```
/app/
  aimbot_direct.py           # Core aimbot v14 (EMA Tracker, KMBox)
  competitive_audio.py       # eRayz Audio (SURGICAL PRO v6)
  titan_two.py               # Titan Two Utilities (pixels_to_stick, Macros)
  titan_antirecoil.gpc       # Standalone Anti-Recoil GPC Script (Plan B)
  SCHLACHTPLAN.txt            # Komplettes Hardware-Setup A bis Z
  TITAN_TWO_SETUP.md          # Input Translator Anleitung
  warzone_audio_setup.bat    # Installer v6
  warzone_audio_start.bat    # Launcher v6
  backend/
    server.py                # FastAPI (Download-API + Detection)
    yolo_onnx.py             # ONNX Runtime Inference
    kmbox_net.py             # KMBox Net UDP Client
```

## Stand: Apr 2026
- Audio: SURGICAL PRO v6 aktiv
- Aimbot: Stabil v14, wartet auf Titan Two Konfiguration
- Titan Two: Hardware da, Input Translator muss auf Slot geladen werden
