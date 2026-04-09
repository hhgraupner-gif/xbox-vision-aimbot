# Xbox Vision AI + eRayz Audio — PRD

## Ziel
1. **Aimbot**: Computer Vision AI fuer RemotePlay Xbox (BO7/Warzone). Erkennt Gegner via YOLO, gibt sanfte Aim-Hilfe ("Soft Assist"), laeuft lokal auf Windows PC mit Capture Card.
2. **eRayz Audio**: Competitive Audio Tool fuer Warzone/BO7. Real-time DSP mit Footstep Enhancement, Gunfire Suppression, Dynamic Compression. Kommerzielles Produkt.

## Hardware-Kette
AVerMedia GC571 → OpenCV Capture → YOLO ONNX (DirectML) → KMBox Net (aktuell) / Titan Two (pending)

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

## Erledigte Features — eRayz Audio
- [x] Real-time Audio DSP Engine (sounddevice + scipy)
- [x] 6-Band Surgical Peaking EQ
- [x] Dynamic Compression + Noise Gate
- [x] Spatial Audio Widening
- [x] Footstep Enhancement (Body/Texture/Edge)
- [x] Gunfire Suppression
- [x] DT 990 Pro Treble Taming (8kHz)
- [x] Map-spezifische Presets (Warzone, Multiplayer, Resurgence, Rebirth, Heavens, RANKED)
- [x] Live Step-Radar (OpenCV)
- [x] Duplex WASAPI Stream + Ring-Buffer Fallback
- [x] Hotkeys fuer alle Parameter
- [x] Geraete-Config wird gespeichert (JSON)
- [x] Professionelle Distribution Scripts (warzone_audio_setup.bat, warzone_audio_start.bat)
- [x] TinyURL-Maskierung fuer kommerzielle Verteilung
- [x] Multi-Page Setup Flow mit ASCII-Logo, Fortschrittsanzeige, Fehlerbehandlung (2026-02)

## Offene Issues
- [ ] Audio Stuttering im Warzone Main Menu (P1 — wartet auf Sound Blaster Z / ASIO)
- [ ] KMBox/XIM Deadzone Problem (P0 — wird durch Titan Two geloest)
- [ ] Leichtes Jittern bei engen Ziel-Clustern (P1 — durch Sticky Aim stark reduziert)

## Naechste Schritte
1. **Titan Two Setup** — Hardware anschliessen, Gtuner IV, GPC Script laden (wartet auf Hardware)
2. **Sound Blaster Z ASIO** — ASIO Backend fuer stotterfreies Audio (wartet auf Hardware)
3. Sensitivity-Tuning mit Titan Two
4. Anti-Recoil Profile testen

## Backlog (P2)
- [ ] Besseres YOLO-Modell (mAP50 > 0.65)
- [ ] Triggerbot (erst bei 100% Teammate-Schutz)
- [ ] Aim-Humanisierung (Bezier-Kurven)
- [ ] Kill-Feed Reader
