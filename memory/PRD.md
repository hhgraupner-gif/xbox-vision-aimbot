# Xbox Vision AI — Aimbot PRD

## Ziel
Computer Vision AI fuer RemotePlay Xbox (BO7/Warzone). Erkennt Gegner via YOLO, gibt sanfte Aim-Hilfe ("Soft Assist"), laeuft lokal auf Windows PC mit Capture Card.

## Hardware-Kette
AVerMedia GC571 → OpenCV Capture → YOLO ONNX (DirectML) → KMBox Net (aktuell) / Titan Two (ab Freitag)

## Erledigte Features
- [x] Threaded Capture + Inference (70-90 FPS)
- [x] YOLOv8 ONNX mit DirectML GPU
- [x] Custom BO7 V4/V5 Modell
- [x] Dead-Body Filter (Breite > Hoehe)
- [x] Sky-Filter (obere 10%) + Ground-Filter (untere 12%)
- [x] Loot-Box Filter (min_box_height)
- [x] Teammate-Schutz via Namensschild-Farberkennung (Blau/Gruen/Gelb/Orange)
- [x] Soft Assist Aiming (sanftes Nudging statt Hard-Lock)
- [x] ROI Cropping (640x640 Mitte)
- [x] Config Hotkeys (STR, FOV, DZ, CD, Conf)
- [x] Overlay Toggle (V-Taste fuer FPS Boost)
- [x] Minimap Reader (Legacy)
- [x] Titan Two Scripts vorbereitet (titan_two.py, aimbot_gpc.gpc)
- [x] **Sticky Aim Close-Fight** (Abgestufte Ziel-Klebrigkeit basierend auf Bbox-Groesse, 2026-04-09)

## Offene Issues
- [ ] Leichtes Jittern bei engen Ziel-Clustern (P1 — durch Sticky Aim stark reduziert)
- [ ] KMBox/XIM Deadzone Problem (P0 — wird durch Titan Two geloest)
- [ ] Iron Sights verdecken YOLO Detection (P2 — Workaround: Red Dot benutzen)

## Naechste Schritte
1. **Titan Two Setup** (Freitag) — Hardware anschliessen, Gtuner IV konfigurieren, GPC Script laden
2. Sensitivity-Tuning mit Titan Two (1:1 Analog-Stick statt Maus-Emulation)
3. Anti-Recoil Profile im Spiel testen
4. Macro-Timing (Dropshot, Snaking, Slide-Cancel)

## Backlog (P2)
- [ ] Besseres YOLO-Modell (mAP50 > 0.65)
- [ ] Triggerbot (erst bei 100% Teammate-Schutz)
- [ ] Aim-Humanisierung (Bezier-Kurven)
- [ ] Kill-Feed Reader
