# Xbox Vision AI — Aimbot v6 PRD

## Problemstellung
Computer Vision AI Aimbot fuer Xbox RemotePlay via XIM Matrix / Titan Two.
Erkennt Gegner in Call of Duty (BO7 / Warzone) und bietet praezises Aim-Assist.

## Hardware-Setup
- Capture Card: AVerMedia GC571
- Input: KMBox Net (192.168.2.188:32778, UUID: C14AE466) → XIM Matrix → Xbox
- GPU: AMD RX 7800 XT (DirectML)
- Controller: Scuf Valor Pro (via XIM Matrix)
- Titan Two: Bestellt (~1 Woche), Support vorbereitet

## Architektur v6
```
Capture Card → OpenCV (1920x1080) → YOLO11s ONNX (DirectML GPU)
    → Kalman-Filter Tracking → Speed Curves → XIM ADS-Kompensation
    → KMBox Net UDP → XIM Matrix / Titan Two → Xbox Controller
```

## Dateien
```
/app/
├── aimbot_direct.py          # Hauptprogramm v6 (Kalman + Speed Curves)
├── backend/
│   ├── yolo_onnx.py          # YOLO ONNX Inference Wrapper (COCO/FPS/Custom)
│   ├── kmbox_net.py          # KMBox Net UDP Client
│   ├── yolo11s.onnx          # YOLO11s COCO Modell (80 Klassen, 36MB)
│   ├── sunxds_640.onnx       # SunOner FPS Modell (10 Klassen, 22MB)
│   ├── sunxds_nano_320.onnx  # SunOner FPS Nano (10 Klassen, 6MB)
│   ├── bo7_v5_640.onnx       # BO7 Custom (1 Klasse, 37MB)
│   └── server.py             # Download-API fuer PowerShell
```

## Implementiert (v6)
- [x] YOLO11s COCO Modell (frisch exportiert, mAP50 ~70% person)
- [x] Kalman-Filter Tracker (Prediction + Velocity Estimation)
- [x] Speed Curves (Magnet-Effekt: nah=klebrig, weit=Snap)
- [x] XIM ADS-Kompensation (Boost + Minimum-Clamp)
- [x] Adaptive Korrektur-Rate (80-200ms basierend auf Distanz)
- [x] Leichen-Filter (Breite > 1.2 * Hoehe)
- [x] Himmel-Filter (obere 10%) + Boden-Filter (untere 12%)
- [x] FOV-Begrenzung (250px default)
- [x] 2 Profile (Assist / Aimbot)
- [x] 3 Modelle umschaltbar (COCO / FPS / Nano)
- [x] Live-Anpassung (Sensitivity, FOV, Confidence, Speed X/Y)
- [x] Sauberer Code (kein Legacy-Muell)

## Pending (User-Test noetig)
- [ ] User muss v6 herunterladen und testen
- [ ] Speed Curves / Sensitivity Feintuning nach User-Feedback

## Upcoming (P1)
- [ ] Anti-Recoil (konstante Y-Korrektur beim Schiessen)
- [ ] Triggerbot (Auto-Schuss wenn Fadenkreuz auf Ziel)
- [ ] Titan Two Support (wenn Geraet ankommt)

## Backlog (P2)
- [ ] Scuf Envision Pro Passthrough
- [ ] Custom YOLO Training (BO7-spezifisch, mAP50 > 70%)
- [ ] Bézier-Kurven fuer noch menschlichere Mausbewegung

## Technische Details
- Kalman State: [x, y, vx, vy] — Position + Geschwindigkeit
- Prediction: 2 Frames Lookahead fuer bewegende Gegner
- Speed Curves: 6 Stufen interpoliert (15px → 9999px)
- XIM Boost: 3.0x | Minimum: 25px | Maximum: 600px
- KMBox: move_auto() mit adaptiver Dauer (80-200ms)
