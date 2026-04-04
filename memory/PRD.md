# Xbox Vision AI — Aimbot v7 PRD

## Problemstellung
Computer Vision AI Aimbot fuer Xbox RemotePlay via XIM Matrix / Titan Two.
Erkennt Gegner in Call of Duty (BO7 / Warzone) und bietet praezises Aim-Assist.

## Hardware-Setup
- Capture Card: AVerMedia GC571
- Input: KMBox Net (192.168.2.188:32778, UUID: C14AE466) → XIM Matrix → Xbox
- GPU: AMD RX 7800 XT (DirectML)
- Controller: Scuf Valor Pro (via XIM Matrix)
- Titan Two: Bestellt (~1 Woche), als Backup/Upgrade

## Architektur v7
```
Capture Card → OpenCV (1920x1080) → YOLO ONNX (DirectML GPU)
    → Kalman-Filter (glaettet YOLO-Jitter)
    → delta / smooth (Profi-Methode!)
    → kmbox_net.move() jeden Frame
    → XIM Matrix (Sync=0, Smooth=0, DZ=0)
    → Xbox Controller
```

## Dateien
```
/app/
├── aimbot_direct.py          # Hauptprogramm v7 (delta/smooth — Profi-Methode)
├── backend/
│   ├── yolo_onnx.py          # YOLO ONNX Inference Wrapper
│   ├── kmbox_net.py          # KMBox Net UDP Client
│   ├── sunxds_0.5.6.onnx     # SunOner FPS Modell (aktiv)
│   ├── sunxds_640.onnx       # SunOner FPS 640px
│   ├── sunxds_nano_320.onnx  # SunOner FPS Nano 320px
│   └── server.py             # Download-API fuer PowerShell
```

## Implementiert (v7) — Profi-Methode
- [x] Aim-Mathe: `delta / smooth` (wie DMA-Aimbots, SunOner, Axiom)
- [x] `kmbox_net.move()` jeden Frame (kein move_auto, kein Cooldown)
- [x] Kalman-Filter (nur fuer YOLO-Jitter-Glaettung)
- [x] SunOner FPS Modell (30K FPS-Game Bilder)
- [x] Leichen-Filter, Himmel-Filter, Boden-Filter
- [x] FOV-Begrenzung (250px default)
- [x] 2 Profile (Assist: smooth=7 / Aimbot: smooth=4)
- [x] Live-Anpassung (Smooth, FOV, Conf, SpeedXY)
- [x] Head-Bonus Targeting

## Pending (User-Test noetig)
- [ ] User muss v7 herunterladen und testen
- [ ] XIM Matrix Settings umstellen (Sync=0, Smooth=0, DZ=0, Linear)
- [ ] XIM Velocity Calibration durchfuehren
- [ ] Smooth-Faktor feinjustieren (Tasten 5/6)

## XIM Matrix Settings (KRITISCH)
| Setting | Wert |
|---|---|
| Synch | 0 (Off) |
| Smoothing | 0 |
| Inner Deadzone | 0 |
| Aiming Curve | Linear |
| In-Game Sensitivity | Maximum |

## Upcoming (P1)
- [ ] Anti-Recoil (konstante Y-Korrektur beim Schiessen)
- [ ] Triggerbot (Auto-Schuss wenn Fadenkreuz auf Ziel)
- [ ] Titan Two Support (wenn Geraet ankommt)

## Backlog (P2)
- [ ] Scuf Envision Pro Passthrough
- [ ] Custom YOLO Training (BO7-spezifisch)
- [ ] Bezier-Kurven fuer menschlichere Mausbewegung

## Technische Details v7
- Aim-Formel: move_x = (target_x - center_x) / smooth * speed_x
- Kalman State: [x, y, vx, vy] — glaettet YOLO Detection Jitter
- Smooth-Faktor: 3-8 (einstellbar, Profis nutzen 3-6)
- Max Move/Frame: 80px (Sicherheitslimit)
- Deadzone: 4-8px (je nach Profil)
