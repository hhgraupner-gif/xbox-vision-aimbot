# Xbox Vision AI — PRD

## Problemstellung
Computer Vision AI Aimbot fuer Xbox RemotePlay via XIM Matrix / Titan Two.
Erkennt Gegner in Call of Duty (BO7 / Warzone) und bietet Aim-Assist.

## Hardware-Setup
- Capture Card: AVerMedia GC571
- Input: KMBox Net (192.168.2.188:32778, UUID: C14AE466) → XIM Matrix → Xbox
- GPU: AMD RX 7800 XT (DirectML)
- Controller: Scuf Valor Pro (via XIM Matrix)
- Titan Two: Bestellt (~2 Wochen), fuer vollen Aimbot-Lock

## Aktuelle Architektur (v10 Soft Assist + XIM Config)
```
Capture Card → OpenCV (1920x1080) → YOLO ONNX (SunOner FPS Modell)
    → EMA-Tracker (glaettet Erkennung)
    → Sanfte Schubser (STRENGTH 0.7, DEADZONE 12)
    → kmbox_net.move() jeden Frame
    → XIM Matrix (Klassisch, Weichheit 50, Sync 32, 10K DPI, 15cm/360)
    → Xbox Controller
```

## Dateien
```
/app/
├── aimbot_direct.py          # Soft Assist v10 (sanfte AI-Schubser + XIM)
├── backend/
│   ├── yolo_onnx.py          # YOLO ONNX Inference Wrapper
│   ├── kmbox_net.py          # KMBox Net UDP Client
│   ├── sunxds_0.5.6.onnx     # SunOner FPS Modell (aktiv)
│   └── server.py             # Download-API fuer PowerShell
```

## XIM Matrix Config (AKTUELL — Optimiert)
| Setting | Wert |
|---|---|
| Spiel | Call of Duty: Black Ops 7 [Dynamic] |
| DPI | 10000 |
| Geschwindigkeit | 15.0 cm/360 |
| Smoothing | Klassisch: Weichheit 50, Abbaurate 6.5, Sync 32 |
| Quantisierung | Magnitude 15%, Winkel 10% |
| SAB | Alles auf 0 (direkte Uebersetzung) |
| Smart Actions | Rotational Aim Assist (Aim Magnitude 2.5, Angle 180) |
| ADS Delay | 0ms / 0ms |
| In-Game Sens | Maximum (4.00/4.00) |
| In-Game ADS Multi | 1.00 |
| In-Game Response | Dynamic |
| In-Game FOV | 100 |
| In-Game Deadzone | 5 |

## Implementiert
- [x] SunOner FPS Modell (30K+ FPS-Screenshots)
- [x] Soft Assist v10 (sanfte Schubser MIT XIM zusammen)
- [x] EMA-Tracker fuer stabile Erkennung
- [x] Leichen-Filter, Himmel-Filter, Boden-Filter
- [x] Loot-Filter (Confidence 0.40, Min Box 40px)
- [x] FOV-Begrenzung (180px)
- [x] XIM Matrix Profi-Config (Klassisch + Smart Actions)
- [x] Download-API fuer PowerShell

## Versionshistorie
- v7: Profi-Methode (delta/smooth statt move_auto)
- v8: Snap-Zone + Min-Clamp + sofortiger Lock
- v9: FULL BODY LOCK (delta * STRENGTH)
- v10: SOFT ASSIST (sanfte Schubser MIT XIM, nicht dagegen)

## Wartend
- [ ] Titan Two Lieferung (~2 Wochen)
- [ ] User muss sich mit neuen XIM-Settings einspielen (Muscle Memory)

## Upcoming (nach Titan Two)
- [ ] Titan Two Integration (exakte Stick-Werte statt Maus-Pixel)
- [ ] Voller Aimbot-Lock via Titan Two GPC Scripting
- [ ] Anti-Recoil via Titan Two (native Unterstuetzung)
- [ ] Triggerbot (Auto-Schuss wenn Fadenkreuz auf Ziel)

## Backlog
- [ ] Custom YOLO Training auf BO7 (Screenshots sammeln)
- [ ] Teammate-Erkennung (Freund vs Feind)
- [ ] Scuf Envision Pro Passthrough
