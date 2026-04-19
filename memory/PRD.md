# Xbox Vision AI + eRayz Audio — PRD

## Ziel
1. **Aimbot**: Computer Vision AI fuer Xbox (BO7/Warzone). Erkennt Gegner via YOLO, gibt sanfte Aim-Hilfe via KMBox Net + Titan Two.
2. **eRayz Audio**: Competitive Audio Tool. Kommerzielles Produkt.

## Hardware-Kette (FINAL — FUNKTIONIERT)
```
Xbox HDMI → GC571 HDMI In → GC571 HDMI Out → Monitor
Xbox USB  ← Titan Two OUTPUT
             Titan Two INPUT-A ← KMBox Net Links Unten (Port 1, Maus-Ausgang)
             Titan Two INPUT-B ← Xbox Controller (USB, Authentication)
             Titan Two PROG   → PC USB (Gtuner IV MUSS offen bleiben!)
KMBox Net Links Oben (Port 2) → PC USB (Netzwerk/UDP)
```

## WICHTIG: Gtuner IV muss IMMER offen bleiben
Der Input Translator ist zu gross fuer interne Memory Slots (max 170 bytes).
Er laeuft nur im "Test and Debug" Modus ueber PROG-Kabel.

## KMBox Net Port-Belegung
- Links Unten = Port 1 (Gaming/Ausgang, gibt Strom) → Titan Two INPUT-A
- Links Oben = Port 2 (Netzwerk) → PC USB
- Rechts Oben/Unten = HID Ports (leer lassen)
- IP: 192.168.2.188, Port: 32778, UUID: C14AE466

## Erledigte Features — Aimbot
- [x] YOLO ONNX + DirectML GPU (70-90 FPS)
- [x] Custom BO7 V5 Modell
- [x] Titan Two Integration via KMBox + Input Translator
- [x] Sanftes Tracking (konstante Geschwindigkeit, Rate-Limiter 18ms)
- [x] Anti-Oszillation (pendelt erkennen + stoppen)
- [x] Teammate-Schutz, Dead-Body Filter, Sky/Ground Filter
- [x] Sticky Aim, ROI Cropping, Config Hotkeys

## Erledigte Features — eRayz Audio
- [x] SURGICAL PRO v6 (9-Band EQ, Compression, Spatial)

## Erledigte Features — Titan Two
- [x] titan_antirecoil.gpc (Standalone, kompiliert OK)
- [x] Input Translator Setup (Online Resources → Test and Debug)
- [x] SCHLACHTPLAN.txt + FEINTUNING_GUIDE.txt

## Naechste Schritte (Feintuning)
1. Speed-Werte im Aimbot anpassen (natuerlicher fuer Killcam)
2. Gtuner Sensitivity tunen (4.0-6.0 Range)
3. Rate-Limiter anpassen (18-25ms)
4. FOV optimieren (150-250)

## Backlog
- [ ] Sound Blaster Z ASIO Backend
- [ ] Besseres YOLO-Modell (mAP50 > 0.65)

## Architektur
```
/app/
  aimbot_direct.py           # Core aimbot v15 (Titan Two Tracking)
  competitive_audio.py       # eRayz Audio (SURGICAL PRO v6)
  titan_two.py               # Titan Two Utilities
  titan_antirecoil.gpc       # Standalone Anti-Recoil
  SCHLACHTPLAN.txt            # Hardware Setup A-Z
  FEINTUNING_GUIDE.txt       # Tuning Guide fuer Aimbot + Gtuner
  TITAN_TWO_SETUP.md          # Input Translator Anleitung
  backend/
    server.py                # FastAPI
    yolo_onnx.py             # ONNX Runtime
    kmbox_net.py             # KMBox Net UDP
```

## Stand: Apr 2026
- Aimbot: FUNKTIONIERT mit Titan Two! Feintuning noetig.
- Audio: SURGICAL PRO v6 aktiv
- Hardware: Titan Two + KMBox + GC571 komplett verkabelt
