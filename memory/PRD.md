# Xbox Vision AI + eRayz Audio — PRD

## Ziel
1. **Aimbot**: Computer Vision AI fuer Xbox. YOLO + KMBox Net + Titan Two.
2. **eRayz Audio ULTIMATE**: Competitive Audio Tool. 16-Band EQ, Gunfire Ducker, HRTF Spatial, DT990 Pro optimiert.

## Hardware-Kette (FINAL)
```
Xbox HDMI → GC571 → GC571 HDMI Out → Monitor
Xbox USB  ← Titan Two OUTPUT
             Titan Two INPUT-A ← KMBox Net Links Unten
             Titan Two INPUT-B ← Xbox Controller USB
             Titan Two PROG   → PC USB (Gtuner IV offen!)
KMBox Net Links Oben → PC USB (Netzwerk)
Audio: Xbox HDMI → GC571 → eRayz DSP → GameDAC (USB) → DT990 Pro
```

## KMBox Net: IP 192.168.2.188, Port 32778, UUID C14AE466

## Erledigte Features — Aimbot
- [x] YOLO ONNX + DirectML (sunxds_0.5.6)
- [x] Titan Two Integration (KMBox → Input Translator)
- [x] Proportional Tracking v2 (sqrt-Kurve, Rate-Limiter 16ms)
- [x] Anti-Oszillation, Teammate-Schutz, Sticky Aim
- [x] Warzone Modell trainiert (warzone_v1.onnx — weniger gut als sunxds)
- [x] GPC Script v3 (Sticky AA 3.5, Rotational AA Timer, Anti-Recoil nur ADS+Fire, Hair Trigger)
- [x] 3 Modell-Modi: fps/warzone/coco (Taste M)

## Erledigte Features — eRayz Audio ULTIMATE v2
- [x] 16-Band Parametric EQ (Pro Warzone Frequenzen)
- [x] DT990 Pro Harman Korrektur (Sub-Cut -10dB, Boom -7dB, Spike -8dB)
- [x] Step Enhancement (Impact 115Hz, Body 250Hz, Presence 770Hz, Texture 1.5kHz, Direction 3.17kHz, Clarity 4.5kHz)
- [x] Cuts: Mud -7dB, Gunfire -9dB, Streaks -4dB
- [x] Gunfire Ducker 5:1 (eigene Schuesse leiser)
- [x] Loudness Compression 6:1 @ -32dB (leise Steps lauter)
- [x] HRTF Spatial (Cross-Feed 0.20, Delay 0.4ms, High-Shelf -3dB)
- [x] Spatial Width 2.0x
- [x] GUI mit Live-Slidern (Tkinter)
- [x] Desktop-Icon + Installer + Uninstaller (.bat)
- [x] Config wird gespeichert (JSON)

## Erledigte Features — Titan Two
- [x] GPC v3: AA + Anti-Recoil + Hair Trigger
- [x] Input Translator (Test and Debug Modus)
- [x] 1ms Response, Inframe Out/In Settings

## Offene Issues
- [ ] Gtuner IV muss immer offen bleiben (Input Translator zu gross fuer Slots)

## Naechste Schritte
- Aimbot .exe bauen (PyInstaller, Doppelklick)
- sunxds_0.7.8 Modell ($5 Patreon, naechsten Monat)
- Sound Blaster Z ASIO Backend

## Backlog
- [ ] Eigenes YOLO Training mit 500+ BO7 Screenshots
- [ ] Besseres YOLO-Modell (mAP50 > 0.65)
- [ ] Aim-Humanisierung (Bezier-Kurven)

## Stand: Apr 2026
- Aimbot: Funktioniert mit Titan Two + KMBox, Proportional Tracking v2
- Audio: eRayz ULTIMATE v2 aktiv, 16-Band EQ, InsuredFrames-Level
- Hardware: Komplett verkabelt und getestet
