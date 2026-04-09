# WARZONE COMPETITIVE AUDIO — Setup-Anleitung

## Dein Setup
- **Headset:** Beyerdynamic DT 990 Pro
- **DAC:** SteelSeries GameDAC
- **Capture:** AVerMedia GC571
- **Software:** SteelSeries Sonar (optional)

---

## 1. Installation

```powershell
pip install sounddevice numpy scipy opencv-python
```

## 2. Download

```powershell
cd $env:USERPROFILE\Downloads
Invoke-WebRequest "https://bo7-aimbot-vision.preview.emergentagent.com/api/download/competitive_audio.py" -OutFile "competitive_audio.py"
```

## 3. Geraete pruefen

```powershell
python competitive_audio.py --list
```
Merke dir die Nummern fuer:
- **Input** = AVerMedia GC571 (oder wie sie bei dir heisst)
- **Output** = GameDAC / SteelSeries (wo dein DT 990 Pro dran haengt)

## 4. Starten

```powershell
# Automatisch (Capture Card wird gesucht):
python competitive_audio.py

# Oder manuell mit Device-Nummern:
python competitive_audio.py --input 5 --output 3
```

## 5. SteelSeries Sonar

**Option A (empfohlen):** Sonar auf **FLAT / Aus** stellen.
Unser Tool macht alles besser und gezielter als Sonars Standard-Presets.

**Option B:** Sonar anlassen, aber NUR als Routing-Tool verwenden.
In dem Fall: Sonar EQ komplett flat (alle Regler auf 0).

## 6. In-Game Audio Settings (Warzone)

- **Audio Mix:** Headphones
- **Master Volume:** 80-100
- **Effects Volume:** 100 (WICHTIG!)
- **Music Volume:** 0-5
- **Dialogue Volume:** 30-50
- **Mono Audio:** AUS

---

## Steuerung (im Radar-Fenster)

| Taste | Funktion |
|-------|----------|
| 1 | Preset: Warzone (Standard) |
| 2 | Preset: Multiplayer |
| 3 | Preset: Resurgence |
| Q / W | Footstep Boost runter / hoch |
| E / R | Compression runter / hoch |
| A / S | Bass Cut Frequenz runter / hoch |
| D / F | Spatial Width runter / hoch |
| T / Z | DT990 Treble Tame runter / hoch |
| G / H | Output Gain runter / hoch |
| M | Mute / Unmute |
| V | Radar an / aus |
| P | Alle Settings anzeigen |
| ESC | Beenden (Config wird gespeichert) |

---

## Was das Tool macht

### Signal-Kette:
```
Xbox Audio → Capture Card → [Unser Tool] → GameDAC → DT 990 Pro

1. Highpass 60Hz     = Explosionen/Fahrzeuge weg
2. DT990 Korrektur   = 8kHz Spitze zaehmen (weniger scharf)
3. Gunfire Cut        = 4-6kHz Schuss-Peaks daempfen
4. Footstep Boost     = 80-3500Hz verstaerken (+12dB)
5. Mid-Clarity        = 1-2.5kHz extra Klarheit (+6dB)
6. Noise Gate         = Stille bleibt still
7. Compression 3.5:1  = Leise Steps lauter, Lautes begrenzen
8. Spatial 1.4x       = Breiteres Stereo = bessere Richtung
9. Soft Limiter       = Kein Clipping, kein Ohrenschaden
```

### DT 990 Pro Besonderheiten:
- Die DT 990 Pro hat eine starke **8kHz Treble-Spitze** — Schuesse klingen scharf.
  Wir daempfen das gezielt um -4dB.
- Die **Mitten (1-3kHz)** sind leicht zurueckgesetzt — genau da wo Schritte leben.
  Wir heben das gezielt an.
- Die offene Bauweise gibt gute Raeumlichkeit — Spatial auf 1.4x verstaerkt das.

---

## Presets

| Preset | Step Boost | Bass Cut | Comp | Spatial | Fuer |
|--------|-----------|----------|------|---------|------|
| **Warzone** | +12dB | <60Hz | 3.5:1 | 1.4x | Battle Royale / Ranked |
| **Multiplayer** | +10dB | <50Hz | 2.5:1 | 1.2x | 6v6 / 10v10 |
| **Resurgence** | +14dB | <70Hz | 4.0:1 | 1.5x | Resurgence (mehr Chaos) |

---

## Tipps

- **Radar-Fenster** muss im Vordergrund sein fuer Tasten-Steuerung
- Config wird automatisch in `audio_config.json` gespeichert
- Kann gleichzeitig mit dem Aimbot laufen (separater Prozess)
- Bei Latenz-Problemen: `--no-radar` fuer minimalen CPU-Verbrauch
