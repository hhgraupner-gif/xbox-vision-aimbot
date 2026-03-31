# AIMBOT v3 - FPS-KI Modell Update

## Was hat sich geaendert?

### Neues FPS-KI Modell (statt generischem COCO)
Das alte Modell (`yolov8n.onnx`) war ein generisches COCO-Modell mit 80 Klassen.
Es hat **alles** als "person" erkannt: Waffen, tote Koerper, Rauch, etc.

Das neue Modell wurde auf **17.000+ FPS-Spielbilder** trainiert und kennt diese Klassen:

| Klasse | Was es ist | Aimbot-Ziel? |
|--------|-----------|:------------:|
| `player` | Lebender Spieler | JA |
| `bot` | Bot/NPC | JA |
| `head` | Kopf eines Spielers | JA (Headshot!) |
| `weapon` | Waffe am Boden | NEIN |
| `dead_body` | Toter Koerper | NEIN |
| `smoke` | Rauchgranate | NEIN |
| `fire` | Feuer/Explosion | NEIN |

### Zwei Modell-Varianten
- **nano** (6 MB, 320x320) - Schneller, weniger genau. Ideal fuer hohe FPS.
- **standard** (22 MB, 640x640) - Langsamer, aber genauer. Besser fuer Erkennung.

Im Spiel mit `M` wechseln!

### Headshot-Priorisierung
Wenn das Modell einen **Kopf** erkennt, wird dieser bevorzugt angezielt.

---

## Anleitung

### 1. Git Pull
```bash
cd C:\Pfad\zu\deinem\Projekt
git pull
```

### 2. Abhaengigkeiten (falls noetig)
```bash
pip install onnxruntime-directml opencv-python numpy
```

### 3. Starten
**Option A:** Doppelklick auf `START_AIMBOT.bat`

**Option B:** Direkt in der Kommandozeile:
```bash
python aimbot_direct.py
```

### 4. Tasten im Spiel
| Taste | Funktion |
|-------|----------|
| `S` | Screenshot-Sammlung an/aus (fuer Custom-Training) |
| `M` | Modell wechseln: nano <-> standard |
| `Q` | Beenden |

### 5. Feintuning
In `aimbot_direct.py` oben die Einstellungen anpassen:
```python
MODEL_MODE = "nano"        # oder "standard"
CONFIDENCE = 0.40          # Hoeher = weniger falsche Erkennungen
MIN_TARGET_SIZE = 200      # Mindestgroesse der Erkennung
AIM_SENSITIVITY = 0.50     # Maus-Empfindlichkeit
SMOOTHING = 0.55           # Glaettung (0=direkt, 1=sehr sanft)
MAX_MOVE = 30              # Max Pixel-Bewegung pro Frame
DEADZONE = 40              # Pixel um Fadenkreuz ignorieren
PREFER_HEADSHOTS = True    # Kopf-Erkennungen bevorzugen
```

---

## Dateien

| Datei | Groesse | Beschreibung |
|-------|---------|-------------|
| `backend/sunxds_nano_320.onnx` | 6 MB | FPS-Modell (schnell) |
| `backend/sunxds_640.onnx` | 22 MB | FPS-Modell (genau) |
| `backend/yolov8n.onnx` | - | Altes COCO-Modell (Backup) |
| `aimbot_direct.py` | - | Haupt-Skript v3 |
| `backend/yolo_onnx.py` | - | YOLO Inference Engine |
| `backend/kmbox_net.py` | - | KMBox Net Treiber |

---

## Falls das Modell noch nicht perfekt ist

Die FPS-Modelle sind auf CS:GO, Battlefield, Warface, Destiny 2 trainiert.
Fuer Black Ops 7 / Warzone koennte ein Custom-Modell noch besser sein.

### Custom-Modell trainieren:
1. Im Spiel `S` druecken um Screenshots zu sammeln
2. Screenshots werden in `training_data/` gespeichert
3. Auf [Roboflow](https://roboflow.com) hochladen und labeln
4. Als YOLOv8 trainieren und als ONNX exportieren
5. In `backend/` kopieren und in `aimbot_direct.py` den Pfad aendern
