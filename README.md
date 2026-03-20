# Xbox Vision AI - Aimbot für Xbox Remote Play

## 🎮 Scuf Valor Pro Edition

Computer Vision Aimbot mit YOLO-Gegnererkennung für Xbox Remote Play.

---

## ⚡ Schnellstart

### 1. Installation (einmalig)
```
Doppelklick auf: INSTALL.bat
```
Installiert alle benötigten Pakete und lädt das YOLO-Modell.

### 2. App starten
```
Doppelklick auf: START_AIMBOT.bat
```
Startet Backend, Frontend und öffnet den Browser.

### 3. App stoppen
```
Doppelklick auf: STOP_ALL.bat
```
Beendet alle Prozesse.

---

## 🕹️ Controller-Belegung (Scuf Valor Pro)

| Taste | Funktion |
|-------|----------|
| **LB** | Aim Assist Ein/Aus |
| **LT** | ADS - Aim Assist aktiv beim Zielen |
| **RT** | Schießen |
| **RS Click** | Snap zum nächsten Ziel |
| **RB** | Zwischen Zielen wechseln |
| **Back/View** | Overlay Ein/Aus |
| **LB + RB** | 🚨 NOTAUS - Alles deaktivieren |

### Scuf Paddle Empfehlung:
- Paddle 1 → A (Quick Aim)
- Paddle 2 → B (Target Lock)

---

## 📋 Spielablauf

1. **Xbox einschalten** → Spiel starten (Single-Player!)
2. **PC:** Xbox App → Remote Play → **Vollbild**
3. **START_AIMBOT.bat** ausführen
4. Im Browser: **Demo Mode AUS** → **START** drücken
5. **LB drücken** = Aim Assist aktivieren
6. Spielen! 🎯

---

## ⚙️ Einstellungen

### Game Profiles
- **Default** - Standard
- **Call of Duty: Warzone** - Schnelle Ziele, hohe Präzision
- **Call of Duty: Black Ops 7** - Maximale Reaktion
- **Call of Duty: Zombies** - Horden-Modus

### In der App anpassen:
- **Settings** → Detection, Aim Assist, Monitor
- **Controller Setup** → Button Bindings ändern

---

## 🖥️ Voraussetzungen

- Windows 10/11
- Python 3.8+
- Node.js 16+
- Xbox App mit Remote Play
- Scuf Valor Pro (oder anderer Xbox Controller)

---

## ⚠️ Wichtig

- **NUR für Single-Player / Private Matches verwenden!**
- Online-Nutzung kann zu Account-Sperren führen
- Controller muss am PC angeschlossen sein
- Xbox Remote Play im Vollbild für beste Ergebnisse

---

## 🔧 Troubleshooting

### "Python nicht gefunden"
→ Python installieren: https://python.org/downloads/
→ Bei Installation: "Add to PATH" anhaken!

### "Node.js nicht gefunden"  
→ Node.js installieren: https://nodejs.org/

### "Controller nicht erkannt"
→ Controller per USB verbinden (stabiler als Wireless)
→ Windows erkennt Xbox Controller automatisch

### "Erkennung funktioniert nicht"
→ Richtigen Monitor in Settings auswählen
→ Xbox Remote Play muss im Vollbild sein
→ "Demo Mode" muss AUS sein

---

## 📁 Projektstruktur

```
xbox-vision-ai/
├── START_AIMBOT.bat    ← App starten
├── INSTALL.bat         ← Einmalige Installation
├── STOP_ALL.bat        ← Alles stoppen
├── backend/            ← Python/FastAPI Server
│   ├── server.py       ← Hauptserver
│   ├── controller.py   ← Controller-Input
│   └── requirements.txt
└── frontend/           ← React UI
    └── src/
        └── pages/
            ├── Dashboard.jsx
            ├── Settings.jsx
            └── ControllerSettings.jsx
```

---

Made with ❤️ for Gaming
