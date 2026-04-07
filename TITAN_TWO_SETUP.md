# TITAN TWO SETUP — Anleitung
## Komplette Einrichtung fuer Aimbot + Macros

---

### 1. Was du brauchst
- **Titan Two** Geraet
- **USB Kabel** (Micro-USB)
- **Gtuner IV** Software ([Download](https://www.consoletuner.com/titan-two-downloads/))
- **Xbox Controller** (fuer Authentifizierung)

---

### 2. Hardware-Verkabelung

```
PC (USB PROG) -----> Titan Two <----- Xbox Controller (INPUT-A oder INPUT-B)
                         |
                     OUTPUT (Micro-USB)
                         |
                       Xbox Konsole
```

| Port | Was anschliessen |
|------|-----------------|
| **PROG** | USB-Kabel zum PC (fuer Python-Kommunikation) |
| **INPUT-A** oder **INPUT-B** | Xbox Controller (Authentifizierung) |
| **OUTPUT** | USB-Kabel zur Xbox Konsole |

**WICHTIG:** Der Xbox Controller MUSS angeschlossen sein fuer die Authentifizierung! Die Xbox denkt, der Titan Two ist der Controller.

---

### 3. Gtuner IV einrichten

1. **Gtuner IV installieren** und oeffnen
2. Titan Two per PROG-USB verbinden → LED wird gruen
3. **GPC Script laden:**
   - Gehe zu "GPC Scripting" Tab
   - Oeffne `aimbot_gpc.gpc`
   - Klicke "Compile" (muss fehlerfrei sein)
   - Klicke "Program to Slot" → waehle einen freien Slot
4. **Ausgabe-Protokoll:**
   - Gehe zu "Device Configuration"
   - Setze "Output Protocol" auf **Xbox One / Series X**
   - Speichern

---

### 4. Python-Script starten

Das Aimbot-Script laeuft innerhalb von Gtuner IV:

1. In Gtuner IV → "Device Monitor" oeffnen
2. Python Plugin aktivieren (falls nicht aktiv)
3. Unser Script wird automatisch die Capture Card oeffnen und AI starten
4. Stick-Werte werden direkt an das GPC-Script gesendet

**Starten:**
```powershell
cd $env:USERPROFILE\Downloads
python aimbot_direct.py
```

**HINWEIS:** Der Aimbot erkennt automatisch ob Gtuner IV aktiv ist.
- **Mit Gtuner IV:** Sendet praezise Stick-Werte → perfektes Tracking
- **Ohne Gtuner IV:** Faellt auf KMBox-Modus zurueck

---

### 5. Titan Two vs. KMBox — Der Unterschied

| Feature | KMBox + XIM | Titan Two |
|---------|-------------|-----------|
| Aim-Praezision | Maus → Stick Uebersetzung (verlustbehaftet) | **Direkte Stick-Werte (1:1)** |
| Deadzone-Problem | Ja (XIM filtert kleine Bewegungen) | **Nein** |
| Macros | Nicht moeglich | **Dropshot, Snaking, Slide-Cancel** |
| Anti-Recoil | Schwierig (Maus-basiert) | **Perfekt (Stick-basiert)** |
| Auto-Fire | Nicht moeglich | **Ja (Pistolen/Semi-Auto)** |
| Controller-Auth | Braucht XIM Matrix | **Eingebaute Auth** |
| Latenz | PC→KMBox→XIM→Xbox (3 Hops) | **PC→TitanTwo→Xbox (1 Hop)** |

---

### 6. Verfuegbare Macros

| Macro | Beschreibung | Aktivierung |
|-------|-------------|-------------|
| **Dropshot** | Feuern + Hinlegen gleichzeitig | Hotkey im Script |
| **Snaking** | Schnell hinlegen/aufstehen Loop | Hotkey im Script |
| **Slide-Cancel** | Tac-Sprint → Slide → Jump | Hotkey im Script |
| **Bunny Hop** | Auto-Jump nach Slides | Hotkey im Script |
| **Auto-Fire** | Dauerfeuer fuer Pistolen | Hotkey im Script |
| **YY Swap** | Waffenwechsel Animation-Cancel | Hotkey im Script |

---

### 7. Anti-Recoil Profile

Umschalten per Hotkey im Script:

| Profil | Waffe | Recoil-Kompensation |
|--------|-------|-------------------|
| default | Standard | 8.0 |
| xm4 | XM4 | 10.0 |
| ak74 | AK-74 | 12.0 |
| mp5 | MP5 | 6.0 |
| smg_low | SMG (wenig) | 5.0 |
| smg_high | SMG (viel) | 9.0 |
| ar_low | AR (wenig) | 7.0 |
| ar_high | AR (viel) | 14.0 |
| lmg | LMG | 11.0 |
| pistol | Pistole | 4.0 |
| kar98 | Sniper | 0.0 |

**HINWEIS:** Die Recoil-Werte sind Startwerte. Du musst sie im Spiel feintunen, weil jede Waffe etwas anders ist. Aendere die Werte in `config.json` unter `recoil_profile`.

---

### 8. Troubleshooting

| Problem | Loesung |
|---------|---------|
| Titan Two wird nicht erkannt | USB-Kabel pruefen, Gtuner IV neu starten |
| Xbox erkennt Controller nicht | Xbox Controller an INPUT-B anschliessen |
| GPC Script Fehler | "Compile" nochmal klicken, Fehlermeldung lesen |
| Aim funktioniert nicht | Pruefen ob GPC Script in einem Slot programmiert ist |
| Macros reagieren nicht | Macro-Command im Python-Script pruefen |
| Latenz zu hoch | USB 2.0 Kabel verwenden (kein Hub!) |
