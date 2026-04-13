# TITAN TWO — Input Translator Setup (Schritt fuer Schritt)

## Deine Hardware-Kette
```
PC --[Ethernet]--> KMBox Net --[USB]--> Titan Two (vorne rechts) --[USB]--> Xbox
```

Der KMBox sendet Mausbewegungen. Der Titan Two uebersetzt sie 1:1 in Right-Stick-Bewegungen.
KEIN GPC Script noetig! Alles ueber den eingebauten Input Translator.

---

## SCHRITT 1: Input Translator erstellen

Du hast den Dialog "Create New Input Translator" schon offen.

1. Bei **File Name** eingeben: `kmbox_mouse`
2. Directory kann so bleiben (Downloads oder wo du willst)
3. Klicke **Create**

Es oeffnet sich jetzt der Input Translator Editor mit mehreren Tabs/Sektionen.

---

## SCHRITT 2: Mouse Mapping konfigurieren

Im Input Translator Editor gibt es mehrere Bereiche. Suche den Bereich **"Mouse Mapping"** oder **"Mouse"**.

1. Klicke auf das **gruene +** (Plus-Symbol) um ein neues Mouse-Mapping hinzuzufuegen
2. Konfiguriere folgende Mappings:

### Mapping 1: Mouse X -> Right Stick X
- **Source (Quelle):** `MOUSE_X`
- **Destination (Ziel):** `STICK_1_X` (= Right Stick horizontal)

### Mapping 2: Mouse Y -> Right Stick Y
- **Source (Quelle):** `MOUSE_Y`
- **Destination (Ziel):** `STICK_1_Y` (= Right Stick vertikal)

### Optional: Maustasten
- **Left Click** -> `BUTTON_5` (RT / Fire)
- **Right Click** -> `BUTTON_8` (LT / ADS)

---

## SCHRITT 3: Mouse Converter Einstellungen (WICHTIG!)

Nach dem Mapping gibt es einen Bereich **"Mouse X/Y Converter"** oder aehnlich.
Hier werden die kritischen Einstellungen gemacht:

### Deadzone: 0.00
Das ist der wichtigste Wert! Setze die Deadzone auf **0** oder **0.00**.
Dadurch werden auch kleinste Mausbewegungen vom KMBox registriert.

### Sensitivity (Empfindlichkeit)
- Starte mit **5** (Mitte)
- Kannst du spaeter anpassen je nachdem wie es sich anfuehlt

### Y/X Ratio
- Lass auf **1.0** (gleiche Empfindlichkeit horizontal und vertikal)

### Conversion Curve
- **Linear** ist am besten fuer den Aimbot
- Falls vorhanden: Keine Beschleunigung (keine "Acceleration")

---

## SCHRITT 4: Speichern

1. **Ctrl+S** oder **File -> Save**
2. Die Datei wird als `.git` Datei gespeichert (Gtuner-Format, nicht Git-Versionskontrolle)

---

## SCHRITT 5: Auf Titan Two laden

1. Stelle sicher, dass der Titan Two **verbunden** ist (rechtes Panel zeigt "TITAN TWO [...]")
2. **Drag & Drop** die gespeicherte `.git` Datei auf einen **Memory Slot** im rechten Panel
   - ODER: Rechtsklick auf Memory Slot 1 -> "Load" -> waehle die Datei
3. Der Slot sollte jetzt den Namen "kmbox_mouse" anzeigen (nicht mehr "Empty")

---

## SCHRITT 6: Testen

1. Starte ein beliebiges Xbox-Spiel
2. Bewege die Maus (KMBox sollte die Bewegungen senden)
3. Der rechte Stick auf der Xbox sollte sich jetzt mitbewegen
4. Wenn es sich bewegt: PERFEKT! Weiter mit Aimbot-Tuning

### Falls nichts passiert:
- Pruefe ob der richtige Memory Slot aktiv ist
- Pruefe ob der Titan Two "CONNECTED" anzeigt
- Gehe zu **Device Configuration** Tab (rechtes Panel, unten) und stelle sicher:
  - Output Protocol: Xbox One / Xbox Series
  - USB Port: korrekt zugewiesen

---

## SCHRITT 7: Feintuning (spaeter)

Sobald die Grundbewegung funktioniert, kannst du folgendes anpassen:

### In Gtuner IV (Input Translator):
- **Sensitivity** hoeher = schnellere Stick-Bewegung pro Maus-Pixel
- **Sensitivity** niedriger = praezisere Kontrolle
- **ADS Sensitivity**: Kann separat eingestellt werden (langsamer beim Zielen)

### Im Aimbot (aimbot_direct.py):
- **Strength**: Wie aggressiv der Aimbot nachzieht
- **max_move**: Maximale Pixelbewegung pro Frame
- **deadzone**: Minimaler Abstand bevor der Aimbot reagiert

### Im Spiel (Xbox Settings):
- Setze die In-Game-Sensitivity auf **MAXIMUM**
- Das gibt dem Titan Two den groessten Bewegungsspielraum

---

## Fehlerbehebung

### "Device Memory Slots" zeigt nichts
-> Titan Two USB-Kabel pruefen, neu einstecken

### Titan Two connected aber Maus bewegt nichts
-> Pruefe ob der KMBox USB-Ausgang im **vorderen rechten** USB-Port des Titan Two steckt
-> Das ist der "Input" Port fuer Tastatur/Maus

### Stick bewegt sich aber sehr langsam
-> Sensitivity im Input Translator erhoehen
-> Mouse DPI am KMBox / in der Maus-Software erhoehen

### Stick bewegt sich ruckartig
-> Conversion Curve auf "Linear" stellen
-> Pruefe ob die Sensitivity nicht zu hoch ist

---

## Zusammenfassung der Kette

```
Aimbot (Python)
    |
    | UDP (192.168.2.188:32778)
    v
KMBox Net (empfaengt Mausbewegungen)
    |
    | USB (physische Mausbewegung)
    v
Titan Two (Input Translator: Mouse -> Right Stick, Deadzone=0)
    |
    | USB (Xbox Controller Emulation)
    v
Xbox (sieht normalen Controller-Input)
```
