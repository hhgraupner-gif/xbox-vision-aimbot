# XIM Matrix Setup - Aimbot Mausbewegungen zu Xbox Controller

## So funktioniert es
Der Aimbot bewegt die PC-Maus mit PyAutoGUI. Der XIM Matrix faengt diese Mausbewegungen ab und uebersetzt sie in rechten Stick-Bewegungen auf der Xbox. Dein Scuf Valor Pro bleibt gleichzeitig verbunden.

## Voraussetzungen
- XIM Matrix (neueste Firmware von xim.tech/downloads)
- XIM Matrix Manager (PC App, Windows 10/11 x64)
- Bluetooth LE am PC (oder externe Antenne)
- Scuf Valor Pro Controller
- AVerMedia GC571 Capture Card

## Schritt-fuer-Schritt Anleitung

### 1. Firmware Update
```
1. XIM Matrix vom PC trennen
2. XIM Matrix Firmware Updater starten (xim.tech/downloads)
3. Button am XIM Matrix gedrueckt halten + per USB an PC anschliessen
4. LED wird blau -> Update durchfuehren
5. Nach Update: Rainbow-LED = Erfolgreich
```

### 2. XIM Matrix Manager starten
```
1. XIM Matrix Manager (PC Version) oeffnen
2. Lizenz akzeptieren, Game Support runterladen
3. Button am XIM Matrix druecken (blinkt cyan)
4. Manager verbindet sich (blinkt cyan-gruen)
```

### 3. Neues Config erstellen (WICHTIG)
```
1. Oben rechts "..." -> "New Config"
2. Game auswaehlen: "Call of Duty" (oder dein Spiel)
3. Input: "Mouse + Keyboard" auswaehlen
4. Output: "Controller (Xbox)" auswaehlen
5. "New" klicken und 30 Sekunden warten
```

### 4. Mappings konfigurieren (KERNPUNKT)
```
1. Config oeffnen -> Stift-Icon zum Bearbeiten
2. Reiter "Mappings" oeffnen
3. Links: Mouse (als Aiming Source)
4. Rechts: Right Stick (Xbox Controller Output)
5. Mausbewegung -> Right Stick zuweisen
```

### 5. Aim Settings tunen
```
- Sensitivity: Niedrig starten (30-50%), dann hochdrehen
- Deadzone: An das Spiel anpassen (CoD Standard nutzen)
- Smoothing: Mittel-Hoch fuer fluessige Bewegungen
```

### 6. Hardware anschliessen
```
Verbindungsreihenfolge:
1. Scuf Controller -> XIM Matrix (USB Port 3 fuer Auth)
2. PC Maus -> XIM Matrix (oder wird vom PyAutoGUI Skript gesteuert)
3. XIM Matrix -> Xbox (USB)
4. AVerMedia GC571 -> HDMI vom Xbox an der Capture Card
```

### 7. Aimbot starten
```
1. Backend starten: python -m uvicorn server:app --host 0.0.0.0 --port 8001 --reload
2. AIMBOT.html oeffnen
3. Demo Mode AUS
4. Capture Card aktivieren (Device Index testen!)
5. Aimbot aktivieren
6. PyAutoGUI bewegt die Maus -> XIM Matrix uebersetzt zu Stick
```

## Wichtige Tipps
- Die PC-Maus NICHT selbst bewegen wenn der Aimbot aktiv ist
- Am besten eine ZWEITE Maus anschliessen die PyAutoGUI steuert
- Oder: Nur die eine Maus nutzen und den Scuf fuer alles andere
- Sensitivity im Aimbot UND im XIM Matrix muessen zusammenpassen
- Starte mit niedrigen Werten und erhoehe langsam

## Troubleshooting
- "XIM verbindet nicht": Bluetooth pruefen, Firmware update
- "Maus bewegt sich aber Stick nicht": Mapping pruefen (Mouse -> Right Stick)
- "Zu schnell/zu langsam": Sensitivity in BEIDEN Tools anpassen
- "Drift/Twitching": Deadzone im XIM Matrix erhoehen
