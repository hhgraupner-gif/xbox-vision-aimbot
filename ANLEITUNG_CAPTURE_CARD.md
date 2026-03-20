# AVerMedia GC571 Capture Card - Setup Anleitung

## Warum Capture Card?
Ohne Capture Card wuerde der Aimbot den eigenen PC-Bildschirm aufnehmen (Spiegeleffekt).
Die Capture Card nimmt direkt das Xbox-Bild auf, ohne Verzoegerung.

## Voraussetzungen
- AVerMedia GC571 (PCIe, eingebaut in deinen PC)
- Neueste Treiber von: https://www.avermedia.com/support/download
- HDMI Kabel: Xbox -> GC571 HDMI-In
- Python + OpenCV installiert

## Schritt 1: Treiber installieren
```
1. Gehe zu: https://www.avermedia.com/support/download
2. Suche "GC571"
3. Lade Treiber + Firmware Update Tool runter
4. Installieren + PC neustarten
5. In Geraete-Manager pruefen: "AVerMedia" sollte unter Video erscheinen
```

## Schritt 2: Device Index finden
Speichere dieses Skript als `test_capture.py` und fuehre es aus:

```python
import cv2

print("Suche Capture Devices...")
for i in range(10):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if cap.isOpened():
        ret, frame = cap.read()
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"  Device {i}: OK - {int(w)}x{int(h)}")
        cap.release()
    else:
        print(f"  Device {i}: Nicht verfuegbar")

print("\nNotiere dir die Device-Nummer der Capture Card!")
print("Tipp: Die Capture Card hat meist 1920x1080 Aufloesung")
```

Ausfuehren mit:
```
python test_capture.py
```

## Schritt 3: Im Aimbot konfigurieren
```
1. AIMBOT.html oeffnen
2. "Capture Card" Karte in der Sidebar finden
3. "Capture Card aktiv" einschalten
4. Device Index auf die richtige Nummer setzen
5. "CAPTURE CARD TESTEN" klicken
6. Wenn OK: Gruen mit Aufloesung angezeigt
```

## Schritt 4: Live Test
```
1. Xbox anschalten, Spiel starten
2. HDMI von Xbox muss in der GC571 stecken
3. Demo Mode AUS schalten
4. START druecken
5. Du solltest jetzt das Xbox-Bild im Aimbot sehen
```

## Troubleshooting
- "Device konnte nicht geoeffnet werden":
  -> Treiber installiert? PC neugestartet?
  -> HDMI Kabel angeschlossen? Xbox an?
  -> Anderen Device Index probieren (0-5)

- "Schwarzes Bild":
  -> Xbox muss ein Spiel/Menu zeigen
  -> cv2.CAP_DSHOW nutzen (wird automatisch gemacht)

- "Niedrige FPS":
  -> Aufloesung auf 1080p setzen (nicht 4K)
  -> cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) fuer weniger Latenz

- "Falscher Device Index":
  -> Fuehre test_capture.py erneut aus
  -> Die GC571 zeigt meist 1920x1080
  -> Webcam zeigt meist 640x480 oder 1280x720
