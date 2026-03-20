"""
AVerMedia GC571 Capture Card Finder
Fuehre dieses Skript aus um deine Capture Card zu finden.
Starte in PowerShell: python test_capture.py
"""
import cv2
import sys

print("=" * 50)
print("  AVerMedia GC571 - Capture Card Finder")
print("=" * 50)
print()

found = []
for i in range(10):
    # CAP_DSHOW = DirectShow (beste Windows-Kompatibilitaet)
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if cap.isOpened():
        ret, frame = cap.read()
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        status = "BILD OK" if ret else "KEIN BILD"
        print(f"  Device {i}: {status} - {w}x{h} @ {fps:.0f}fps")
        found.append({"id": i, "w": w, "h": h, "has_frame": ret})
    else:
        print(f"  Device {i}: ---")

print()
if not found:
    print("KEIN Device gefunden!")
    print("-> Treiber installiert?")
    print("-> HDMI Kabel angeschlossen?")
    print("-> Xbox eingeschaltet?")
    sys.exit(1)

# Guess capture card (highest resolution device)
best = max(found, key=lambda x: x["w"] * x["h"])
print(f"Vermutlich Capture Card: Device {best['id']} ({best['w']}x{best['h']})")
print()
print(f"-> Setze im AIMBOT.html den Device Index auf: {best['id']}")
print()

# Show preview
answer = input("Vorschau anzeigen? (j/n): ").strip().lower()
if answer == "j":
    print("Druecke Q zum Beenden...")
    cap = cv2.VideoCapture(best["id"], cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("GC571 Capture (Q = Beenden)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
