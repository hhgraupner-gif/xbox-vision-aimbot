"""
AIMBOT - Standalone Python Script (kein Browser noetig)
Laeuft direkt auf dem PC in einer schnellen Schleife.
Capture Card → YOLO → KMBox Net → XIM Matrix → Xbox

Starten: python aimbot_direct.py
Beenden: Strg+C
"""
import cv2
import numpy as np
import time
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from yolo_onnx import YOLODetector
import kmbox_net

# ============================================================
# EINSTELLUNGEN - Hier anpassen!
# ============================================================
CAPTURE_DEVICE = 0          # Capture Card Device Index (0 oder 3)
KMBOX_IP = "192.168.2.188"
KMBOX_PORT = "32778"        # Vom KMBox Display
KMBOX_UUID = "C14AE466"     # Vom KMBox Display

CONFIDENCE = 0.45           # Erkennungs-Schwelle (0.3 = mehr, 0.6 = weniger)
AIM_SENSITIVITY = 0.25      # Wie stark der Aimbot zieht (0.1 = sanft, 0.5 = stark)
SMOOTHING = 0.80            # Glaettung (0.9 = sehr sanft, 0.5 = direkt)
MAX_MOVE = 15               # Max Pixel pro Frame
DEADZONE = 120              # Pixel vom Zentrum - darunter passiert nichts
AIM_POINT = 0.45            # Wo am Gegner zielen (0.15=Kopf, 0.45=Brust, 0.7=Bauch)
MIN_TARGET_SIZE = 600       # Minimale Box-Groesse in Pixel
LOCK_FRAMES = 2             # Frames bis Target als stabil gilt

SHOW_WINDOW = True          # Vorschau-Fenster anzeigen (True/False)
WINDOW_SCALE = 0.5          # Vorschau-Groesse (0.5 = halbe Groesse)
# ============================================================


class TargetTracker:
    def __init__(self):
        self.ema_x = None
        self.ema_y = None
        self.frames_seen = 0
        self.last_seen = 0
        self.prev_mx = 0.0
        self.prev_my = 0.0

    def update(self, rx, ry, alpha=0.3):
        now = time.monotonic()
        if self.ema_x is None or (now - self.last_seen) > 0.4:
            self.ema_x = float(rx)
            self.ema_y = float(ry)
            self.frames_seen = 1
        else:
            self.ema_x = alpha * rx + (1 - alpha) * self.ema_x
            self.ema_y = alpha * ry + (1 - alpha) * self.ema_y
            self.frames_seen += 1
        self.last_seen = now

    def get(self):
        if self.ema_x is None:
            return None
        return (self.ema_x, self.ema_y)

    def stable(self, n=2):
        return self.frames_seen >= n

    def reset(self):
        self.ema_x = None
        self.ema_y = None
        self.frames_seen = 0
        self.prev_mx = 0.0
        self.prev_my = 0.0


def move_aim(tracker, tx, ty, fw, fh):
    """Berechne und sende Mausbewegung ueber KMBox."""
    cx = fw / 2.0
    cy = fh / 2.0
    dx = tx - cx
    dy = ty - cy
    dist = (dx*dx + dy*dy) ** 0.5

    if dist < DEADZONE:
        tracker.prev_mx *= 0.5
        tracker.prev_my *= 0.5
        return False

    mx = (dx / fw) * AIM_SENSITIVITY * 200
    my = (dy / fh) * AIM_SENSITIVITY * 200

    mx = SMOOTHING * tracker.prev_mx + (1 - SMOOTHING) * mx
    my = SMOOTHING * tracker.prev_my + (1 - SMOOTHING) * my

    mag = (mx*mx + my*my) ** 0.5
    if mag > MAX_MOVE:
        s = MAX_MOVE / mag
        mx *= s
        my *= s

    tracker.prev_mx = mx
    tracker.prev_my = my

    ix = int(round(mx))
    iy = int(round(my))

    if abs(ix) >= 1 or abs(iy) >= 1:
        kmbox_net.move(ix, iy)
        return True
    return False


def draw_overlay(frame, detections, target, fps):
    """Zeichne Boxen und Info auf das Frame."""
    h, w = frame.shape[:2]

    # Fadenkreuz
    cv2.line(frame, (w//2-20, h//2), (w//2+20, h//2), (0, 255, 0), 1)
    cv2.line(frame, (w//2, h//2-20), (w//2, h//2+20), (0, 255, 0), 1)

    # Deadzone Kreis
    cv2.circle(frame, (w//2, h//2), DEADZONE, (50, 50, 50), 1)

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        conf = det["confidence"]
        # Rote Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, f'{conf:.0%}', (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    if target:
        # Ziel-Punkt
        cv2.circle(frame, (int(target[0]), int(target[1])), 8, (0, 255, 255), 2)
        cv2.line(frame, (w//2, h//2), (int(target[0]), int(target[1])), (0, 255, 255), 1)

    # Info
    cv2.putText(frame, f'FPS: {fps}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f'Targets: {len(detections)}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, 'AIMBOT AKTIV', (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return frame


def main():
    print("=" * 50)
    print("  XBOX VISION AI - AIMBOT DIRECT")
    print("=" * 50)
    print()

    # 1. YOLO laden
    onnx_path = os.path.join(os.path.dirname(__file__), 'backend', 'yolov8n.onnx')
    if not os.path.exists(onnx_path):
        print(f"FEHLER: {onnx_path} nicht gefunden!")
        print("Zuerst ONNX Modell erstellen (siehe Anleitung)")
        sys.exit(1)

    print(f"[1/3] YOLO laden: {onnx_path}")
    detector = YOLODetector(onnx_path, conf_threshold=CONFIDENCE)
    print("      YOLO geladen!")

    # 2. KMBox verbinden
    print(f"[2/3] KMBox verbinden: {KMBOX_IP}:{KMBOX_PORT}")
    ret = kmbox_net.init(KMBOX_IP, KMBOX_PORT, KMBOX_UUID)
    if ret == 0:
        print("      KMBox verbunden!")
    else:
        print("      WARNUNG: KMBox nicht verbunden - Aimbot laeuft ohne Output")

    # 3. Capture Card oeffnen
    print(f"[3/3] Capture Card oeffnen: Device {CAPTURE_DEVICE}")
    import platform
    if platform.system() == "Windows":
        cap = cv2.VideoCapture(CAPTURE_DEVICE, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(CAPTURE_DEVICE)

    if not cap.isOpened():
        print(f"FEHLER: Device {CAPTURE_DEVICE} nicht verfuegbar!")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"      Capture Card: {w}x{h}")
    print()
    print("AIMBOT LAEUFT! Druecke Strg+C oder Q zum Beenden.")
    print("=" * 50)

    tracker = TargetTracker()
    ema_alpha = max(0.1, 1.0 - SMOOTHING)

    fps = 0
    frame_count = 0
    fps_time = time.monotonic()

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        fh, fw = frame.shape[:2]

        # YOLO Erkennung
        raw_dets = detector.detect(frame, conf_threshold=CONFIDENCE)

        # Filtern
        detections = []
        best_target = None
        min_dist = float('inf')
        center_x = fw / 2.0
        center_y = fh / 2.0

        for det in raw_dets:
            if det["class_name"] != "person":
                continue

            x1, y1, x2, y2 = det["bbox"]
            bw = x2 - x1
            bh = y2 - y1

            if bw * bh < MIN_TARGET_SIZE:
                continue

            ratio = bh / max(1, bw)
            if ratio < 0.8 or ratio > 5.0:
                continue

            cx = (x1 + x2) // 2
            det["bbox"] = [x1, y1, x2, y2]
            detections.append(det)

            # Zielpunkt
            target_y = y1 + int(bh * AIM_POINT)
            d = ((cx - center_x)**2 + (target_y - center_y)**2) ** 0.5
            if d < min_dist:
                min_dist = d
                best_target = (cx, target_y)

        # Aimbot
        if best_target:
            tracker.update(best_target[0], best_target[1], alpha=ema_alpha)
            if tracker.stable(LOCK_FRAMES):
                pos = tracker.get()
                if pos:
                    move_aim(tracker, pos[0], pos[1], fw, fh)
        else:
            if tracker.frames_seen > 0:
                tracker.frames_seen = max(0, tracker.frames_seen - 1)
            if tracker.frames_seen == 0:
                tracker.reset()

        # FPS zaehlen
        frame_count += 1
        now = time.monotonic()
        if now - fps_time >= 1.0:
            fps = frame_count
            frame_count = 0
            fps_time = now

        # Vorschau Fenster
        if SHOW_WINDOW:
            display = draw_overlay(frame.copy(), detections, best_target, fps)
            if WINDOW_SCALE != 1.0:
                display = cv2.resize(display, None, fx=WINDOW_SCALE, fy=WINDOW_SCALE)
            cv2.imshow('AIMBOT', display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

    cap.release()
    kmbox_net.close()
    if SHOW_WINDOW:
        cv2.destroyAllWindows()
    print("\nAimbot beendet.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAimbot beendet (Strg+C).")
        kmbox_net.close()
        cv2.destroyAllWindows()
