"""
AIMBOT DIRECT v2 - Mit ADS-Erkennung + Screenshot-Sammlung
Capture Card → YOLO → KMBox Net → XIM Matrix → Xbox

Features:
- Visuelle ADS-Erkennung (erkennt wenn du zielst)
- Screenshot-Sammlung fuer Custom-Modell Training
- Verbesserte Tracking-Logik

Starten:     python aimbot_direct.py
Beenden:     Q oder Strg+C
Screenshots: S druecken zum Speichern ein/ausschalten
"""
import cv2
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from yolo_onnx import YOLODetector
import kmbox_net

# ============================================================
# EINSTELLUNGEN
# ============================================================
CAPTURE_DEVICE = 0
KMBOX_IP = "192.168.2.188"
KMBOX_PORT = "32778"
KMBOX_UUID = "C14AE466"

# Erkennung
CONFIDENCE = 0.35
MIN_TARGET_SIZE = 400
AIM_POINT = 0.45

# Aimbot - NUR wenn ADS aktiv
AIM_SENSITIVITY = 0.50
SMOOTHING = 0.55
MAX_MOVE = 30
DEADZONE = 40
LOCK_FRAMES = 1

# ADS Erkennung
ADS_DETECTION = True           # ADS visuell erkennen
ADS_ZOOM_THRESHOLD = 12.0      # Schwelle fuer Zoom-Erkennung

# Screenshot Sammlung
COLLECT_SCREENSHOTS = False     # S druecken zum toggeln
SCREENSHOT_INTERVAL = 0.5      # Sekunden zwischen Screenshots
SCREENSHOT_FOLDER = "training_data"

# Anzeige
SHOW_WINDOW = True
WINDOW_SCALE = 0.5
# ============================================================


class ADSDetector:
    """Erkennt ob der Spieler gerade ADS (Aim Down Sight) benutzt.
    Methode: Vergleicht aufeinanderfolgende Frames auf Zoom-Aenderung.
    Wenn ADS gedrueckt wird, zoomt die Kamera rein (FOV wird kleiner).
    """
    def __init__(self):
        self.prev_gray_center = None
        self.ads_active = False
        self.ads_confidence = 0.0
        self.frame_count = 0
        self.baseline_sharpness = None
        self.sharpness_history = []

    def update(self, frame):
        """Analysiere Frame und bestimme ADS-Status."""
        h, w = frame.shape[:2]

        # Zentrale Region (wo Scope/Visier erscheint)
        cx, cy = w // 2, h // 2
        region_size = 200
        center = frame[cy-region_size:cy+region_size, cx-region_size:cx+region_size]

        if center.size == 0:
            return self.ads_active

        gray = cv2.cvtColor(center, cv2.COLOR_BGR2GRAY)

        # Methode 1: Schaerfe messen (ADS = schaerfer wegen Zoom)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()

        self.sharpness_history.append(sharpness)
        if len(self.sharpness_history) > 30:
            self.sharpness_history.pop(0)

        # Baseline nach 10 Frames setzen (hip-fire Schaerfe)
        if len(self.sharpness_history) >= 10 and self.baseline_sharpness is None:
            self.baseline_sharpness = np.median(self.sharpness_history)

        if self.baseline_sharpness is not None and self.baseline_sharpness > 0:
            # Wenn aktuelle Schaerfe deutlich hoeher als Baseline = ADS
            ratio = sharpness / self.baseline_sharpness
            self.ads_confidence = ratio

            if ratio > 1.3:  # 30% schaerfer = wahrscheinlich ADS
                self.ads_active = True
            elif ratio < 1.1:  # Zurueck zu normal
                self.ads_active = False

        # Methode 2: Optischer Fluss (Zoom-Bewegung erkennen)
        if self.prev_gray_center is not None and self.prev_gray_center.shape == gray.shape:
            diff = cv2.absdiff(gray, self.prev_gray_center)
            mean_diff = np.mean(diff)

            # Grosser ploetzlicher Unterschied = ADS Uebergang
            if mean_diff > ADS_ZOOM_THRESHOLD:
                self.ads_active = True
                # Neue Baseline setzen nach ADS-Wechsel
                self.baseline_sharpness = None
                self.sharpness_history.clear()

        self.prev_gray_center = gray.copy()
        self.frame_count += 1

        return self.ads_active


class TargetTracker:
    """Verbessertes Target Tracking mit Prediction."""
    def __init__(self):
        self.ema_x = None
        self.ema_y = None
        self.vel_x = 0.0
        self.vel_y = 0.0
        self.frames_seen = 0
        self.last_seen = 0
        self.prev_mx = 0.0
        self.prev_my = 0.0
        self.locked = False

    def update(self, rx, ry, alpha=0.4):
        now = time.monotonic()
        dt = now - self.last_seen if self.last_seen > 0 else 0.033

        if self.ema_x is None or (now - self.last_seen) > 0.3:
            self.ema_x = float(rx)
            self.ema_y = float(ry)
            self.vel_x = 0.0
            self.vel_y = 0.0
            self.frames_seen = 1
            self.locked = False
        else:
            # Velocity berechnen (fuer Prediction)
            old_x, old_y = self.ema_x, self.ema_y
            self.ema_x = alpha * rx + (1 - alpha) * self.ema_x
            self.ema_y = alpha * ry + (1 - alpha) * self.ema_y

            if dt > 0:
                new_vx = (self.ema_x - old_x) / dt
                new_vy = (self.ema_y - old_y) / dt
                self.vel_x = 0.7 * self.vel_x + 0.3 * new_vx
                self.vel_y = 0.7 * self.vel_y + 0.3 * new_vy

            self.frames_seen += 1

        self.last_seen = now

    def get_predicted(self, lookahead=0.02):
        """Gibt vorhergesagte Position zurueck (kompensiert Latenz)."""
        if self.ema_x is None:
            return None
        px = self.ema_x + self.vel_x * lookahead
        py = self.ema_y + self.vel_y * lookahead
        return (px, py)

    def stable(self, n=1):
        return self.frames_seen >= n

    def reset(self):
        self.ema_x = None
        self.ema_y = None
        self.vel_x = 0.0
        self.vel_y = 0.0
        self.frames_seen = 0
        self.prev_mx = 0.0
        self.prev_my = 0.0
        self.locked = False


def move_aim(tracker, tx, ty, fw, fh):
    cx = fw / 2.0
    cy = fh / 2.0
    dx = tx - cx
    dy = ty - cy
    dist = (dx*dx + dy*dy) ** 0.5

    if dist < DEADZONE:
        tracker.prev_mx *= 0.3
        tracker.prev_my *= 0.3
        return False

    mx = (dx / fw) * AIM_SENSITIVITY * 250
    my = (dy / fh) * AIM_SENSITIVITY * 250

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


def draw_overlay(frame, detections, target, fps, ads_active, collecting, tracker):
    h, w = frame.shape[:2]

    # Fadenkreuz
    color = (0, 255, 255) if ads_active else (0, 255, 0)
    cv2.line(frame, (w//2-25, h//2), (w//2+25, h//2), color, 2)
    cv2.line(frame, (w//2, h//2-25), (w//2, h//2+25), color, 2)

    # Deadzone
    cv2.circle(frame, (w//2, h//2), DEADZONE, (50, 50, 50), 1)

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        conf = det["confidence"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, f'{conf:.0%}', (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    if target:
        tx, ty = int(target[0]), int(target[1])
        cv2.circle(frame, (tx, ty), 10, (0, 255, 255), 3)
        cv2.line(frame, (w//2, h//2), (tx, ty), (0, 255, 255), 2)

    # Status
    cv2.putText(frame, f'FPS: {fps}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f'Targets: {len(detections)}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # ADS Status
    if ads_active:
        cv2.putText(frame, 'ADS AKTIV - AIMBOT AN', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    else:
        cv2.putText(frame, 'Hip-Fire - Aimbot aus', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)

    if tracker.locked:
        cv2.putText(frame, 'TARGET LOCKED', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    if collecting:
        cv2.putText(frame, 'SAMMELT SCREENSHOTS', (w-350, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Tastenbelegung
    cv2.putText(frame, 'S=Screenshots | Q=Beenden', (10, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)

    return frame


def main():
    print("=" * 50)
    print("  XBOX VISION AI - AIMBOT v2")
    print("  ADS-Erkennung + Screenshot-Sammlung")
    print("=" * 50)
    print()

    # Ordner fuer Screenshots
    screenshot_dir = os.path.join(os.path.dirname(__file__), SCREENSHOT_FOLDER)
    os.makedirs(screenshot_dir, exist_ok=True)

    # 1. YOLO
    onnx_path = os.path.join(os.path.dirname(__file__), 'backend', 'yolov8n.onnx')
    if not os.path.exists(onnx_path):
        print(f"FEHLER: {onnx_path} nicht gefunden!")
        sys.exit(1)
    print("[1/3] YOLO laden...")
    detector = YOLODetector(onnx_path, conf_threshold=CONFIDENCE)
    print("      OK!")

    # 2. KMBox
    print(f"[2/3] KMBox verbinden: {KMBOX_IP}:{KMBOX_PORT}")
    ret = kmbox_net.init(KMBOX_IP, KMBOX_PORT, KMBOX_UUID)
    if ret == 0:
        print("      KMBox verbunden!")
    else:
        print("      WARNUNG: KMBox nicht verbunden")

    # 3. Capture Card
    print(f"[3/3] Capture Card: Device {CAPTURE_DEVICE}")
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
    print(f"      Capture: {w}x{h}")
    print()
    print("BEREIT! Druecke S fuer Screenshot-Sammlung an/aus.")
    print("Aimbot aktiviert sich automatisch wenn du ADS drueckst.")
    print("=" * 50)

    tracker = TargetTracker()
    ads_detector = ADSDetector()
    collecting = COLLECT_SCREENSHOTS
    last_screenshot = 0
    screenshot_count = len([f for f in os.listdir(screenshot_dir) if f.endswith('.jpg')])

    fps = 0
    frame_count = 0
    fps_time = time.monotonic()

    while True:
        ret_cap, frame = cap.read()
        if not ret_cap:
            continue

        fh, fw = frame.shape[:2]

        # ADS erkennen
        ads_active = True  # Default: immer an
        if ADS_DETECTION:
            ads_active = ads_detector.update(frame)

        # YOLO Erkennung (immer, fuer rote Boxen)
        raw_dets = detector.detect(frame, conf_threshold=CONFIDENCE)

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

            cx_det = (x1 + x2) // 2
            detections.append(det)

            target_y = y1 + int(bh * AIM_POINT)
            d = ((cx_det - center_x)**2 + (target_y - center_y)**2) ** 0.5
            if d < min_dist:
                min_dist = d
                best_target = (cx_det, target_y)

        # Aimbot - NUR wenn ADS aktiv
        if best_target and ads_active:
            ema_alpha = max(0.15, 1.0 - SMOOTHING)
            tracker.update(best_target[0], best_target[1], alpha=ema_alpha)
            if tracker.stable(LOCK_FRAMES):
                pos = tracker.get_predicted(lookahead=0.03)
                if pos:
                    tracker.locked = True
                    move_aim(tracker, pos[0], pos[1], fw, fh)
        else:
            if not ads_active:
                tracker.reset()
            elif tracker.frames_seen > 0:
                tracker.frames_seen = max(0, tracker.frames_seen - 1)
                if tracker.frames_seen == 0:
                    tracker.reset()

        # Screenshots sammeln
        now = time.monotonic()
        if collecting and (now - last_screenshot) >= SCREENSHOT_INTERVAL:
            fname = os.path.join(screenshot_dir, f'frame_{screenshot_count:05d}.jpg')
            cv2.imwrite(fname, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            screenshot_count += 1
            last_screenshot = now

        # FPS
        frame_count += 1
        if now - fps_time >= 1.0:
            fps = frame_count
            frame_count = 0
            fps_time = now

        # Anzeige
        if SHOW_WINDOW:
            display = draw_overlay(frame.copy(), detections, best_target, fps, ads_active, collecting, tracker)
            if WINDOW_SCALE != 1.0:
                display = cv2.resize(display, None, fx=WINDOW_SCALE, fy=WINDOW_SCALE)
            cv2.imshow('AIMBOT v2', display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('s'):
                collecting = not collecting
                status = "AN" if collecting else "AUS"
                print(f"Screenshot-Sammlung: {status} ({screenshot_count} gespeichert)")

    cap.release()
    kmbox_net.close()
    if SHOW_WINDOW:
        cv2.destroyAllWindows()
    print(f"\nBeendet. {screenshot_count} Screenshots in '{screenshot_dir}/'")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBeendet (Strg+C).")
        kmbox_net.close()
        cv2.destroyAllWindows()
