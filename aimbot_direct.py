"""
AIMBOT VISION v11 — PERFORMANCE OPTIMIERT
==========================================
Soft Assist mit 3 Performance-Boostern:
  1. ROI-Cropping: Nur Bildmitte analysieren (640x640 statt 1920x1080)
  2. Threading: Capture + AI laufen parallel
  3. 60 FPS Capture

Steuerung:
  A     = Assist an/aus
  1/2   = STRENGTH runter/hoch
  3/4   = FOV kleiner/groesser
  5/6   = Deadzone kleiner/groesser
  7/8   = Confidence runter/hoch
  9/0   = Cooldown -/+
  C     = ROI Cropping an/aus
  M     = Modell wechseln
  ESC   = Beenden
"""

import os
import sys
import time
import math
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

import cv2
import numpy as np
from yolo_onnx import YOLODetector, TARGET_CLASSES, IGNORE_CLASSES

try:
    import kmbox_net
    KMBOX_AVAILABLE = True
except Exception:
    KMBOX_AVAILABLE = False


# ============================================================
# HARDWARE
# ============================================================
KMBOX_IP = "192.168.2.188"
KMBOX_PORT = 32778
KMBOX_UUID = "C14AE466"
CAPTURE_DEVICE = 0

# ============================================================
# MODELL
# ============================================================
MODEL_MODE = "fps"
CONFIDENCE = 0.40
FOV_RADIUS = 180

# ============================================================
# SOFT ASSIST
# ============================================================
STRENGTH = 0.7
DEADZONE = 15
MAX_MOVE = 30
COOLDOWN_FRAMES = 4

# ============================================================
# PERFORMANCE
# ============================================================
USE_ROI_CROP = True         # Nur Bildmitte analysieren
ROI_SIZE = 640              # 640x640 Pixel aus der Mitte (= Modell-Groesse!)
TARGET_FPS = 60             # Capture Card FPS

# ============================================================
# FILTER
# ============================================================
SHOW_WINDOW = True
WINDOW_SCALE = 0.5
DEAD_BODY_RATIO = 1.2
SKY_FILTER_RATIO = 0.10
GROUND_FILTER_RATIO = 0.88
MIN_BOX_HEIGHT = 40


def get_model_path(mode):
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend')
    priority = {
        "fps":  ['sunxds_0.5.6.onnx', 'sunxds_640.onnx', 'bo7_v5_640.onnx', 'yolo11s.onnx'],
        "coco": ['yolo11s.onnx', 'sunxds_0.5.6.onnx', 'sunxds_640.onnx'],
        "nano": ['sunxds_nano_320.onnx', 'sunxds_0.5.6.onnx', 'sunxds_640.onnx'],
    }
    candidates = priority.get(mode, priority["coco"])
    for name in candidates:
        path = os.path.join(base, name)
        if os.path.exists(path):
            return path
    if os.path.isdir(base):
        for f in os.listdir(base):
            if f.endswith('.onnx'):
                return os.path.join(base, f)
    return None


# ============================================================
# THREADED CAPTURE — Laeuft im Hintergrund, immer neuestes Frame
# ============================================================
class FastCapture:
    def __init__(self, device, width=1920, height=1080, fps=60):
        self.cap = cv2.VideoCapture(device, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(device)
        if not self.cap.isOpened():
            raise RuntimeError("Capture Card nicht gefunden!")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        # Puffer klein halten fuer minimale Latenz
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.frame = None
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        print(f"  Capture: {self.width}x{self.height} @ {actual_fps} FPS (Buffer: 1)")

    def _capture_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame

    def read(self):
        with self.lock:
            return self.frame is not None, self.frame.copy() if self.frame is not None else None

    def release(self):
        self.running = False
        self.thread.join(timeout=2)
        self.cap.release()


# ============================================================
# TRACKER
# ============================================================
class SmoothTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.x = None
        self.y = None
        self.alpha = 0.4
        self.frames = 0
        self.lost = 0

    def update(self, mx, my):
        if self.x is None:
            self.x, self.y = mx, my
        else:
            self.x = self.x + self.alpha * (mx - self.x)
            self.y = self.y + self.alpha * (my - self.y)
        self.frames += 1
        self.lost = 0

    def mark_lost(self):
        self.lost += 1
        if self.lost > 5:
            self.reset()

    def get_position(self):
        if self.x is None:
            return None
        return (self.x, self.y)

    @property
    def locked(self):
        return self.frames >= 1 and self.lost == 0


# ============================================================
# ZIELAUSWAHL
# ============================================================
def pick_best_target(detections, frame_w, frame_h):
    cx, cy = frame_w / 2.0, frame_h / 2.0
    best = None
    best_dist = float('inf')

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        bw, bh = x2 - x1, y2 - y1
        cls = det["class_name"]

        if bw > bh * DEAD_BODY_RATIO:
            continue
        if bh < MIN_BOX_HEIGHT:
            continue
        mid_y = (y1 + y2) / 2.0
        if mid_y < frame_h * SKY_FILTER_RATIO:
            continue
        if mid_y > frame_h * GROUND_FILTER_RATIO:
            continue
        if cls in IGNORE_CLASSES:
            continue

        if cls == "head":
            tx, ty = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        else:
            tx, ty = (x1 + x2) / 2.0, y1 + bh * 0.35

        dist = math.sqrt((tx - cx) ** 2 + (ty - cy) ** 2)
        if dist > FOV_RADIUS:
            continue

        effective_dist = dist * (0.5 if cls == "head" else 1.0)
        if effective_dist < best_dist:
            best_dist = effective_dist
            best = (tx, ty, det)

    return best if best else (None, None, None)


# ============================================================
# ROI CROPPING — Nur Bildmitte fuer die AI
# ============================================================
def crop_center(frame, crop_size):
    """Schneidet crop_size x crop_size aus der Mitte.
    Gibt (crop, offset_x, offset_y) zurueck."""
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    half = crop_size // 2
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(w, cx + half)
    y2 = min(h, cy + half)
    return frame[y1:y2, x1:x2], x1, y1


def offset_detections(detections, off_x, off_y):
    """Verschiebt Bounding Boxes vom Crop-Raum in den Fullframe-Raum."""
    for d in detections:
        d["bbox"][0] += off_x
        d["bbox"][1] += off_y
        d["bbox"][2] += off_x
        d["bbox"][3] += off_y
    return detections


# ============================================================
# OVERLAY
# ============================================================
def draw_overlay(frame, dets, aim_pos, tracker, fps, inf_ms, active, strength, use_roi):
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2

    # ROI-Bereich anzeigen
    if use_roi:
        half = ROI_SIZE // 2
        cv2.rectangle(frame, (cx-half, cy-half), (cx+half, cy+half), (50, 50, 50), 1)

    col = (0, 200, 100) if active else (80, 80, 80)
    cv2.circle(frame, (cx, cy), FOV_RADIUS, col, 1)
    cv2.line(frame, (cx-10, cy), (cx+10, cy), (255, 255, 255), 1)
    cv2.line(frame, (cx, cy-10), (cx, cy+10), (255, 255, 255), 1)

    for d in dets:
        x1, y1, x2, y2 = d["bbox"]
        c = d["class_name"]
        cf = d["confidence"]
        color = (0, 0, 255) if c in TARGET_CLASSES else (128, 128, 128) if c in IGNORE_CLASSES else (255, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{c} {cf:.0%}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    if aim_pos and active and tracker.locked:
        ax, ay = int(aim_pos[0]), int(aim_pos[1])
        cv2.line(frame, (cx, cy), (ax, ay), (0, 200, 100), 1)
        cv2.circle(frame, (ax, ay), 5, (0, 200, 100), -1)

    mode = "ASSIST ON" if active else "ASSIST OFF"
    roi_txt = "ROI" if use_roi else "FULL"
    lock = " | LOCKED" if tracker.locked and active else ""
    cv2.putText(frame, f"FPS:{fps:.0f} | AI:{inf_ms:.0f}ms | {roi_txt} | {mode}{lock}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 100) if active else (150, 150, 150), 2)
    cv2.putText(frame, f"STR:{strength:.1f} | DZ:{DEADZONE} | CD:{COOLDOWN_FRAMES} | FOV:{FOV_RADIUS}",
                (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    return frame


# ============================================================
# HAUPTPROGRAMM
# ============================================================
def main():
    global CONFIDENCE, FOV_RADIUS, STRENGTH, DEADZONE, MODEL_MODE
    global COOLDOWN_FRAMES, USE_ROI_CROP

    print("=" * 60)
    print("  SOFT ASSIST v11 — PERFORMANCE OPTIMIERT")
    print("  ROI-Crop + Threading + 60 FPS")
    print("=" * 60)

    # KMBox
    if KMBOX_AVAILABLE:
        try:
            kmbox_net.init(KMBOX_IP, KMBOX_PORT, KMBOX_UUID)
            print(f"  KMBox: OK")
        except Exception as e:
            print(f"  KMBox: {e}")
    else:
        print("  KMBox nicht da")

    # YOLO
    model_path = get_model_path(MODEL_MODE)
    if not model_path:
        print("FEHLER: Kein Modell!")
        return
    print(f"  Modell: {os.path.basename(model_path)}")
    detector = YOLODetector(model_path)

    # Capture (Threaded!)
    print(f"\n  Starte Threaded Capture...")
    try:
        cap = FastCapture(CAPTURE_DEVICE, 1920, 1080, TARGET_FPS)
    except RuntimeError as e:
        print(f"  FEHLER: {e}")
        return
    fw, fh = cap.width, cap.height

    print(f"\n  ROI-Crop: {'EIN' if USE_ROI_CROP else 'AUS'} ({ROI_SIZE}x{ROI_SIZE})")
    print(f"  → Analysiert nur Bildmitte statt volles {fw}x{fh}")
    print(f"\n  Tasten: A=Assist 1/2=Strength C=ROI ESC=Quit")
    print("=" * 60)

    tracker = SmoothTracker()
    active = True
    fc = 0
    fps = 0.0
    fps_t = time.monotonic()
    strength = STRENGTH
    cooldown = 0
    inf_ms = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.001)
                continue

            fc += 1
            now = time.monotonic()
            if fc % 30 == 0:
                el = now - fps_t
                fps = 30.0 / el if el > 0 else 0
                fps_t = now

            # === ROI CROPPING ===
            if USE_ROI_CROP:
                roi, off_x, off_y = crop_center(frame, ROI_SIZE)
                t0 = time.monotonic()
                all_dets = detector.detect(roi, conf_threshold=CONFIDENCE)
                inf_ms = (time.monotonic() - t0) * 1000
                # Koordinaten zurueck in Fullframe-Raum
                all_dets = offset_detections(all_dets, off_x, off_y)
            else:
                t0 = time.monotonic()
                all_dets = detector.detect(frame, conf_threshold=CONFIDENCE)
                inf_ms = (time.monotonic() - t0) * 1000

            targets = [d for d in all_dets if d["class_name"] in TARGET_CLASSES]
            tx, ty, tdet = pick_best_target(targets, fw, fh)

            aim_pos = None
            cx, cy = fw / 2.0, fh / 2.0

            if tx is not None:
                tracker.update(tx, ty)
                pos = tracker.get_position()
                if pos:
                    aim_pos = pos

                    if active and tracker.locked:
                        dx = pos[0] - cx
                        dy = pos[1] - cy
                        dist = math.sqrt(dx * dx + dy * dy)

                        if dist > DEADZONE and cooldown <= 0:
                            mx = dx * strength
                            my = dy * strength
                            mx = max(-MAX_MOVE, min(MAX_MOVE, mx))
                            my = max(-MAX_MOVE, min(MAX_MOVE, my))
                            ix = int(round(mx))
                            iy = int(round(my))
                            if (abs(ix) > 0 or abs(iy) > 0) and KMBOX_AVAILABLE:
                                try:
                                    kmbox_net.move(ix, iy)
                                    cooldown = COOLDOWN_FRAMES
                                except Exception:
                                    pass
            else:
                tracker.mark_lost()

            if cooldown > 0:
                cooldown -= 1

            if SHOW_WINDOW:
                disp = draw_overlay(frame.copy(), all_dets, aim_pos, tracker,
                                    fps, inf_ms, active, strength, USE_ROI_CROP)
                if WINDOW_SCALE != 1.0:
                    disp = cv2.resize(disp, (int(fw * WINDOW_SCALE), int(fh * WINDOW_SCALE)))
                cv2.imshow("SOFT ASSIST v11", disp)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            elif key == ord('a'):
                active = not active
                print(f"Assist: {'EIN' if active else 'AUS'}")
                if not active:
                    tracker.reset()
            elif key == ord('c'):
                USE_ROI_CROP = not USE_ROI_CROP
                print(f"ROI-Crop: {'EIN' if USE_ROI_CROP else 'AUS'}")
            elif key == ord('1'):
                strength = max(0.1, round(strength - 0.1, 1))
                print(f"Strength: {strength}")
            elif key == ord('2'):
                strength = min(5.0, round(strength + 0.1, 1))
                print(f"Strength: {strength}")
            elif key == ord('3'):
                FOV_RADIUS = max(50, FOV_RADIUS - 25)
                print(f"FOV: {FOV_RADIUS}px")
            elif key == ord('4'):
                FOV_RADIUS = min(400, FOV_RADIUS + 25)
                print(f"FOV: {FOV_RADIUS}px")
            elif key == ord('5'):
                DEADZONE = max(5, DEADZONE - 5)
                print(f"Deadzone: {DEADZONE}px")
            elif key == ord('6'):
                DEADZONE = min(60, DEADZONE + 5)
                print(f"Deadzone: {DEADZONE}px")
            elif key == ord('7'):
                CONFIDENCE = max(0.15, round(CONFIDENCE - 0.05, 2))
                print(f"Confidence: {CONFIDENCE}")
            elif key == ord('8'):
                CONFIDENCE = min(0.80, round(CONFIDENCE + 0.05, 2))
                print(f"Confidence: {CONFIDENCE}")
            elif key == ord('9'):
                COOLDOWN_FRAMES = max(0, COOLDOWN_FRAMES - 1)
                print(f"Cooldown: {COOLDOWN_FRAMES} Frames")
            elif key == ord('0'):
                COOLDOWN_FRAMES = min(10, COOLDOWN_FRAMES + 1)
                print(f"Cooldown: {COOLDOWN_FRAMES} Frames")
            elif key == ord('m'):
                modes = ["fps", "coco", "nano"]
                idx = modes.index(MODEL_MODE) if MODEL_MODE in modes else 0
                MODEL_MODE = modes[(idx + 1) % len(modes)]
                np2 = get_model_path(MODEL_MODE)
                if np2:
                    print(f"Lade: {MODEL_MODE}...")
                    detector = YOLODetector(np2)
                    tracker.reset()
                else:
                    MODEL_MODE = modes[idx]

    except KeyboardInterrupt:
        print("\nStop.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if KMBOX_AVAILABLE:
            try:
                kmbox_net.close()
            except Exception:
                pass
        print("Beendet.")


if __name__ == "__main__":
    main()
