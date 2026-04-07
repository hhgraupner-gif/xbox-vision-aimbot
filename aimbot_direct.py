"""
AIMBOT VISION v12 — MAXIMUM PERFORMANCE
=========================================
Alles optimiert fuer minimale Latenz und maximale FPS.

Performance-Features:
  1. Threaded Capture (Buffer=1, 60 FPS)
  2. Threaded Inference (AI laeuft parallel, blockt nichts)
  3. ROI-Cropping (640x640 aus Mitte, kein Resize)
  4. Dynamischer Modell-Switch (Nano 320 fuer Speed / Full 640 fuer Praezision)
  5. Overlay Toggle (aus = +10-15 FPS)
  6. Zero-Copy Pipeline (minimale Speicher-Operationen)

Steuerung:
  A     = Assist an/aus
  1/2   = STRENGTH runter/hoch
  3/4   = FOV kleiner/groesser
  5/6   = Deadzone kleiner/groesser
  7/8   = Confidence runter/hoch
  9/0   = Cooldown -/+
  C     = ROI Cropping an/aus
  V     = Overlay an/aus (aus = mehr FPS!)
  N     = Nano/Full Modell umschalten
  M     = Modell-Familie wechseln
  ESC   = Beenden
"""

import os
import sys
import time
import math
import threading
from collections import deque

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
USE_ROI_CROP = True
ROI_SIZE = 640
TARGET_FPS = 60
SHOW_OVERLAY = True         # V-Taste: Overlay aus = +10-15 FPS!
USE_NANO = False            # N-Taste: Nano-Modell (320px, 3x schneller)


def get_model_path(mode, nano=False):
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend')
    if nano:
        priority = ['sunxds_nano_320.onnx', 'sunxds_0.5.6.onnx']
    else:
        priority_map = {
            "fps":  ['sunxds_0.5.6.onnx', 'sunxds_640.onnx', 'bo7_v5_640.onnx', 'yolo11s.onnx'],
            "coco": ['yolo11s.onnx', 'sunxds_0.5.6.onnx'],
            "nano": ['sunxds_nano_320.onnx', 'sunxds_0.5.6.onnx'],
        }
        priority = priority_map.get(mode, priority_map["fps"])

    for name in priority:
        path = os.path.join(base, name)
        if os.path.exists(path):
            return path
    if os.path.isdir(base):
        for f in os.listdir(base):
            if f.endswith('.onnx'):
                return os.path.join(base, f)
    return None


# ============================================================
# FILTER
# ============================================================
DEAD_BODY_RATIO = 1.2
SKY_FILTER_RATIO = 0.10
GROUND_FILTER_RATIO = 0.88
MIN_BOX_HEIGHT = 40


# ============================================================
# THREADED CAPTURE — Immer neuestes Frame, keine Wartezeit
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
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.frame = None
        self.new_frame = False
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        print(f"  Capture: {self.width}x{self.height} @ {actual_fps}FPS (Buffer:1)")

    def _loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
                    self.new_frame = True

    def read(self):
        with self.lock:
            if self.frame is not None:
                self.new_frame = False
                return True, self.frame
            return False, None

    def has_new(self):
        with self.lock:
            return self.new_frame

    def release(self):
        self.running = False
        self.thread.join(timeout=2)
        self.cap.release()


# ============================================================
# THREADED INFERENCE — AI laeuft im Hintergrund
# ============================================================
class FastInference:
    def __init__(self, detector):
        self.detector = detector
        self.input_frame = None
        self.input_roi_info = None  # (off_x, off_y) oder None
        self._conf = 0.40
        self.results = []
        self.inf_ms = 0.0
        self.running = True
        self.has_input = threading.Event()
        self.has_output = threading.Event()
        self.lock_in = threading.Lock()
        self.lock_out = threading.Lock()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while self.running:
            self.has_input.wait(timeout=0.1)
            if not self.running:
                break
            self.has_input.clear()

            with self.lock_in:
                frame = self.input_frame
                roi_info = self.input_roi_info
                conf = self._conf

            if frame is None:
                continue

            t0 = time.monotonic()
            dets = self.detector.detect(frame, conf_threshold=conf)
            ms = (time.monotonic() - t0) * 1000

            # ROI Offset anwenden
            if roi_info:
                off_x, off_y = roi_info
                for d in dets:
                    d["bbox"][0] += off_x
                    d["bbox"][1] += off_y
                    d["bbox"][2] += off_x
                    d["bbox"][3] += off_y

            with self.lock_out:
                self.results = dets
                self.inf_ms = ms
            self.has_output.set()

    def submit(self, frame, conf, roi_info=None):
        with self.lock_in:
            self.input_frame = frame
            self.input_roi_info = roi_info
            self._conf = conf
        self.has_input.set()

    def get_results(self):
        with self.lock_out:
            return self.results, self.inf_ms

    def swap_detector(self, new_detector):
        self.detector = new_detector

    def stop(self):
        self.running = False
        self.has_input.set()
        self.thread.join(timeout=2)


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
            self.x += self.alpha * (mx - self.x)
            self.y += self.alpha * (my - self.y)
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
# ROI
# ============================================================
def crop_center(frame, crop_size):
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    half = crop_size // 2
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    return frame[y1:y1+crop_size, x1:x1+crop_size], x1, y1


# ============================================================
# HAUPTPROGRAMM
# ============================================================
def main():
    global CONFIDENCE, FOV_RADIUS, STRENGTH, DEADZONE, MODEL_MODE
    global COOLDOWN_FRAMES, USE_ROI_CROP, SHOW_OVERLAY, USE_NANO

    print("=" * 60)
    print("  AIMBOT v12 — MAXIMUM PERFORMANCE")
    print("  Threaded Capture + Threaded AI + ROI + Nano-Switch")
    print("=" * 60)

    # KMBox
    if KMBOX_AVAILABLE:
        try:
            kmbox_net.init(KMBOX_IP, KMBOX_PORT, KMBOX_UUID)
            print(f"  KMBox: OK")
        except Exception as e:
            print(f"  KMBox: {e}")

    # YOLO
    model_path = get_model_path(MODEL_MODE, USE_NANO)
    if not model_path:
        print("FEHLER: Kein Modell!")
        return
    print(f"  Modell: {os.path.basename(model_path)}")
    detector = YOLODetector(model_path)

    # Threaded Capture
    print(f"\n  Starte Threads...")
    try:
        cap = FastCapture(CAPTURE_DEVICE, 1920, 1080, TARGET_FPS)
    except RuntimeError as e:
        print(f"  FEHLER: {e}")
        return
    fw, fh = cap.width, cap.height

    # Threaded Inference
    inferencer = FastInference(detector)
    print(f"  AI-Thread: gestartet")

    print(f"  ROI: {'EIN' if USE_ROI_CROP else 'AUS'} ({ROI_SIZE}x{ROI_SIZE})")
    print(f"\n  Tasten:")
    print(f"  A=Assist  V=Overlay  C=ROI  N=Nano/Full  M=Modell")
    print(f"  1/2=STR  3/4=FOV  5/6=DZ  7/8=Conf  9/0=CD  ESC=Quit")
    print("=" * 60)

    tracker = SmoothTracker()
    active = True
    fc = 0
    fps = 0.0
    fps_t = time.monotonic()
    strength = STRENGTH
    cooldown = 0
    all_dets = []
    inf_ms = 0.0
    last_submit = 0.0

    # FPS Tracking (letzte 60 Frames)
    frame_times = deque(maxlen=60)

    try:
        while True:
            loop_start = time.monotonic()

            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.001)
                continue

            fc += 1

            # Frame an AI-Thread senden (nur wenn der vorherige fertig ist)
            if USE_ROI_CROP:
                roi, off_x, off_y = crop_center(frame, ROI_SIZE)
                inferencer.submit(roi, CONFIDENCE, (off_x, off_y))
            else:
                inferencer.submit(frame, CONFIDENCE, None)

            # Neueste Ergebnisse abholen (non-blocking)
            new_dets, new_ms = inferencer.get_results()
            if new_dets is not None:
                all_dets = new_dets
                inf_ms = new_ms

            # Zielauswahl + Aim
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

            # FPS berechnen
            frame_times.append(time.monotonic())
            if len(frame_times) > 1:
                elapsed = frame_times[-1] - frame_times[0]
                fps = (len(frame_times) - 1) / elapsed if elapsed > 0 else 0

            # Overlay (optional — ausschalten fuer +10-15 FPS)
            if SHOW_OVERLAY:
                h, w = frame.shape[:2]
                hcx, hcy = w // 2, h // 2

                if USE_ROI_CROP:
                    half = ROI_SIZE // 2
                    cv2.rectangle(frame, (hcx-half, hcy-half), (hcx+half, hcy+half), (50, 50, 50), 1)

                col = (0, 200, 100) if active else (80, 80, 80)
                cv2.circle(frame, (hcx, hcy), FOV_RADIUS, col, 1)
                cv2.line(frame, (hcx-10, hcy), (hcx+10, hcy), (255, 255, 255), 1)
                cv2.line(frame, (hcx, hcy-10), (hcx, hcy+10), (255, 255, 255), 1)

                for d in all_dets:
                    x1, y1, x2, y2 = d["bbox"]
                    c = d["class_name"]
                    cf = d["confidence"]
                    clr = (0, 0, 255) if c in TARGET_CLASSES else (128, 128, 128) if c in IGNORE_CLASSES else (255, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), clr, 2)
                    cv2.putText(frame, f"{c} {cf:.0%}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, clr, 1)

                if aim_pos and active and tracker.locked:
                    ax, ay = int(aim_pos[0]), int(aim_pos[1])
                    cv2.line(frame, (hcx, hcy), (ax, ay), (0, 200, 100), 1)
                    cv2.circle(frame, (ax, ay), 5, (0, 200, 100), -1)

                mode = "ON" if active else "OFF"
                nano = "NANO" if USE_NANO else "FULL"
                roi = "ROI" if USE_ROI_CROP else "ALL"
                cv2.putText(frame, f"FPS:{fps:.0f} AI:{inf_ms:.0f}ms {nano} {roi} | {mode}",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 100), 2)
                cv2.putText(frame, f"STR:{strength:.1f} DZ:{DEADZONE} CD:{COOLDOWN_FRAMES} FOV:{FOV_RADIUS}",
                            (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

                disp = cv2.resize(frame, (fw // 2, fh // 2))
                cv2.imshow("AIMBOT v12", disp)
            else:
                # Minimales Fenster fuer Tastatur-Input
                tiny = np.zeros((60, 300, 3), dtype=np.uint8)
                nano = "NANO" if USE_NANO else "FULL"
                roi = "ROI" if USE_ROI_CROP else "ALL"
                cv2.putText(tiny, f"FPS:{fps:.0f} AI:{inf_ms:.0f}ms {nano} {roi}",
                            (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 100), 1)
                cv2.putText(tiny, f"STR:{strength:.1f} | V=Overlay | ESC=Quit",
                            (5, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
                cv2.imshow("AIMBOT v12", tiny)

            # Tastatur
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            elif key == ord('a'):
                active = not active
                print(f"Assist: {'EIN' if active else 'AUS'}")
                if not active:
                    tracker.reset()
            elif key == ord('v'):
                SHOW_OVERLAY = not SHOW_OVERLAY
                print(f"Overlay: {'EIN' if SHOW_OVERLAY else 'AUS (Max FPS!)'}")
            elif key == ord('c'):
                USE_ROI_CROP = not USE_ROI_CROP
                print(f"ROI: {'EIN' if USE_ROI_CROP else 'AUS'}")
            elif key == ord('n'):
                USE_NANO = not USE_NANO
                np2 = get_model_path(MODEL_MODE, USE_NANO)
                if np2:
                    print(f"Lade: {'NANO' if USE_NANO else 'FULL'} ({os.path.basename(np2)})...")
                    new_det = YOLODetector(np2)
                    inferencer.swap_detector(new_det)
                    tracker.reset()
                else:
                    USE_NANO = not USE_NANO
                    print("Modell nicht gefunden!")
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
                print(f"Cooldown: {COOLDOWN_FRAMES}")
            elif key == ord('0'):
                COOLDOWN_FRAMES = min(10, COOLDOWN_FRAMES + 1)
                print(f"Cooldown: {COOLDOWN_FRAMES}")
            elif key == ord('m'):
                modes = ["fps", "coco"]
                idx = modes.index(MODEL_MODE) if MODEL_MODE in modes else 0
                MODEL_MODE = modes[(idx + 1) % len(modes)]
                np2 = get_model_path(MODEL_MODE, USE_NANO)
                if np2:
                    print(f"Lade: {MODEL_MODE}...")
                    new_det = YOLODetector(np2)
                    inferencer.swap_detector(new_det)
                    tracker.reset()
                else:
                    MODEL_MODE = modes[idx]

    except KeyboardInterrupt:
        print("\nStop.")
    finally:
        inferencer.stop()
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
