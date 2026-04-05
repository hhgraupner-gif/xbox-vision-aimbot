"""
AIMBOT VISION v10 — SOFT ASSIST
=================================
Leichter AI-Assist der MIT dem XIM Matrix zusammenarbeitet.
Gibt nur sanfte Schubser Richtung Gegner — der XIM glaettet den Rest.

NICHT als voller Aimbot gedacht! Nur als Unterstuetzung zum XIM Aim Assist.
Funktioniert perfekt mit: Weichheit 50, Sync 32, 10000 DPI, 15cm/360

Steuerung:
  A     = Assist an/aus (Toggle)
  1/2   = STRENGTH runter/hoch (0.1-Schritte, fein!)
  3/4   = FOV kleiner/groesser
  7/8   = Confidence runter/hoch
  M     = Modell wechseln
  ESC   = Beenden
"""

import os
import sys
import time
import math

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
# SOFT ASSIST — Sanfte Schubser, XIM glaettet
# ============================================================
# STRENGTH niedrig! Wir UNTERSTUETZEN den XIM Aim Assist nur.
# Der XIM mit Weichheit 50 + Sync 32 glaettet unsere Korrekturen.

STRENGTH = 0.7              # Etwas staerker (0.3 = kaum, 1.0 = deutlich)
DEADZONE = 12               # Kleiner = greift frueher ein
MAX_MOVE = 40               # Kleine Moves — XIM macht den Rest

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
# EINFACHER TRACKER (Kalman fuer Glaettung)
# ============================================================
class SmoothTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.x = None
        self.y = None
        self.alpha = 0.4  # EMA-Faktor: 0.3=sehr glatt, 0.6=reaktiv
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

        # Brust-Mitte
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
# OVERLAY
# ============================================================
def draw_overlay(frame, dets, aim_pos, tracker, fps, active, strength):
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2

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
    lock = " | LOCKED" if tracker.locked and active else ""
    cv2.putText(frame, f"FPS:{fps:.0f} | SOFT ASSIST | {mode}{lock}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 100) if active else (150, 150, 150), 2)
    cv2.putText(frame, f"Strength: {strength:.1f} | DZ: {DEADZONE} | FOV: {FOV_RADIUS}",
                (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    return frame


# ============================================================
# HAUPTPROGRAMM
# ============================================================
def main():
    global CONFIDENCE, FOV_RADIUS, STRENGTH, DEADZONE, MODEL_MODE

    print("=" * 55)
    print("  SOFT ASSIST v10 — XIM Unterstuetzung")
    print("  Sanfte AI-Schubser + XIM Aim Assist = Klebrig!")
    print("=" * 55)

    if KMBOX_AVAILABLE:
        try:
            kmbox_net.init(KMBOX_IP, KMBOX_PORT, KMBOX_UUID)
            print(f"  KMBox: OK")
        except Exception as e:
            print(f"  KMBox: {e}")
    else:
        print("  KMBox nicht da")

    model_path = get_model_path(MODEL_MODE)
    if not model_path:
        print("FEHLER: Kein Modell!")
        return
    print(f"  Modell: {os.path.basename(model_path)}")
    detector = YOLODetector(model_path)

    cap = cv2.VideoCapture(CAPTURE_DEVICE, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAPTURE_DEVICE)
    if not cap.isOpened():
        print("FEHLER: Keine Capture Card!")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  Capture: {fw}x{fh}")
    print(f"\n  A=Assist an/aus | 1/2=Strength | ESC=Quit")
    print("=" * 55)

    tracker = SmoothTracker()
    active = True  # Standardmaessig AN
    fc = 0
    fps = 0.0
    fps_t = time.monotonic()
    strength = STRENGTH

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            fc += 1
            now = time.monotonic()
            if fc % 30 == 0:
                el = now - fps_t
                fps = 30.0 / el if el > 0 else 0
                fps_t = now

            all_dets = detector.detect(frame, conf_threshold=CONFIDENCE)
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

                        # Nur korrigieren wenn deutlich daneben
                        if dist > DEADZONE:
                            mx = dx * strength
                            my = dy * strength

                            mx = max(-MAX_MOVE, min(MAX_MOVE, mx))
                            my = max(-MAX_MOVE, min(MAX_MOVE, my))

                            ix = int(round(mx))
                            iy = int(round(my))

                            if (abs(ix) > 0 or abs(iy) > 0) and KMBOX_AVAILABLE:
                                try:
                                    kmbox_net.move(ix, iy)
                                except Exception:
                                    pass
            else:
                tracker.mark_lost()

            if SHOW_WINDOW:
                disp = draw_overlay(frame.copy(), all_dets, aim_pos, tracker, fps, active, strength)
                if WINDOW_SCALE != 1.0:
                    disp = cv2.resize(disp, (int(fw * WINDOW_SCALE), int(fh * WINDOW_SCALE)))
                cv2.imshow("SOFT ASSIST v10", disp)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            elif key == ord('a'):
                active = not active
                print(f"Assist: {'EIN' if active else 'AUS'}")
                if not active:
                    tracker.reset()
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
