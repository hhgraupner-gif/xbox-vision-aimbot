"""
AIMBOT VISION v9 — FULL BODY LOCK
===================================
Radikal vereinfacht nach Analyse der Profi-Aimbots.

KERN-LOGIK (wie die TikTok-Aimbots):
  move_x = delta_x * STRENGTH
  move_y = delta_y * STRENGTH
  → kmbox_net.move(mx, my) JEDEN FRAME

Ein einziger Parameter: STRENGTH
  - Zu schwach? STRENGTH hoch (Taste 2)
  - Overshoot? STRENGTH runter (Taste 1)
  - Perfekt wenn Fadenkreuz SOFORT auf Gegner springt und bleibt

Steuerung:
  A     = ADS an/aus
  1/2   = STRENGTH runter/hoch (WICHTIGSTER WERT!)
  3/4   = FOV kleiner/groesser
  5/6   = Anti-Recoil schwaecher/staerker
  R     = Anti-Recoil an/aus
  P     = Profil (Lock / Assist)
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
CONFIDENCE = 0.30
FOV_RADIUS = 200

# ============================================================
# AIM — EIN PARAMETER: STRENGTH
# ============================================================
# So machen es die Profi-Aimbots:
#   move = delta * STRENGTH
# STRENGTH = Pixel-zu-Maus Umrechnungsfaktor
# Wenn STRENGTH richtig eingestellt ist:
#   → Gegner 50px entfernt → Fadenkreuz springt GENAU 50px
#   → PERFEKTER LOCK

STRENGTH = 4.0              # START-WERT. User muss tunen!
                             # Zu schwach → hoeher (Taste 2)
                             # Overshoot → niedriger (Taste 1)
                             # Profi-Bereich: 2.0 - 8.0

# Sicherheits-Limit
MAX_MOVE = 150              # Max Pixel pro Frame

# Anti-Recoil: Zieht nach unten waehrend Lock (gegen Rueckstoss)
ANTI_RECOIL = 8.0           # Pixel nach unten pro Frame
ANTI_RECOIL_ENABLED = True

# Kalman Prediction — DEFAULT AUS (verstaerkt Jitter!)
PREDICTION_FRAMES = 0

# Deadzone: Wenn Ziel naeher als X Pixel → NICHT bewegen (verhindert Mikro-Zittern)
AIM_DEADZONE = 5

# ============================================================
# PROFILE
# ============================================================
PROFILES = {
    "lock": {
        "name": "FULL LOCK",
        "strength": 4.0,
    },
    "assist": {
        "name": "SOFT ASSIST",
        "strength": 2.0,
    },
}
PROFILE_ORDER = ["lock", "assist"]

# ============================================================
# FILTER
# ============================================================
SHOW_WINDOW = True
WINDOW_SCALE = 0.5
DEAD_BODY_RATIO = 1.2
SKY_FILTER_RATIO = 0.10
GROUND_FILTER_RATIO = 0.88
MIN_BOX_HEIGHT = 25


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
# KALMAN TRACKER — Schnell & Reaktiv
# ============================================================
class KalmanTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.x = np.zeros(4, dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64) * 500.0
        self.frames_seen = 0
        self.frames_lost = 0
        self.locked = False
        self.last_update = 0.0
        self.last_dt = 0.033
        self.H = np.array([[1,0,0,0],[0,1,0,0]], dtype=np.float64)
        # R HOCH = glaettet YOLO-Jitter STARK (das war zu niedrig!)
        self.R = np.eye(2, dtype=np.float64) * 30.0
        # Q Position niedrig, Velocity moderat
        self.Q_base = np.diag([1.0, 1.0, 6.0, 6.0])

    def predict(self, dt=None):
        if dt is None:
            dt = self.last_dt
        F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], dtype=np.float64)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q_base * dt

    def update(self, mx, my):
        now = time.monotonic()
        if self.last_update > 0:
            self.last_dt = max(0.005, min(0.200, now - self.last_update))
        z = np.array([mx, my], dtype=np.float64)

        if self.frames_seen == 0:
            self.x[:2] = [mx, my]
            self.x[2:] = [0.0, 0.0]
            self.P = np.eye(4, dtype=np.float64) * 30.0
        else:
            self.predict(self.last_dt)
            S = self.H @ self.P @ self.H.T + self.R
            K = self.P @ self.H.T @ np.linalg.inv(S)
            self.x = self.x + K @ (z - self.H @ self.x)
            self.P = (np.eye(4) - K @ self.H) @ self.P

        self.frames_seen += 1
        self.frames_lost = 0
        self.locked = True  # Sofort ab Frame 1!
        self.last_update = now

    def mark_lost(self):
        self.frames_lost += 1
        if self.frames_lost > 4:
            self.reset()
        self.locked = False

    def get_position(self):
        if self.frames_seen == 0:
            return None
        return (float(self.x[0]), float(self.x[1]))

    def get_velocity(self):
        return (float(self.x[2]), float(self.x[3]))

    def get_predicted_position(self, frames=2):
        if self.frames_seen < 2:
            return self.get_position()
        dt = self.last_dt * frames
        return (float(self.x[0] + self.x[2] * dt),
                float(self.x[1] + self.x[3] * dt))


# ============================================================
# ZIELAUSWAHL — Naechster Gegner zur Mitte
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

        # Zielpunkt: Brust (40% von oben)
        if cls == "head":
            tx, ty = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        else:
            tx, ty = (x1 + x2) / 2.0, y1 + bh * 0.35

        dist = math.sqrt((tx - cx) ** 2 + (ty - cy) ** 2)
        if dist > FOV_RADIUS:
            continue

        # Kopf bevorzugen
        effective_dist = dist * (0.5 if cls == "head" else 1.0)
        if effective_dist < best_dist:
            best_dist = effective_dist
            best = (tx, ty, det)

    return best if best else (None, None, None)


# ============================================================
# OVERLAY
# ============================================================
def draw_overlay(frame, dets, aim_pos, tracker, fps, ads, profile, strength):
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2

    # FOV
    col = (0, 255, 0) if ads else (80, 80, 80)
    cv2.circle(frame, (cx, cy), FOV_RADIUS, col, 1)

    # Fadenkreuz
    cv2.line(frame, (cx-12, cy), (cx+12, cy), (255, 255, 255), 1)
    cv2.line(frame, (cx, cy-12), (cx, cy+12), (255, 255, 255), 1)

    # Boxen
    for d in dets:
        x1, y1, x2, y2 = d["bbox"]
        c = d["class_name"]
        cf = d["confidence"]
        color = (0, 0, 255) if c in TARGET_CLASSES else (128, 128, 128) if c in IGNORE_CLASSES else (255, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{c} {cf:.0%}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Aim-Linie
    if aim_pos and ads:
        ax, ay = int(aim_pos[0]), int(aim_pos[1])
        cv2.line(frame, (cx, cy), (ax, ay), (0, 255, 255), 2)
        cv2.circle(frame, (ax, ay), 6, (0, 0, 255), -1)

    # Status
    mode = "LOCKED" if tracker.locked and ads else "ADS ON" if ads else "ADS OFF"
    cv2.putText(frame, f"FPS:{fps:.0f} | {profile['name']} | {mode}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"STRENGTH: {strength:.1f} | Recoil: {ANTI_RECOIL:.0f} | DZ: {AIM_DEADZONE}px | FOV: {FOV_RADIUS}",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return frame


# ============================================================
# HAUPTPROGRAMM
# ============================================================
def main():
    global CONFIDENCE, FOV_RADIUS, STRENGTH
    global MODEL_MODE, ANTI_RECOIL, ANTI_RECOIL_ENABLED, PREDICTION_FRAMES

    print("=" * 60)
    print("  AIMBOT v9 — FULL BODY LOCK")
    print("  move = delta * STRENGTH (wie die Profis)")
    print("=" * 60)

    # KMBox
    if KMBOX_AVAILABLE:
        try:
            kmbox_net.init(KMBOX_IP, KMBOX_PORT, KMBOX_UUID)
            print(f"  KMBox: {KMBOX_IP}:{KMBOX_PORT} OK")
        except Exception as e:
            print(f"  KMBox WARNUNG: {e}")
    else:
        print("  KMBox nicht da — Anzeige-Modus")

    # YOLO
    model_path = get_model_path(MODEL_MODE)
    if not model_path:
        print("FEHLER: Kein Modell!")
        return
    print(f"  Modell: {os.path.basename(model_path)}")
    detector = YOLODetector(model_path)

    # Capture
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

    print("\n  BEREIT! Taste A = ADS | 1/2 = STRENGTH | ESC = Quit")
    print("=" * 60)

    tracker = KalmanTracker()
    prof_idx = 0
    profile = PROFILES[PROFILE_ORDER[prof_idx]]
    strength = profile["strength"]
    ads = False
    fc = 0
    fps = 0.0
    fps_t = time.monotonic()

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

            # Detect
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

                    # ==========================================
                    # FULL BODY LOCK — Mit Jitter-Schutz
                    # ==========================================
                    if ads and tracker.locked:
                        # Prediction nur wenn eingeschaltet
                        if PREDICTION_FRAMES > 0 and tracker.frames_seen >= 3:
                            aim = tracker.get_predicted_position(PREDICTION_FRAMES)
                        else:
                            aim = pos
                        aim_pos = aim

                        dx = aim[0] - cx
                        dy = aim[1] - cy
                        dist = math.sqrt(dx * dx + dy * dy)

                        # DEADZONE: Wenn wir schon drauf sind → NICHT bewegen!
                        if dist > AIM_DEADZONE:
                            mx = dx * strength
                            my = dy * strength

                            # Anti-Recoil
                            if ANTI_RECOIL_ENABLED:
                                my += ANTI_RECOIL

                            # Limit
                            mx = max(-MAX_MOVE, min(MAX_MOVE, mx))
                            my = max(-MAX_MOVE, min(MAX_MOVE, my))

                            ix = int(round(mx))
                            iy = int(round(my))
                            if (abs(ix) > 0 or abs(iy) > 0) and KMBOX_AVAILABLE:
                                try:
                                    kmbox_net.move(ix, iy)
                                except Exception:
                                    pass
                        elif ANTI_RECOIL_ENABLED:
                            # In Deadzone: Nur Anti-Recoil
                            iy = int(round(ANTI_RECOIL))
                            if iy > 0 and KMBOX_AVAILABLE:
                                try:
                                    kmbox_net.move(0, iy)
                                except Exception:
                                    pass
            else:
                tracker.mark_lost()

            # Anzeige
            if SHOW_WINDOW:
                disp = draw_overlay(frame.copy(), all_dets, aim_pos, tracker, fps, ads, profile, strength)
                if WINDOW_SCALE != 1.0:
                    disp = cv2.resize(disp, (int(fw * WINDOW_SCALE), int(fh * WINDOW_SCALE)))
                cv2.imshow("AIMBOT v9", disp)

            # Tasten
            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                break
            elif key == ord('a'):
                ads = not ads
                print(f"ADS: {'EIN' if ads else 'AUS'}")
                if not ads:
                    tracker.reset()
            elif key == ord('1'):
                strength = max(0.5, round(strength - 0.5, 1))
                print(f"STRENGTH: {strength}")
            elif key == ord('2'):
                strength = min(15.0, round(strength + 0.5, 1))
                print(f"STRENGTH: {strength}")
            elif key == ord('3'):
                FOV_RADIUS = max(50, FOV_RADIUS - 25)
                print(f"FOV: {FOV_RADIUS}px")
            elif key == ord('4'):
                FOV_RADIUS = min(500, FOV_RADIUS + 25)
                print(f"FOV: {FOV_RADIUS}px")
            elif key == ord('5'):
                ANTI_RECOIL = max(0.0, round(ANTI_RECOIL - 1.0, 1))
                print(f"Anti-Recoil: {ANTI_RECOIL}px")
            elif key == ord('6'):
                ANTI_RECOIL = min(25.0, round(ANTI_RECOIL + 1.0, 1))
                print(f"Anti-Recoil: {ANTI_RECOIL}px")
            elif key == ord('r'):
                ANTI_RECOIL_ENABLED = not ANTI_RECOIL_ENABLED
                print(f"Anti-Recoil: {'EIN' if ANTI_RECOIL_ENABLED else 'AUS'}")
            elif key == ord('p'):
                prof_idx = (prof_idx + 1) % len(PROFILE_ORDER)
                profile = PROFILES[PROFILE_ORDER[prof_idx]]
                strength = profile["strength"]
                print(f"Profil: {profile['name']} (Strength: {strength})")
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
            elif key == ord('7'):
                CONFIDENCE = max(0.15, round(CONFIDENCE - 0.05, 2))
                print(f"Confidence: {CONFIDENCE}")
            elif key == ord('8'):
                CONFIDENCE = min(0.80, round(CONFIDENCE + 0.05, 2))
                print(f"Confidence: {CONFIDENCE}")
            elif key == ord('9'):
                PREDICTION_FRAMES = max(0, PREDICTION_FRAMES - 1)
                print(f"Prediction: {PREDICTION_FRAMES}F")
            elif key == ord('0'):
                PREDICTION_FRAMES = min(6, PREDICTION_FRAMES + 1)
                print(f"Prediction: {PREDICTION_FRAMES}F")

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
