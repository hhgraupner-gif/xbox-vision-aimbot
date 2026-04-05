"""
AIMBOT VISION v8 — Aggressives Lock-On
========================================
Computer Vision Aimbot fuer Xbox RemotePlay via XIM Matrix.

AENDERUNGEN v8:
  - Aimbot-Profil als Standard (nicht mehr Assist!)
  - Smooth 2.0 default (war 7.0 — DESWEGEN war es so schwach!)
  - Minimum-Move Clamp (kleine Korrekturen werden hochgeboostet)
  - Snap-Zone: Nah am Ziel = sofort drauf (smooth=1)
  - Sofortiger Lock ab Frame 1 (nicht mehr 2 Frames warten)
  - Kalman Prediction fuer bewegende Gegner

Steuerung:
  A = ADS Toggle | P = Profil | M = Modell | R = Anti-Recoil
  1/2 = FOV | 3/4 = Conf | 5/6 = Smooth | 7/8 = SpeedX | 9/0 = SpeedY
  +/- = Anti-Recoil Staerke | [/] = Prediction Frames | ESC = Beenden
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
CONFIDENCE = 0.25
FOV_RADIUS = 250

# ============================================================
# AIM-KONFIGURATION v8
# ============================================================
SMOOTH_FACTOR = 2.0         # Default jetzt 2.0 (war 7.0 — das war das Problem!)
SPEED_X = 1.0
SPEED_Y = 0.80
DEADZONE = 3                # Kleiner = reagiert frueher
MAX_MOVE_PER_FRAME = 120    # Erhoet fuer aggressiveres Tracking

# ADS-BOOST: Kompensiert die niedrige In-Game ADS-Sensitivity!
# Dein Spiel: ADS-Multiplikator = 1.0 (Standard)
# Das Spiel reduziert Mausbewegungen im ADS um ca. 60-70%
# Wir gleichen das hier aus, OHNE die In-Game-Settings zu aendern
ADS_BOOST = 3.0             # Multipliziert alle Aim-Moves (2.0-4.0 empfohlen)

# Minimum-Move: XIM ignoriert winzige Bewegungen (1-2px)
MIN_MOVE = 4

# Snap-Zone: Wenn Ziel naeher als SNAP_RADIUS Pixel → Smooth wird 1.0 (sofort drauf!)
SNAP_RADIUS = 40            # Innerhalb 40px = Instant-Lock

# Anti-Recoil (nur wenn auf Gegner gelockt)
ANTI_RECOIL = 6.0
ANTI_RECOIL_ENABLED = True

# Kalman Prediction
PREDICTION_FRAMES = 4

# ============================================================
# PROFILE — Aimbot ist jetzt DEFAULT!
# ============================================================
PROFILES = {
    "aimbot": {
        "name": "AIMBOT",
        "smooth": 2.0,      # Aggressiv!
        "deadzone": 3,
    },
    "assist": {
        "name": "AIM-ASSIST",
        "smooth": 5.0,      # Sanfter
        "deadzone": 8,
    },
}
PROFILE_ORDER = ["aimbot", "assist"]

# ============================================================
# ANZEIGE / FILTER
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
# KALMAN-FILTER TRACKER
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
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
        self.R = np.eye(2, dtype=np.float64) * 8.0   # Weniger Rauschen = schnellere Reaktion
        self.Q_base = np.diag([2.0, 2.0, 12.0, 12.0])  # Mehr Prozess-Rauschen = folgt schneller

    def predict(self, dt=None):
        if dt is None:
            dt = self.last_dt
        F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], dtype=np.float64)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q_base * dt
        return self.x[:2].copy()

    def update(self, mx, my):
        now = time.monotonic()
        if self.last_update > 0:
            self.last_dt = max(0.005, min(0.200, now - self.last_update))
        z = np.array([mx, my], dtype=np.float64)

        if self.frames_seen == 0:
            self.x[:2] = [mx, my]
            self.x[2:] = [0.0, 0.0]
            self.P = np.eye(4, dtype=np.float64) * 50.0
        else:
            self.predict(self.last_dt)
            S = self.H @ self.P @ self.H.T + self.R
            K = self.P @ self.H.T @ np.linalg.inv(S)
            self.x = self.x + K @ (z - self.H @ self.x)
            self.P = (np.eye(4) - K @ self.H) @ self.P

        self.frames_seen += 1
        self.frames_lost = 0
        # SOFORT locken ab Frame 1! (war vorher Frame 2)
        self.locked = self.frames_seen >= 1
        self.last_update = now

    def mark_lost(self):
        self.frames_lost += 1
        if self.frames_lost > 5:
            self.reset()
        self.locked = False

    def get_position(self):
        if self.frames_seen == 0:
            return None
        return (float(self.x[0]), float(self.x[1]))

    def get_velocity(self):
        return (float(self.x[2]), float(self.x[3]))

    def get_predicted_position(self, lookahead_frames=2):
        if self.frames_seen < 3:
            return self.get_position()
        dt = self.last_dt * lookahead_frames
        px = self.x[0] + self.x[2] * dt
        py = self.x[1] + self.x[3] * dt
        return (float(px), float(py))


# ============================================================
# ZIELAUSWAHL
# ============================================================
def pick_best_target(detections, frame_w, frame_h):
    cx, cy = frame_w / 2.0, frame_h / 2.0
    best = None
    best_score = float('inf')

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        bw, bh = x2 - x1, y2 - y1
        class_name = det["class_name"]

        if bw > bh * DEAD_BODY_RATIO:
            continue
        if bh < MIN_BOX_HEIGHT:
            continue
        center_y = (y1 + y2) / 2.0
        if center_y < frame_h * SKY_FILTER_RATIO:
            continue
        if center_y > frame_h * GROUND_FILTER_RATIO:
            continue
        if class_name in IGNORE_CLASSES:
            continue

        # Zielpunkt: Brust-Mitte
        if class_name == "head":
            tx, ty = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        else:
            tx, ty = (x1 + x2) / 2.0, y1 + bh * 0.40

        dist = math.sqrt((tx - cx) ** 2 + (ty - cy) ** 2)
        if dist > FOV_RADIUS:
            continue

        score = dist * (0.6 if class_name == "head" else 1.0)
        if score < best_score:
            best_score = score
            best = (tx, ty, det)

    return best if best else (None, None, None)


# ============================================================
# OVERLAY
# ============================================================
def draw_overlay(frame, all_dets, target_pos, tracker, fps, ads_active, profile, fov_radius, smooth):
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2

    color = (0, 255, 0) if ads_active else (100, 100, 100)
    cv2.circle(frame, (cx, cy), fov_radius, color, 1)
    cv2.line(frame, (cx - 15, cy), (cx + 15, cy), (255, 255, 255), 1)
    cv2.line(frame, (cx, cy - 15), (cx, cy + 15), (255, 255, 255), 1)

    for det in all_dets:
        x1, y1, x2, y2 = det["bbox"]
        cls = det["class_name"]
        conf = det["confidence"]
        if cls in TARGET_CLASSES:
            c = (0, 0, 255)
        elif cls in IGNORE_CLASSES:
            c = (128, 128, 128)
        else:
            c = (255, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
        cv2.putText(frame, f"{cls} {conf:.0%}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)

    if target_pos and ads_active:
        tx, ty = int(target_pos[0]), int(target_pos[1])
        cv2.line(frame, (cx, cy), (tx, ty), (0, 255, 255), 2)
        cv2.circle(frame, (tx, ty), 8, (0, 255, 255), 2)

    status = f"FPS: {fps:.0f} | {profile['name']} | ADS: {'ON' if ads_active else 'OFF'}"
    if tracker.locked:
        vx, vy = tracker.get_velocity()
        status += f" | LOCKED v=({vx:.0f},{vy:.0f})"
    cv2.putText(frame, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    info = f"Smooth: {smooth:.1f} | FOV: {fov_radius} | ADS-Boost: {ADS_BOOST:.1f}x | Speed: {SPEED_X:.1f}/{SPEED_Y:.1f}"
    info2 = f"Recoil: {ANTI_RECOIL:.0f}px({'ON' if ANTI_RECOIL_ENABLED else 'OFF'}) | Pred: {PREDICTION_FRAMES}F | Snap: <{SNAP_RADIUS}px"
    cv2.putText(frame, info, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(frame, info2, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return frame


# ============================================================
# HAUPTPROGRAMM
# ============================================================
def main():
    global CONFIDENCE, FOV_RADIUS, SMOOTH_FACTOR
    global SPEED_X, SPEED_Y, MODEL_MODE
    global ANTI_RECOIL, ANTI_RECOIL_ENABLED, PREDICTION_FRAMES, ADS_BOOST

    print("=" * 60)
    print("  AIMBOT VISION v8 — Aggressives Lock-On")
    print("=" * 60)

    # --- 1. KMBox ---
    print("\n[1/3] KMBox verbinden...")
    if KMBOX_AVAILABLE:
        try:
            kmbox_net.init(KMBOX_IP, KMBOX_PORT, KMBOX_UUID)
            print(f"  OK: {KMBOX_IP}:{KMBOX_PORT}")
        except Exception as e:
            print(f"  WARNUNG: KMBox nicht erreichbar ({e})")
    else:
        print("  KMBox nicht verfuegbar — Anzeige-Modus")

    # --- 2. YOLO ---
    print(f"\n[2/3] Modell laden ({MODEL_MODE})...")
    model_path = get_model_path(MODEL_MODE)
    if not model_path:
        print("  FEHLER: Kein ONNX-Modell gefunden!")
        return
    print(f"  Datei: {os.path.basename(model_path)}")
    detector = YOLODetector(model_path)

    # --- 3. Capture ---
    print(f"\n[3/3] Capture Card (Device {CAPTURE_DEVICE})...")
    cap = cv2.VideoCapture(CAPTURE_DEVICE, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAPTURE_DEVICE)
    if not cap.isOpened():
        print("  FEHLER: Capture Card nicht gefunden!")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  Capture: {fw}x{fh}")

    print("\n" + "=" * 60)
    print("  BEREIT! Profil: AIMBOT (Smooth 2.0)")
    print("=" * 60)
    print("  Tasten: A=ADS P=Profil R=Recoil 5/6=Smooth ESC=Quit")
    print("=" * 60)

    # --- Variablen ---
    tracker = KalmanTracker()
    profile_idx = 0  # Startet jetzt mit AIMBOT (nicht Assist!)
    profile = PROFILES[PROFILE_ORDER[profile_idx]]
    ads_active = False
    frame_count = 0
    fps = 0.0
    fps_timer = time.monotonic()
    smooth = profile["smooth"]
    deadzone = profile["deadzone"]

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame_count += 1
            now = time.monotonic()

            if frame_count % 30 == 0:
                elapsed = now - fps_timer
                fps = 30.0 / elapsed if elapsed > 0 else 0
                fps_timer = now

            # --- YOLO ---
            all_dets = detector.detect(frame, conf_threshold=CONFIDENCE)
            target_dets = [d for d in all_dets if d["class_name"] in TARGET_CLASSES]
            tx, ty, target_det = pick_best_target(target_dets, fw, fh)

            target_pos = None
            cx, cy = fw / 2.0, fh / 2.0

            if tx is not None:
                tracker.update(tx, ty)
                pos = tracker.get_position()
                if pos:
                    target_pos = pos

                    # ========================================
                    # v8 AIM-LOGIK: Snap + Prediction + Clamp
                    # ========================================
                    if ads_active and tracker.locked:
                        # Prediction
                        if PREDICTION_FRAMES > 0 and tracker.frames_seen >= 3:
                            pred = tracker.get_predicted_position(PREDICTION_FRAMES)
                            aim_x, aim_y = pred[0], pred[1]
                        else:
                            aim_x, aim_y = pos[0], pos[1]

                        dx = aim_x - cx
                        dy = aim_y - cy
                        dist = math.sqrt(dx * dx + dy * dy)

                        if dist > deadzone:
                            # SNAP-ZONE: Nah am Ziel = Smooth 1.0 (sofort!)
                            if dist < SNAP_RADIUS:
                                effective_smooth = 1.0
                            else:
                                effective_smooth = smooth

                            move_x = (dx / effective_smooth) * SPEED_X * ADS_BOOST
                            move_y = (dy / effective_smooth) * SPEED_Y * ADS_BOOST

                            # Anti-Recoil
                            if ANTI_RECOIL_ENABLED and ANTI_RECOIL > 0:
                                move_y += ANTI_RECOIL

                            # MINIMUM-CLAMP: Kleine Moves hochsetzen!
                            # XIM ignoriert 1-2px — mindestens MIN_MOVE senden
                            if abs(move_x) > 0.5 and abs(move_x) < MIN_MOVE:
                                move_x = MIN_MOVE if move_x > 0 else -MIN_MOVE
                            if abs(move_y) > 0.5 and abs(move_y) < MIN_MOVE:
                                move_y = MIN_MOVE if move_y > 0 else -MIN_MOVE

                            # Sicherheits-Limit
                            move_x = max(-MAX_MOVE_PER_FRAME, min(MAX_MOVE_PER_FRAME, move_x))
                            move_y = max(-MAX_MOVE_PER_FRAME, min(MAX_MOVE_PER_FRAME, move_y))

                            ix = int(round(move_x))
                            iy = int(round(move_y))
                            if (abs(ix) > 0 or abs(iy) > 0) and KMBOX_AVAILABLE:
                                try:
                                    kmbox_net.move(ix, iy)
                                except Exception:
                                    pass

                        elif ANTI_RECOIL_ENABLED and ANTI_RECOIL > 0:
                            iy = int(round(ANTI_RECOIL))
                            if iy > 0 and KMBOX_AVAILABLE:
                                try:
                                    kmbox_net.move(0, iy)
                                except Exception:
                                    pass
            else:
                tracker.mark_lost()

            # --- Anzeige ---
            if SHOW_WINDOW:
                display = draw_overlay(
                    frame.copy(), all_dets, target_pos,
                    tracker, fps, ads_active, profile, FOV_RADIUS, smooth
                )
                if WINDOW_SCALE != 1.0:
                    display = cv2.resize(display, (int(fw * WINDOW_SCALE), int(fh * WINDOW_SCALE)))
                cv2.imshow("AIMBOT v8", display)

            # --- Tastatur ---
            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                break
            elif key == ord('a'):
                ads_active = not ads_active
                print(f"ADS: {'EIN' if ads_active else 'AUS'}")
                if not ads_active:
                    tracker.reset()
            elif key == ord('p'):
                profile_idx = (profile_idx + 1) % len(PROFILE_ORDER)
                profile = PROFILES[PROFILE_ORDER[profile_idx]]
                smooth = profile["smooth"]
                deadzone = profile["deadzone"]
                print(f"Profil: {profile['name']} (Smooth: {smooth}, DZ: {deadzone})")
            elif key == ord('m'):
                modes = ["coco", "fps", "nano"]
                idx = modes.index(MODEL_MODE) if MODEL_MODE in modes else 0
                MODEL_MODE = modes[(idx + 1) % len(modes)]
                new_path = get_model_path(MODEL_MODE)
                if new_path:
                    print(f"Lade: {MODEL_MODE} ({os.path.basename(new_path)})...")
                    detector = YOLODetector(new_path)
                    tracker.reset()
                else:
                    print(f"Modell '{MODEL_MODE}' nicht gefunden!")
                    MODEL_MODE = modes[idx]
            elif key == ord('1'):
                FOV_RADIUS = max(50, FOV_RADIUS - 25)
                print(f"FOV: {FOV_RADIUS}px")
            elif key == ord('2'):
                FOV_RADIUS = min(500, FOV_RADIUS + 25)
                print(f"FOV: {FOV_RADIUS}px")
            elif key == ord('3'):
                CONFIDENCE = max(0.15, round(CONFIDENCE - 0.05, 2))
                print(f"Confidence: {CONFIDENCE:.2f}")
            elif key == ord('4'):
                CONFIDENCE = min(0.80, round(CONFIDENCE + 0.05, 2))
                print(f"Confidence: {CONFIDENCE:.2f}")
            elif key == ord('5'):
                smooth = max(1.0, round(smooth - 0.5, 1))
                print(f"Smooth: {smooth}")
            elif key == ord('6'):
                smooth = min(15.0, round(smooth + 0.5, 1))
                print(f"Smooth: {smooth}")
            elif key == ord('7'):
                SPEED_X = max(0.10, round(SPEED_X - 0.10, 2))
                print(f"Speed X: {SPEED_X:.2f}")
            elif key == ord('8'):
                SPEED_X = min(3.00, round(SPEED_X + 0.10, 2))
                print(f"Speed X: {SPEED_X:.2f}")
            elif key == ord('9'):
                SPEED_Y = max(0.10, round(SPEED_Y - 0.10, 2))
                print(f"Speed Y: {SPEED_Y:.2f}")
            elif key == ord('0'):
                SPEED_Y = min(3.00, round(SPEED_Y + 0.10, 2))
                print(f"Speed Y: {SPEED_Y:.2f}")
            elif key == ord('r'):
                ANTI_RECOIL_ENABLED = not ANTI_RECOIL_ENABLED
                print(f"Anti-Recoil: {'EIN' if ANTI_RECOIL_ENABLED else 'AUS'} ({ANTI_RECOIL}px)")
            elif key == ord('+') or key == ord('='):
                ANTI_RECOIL = min(15.0, round(ANTI_RECOIL + 0.5, 1))
                print(f"Anti-Recoil: {ANTI_RECOIL}px")
            elif key == ord('-'):
                ANTI_RECOIL = max(0.0, round(ANTI_RECOIL - 0.5, 1))
                print(f"Anti-Recoil: {ANTI_RECOIL}px")
            elif key == ord(']'):
                PREDICTION_FRAMES = min(5, PREDICTION_FRAMES + 1)
                print(f"Prediction: {PREDICTION_FRAMES} Frames")
            elif key == ord('['):
                PREDICTION_FRAMES = max(0, PREDICTION_FRAMES - 1)
                print(f"Prediction: {PREDICTION_FRAMES} Frames")
            elif key == ord('b'):
                ADS_BOOST = round(ADS_BOOST + 0.5, 1)
                if ADS_BOOST > 6.0:
                    ADS_BOOST = 1.0
                print(f"ADS-Boost: {ADS_BOOST}x")

    except KeyboardInterrupt:
        print("\nUnterbrochen.")
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
