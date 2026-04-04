"""
AIMBOT VISION v7 — Profi-Methode
=================================
Computer Vision Aimbot fuer Xbox RemotePlay via XIM Matrix.

NEUER ANSATZ (wie DMA/Profi-Aimbots):
  Jeden Frame: delta = (ziel - mitte) / SMOOTH
  → kmbox_net.move(dx, dy) — einfacher relativer Move
  → KEIN move_auto, KEIN Cooldown, KEIN Speed Curve

Hardware-Kette:
  Capture Card → OpenCV → YOLO (ONNX/DirectML) → KMBox Net → XIM Matrix → Xbox

XIM Matrix Einstellungen (WICHTIG!):
  - Synch: 0 (Off)
  - Smoothing: 0
  - Inner Deadzone: 0
  - Aiming Curve: Linear
  - Velocity Calibration durchfuehren!

Steuerung:
  A = ADS Toggle (Aimbot aktiv/inaktiv)
  P = Profil wechseln (Assist / Aimbot)
  M = Modell wechseln (COCO / FPS / Nano)
  1/2 = FOV +/-
  3/4 = Confidence +/-
  5/6 = Smooth-Faktor +/-
  7/8 = Speed X +/-
  9/0 = Speed Y +/-
  ESC = Beenden
"""

import os
import sys
import time
import math

# Backend-Pfad hinzufuegen
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

import cv2
import numpy as np
from yolo_onnx import YOLODetector, TARGET_CLASSES, IGNORE_CLASSES

# KMBox importieren (funktioniert nur auf Windows mit Netzwerk-Zugang)
try:
    import kmbox_net
    KMBOX_AVAILABLE = True
except Exception:
    KMBOX_AVAILABLE = False


# ============================================================
# HARDWARE-KONFIGURATION
# ============================================================
KMBOX_IP = "192.168.2.188"
KMBOX_PORT = 32778
KMBOX_UUID = "C14AE466"
CAPTURE_DEVICE = 0

# ============================================================
# MODELL-KONFIGURATION
# ============================================================
MODEL_MODE = "fps"
CONFIDENCE = 0.25
FOV_RADIUS = 250

# ============================================================
# AIM-KONFIGURATION — PROFI-METHODE
# ============================================================
# So machen es DMA-Aimbots und SunOner:
#   move_x = (ziel_x - mitte_x) / SMOOTH_FACTOR
#   move_y = (ziel_y - mitte_y) / SMOOTH_FACTOR
# Fertig. Kein Speed Curve, kein move_auto, kein Cooldown.

SMOOTH_FACTOR = 5.0         # Teilungsfaktor (hoeher = sanfter, niedriger = aggressiver)
                             # Profi-Bereich: 3-8

SPEED_X = 1.0               # X-Achsen Multiplikator
SPEED_Y = 0.80              # Y-Achse etwas reduziert (CoD: vertikale Sens hoeher)

# Deadzone: Wenn Ziel naeher als X Pixel an Mitte → nicht bewegen
DEADZONE = 5                # In Pixeln (klein! XIM soll feinjustieren)

# Max Pixel pro Frame (Sicherheit gegen Spinning)
MAX_MOVE_PER_FRAME = 80     # Mehr als 80px/Frame = verdaechtig, begrenzen

# ============================================================
# PROFILE
# ============================================================
PROFILES = {
    "assist": {
        "name": "AIM-ASSIST",
        "smooth": 7.0,      # Sanfter
        "deadzone": 8,
    },
    "aimbot": {
        "name": "AIMBOT",
        "smooth": 4.0,      # Aggressiver
        "deadzone": 4,
    },
}
PROFILE_ORDER = ["assist", "aimbot"]

# ============================================================
# ANZEIGE
# ============================================================
SHOW_WINDOW = True
WINDOW_SCALE = 0.5

# ============================================================
# FILTER
# ============================================================
DEAD_BODY_RATIO = 1.2       # Breite > 1.2 * Hoehe → Leiche
SKY_FILTER_RATIO = 0.10     # Obere 10% = Himmel
GROUND_FILTER_RATIO = 0.88  # Untere 12% = Boden/HUD
MIN_BOX_HEIGHT = 25


# ============================================================
# HILFSFUNKTIONEN
# ============================================================

def get_model_path(mode):
    """Gibt den besten verfuegbaren Modell-Pfad zurueck."""
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
# KALMAN-FILTER TRACKER (glaettet YOLO-Jitter)
# ============================================================
# Der Kalman-Filter ist NICHT fuer die Aim-Mathematik!
# Er glaettet nur die YOLO-Erkennungen (die von Frame zu Frame wackeln).
# Die eigentliche Aim-Berechnung ist simpel: delta / smooth.

class KalmanTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.x = np.zeros(4, dtype=np.float64)         # [x, y, vx, vy]
        self.P = np.eye(4, dtype=np.float64) * 500.0
        self.frames_seen = 0
        self.frames_lost = 0
        self.locked = False
        self.last_update = 0.0
        self.last_dt = 0.033

        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
        self.R = np.eye(2, dtype=np.float64) * 12.0
        self.Q_base = np.diag([1.0, 1.0, 8.0, 8.0])

    def predict(self, dt=None):
        if dt is None:
            dt = self.last_dt
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1]
        ], dtype=np.float64)
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
            self.P = np.eye(4, dtype=np.float64) * 100.0
        else:
            self.predict(self.last_dt)
            S = self.H @ self.P @ self.H.T + self.R
            K = self.P @ self.H.T @ np.linalg.inv(S)
            self.x = self.x + K @ (z - self.H @ self.x)
            self.P = (np.eye(4) - K @ self.H) @ self.P

        self.frames_seen += 1
        self.frames_lost = 0
        self.locked = self.frames_seen >= 2
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

        # Filter
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

        # Zielpunkt
        if class_name == "head":
            tx, ty = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        else:
            tx, ty = (x1 + x2) / 2.0, y1 + bh * 0.25

        # FOV-Check
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

    # FOV Kreis
    color = (0, 255, 0) if ads_active else (100, 100, 100)
    cv2.circle(frame, (cx, cy), fov_radius, color, 1)

    # Fadenkreuz
    cv2.line(frame, (cx - 15, cy), (cx + 15, cy), (255, 255, 255), 1)
    cv2.line(frame, (cx, cy - 15), (cx, cy + 15), (255, 255, 255), 1)

    # Erkennungen
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

    # Tracker-Linie
    if target_pos and ads_active:
        tx, ty = int(target_pos[0]), int(target_pos[1])
        cv2.line(frame, (cx, cy), (tx, ty), (0, 255, 255), 2)
        cv2.circle(frame, (tx, ty), 8, (0, 255, 255), 2)

    # Status
    status = f"FPS: {fps:.0f} | {profile['name']} | ADS: {'ON' if ads_active else 'OFF'}"
    if tracker.locked:
        vx, vy = tracker.get_velocity()
        status += f" | LOCKED v=({vx:.0f},{vy:.0f})"
    cv2.putText(frame, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    info = f"Smooth: {smooth:.1f} | FOV: {fov_radius} | Conf: {CONFIDENCE:.2f} | SpeedXY: {SPEED_X:.2f}/{SPEED_Y:.2f}"
    cv2.putText(frame, info, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return frame


# ============================================================
# HAUPTPROGRAMM
# ============================================================

def main():
    global CONFIDENCE, FOV_RADIUS, SMOOTH_FACTOR
    global SPEED_X, SPEED_Y, MODEL_MODE

    print("=" * 60)
    print("  AIMBOT VISION v7 — Profi-Methode")
    print("  delta / smooth → move() → jeden Frame")
    print("=" * 60)

    # --- 1. KMBox verbinden ---
    print("\n[1/3] KMBox verbinden...")
    if KMBOX_AVAILABLE:
        try:
            kmbox_net.init(KMBOX_IP, KMBOX_PORT, KMBOX_UUID)
            print(f"  OK: {KMBOX_IP}:{KMBOX_PORT}")
        except Exception as e:
            print(f"  WARNUNG: KMBox nicht erreichbar ({e})")
    else:
        print("  KMBox nicht verfuegbar — Anzeige-Modus")

    # --- 2. YOLO Modell laden ---
    print(f"\n[2/3] Modell laden ({MODEL_MODE})...")
    model_path = get_model_path(MODEL_MODE)
    if not model_path:
        print("  FEHLER: Kein ONNX-Modell gefunden!")
        return
    print(f"  Datei: {os.path.basename(model_path)}")
    detector = YOLODetector(model_path)

    # --- 3. Capture Card ---
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

    # --- Status ---
    print("\n" + "=" * 60)
    print("  BEREIT!")
    print("=" * 60)
    print(f"  Methode: delta / smooth → move() (wie Profi-Aimbots)")
    print(f"  Smooth: {SMOOTH_FACTOR} | Deadzone: {DEADZONE}px | Max: {MAX_MOVE_PER_FRAME}px")
    print(f"  Speed: X={SPEED_X} Y={SPEED_Y}")
    print(f"  FOV: {FOV_RADIUS}px | Confidence: {CONFIDENCE}")
    print()
    print("  WICHTIG — XIM Matrix Settings:")
    print("    Synch: 0 (Off) | Smoothing: 0 | Deadzone: 0 | Curve: Linear")
    print()
    print("  Tasten:")
    print("    A=ADS  P=Profil  M=Modell  ESC=Beenden")
    print("    1/2=FOV  3/4=Conf  5/6=Smooth  7/8=SpeedX  9/0=SpeedY")
    print("=" * 60)

    # --- Variablen ---
    tracker = KalmanTracker()
    profile_idx = 0
    profile = PROFILES[PROFILE_ORDER[profile_idx]]
    ads_active = False
    frame_count = 0
    fps = 0.0
    fps_timer = time.monotonic()
    smooth = profile["smooth"]
    deadzone = profile["deadzone"]

    # --- Hauptschleife ---
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame_count += 1
            now = time.monotonic()

            # FPS (alle 30 Frames)
            if frame_count % 30 == 0:
                elapsed = now - fps_timer
                fps = 30.0 / elapsed if elapsed > 0 else 0
                fps_timer = now

            # --- YOLO Inference ---
            all_dets = detector.detect(frame, conf_threshold=CONFIDENCE)
            target_dets = [d for d in all_dets if d["class_name"] in TARGET_CLASSES]
            tx, ty, target_det = pick_best_target(target_dets, fw, fh)

            # --- Tracking + Aim ---
            target_pos = None
            cx, cy = fw / 2.0, fh / 2.0

            if tx is not None:
                tracker.update(tx, ty)
                pos = tracker.get_position()
                if pos:
                    target_pos = pos

                    # ========================================
                    # PROFI AIM-MATHEMATIK (simpel!)
                    # ========================================
                    if ads_active and tracker.locked:
                        dx = pos[0] - cx
                        dy = pos[1] - cy
                        dist = math.sqrt(dx * dx + dy * dy)

                        if dist > deadzone:
                            # Einfache Division — genau wie DMA-Aimbots
                            move_x = (dx / smooth) * SPEED_X
                            move_y = (dy / smooth) * SPEED_Y

                            # Sicherheits-Limit
                            move_x = max(-MAX_MOVE_PER_FRAME, min(MAX_MOVE_PER_FRAME, move_x))
                            move_y = max(-MAX_MOVE_PER_FRAME, min(MAX_MOVE_PER_FRAME, move_y))

                            # Senden — jeden Frame, kein Cooldown!
                            ix = int(round(move_x))
                            iy = int(round(move_y))
                            if (abs(ix) > 0 or abs(iy) > 0) and KMBOX_AVAILABLE:
                                try:
                                    kmbox_net.move(ix, iy)
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
                cv2.imshow("AIMBOT v7", display)

            # --- Tastatur ---
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
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
