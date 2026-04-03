"""
AIMBOT VISION v6 — Profi-System
================================
Computer Vision Aimbot fuer Xbox RemotePlay via XIM Matrix.

Hardware-Kette:
  Capture Card (AVerMedia GC571) → OpenCV → YOLO11s (DirectML) → KMBox Net → XIM Matrix → Xbox

Aim-Algorithmus:
  Speed Curves + PD-Controller + XIM ADS-Kompensation + Anti-Jitter EMA

Steuerung:
  A = ADS Toggle (Aimbot aktiv/inaktiv)
  P = Profil wechseln (Assist / Aimbot)
  M = Modell wechseln (COCO / FPS / Nano)
  1/2 = FOV +/-
  3/4 = Confidence +/-
  5/6 = Sensitivity +/-
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
# "coco" = YOLO11s 80-Klassen (Primaer, beste Person-Erkennung)
# "fps"  = SunOner 10-Klassen (FPS-spezifisch)
# "nano" = SunOner 320px (schnell, weniger genau)
MODEL_MODE = "coco"

CONFIDENCE = 0.35           # Mindest-Confidence (0.20 - 0.80)
FOV_RADIUS = 250            # Aimbot FOV in Pixeln (nur Ziele innerhalb werden getrackt)

# ============================================================
# AIM-KONFIGURATION — SPEED CURVES (Magnet-Effekt)
# ============================================================
# Format: (max_distanz_pixel, multiplikator)
# Nah = langsam/klebrig, Weit = schnell → Profi "Magnet-Effekt"

SPEED_CURVE_NORMAL = [
    (15,   0.06),   # Sehr nah: Kaum bewegen (anti-jitter)
    (40,   0.20),   # Nah: Sanfte Mikro-Korrekturen
    (80,   0.45),   # Mittel-nah: Kontrolliertes Nachfuehren
    (140,  0.75),   # Mittel: Starkes Anziehen
    (220,  1.00),   # Weit: Voller Pull
    (9999, 1.20),   # Sehr weit: Maximaler Snap
]

SPEED_CURVE_LOCKED = [
    # Wenn bereits auf Ziel gelockt: Extra sanft halten
    (12,   0.03),   # Minimal: Praktisch stillstehen
    (30,   0.12),   # Sehr nah: Feinste Korrekturen
    (60,   0.30),   # Nah: Sanftes Nachfuehren
    (120,  0.55),   # Mittel: Kontrolliert folgen
    (200,  0.80),   # Weit: Zuegig nachziehen
    (9999, 1.00),   # Sehr weit: Volle Geschwindigkeit
]

# Achsen-Multiplikatoren (CoD: Y-Achse ist empfindlicher)
SPEED_X_MULTIPLIER = 1.00
SPEED_Y_MULTIPLIER = 0.75

# XIM Matrix ADS-Kompensation
# XIM schluckt ~70-80% der Mausbewegung waehrend ADS
# Muss hoch skaliert werden damit Korrekturen ankommen
XIM_ADS_BOOST = 3.0
XIM_MIN_MOVE = 25.0         # Minimum-Pixel damit XIM es registriert
MAX_CORRECTION = 600.0      # Maximum pro Korrektur

# Globaler Sensitivity-Multiplikator (Taste 5/6)
KMBOX_SENSITIVITY = 1.00

# ============================================================
# PROFILE
# ============================================================
PROFILES = {
    "assist": {
        "name": "AIM-ASSIST",
        "speed": 0.70,       # 70% der Speed Curve
        "deadzone": 30,      # Groessere Deadzone = weniger Micro-Tracking
    },
    "aimbot": {
        "name": "AIMBOT",
        "speed": 1.00,       # 100% Speed Curve
        "deadzone": 18,      # Kleinere Deadzone = praeziser
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
# Leichen-Filter: Bounding Box breiter als hoch → wahrscheinlich liegend
DEAD_BODY_RATIO = 1.2       # Breite > 1.2 * Hoehe → ignorieren
# Himmel-Filter: Erkennungen in den oberen X% ignorieren
SKY_FILTER_RATIO = 0.10     # Obere 10% = Himmel
# Boden-Filter: Erkennungen in den unteren X% ignorieren
GROUND_FILTER_RATIO = 0.88  # Untere 12% = Boden/HUD
# Minimum-Hoehe: Zu kleine Boxen ignorieren
MIN_BOX_HEIGHT = 25


# ============================================================
# HILFSFUNKTIONEN
# ============================================================

def get_model_path(mode):
    """Gibt den besten verfuegbaren Modell-Pfad zurueck."""
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend')

    priority = {
        "coco": ['yolo11s.onnx', 'bo7_v5_640.onnx', 'sunxds_640.onnx'],
        "fps":  ['sunxds_640.onnx', 'yolo11s.onnx', 'bo7_v5_640.onnx'],
        "nano": ['sunxds_nano_320.onnx', 'sunxds_640.onnx', 'yolo11s.onnx'],
    }

    candidates = priority.get(mode, priority["coco"])
    for name in candidates:
        path = os.path.join(base, name)
        if os.path.exists(path):
            return path

    # Letzter Fallback: Erstes .onnx im Ordner
    if os.path.isdir(base):
        for f in os.listdir(base):
            if f.endswith('.onnx'):
                return os.path.join(base, f)
    return None


def get_speed_multiplier(dist, curve):
    """Interpoliert den Speed-Curve Multiplikator fuer eine gegebene Distanz."""
    prev_dist, prev_mult = 0, curve[0][1]
    for max_dist, mult in curve:
        if dist <= max_dist:
            # Lineare Interpolation zwischen Stufen
            if max_dist == prev_dist:
                return mult
            t = (dist - prev_dist) / (max_dist - prev_dist)
            return prev_mult + t * (mult - prev_mult)
        prev_dist, prev_mult = max_dist, mult
    return curve[-1][1]


# ============================================================
# TARGET TRACKER (EMA-basiert, anti-jitter)
# ============================================================

class TargetTracker:
    """Trackt ein einzelnes Ziel mit Exponential Moving Average."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.ema_x = 0.0
        self.ema_y = 0.0
        self.frames_seen = 0
        self.frames_lost = 0
        self.locked = False
        self.last_update = 0.0

    def update(self, x, y, alpha=0.35):
        """Update Position mit EMA-Glaettung."""
        if self.frames_seen == 0:
            self.ema_x = x
            self.ema_y = y
        else:
            self.ema_x = alpha * x + (1.0 - alpha) * self.ema_x
            self.ema_y = alpha * y + (1.0 - alpha) * self.ema_y
        self.frames_seen += 1
        self.frames_lost = 0
        self.last_update = time.monotonic()

    def mark_lost(self):
        """Ziel nicht mehr erkannt."""
        self.frames_lost += 1
        if self.frames_lost > 5:
            self.reset()

    def is_stable(self, min_frames=2):
        """Ziel mindestens N Frames hintereinander erkannt?"""
        return self.frames_seen >= min_frames

    def get_position(self):
        """Gibt geglättete Position zurueck."""
        if self.frames_seen == 0:
            return None
        return (self.ema_x, self.ema_y)


# ============================================================
# AIM CONTROLLER (Sendet Korrekturen an KMBox)
# ============================================================

class AimController:
    """Profi Aim Controller fuer XIM Matrix.

    Features:
    - Speed Curves fuer Magnet-Effekt
    - Adaptive Dauer (weit=schnell, nah=sanft)
    - XIM ADS-Kompensation + Minimum-Clamp
    - Command-Overlap Schutz
    """

    def __init__(self):
        self.last_send_time = 0.0
        self.last_duration_s = 0.0
        self.correction_count = 0

    def calc_correction(self, tx, ty, fw, fh, profile, is_locked):
        """Berechnet Aim-Korrektur mit Speed Curves.

        Returns: (mx, my, dist) oder (0, 0, dist) wenn in Deadzone
        """
        cx = fw / 2.0
        cy = fh / 2.0
        dx = tx - cx
        dy = ty - cy
        dist = math.sqrt(dx * dx + dy * dy)

        if dist < profile["deadzone"]:
            return 0, 0, dist

        # 1. SPEED CURVE — Magnet-Effekt
        curve = SPEED_CURVE_LOCKED if is_locked else SPEED_CURVE_NORMAL
        speed_mult = get_speed_multiplier(dist, curve)

        # 2. Korrektur berechnen
        mx = dx * speed_mult * profile["speed"] * SPEED_X_MULTIPLIER * KMBOX_SENSITIVITY
        my = dy * speed_mult * profile["speed"] * SPEED_Y_MULTIPLIER * KMBOX_SENSITIVITY

        # 3. XIM ADS-Boost (kompensiert XIM-Daempfung)
        mx *= XIM_ADS_BOOST
        my *= XIM_ADS_BOOST

        # 4. XIM Minimum-Clamp (zu kleine Werte hochziehen)
        mag = math.sqrt(mx * mx + my * my)
        if 0 < mag < XIM_MIN_MOVE:
            scale = XIM_MIN_MOVE / mag
            mx *= scale
            my *= scale

        # 5. Maximum deckeln
        mag = math.sqrt(mx * mx + my * my)
        if mag > MAX_CORRECTION:
            scale = MAX_CORRECTION / mag
            mx *= scale
            my *= scale

        return mx, my, dist

    def send_correction(self, mx, my, dist):
        """Sendet move_auto an KMBox mit adaptiver Dauer."""
        if not KMBOX_AVAILABLE:
            return False

        now = time.monotonic()
        # Cooldown = Dauer der letzten Bewegung (kein Command-Overlap!)
        if now - self.last_send_time < self.last_duration_s:
            return False

        ix = int(round(mx))
        iy = int(round(my))
        if abs(ix) < 3 and abs(iy) < 3:
            return False

        # ADAPTIVE DAUER: Weit=kurz (schnelle Snaps), Nah=lang (smooth)
        if dist > 150:
            duration_ms = 80     # ~12 Korrekturen/Sek
        elif dist > 80:
            duration_ms = 120    # ~8 Korrekturen/Sek
        elif dist > 40:
            duration_ms = 160    # ~6 Korrekturen/Sek
        else:
            duration_ms = 200    # ~5 Korrekturen/Sek

        try:
            kmbox_net.move_auto(ix, iy, ms=duration_ms)
        except Exception:
            return False

        self.last_send_time = now
        self.last_duration_s = duration_ms / 1000.0
        self.correction_count += 1
        return True

    def reset(self):
        self.correction_count = 0


# ============================================================
# ZIELAUSWAHL — Waehlt bestes Ziel aus allen Erkennungen
# ============================================================

def pick_best_target(detections, frame_w, frame_h):
    """Waehlt das beste Ziel aus den Erkennungen.

    Prioritaet:
    1. Innerhalb FOV
    2. Head-Shots haben Bonus
    3. Naehestes zur Bildmitte

    Returns: (target_x, target_y, detection) oder (None, None, None)
    """
    cx = frame_w / 2.0
    cy = frame_h / 2.0
    best = None
    best_score = float('inf')

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        bw = x2 - x1
        bh = y2 - y1
        class_name = det["class_name"]

        # --- FILTER ---

        # Leichen-Filter (liegend = breiter als hoch)
        if bw > bh * DEAD_BODY_RATIO:
            continue

        # Minimum-Hoehe Filter
        if bh < MIN_BOX_HEIGHT:
            continue

        # Himmel-Filter (obere 10%)
        center_y = (y1 + y2) / 2.0
        if center_y < frame_h * SKY_FILTER_RATIO:
            continue

        # Boden-Filter (untere 12%)
        if center_y > frame_h * GROUND_FILTER_RATIO:
            continue

        # Ignorierte Klassen
        if class_name in IGNORE_CLASSES:
            continue

        # --- ZIELPUNKT ---
        if class_name == "head":
            # Head: Mitte der Box
            tx = (x1 + x2) / 2.0
            ty = (y1 + y2) / 2.0
        else:
            # Body: Oberes Drittel (Schulter/Kopf-Bereich)
            tx = (x1 + x2) / 2.0
            ty = y1 + bh * 0.25

        # FOV-Check
        dx = tx - cx
        dy = ty - cy
        dist = math.sqrt(dx * dx + dy * dy)
        if dist > FOV_RADIUS:
            continue

        # Score: Distanz (naeher = besser), Head-Bonus
        score = dist
        if class_name == "head":
            score *= 0.6  # Head-Bonus: 40% naeher gewichtet

        if score < best_score:
            best_score = score
            best = (tx, ty, det)

    if best:
        return best[0], best[1], best[2]
    return None, None, None


# ============================================================
# OVERLAY ZEICHNEN
# ============================================================

def draw_overlay(frame, all_dets, target_pos, tracker, fps, ads_active, profile_name, fov_radius):
    """Zeichnet HUD-Overlay auf den Frame."""
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2

    # FOV Kreis
    fov_color = (0, 255, 0) if ads_active else (100, 100, 100)
    cv2.circle(frame, (cx, cy), fov_radius, fov_color, 1)

    # Fadenkreuz
    cv2.line(frame, (cx - 15, cy), (cx + 15, cy), (255, 255, 255), 1)
    cv2.line(frame, (cx, cy - 15), (cx, cy + 15), (255, 255, 255), 1)

    # Alle Erkennungen zeichnen
    for det in all_dets:
        x1, y1, x2, y2 = det["bbox"]
        cls = det["class_name"]
        conf = det["confidence"]

        if cls in TARGET_CLASSES:
            color = (0, 0, 255)  # Rot = Ziel
        elif cls in IGNORE_CLASSES:
            color = (128, 128, 128)  # Grau = ignoriert
        else:
            color = (255, 255, 0)  # Cyan = sonstige

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{cls} {conf:.0%}"
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Tracker-Linie zum Ziel
    if target_pos and ads_active:
        tx, ty = int(target_pos[0]), int(target_pos[1])
        cv2.line(frame, (cx, cy), (tx, ty), (0, 255, 255), 2)
        cv2.circle(frame, (tx, ty), 8, (0, 255, 255), 2)

    # Status-Text
    status = f"FPS: {fps:.0f} | {profile_name} | ADS: {'ON' if ads_active else 'OFF'}"
    if tracker.locked:
        status += f" | LOCKED (#{tracker.frames_seen})"
    cv2.putText(frame, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Sensitivity-Info
    info = f"Sens: {KMBOX_SENSITIVITY:.2f} | FOV: {fov_radius} | Conf: {CONFIDENCE:.2f}"
    cv2.putText(frame, info, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return frame


# ============================================================
# HAUPTPROGRAMM
# ============================================================

def main():
    global CONFIDENCE, FOV_RADIUS, KMBOX_SENSITIVITY
    global SPEED_X_MULTIPLIER, SPEED_Y_MULTIPLIER
    global MODEL_MODE

    print("=" * 60)
    print("  AIMBOT VISION v6 — Profi-System")
    print("=" * 60)

    # --- 1. KMBox verbinden ---
    print("\n[1/3] KMBox verbinden...")
    if KMBOX_AVAILABLE:
        try:
            kmbox_net.init(KMBOX_IP, KMBOX_PORT, KMBOX_UUID)
            print(f"  KMBox verbunden: {KMBOX_IP}:{KMBOX_PORT}")
        except Exception as e:
            print(f"  WARNUNG: KMBox nicht erreichbar ({e})")
            print(f"  → Aimbot laeuft im Anzeige-Modus (keine Mausbewegung)")
    else:
        print("  KMBox Modul nicht verfuegbar (Windows erforderlich)")
        print("  → Anzeige-Modus")

    # --- 2. YOLO Modell laden ---
    print(f"\n[2/3] YOLO Modell laden (Modus: {MODEL_MODE})...")
    model_path = get_model_path(MODEL_MODE)
    if not model_path:
        print("  FEHLER: Kein ONNX-Modell gefunden im backend/ Ordner!")
        print("  Benoetigte Datei: backend/yolo11s.onnx")
        return

    print(f"  Datei: {os.path.basename(model_path)}")
    detector = YOLODetector(model_path)

    # --- 3. Capture Card oeffnen ---
    print(f"\n[3/3] Capture Card oeffnen (Device {CAPTURE_DEVICE})...")
    cap = cv2.VideoCapture(CAPTURE_DEVICE, cv2.CAP_DSHOW)
    if not cap.isOpened():
        # Fallback ohne DirectShow
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
    print("  SYSTEM BEREIT!")
    print("=" * 60)
    print(f"  Profil: {PROFILES[PROFILE_ORDER[0]]['name']}")
    print(f"  Speed Curves: {len(SPEED_CURVE_NORMAL)} Stufen (Magnet-Effekt)")
    print(f"  XIM ADS-Boost: {XIM_ADS_BOOST} | Min: {XIM_MIN_MOVE}px | Max: {MAX_CORRECTION}px")
    print(f"  Sensitivity: {KMBOX_SENSITIVITY:.2f}")
    print(f"  FOV: {FOV_RADIUS}px | Confidence: {CONFIDENCE:.2f}")
    print(f"  Filter: Leichen(>{DEAD_BODY_RATIO}x), Himmel(<{SKY_FILTER_RATIO*100:.0f}%), Min-H({MIN_BOX_HEIGHT}px)")
    print()
    print("  Steuerung:")
    print("    A = ADS Toggle | P = Profil | M = Modell")
    print("    1/2 = FOV | 3/4 = Confidence | 5/6 = Sensitivity")
    print("    7/8 = Speed X | 9/0 = Speed Y | ESC = Beenden")
    print("=" * 60)

    # --- Variablen ---
    tracker = TargetTracker()
    aim_ctrl = AimController()
    profile_idx = 0
    profile = PROFILES[PROFILE_ORDER[profile_idx]]
    ads_active = False
    frame_count = 0
    fps = 0.0
    fps_timer = time.monotonic()

    # --- Hauptschleife ---
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame_count += 1
            now = time.monotonic()

            # FPS berechnen (alle 30 Frames)
            if frame_count % 30 == 0:
                elapsed = now - fps_timer
                fps = 30.0 / elapsed if elapsed > 0 else 0
                fps_timer = now

            # --- YOLO Inference ---
            all_dets = detector.detect(frame, conf_threshold=CONFIDENCE)

            # --- Nur Target-Klassen fuer Aimbot ---
            target_dets = [d for d in all_dets if d["class_name"] in TARGET_CLASSES]

            # --- Bestes Ziel waehlen ---
            tx, ty, target_det = pick_best_target(target_dets, fw, fh)

            # --- Tracking + Aim ---
            target_pos = None

            if tx is not None and ads_active:
                # Adaptiver EMA: Schnell erfassen, sanft halten
                alpha = 0.50 if tracker.frames_seen < 3 else 0.30
                tracker.update(tx, ty, alpha=alpha)

                if tracker.is_stable(2):
                    pos = tracker.get_position()
                    if pos:
                        tracker.locked = True
                        target_pos = pos

                        # Korrektur berechnen + senden
                        mx, my, dist = aim_ctrl.calc_correction(
                            pos[0], pos[1], fw, fh, profile, tracker.locked
                        )
                        if mx != 0 or my != 0:
                            aim_ctrl.send_correction(mx, my, dist)
            elif tx is not None:
                # Nicht ADS: Trotzdem tracken (fuer schnellen Lock beim ADS-Druecken)
                alpha = 0.40 if tracker.frames_seen < 3 else 0.25
                tracker.update(tx, ty, alpha=alpha)
                target_pos = tracker.get_position()
                tracker.locked = False
            else:
                tracker.mark_lost()
                tracker.locked = False

            # --- Anzeige ---
            if SHOW_WINDOW:
                display = draw_overlay(
                    frame.copy(), all_dets, target_pos,
                    tracker, fps, ads_active, profile["name"], FOV_RADIUS
                )

                if WINDOW_SCALE != 1.0:
                    new_w = int(fw * WINDOW_SCALE)
                    new_h = int(fh * WINDOW_SCALE)
                    display = cv2.resize(display, (new_w, new_h))

                cv2.imshow("AIMBOT v6", display)

            # --- Tastatur-Steuerung ---
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                print("Beende...")
                break

            elif key == ord('a'):
                ads_active = not ads_active
                state = "EIN" if ads_active else "AUS"
                print(f"ADS: {state}")
                if not ads_active:
                    tracker.reset()
                    aim_ctrl.reset()

            elif key == ord('p'):
                profile_idx = (profile_idx + 1) % len(PROFILE_ORDER)
                profile = PROFILES[PROFILE_ORDER[profile_idx]]
                print(f"Profil: {profile['name']} (Speed: {profile['speed']}, DZ: {profile['deadzone']})")

            elif key == ord('m'):
                modes = ["coco", "fps", "nano"]
                current_idx = modes.index(MODEL_MODE) if MODEL_MODE in modes else 0
                MODEL_MODE = modes[(current_idx + 1) % len(modes)]
                new_path = get_model_path(MODEL_MODE)
                if new_path:
                    print(f"Lade Modell: {MODEL_MODE} ({os.path.basename(new_path)})...")
                    detector = YOLODetector(new_path)
                    tracker.reset()
                else:
                    print(f"Modell '{MODEL_MODE}' nicht gefunden!")
                    MODEL_MODE = modes[current_idx]  # Zurueck

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
                KMBOX_SENSITIVITY = max(0.10, round(KMBOX_SENSITIVITY - 0.10, 2))
                print(f"Sensitivity: {KMBOX_SENSITIVITY:.2f}")
            elif key == ord('6'):
                KMBOX_SENSITIVITY = min(3.00, round(KMBOX_SENSITIVITY + 0.10, 2))
                print(f"Sensitivity: {KMBOX_SENSITIVITY:.2f}")
            elif key == ord('7'):
                SPEED_X_MULTIPLIER = max(0.10, round(SPEED_X_MULTIPLIER - 0.10, 2))
                print(f"Speed X: {SPEED_X_MULTIPLIER:.2f}")
            elif key == ord('8'):
                SPEED_X_MULTIPLIER = min(3.00, round(SPEED_X_MULTIPLIER + 0.10, 2))
                print(f"Speed X: {SPEED_X_MULTIPLIER:.2f}")
            elif key == ord('9'):
                SPEED_Y_MULTIPLIER = max(0.10, round(SPEED_Y_MULTIPLIER - 0.10, 2))
                print(f"Speed Y: {SPEED_Y_MULTIPLIER:.2f}")
            elif key == ord('0'):
                SPEED_Y_MULTIPLIER = min(3.00, round(SPEED_Y_MULTIPLIER + 0.10, 2))
                print(f"Speed Y: {SPEED_Y_MULTIPLIER:.2f}")

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
