"""
AIMBOT VISION v14 — ADVANCED TRACKING
=======================================
Neu in v14:
  - Velocity Prediction: Zielt wohin der Gegner sich BEWEGT
  - Dynamic Strength: Starker Snap bei Distanz, Praezision bei Naehe
  - Sticky Target: Bleibt auf aktuellem Ziel, springt nicht hin und her
  - Schnellerer Tracker: Alpha 0.6 (war 0.4)
  - Cooldown 2 Frames (war 4), Max-Move 50px (war 30)
  - Groesse-Bonus: Naehere (groessere) Gegner werden bevorzugt

Performance-Optimierungen:
  1. Threaded Capture (Buffer=1, immer neuestes Frame)
  2. Threaded Inference (AI laeuft parallel, non-blocking)
  3. Inference-Guard (neuer Frame nur wenn AI fertig)
  4. ROI-Cropping (640x640 Mitte, kein Full-Screen Resize)
  5. Display-Throttle (Overlay nur jeden 2. Frame = weniger GPU-Last)
  6. Minimap-Throttle (nur jeden 3. Frame = spart ~0.5ms)
  7. INTER_NEAREST Resize (schnellster Resize-Algorithmus)
  8. Pre-Computed Werte (Screen-Center etc. einmal berechnen)
  9. Pre-Alloc Buffers (kein numpy.zeros() pro Frame)
  10. Overlay Toggle (V-Taste: aus = +10-15 FPS extra)

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
  B     = Minimap Debug an/aus
  F     = Teammate-Schutz an/aus
  Pfeiltasten = Minimap-Position verschieben (bei Debug-Modus)
  I/K=hoch/runter  J/L=links/rechts (Minimap, bei Debug)
  +/-   = Minimap groesser/kleiner
  ESC   = Beenden
"""

import os
import sys
import time
import math
import json
import threading
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

import cv2
import numpy as np
from yolo_onnx import YOLODetector, TARGET_CLASSES, IGNORE_CLASSES
from minimap_reader import MinimapReader

try:
    import kmbox_net
    KMBOX_AVAILABLE = True
except Exception:
    KMBOX_AVAILABLE = False

try:
    from titan_two import TitanTwo, pixels_to_stick, RECOIL_PROFILES, get_recoil_profile
    from titan_two import MACRO_NONE, MACRO_DROPSHOT, MACRO_SNAKING, MACRO_SLIDE_CANCEL
    from titan_two import MACRO_BUNNY_HOP, MACRO_AUTO_FIRE, MACRO_YY_SWAP
    TITAN_MODULE_AVAILABLE = True
except Exception:
    TITAN_MODULE_AVAILABLE = False

# Titan Two wird in main() initialisiert (braucht gtuner Modul)


# ============================================================
# CONFIG FILE
# ============================================================
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

DEFAULT_CONFIG = {
    "kmbox_ip": "192.168.2.188",
    "kmbox_port": 32778,
    "kmbox_uuid": "C14AE466",
    "capture_device": 0,
    "model_mode": "fps",
    "confidence": 0.32,
    "fov_radius": 280,
    "strength": 4.8,
    "deadzone": 3,
    "max_move": 100,
    "cooldown_frames": 0,
    "use_roi_crop": True,
    "roi_size": 640,
    "target_fps": 60,
    "show_overlay": True,
    "use_nano": False,
    "dead_body_ratio": 1.0,
    "sky_filter_ratio": 0.10,
    "ground_filter_ratio": 0.88,
    "min_box_height": 30,
    "minimap_enabled": True,
    "minimap_x": 40,
    "minimap_y": 140,
    "minimap_size": 200,
    "teammate_protection": True,
    "teammate_tolerance": 35,
    "game_fov": 100,
    "input_mode": "auto",
    "titan_speed_x": 55.0,
    "titan_speed_y": 55.0,
    "titan_sensitivity": 1.0,
    "recoil_profile": "default",
    "anti_recoil_enabled": False,
    "macro_dropshot_key": "q",
    "macro_snaking_key": "e",
    "macro_slidecancel_key": "r",
    "macro_autofire_key": "t",
}


def load_config():
    """Laedt Config aus JSON oder erstellt Default."""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                saved = json.load(f)
            cfg = DEFAULT_CONFIG.copy()
            cfg.update(saved)
            print(f"  Config geladen: {CONFIG_FILE}")
            return cfg
        except Exception as e:
            print(f"  Config Fehler: {e} — nutze Default")
    return DEFAULT_CONFIG.copy()


def save_config(cfg):
    """Speichert Config als JSON."""
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(cfg, f, indent=2)
    except Exception:
        pass


# ============================================================
# MODEL PATH
# ============================================================
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
# THREADED CAPTURE
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

    def release(self):
        self.running = False
        self.thread.join(timeout=2)
        self.cap.release()


# ============================================================
# THREADED INFERENCE
# ============================================================
class FastInference:
    def __init__(self, detector):
        self.detector = detector
        self.input_frame = None
        self.input_roi_info = None
        self._conf = 0.40
        self.results = []
        self.inf_ms = 0.0
        self.busy = False
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
            self.busy = True

            with self.lock_in:
                frame = self.input_frame
                roi_info = self.input_roi_info
                conf = self._conf

            if frame is None:
                self.busy = False
                continue

            t0 = time.monotonic()
            dets = self.detector.detect(frame, conf_threshold=conf)
            ms = (time.monotonic() - t0) * 1000

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
            self.busy = False
            self.has_output.set()

    def submit(self, frame, conf, roi_info=None):
        """Nur senden wenn AI-Thread nicht beschaeftigt (kein Stau)."""
        if self.busy:
            return False
        with self.lock_in:
            self.input_frame = frame
            self.input_roi_info = roi_info
            self._conf = conf
        self.has_input.set()
        return True

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
# TRACKER MIT VELOCITY PREDICTION
# ============================================================
class SmoothTracker:
    """Verfolgt ein Ziel mit EMA-Glaettung + Velocity Prediction + Oszillations-Erkennung."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.x = None
        self.y = None
        self.prev_x = None
        self.prev_y = None
        self.vx = 0.0           # Geschwindigkeit X (px/frame)
        self.vy = 0.0           # Geschwindigkeit Y (px/frame)
        self.alpha = 0.5
        self.frames = 0
        self.lost = 0
        self.target_id = None
        self.target_h = 0
        self.prev_dx = 0.0
        self.prev_dy = 0.0
        self.osc_count = 0

    def update(self, mx, my, bbox_h=0):
        # Dynamischer Alpha: Close = instant, Range = schnell
        if bbox_h > 120:
            alpha = 0.95  # Close: sofort
        elif bbox_h > 80:
            alpha = 0.85
        elif bbox_h > 50:
            alpha = 0.75
        else:
            alpha = 0.65

        if self.x is None:
            self.x, self.y = mx, my
            self.prev_x, self.prev_y = mx, my
        else:
            self.prev_x, self.prev_y = self.x, self.y
            self.x += alpha * (mx - self.x)
            self.y += alpha * (my - self.y)

            # Velocity berechnen (EMA-geglaettet)
            raw_vx = self.x - self.prev_x
            raw_vy = self.y - self.prev_y
            self.vx = 0.5 * self.vx + 0.5 * raw_vx
            self.vy = 0.5 * self.vy + 0.5 * raw_vy

        self.target_h = bbox_h
        self.frames += 1
        self.lost = 0

    def get_predicted_position(self, lead_frames=2.5):
        """Position + Velocity Prediction (zielt VORAUS).
        Nur bei echtem Movement — ignoriert Box-Jitter."""
        if self.x is None:
            return None
        if self.frames < 3:
            return (self.x, self.y)
        # Nur vorhersagen wenn Geschwindigkeit > Jitter-Schwelle (2px/frame)
        speed = math.sqrt(self.vx * self.vx + self.vy * self.vy)
        if speed < 2.0:
            return (self.x, self.y)
        px = self.x + self.vx * lead_frames
        py = self.y + self.vy * lead_frames
        return (px, py)

    def check_oscillation(self, dx, dy):
        """Erkennt ob Aim hin-und-her pendelt. Returns damping factor 0.0-1.0."""
        # Richtungswechsel erkennen (Vorzeichen aendert sich)
        if (dx * self.prev_dx < 0) or (dy * self.prev_dy < 0):
            self.osc_count = min(self.osc_count + 1, 6)
        else:
            self.osc_count = max(self.osc_count - 1, 0)

        self.prev_dx = dx
        self.prev_dy = dy

        # Je mehr Oszillation, desto staerker daempfen
        if self.osc_count >= 4:
            return 0.2   # Starkes Daempfen
        elif self.osc_count >= 2:
            return 0.5   # Mittleres Daempfen
        return 1.0       # Kein Daempfen

    def mark_lost(self):
        self.lost += 1
        if self.lost > 4:
            self.reset()

    def get_position(self):
        """Gibt geglättete Position zurueck."""
        if self.x is None:
            return None
        return (self.x, self.y)

    def get_raw_position(self):
        if self.x is None:
            return None
        return (self.x, self.y)

    @property
    def locked(self):
        return self.frames >= 1 and self.lost == 0


# ============================================================
# ZIELAUSWAHL MIT TEAMMATE-SCHUTZ + STICKY TARGET
# ============================================================
def pick_best_target(detections, frame_w, frame_h, cfg, minimap=None, sticky_pos=None, frame=None, sticky_h=0, sticky_frames=0):
    """Waehlt das beste Ziel mit CLOSE-FIGHT STICKY AIM.
    
    sticky_h:      Bbox-Hoehe des aktuell getrackten Ziels (groesser = naeher)
    sticky_frames: Wie viele Frames das aktuelle Ziel schon gelockt ist
    """
    cx, cy = frame_w / 2.0, frame_h / 2.0
    fov = cfg["fov_radius"]
    dead_ratio = cfg["dead_body_ratio"]
    min_h = cfg["min_box_height"]
    sky = cfg["sky_filter_ratio"]
    ground = cfg["ground_filter_ratio"]
    tm_protect = cfg["teammate_protection"] and frame is not None

    # CLOSE-FIGHT STICKY: Maximaler Kleber
    if sticky_h > 120:
        sticky_radius = 500
        sticky_bonus = 0.02
        min_lock = 30
    elif sticky_h > 80:
        sticky_radius = 350
        sticky_bonus = 0.05
        min_lock = 18
    elif sticky_h > 50:
        sticky_radius = 200
        sticky_bonus = 0.10
        min_lock = 10
    else:
        sticky_radius = 120
        sticky_bonus = 0.20
        min_lock = 5

    best = None
    best_score = float('inf')

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        bw, bh = x2 - x1, y2 - y1
        cls = det["class_name"]

        if bw > bh * dead_ratio:
            continue
        if bh < min_h:
            continue
        mid_y = (y1 + y2) / 2.0
        if mid_y < frame_h * sky:
            continue
        if mid_y > frame_h * ground:
            continue
        if cls in IGNORE_CLASSES:
            continue

        # Zielpunkt: Mitte X, OBERKOERPER Y (35% von oben — stabiler als Box-Mitte)
        tx = (x1 + x2) / 2.0
        ty = y1 + (y2 - y1) * 0.35

        dist = math.sqrt((tx - cx) ** 2 + (ty - cy) ** 2)
        if dist > fov:
            continue

        # TEAMMATE-SCHUTZ: Blaues/Gruenes Namensschild OBERHALB der Box?
        if tm_protect and _has_teammate_nameplate(frame, x1, y1, x2, bw, frame_w, frame_h):
            continue

        # Score berechnen
        score = dist

        # Groesse-Bonus (naehere Ziele bevorzugen)
        if bh > 80:
            score *= 0.7

        # STICKY AIM — Close-Fight Kleber
        if sticky_pos is not None:
            stick_dist = math.sqrt((tx - sticky_pos[0]) ** 2 + (ty - sticky_pos[1]) ** 2)
            if stick_dist < sticky_radius:
                # Dieses Ziel ist unser aktuelles — massiver Bonus
                score *= sticky_bonus

                # MINIMUM LOCK: Innerhalb der Lock-Zeit ist dieses Ziel
                # praktisch unschlagbar (Score wird auf fast 0 gesetzt)
                if sticky_frames < min_lock:
                    score *= 0.01

        if score < best_score:
            best_score = score
            best = (tx, ty, det)

    return best if best else (None, None, None)


# Teammate-Farberkennung: ALLE Teammate-Farben (MP + Warzone)
_BLUE_LOW = np.array([90, 120, 120])
_BLUE_HIGH = np.array([130, 255, 255])
_GREEN_LOW = np.array([35, 120, 120])
_GREEN_HIGH = np.array([85, 255, 255])
_YELLOW_LOW = np.array([18, 120, 120])
_YELLOW_HIGH = np.array([34, 255, 255])
_ORANGE_LOW = np.array([8, 120, 120])
_ORANGE_HIGH = np.array([18, 255, 255])

def _has_teammate_nameplate(frame, x1, y1, x2, bw, fw, fh):
    """Prueft ob OBERHALB einer Detection ein Teammate-Namensschild ist.
    Erkennt: Blau, Gruen, Gelb, Orange (MP + Warzone)."""
    pad = int(bw * 0.2)
    nx1 = max(0, int(x1) - pad)
    nx2 = min(fw, int(x2) + pad)
    ny1 = max(0, int(y1) - 35)
    ny2 = max(0, int(y1) - 2)

    if ny2 <= ny1 or nx2 <= nx1:
        return False

    region = frame[ny1:ny2, nx1:nx2]
    if region.size == 0:
        return False

    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    tm_pixels = (
        cv2.countNonZero(cv2.inRange(hsv, _BLUE_LOW, _BLUE_HIGH))
        + cv2.countNonZero(cv2.inRange(hsv, _GREEN_LOW, _GREEN_HIGH))
        + cv2.countNonZero(cv2.inRange(hsv, _YELLOW_LOW, _YELLOW_HIGH))
        + cv2.countNonZero(cv2.inRange(hsv, _ORANGE_LOW, _ORANGE_HIGH))
    )
    total_pixels = region.shape[0] * region.shape[1]

    return (tm_pixels / total_pixels) > 0.05 if total_pixels > 0 else False


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
    cfg = load_config()

    print("=" * 60)
    print("  AIMBOT v14 — ADVANCED TRACKING")
    print("  Velocity Prediction | Dynamic Strength | Sticky Target")
    print("=" * 60)

    # KMBox
    if KMBOX_AVAILABLE:
        try:
            kmbox_net.init(cfg["kmbox_ip"], cfg["kmbox_port"], cfg["kmbox_uuid"])
            print(f"  KMBox: OK")
        except Exception as e:
            print(f"  KMBox: {e}")

    # Titan Two
    titan = None
    if TITAN_MODULE_AVAILABLE:
        titan = TitanTwo()
        if not titan.connected:
            titan = None

    # Input Modus bestimmen
    input_mode = cfg["input_mode"]
    if input_mode == "auto":
        if titan and titan.connected:
            input_mode = "titan"
        elif KMBOX_AVAILABLE:
            input_mode = "kmbox"
        else:
            input_mode = "none"
    print(f"  Input: {input_mode.upper()}"
          f"{' (Titan Two → praezise Stick-Werte!)' if input_mode == 'titan' else ''}"
          f"{' (KMBox → Maus-Emulation)' if input_mode == 'kmbox' else ''}")

    # YOLO
    model_path = get_model_path(cfg["model_mode"], cfg["use_nano"])
    if not model_path:
        print("FEHLER: Kein Modell!")
        return
    print(f"  Modell: {os.path.basename(model_path)}")
    detector = YOLODetector(model_path)

    # Capture
    print(f"\n  Starte Threads...")
    try:
        cap = FastCapture(cfg["capture_device"], 1920, 1080, cfg["target_fps"])
    except RuntimeError as e:
        print(f"  FEHLER: {e}")
        return
    fw, fh = cap.width, cap.height

    # Inference Thread
    inferencer = FastInference(detector)
    print(f"  AI-Thread: gestartet")

    # Minimap Reader
    minimap = MinimapReader(
        x=cfg["minimap_x"],
        y=cfg["minimap_y"],
        size=cfg["minimap_size"],
    )
    print(f"  Minimap: {'EIN' if cfg['minimap_enabled'] else 'AUS'} "
          f"(x={cfg['minimap_x']}, y={cfg['minimap_y']}, {cfg['minimap_size']}px)")
    print(f"  Teammate-Schutz: {'EIN' if cfg['teammate_protection'] else 'AUS'} "
          f"(Toleranz: {cfg['teammate_tolerance']}°)")

    print(f"  ROI: {'EIN' if cfg['use_roi_crop'] else 'AUS'} ({cfg['roi_size']}x{cfg['roi_size']})")
    print(f"  Config: {CONFIG_FILE}")
    print(f"\n  Tasten:")
    print(f"  A=Assist  V=Overlay  C=ROI  N=Nano/Full  M=Modell")
    print(f"  B=Minimap-Debug  F=Teammate-Schutz")
    print(f"  Pfeile: I/K=hoch/runter  J/L=links/rechts  +/-=Groesse")
    print(f"  1/2=STR  3/4=FOV  5/6=DZ  7/8=Conf  9/0=CD  ESC=Quit")
    print("=" * 60)

    tracker = SmoothTracker()
    active = True
    fc = 0
    fps = 0.0
    strength = cfg["strength"]
    cooldown = 0
    all_dets = []
    inf_ms = 0.0
    show_minimap_debug = False

    frame_times = deque(maxlen=60)

    # Pre-computed (einmal berechnen statt jeden Frame)
    scr_cx = fw / 2.0
    scr_cy = fh / 2.0
    hcx = fw // 2
    hcy = fh // 2
    disp_w = fw // 2
    disp_h = fh // 2

    # Throttle-Counter
    tm_count = 0
    en_count = 0

    # Pre-alloc fuer Overlay-aus Modus
    tiny = np.zeros((60, 300, 3), dtype=np.uint8)

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.001)
                continue

            fc += 1

            # Frame an AI-Thread senden (nur wenn nicht beschaeftigt)
            if cfg["use_roi_crop"]:
                roi, off_x, off_y = crop_center(frame, cfg["roi_size"])
                inferencer.submit(roi, cfg["confidence"], (off_x, off_y))
            else:
                inferencer.submit(frame, cfg["confidence"], None)

            # Minimap nur jeden 3. Frame lesen (Position aendert sich langsam)
            if cfg["minimap_enabled"] and fc % 3 == 0:
                teammates, enemies = minimap.update(frame)
                tm_count = len(teammates)
                en_count = len(enemies)

            # Neueste AI-Ergebnisse (non-blocking)
            new_dets, new_ms = inferencer.get_results()
            if new_dets is not None:
                all_dets = new_dets
                inf_ms = new_ms

            # Zielauswahl + Aim (JEDEN Frame — Latenz-kritisch!)
            targets = [d for d in all_dets if d["class_name"] in TARGET_CLASSES]
            sticky = tracker.get_raw_position()
            tx, ty, tdet = pick_best_target(
                targets, fw, fh, cfg,
                sticky_pos=sticky, frame=frame,
                sticky_h=tracker.target_h,
                sticky_frames=tracker.frames,
            )

            aim_pos = None

            if tx is not None:
                # Bbox-Hoehe des gewaehlten Ziels an Tracker weitergeben
                det_h = tdet["bbox"][3] - tdet["bbox"][1] if tdet else 0
                tracker.update(tx, ty, bbox_h=det_h)

                # Velocity Prediction: Ziele voraus wo der Gegner HINLAEUFT
                pred = tracker.get_predicted_position(lead_frames=2.5)
                pos = pred if pred else tracker.get_position()

                if pos:
                    aim_pos = pos

                    if active and tracker.locked:
                        dx = pos[0] - scr_cx
                        dy = pos[1] - scr_cy
                        dist = math.sqrt(dx * dx + dy * dy)

                        if dist > cfg["deadzone"] and cooldown <= 0:
                            if input_mode == "titan" and titan:
                                sx, sy = pixels_to_stick(
                                    dx, dy, fw, fh,
                                    sensitivity=cfg["titan_sensitivity"],
                                    speed_x=cfg["titan_speed_x"],
                                    speed_y=cfg["titan_speed_y"],
                                )
                                osc_damp = tracker.check_oscillation(dx, dy)
                                titan.set_aim(sx * osc_damp, sy * osc_damp)
                                if cfg["anti_recoil_enabled"] and titan.is_firing():
                                    recoil_y = get_recoil_profile(cfg["recoil_profile"])
                                    titan.set_anti_recoil(recoil_y)
                                cooldown = cfg["cooldown_frames"]

                            elif input_mode == "kmbox" and KMBOX_AVAILABLE:
                                # AGGRESSIVE TRACKING mit move_auto
                                dyn_str = strength
                                if det_h > 120:
                                    dyn_str = min(strength * 2.5, 8.0)
                                elif det_h > 80:
                                    dyn_str = min(strength * 2.0, 6.5)
                                elif det_h > 50:
                                    dyn_str = min(strength * 1.6, 5.0)

                                mx = dx * dyn_str
                                my = dy * dyn_str

                                # Anti-Pendel: Daempfung nah am Ziel
                                if dist < 18:
                                    mx *= dist / 18.0
                                    my *= dist / 18.0

                                lim = cfg["max_move"]
                                mx = max(-lim, min(lim, mx))
                                my = max(-lim, min(lim, my))
                                ix = int(round(mx))
                                iy = int(round(my))

                                # XIM Bypass: min 6px
                                if ix != 0 or iy != 0:
                                    if 0 < abs(ix) < 6:
                                        ix = 6 if ix > 0 else -6
                                    if 0 < abs(iy) < 6:
                                        iy = 6 if iy > 0 else -6
                                    try:
                                        kmbox_net.move_auto(ix, iy, 22)
                                        cooldown = cfg["cooldown_frames"]
                                    except Exception:
                                        try:
                                            kmbox_net.move(ix, iy)
                                            cooldown = cfg["cooldown_frames"]
                                        except Exception:
                                            pass
            else:
                tracker.mark_lost()

            if cooldown > 0:
                cooldown -= 1

            # FPS berechnen
            now = time.monotonic()
            frame_times.append(now)
            if len(frame_times) > 1:
                elapsed = frame_times[-1] - frame_times[0]
                fps = (len(frame_times) - 1) / elapsed if elapsed > 0 else 0

            # ===== DISPLAY (jeden 2. Frame — spart ~5-8ms) =====
            render_frame = (fc % 2 == 0)

            if render_frame and cfg["show_overlay"]:
                if cfg["use_roi_crop"]:
                    half = cfg["roi_size"] // 2
                    cv2.rectangle(frame, (hcx-half, hcy-half), (hcx+half, hcy+half), (50, 50, 50), 1)

                col = (0, 200, 100) if active else (80, 80, 80)
                cv2.circle(frame, (hcx, hcy), cfg["fov_radius"], col, 1)
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
                    # Close-Fight: Dickere Linie + groesserer Punkt = sichtbarer Sticky
                    if tracker.target_h > 120:
                        cv2.line(frame, (hcx, hcy), (ax, ay), (0, 255, 255), 2)
                        cv2.circle(frame, (ax, ay), 8, (0, 255, 255), -1)
                    elif tracker.target_h > 80:
                        cv2.line(frame, (hcx, hcy), (ax, ay), (0, 220, 180), 2)
                        cv2.circle(frame, (ax, ay), 6, (0, 220, 180), -1)
                    else:
                        cv2.line(frame, (hcx, hcy), (ax, ay), (0, 200, 100), 1)
                        cv2.circle(frame, (ax, ay), 5, (0, 200, 100), -1)

                mode = "ON" if active else "OFF"
                nano = "NANO" if cfg["use_nano"] else "FULL"
                roi_t = "ROI" if cfg["use_roi_crop"] else "ALL"
                tm_t = f"TM:{tm_count}" if cfg["minimap_enabled"] else ""
                en_t = f"EN:{en_count}" if cfg["minimap_enabled"] and en_count > 0 else ""
                prot = "SCHUTZ" if cfg["teammate_protection"] else ""
                # Sticky-Anzeige
                sticky_info = ""
                if tracker.locked and tracker.target_h > 80:
                    sticky_info = f"STICKY({tracker.frames}f)"

                cv2.putText(frame, f"FPS:{fps:.0f} AI:{inf_ms:.0f}ms {nano} {roi_t} | {mode} {tm_t} {en_t} {prot} {sticky_info}",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 100), 2)
                cv2.putText(frame, f"STR:{strength:.1f} DZ:{cfg['deadzone']} CD:{cfg['cooldown_frames']} FOV:{cfg['fov_radius']}",
                            (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

                if cfg["minimap_enabled"]:
                    mmx, mmy, mms = cfg["minimap_x"], cfg["minimap_y"], cfg["minimap_size"]
                    cv2.rectangle(frame, (mmx, mmy), (mmx+mms, mmy+mms), (255, 200, 0), 1)

                disp = cv2.resize(frame, (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)
                cv2.imshow("AIMBOT v14", disp)

            elif render_frame:
                # Overlay AUS: Minimales Status-Fenster (pre-alloc, kein neues Array)
                tiny[:] = 0
                nano = "NANO" if cfg["use_nano"] else "FULL"
                roi_t = "ROI" if cfg["use_roi_crop"] else "ALL"
                cv2.putText(tiny, f"FPS:{fps:.0f} AI:{inf_ms:.0f}ms {nano} {roi_t}",
                            (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 100), 1)
                cv2.putText(tiny, f"STR:{strength:.1f} | V=Overlay | ESC=Quit",
                            (5, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
                cv2.imshow("AIMBOT v14", tiny)

            # Minimap Debug (nur wenn aktiv UND Render-Frame)
            if render_frame and show_minimap_debug and cfg["minimap_enabled"]:
                debug_img = minimap.get_debug_image(frame)
                if debug_img.size > 0:
                    cv2.imshow("Minimap Debug", debug_img)

            # Tastatur (muss JEDEN Frame pruefen!)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            elif key == ord('a'):
                active = not active
                print(f"Assist: {'EIN' if active else 'AUS'}")
                if not active:
                    tracker.reset()
            elif key == ord('v'):
                cfg["show_overlay"] = not cfg["show_overlay"]
                print(f"Overlay: {'EIN' if cfg['show_overlay'] else 'AUS (Max FPS!)'}")
            elif key == ord('c'):
                cfg["use_roi_crop"] = not cfg["use_roi_crop"]
                print(f"ROI: {'EIN' if cfg['use_roi_crop'] else 'AUS'}")
            elif key == ord('b'):
                show_minimap_debug = not show_minimap_debug
                print(f"Minimap Debug: {'EIN' if show_minimap_debug else 'AUS'}")
            elif key == ord('f'):
                cfg["teammate_protection"] = not cfg["teammate_protection"]
                print(f"Teammate-Schutz: {'EIN' if cfg['teammate_protection'] else 'AUS'}")
            elif key == ord('n'):
                cfg["use_nano"] = not cfg["use_nano"]
                np2 = get_model_path(cfg["model_mode"], cfg["use_nano"])
                if np2:
                    print(f"Lade: {'NANO' if cfg['use_nano'] else 'FULL'} ({os.path.basename(np2)})...")
                    new_det = YOLODetector(np2)
                    inferencer.swap_detector(new_det)
                    tracker.reset()
                else:
                    cfg["use_nano"] = not cfg["use_nano"]
                    print("Modell nicht gefunden!")
            elif key == ord('1'):
                strength = max(0.1, round(strength - 0.1, 1))
                cfg["strength"] = strength
                print(f"Strength: {strength}")
            elif key == ord('2'):
                strength = min(5.0, round(strength + 0.1, 1))
                cfg["strength"] = strength
                print(f"Strength: {strength}")
            elif key == ord('3'):
                cfg["fov_radius"] = max(50, cfg["fov_radius"] - 25)
                print(f"FOV: {cfg['fov_radius']}px")
            elif key == ord('4'):
                cfg["fov_radius"] = min(400, cfg["fov_radius"] + 25)
                print(f"FOV: {cfg['fov_radius']}px")
            elif key == ord('5'):
                cfg["deadzone"] = max(5, cfg["deadzone"] - 5)
                print(f"Deadzone: {cfg['deadzone']}px")
            elif key == ord('6'):
                cfg["deadzone"] = min(60, cfg["deadzone"] + 5)
                print(f"Deadzone: {cfg['deadzone']}px")
            elif key == ord('7'):
                cfg["confidence"] = max(0.15, round(cfg["confidence"] - 0.05, 2))
                print(f"Confidence: {cfg['confidence']}")
            elif key == ord('8'):
                cfg["confidence"] = min(0.80, round(cfg["confidence"] + 0.05, 2))
                print(f"Confidence: {cfg['confidence']}")
            elif key == ord('9'):
                cfg["cooldown_frames"] = max(0, cfg["cooldown_frames"] - 1)
                print(f"Cooldown: {cfg['cooldown_frames']}")
            elif key == ord('0'):
                cfg["cooldown_frames"] = min(10, cfg["cooldown_frames"] + 1)
                print(f"Cooldown: {cfg['cooldown_frames']}")
            elif key == ord('m'):
                modes = ["fps", "coco"]
                idx = modes.index(cfg["model_mode"]) if cfg["model_mode"] in modes else 0
                cfg["model_mode"] = modes[(idx + 1) % len(modes)]
                np2 = get_model_path(cfg["model_mode"], cfg["use_nano"])
                if np2:
                    print(f"Lade: {cfg['model_mode']}...")
                    new_det = YOLODetector(np2)
                    inferencer.swap_detector(new_det)
                    tracker.reset()
                else:
                    cfg["model_mode"] = modes[idx]

            # Minimap-Position IJKL (nur wenn Debug aktiv)
            elif show_minimap_debug and key == ord('i'):  # Hoch
                cfg["minimap_y"] = max(0, cfg["minimap_y"] - 5)
                minimap.set_position(y=cfg["minimap_y"])
                print(f"Minimap Y: {cfg['minimap_y']}")
            elif show_minimap_debug and key == ord('k'):  # Runter
                cfg["minimap_y"] = min(900, cfg["minimap_y"] + 5)
                minimap.set_position(y=cfg["minimap_y"])
                print(f"Minimap Y: {cfg['minimap_y']}")
            elif show_minimap_debug and key == ord('j'):  # Links
                cfg["minimap_x"] = max(0, cfg["minimap_x"] - 5)
                minimap.set_position(x=cfg["minimap_x"])
                print(f"Minimap X: {cfg['minimap_x']}")
            elif show_minimap_debug and key == ord('l'):  # Rechts
                cfg["minimap_x"] = min(1700, cfg["minimap_x"] + 5)
                minimap.set_position(x=cfg["minimap_x"])
                print(f"Minimap X: {cfg['minimap_x']}")
            elif show_minimap_debug and key == ord('+'):
                cfg["minimap_size"] = min(400, cfg["minimap_size"] + 10)
                minimap.set_position(size=cfg["minimap_size"])
                print(f"Minimap Size: {cfg['minimap_size']}")
            elif show_minimap_debug and key == ord('-'):
                cfg["minimap_size"] = max(80, cfg["minimap_size"] - 10)
                minimap.set_position(size=cfg["minimap_size"])
                print(f"Minimap Size: {cfg['minimap_size']}")

            # Titan Two Macros (nur wenn Titan Two aktiv)
            elif key == ord('q') and input_mode == "titan" and titan:
                titan.set_macro(MACRO_DROPSHOT)
                print("MACRO: Dropshot!")
            elif key == ord('e') and input_mode == "titan" and titan:
                titan.set_macro(MACRO_SNAKING)
                print("MACRO: Snaking!")
            elif key == ord('r') and input_mode == "titan" and titan:
                titan.set_macro(MACRO_SLIDE_CANCEL)
                print("MACRO: Slide-Cancel!")
            elif key == ord('t') and input_mode == "titan" and titan:
                titan.set_macro(MACRO_AUTO_FIRE)
                print("MACRO: Auto-Fire!")
            elif key == ord('g'):
                cfg["anti_recoil_enabled"] = not cfg["anti_recoil_enabled"]
                print(f"Anti-Recoil: {'EIN' if cfg['anti_recoil_enabled'] else 'AUS'}"
                      f" (Profil: {cfg['recoil_profile']})")
            elif key == ord('h'):
                # Recoil-Profil durchschalten
                profiles = list(RECOIL_PROFILES.keys()) if TITAN_MODULE_AVAILABLE else ["default"]
                idx = profiles.index(cfg["recoil_profile"]) if cfg["recoil_profile"] in profiles else 0
                cfg["recoil_profile"] = profiles[(idx + 1) % len(profiles)]
                rval = get_recoil_profile(cfg["recoil_profile"]) if TITAN_MODULE_AVAILABLE else 0
                print(f"Recoil-Profil: {cfg['recoil_profile']} (Y={rval})")

    except KeyboardInterrupt:
        print("\nStop.")
    finally:
        # Config speichern bei Beenden
        save_config(cfg)
        print(f"  Config gespeichert: {CONFIG_FILE}")

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
