"""
AIMBOT VISION v13 — MINIMAP READER + CONFIG + PERF
====================================================
Neu in v13:
  - Minimap-Reader: Erkennt Teammates (blau/gruen) auf der Minimap
  - Teammate-Schutz: Zielt NICHT auf Spieler in Teammate-Richtung
  - Config: Settings in config.json gespeichert/geladen
  - Minimap Debug: Taste B zeigt erkannte Minimap

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
    "confidence": 0.40,
    "fov_radius": 180,
    "strength": 0.7,
    "deadzone": 15,
    "max_move": 30,
    "cooldown_frames": 4,
    "use_roi_crop": True,
    "roi_size": 640,
    "target_fps": 60,
    "show_overlay": True,
    "use_nano": False,
    "dead_body_ratio": 1.2,
    "sky_filter_ratio": 0.10,
    "ground_filter_ratio": 0.88,
    "min_box_height": 40,
    "minimap_enabled": True,
    "minimap_x": 22,
    "minimap_y": 42,
    "minimap_size": 195,
    "teammate_protection": True,
    "teammate_tolerance": 30,
    "game_fov": 100,
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
# ZIELAUSWAHL MIT TEAMMATE-SCHUTZ
# ============================================================
def pick_best_target(detections, frame_w, frame_h, cfg, minimap=None):
    cx, cy = frame_w / 2.0, frame_h / 2.0
    fov = cfg["fov_radius"]
    dead_ratio = cfg["dead_body_ratio"]
    min_h = cfg["min_box_height"]
    sky = cfg["sky_filter_ratio"]
    ground = cfg["ground_filter_ratio"]
    tm_protect = cfg["teammate_protection"] and minimap is not None

    best = None
    best_dist = float('inf')

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

        if cls == "head":
            tx, ty = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        else:
            tx, ty = (x1 + x2) / 2.0, y1 + bh * 0.35

        dist = math.sqrt((tx - cx) ** 2 + (ty - cy) ** 2)
        if dist > fov:
            continue

        # TEAMMATE-SCHUTZ: Pruefen ob Ziel in Richtung eines Teammates liegt
        if tm_protect and minimap.is_teammate_direction(
            tx, cx, fov_deg=cfg["game_fov"], tolerance=cfg["teammate_tolerance"]
        ):
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
    cfg = load_config()

    print("=" * 60)
    print("  AIMBOT v13 — MINIMAP READER + CONFIG")
    print("  Teammate-Schutz | Auto-Save | Debug-Modus")
    print("=" * 60)

    # KMBox
    if KMBOX_AVAILABLE:
        try:
            kmbox_net.init(cfg["kmbox_ip"], cfg["kmbox_port"], cfg["kmbox_uuid"])
            print(f"  KMBox: OK")
        except Exception as e:
            print(f"  KMBox: {e}")

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
            mm = minimap if cfg["minimap_enabled"] and cfg["teammate_protection"] else None
            tx, ty, tdet = pick_best_target(targets, fw, fh, cfg, minimap=mm)

            aim_pos = None

            if tx is not None:
                tracker.update(tx, ty)
                pos = tracker.get_position()
                if pos:
                    aim_pos = pos

                    if active and tracker.locked:
                        dx = pos[0] - scr_cx
                        dy = pos[1] - scr_cy
                        dist = math.sqrt(dx * dx + dy * dy)

                        if dist > cfg["deadzone"] and cooldown <= 0:
                            mx = dx * strength
                            my = dy * strength
                            mx = max(-cfg["max_move"], min(cfg["max_move"], mx))
                            my = max(-cfg["max_move"], min(cfg["max_move"], my))
                            ix = int(round(mx))
                            iy = int(round(my))
                            if (abs(ix) > 0 or abs(iy) > 0) and KMBOX_AVAILABLE:
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
                    cv2.line(frame, (hcx, hcy), (ax, ay), (0, 200, 100), 1)
                    cv2.circle(frame, (ax, ay), 5, (0, 200, 100), -1)

                mode = "ON" if active else "OFF"
                nano = "NANO" if cfg["use_nano"] else "FULL"
                roi_t = "ROI" if cfg["use_roi_crop"] else "ALL"
                tm_t = f"TM:{tm_count}" if cfg["minimap_enabled"] else ""
                en_t = f"EN:{en_count}" if cfg["minimap_enabled"] and en_count > 0 else ""
                prot = "SCHUTZ" if cfg["teammate_protection"] else ""

                cv2.putText(frame, f"FPS:{fps:.0f} AI:{inf_ms:.0f}ms {nano} {roi_t} | {mode} {tm_t} {en_t} {prot}",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 100), 2)
                cv2.putText(frame, f"STR:{strength:.1f} DZ:{cfg['deadzone']} CD:{cfg['cooldown_frames']} FOV:{cfg['fov_radius']}",
                            (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

                if cfg["minimap_enabled"]:
                    mmx, mmy, mms = cfg["minimap_x"], cfg["minimap_y"], cfg["minimap_size"]
                    cv2.rectangle(frame, (mmx, mmy), (mmx+mms, mmy+mms), (255, 200, 0), 1)

                disp = cv2.resize(frame, (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)
                cv2.imshow("AIMBOT v13", disp)

            elif render_frame:
                # Overlay AUS: Minimales Status-Fenster (pre-alloc, kein neues Array)
                tiny[:] = 0
                nano = "NANO" if cfg["use_nano"] else "FULL"
                roi_t = "ROI" if cfg["use_roi_crop"] else "ALL"
                cv2.putText(tiny, f"FPS:{fps:.0f} AI:{inf_ms:.0f}ms {nano} {roi_t}",
                            (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 100), 1)
                cv2.putText(tiny, f"STR:{strength:.1f} | V=Overlay | ESC=Quit",
                            (5, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
                cv2.imshow("AIMBOT v13", tiny)

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
