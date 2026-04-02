"""
AIMBOT DIRECT v4 - Scuf Passthrough + FPS-KI + Multi-ADS-Trigger
Scuf Envision Pro -> PC -> KMBox Net -> XIM Matrix -> Xbox
Capture Card -> YOLO (FPS) -> Aim-Assist auf Stick-Bewegung

Features:
- Scuf Envision Pro Passthrough (komplett ueber PC geroutet)
- FPS-spezifisches YOLO-Modell (erkennt Spieler, Koepfe)
- Multi-ADS-Trigger: Scuf LT, Tastatur, KMBox, Visuell
- Aimbot-Korrektur wird auf Stick-Aim ADDIERT (natuerliches Gefuehl)
- Headshot-Priorisierung + Teammate-Filter
- Screenshot-Sammlung fuer Custom-Modell Training

Starten:     python aimbot_direct.py
Beenden:     Q oder Strg+C
Screenshots: S druecken zum Speichern ein/ausschalten
Modell:      M druecken zum Wechseln (nano <-> standard)
"""
import cv2
import numpy as np
import time
import sys
import os
import ctypes

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from yolo_onnx import YOLODetector, TARGET_CLASSES
import kmbox_net
from scuf_passthrough import ScufPassthrough, print_button_map

# Windows API fuer Tastenerkennung (kein pip install noetig)
try:
    user32 = ctypes.windll.user32
    def is_key_pressed(vk_code):
        """Prueft ob eine Taste gerade gedrueckt ist (Windows)."""
        return (user32.GetAsyncKeyState(vk_code) & 0x8000) != 0
    KEY_DETECTION_AVAILABLE = True
except (AttributeError, OSError):
    def is_key_pressed(vk_code):
        return False
    KEY_DETECTION_AVAILABLE = False


# ============================================================
# SCUF / XINPUT CONTROLLER AUSLESEN (Windows, kein pip noetig)
# ============================================================
XINPUT_AVAILABLE = False
try:
    xinput_dll = ctypes.windll.xinput1_4
    XINPUT_AVAILABLE = True
except (AttributeError, OSError):
    try:
        xinput_dll = ctypes.windll.xinput1_3
        XINPUT_AVAILABLE = True
    except (AttributeError, OSError):
        try:
            xinput_dll = ctypes.windll.xinput9_1_0
            XINPUT_AVAILABLE = True
        except (AttributeError, OSError):
            xinput_dll = None

if XINPUT_AVAILABLE:
    class XINPUT_GAMEPAD(ctypes.Structure):
        _fields_ = [
            ("wButtons", ctypes.c_ushort),
            ("bLeftTrigger", ctypes.c_ubyte),
            ("bRightTrigger", ctypes.c_ubyte),
            ("sThumbLX", ctypes.c_short),
            ("sThumbLY", ctypes.c_short),
            ("sThumbRX", ctypes.c_short),
            ("sThumbRY", ctypes.c_short),
        ]

    class XINPUT_STATE(ctypes.Structure):
        _fields_ = [
            ("dwPacketNumber", ctypes.c_ulong),
            ("Gamepad", XINPUT_GAMEPAD),
        ]

    # Button Konstanten
    XINPUT_GAMEPAD_LB = 0x0100
    XINPUT_GAMEPAD_RB = 0x0200
    XINPUT_GAMEPAD_A  = 0x1000
    XINPUT_GAMEPAD_B  = 0x2000
    XINPUT_GAMEPAD_X  = 0x4000
    XINPUT_GAMEPAD_Y  = 0x8000

    def get_xinput_state(controller_id=0):
        """Liest den XInput Controller Status. Returns None wenn nicht verbunden."""
        state = XINPUT_STATE()
        ret = xinput_dll.XInputGetState(controller_id, ctypes.byref(state))
        if ret == 0:
            return state
        return None

    def is_scuf_ads_pressed(controller_id=0, trigger_threshold=50):
        """Prueft ob LT (Aim Down Sight) am Scuf gedrueckt ist."""
        state = get_xinput_state(controller_id)
        if state is None:
            return False
        # Left Trigger > threshold = ADS aktiv
        return state.Gamepad.bLeftTrigger > trigger_threshold

    def find_scuf_controller():
        """Findet den Scuf Controller (durchsucht alle 4 XInput Slots)."""
        for i in range(4):
            state = get_xinput_state(i)
            if state is not None:
                return i
        return -1

# ============================================================
# EINSTELLUNGEN
# ============================================================
CAPTURE_DEVICE = 0
KMBOX_IP = "192.168.2.188"
KMBOX_PORT = "32778"
KMBOX_UUID = "C14AE466"

# Modell-Auswahl: "bo7" (custom), "nano" (schnell, 320px) oder "standard" (genauer, 640px)
MODEL_MODE = "bo7"

# Erkennung
CONFIDENCE = 0.50           # Nur sichere Erkennungen
MIN_TARGET_SIZE = 400       # Kleine Boxen ignorieren
AIM_POINT_BODY = 0.40       # Zielpunkt am Koerper (0.40 = obere Brust)
PREFER_HEADSHOTS = False    # AUS: Verhindert Springen zwischen Kopf/Koerper
MAX_AIM_RADIUS = 300        # Nur Ziele nah am Fadenkreuz (kleiner = weniger Himmel-Aiming)
MIN_MOUSE_MOVE = 1          # Nur sub-pixel Bewegungen ignorieren

# ============================================================
# KMBOX SENSITIVITY — Steuert wie stark die Maus pro Pixel Fehler bewegt wird
# Taste 5/6 zum live anpassen
KMBOX_SENSITIVITY = 0.45
# ============================================================

# ============================================================
# AIMBOT PROFILE: Taste 1 = Aim-Assist, Taste 2 = Aimbot
# ============================================================
PROFILES = {
    "assist": {
        "name": "AIM-ASSIST",
        "speed": 0.60,
        "smoothing": 0.40,
        "max_move": 127,        # KMBox HID max
        "deadzone": 20,
    },
    "aimbot": {
        "name": "AIMBOT",
        "speed": 0.85,
        "smoothing": 0.25,
        "max_move": 127,
        "deadzone": 12,
    },
}
ACTIVE_PROFILE = "assist"  # Standard: Aim-Assist (sanft, sicherer Start)

# ADS Erkennung / Trigger-Modus
# "scuf"     = Halte LT am Scuf inVision Pro (zuverlaessig, bester Modus)
# "keyboard" = Halte Taste X am PC (zuverlaessig)
# "kmbox"    = Halte rechte Maustaste an KMBox-Maus (zuverlaessig)
# "visual"   = Automatisch per Zoom-Erkennung (unzuverlaessig)
ADS_MODE = "always"         # Standard: Immer an (fuer Tests ohne Scuf)
ADS_KEY = 0x58              # 0x58 = X-Taste (Virtual Key Code)
ADS_ZOOM_THRESHOLD = 12.0
SCUF_CONTROLLER_ID = -1     # -1 = automatisch finden
SCUF_TRIGGER_THRESHOLD = 50 # LT Empfindlichkeit (0-255, niedriger = empfindlicher)

# Screenshot Sammlung
COLLECT_SCREENSHOTS = False
SCREENSHOT_INTERVAL = 0.5
SCREENSHOT_FOLDER = "training_data"

# Anzeige
SHOW_WINDOW = True
WINDOW_SCALE = 0.5
# ============================================================


def get_model_path(mode):
    """Gibt den Modell-Pfad zurueck."""
    base = os.path.join(os.path.dirname(__file__), 'backend')
    if mode == "bo7":
        path = os.path.join(base, 'bo7_custom_320.onnx')
        if os.path.exists(path):
            return path
    if mode == "nano" or mode == "bo7":
        path = os.path.join(base, 'sunxds_nano_320.onnx')
        if os.path.exists(path):
            return path
    elif mode == "standard":
        path = os.path.join(base, 'sunxds_640.onnx')
        if os.path.exists(path):
            return path
    # Fallback: altes COCO-Modell
    fallback = os.path.join(base, 'yolov8n.onnx')
    if os.path.exists(fallback):
        print(f"  WARNUNG: FPS-Modell nicht gefunden, nutze COCO-Fallback")
        return fallback
    return None


class ADSDetector:
    """Erkennt ob der Spieler gerade ADS (Aim Down Sight) benutzt."""
    def __init__(self):
        self.prev_gray_center = None
        self.ads_active = False
        self.ads_confidence = 0.0
        self.frame_count = 0
        self.baseline_sharpness = None
        self.sharpness_history = []

    def update(self, frame):
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        region_size = 200
        center = frame[cy-region_size:cy+region_size, cx-region_size:cx+region_size]

        if center.size == 0:
            return self.ads_active

        gray = cv2.cvtColor(center, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()

        self.sharpness_history.append(sharpness)
        if len(self.sharpness_history) > 30:
            self.sharpness_history.pop(0)

        if len(self.sharpness_history) >= 10 and self.baseline_sharpness is None:
            self.baseline_sharpness = np.median(self.sharpness_history)

        if self.baseline_sharpness is not None and self.baseline_sharpness > 0:
            ratio = sharpness / self.baseline_sharpness
            self.ads_confidence = ratio
            if ratio > 1.3:
                self.ads_active = True
            elif ratio < 1.1:
                self.ads_active = False

        if self.prev_gray_center is not None and self.prev_gray_center.shape == gray.shape:
            diff = cv2.absdiff(gray, self.prev_gray_center)
            mean_diff = np.mean(diff)
            if mean_diff > ADS_ZOOM_THRESHOLD:
                self.ads_active = True
                self.baseline_sharpness = None
                self.sharpness_history.clear()

        self.prev_gray_center = gray.copy()
        self.frame_count += 1
        return self.ads_active


class TargetTracker:
    """Verbessertes Target Tracking mit Prediction und Anti-Jitter."""
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
        self.last_raw_x = 0.0
        self.last_raw_y = 0.0

    def update(self, rx, ry, alpha=0.4):
        now = time.monotonic()
        dt = now - self.last_seen if self.last_seen > 0 else 0.033

        if self.ema_x is None or (now - self.last_seen) > 0.3:
            # Neues Ziel oder zu lange kein Update
            self.ema_x = float(rx)
            self.ema_y = float(ry)
            self.vel_x = 0.0
            self.vel_y = 0.0
            self.frames_seen = 1
            self.locked = False
        else:
            # Target-Sprung erkennen: Wenn neue Position > 150px entfernt, 
            # ist es wahrscheinlich ein anderes Ziel → Reset
            jump = ((rx - self.ema_x)**2 + (ry - self.ema_y)**2) ** 0.5
            if jump > 150:
                self.ema_x = float(rx)
                self.ema_y = float(ry)
                self.vel_x = 0.0
                self.vel_y = 0.0
                self.frames_seen = 1
                self.locked = False
            else:
                old_x, old_y = self.ema_x, self.ema_y
                self.ema_x = alpha * rx + (1 - alpha) * self.ema_x
                self.ema_y = alpha * ry + (1 - alpha) * self.ema_y
                if dt > 0:
                    new_vx = (self.ema_x - old_x) / dt
                    new_vy = (self.ema_y - old_y) / dt
                    self.vel_x = 0.8 * self.vel_x + 0.2 * new_vx
                    self.vel_y = 0.8 * self.vel_y + 0.2 * new_vy
                self.frames_seen += 1

        self.last_raw_x = rx
        self.last_raw_y = ry
        self.last_seen = now

    def get_predicted(self, lookahead=0.02):
        """Gibt die aktuelle Zielposition zurueck. 
        Keine Velocity-Prediction mehr — verursacht Himmel-Snapping."""
        if self.ema_x is None:
            return None
        # Direkt die EMA-Position verwenden, keine Vorhersage
        return (self.ema_x, self.ema_y)

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


def calc_aim_correction(tracker, tx, ty, fw, fh, profile):
    """Simple P-Controller: Fehler in Pixel * Gain = Mausbewegung.
    
    Je weiter das Ziel vom Fadenkreuz, desto staerker die Korrektur.
    Natuerlich proportional — kein kompliziertes Ramping.
    """
    cx = fw / 2.0
    cy = fh / 2.0
    dx = tx - cx
    dy = ty - cy
    dist = (dx*dx + dy*dy) ** 0.5

    dz = profile["deadzone"]
    speed = profile["speed"]
    smooth = profile["smoothing"]
    max_mv = profile["max_move"]

    if dist < dz:
        tracker.prev_mx *= 0.3
        tracker.prev_my *= 0.3
        return 0, 0

    # Einfacher P-Controller: Fehler * Speed * Sensitivity
    mx = dx * KMBOX_SENSITIVITY * speed
    my = dy * KMBOX_SENSITIVITY * speed

    # Smoothing mit vorheriger Bewegung
    mx = smooth * tracker.prev_mx + (1 - smooth) * mx
    my = smooth * tracker.prev_my + (1 - smooth) * my

    # Max Speed begrenzen
    mag = (mx*mx + my*my) ** 0.5
    if mag > max_mv:
        s = max_mv / mag
        mx *= s
        my *= s

    tracker.prev_mx = mx
    tracker.prev_my = my

    return mx, my


class AimAccumulator:
    """Akkumuliert Aim-Korrekturen ueber mehrere Frames.
    Sendet nur alle N Frames eine groessere move_auto Bewegung,
    damit die XIM Matrix die Bewegung nicht als Rauschen filtert.
    """
    def __init__(self, send_every=3, move_duration_ms=50):
        self.send_every = send_every       # Alle N Frames senden
        self.move_duration_ms = move_duration_ms  # Dauer der move_auto Bewegung
        self.acc_x = 0.0                   # Akkumulierte X-Korrektur
        self.acc_y = 0.0                   # Akkumulierte Y-Korrektur
        self.frame_count = 0               # Frame-Zaehler

    def accumulate(self, mx, my):
        """Fuegt eine Frame-Korrektur hinzu."""
        self.acc_x += mx
        self.acc_y += my
        self.frame_count += 1

    def should_send(self):
        """Prueft ob jetzt gesendet werden soll."""
        return self.frame_count >= self.send_every

    def send(self):
        """Sendet die akkumulierte Bewegung per move_auto und resettet.
        Returns True wenn eine Bewegung gesendet wurde.
        """
        ix = int(round(self.acc_x))
        iy = int(round(self.acc_y))
        sent = False

        # Nur senden wenn Bewegung gross genug (XIM Mindest-Schwelle)
        if abs(ix) >= 5 or abs(iy) >= 5:
            kmbox_net.move_auto(ix, iy, ms=self.move_duration_ms)
            sent = True

        # Reset
        self.acc_x = 0.0
        self.acc_y = 0.0
        self.frame_count = 0
        return sent

    def reset(self):
        """Komplett zuruecksetzen (z.B. wenn Ziel verloren)."""
        self.acc_x = 0.0
        self.acc_y = 0.0
        self.frame_count = 0


# Globaler Akkumulator
aim_accumulator = AimAccumulator(send_every=3, move_duration_ms=50)


def move_aim(tracker, tx, ty, fw, fh, profile):
    """Akkumuliert Aim-Korrekturen und sendet alle 3 Frames eine grosse move_auto.
    
    Problem: XIM Matrix ignoriert kleine Mausbewegungen (<10px).
    Loesung: 3 Frames Korrekturen sammeln, dann eine groessere Bewegung senden.
    Bei 60 FPS: ~20 Korrekturen/Sek statt 60 winzige.
    """
    mx, my = calc_aim_correction(tracker, tx, ty, fw, fh, profile)

    # Korrektur akkumulieren
    aim_accumulator.accumulate(mx, my)

    # Alle 3 Frames: Akkumulierte Bewegung senden
    if aim_accumulator.should_send():
        return aim_accumulator.send()

    return False


def is_teammate(frame, bbox):
    """Prueft ob ein erkannter Spieler ein Teammate ist.
    Teammates in CoD haben blaue/gruene Namenschilder ueber dem Kopf.
    Gegner haben rote oder gar keine.
    """
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    fh, fw = frame.shape[:2]

    # Bereich UEBER der Bounding Box pruefen (Namensschild)
    check_h = max(20, int((y2 - y1) * 0.25))
    check_top = max(0, y1 - check_h)
    check_left = max(0, x1 - 10)
    check_right = min(fw, x2 + 10)

    if check_top >= y1 or check_right <= check_left:
        return False

    region = frame[check_top:y1, check_left:check_right]
    if region.size == 0:
        return False

    # BGR: Blau und Gruen erkennen (Teammate-Farben)
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

    # Blau: H=90-130, S>80, V>80
    blue_mask = cv2.inRange(hsv, (90, 80, 80), (130, 255, 255))
    # Gruen: H=40-80, S>80, V>80
    green_mask = cv2.inRange(hsv, (40, 80, 80), (80, 255, 255))
    # Cyan/Tuerkis: H=80-95, S>80, V>80
    cyan_mask = cv2.inRange(hsv, (80, 80, 80), (95, 255, 255))

    teammate_pixels = cv2.countNonZero(blue_mask) + cv2.countNonZero(green_mask) + cv2.countNonZero(cyan_mask)
    total_pixels = region.shape[0] * region.shape[1]

    if total_pixels == 0:
        return False

    ratio = teammate_pixels / total_pixels
    # Wenn mehr als 5% der Pixel ueber dem Spieler blau/gruen sind = Teammate
    return ratio > 0.05


def pick_best_target(detections, center_x, center_y, prefer_head=False, frame=None):
    """Waehlt das beste Ziel aus den Erkennungen.
    Nutzt NUR body-Erkennung fuer konsistenten Zielpunkt.
    Head-Erkennung wird ignoriert (verhindert Springen).
    """
    bodies = []

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        bw = x2 - x1
        bh = y2 - y1
        if bw * bh < MIN_TARGET_SIZE:
            continue

        # Mittelpunkt der Box
        cx_det = (x1 + x2) / 2.0
        cy_det = (y1 + y2) / 2.0

        # Tote Koerper Filter: Liegende Bboxen ignorieren (Breite > 1.8x Hoehe)
        if bw > bh * 1.8:
            continue

        # Obere 12% vom Bildschirm ignorieren (Himmel/HUD-Bereich)
        frame_height = center_y * 2
        if cy_det < frame_height * 0.12:
            continue

        # Untere 10% ignorieren (HUD/Killfeed)
        if cy_det > frame_height * 0.90:
            continue

        # MAX_AIM_RADIUS: Zu weit vom Fadenkreuz = ignorieren
        dist_from_center = ((cx_det - center_x)**2 + (cy_det - center_y)**2) ** 0.5
        if dist_from_center > MAX_AIM_RADIUS:
            continue

        # Teammate-Check: Blaue/Gruene Markierung = ueberspringen
        if frame is not None and is_teammate(frame, det["bbox"]):
            det["_teammate"] = True
            continue

        class_name = det["class_name"]

        # NUR Body/Player Detections verwenden (konsistenter Zielpunkt)
        if class_name in ("player", "bot", "person"):
            target_y = y1 + int(bh * AIM_POINT_BODY)
            d = ((cx_det - center_x)**2 + (target_y - center_y)**2) ** 0.5
            bodies.append((cx_det, target_y, d, det))
        elif class_name == "head":
            # Head-Box in Body-aehnlichen Zielpunkt umrechnen
            # (Mitte der Head-Box statt oben drueber zu aimen)
            head_cy = (y1 + y2) / 2.0
            # Etwas UNTER die Head-Mitte zielen (realistischer)
            target_y = head_cy + bh * 0.3
            d = ((cx_det - center_x)**2 + (target_y - center_y)**2) ** 0.5
            # Head nur nehmen wenn kein Body da ist (niedriger Prio)
            bodies.append((cx_det, target_y, d + 100, det))  # +100 = niedrigere Prio

    # Naechstes Ziel zum Fadenkreuz nehmen
    if bodies:
        bodies.sort(key=lambda x: x[2])
        best = bodies[0]
        return (best[0], best[1]), best[3]

    return None, None


# Farben fuer verschiedene Klassen
CLASS_COLORS = {
    "player": (0, 0, 255),      # Rot
    "bot": (0, 0, 255),         # Rot
    "head": (0, 165, 255),      # Orange
    "person": (0, 0, 255),      # Rot (COCO fallback)
    "weapon": (128, 128, 128),  # Grau
    "dead_body": (80, 80, 80),  # Dunkelgrau
    "smoke": (200, 200, 200),   # Hellgrau
    "fire": (0, 100, 255),      # Orange-Rot
}


def draw_status_bar(frame, model_mode, collecting, screenshot_count, fps, ads_active, tracker, profile_name):
    """Zeichnet eine gut sichtbare Statusleiste oben im Bild."""
    h, w = frame.shape[:2]
    bar_h = 38

    # Hintergrund: halbtransparenter schwarzer Balken oben
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    x = 10
    y = 27

    # --- MODELL ---
    if model_mode == "bo7":
        tag = "BO7 CUSTOM"
        tag_color = (0, 255, 0)     # Gruen
    elif model_mode == "nano":
        tag = "NANO 320"
        tag_color = (0, 200, 255)   # Gelb-Orange
    else:
        tag = "STANDARD 640"
        tag_color = (255, 180, 0)   # Blau-Cyan

    (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x-4, 6), (x + tw + 8, 33), tag_color, -1)
    cv2.putText(frame, tag, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    x += tw + 16

    # --- PROFIL ---
    if profile_name == "AIMBOT":
        prof_color = (0, 0, 255)    # Rot
    else:
        prof_color = (0, 180, 0)    # Gruen

    (tw_p, _), _ = cv2.getTextSize(profile_name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x-4, 6), (x + tw_p + 8, 33), prof_color, -1)
    cv2.putText(frame, profile_name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    x += tw_p + 16

    # --- SCREENSHOTS ---
    if collecting:
        scr_text = f"REC ({screenshot_count})"
        scr_color = (0, 0, 255)     # Rot = Aufnahme laeuft
        cv2.circle(frame, (x + 6, y - 6), 6, (0, 0, 255), -1)
        x += 18
    else:
        scr_text = "REC: AUS"
        scr_color = (120, 120, 120) # Grau = inaktiv

    cv2.putText(frame, scr_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, scr_color, 2)
    (tw2, _), _ = cv2.getTextSize(scr_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    x += tw2 + 16

    # --- FPS ---
    fps_color = (0, 255, 0) if fps >= 30 else (0, 200, 255) if fps >= 15 else (0, 0, 255)
    cv2.putText(frame, f"FPS: {fps}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, fps_color, 2)

    # --- ADS / AIMBOT Status (rechte Seite) ---
    if ads_active:
        ads_text = "ADS: AN"
        ads_color = (0, 255, 255)
    else:
        ads_text = "ADS: AUS"
        ads_color = (100, 100, 100)

    # Zeige aktuellen ADS-Modus
    mode_short = {"scuf": "[LT]", "keyboard": "[X]", "kmbox": "[MAUS]", "visual": "[AUTO]", "always": "[ON]"}
    ads_text += " " + mode_short.get(ADS_MODE, "")

    (tw_ads, _), _ = cv2.getTextSize(ads_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    ads_x = w - tw_ads - 15
    cv2.putText(frame, ads_text, (ads_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, ads_color, 2)

    if tracker.locked:
        lock_text = "LOCKED"
        (tw_lock, _), _ = cv2.getTextSize(lock_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.putText(frame, lock_text, (ads_x - tw_lock - 15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)


def draw_overlay(frame, all_detections, target_dets, target_pos, fps, ads_active, collecting, tracker, model_mode, screenshot_count, profile_name):
    h, w = frame.shape[:2]

    # === STATUSLEISTE OBEN (gut sichtbar) ===
    draw_status_bar(frame, model_mode, collecting, screenshot_count, fps, ads_active, tracker, profile_name)

    # Fadenkreuz
    color = (0, 255, 255) if ads_active else (0, 255, 0)
    cv2.line(frame, (w//2-25, h//2), (w//2+25, h//2), color, 2)
    cv2.line(frame, (w//2, h//2-25), (w//2, h//2+25), color, 2)
    cv2.circle(frame, (w//2, h//2), PROFILES[ACTIVE_PROFILE]["deadzone"], (50, 50, 50), 1)

    # Alle Erkennungen zeichnen
    for det in all_detections:
        x1, y1, x2, y2 = det["bbox"]
        conf = det["confidence"]
        cls = det["class_name"]
        is_target = cls in TARGET_CLASSES or cls == "person"
        is_tm = det.get("_teammate", False)

        if is_tm:
            # Teammate: blau, durchgestrichen
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 150, 0), 2)
            cv2.putText(frame, 'TEAM', (x1, y1-8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 150, 0), 2)
        elif is_target:
            tgt_color = (0, 0, 255)  # Rot fuer Ziele
            cv2.rectangle(frame, (x1, y1), (x2, y2), tgt_color, 2)
            cv2.putText(frame, f'{cls} {conf:.0%}', (x1, y1-8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, tgt_color, 2)
        else:
            other_color = (100, 100, 100)  # Grau fuer Nicht-Ziele
            cv2.rectangle(frame, (x1, y1), (x2, y2), other_color, 1)
            cv2.putText(frame, f'{cls}', (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, other_color, 1)

    # Ziel-Markierung
    if target_pos:
        tx, ty = int(target_pos[0]), int(target_pos[1])
        cv2.circle(frame, (tx, ty), 12, (0, 255, 255), 3)
        cv2.line(frame, (w//2, h//2), (tx, ty), (0, 255, 255), 2)

    # Ziel-Anzahl links unter Statusleiste
    cv2.putText(frame, f'Ziele: {len(target_dets)}', (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Tastenbelegung unten
    cv2.putText(frame, f'1=Assist | 2=Aimbot | 3=ADS | 5/6=Sens({KMBOX_SENSITIVITY:.2f}) | 7=Test | Q=Quit', (10, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

    return frame


def main():
    global ADS_MODE, SCUF_CONTROLLER_ID, KMBOX_SENSITIVITY
    print("=" * 55)
    print("  XBOX VISION AI - AIMBOT v4")
    print("  Scuf Passthrough + FPS-KI + ADS-Trigger")
    print("=" * 55)
    print()

    screenshot_dir = os.path.join(os.path.dirname(__file__), SCREENSHOT_FOLDER)
    os.makedirs(screenshot_dir, exist_ok=True)

    # 1. YOLO laden
    current_mode = MODEL_MODE
    onnx_path = get_model_path(current_mode)
    if not onnx_path:
        print("FEHLER: Kein YOLO-Modell gefunden!")
        print("Erwartete Dateien:")
        print("  backend/sunxds_nano_320.onnx (schnell)")
        print("  backend/sunxds_640.onnx (genau)")
        sys.exit(1)

    print(f"[1/3] YOLO laden ({current_mode})...")
    detector = YOLODetector(onnx_path, conf_threshold=CONFIDENCE)
    model_info = f"{current_mode} {'FPS' if detector.is_fps_model else 'COCO'}"
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
    print("BEREIT!")
    if detector.is_fps_model:
        print("FPS-Modell geladen: Erkennt Spieler + Koepfe")
        print("Ignoriert: Waffen, Tote, Rauch, Feuer")
    else:
        print("COCO-Modell geladen (Fallback)")
    print()
    print("1 = Aim-Assist (sanft)")
    print("2 = Aimbot (aggressiv)")
    print("3 = ADS-Trigger umschalten:")
    print("    [scuf]     Halte LT am Scuf inVision Pro")
    print("    [keyboard] Halte X-Taste am PC")
    print("    [kmbox]    Halte rechte Maustaste (KMBox)")
    print("    [visual]   Automatisch (Zoom-Erkennung)")
    print("    [always]   Immer an")
    print("5/6 = KMBox Sensitivity runter/rauf (WICHTIG!)")
    print("7 = Kalibrierungs-Test (sendet Test-Bewegung)")
    print("S = Screenshot-Sammlung an/aus")
    print("M = Modell wechseln (nano/standard)")
    print("Q = Beenden")
    print(f"ADS-Trigger: {ADS_MODE.upper()}")
    print(f"KMBox Sensitivity: {KMBOX_SENSITIVITY:.2f}")
    print("=" * 55)

    tracker = TargetTracker()
    ads_detector = ADSDetector()
    collecting = COLLECT_SCREENSHOTS
    last_screenshot = 0
    screenshot_count = len([f for f in os.listdir(screenshot_dir) if f.endswith('.jpg')])
    active_profile_key = ACTIVE_PROFILE
    profile = PROFILES[active_profile_key]
    print(f"Aktives Profil: {profile['name']}")
    print(f"Aim-Batching: Alle {aim_accumulator.send_every} Frames, {aim_accumulator.move_duration_ms}ms Dauer")

    # Scuf Controller suchen
    global SCUF_CONTROLLER_ID
    scuf = None
    if XINPUT_AVAILABLE:
        scuf = ScufPassthrough(kmbox_net)
        scuf_id = scuf.find_controller()
        if scuf_id >= 0:
            SCUF_CONTROLLER_ID = scuf_id
            print(f"Scuf Controller gefunden: Slot {scuf_id}")
            print_button_map()
        else:
            print("Kein XInput Controller gefunden")
            scuf = None
            if ADS_MODE == "scuf":
                ADS_MODE = "keyboard"
                print("  -> Fallback: Tastatur-Modus (halte X)")
    else:
        print("XInput nicht verfuegbar (nur Windows)")
        if ADS_MODE == "scuf":
            ADS_MODE = "keyboard"

    fps = 0
    frame_count = 0
    fps_time = time.monotonic()

    while True:
        ret_cap, frame = cap.read()
        if not ret_cap:
            continue

        fh, fw = frame.shape[:2]

        # YOLO Erkennung - ALLE Klassen (fuer Anzeige)
        all_dets = detector.detect(frame, conf_threshold=CONFIDENCE)

        # Nur Ziel-Klassen fuer Aimbot
        if detector.is_fps_model:
            target_dets = [d for d in all_dets if d["class_name"] in TARGET_CLASSES]
        else:
            target_dets = [d for d in all_dets if d["class_name"] == "person"]

        center_x = fw / 2.0
        center_y = fh / 2.0

        # Bestes Ziel waehlen (mit Headshot-Priorisierung + Teammate-Filter)
        best_target, target_det = pick_best_target(
            target_dets, center_x, center_y, prefer_head=PREFER_HEADSHOTS, frame=frame
        )

        # ADS erkennen (je nach Modus)
        ads_active = False
        aimbot_dx, aimbot_dy = 0, 0

        if ADS_MODE == "scuf" and scuf and scuf.is_available():
            # Scuf-Modus: Pruefen ob noch verbunden
            state = scuf.read_state()
            if state is None:
                print("Scuf getrennt! Wechsle zu IMMER AN")
                ADS_MODE = "always"
                ads_active = True
            else:
                # Aimbot-Korrektur berechnen WENN Ziel vorhanden
                if best_target:
                    tracker.update(best_target[0], best_target[1], alpha=profile["ema_alpha"])
                    if tracker.stable(profile["lock_frames"]):
                        pos = tracker.get_predicted(lookahead=profile["lookahead"])
                        if pos:
                            aimbot_dx, aimbot_dy = calc_aim_correction(tracker, pos[0], pos[1], fw, fh, profile)
                            tracker.locked = True

                # Passthrough sendet Stick+Buttons+Trigger UND addiert Aimbot-Korrektur
                ads_active = scuf.update(
                    aimbot_override_x=aimbot_dx if best_target else 0,
                    aimbot_override_y=aimbot_dy if best_target else 0
                )

                # Wenn kein Ziel, Tracker zuruecksetzen
                if not best_target:
                    if tracker.frames_seen > 0:
                        tracker.frames_seen = max(0, tracker.frames_seen - 1)
                        if tracker.frames_seen == 0:
                            tracker.reset()

        else:
            # Nicht-Scuf Modi: Original-Logik
            if ADS_MODE == "keyboard":
                ads_active = is_key_pressed(ADS_KEY)
            elif ADS_MODE == "kmbox":
                ads_active = kmbox_net.is_mouse_right_pressed()
            elif ADS_MODE == "visual":
                ads_active = ads_detector.update(frame)
            else:
                ads_active = True

            # Aimbot - NUR wenn ADS aktiv
            if best_target and ads_active:
                # Confidence-Check
                if target_det and target_det.get("confidence", 0) >= 0.50:
                    tracker.update(best_target[0], best_target[1], alpha=0.45)
                    if tracker.stable(1):
                        pos = tracker.get_predicted()
                        if pos:
                            tracker.locked = True
                            move_aim(tracker, pos[0], pos[1], fw, fh, profile)
                else:
                    # Confidence zu niedrig: Nicht aimen, Tracker beibehalten
                    pass
            else:
                if not best_target:
                    # Sofort stoppen wenn kein Ziel — keine Geister-Bewegungen!
                    tracker.reset()
                    aim_accumulator.reset()
                elif not ads_active:
                    tracker.reset()
                    aim_accumulator.reset()

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
            display = draw_overlay(
                frame.copy(), all_dets, target_dets, best_target,
                fps, ads_active, collecting, tracker, current_mode, screenshot_count,
                profile["name"]
            )
            if WINDOW_SCALE != 1.0:
                display = cv2.resize(display, None, fx=WINDOW_SCALE, fy=WINDOW_SCALE)
            cv2.imshow('AIMBOT v3 - FPS KI', display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('s'):
                collecting = not collecting
                status = "AN" if collecting else "AUS"
                print(f"Screenshot-Sammlung: {status} ({screenshot_count} gespeichert)")
            elif key == ord('1'):
                active_profile_key = "assist"
                profile = PROFILES[active_profile_key]
                tracker.reset()
                print(f"Profil: {profile['name']} (sanft)")
            elif key == ord('2'):
                active_profile_key = "aimbot"
                profile = PROFILES[active_profile_key]
                tracker.reset()
                print(f"Profil: {profile['name']} (aggressiv)")
            elif key == ord('m'):
                # Modell wechseln: bo7 -> nano -> standard -> bo7
                cycle = {"bo7": "nano", "nano": "standard", "standard": "bo7"}
                new_mode = cycle.get(current_mode, "bo7")
                new_path = get_model_path(new_mode)
                if new_path:
                    print(f"Lade Modell: {new_mode}...")
                    detector = YOLODetector(new_path, conf_threshold=CONFIDENCE)
                    current_mode = new_mode
                    tracker.reset()
                    print(f"Modell gewechselt: {current_mode}")
                else:
                    # Skip missing model
                    new_mode2 = cycle.get(new_mode, "bo7")
                    new_path2 = get_model_path(new_mode2)
                    if new_path2:
                        detector = YOLODetector(new_path2, conf_threshold=CONFIDENCE)
                        current_mode = new_mode2
                        tracker.reset()
                        print(f"Modell '{new_mode}' nicht gefunden, nutze: {current_mode}")
                    else:
                        print(f"Kein alternatives Modell gefunden!")
            elif key == ord('3'):
                # ADS-Trigger-Modus umschalten
                ads_cycle = {"scuf": "keyboard", "keyboard": "kmbox", "kmbox": "visual", "visual": "always", "always": "scuf"}
                ADS_MODE = ads_cycle.get(ADS_MODE, "scuf")
                tracker.reset()
                mode_names = {
                    "scuf": "SCUF CONTROLLER (halte LT)",
                    "keyboard": f"TASTATUR (halte X-Taste)",
                    "kmbox": "KMBOX (rechte Maustaste)",
                    "visual": "VISUELL (automatisch)",
                    "always": "IMMER AN",
                }
                print(f"ADS-Trigger: {mode_names.get(ADS_MODE, ADS_MODE)}")
            elif key == ord('5'):
                # KMBOX Multiplier runter
                KMBOX_SENSITIVITY = max(0.02, KMBOX_SENSITIVITY - 0.02)
                print(f"KMBOX Sensitivity: {KMBOX_SENSITIVITY:.2f}")
            elif key == ord('6'):
                # KMBOX Sensitivity rauf
                KMBOX_SENSITIVITY = min(1.0, KMBOX_SENSITIVITY + 0.02)
                print(f"KMBOX Sensitivity: {KMBOX_SENSITIVITY:.2f}")
            elif key == ord('7'):
                # Kalibrierungs-Test
                test_val = int(100 * KMBOX_SENSITIVITY * 10)
                if test_val < 5:
                    test_val = 5
                print(f"=== KALIBRIERUNGS-TEST (Sens: {KMBOX_SENSITIVITY:.2f}, Wert: {test_val}px) ===")
                print("  RECHTS...")
                kmbox_net.move_auto(test_val, 0, ms=200)
                time.sleep(0.8)
                print("  LINKS...")
                kmbox_net.move_auto(-test_val, 0, ms=200)
                time.sleep(0.8)
                print("  UNTEN...")
                kmbox_net.move_auto(0, test_val, ms=200)
                time.sleep(0.8)
                print("  OBEN...")
                kmbox_net.move_auto(0, -test_val, ms=200)
                print(f"=== 5=weniger 6=mehr, dann 7 nochmal ===")

    cap.release()
    if scuf:
        scuf.stop()
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
