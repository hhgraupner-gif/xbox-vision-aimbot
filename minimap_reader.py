"""
MINIMAP READER — BO7 Multiplayer
=================================
Liest die Minimap aus dem Capture-Card-Bild und erkennt:
  - Teammates (blau)
  - Freunde (gruen)
  - Feinde (rot, wenn UAV / Schiessen ohne Daempfer)
  - Eigene Position (pink/magenta, immer Mitte)

Nutzt reine OpenCV HSV-Farberkennung, kein AI noetig.
Verbrauch: < 1ms pro Frame.

BO7 Minimap rotiert MIT dem Spieler:
  - Oben = Blickrichtung (Bildschirm-Mitte)
  - Rechts auf Minimap = Rechts auf Bildschirm
"""

import cv2
import numpy as np
import math


# ============================================================
# HSV-Farbbereiche (OpenCV: H=0-180, S=0-255, V=0-255)
# Strenge Filter: Spieler-Icons sind SEHR gesaettigt + hell,
# Karten-Elemente sind blass/dunkel → hohe S/V Schwellen
# ============================================================

# Teammates (blau) — Helles, gesaettigtes Blau
BLUE_LOW = np.array([100, 150, 150])
BLUE_HIGH = np.array([125, 255, 255])

# Freunde (gruen) — Helles, gesaettigtes Gruen
GREEN_LOW = np.array([40, 150, 150])
GREEN_HIGH = np.array([80, 255, 255])

# Feinde (rot) — Helles, gesaettigtes Rot (wrapt um 0/180!)
RED_LOW_A = np.array([0, 150, 150])
RED_HIGH_A = np.array([8, 255, 255])
RED_LOW_B = np.array([172, 150, 150])
RED_HIGH_B = np.array([180, 255, 255])

# Minimum Blob-Flaeche (Pixel) — Spieler-Icons sind groesser als Rauschen
MIN_BLOB_AREA = 30


class MinimapReader:
    """Liest Teammate/Feind-Positionen aus der BO7-Minimap."""

    def __init__(self, x=22, y=42, size=195):
        """
        x, y:  Minimap Top-Left Position auf 1080p Frame
        size:  Minimap Breite/Hoehe in Pixel
        """
        self.x = x
        self.y = y
        self.size = size
        self.teammates = []
        self.enemies = []
        self._kernel = np.ones((3, 3), np.uint8)

    def update(self, frame):
        """Analysiert die Minimap im Frame.

        Returns: (teammates, enemies) — Listen von Dicts:
            {"angle": float, "distance": float, "pixel": (x, y)}
            angle: 0=oben(vorwaerts), 90=rechts, 180=hinten, 270=links
        """
        h, w = frame.shape[:2]
        x2 = min(self.x + self.size, w)
        y2 = min(self.y + self.size, h)
        minimap = frame[self.y:y2, self.x:x2]

        if minimap.size == 0:
            self.teammates = []
            self.enemies = []
            return self.teammates, self.enemies

        hsv = cv2.cvtColor(minimap, cv2.COLOR_BGR2HSV)

        # Teammates: blau + gruen
        blue = cv2.inRange(hsv, BLUE_LOW, BLUE_HIGH)
        green = cv2.inRange(hsv, GREEN_LOW, GREEN_HIGH)
        tm_mask = cv2.bitwise_or(blue, green)

        # Feinde: rot
        r1 = cv2.inRange(hsv, RED_LOW_A, RED_HIGH_A)
        r2 = cv2.inRange(hsv, RED_LOW_B, RED_HIGH_B)
        en_mask = cv2.bitwise_or(r1, r2)

        self.teammates = self._find_blobs(tm_mask)
        self.enemies = self._find_blobs(en_mask)

        return self.teammates, self.enemies

    def _find_blobs(self, mask):
        """Findet Blob-Zentren und berechnet Winkel vom Minimap-Zentrum."""
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cx = self.size // 2
        cy = self.size // 2
        max_dist = self.size * 0.45  # Max 45% vom Zentrum (Spieler nicht am Rand)
        results = []

        for c in contours:
            area = cv2.contourArea(c)
            if area < MIN_BLOB_AREA:
                continue
            # Zu grosse Blobs sind Karten-Elemente, keine Spieler-Icons
            if area > 500:
                continue

            M = cv2.moments(c)
            if M["m00"] == 0:
                continue

            bx = int(M["m10"] / M["m00"])
            by = int(M["m01"] / M["m00"])

            # Abstand zum Zentrum (eigene Position)
            dx = bx - cx
            dy = cy - by  # Y invertiert (oben = vorwaerts)
            dist = math.sqrt(dx * dx + dy * dy)

            # Winkel: 0=oben, 90=rechts, 180=unten, 270=links
            angle = math.degrees(math.atan2(dx, dy)) % 360

            # Zu nah am Zentrum = eigener Spieler, ignorieren
            if dist < 10:
                continue
            # Zu weit weg = wahrscheinlich Karten-Artefakt
            if dist > max_dist:
                continue

            results.append({
                "angle": angle,
                "distance": dist,
                "pixel": (bx, by),
            })

        return results

    def is_teammate_direction(self, target_x, screen_cx, fov_deg=100, tolerance=30):
        """Prueft ob ein Bildschirm-Ziel in Richtung eines Teammates liegt.

        target_x:   X-Position des Ziels auf dem Bildschirm
        screen_cx:  Bildschirm-Mitte X
        fov_deg:    In-Game FOV in Grad
        tolerance:  Winkel-Toleranz in Grad

        Returns: True wenn Ziel vermutlich ein Teammate ist.
        """
        if not self.teammates:
            return False

        # Bildschirm-X → Winkel (Mitte = 0°, rechts = positiv)
        dx = target_x - screen_cx
        half_fov = fov_deg / 2.0
        # Normalisieren: Bildschirmrand = halber FOV
        screen_angle = (dx / screen_cx) * half_fov

        # BO7 Minimap rotiert mit Spieler → Oben = Vorwaerts = 0°
        # Bildschirm-Mitte = Vorwaerts = 0°
        # Also: screen_angle direkt vergleichbar mit minimap angle

        # Minimap-Winkel in -180..180 konvertieren fuer Vergleich
        for tm in self.teammates:
            tm_angle = tm["angle"]
            if tm_angle > 180:
                tm_angle -= 360

            diff = abs(screen_angle - tm_angle)
            if diff > 180:
                diff = 360 - diff

            if diff < tolerance:
                return True

        return False

    def get_debug_image(self, frame):
        """Gibt ein Debug-Bild der Minimap mit Markierungen zurueck."""
        h, w = frame.shape[:2]
        x2 = min(self.x + self.size, w)
        y2 = min(self.y + self.size, h)
        debug = frame[self.y:y2, self.x:x2].copy()

        if debug.size == 0:
            return debug

        cx = self.size // 2
        cy = self.size // 2

        # Zentrum (eigener Spieler)
        cv2.circle(debug, (cx, cy), 4, (255, 0, 255), -1)

        # Teammates (blau)
        for tm in self.teammates:
            px, py = tm["pixel"]
            cv2.circle(debug, (px, py), 6, (255, 200, 0), 2)
            cv2.putText(debug, f"{tm['angle']:.0f}", (px+8, py),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 200, 0), 1)

        # Feinde (rot)
        for en in self.enemies:
            px, py = en["pixel"]
            cv2.circle(debug, (px, py), 6, (0, 0, 255), 2)
            cv2.putText(debug, f"{en['angle']:.0f}", (px+8, py),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        # Auf 2x vergroessern fuer bessere Sichtbarkeit
        debug = cv2.resize(debug, (self.size * 2, self.size * 2),
                           interpolation=cv2.INTER_NEAREST)

        return debug

    def set_position(self, x=None, y=None, size=None):
        """Minimap-Position anpassen."""
        if x is not None:
            self.x = max(0, x)
        if y is not None:
            self.y = max(0, y)
        if size is not None:
            self.size = max(50, min(400, size))

    def to_dict(self):
        """Fuer Config-Speicherung."""
        return {"x": self.x, "y": self.y, "size": self.size}

    def from_dict(self, d):
        """Aus Config laden."""
        self.x = d.get("x", self.x)
        self.y = d.get("y", self.y)
        self.size = d.get("size", self.size)
