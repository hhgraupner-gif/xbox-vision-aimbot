"""
TITAN TWO KOMMUNIKATION — Python Modul
========================================
Sendet Stick-Werte und Macro-Befehle an den Titan Two
ueber das Gtuner IV GCV-Protokoll.

WICHTIG: Dieses Modul funktioniert NUR innerhalb der Gtuner IV
Python-Umgebung, wo das 'gtuner' Modul verfuegbar ist.

Stick-Werte: -100.0 bis +100.0
  STICK_1_X = Rechter Stick X (Aim links/rechts)
  STICK_1_Y = Rechter Stick Y (Aim hoch/runter)
  STICK_2_X = Linker Stick X (Bewegung links/rechts)
  STICK_2_Y = Linker Stick Y (Bewegung vor/zurueck)

Button-Werte: 0 (los) oder 100 (gedrueckt)
"""

import struct

# ============================================================
# TITAN TWO I/O INDIZES (Xbox Layout)
# ============================================================

# Buttons
BUTTON_XBOX    = 0   # Xbox Button
BUTTON_VIEW    = 1   # View / Select
BUTTON_MENU    = 2   # Menu / Start
BUTTON_RB      = 3   # Right Bumper
BUTTON_RT      = 4   # Right Trigger (FIRE)
BUTTON_RS      = 5   # Right Stick Click
BUTTON_LB      = 6   # Left Bumper
BUTTON_LT      = 7   # Left Trigger (ADS)
BUTTON_LS      = 8   # Left Stick Click (Sprint)
BUTTON_UP      = 9   # D-Pad Up
BUTTON_DOWN    = 10  # D-Pad Down
BUTTON_LEFT    = 11  # D-Pad Left
BUTTON_RIGHT   = 12  # D-Pad Right
BUTTON_Y       = 13  # Y
BUTTON_B       = 14  # B (Crouch/Prone)
BUTTON_A       = 15  # A (Jump)
BUTTON_X       = 16  # X (Reload/Use)

# Sticks
STICK_RX       = 17  # Right Stick X (Aim L/R) — STICK_1_X in GPC
STICK_RY       = 18  # Right Stick Y (Aim U/D) — STICK_1_Y in GPC
STICK_LX       = 19  # Left Stick X (Move L/R) — STICK_2_X in GPC
STICK_LY       = 20  # Left Stick Y (Move F/B) — STICK_2_Y in GPC

# GCV Data Offsets (Byte-Positionen fuer gcv_read)
GCV_STICK_RX   = 0   # Bytes 0-3:  Right Stick X
GCV_STICK_RY   = 4   # Bytes 4-7:  Right Stick Y
GCV_MACRO_CMD  = 8   # Bytes 8-11: Macro Command
GCV_EXTRA      = 12  # Bytes 12-15: Extra Data

# Macro Commands (werden an GPC gesendet)
MACRO_NONE         = 0
MACRO_DROPSHOT     = 1
MACRO_SNAKING      = 2
MACRO_SLIDE_CANCEL = 3
MACRO_BUNNY_HOP    = 4
MACRO_AUTO_FIRE    = 5
MACRO_YY_SWAP      = 6


class TitanTwo:
    """Interface fuer Titan Two via Gtuner IV GCV-Protokoll."""

    def __init__(self):
        self._gtuner = None
        self._connected = False
        self._stick_rx = 0.0
        self._stick_ry = 0.0
        self._macro_cmd = MACRO_NONE
        self._sensitivity = 1.0
        self._anti_recoil_y = 0.0

        try:
            import gtuner
            self._gtuner = gtuner
            self._connected = True
            print("  Titan Two: Verbunden (Gtuner IV)")
        except ImportError:
            print("  Titan Two: NICHT verfuegbar (Gtuner IV nicht aktiv)")
            print("  → Starte das Script innerhalb von Gtuner IV!")
            self._connected = False

    @property
    def connected(self):
        return self._connected

    def get_button(self, button_id):
        """Liest aktuellen Button-Zustand vom Controller."""
        if not self._connected:
            return 0
        try:
            return self._gtuner.get_actual(button_id)
        except Exception:
            return 0

    def is_ads(self):
        """Ist LT/ADS gedrueckt?"""
        return self.get_button(BUTTON_LT) > 50

    def is_firing(self):
        """Ist RT/Fire gedrueckt?"""
        return self.get_button(BUTTON_RT) > 50

    def set_aim(self, stick_rx, stick_ry):
        """Setzt Aim-Stick-Werte (-100 bis +100)."""
        self._stick_rx = max(-100.0, min(100.0, stick_rx))
        self._stick_ry = max(-100.0, min(100.0, stick_ry))

    def set_macro(self, macro_cmd):
        """Aktiviert ein Macro (MACRO_DROPSHOT, MACRO_SNAKING, etc.)."""
        self._macro_cmd = macro_cmd

    def set_anti_recoil(self, recoil_y):
        """Setzt Anti-Recoil Kompensation (positiv = nach unten ziehen)."""
        self._anti_recoil_y = recoil_y

    def send(self, gcvdata):
        """Packt alle Werte in gcvdata fuer das GPC-Script.

        Wird von process() aufgerufen. gcvdata ist ein bytearray.
        Format:
          Bytes 0-3:   Right Stick X (fix32)
          Bytes 4-7:   Right Stick Y (fix32, inkl. Anti-Recoil)
          Bytes 8-11:  Macro Command (int32)
          Bytes 12-15: Reserved
        """
        # Stick Y inkl. Anti-Recoil
        total_ry = self._stick_ry + self._anti_recoil_y
        total_ry = max(-100.0, min(100.0, total_ry))

        # Fix32 Format: float * 0x10000 → 4 Bytes Big-Endian
        gcvdata.extend(
            int(float(self._stick_rx) * 0x10000).to_bytes(4, byteorder="big", signed=True)
        )
        gcvdata.extend(
            int(float(total_ry) * 0x10000).to_bytes(4, byteorder="big", signed=True)
        )
        gcvdata.extend(
            int(self._macro_cmd).to_bytes(4, byteorder="big", signed=True)
        )
        gcvdata.extend(
            int(0).to_bytes(4, byteorder="big", signed=True)
        )

        # Reset nach Senden
        self._stick_rx = 0.0
        self._stick_ry = 0.0
        self._macro_cmd = MACRO_NONE

        return gcvdata

    def reset(self):
        """Setzt alle Werte zurueck."""
        self._stick_rx = 0.0
        self._stick_ry = 0.0
        self._macro_cmd = MACRO_NONE
        self._anti_recoil_y = 0.0


def pixels_to_stick(dx, dy, screen_w, screen_h, sensitivity=1.0, speed_x=55.0, speed_y=55.0):
    """Konvertiert Pixel-Delta (Aim-Offset) in Stick-Werte (-100 bis +100).

    dx, dy:        Pixel-Abstand vom Fadenkreuz zum Ziel
    screen_w/h:    Bildschirmaufloesung
    sensitivity:   Multiplikator
    speed_x/y:     Basis-Geschwindigkeit

    Returns: (stick_x, stick_y)
    """
    # Normalisieren: Pixel → Anteil des Bildschirms
    norm_x = dx / (screen_w / 2.0)
    norm_y = dy / (screen_h / 2.0)

    # In Stick-Werte umrechnen
    stick_x = norm_x * speed_x * sensitivity
    stick_y = norm_y * speed_y * sensitivity

    # Clampen
    stick_x = max(-100.0, min(100.0, stick_x))
    stick_y = max(-100.0, min(100.0, stick_y))

    return stick_x, stick_y


# ============================================================
# ANTI-RECOIL PROFILE PRO WAFFE (BO7)
# ============================================================
RECOIL_PROFILES = {
    "default":    {"recoil_y": 8.0,  "name": "Standard"},
    "xm4":        {"recoil_y": 10.0, "name": "XM4"},
    "ak74":       {"recoil_y": 12.0, "name": "AK-74"},
    "mp5":        {"recoil_y": 6.0,  "name": "MP5"},
    "kar98":      {"recoil_y": 0.0,  "name": "Kar98k (Sniper)"},
    "smg_low":    {"recoil_y": 5.0,  "name": "SMG (wenig Recoil)"},
    "smg_high":   {"recoil_y": 9.0,  "name": "SMG (viel Recoil)"},
    "ar_low":     {"recoil_y": 7.0,  "name": "AR (wenig Recoil)"},
    "ar_high":    {"recoil_y": 14.0, "name": "AR (viel Recoil)"},
    "lmg":        {"recoil_y": 11.0, "name": "LMG"},
    "pistol":     {"recoil_y": 4.0,  "name": "Pistole"},
}


def get_recoil_profile(name):
    """Gibt Anti-Recoil Y-Wert fuer eine Waffe zurueck."""
    profile = RECOIL_PROFILES.get(name, RECOIL_PROFILES["default"])
    return profile["recoil_y"]
