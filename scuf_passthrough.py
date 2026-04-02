"""
SCUF ENVISION PRO → KMBox Net Passthrough
Liest den Scuf Controller per XInput und sendet alles als Maus/Tastatur an KMBox.
XIM Matrix empfaengt Maus/Tastatur und macht daraus Xbox-Controller-Signale.

Scuf → PC (XInput) → Python → KMBox (Maus+Tastatur) → XIM Matrix → Xbox
"""
import ctypes
import math

# ============================================================
# HID Keyboard Scancodes (USB HID Usage Table)
# ============================================================
HID_KEY_A = 0x04
HID_KEY_B = 0x05
HID_KEY_C = 0x06
HID_KEY_D = 0x07
HID_KEY_E = 0x08
HID_KEY_F = 0x09
HID_KEY_G = 0x0A
HID_KEY_H = 0x0B
HID_KEY_I = 0x0C
HID_KEY_J = 0x0D
HID_KEY_K = 0x0E
HID_KEY_L = 0x0F
HID_KEY_M = 0x10
HID_KEY_N = 0x11
HID_KEY_O = 0x12
HID_KEY_P = 0x13
HID_KEY_Q = 0x14
HID_KEY_R = 0x15
HID_KEY_S = 0x16
HID_KEY_T = 0x17
HID_KEY_V = 0x19
HID_KEY_W = 0x1A
HID_KEY_X = 0x1B
HID_KEY_Z = 0x1D
HID_KEY_1 = 0x1E
HID_KEY_2 = 0x1F
HID_KEY_3 = 0x20
HID_KEY_4 = 0x21
HID_KEY_SPACE = 0x2C
HID_KEY_TAB = 0x2B
HID_KEY_ESC = 0x29
HID_KEY_F1 = 0x3A
HID_KEY_F2 = 0x3B
HID_KEY_F3 = 0x3C
HID_KEY_F4 = 0x3D

# Modifier bits
MOD_LSHIFT = 0x02
MOD_LCTRL = 0x01

# ============================================================
# XInput Button Constants
# ============================================================
XINPUT_DPAD_UP    = 0x0001
XINPUT_DPAD_DOWN  = 0x0002
XINPUT_DPAD_LEFT  = 0x0004
XINPUT_DPAD_RIGHT = 0x0008
XINPUT_START      = 0x0010
XINPUT_BACK       = 0x0020
XINPUT_LSTICK     = 0x0040
XINPUT_RSTICK     = 0x0080
XINPUT_LB         = 0x0100
XINPUT_RB         = 0x0200
XINPUT_A          = 0x1000
XINPUT_B          = 0x2000
XINPUT_X          = 0x4000
XINPUT_Y          = 0x8000

# ============================================================
# XIM Matrix Tastenbelegung (Scuf-Button → Tastatur-Key)
# Diese Belegung muss im XIM Matrix identisch konfiguriert sein!
# ============================================================
BUTTON_MAP = {
    # Scuf Button    → HID Key         → XIM mappt zu Xbox
    XINPUT_A:        HID_KEY_SPACE,    # A (Springen)     → Space
    XINPUT_B:        HID_KEY_C,        # B (Ducken)       → C
    XINPUT_X:        HID_KEY_R,        # X (Nachladen)    → R
    XINPUT_Y:        HID_KEY_T,        # Y (Waffe wechs.) → T
    XINPUT_LB:       HID_KEY_Q,        # LB (Taktik)      → Q
    XINPUT_RB:       HID_KEY_E,        # RB (Letal)       → E
    XINPUT_DPAD_UP:  HID_KEY_1,        # D-Up             → 1
    XINPUT_DPAD_DOWN: HID_KEY_2,       # D-Down           → 2
    XINPUT_DPAD_LEFT: HID_KEY_3,       # D-Left           → 3
    XINPUT_DPAD_RIGHT: HID_KEY_4,      # D-Right          → 4
    XINPUT_START:    HID_KEY_ESC,      # Start            → ESC
    XINPUT_BACK:     HID_KEY_TAB,      # Back/Select      → TAB
    XINPUT_LSTICK:   HID_KEY_F1,       # L3 (Sprint)      → F1
    XINPUT_RSTICK:   HID_KEY_F2,       # R3 (Nahkampf)    → F2
}

# Trigger → Maustasten
# LT (Zielen) = Rechte Maustaste
# RT (Schiessen) = Linke Maustaste
LT_THRESHOLD = 50    # 0-255, niedriger = empfindlicher
RT_THRESHOLD = 50

# Stick Einstellungen
RIGHT_STICK_DEADZONE = 8000   # XInput standard: 8689
LEFT_STICK_DEADZONE = 8000    # XInput standard: 7849
RIGHT_STICK_SENSITIVITY = 15  # Maus-Pixel pro maximaler Auslenkung
LEFT_STICK_EXPO = 1.5         # Exponential Curve (1.0=linear, 2.0=stark)


class ScufPassthrough:
    """Liest Scuf Envision Pro und sendet alles an KMBox Net."""

    def __init__(self, kmbox_module, controller_id=0):
        self.km = kmbox_module
        self.controller_id = controller_id
        self.enabled = False
        self.last_buttons = 0
        self.last_lt = 0
        self.last_rt = 0
        self.last_keys_sent = []
        self.last_modifier = 0
        self.ads_active = False
        self.rt_active = False

        # XInput laden
        self.xinput = None
        try:
            self.xinput = ctypes.windll.xinput1_4
        except (AttributeError, OSError):
            try:
                self.xinput = ctypes.windll.xinput1_3
            except (AttributeError, OSError):
                try:
                    self.xinput = ctypes.windll.xinput9_1_0
                except (AttributeError, OSError):
                    pass

        if self.xinput is None:
            print("FEHLER: XInput nicht verfuegbar!")
            return

        # XInput Structures
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

        self._state_type = XINPUT_STATE
        self.enabled = True
        print(f"Scuf Passthrough bereit (Controller {controller_id})")

    def is_available(self):
        return self.xinput is not None and self.enabled

    def find_controller(self):
        """Sucht verbundenen Controller."""
        if not self.xinput:
            return -1
        state = self._state_type()
        for i in range(4):
            if self.xinput.XInputGetState(i, ctypes.byref(state)) == 0:
                self.controller_id = i
                return i
        return -1

    def read_state(self):
        """Liest aktuellen Controller-Status."""
        if not self.xinput:
            return None
        state = self._state_type()
        ret = self.xinput.XInputGetState(self.controller_id, ctypes.byref(state))
        if ret != 0:
            return None
        return state

    def _apply_deadzone(self, value, deadzone):
        """Deadzone anwenden und normalisieren (-1.0 bis 1.0)."""
        if abs(value) < deadzone:
            return 0.0
        sign = 1 if value > 0 else -1
        normalized = (abs(value) - deadzone) / (32767 - deadzone)
        return sign * min(1.0, normalized)

    def _apply_expo(self, value, expo):
        """Exponentielle Kurve anwenden (feinere Kontrolle bei kleinen Bewegungen)."""
        sign = 1 if value >= 0 else -1
        return sign * (abs(value) ** expo)

    def update(self, aimbot_override_x=0, aimbot_override_y=0):
        """Liest Scuf und sendet an KMBox. Gibt ads_active zurueck.
        
        aimbot_override_x/y: Wenn Aimbot aktiv, werden diese Werte 
        ZUM Stick-Input ADDIERT (Aim-Assist).
        """
        state = self.read_state()
        if state is None:
            return False

        gp = state.Gamepad

        # === RECHTER STICK → MAUSBEWEGUNG ===
        rx = self._apply_deadzone(gp.sThumbRX, RIGHT_STICK_DEADZONE)
        ry = self._apply_deadzone(-gp.sThumbRY, RIGHT_STICK_DEADZONE)  # Y invertiert

        # Exponentielle Kurve
        rx = self._apply_expo(rx, LEFT_STICK_EXPO)
        ry = self._apply_expo(ry, LEFT_STICK_EXPO)

        # In Maus-Pixel umrechnen
        mouse_x = int(rx * RIGHT_STICK_SENSITIVITY + aimbot_override_x)
        mouse_y = int(ry * RIGHT_STICK_SENSITIVITY + aimbot_override_y)

        if abs(mouse_x) >= 1 or abs(mouse_y) >= 1:
            self.km.move(mouse_x, mouse_y)

        # === LINKER STICK → WASD ===
        lx = self._apply_deadzone(gp.sThumbLX, LEFT_STICK_DEADZONE)
        ly = self._apply_deadzone(gp.sThumbLY, LEFT_STICK_DEADZONE)

        keys = []
        modifier = 0

        if ly > 0.3:
            keys.append(HID_KEY_W)   # Vorwaerts
        if ly < -0.3:
            keys.append(HID_KEY_S)   # Rueckwaerts
        if lx < -0.3:
            keys.append(HID_KEY_A)   # Links
        if lx > 0.3:
            keys.append(HID_KEY_D)   # Rechts

        # === BUTTONS → TASTATUR ===
        buttons = gp.wButtons
        for btn_mask, hid_key in BUTTON_MAP.items():
            if buttons & btn_mask:
                keys.append(hid_key)

        # L3 gedrueckt = Sprint (Shift modifier)
        if buttons & XINPUT_LSTICK:
            modifier |= MOD_LSHIFT

        # === TRIGGER → MAUSTASTEN ===
        self.ads_active = gp.bLeftTrigger > LT_THRESHOLD
        self.rt_active = gp.bRightTrigger > RT_THRESHOLD

        # LT = Rechte Maustaste (ADS/Zielen)
        if self.ads_active and self.last_lt <= LT_THRESHOLD:
            self.km.right(1)   # Rechte Maustaste druecken
        elif not self.ads_active and self.last_lt > LT_THRESHOLD:
            self.km.right(0)   # Rechte Maustaste loslassen

        # RT = Linke Maustaste (Schiessen)
        if self.rt_active and self.last_rt <= RT_THRESHOLD:
            self.km.left(1)    # Linke Maustaste druecken
        elif not self.rt_active and self.last_rt > RT_THRESHOLD:
            self.km.left(0)    # Linke Maustaste loslassen

        # === TASTATUR SENDEN ===
        if keys != self.last_keys_sent or modifier != self.last_modifier:
            self.km.keyboard(modifier, keys[:6])  # Max 6 gleichzeitige Tasten
            self.last_keys_sent = keys[:]
            self.last_modifier = modifier

        # State speichern
        self.last_buttons = buttons
        self.last_lt = gp.bLeftTrigger
        self.last_rt = gp.bRightTrigger

        return self.ads_active

    def stop(self):
        """Alle Tasten/Maustasten loslassen."""
        self.km.keyboard_release()
        self.km.left(0)
        self.km.right(0)
        self.last_keys_sent = []
        self.last_modifier = 0


# Druckbare Zusammenfassung der Belegung
def print_button_map():
    """Zeigt die aktuelle Tastenbelegung an."""
    btn_names = {
        XINPUT_A: "A", XINPUT_B: "B", XINPUT_X: "X", XINPUT_Y: "Y",
        XINPUT_LB: "LB", XINPUT_RB: "RB",
        XINPUT_DPAD_UP: "D-Up", XINPUT_DPAD_DOWN: "D-Down",
        XINPUT_DPAD_LEFT: "D-Left", XINPUT_DPAD_RIGHT: "D-Right",
        XINPUT_START: "Start", XINPUT_BACK: "Back",
        XINPUT_LSTICK: "L3", XINPUT_RSTICK: "R3",
    }
    key_names = {
        HID_KEY_SPACE: "Space", HID_KEY_C: "C", HID_KEY_R: "R",
        HID_KEY_T: "T", HID_KEY_Q: "Q", HID_KEY_E: "E",
        HID_KEY_1: "1", HID_KEY_2: "2", HID_KEY_3: "3", HID_KEY_4: "4",
        HID_KEY_ESC: "ESC", HID_KEY_TAB: "TAB",
        HID_KEY_F1: "F1", HID_KEY_F2: "F2",
        HID_KEY_W: "W", HID_KEY_A: "A", HID_KEY_S: "S", HID_KEY_D: "D",
    }
    print("\n=== SCUF → XIM TASTENBELEGUNG ===")
    print(f"  Rechter Stick  → Maus (Zielen)")
    print(f"  Linker Stick   → W/A/S/D (Laufen)")
    print(f"  LT (Zielen)    → Rechte Maustaste")
    print(f"  RT (Schiessen) → Linke Maustaste")
    for btn_mask, hid_key in BUTTON_MAP.items():
        bn = btn_names.get(btn_mask, f"0x{btn_mask:04X}")
        kn = key_names.get(hid_key, f"0x{hid_key:02X}")
        print(f"  {bn:14s} → {kn}")
    print("=" * 34)
    print("\nDiese Belegung muss im XIM Matrix identisch sein!")
    print("XIM Config: Maus→RS, WASD→LS, LMaus→RT, RMaus→LT, etc.\n")
