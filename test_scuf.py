"""
Scuf Envision Pro Test-Script
Zeigt LIVE alle Stick/Button/Trigger-Werte an.
Beenden mit Strg+C
"""
import ctypes
import time
import sys

# XInput laden
xinput = None
for dll_name in ['xinput1_4', 'xinput1_3', 'xinput9_1_0']:
    try:
        xinput = getattr(ctypes.windll, dll_name)
        print(f"XInput geladen: {dll_name}")
        break
    except (AttributeError, OSError):
        continue

if xinput is None:
    print("FEHLER: XInput nicht gefunden!")
    sys.exit(1)

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

# Controller suchen
print("\nSuche Controller...")
found_id = -1
for i in range(4):
    state = XINPUT_STATE()
    ret = xinput.XInputGetState(i, ctypes.byref(state))
    if ret == 0:
        print(f"  Controller gefunden auf Slot {i}!")
        found_id = i
        break
    else:
        print(f"  Slot {i}: leer")

if found_id == -1:
    print("\nKein Controller gefunden! Ist der Scuf per USB angeschlossen?")
    sys.exit(1)

# Button-Namen
BTN_NAMES = {
    0x0001: "D-Up", 0x0002: "D-Down", 0x0004: "D-Left", 0x0008: "D-Right",
    0x0010: "Start", 0x0020: "Back", 0x0040: "L3", 0x0080: "R3",
    0x0100: "LB", 0x0200: "RB", 0x1000: "A", 0x2000: "B",
    0x4000: "X", 0x8000: "Y",
}

print(f"\n=== SCUF LIVE TEST (Slot {found_id}) ===")
print("Druecke Buttons, bewege Sticks, druecke Trigger...")
print("Beenden mit Strg+C\n")

try:
    while True:
        state = XINPUT_STATE()
        ret = xinput.XInputGetState(found_id, ctypes.byref(state))
        if ret != 0:
            print("Controller getrennt!")
            break

        gp = state.Gamepad

        # Buttons
        pressed = []
        for mask, name in BTN_NAMES.items():
            if gp.wButtons & mask:
                pressed.append(name)
        btn_str = ", ".join(pressed) if pressed else "-"

        # Output
        line = (
            f"LStick: {gp.sThumbLX:+6d},{gp.sThumbLY:+6d} | "
            f"RStick: {gp.sThumbRX:+6d},{gp.sThumbRY:+6d} | "
            f"LT: {gp.bLeftTrigger:3d} | "
            f"RT: {gp.bRightTrigger:3d} | "
            f"Buttons: {btn_str}"
        )
        print(f"\r{line:<100}", end="", flush=True)

        time.sleep(0.016)  # ~60fps

except KeyboardInterrupt:
    print("\n\nTest beendet!")
