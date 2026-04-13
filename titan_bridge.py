"""
TITAN TWO BRIDGE — Laeuft in Gtuner IV Python Console
=======================================================
Empfaengt Aim-Daten vom Aimbot (UDP localhost:5555)
und leitet sie per GCV an den Titan Two weiter.

STARTEN: In Gtuner IV → Tools → Python Scripting → Dieses Script oeffnen → Run

Format vom Aimbot:
  4 Bytes: stick_rx (float, little-endian)
  4 Bytes: stick_ry (float, little-endian)
  4 Bytes: macro_cmd (int32, little-endian)
"""

import socket
import struct
import gtuner

UDP_PORT = 5555
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("127.0.0.1", UDP_PORT))
sock.settimeout(0.002)  # 2ms timeout — kein Blockieren

print(f"Titan Bridge: Warte auf UDP Port {UDP_PORT}...")
print("Starte jetzt den Aimbot: python aimbot_direct.py --input titan")

stick_rx = 0.0
stick_ry = 0.0
macro_cmd = 0

def process(gcvdata):
    """Wird von Gtuner IV jeden Frame aufgerufen (~1000Hz)."""
    global stick_rx, stick_ry, macro_cmd

    # Alle wartenden UDP Pakete lesen (nur neuestes zaehlt)
    while True:
        try:
            data, addr = sock.recvfrom(12)
            if len(data) == 12:
                stick_rx, stick_ry, macro_cmd = struct.unpack("<ffI", data)
        except socket.timeout:
            break
        except Exception:
            break

    # GCV Daten an Titan Two senden
    gcvdata.extend(
        int(float(stick_rx) * 0x10000).to_bytes(4, byteorder="big", signed=True)
    )
    gcvdata.extend(
        int(float(stick_ry) * 0x10000).to_bytes(4, byteorder="big", signed=True)
    )
    gcvdata.extend(
        int(macro_cmd).to_bytes(4, byteorder="big", signed=True)
    )
    gcvdata.extend(
        int(0).to_bytes(4, byteorder="big", signed=True)
    )

    # Stick-Werte zerfallen lassen (kein Input = nichts bewegen)
    stick_rx *= 0.3
    stick_ry *= 0.3
    if abs(stick_rx) < 0.5:
        stick_rx = 0.0
    if abs(stick_ry) < 0.5:
        stick_ry = 0.0
    macro_cmd = 0

    return gcvdata
