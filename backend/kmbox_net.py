"""
KMBox Net - Pure Python Client (no .pyd needed)
Communicates with KMBox Net hardware via UDP protocol.
Works with any Python version on Windows.
"""
import socket
import struct
import random
import logging

logger = logging.getLogger(__name__)

# KMBox Net Command IDs (from official C header)
CMD_CONNECT     = 0xaf3c2828
CMD_MOUSE_MOVE  = 0xaede7345
CMD_MOUSE_LEFT  = 0x9823AE8D
CMD_MOUSE_MIDDLE = 0x97a3AE8D
CMD_MOUSE_RIGHT = 0x238d8212
CMD_MOUSE_WHEEL = 0xffeead38
CMD_MOUSE_AUTOMOVE = 0xaede7346
CMD_KEYBOARD_ALL = 0x123c2c2f
CMD_REBOOT      = 0xaa8855aa
CMD_BAZER_MOVE  = 0xa238455a
CMD_MONITOR     = 0x27388020


class KMBoxNet:
    """Pure Python KMBox Net controller."""

    def __init__(self):
        self.sock = None
        self.addr = None
        self.mac = 0
        self.indexpts = 0
        self.connected = False

    def init(self, ip, port, uuid):
        """Initialize connection to KMBox Net device.
        ip: Device IP (e.g. '192.168.2.188')
        port: Device port (from display, e.g. '8338')
        uuid: Device UUID (from display, e.g. 'C14AE486')
        """
        self.addr = (ip, int(port))
        self.mac = int(uuid, 16)
        self.indexpts = 0

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(1.0)

        # Send connect command
        packet = self._build_header(CMD_CONNECT)
        # Pad to standard packet size (header + mouse struct = 64 bytes)
        packet += b'\x00' * 48
        try:
            self.sock.sendto(packet, self.addr)
            # Try to receive response
            try:
                data, _ = self.sock.recvfrom(1024)
                self.connected = True
                logger.info(f"KMBox Net connected: {ip}:{port} UUID={uuid}")
                return 0
            except socket.timeout:
                # Some firmware versions don't respond, but connection still works
                self.connected = True
                logger.warning(f"KMBox Net no response but assuming connected: {ip}:{port}")
                return 0
        except Exception as e:
            logger.error(f"KMBox Net connection failed: {e}")
            return -1

    def _build_header(self, cmd):
        """Build 16-byte command header: mac(4) + rand(4) + indexpts(4) + cmd(4)"""
        self.indexpts += 1
        return struct.pack('<IIII',
            self.mac,
            random.randint(0, 0xFFFFFFFF),
            self.indexpts,
            cmd
        )

    def _build_mouse(self, button=0, x=0, y=0, wheel=0):
        """Build 48-byte soft_mouse_t: button(4) + x(4) + y(4) + wheel(4) + point[10](40)"""
        data = struct.pack('<iiii', button, x, y, wheel)
        data += b'\x00' * 40  # point[10] padding
        return data

    def _send_mouse_cmd(self, cmd, button=0, x=0, y=0, wheel=0):
        """Send a mouse command packet."""
        if self.sock is None:
            return -1
        packet = self._build_header(cmd) + self._build_mouse(button, x, y, wheel)
        try:
            self.sock.sendto(packet, self.addr)
            return 0
        except Exception as e:
            logger.error(f"KMBox send error: {e}")
            return -1

    def move(self, x, y):
        """Move mouse relative by (x, y) pixels."""
        return self._send_mouse_cmd(CMD_MOUSE_MOVE, x=int(x), y=int(y))

    def move_auto(self, x, y, ms=100):
        """Move mouse with simulated human movement over ms milliseconds."""
        if self.sock is None:
            return -1
        header = self._build_header(CMD_MOUSE_AUTOMOVE)
        # For automove, rand field = duration in ms
        header = struct.pack('<IIII',
            self.mac,
            ms,
            self.indexpts,
            CMD_MOUSE_AUTOMOVE
        )
        mouse = self._build_mouse(x=int(x), y=int(y))
        try:
            self.sock.sendto(header + mouse, self.addr)
            return 0
        except Exception as e:
            logger.error(f"KMBox auto move error: {e}")
            return -1

    def left(self, state):
        """Left mouse button: 1=press, 0=release."""
        return self._send_mouse_cmd(CMD_MOUSE_LEFT, button=1 if state else 0)

    def right(self, state):
        """Right mouse button: 1=press, 0=release."""
        return self._send_mouse_cmd(CMD_MOUSE_RIGHT, button=1 if state else 0)

    def middle(self, state):
        """Middle mouse button: 1=press, 0=release."""
        return self._send_mouse_cmd(CMD_MOUSE_MIDDLE, button=1 if state else 0)

    def wheel(self, delta):
        """Mouse wheel: positive=down, negative=up."""
        return self._send_mouse_cmd(CMD_MOUSE_WHEEL, wheel=int(delta))

    def mouse_all(self, button, x, y, wheel):
        """Combined mouse command."""
        return self._send_mouse_cmd(CMD_MOUSE_MOVE, button=int(button), x=int(x), y=int(y), wheel=int(wheel))

    def close(self):
        """Close the connection."""
        if self.sock:
            self.sock.close()
            self.sock = None
            self.connected = False


# Global instance
_kmbox = KMBoxNet()

def init(ip, port, uuid):
    return _kmbox.init(ip, port, uuid)

def move(x, y):
    return _kmbox.move(x, y)

def move_auto(x, y, ms=100):
    return _kmbox.move_auto(x, y, ms)

def left(state):
    return _kmbox.left(state)

def right(state):
    return _kmbox.right(state)

def middle(state):
    return _kmbox.middle(state)

def wheel(delta):
    return _kmbox.wheel(delta)

def mouse_all(button, x, y, wheel):
    return _kmbox.mouse_all(button, x, y, wheel)

def is_connected():
    return _kmbox.connected

def close():
    return _kmbox.close()
