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
        """Move mouse relative by (x, y) pixels. 
        Teilt grosse Bewegungen in kleinere Schritte auf (KMBox Firmware Limit).
        """
        MAX_STEP = 100  # Max Pixel pro Einzelbewegung
        x = int(x)
        y = int(y)
        
        # Kleine Bewegungen direkt senden
        if abs(x) <= MAX_STEP and abs(y) <= MAX_STEP:
            return self._send_mouse_cmd(CMD_MOUSE_MOVE, x=x, y=y)
        
        # Grosse Bewegungen in Schritte aufteilen
        steps = max(abs(x) // MAX_STEP, abs(y) // MAX_STEP, 1)
        step_x = x / steps
        step_y = y / steps
        
        for i in range(steps):
            sx = int(round(step_x * (i + 1))) - int(round(step_x * i))
            sy = int(round(step_y * (i + 1))) - int(round(step_y * i))
            self._send_mouse_cmd(CMD_MOUSE_MOVE, x=sx, y=sy)
        
        return 0

    def monitor(self, port=10000):
        """Enable physical mouse/keyboard monitoring on given port."""
        if self.sock is None:
            return -1
        header = struct.pack('<IIII', self.mac, port, self.indexpts, CMD_MONITOR)
        self.indexpts += 1
        mouse = b'\x00' * 48
        try:
            self.sock.sendto(header + mouse, self.addr)
            # Set up monitor receive socket
            if not hasattr(self, '_mon_sock') or self._mon_sock is None:
                self._mon_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self._mon_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self._mon_sock.bind(('0.0.0.0', port))
                self._mon_sock.settimeout(0.001)
            self._mon_port = port
            self._mouse_state = 0
            logger.info(f"KMBox monitor enabled on port {port}")
            return 0
        except Exception as e:
            logger.error(f"KMBox monitor error: {e}")
            return -1

    def poll_monitor(self):
        """Poll for monitor data (non-blocking). Returns mouse button state."""
        if not hasattr(self, '_mon_sock') or self._mon_sock is None:
            return 0
        try:
            while True:
                data, _ = self._mon_sock.recvfrom(256)
                if len(data) >= 20:
                    # Parse: head(16) + button(4) + x(4) + y(4) + ...
                    btn = struct.unpack_from('<i', data, 16)[0]
                    self._mouse_state = btn
        except (socket.timeout, BlockingIOError):
            pass
        except Exception:
            pass
        return getattr(self, '_mouse_state', 0)

    def is_mouse_right_pressed(self):
        """Check if physical right mouse button is pressed."""
        state = self.poll_monitor()
        return (state & 0x02) != 0

    def is_mouse_left_pressed(self):
        """Check if physical left mouse button is pressed."""
        state = self.poll_monitor()
        return (state & 0x01) != 0

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

    def keyboard(self, modifier=0, keys=None):
        """Send keyboard state. 
        modifier: bitmask (1=LCtrl, 2=LShift, 4=LAlt, 8=LGUI, 16=RCtrl, 32=RShift, 64=RAlt, 128=RGUI)
        keys: list of up to 6 HID key codes pressed simultaneously
        """
        if self.sock is None:
            return -1
        header = self._build_header(CMD_KEYBOARD_ALL)
        # Build keyboard HID report: modifier(1) + reserved(1) + key[6]
        key_data = [0] * 8
        key_data[0] = modifier & 0xFF
        key_data[1] = 0  # reserved
        if keys:
            for i, k in enumerate(keys[:6]):
                key_data[2 + i] = k & 0xFF
        # Pack as 8 bytes + pad to 48 bytes (same as mouse payload size)
        payload = bytes(key_data) + b'\x00' * 40
        try:
            self.sock.sendto(header + payload, self.addr)
            return 0
        except Exception as e:
            logger.error(f"KMBox keyboard error: {e}")
            return -1

    def keyboard_release(self):
        """Release all keyboard keys."""
        return self.keyboard(0, [])

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

def monitor(port=10000):
    return _kmbox.monitor(port)

def is_mouse_right_pressed():
    return _kmbox.is_mouse_right_pressed()

def is_mouse_left_pressed():
    return _kmbox.is_mouse_left_pressed()

def is_connected():
    return _kmbox.connected

def close():
    return _kmbox.close()

def keyboard(modifier=0, keys=None):
    return _kmbox.keyboard(modifier, keys)

def keyboard_release():
    return _kmbox.keyboard_release()
