"""
COMPETITIVE AUDIO TOOL — Warzone Ranked
=========================================
Echtzeit Audio-Enhancement fuer Footsteps in Call of Duty.

Features:
  - Footstep Frequency Boost (300Hz-2.5kHz)
  - Explosion/Gunfire Suppression (unter 200Hz gedaempft)
  - Dynamic Compression (leise Steps lauter, laute Sounds leiser)
  - Stereo Spatial Enhancement (breiteres Stereo-Bild)
  - Visueller Audio-Radar (zeigt Richtung erkannter Steps)
  - Noise Gate (Stille bleibt still)
  - Mehrere Presets (Warzone, Multiplayer, Resurgence)

Voraussetzungen:
  pip install sounddevice numpy scipy

Steuerung:
  1/2/3   = Preset: Warzone / Multiplayer / Resurgence
  Q/W     = Footstep Boost runter/hoch
  E/R     = Compression runter/hoch
  A/S     = Bass Cut runter/hoch
  D/F     = Spatial Width runter/hoch
  M       = Mute/Unmute
  V       = Radar an/aus
  ESC     = Beenden

Audio-Kette:
  Capture Card Input → Highpass → Footstep Bandboost → Compression → Spatial → Output
"""

import sys
import os
import json
import time
import threading
import argparse
from collections import deque

import numpy as np
from scipy import signal

try:
    import sounddevice as sd
    SD_AVAILABLE = True
except ImportError:
    SD_AVAILABLE = False
    print("FEHLER: 'sounddevice' nicht installiert!")
    print("Installiere mit: pip install sounddevice")
    print()

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# ============================================================
# CONFIG
# ============================================================
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio_config.json")

DEFAULT_CONFIG = {
    "input_device": None,       # None = Default, oder Device-Index
    "output_device": None,      # None = Default, oder Device-Index
    "samplerate": 48000,
    "blocksize": 512,           # ~10ms bei 48kHz
    "channels": 2,              # Stereo
    "preset": "warzone",
    "footstep_boost_db": 10.0,  # dB Boost im Step-Band
    "bass_cut_freq": 180,       # Hz — alles darunter wird gedaempft
    "step_low_freq": 300,       # Hz — untere Grenze Step-Band
    "step_high_freq": 2500,     # Hz — obere Grenze Step-Band
    "compression_ratio": 3.0,   # Kompression (1=aus, 4=stark)
    "compression_threshold": -20.0,  # dB Threshold
    "noise_gate_db": -50.0,     # Unter diesem Pegel = Stille
    "spatial_width": 1.3,       # Stereo-Breite (1.0=normal, 2.0=extrabreit)
    "output_gain_db": 0.0,      # Master-Lautstaerke
    "show_radar": True,
}

PRESETS = {
    "warzone": {
        "footstep_boost_db": 12.0,
        "bass_cut_freq": 200,
        "step_low_freq": 250,
        "step_high_freq": 3000,
        "compression_ratio": 3.5,
        "compression_threshold": -22.0,
        "noise_gate_db": -48.0,
        "spatial_width": 1.4,
    },
    "multiplayer": {
        "footstep_boost_db": 10.0,
        "bass_cut_freq": 150,
        "step_low_freq": 300,
        "step_high_freq": 2500,
        "compression_ratio": 2.5,
        "compression_threshold": -18.0,
        "noise_gate_db": -45.0,
        "spatial_width": 1.2,
    },
    "resurgence": {
        "footstep_boost_db": 14.0,
        "bass_cut_freq": 220,
        "step_low_freq": 200,
        "step_high_freq": 3500,
        "compression_ratio": 4.0,
        "compression_threshold": -25.0,
        "noise_gate_db": -50.0,
        "spatial_width": 1.5,
    },
}


def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                saved = json.load(f)
            cfg = DEFAULT_CONFIG.copy()
            cfg.update(saved)
            return cfg
        except Exception:
            pass
    return DEFAULT_CONFIG.copy()


def save_config(cfg):
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(cfg, f, indent=2)
    except Exception:
        pass


def apply_preset(cfg, preset_name):
    """Wendet ein Preset an."""
    if preset_name in PRESETS:
        cfg.update(PRESETS[preset_name])
        cfg["preset"] = preset_name
        print(f"  Preset: {preset_name.upper()}")


# ============================================================
# AUDIO FILTER DESIGN
# ============================================================
class AudioProcessor:
    """Echtzeit Audio-Verarbeitung fuer Competitive Gaming."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.sr = cfg["samplerate"]
        self.nyq = self.sr / 2.0
        self.muted = False

        # Filter-States (fuer kontinuierliche Verarbeitung ohne Klicks)
        self._rebuild_filters()

        # RMS History fuer Radar
        self.rms_left_hist = deque(maxlen=30)
        self.rms_right_hist = deque(maxlen=30)
        self.rms_step_left = 0.0
        self.rms_step_right = 0.0
        self.peak_direction = 0.0  # -1.0 = links, +1.0 = rechts
        self.step_energy = 0.0     # 0-1 Staerke des erkannten Steps

    def _rebuild_filters(self):
        """Baut alle Filter neu (bei Config-Aenderung)."""
        sr = self.sr
        nyq = self.nyq
        cfg = self.cfg

        # 1. Highpass — Bass Cut (Explosionen, Fahrzeuge daempfen)
        bass_freq = max(20, min(cfg["bass_cut_freq"], nyq * 0.9))
        self.sos_hp = signal.butter(4, bass_freq / nyq, btype='highpass', output='sos')

        # 2. Bandpass — Footstep-Band isolieren (fuer Boost)
        lo = max(20, min(cfg["step_low_freq"], nyq * 0.8))
        hi = max(lo + 50, min(cfg["step_high_freq"], nyq * 0.9))
        self.sos_bp = signal.butter(3, [lo / nyq, hi / nyq], btype='bandpass', output='sos')

        # 3. Notch-Filter fuer typische Stoerfrequenzen (Schuss-Peaks bei ~4-6kHz)
        notch_freq = 5000.0
        if notch_freq < nyq:
            self.sos_notch = signal.iirnotch(notch_freq, Q=5.0, fs=sr)
        else:
            self.sos_notch = None

        # Filter-Zustaende (pro Kanal)
        self.zi_hp_L = np.zeros((self.sos_hp.shape[0], 2))
        self.zi_hp_R = np.zeros((self.sos_hp.shape[0], 2))
        self.zi_bp_L = np.zeros((self.sos_bp.shape[0], 2))
        self.zi_bp_R = np.zeros((self.sos_bp.shape[0], 2))

        # Kompressor-State
        self.comp_env = 0.0  # Envelope follower

        # Gain Werte vorberechnen
        self.step_gain = 10 ** (cfg["footstep_boost_db"] / 20.0)
        self.out_gain = 10 ** (cfg["output_gain_db"] / 20.0)
        self.gate_thresh = 10 ** (cfg["noise_gate_db"] / 20.0)
        self.comp_thresh_lin = 10 ** (cfg["compression_threshold"] / 20.0)

    def update_config(self):
        """Wird aufgerufen wenn Config sich aendert."""
        self._rebuild_filters()

    def process(self, data):
        """Verarbeitet einen Stereo-Block (numpy array shape [frames, 2]).
        Returns: verarbeitetes Array gleicher Groesse."""
        if self.muted:
            return np.zeros_like(data)

        cfg = self.cfg
        ch = data.shape[1] if data.ndim > 1 else 1

        if ch >= 2:
            left = data[:, 0].copy()
            right = data[:, 1].copy()
        else:
            left = data.flatten().copy()
            right = left.copy()

        # ============ 1. HIGHPASS — Bass Cut ============
        left, self.zi_hp_L = signal.sosfilt(self.sos_hp, left, zi=self.zi_hp_L)
        right, self.zi_hp_R = signal.sosfilt(self.sos_hp, right, zi=self.zi_hp_R)

        # ============ 2. FOOTSTEP BOOST ============
        # Bandpass isoliert Step-Frequenzen, dann mit Gain addieren
        step_L, self.zi_bp_L = signal.sosfilt(self.sos_bp, left, zi=self.zi_bp_L)
        step_R, self.zi_bp_R = signal.sosfilt(self.sos_bp, right, zi=self.zi_bp_R)

        # Step-Band zum Original addieren (gewichtet)
        left = left + step_L * (self.step_gain - 1.0)
        right = right + step_R * (self.step_gain - 1.0)

        # RMS fuer Radar tracken
        self.rms_step_left = float(np.sqrt(np.mean(step_L ** 2)))
        self.rms_step_right = float(np.sqrt(np.mean(step_R ** 2)))

        total_step = self.rms_step_left + self.rms_step_right
        if total_step > 0.001:
            self.peak_direction = (self.rms_step_right - self.rms_step_left) / total_step
            self.step_energy = min(1.0, total_step * 15.0)
        else:
            self.step_energy *= 0.85  # Langsam abfallen

        # ============ 3. NOISE GATE ============
        rms_total = np.sqrt(np.mean(left ** 2) + np.mean(right ** 2))
        if rms_total < self.gate_thresh:
            left *= 0.05   # Fast-Mute, nicht komplett (kein Klick)
            right *= 0.05

        # ============ 4. DYNAMIC COMPRESSION ============
        ratio = cfg["compression_ratio"]
        if ratio > 1.0:
            # Peak-Level
            peak = max(np.max(np.abs(left)), np.max(np.abs(right)), 1e-10)

            # Envelope follower (smooth)
            attack = 0.002
            release = 0.05
            if peak > self.comp_env:
                self.comp_env += attack * (peak - self.comp_env)
            else:
                self.comp_env += release * (peak - self.comp_env)
            self.comp_env = max(self.comp_env, 1e-10)

            # Gain Reduction
            if self.comp_env > self.comp_thresh_lin:
                over_db = 20.0 * np.log10(self.comp_env / self.comp_thresh_lin)
                reduce_db = over_db * (1.0 - 1.0 / ratio)
                comp_gain = 10 ** (-reduce_db / 20.0)
            else:
                comp_gain = 1.0

            # Make-Up Gain (leise Sounds anheben)
            makeup = 10 ** ((reduce_db * 0.5 if self.comp_env > self.comp_thresh_lin else 0) / 20.0)

            left *= comp_gain * makeup
            right *= comp_gain * makeup

        # ============ 5. SPATIAL ENHANCEMENT ============
        width = cfg["spatial_width"]
        if width != 1.0:
            mid = (left + right) * 0.5
            side = (left - right) * 0.5
            side *= width
            left = mid + side
            right = mid - side

        # ============ 6. OUTPUT GAIN + LIMITER ============
        left *= self.out_gain
        right *= self.out_gain

        # Soft Limiter (verhindert Clipping ohne harten Cut)
        left = np.tanh(left)
        right = np.tanh(right)

        # Output zusammenbauen
        if ch >= 2:
            out = np.column_stack([left, right])
            # Falls mehr als 2 Kanaele, Rest mit Nullen fuellen
            if data.shape[1] > 2:
                extra = np.zeros((len(left), data.shape[1] - 2), dtype=data.dtype)
                out = np.column_stack([out, extra])
        else:
            out = left.reshape(-1, 1)

        return out.astype(np.float32)


# ============================================================
# VISUELLER AUDIO RADAR
# ============================================================
class AudioRadar:
    """Zeigt einen visuellen Radar fuer Audio-Richtung."""

    def __init__(self, size=300):
        self.size = size
        self.cx = size // 2
        self.cy = size // 2
        self.radius = size // 2 - 30
        self.running = True

        # History fuer Trail-Effekt
        self.dir_history = deque(maxlen=20)

    def draw(self, direction, energy, preset_name, cfg):
        """Zeichnet den Radar. direction: -1..+1, energy: 0..1"""
        img = np.zeros((self.size, self.size, 3), dtype=np.uint8)

        # Hintergrund-Kreis
        cv2.circle(img, (self.cx, self.cy), self.radius, (30, 30, 30), -1)
        cv2.circle(img, (self.cx, self.cy), self.radius, (60, 60, 60), 2)

        # Kreuz in der Mitte
        cv2.line(img, (self.cx - 8, self.cy), (self.cx + 8, self.cy), (60, 60, 60), 1)
        cv2.line(img, (self.cx, self.cy - 8), (self.cx, self.cy + 8), (60, 60, 60), 1)

        # L/R Labels
        cv2.putText(img, "L", (8, self.cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        cv2.putText(img, "R", (self.size - 20, self.cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

        # Richtungsanzeige-Punkt
        if energy > 0.05:
            self.dir_history.append((direction, energy))

            # Trail zeichnen (alte Punkte dunkler)
            for i, (d, e) in enumerate(self.dir_history):
                alpha = (i + 1) / len(self.dir_history)
                px = int(self.cx + d * self.radius * 0.8)
                py = int(self.cy)
                r = int(3 + e * 12 * alpha)
                brightness = int(80 * alpha)
                cv2.circle(img, (px, py), r, (0, brightness, 0), -1)

            # Aktueller Punkt (hell)
            px = int(self.cx + direction * self.radius * 0.8)
            py = self.cy
            r = int(5 + energy * 15)

            # Farbe nach Staerke: gruen → gelb → rot
            if energy > 0.7:
                color = (0, 100, 255)   # Orange/Rot = sehr nah
            elif energy > 0.4:
                color = (0, 220, 255)   # Gelb = mittel
            else:
                color = (0, 200, 100)   # Gruen = leise

            cv2.circle(img, (px, py), r, color, -1)

            # Richtungslinie
            cv2.line(img, (self.cx, self.cy), (px, py), color, 2)

        # Info-Text
        preset_text = cfg["preset"].upper()
        boost_text = f"BOOST:{cfg['footstep_boost_db']:.0f}dB"
        comp_text = f"COMP:{cfg['compression_ratio']:.1f}x"
        spatial_text = f"SPA:{cfg['spatial_width']:.1f}x"
        bass_text = f"BASS<{cfg['bass_cut_freq']}Hz"

        cv2.putText(img, f"[{preset_text}] {boost_text}", (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 200, 100), 1)
        cv2.putText(img, f"{comp_text} {spatial_text} {bass_text}", (8, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (150, 150, 150), 1)

        # Energie-Bar unten
        bar_w = int(energy * (self.size - 20))
        bar_color = (0, 200, 100) if energy < 0.5 else (0, 220, 255) if energy < 0.8 else (0, 100, 255)
        cv2.rectangle(img, (10, self.size - 18), (10 + bar_w, self.size - 8), bar_color, -1)
        cv2.rectangle(img, (10, self.size - 18), (self.size - 10, self.size - 8), (60, 60, 60), 1)

        return img


# ============================================================
# GERAETE AUFLISTEN
# ============================================================
def list_devices():
    """Zeigt alle Audio-Geraete."""
    print("\n  Verfuegbare Audio-Geraete:")
    print("  " + "=" * 60)
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        inp = d['max_input_channels']
        out = d['max_output_channels']
        direction = ""
        if inp > 0 and out > 0:
            direction = "[IN/OUT]"
        elif inp > 0:
            direction = "[INPUT] "
        elif out > 0:
            direction = "[OUTPUT]"

        marker = ""
        if i == sd.default.device[0]:
            marker += " ← DEFAULT INPUT"
        if i == sd.default.device[1]:
            marker += " ← DEFAULT OUTPUT"

        name = d['name']
        sr = int(d['default_samplerate'])
        print(f"  {i:3d}: {direction} {name} ({sr}Hz, in:{inp} out:{out}){marker}")

    print("  " + "=" * 60)
    return devices


def find_capture_card(devices):
    """Versucht automatisch die Capture Card zu finden."""
    keywords = ["avermedia", "gc571", "capture", "game capture", "hdmi"]
    for i, d in enumerate(devices):
        name_lower = d['name'].lower()
        if d['max_input_channels'] >= 2:
            for kw in keywords:
                if kw in name_lower:
                    return i
    return None


# ============================================================
# HAUPTPROGRAMM
# ============================================================
def main():
    if not SD_AVAILABLE:
        print("Installiere zuerst: pip install sounddevice")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Competitive Audio Tool — Warzone Steps")
    parser.add_argument("--list", action="store_true", help="Geraete auflisten")
    parser.add_argument("--input", type=int, default=None, help="Input Device Index")
    parser.add_argument("--output", type=int, default=None, help="Output Device Index")
    parser.add_argument("--preset", type=str, default=None, choices=["warzone", "multiplayer", "resurgence"])
    parser.add_argument("--no-radar", action="store_true", help="Kein visueller Radar")
    args = parser.parse_args()

    cfg = load_config()

    print("=" * 60)
    print("  COMPETITIVE AUDIO — Warzone Ranked")
    print("  Footstep Enhancement | Compression | Spatial Audio")
    print("=" * 60)

    # Geraete auflisten
    devices = list_devices()

    if args.list:
        return

    # Input-Geraet bestimmen
    input_dev = args.input if args.input is not None else cfg["input_device"]
    if input_dev is None:
        # Auto-Detect Capture Card
        auto = find_capture_card(devices)
        if auto is not None:
            input_dev = auto
            print(f"\n  Auto-Erkannt: Capture Card = Device {auto} ({devices[auto]['name']})")
        else:
            input_dev = sd.default.device[0]
            print(f"\n  Kein Capture Card gefunden — nutze Default Input: Device {input_dev}")

    output_dev = args.output if args.output is not None else cfg["output_device"]
    if output_dev is None:
        output_dev = sd.default.device[1]

    cfg["input_device"] = input_dev
    cfg["output_device"] = output_dev

    # Preset
    if args.preset:
        apply_preset(cfg, args.preset)
    elif cfg["preset"] in PRESETS:
        apply_preset(cfg, cfg["preset"])

    if args.no_radar:
        cfg["show_radar"] = False

    # Samplerate vom Input-Geraet uebernehmen
    dev_info = devices[input_dev]
    sr = int(dev_info['default_samplerate'])
    cfg["samplerate"] = sr
    channels = min(dev_info['max_input_channels'], 2)
    cfg["channels"] = channels

    print(f"\n  Input:  Device {input_dev} — {devices[input_dev]['name']}")
    print(f"  Output: Device {output_dev} — {devices[output_dev]['name']}")
    print(f"  Samplerate: {sr}Hz | Channels: {channels} | Block: {cfg['blocksize']} samples")
    print(f"  Latenz: ~{cfg['blocksize'] / sr * 1000:.1f}ms")
    print(f"\n  Preset: {cfg['preset'].upper()}")
    print(f"  Step Boost: {cfg['footstep_boost_db']}dB ({cfg['step_low_freq']}-{cfg['step_high_freq']}Hz)")
    print(f"  Bass Cut: <{cfg['bass_cut_freq']}Hz")
    print(f"  Compression: {cfg['compression_ratio']}:1 @ {cfg['compression_threshold']}dB")
    print(f"  Spatial: {cfg['spatial_width']}x")
    print(f"  Radar: {'EIN' if cfg['show_radar'] else 'AUS'}")

    print(f"\n  Tasten (im Radar-Fenster):")
    print(f"  1=Warzone  2=MP  3=Resurgence")
    print(f"  Q/W=Boost  E/R=Comp  A/S=BassCut  D/F=Spatial")
    print(f"  M=Mute  V=Radar  ESC=Quit")
    print("=" * 60)

    # Audio Processor
    processor = AudioProcessor(cfg)

    # Radar
    radar = AudioRadar(size=300) if (cfg["show_radar"] and CV2_AVAILABLE) else None

    # Audio Callback
    def audio_callback(indata, outdata, frames, time_info, status):
        if status:
            pass  # Keine Fehlermeldung um Konsole sauber zu halten
        try:
            processed = processor.process(indata)
            outdata[:] = processed[:outdata.shape[0], :outdata.shape[1]]
        except Exception:
            outdata[:] = indata  # Fallback: Pass-Through

    # Stream starten
    print("\n  Starte Audio-Stream...")
    try:
        stream = sd.Stream(
            device=(input_dev, output_dev),
            samplerate=sr,
            blocksize=cfg["blocksize"],
            channels=channels,
            dtype='float32',
            callback=audio_callback,
            latency='low',
        )
        stream.start()
        print("  Audio-Stream: LAEUFT!")
        print("  Druecke ESC im Radar-Fenster oder Ctrl+C zum Beenden.\n")
    except Exception as e:
        print(f"\n  FEHLER beim Starten: {e}")
        print(f"  Versuche: python competitive_audio.py --list")
        print(f"  Dann: python competitive_audio.py --input <INDEX> --output <INDEX>")
        return

    # Hauptschleife (Radar + Hotkeys)
    try:
        frame_count = 0
        while True:
            frame_count += 1

            # Radar zeichnen
            if radar and cfg["show_radar"]:
                img = radar.draw(
                    processor.peak_direction,
                    processor.step_energy,
                    cfg["preset"],
                    cfg,
                )
                cv2.imshow("Audio Radar", img)

            # Tastatur (CV2 oder Sleep)
            if CV2_AVAILABLE and cfg["show_radar"]:
                key = cv2.waitKey(33) & 0xFF  # ~30 FPS fuer Radar
            else:
                time.sleep(0.033)
                key = 255

            if key == 27:  # ESC
                break
            elif key == ord('1'):
                apply_preset(cfg, "warzone")
                processor.update_config()
            elif key == ord('2'):
                apply_preset(cfg, "multiplayer")
                processor.update_config()
            elif key == ord('3'):
                apply_preset(cfg, "resurgence")
                processor.update_config()
            elif key == ord('q'):
                cfg["footstep_boost_db"] = max(0, cfg["footstep_boost_db"] - 2)
                processor.update_config()
                print(f"  Step Boost: {cfg['footstep_boost_db']}dB")
            elif key == ord('w'):
                cfg["footstep_boost_db"] = min(24, cfg["footstep_boost_db"] + 2)
                processor.update_config()
                print(f"  Step Boost: {cfg['footstep_boost_db']}dB")
            elif key == ord('e'):
                cfg["compression_ratio"] = max(1.0, round(cfg["compression_ratio"] - 0.5, 1))
                processor.update_config()
                print(f"  Compression: {cfg['compression_ratio']}:1")
            elif key == ord('r'):
                cfg["compression_ratio"] = min(8.0, round(cfg["compression_ratio"] + 0.5, 1))
                processor.update_config()
                print(f"  Compression: {cfg['compression_ratio']}:1")
            elif key == ord('a'):
                cfg["bass_cut_freq"] = max(50, cfg["bass_cut_freq"] - 20)
                processor.update_config()
                print(f"  Bass Cut: <{cfg['bass_cut_freq']}Hz")
            elif key == ord('s'):
                cfg["bass_cut_freq"] = min(400, cfg["bass_cut_freq"] + 20)
                processor.update_config()
                print(f"  Bass Cut: <{cfg['bass_cut_freq']}Hz")
            elif key == ord('d'):
                cfg["spatial_width"] = max(0.5, round(cfg["spatial_width"] - 0.1, 1))
                processor.update_config()
                print(f"  Spatial: {cfg['spatial_width']}x")
            elif key == ord('f'):
                cfg["spatial_width"] = min(2.5, round(cfg["spatial_width"] + 0.1, 1))
                processor.update_config()
                print(f"  Spatial: {cfg['spatial_width']}x")
            elif key == ord('m'):
                processor.muted = not processor.muted
                print(f"  {'MUTED' if processor.muted else 'UNMUTED'}")
            elif key == ord('v'):
                cfg["show_radar"] = not cfg["show_radar"]
                if not cfg["show_radar"] and CV2_AVAILABLE:
                    cv2.destroyAllWindows()
                print(f"  Radar: {'EIN' if cfg['show_radar'] else 'AUS'}")

    except KeyboardInterrupt:
        print("\n  Stop.")
    finally:
        stream.stop()
        stream.close()
        save_config(cfg)
        print(f"  Config gespeichert: {CONFIG_FILE}")
        if CV2_AVAILABLE:
            cv2.destroyAllWindows()
        print("  Beendet.")


if __name__ == "__main__":
    main()
