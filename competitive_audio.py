"""
WARZONE COMPETITIVE AUDIO V2 — DT 990 Pro Edition
====================================================
Chirurgische Peaking-EQ Filter auf exakten Footstep-Frequenzen.
Basierend auf Spectrogram-Analyse von FPS-Footsteps.

SIGNAL-KETTE:
  1. Highpass 100Hz (5th order) — Rumble/Explosionen komplett weg
  2. Peaking +8dB @ 200Hz Q=1.8 — Footstep BODY (Thud/Aufprall)
  3. Peaking -4dB @ 600Hz Q=1.5 — MUD CUT (Waffen-Body/Ambient raus)
  4. Peaking +8dB @ 2400Hz Q=2.5 — Footstep TEXTURE (Schritte/Richtung)
  5. Peaking +5dB @ 4400Hz Q=1.2 — Footstep EDGE (Klarheit/Definition)
  6. Peaking -6dB @ 5500Hz Q=2.0 — Gunfire CRACK Suppression
  7. Notch  -5dB @ 8000Hz Q=3.0 — DT 990 Pro Treble-Spitze zaehmen
  8. Lowpass 12kHz (2nd order) — Hiss/Rauschen abschneiden
  9. Noise Gate
  10. Dynamic Compression mit Makeup-Gain
  11. Stereo Spatial Enhancement (Side-Channel verstaerkt)
  12. Soft Limiter

INSTALLATION:
  pip install sounddevice numpy scipy opencv-python

STEUERUNG:
  1/2/3   = Preset: Warzone / Multiplayer / Resurgence
  Q/W     = Step Body (200Hz) Boost -/+
  E/R     = Step Texture (2.4kHz) Boost -/+
  A/S     = Step Edge (4.4kHz) Boost -/+
  D/F     = Spatial Width -/+
  T/Z     = Compression -/+
  G/H     = Output Gain -/+
  U/I     = Gunfire Cut -/+
  O       = Mud Cut toggle (600Hz)
  M       = Mute/Unmute
  V       = Radar an/aus
  P       = Alle Settings anzeigen
  ESC     = Beenden
"""

import sys
import os
import json
import time
import argparse
from collections import deque

import numpy as np
from scipy import signal as sig

try:
    import sounddevice as sd
except ImportError:
    sd = None
    print("FEHLER: pip install sounddevice")

try:
    import cv2
except ImportError:
    cv2 = None

# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(SCRIPT_DIR, "audio_config.json")

DEFAULT_CFG = {
    "input_device":   None,
    "output_device":  None,
    "samplerate":     48000,
    "blocksize":      512,
    "channels":       2,
    "preset":         "warzone",

    # === CHIRURGISCHE EQ BANDS ===
    # Footstep Body (Thud/Aufprall auf Boden)
    "step_body_hz":     200,
    "step_body_db":     8.0,
    "step_body_q":      1.8,

    # Mud Cut (Waffen-Body, Ambient-Muell)
    "mud_cut_hz":       600,
    "mud_cut_db":       -4.0,
    "mud_cut_q":        1.5,
    "mud_cut_on":       True,

    # Footstep Texture (Schritte hoerbar, Richtung)
    "step_texture_hz":  2400,
    "step_texture_db":  8.0,
    "step_texture_q":   2.5,

    # Footstep Edge (Klarheit/Definition)
    "step_edge_hz":     4400,
    "step_edge_db":     5.0,
    "step_edge_q":      1.2,

    # Gunfire Crack Suppression
    "gunfire_hz":       5500,
    "gunfire_db":       -6.0,
    "gunfire_q":        2.0,

    # DT990 Treble Tame
    "treble_hz":        8000,
    "treble_db":        -5.0,
    "treble_q":         3.0,

    # Highpass / Lowpass
    "highpass_hz":      100,
    "lowpass_hz":       12000,

    # Dynamic Compression
    "comp_ratio":       4.0,
    "comp_threshold":   -18.0,
    "comp_attack":      0.003,
    "comp_release":     0.06,

    # Noise Gate
    "gate_db":          -46.0,

    # Spatial
    "spatial_width":    1.6,

    # Output
    "output_gain_db":   2.0,

    # Radar
    "show_radar":       True,
}

PRESETS = {
    "warzone": {
        "step_body_db": 8.0, "step_texture_db": 8.0, "step_edge_db": 5.0,
        "mud_cut_db": -4.0, "mud_cut_on": True,
        "gunfire_db": -6.0, "treble_db": -5.0,
        "highpass_hz": 100, "lowpass_hz": 12000,
        "comp_ratio": 4.0, "comp_threshold": -18.0,
        "spatial_width": 1.6, "gate_db": -46.0, "output_gain_db": 2.0,
    },
    "multiplayer": {
        "step_body_db": 6.0, "step_texture_db": 7.0, "step_edge_db": 4.0,
        "mud_cut_db": -3.0, "mud_cut_on": True,
        "gunfire_db": -4.0, "treble_db": -4.0,
        "highpass_hz": 80, "lowpass_hz": 13000,
        "comp_ratio": 3.0, "comp_threshold": -16.0,
        "spatial_width": 1.3, "gate_db": -44.0, "output_gain_db": 1.0,
    },
    "resurgence": {
        "step_body_db": 10.0, "step_texture_db": 10.0, "step_edge_db": 6.0,
        "mud_cut_db": -5.0, "mud_cut_on": True,
        "gunfire_db": -8.0, "treble_db": -6.0,
        "highpass_hz": 110, "lowpass_hz": 11000,
        "comp_ratio": 5.0, "comp_threshold": -20.0,
        "spatial_width": 1.8, "gate_db": -48.0, "output_gain_db": 3.0,
    },
}


def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                saved = json.load(f)
            cfg = DEFAULT_CFG.copy()
            cfg.update(saved)
            return cfg
        except Exception:
            pass
    return DEFAULT_CFG.copy()


def save_config(cfg):
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(cfg, f, indent=2)
    except Exception:
        pass


def apply_preset(cfg, name):
    if name in PRESETS:
        cfg.update(PRESETS[name])
        cfg["preset"] = name


# ============================================================
# PARAMETRIC PEAKING EQ — Audio EQ Cookbook (Robert Bristow-Johnson)
# ============================================================
def make_peaking_eq(fc, gain_db, Q, fs):
    """Erzeugt einen Peaking-EQ als b/a Koeffizienten.
    fc: Centerfrequenz Hz, gain_db: Gain in dB, Q: Guete, fs: Samplerate.
    Basiert auf Audio EQ Cookbook."""
    A = 10 ** (gain_db / 40.0)  # sqrt(10^(dB/20))
    w0 = 2.0 * np.pi * fc / fs
    alpha = np.sin(w0) / (2.0 * Q)

    b0 = 1.0 + alpha * A
    b1 = -2.0 * np.cos(w0)
    b2 = 1.0 - alpha * A
    a0 = 1.0 + alpha / A
    a1 = -2.0 * np.cos(w0)
    a2 = 1.0 - alpha / A

    b = np.array([b0 / a0, b1 / a0, b2 / a0])
    a = np.array([1.0, a1 / a0, a2 / a0])
    return b, a


# ============================================================
# AUDIO PROCESSOR V2
# ============================================================
class CompetitiveAudioV2:
    """Chirurgische Audio-Verarbeitung mit Peaking-EQ auf Step-Frequenzen."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.sr = cfg["samplerate"]
        self.nyq = self.sr / 2.0
        self.muted = False
        self.comp_env = 0.0

        # Radar
        self.step_energy_L = 0.0
        self.step_energy_R = 0.0
        self.peak_dir = 0.0
        self.step_power = 0.0

        self._build()

    def _safe_freq(self, hz):
        return max(20.0, min(float(hz), self.nyq * 0.95))

    def _build(self):
        """Alle Filter bauen."""
        sr = self.sr
        nyq = self.nyq
        c = self.cfg

        # === HIGHPASS (Butterworth 5th order) ===
        f_hp = self._safe_freq(c["highpass_hz"])
        self.sos_hp = sig.butter(5, f_hp / nyq, btype='highpass', output='sos')

        # === LOWPASS (Butterworth 2nd order) ===
        f_lp = self._safe_freq(c["lowpass_hz"])
        self.sos_lp = sig.butter(2, f_lp / nyq, btype='lowpass', output='sos')

        # === PEAKING EQ BANDS ===
        self.eq_bands = []

        # 1. Footstep Body
        if abs(c["step_body_db"]) > 0.1:
            b, a = make_peaking_eq(self._safe_freq(c["step_body_hz"]),
                                   c["step_body_db"], c["step_body_q"], sr)
            self.eq_bands.append(("body", b, a))

        # 2. Mud Cut
        if c["mud_cut_on"] and abs(c["mud_cut_db"]) > 0.1:
            b, a = make_peaking_eq(self._safe_freq(c["mud_cut_hz"]),
                                   c["mud_cut_db"], c["mud_cut_q"], sr)
            self.eq_bands.append(("mud", b, a))

        # 3. Footstep Texture
        if abs(c["step_texture_db"]) > 0.1:
            b, a = make_peaking_eq(self._safe_freq(c["step_texture_hz"]),
                                   c["step_texture_db"], c["step_texture_q"], sr)
            self.eq_bands.append(("texture", b, a))

        # 4. Footstep Edge
        if abs(c["step_edge_db"]) > 0.1:
            b, a = make_peaking_eq(self._safe_freq(c["step_edge_hz"]),
                                   c["step_edge_db"], c["step_edge_q"], sr)
            self.eq_bands.append(("edge", b, a))

        # 5. Gunfire Cut
        if abs(c["gunfire_db"]) > 0.1:
            b, a = make_peaking_eq(self._safe_freq(c["gunfire_hz"]),
                                   c["gunfire_db"], c["gunfire_q"], sr)
            self.eq_bands.append(("gun", b, a))

        # 6. DT990 Treble
        if abs(c["treble_db"]) > 0.1:
            b, a = make_peaking_eq(self._safe_freq(c["treble_hz"]),
                                   c["treble_db"], c["treble_q"], sr)
            self.eq_bands.append(("treble", b, a))

        # Bandpass fuer Step-Radar-Erkennung (2-4kHz)
        bp_lo = self._safe_freq(1500)
        bp_hi = self._safe_freq(4500)
        self.sos_radar = sig.butter(2, [bp_lo / nyq, bp_hi / nyq], btype='bandpass', output='sos')

        # Pre-compute gains
        self.out_gain = 10 ** (c["output_gain_db"] / 20.0)
        self.gate_lin = 10 ** (c["gate_db"] / 20.0)
        self.comp_thresh = 10 ** (c["comp_threshold"] / 20.0)

        # Filter-States initialisieren
        self._init_zi()

    def _init_zi(self):
        """Filter-Zustaende fuer 2 Kanaele."""
        # SOS-States
        z_sos = lambda sos: np.zeros((sos.shape[0], 2))
        self.zi_hp = [z_sos(self.sos_hp), z_sos(self.sos_hp)]
        self.zi_lp = [z_sos(self.sos_lp), z_sos(self.sos_lp)]
        self.zi_radar = [z_sos(self.sos_radar), z_sos(self.sos_radar)]

        # IIR-States (b/a) fuer jedes EQ-Band, pro Kanal
        self.zi_eq = []
        for name, b, a in self.eq_bands:
            zi_L = sig.lfilter_zi(b, a) * 0.0
            zi_R = sig.lfilter_zi(b, a) * 0.0
            self.zi_eq.append([zi_L, zi_R])

    def rebuild(self):
        self._build()

    def process(self, data):
        if self.muted:
            return np.zeros_like(data)

        ch = min(data.shape[1] if data.ndim > 1 else 1, 2)
        if ch >= 2:
            channels = [data[:, 0].copy(), data[:, 1].copy()]
        else:
            channels = [data.flatten().copy(), data.flatten().copy()]

        for i in range(2):
            x = channels[i]

            # === 1. HIGHPASS — Rumble/Explosionen weg ===
            x, self.zi_hp[i] = sig.sosfilt(self.sos_hp, x, zi=self.zi_hp[i])

            # === 2-7. PEAKING EQ BANDS (chirurgisch) ===
            for band_idx, (name, b, a) in enumerate(self.eq_bands):
                x, self.zi_eq[band_idx][i] = sig.lfilter(b, a, x, zi=self.zi_eq[band_idx][i])

            # === 8. LOWPASS — Hiss weg ===
            x, self.zi_lp[i] = sig.sosfilt(self.sos_lp, x, zi=self.zi_lp[i])

            # Radar: Step-Band Energie messen (2-4kHz)
            radar_band, self.zi_radar[i] = sig.sosfilt(self.sos_radar, x, zi=self.zi_radar[i])
            rms = float(np.sqrt(np.mean(radar_band ** 2)))
            if i == 0:
                self.step_energy_L = rms
            else:
                self.step_energy_R = rms

            channels[i] = x

        left, right = channels

        # === 9. NOISE GATE ===
        rms_total = np.sqrt(np.mean(left ** 2) + np.mean(right ** 2))
        if rms_total < self.gate_lin:
            left *= 0.02
            right *= 0.02

        # === 10. DYNAMIC COMPRESSION ===
        ratio = self.cfg["comp_ratio"]
        if ratio > 1.01:
            peak = max(np.max(np.abs(left)), np.max(np.abs(right)), 1e-10)

            att = self.cfg["comp_attack"]
            rel = self.cfg["comp_release"]
            if peak > self.comp_env:
                self.comp_env += att * (peak - self.comp_env)
            else:
                self.comp_env += rel * (peak - self.comp_env)
            self.comp_env = max(self.comp_env, 1e-10)

            if self.comp_env > self.comp_thresh:
                over_db = 20.0 * np.log10(self.comp_env / self.comp_thresh)
                reduce_db = over_db * (1.0 - 1.0 / ratio)
                comp_g = 10 ** (-reduce_db / 20.0)
                # 50% Makeup Gain — leise Steps deutlich anheben
                makeup = 10 ** (reduce_db * 0.50 / 20.0)
            else:
                comp_g = 1.0
                makeup = 1.0

            left *= comp_g * makeup
            right *= comp_g * makeup

        # === 11. SPATIAL ENHANCEMENT ===
        # Mid/Side: Side-Channel verstaerken = bessere Richtung
        w = self.cfg["spatial_width"]
        if abs(w - 1.0) > 0.01:
            mid = (left + right) * 0.5
            side = (left - right) * 0.5
            side *= w
            left = mid + side
            right = mid - side

        # === 12. OUTPUT GAIN + SOFT LIMITER ===
        left *= self.out_gain
        right *= self.out_gain
        left = np.tanh(left)
        right = np.tanh(right)

        # Radar-Daten
        total = self.step_energy_L + self.step_energy_R
        if total > 0.0003:
            self.peak_dir = (self.step_energy_R - self.step_energy_L) / total
            self.step_power = min(1.0, total * 25.0)
        else:
            self.step_power *= 0.82

        out = np.column_stack([left, right])
        if data.ndim > 1 and data.shape[1] > 2:
            extra = np.zeros((len(left), data.shape[1] - 2), dtype=np.float32)
            out = np.column_stack([out, extra])
        return out.astype(np.float32)


# ============================================================
# RADAR
# ============================================================
class StepRadar:
    SZ = 320

    def __init__(self):
        self.cx = self.SZ // 2
        self.cy = self.SZ // 2
        self.r = self.SZ // 2 - 35
        self.trail = deque(maxlen=25)

    def draw(self, direction, power, cfg):
        img = np.zeros((self.SZ, self.SZ, 3), dtype=np.uint8)

        cv2.circle(img, (self.cx, self.cy), self.r, (25, 25, 30), -1)
        cv2.circle(img, (self.cx, self.cy), self.r, (50, 50, 60), 2)
        cv2.circle(img, (self.cx, self.cy), self.r // 2, (35, 35, 40), 1)
        cv2.line(img, (self.cx - 10, self.cy), (self.cx + 10, self.cy), (50, 50, 60), 1)
        cv2.line(img, (self.cx, self.cy - 10), (self.cx, self.cy + 10), (50, 50, 60), 1)
        cv2.putText(img, "L", (6, self.cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 90), 1)
        cv2.putText(img, "R", (self.SZ - 18, self.cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 90), 1)

        if power > 0.04:
            self.trail.append((direction, power))
            for idx, (d, p) in enumerate(self.trail):
                a = (idx + 1) / len(self.trail)
                px = int(self.cx + d * self.r * 0.85)
                sz = int(2 + p * 10 * a)
                cv2.circle(img, (px, self.cy), sz, (0, int(60 * a), 0), -1)

            px = int(self.cx + direction * self.r * 0.85)
            sz = int(4 + power * 18)
            col = (0, 80, 255) if power > 0.7 else (0, 200, 255) if power > 0.4 else (0, 200, 120)
            cv2.circle(img, (px, self.cy), sz, col, -1)
            cv2.line(img, (self.cx, self.cy), (px, self.cy), col, 2)

        p = cfg["preset"].upper()
        cv2.putText(img, f"[{p}] BODY:{cfg['step_body_db']:.0f}dB TEX:{cfg['step_texture_db']:.0f}dB EDGE:{cfg['step_edge_db']:.0f}dB",
                    (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 180, 100), 1)
        cv2.putText(img, f"MUD:{cfg['mud_cut_db']:.0f}dB GUN:{cfg['gunfire_db']:.0f}dB DT990:{cfg['treble_db']:.0f}dB",
                    (6, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (130, 130, 140), 1)
        cv2.putText(img, f"COMP:{cfg['comp_ratio']:.1f}x SPA:{cfg['spatial_width']:.1f}x GAIN:{cfg['output_gain_db']:.0f}dB HP:{cfg['highpass_hz']}Hz",
                    (6, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (130, 130, 140), 1)

        bw = int(power * (self.SZ - 20))
        bc = (0, 180, 100) if power < 0.4 else (0, 200, 255) if power < 0.7 else (0, 80, 255)
        cv2.rectangle(img, (10, self.SZ - 16), (10 + bw, self.SZ - 6), bc, -1)
        cv2.rectangle(img, (10, self.SZ - 16), (self.SZ - 10, self.SZ - 6), (50, 50, 60), 1)
        return img


# ============================================================
# DEVICES
# ============================================================
def list_devices():
    devs = sd.query_devices()
    print("\n  Audio-Geraete:")
    print("  " + "-" * 70)
    for i, d in enumerate(devs):
        inp, out = d['max_input_channels'], d['max_output_channels']
        tag = "[IN]    " if inp > 0 and out == 0 else "[OUT]   " if out > 0 and inp == 0 else "[IN/OUT]" if inp > 0 else "[---]   "
        mark = ""
        if i == sd.default.device[0]:
            mark += " << IN"
        if i == sd.default.device[1]:
            mark += " << OUT"
        print(f"  {i:3d} {tag} {d['name'][:48]:<48} {int(d['default_samplerate'])}Hz in:{inp} out:{out}{mark}")
    print("  " + "-" * 70)
    return devs


def auto_find_capture(devs):
    for i, d in enumerate(devs):
        if d['max_input_channels'] >= 2:
            name = d['name'].lower()
            for kw in ["avermedia", "gc571", "capture", "game capture", "live gamer"]:
                if kw in name:
                    return i
    return None


# ============================================================
# MAIN
# ============================================================
def main():
    if sd is None:
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Warzone Competitive Audio V2")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--input", type=int, default=None)
    parser.add_argument("--output", type=int, default=None)
    parser.add_argument("--preset", choices=["warzone", "multiplayer", "resurgence"])
    parser.add_argument("--no-radar", action="store_true")
    args = parser.parse_args()

    cfg = load_config()

    print()
    print("  " + "=" * 58)
    print("  WARZONE COMPETITIVE AUDIO V2")
    print("  Chirurgische Peaking-EQ | DT 990 Pro Edition")
    print("  " + "=" * 58)

    devs = list_devices()
    if args.list:
        return

    in_dev = args.input if args.input is not None else cfg["input_device"]
    if in_dev is None:
        auto = auto_find_capture(devs)
        if auto is not None:
            in_dev = auto
            print(f"\n  Capture Card: [{auto}] {devs[auto]['name']}")
        else:
            in_dev = sd.default.device[0]
            print(f"\n  Default Input: [{in_dev}]")

    out_dev = args.output if args.output is not None else cfg["output_device"]
    if out_dev is None:
        out_dev = sd.default.device[1]

    cfg["input_device"] = in_dev
    cfg["output_device"] = out_dev

    if args.preset:
        apply_preset(cfg, args.preset)
    elif cfg["preset"] in PRESETS:
        apply_preset(cfg, cfg["preset"])
    if args.no_radar:
        cfg["show_radar"] = False

    sr = int(devs[in_dev]['default_samplerate'])
    cfg["samplerate"] = sr
    ch = min(devs[in_dev]['max_input_channels'], 2)
    cfg["channels"] = ch

    print(f"\n  Input:    [{in_dev}] {devs[in_dev]['name']}")
    print(f"  Output:   [{out_dev}] {devs[out_dev]['name']}")
    print(f"  SR: {sr}Hz | CH: {ch} | Block: {cfg['blocksize']} (~{cfg['blocksize']/sr*1000:.1f}ms)")
    print(f"\n  Preset: {cfg['preset'].upper()}")
    print(f"  EQ: Body +{cfg['step_body_db']}dB@{cfg['step_body_hz']}Hz"
          f" | Tex +{cfg['step_texture_db']}dB@{cfg['step_texture_hz']}Hz"
          f" | Edge +{cfg['step_edge_db']}dB@{cfg['step_edge_hz']}Hz")
    print(f"  Cut: Mud {cfg['mud_cut_db']}dB@{cfg['mud_cut_hz']}Hz"
          f" | Gun {cfg['gunfire_db']}dB@{cfg['gunfire_hz']}Hz"
          f" | DT990 {cfg['treble_db']}dB@{cfg['treble_hz']}Hz")
    print(f"  HP: {cfg['highpass_hz']}Hz | LP: {cfg['lowpass_hz']}Hz"
          f" | Comp: {cfg['comp_ratio']}:1 | Spatial: {cfg['spatial_width']}x"
          f" | Gain: {cfg['output_gain_db']}dB")
    print(f"\n  Tasten: 1=WZ 2=MP 3=RES | Q/W=Body E/R=Texture A/S=Edge")
    print(f"          D/F=Spatial T/Z=Comp G/H=Gain U/I=GunCut O=Mud M=Mute V=Radar ESC=Quit")
    print("  " + "=" * 58)

    proc = CompetitiveAudioV2(cfg)
    radar = StepRadar() if (cfg["show_radar"] and cv2 is not None) else None

    def callback(indata, outdata, frames, time_info, status):
        try:
            outdata[:] = proc.process(indata)[:outdata.shape[0], :outdata.shape[1]]
        except Exception:
            outdata[:] = indata

    print("\n  Starte Audio-Stream...")
    try:
        stream = sd.Stream(
            device=(in_dev, out_dev), samplerate=sr, blocksize=cfg["blocksize"],
            channels=ch, dtype='float32', callback=callback, latency='low',
        )
        stream.start()
        print("  LAEUFT!\n")
    except Exception as e:
        print(f"\n  FEHLER: {e}")
        print(f"  Tipp: python competitive_audio.py --list")
        return

    try:
        while True:
            if radar and cfg["show_radar"] and cv2 is not None:
                img = radar.draw(proc.peak_dir, proc.step_power, cfg)
                cv2.imshow("Step Radar", img)

            if cv2 is not None and cfg["show_radar"]:
                key = cv2.waitKey(33) & 0xFF
            else:
                time.sleep(0.033)
                key = 255

            if key == 27:
                break

            elif key == ord('1'):
                apply_preset(cfg, "warzone"); proc.rebuild()
                print("  >> WARZONE")
            elif key == ord('2'):
                apply_preset(cfg, "multiplayer"); proc.rebuild()
                print("  >> MULTIPLAYER")
            elif key == ord('3'):
                apply_preset(cfg, "resurgence"); proc.rebuild()
                print("  >> RESURGENCE")

            # Step Body
            elif key == ord('q'):
                cfg["step_body_db"] = max(0, cfg["step_body_db"] - 2); proc.rebuild()
                print(f"  Body: {cfg['step_body_db']:.0f}dB @ {cfg['step_body_hz']}Hz")
            elif key == ord('w'):
                cfg["step_body_db"] = min(18, cfg["step_body_db"] + 2); proc.rebuild()
                print(f"  Body: {cfg['step_body_db']:.0f}dB @ {cfg['step_body_hz']}Hz")

            # Step Texture
            elif key == ord('e'):
                cfg["step_texture_db"] = max(0, cfg["step_texture_db"] - 2); proc.rebuild()
                print(f"  Texture: {cfg['step_texture_db']:.0f}dB @ {cfg['step_texture_hz']}Hz")
            elif key == ord('r'):
                cfg["step_texture_db"] = min(18, cfg["step_texture_db"] + 2); proc.rebuild()
                print(f"  Texture: {cfg['step_texture_db']:.0f}dB @ {cfg['step_texture_hz']}Hz")

            # Step Edge
            elif key == ord('a'):
                cfg["step_edge_db"] = max(0, cfg["step_edge_db"] - 2); proc.rebuild()
                print(f"  Edge: {cfg['step_edge_db']:.0f}dB @ {cfg['step_edge_hz']}Hz")
            elif key == ord('s'):
                cfg["step_edge_db"] = min(18, cfg["step_edge_db"] + 2); proc.rebuild()
                print(f"  Edge: {cfg['step_edge_db']:.0f}dB @ {cfg['step_edge_hz']}Hz")

            # Spatial
            elif key == ord('d'):
                cfg["spatial_width"] = max(0.5, round(cfg["spatial_width"] - 0.1, 1))
                print(f"  Spatial: {cfg['spatial_width']:.1f}x")
            elif key == ord('f'):
                cfg["spatial_width"] = min(3.0, round(cfg["spatial_width"] + 0.1, 1))
                print(f"  Spatial: {cfg['spatial_width']:.1f}x")

            # Compression
            elif key == ord('t'):
                cfg["comp_ratio"] = max(1.0, round(cfg["comp_ratio"] - 0.5, 1)); proc.rebuild()
                print(f"  Comp: {cfg['comp_ratio']:.1f}:1")
            elif key == ord('z'):
                cfg["comp_ratio"] = min(8.0, round(cfg["comp_ratio"] + 0.5, 1)); proc.rebuild()
                print(f"  Comp: {cfg['comp_ratio']:.1f}:1")

            # Output Gain
            elif key == ord('g'):
                cfg["output_gain_db"] = max(-12, cfg["output_gain_db"] - 1); proc.rebuild()
                print(f"  Gain: {cfg['output_gain_db']:.0f}dB")
            elif key == ord('h'):
                cfg["output_gain_db"] = min(18, cfg["output_gain_db"] + 1); proc.rebuild()
                print(f"  Gain: {cfg['output_gain_db']:.0f}dB")

            # Gunfire Cut
            elif key == ord('u'):
                cfg["gunfire_db"] = min(0, cfg["gunfire_db"] + 1); proc.rebuild()
                print(f"  Gun: {cfg['gunfire_db']:.0f}dB")
            elif key == ord('i'):
                cfg["gunfire_db"] = max(-15, cfg["gunfire_db"] - 1); proc.rebuild()
                print(f"  Gun: {cfg['gunfire_db']:.0f}dB")

            # Mud Cut Toggle
            elif key == ord('o'):
                cfg["mud_cut_on"] = not cfg["mud_cut_on"]; proc.rebuild()
                print(f"  Mud Cut: {'EIN' if cfg['mud_cut_on'] else 'AUS'}")

            elif key == ord('m'):
                proc.muted = not proc.muted
                print(f"  {'MUTED' if proc.muted else 'UNMUTED'}")
            elif key == ord('v'):
                cfg["show_radar"] = not cfg["show_radar"]
                if not cfg["show_radar"] and cv2 is not None:
                    cv2.destroyAllWindows()
                print(f"  Radar: {'EIN' if cfg['show_radar'] else 'AUS'}")
            elif key == ord('p'):
                print(f"\n  === CONFIG ===")
                for k, v in sorted(cfg.items()):
                    if k not in ("input_device", "output_device"):
                        print(f"    {k}: {v}")
                print()

    except KeyboardInterrupt:
        print("\n  Ctrl+C")
    finally:
        stream.stop()
        stream.close()
        save_config(cfg)
        print(f"  Config gespeichert: {CONFIG_FILE}")
        if cv2 is not None:
            cv2.destroyAllWindows()
        print("  Beendet.")


if __name__ == "__main__":
    main()
