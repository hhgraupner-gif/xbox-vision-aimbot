"""
eRayz Audio — TRANSIENT ENGINE v5
===================================
Per-Band Transient Detection | Sustained Sound Killer
Footstep Isolation | Phase-Aligned Crossover
DT 990 Pro | Raw HDMI 48kHz

Die Idee: Footsteps = kurze, scharfe TRANSIENTEN.
Gunfire, Wind, Ambient = SUSTAINED Sounds.
→ Transienten boosten, Sustained killen = NUR Steps hoeren.

STEUERUNG:
  Q/W    Band A (Step Thump 100-300Hz) -/+
  E/R    Band B (Mud Kill 300-900Hz) -/+
  A/S    Band C (Step Detail 900-4500Hz) -/+
  D/F    Band D (Gunfire Kill 4500Hz+) -/+
  T/Z    Transient Sensitivity -/+
  G/H    Output Gain -/+
  X/C    Spatial -/+
  J/K    Sustain Kill -/+
  B      BYPASS
  M      Mute
  V      Radar
  P      Settings
  ESC    Beenden
"""

import sys, os, json, time, argparse, threading, math
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(SCRIPT_DIR, "audio_config.json")

DEFAULT_CFG = {
    "input_device": None, "output_device": None,
    "input_name": "", "output_name": "",
    "samplerate": 48000, "blocksize": 256, "channels": 2,
    # ══════════════════════════════════════════════════════════
    # TRANSIENT ENGINE v5 — Pro Audio vom Feinsten
    #
    # Crossover Points (Phase-Aligned Linkwitz-Riley)
    #   100Hz ─── 300Hz ─── 900Hz ─── 4500Hz ─── 8000Hz
    #   │ Band A  │ Band B  │  Band C  │  Band D  │
    #   │ Thump   │ Mud     │  Detail  │  Gunfire │
    # ══════════════════════════════════════════════════════════
    "xover_ab": 300,
    "xover_bc": 900,
    "xover_cd": 4500,
    "highpass_hz": 100,
    "lowpass_hz": 8000,
    #
    # Per-Band Gain
    "gain_a_db": 8.0,      # Step Thump — nur Transienten
    "gain_b_db": -20.0,    # Mud — KOMPLETT TOT
    "gain_c_db": 20.0,     # Step Detail — NUR STEPS
    "gain_d_db": -20.0,    # Gunfire — TOT
    #
    # Per-Band Compression (Soft-Knee)
    "comp_a_ratio": 8.0,  "comp_a_thresh": -30.0,
    "comp_b_ratio": 1.0,  "comp_b_thresh": -10.0,
    "comp_c_ratio": 12.0, "comp_c_thresh": -35.0,
    "comp_d_ratio": 20.0, "comp_d_thresh": -8.0,
    #
    # TRANSIENT — MAXIMAL. Nur Steps durchlassen.
    "transient_sensitivity": 2.5,
    "sustain_kill": 0.95,
    #
    # DT 990 Pro Fix
    "dt990_hz": 8000, "dt990_db": -8.0, "dt990_q": 2.5,
    #
    # Spatial
    "spatial_width": 1.8,
    #
    # Output
    "output_gain_db": 5.0,
    "gate_db": -62.0,
    "show_radar": True,
}


def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                cfg = DEFAULT_CFG.copy()
                cfg.update(json.load(f))
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


def make_peak_eq(fc, db, Q, fs):
    ny = fs / 2.0
    fc = max(20.0, min(float(fc), ny * 0.95))
    A = 10 ** (db / 40.0)
    w0 = 2 * np.pi * fc / fs
    al = np.sin(w0) / (2 * Q)
    b0, b1, b2 = 1 + al*A, -2*np.cos(w0), 1 - al*A
    a0, a1, a2 = 1 + al/A, -2*np.cos(w0), 1 - al/A
    return np.array([b0/a0, b1/a0, b2/a0]), np.array([1.0, a1/a0, a2/a0])


class EnvelopeFollower:
    """Trackt Audio-Hüllkurve mit einstellbarem Attack/Release."""
    def __init__(self, attack_ms, release_ms, sr):
        self.att = math.exp(-1.0 / (sr * attack_ms / 1000.0))
        self.rel = math.exp(-1.0 / (sr * release_ms / 1000.0))
        self.env = 0.0

    def process_block(self, x):
        """Gibt den Envelope-Wert am Ende des Blocks zurück."""
        peak = float(np.max(np.abs(x)))
        if peak > self.env:
            self.env = self.att * self.env + (1 - self.att) * peak
        else:
            self.env = self.rel * self.env + (1 - self.rel) * peak
        self.env = max(self.env, 1e-10)
        return self.env


class TransientDetector:
    """Erkennt Transienten durch Vergleich Fast/Slow Envelope."""
    def __init__(self, sr):
        self.fast = EnvelopeFollower(attack_ms=0.5, release_ms=8, sr=sr)
        self.slow = EnvelopeFollower(attack_ms=40, release_ms=150, sr=sr)

    def process(self, x):
        """Returns: (transient_ratio, fast_env, slow_env)
        ratio > 1.5 = Transient (Step)
        ratio < 0.8 = Sustained (Ambient)
        """
        f = self.fast.process_block(x)
        s = self.slow.process_block(x)
        ratio = f / max(s, 1e-10)
        return ratio, f, s


# ============================================================
# TRANSIENT ENGINE
# ============================================================
class TransientEngine:
    def __init__(self, cfg):
        self.cfg = cfg
        self.sr = cfg["samplerate"]
        self.ny = self.sr / 2.0
        self.muted = False
        self.bypass = False
        self.step_L = 0.0
        self.step_R = 0.0
        self.peak_dir = 0.0
        self.step_pwr = 0.0
        # Per-band compressor envelopes [A,B,C,D] x [L,R]
        self.comp_env = [[1e-10, 1e-10] for _ in range(4)]
        # Per-band transient detectors [A,B,C,D] x [L,R]
        self.td = [[TransientDetector(self.sr), TransientDetector(self.sr)] for _ in range(4)]
        # Transient ratios for display
        self.t_ratios = [0.0] * 4
        self._build()

    def _f(self, hz):
        return max(20.0, min(float(hz), self.ny * 0.95))

    def _build(self):
        c = self.cfg
        ny = self.ny

        # Highpass + Lowpass (Pre-filter)
        self.sos_hp = sig.butter(5, self._f(c["highpass_hz"]) / ny, 'highpass', output='sos')
        self.sos_lp = sig.butter(3, self._f(c["lowpass_hz"]) / ny, 'lowpass', output='sos')

        # Linkwitz-Riley Crossovers (4th order = 2x Butterworth 2nd)
        ab = self._f(c["xover_ab"]) / ny
        bc = self._f(c["xover_bc"]) / ny
        cd = self._f(c["xover_cd"]) / ny

        self.sos_a = sig.butter(4, ab, 'lowpass', output='sos')
        self.sos_b = sig.butter(3, [ab, bc], 'bandpass', output='sos')
        self.sos_c = sig.butter(3, [bc, cd], 'bandpass', output='sos')
        self.sos_d = sig.butter(4, cd, 'highpass', output='sos')

        # Radar band
        lo_r, hi_r = self._f(900) / ny, self._f(4500) / ny
        self.sos_rd = sig.butter(2, [lo_r, hi_r], 'bandpass', output='sos')

        # DT990 notch
        if abs(c["dt990_db"]) > 0.1:
            self.notch_b, self.notch_a = make_peak_eq(
                self._f(c["dt990_hz"]), c["dt990_db"], c["dt990_q"], self.sr)
            self.has_notch = True
        else:
            self.has_notch = False

        # Gains
        self.gains = [10 ** (c[f"gain_{b}_db"] / 20) for b in "abcd"]
        self.g_out = 10 ** (c["output_gain_db"] / 20)
        self.g_gate = 10 ** (c["gate_db"] / 20)

        # Comp
        self.ct = [10 ** (c[f"comp_{b}_thresh"] / 20) for b in "abcd"]
        self.cr = [c[f"comp_{b}_ratio"] for b in "abcd"]

        self._zi()

    def _zi(self):
        z = lambda s: np.zeros((s.shape[0], 2))
        self.zi_hp = [z(self.sos_hp), z(self.sos_hp)]
        self.zi_lp = [z(self.sos_lp), z(self.sos_lp)]
        self.zi_a = [z(self.sos_a), z(self.sos_a)]
        self.zi_b = [z(self.sos_b), z(self.sos_b)]
        self.zi_c = [z(self.sos_c), z(self.sos_c)]
        self.zi_d = [z(self.sos_d), z(self.sos_d)]
        self.zi_rd = [z(self.sos_rd), z(self.sos_rd)]
        if self.has_notch:
            self.zi_notch = [sig.lfilter_zi(self.notch_b, self.notch_a) * 0,
                             sig.lfilter_zi(self.notch_b, self.notch_a) * 0]

    def rebuild(self):
        self._build()

    def _soft_compress(self, x, band_idx, ch_idx):
        """Soft-Knee Compression mit Makeup Gain."""
        ratio = self.cr[band_idx]
        if ratio <= 1.01:
            return x
        thresh = self.ct[band_idx]
        pk = max(float(np.max(np.abs(x))), 1e-10)
        env = self.comp_env[band_idx][ch_idx]
        coeff = 0.001 if pk > env else 0.035
        env += coeff * (pk - env)
        env = max(env, 1e-10)
        self.comp_env[band_idx][ch_idx] = env
        if env > thresh:
            over_db = 20 * math.log10(env / thresh)
            # Soft knee: sanfter Übergang
            knee_db = min(over_db, 6.0)
            soft = over_db - knee_db * 0.3
            reduce_db = soft * (1 - 1 / ratio)
            g = 10 ** (-reduce_db / 20)
            makeup = 10 ** (reduce_db * 0.45 / 20)
            return x * g * makeup
        return x

    def _transient_shape(self, x, band_idx, ch_idx):
        """Transient Enhancement + Sustained Reduction."""
        sens = self.cfg["transient_sensitivity"]
        kill = self.cfg["sustain_kill"]
        if sens < 0.01 and kill < 0.01:
            return x

        ratio, _, _ = self.td[band_idx][ch_idx].process(x)

        # Update display (average of L/R)
        if ch_idx == 0:
            self.t_ratios[band_idx] = ratio

        if ratio > 1.8:
            # TRANSIENT DETECTED → Boost
            boost = 1.0 + (ratio - 1.8) * sens * 0.5
            boost = min(boost, 1.0 + sens)
            return x * boost
        elif ratio < 0.7:
            # SUSTAINED SOUND → Duck
            duck = 1.0 - kill * (0.7 - ratio) * 2
            duck = max(duck, 1.0 - kill)
            return x * duck
        return x

    def process(self, data):
        if self.muted:
            return np.zeros_like(data)
        if self.bypass:
            return data.copy()

        ch = min(data.shape[1] if data.ndim > 1 else 1, 2)
        C = [data[:, 0].copy(), data[:, 1].copy()] if ch >= 2 else [data.flatten().copy()] * 2

        for i in range(2):
            x = C[i]

            # Pre-filter
            x, self.zi_hp[i] = sig.sosfilt(self.sos_hp, x, zi=self.zi_hp[i])
            x, self.zi_lp[i] = sig.sosfilt(self.sos_lp, x, zi=self.zi_lp[i])

            # Split into 4 bands
            a, self.zi_a[i] = sig.sosfilt(self.sos_a, x, zi=self.zi_a[i])
            b, self.zi_b[i] = sig.sosfilt(self.sos_b, x, zi=self.zi_b[i])
            c, self.zi_c[i] = sig.sosfilt(self.sos_c, x, zi=self.zi_c[i])
            d, self.zi_d[i] = sig.sosfilt(self.sos_d, x, zi=self.zi_d[i])

            bands = [a, b, c, d]

            # Per-band: Gain → Transient Shape → Compress
            for j in range(4):
                bands[j] = bands[j] * self.gains[j]
                bands[j] = self._transient_shape(bands[j], j, i)
                bands[j] = self._soft_compress(bands[j], j, i)

            # Recombine
            x = bands[0] + bands[1] + bands[2] + bands[3]

            # DT990 Notch
            if self.has_notch:
                x, self.zi_notch[i] = sig.lfilter(self.notch_b, self.notch_a, x, zi=self.zi_notch[i])

            # Radar
            r, self.zi_rd[i] = sig.sosfilt(self.sos_rd, x, zi=self.zi_rd[i])
            rms = float(np.sqrt(np.mean(r * r)))
            if i == 0:
                self.step_L = rms
            else:
                self.step_R = rms

            C[i] = x

        L, R = C

        # Noise Gate
        level = math.sqrt(float(np.mean(L * L)) + float(np.mean(R * R)))
        if level < self.g_gate:
            L *= 0.01
            R *= 0.01

        # Spatial
        w = self.cfg["spatial_width"]
        if abs(w - 1) > 0.01:
            m, s = (L + R) * 0.5, (L - R) * 0.5 * w
            L, R = m + s, m - s

        # Output + Soft Limiter
        L = np.tanh(L * self.g_out)
        R = np.tanh(R * self.g_out)

        # Radar
        t = self.step_L + self.step_R
        if t > 0.0003:
            self.peak_dir = (self.step_R - self.step_L) / t
            self.step_pwr = min(1.0, t * 25)
        else:
            self.step_pwr *= 0.82

        out = np.column_stack([L, R])
        if data.ndim > 1 and data.shape[1] > 2:
            out = np.column_stack([out, np.zeros((len(L), data.shape[1] - 2), dtype=np.float32)])
        return out.astype(np.float32)


# ============================================================
# RADAR
# ============================================================
class Radar:
    SZ = 360

    def __init__(self):
        self.cx, self.cy = self.SZ // 2, self.SZ // 2
        self.r = self.SZ // 2 - 40
        self.trail = deque(maxlen=30)

    def draw(self, proc, cfg):
        img = np.zeros((self.SZ, self.SZ, 3), dtype=np.uint8)

        # Background
        cv2.circle(img, (self.cx, self.cy), self.r, (20, 22, 28), -1)
        cv2.circle(img, (self.cx, self.cy), self.r, (45, 50, 60), 2)
        cv2.circle(img, (self.cx, self.cy), self.r // 2, (30, 33, 40), 1)
        cv2.circle(img, (self.cx, self.cy), self.r // 4, (25, 28, 35), 1)
        cv2.line(img, (self.cx - 8, self.cy), (self.cx + 8, self.cy), (45, 50, 60), 1)
        cv2.line(img, (self.cx, self.cy - 8), (self.cx, self.cy + 8), (45, 50, 60), 1)
        cv2.putText(img, "L", (4, self.cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (70, 75, 85), 1)
        cv2.putText(img, "R", (self.SZ - 16, self.cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (70, 75, 85), 1)

        d, pw = proc.peak_dir, proc.step_pwr

        if pw > 0.04:
            self.trail.append((d, pw))
            for idx, (dd, pp) in enumerate(self.trail):
                a = (idx + 1) / len(self.trail)
                cv2.circle(img, (int(self.cx + dd * self.r * 0.85), self.cy),
                           int(2 + pp * 8 * a), (0, int(50 * a), 0), -1)
            px = int(self.cx + d * self.r * 0.85)
            sz = int(4 + pw * 16)
            col = (0, 60, 255) if pw > 0.7 else (0, 180, 255) if pw > 0.4 else (0, 200, 100)
            cv2.circle(img, (px, self.cy), sz, col, -1)
            cv2.line(img, (self.cx, self.cy), (px, self.cy), col, 2)

        # Bypass
        if proc.bypass:
            cv2.putText(img, "BYPASS", (self.SZ // 2 - 50, self.SZ // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Band info
        y = 16
        cv2.putText(img, f"[TRANSIENT ENGINE v5]", (6, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 200, 120), 1)
        y += 14
        cv2.putText(img, f"A:{cfg['gain_a_db']:+.0f} B:{cfg['gain_b_db']:+.0f} C:{cfg['gain_c_db']:+.0f} D:{cfg['gain_d_db']:+.0f} | SPA:{cfg['spatial_width']:.1f}",
                    (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (120, 125, 140), 1)
        y += 14
        cv2.putText(img, f"TRANS:{cfg['transient_sensitivity']:.1f} KILL:{cfg['sustain_kill']:.1f} GAIN:{cfg['output_gain_db']:+.0f}",
                    (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (120, 125, 140), 1)

        # Transient meters per band
        y += 16
        labels = ["A:Thump", "B:Mud  ", "C:Step ", "D:Gun  "]
        colors = [(0, 200, 100), (0, 80, 180), (0, 255, 220), (0, 60, 200)]
        for j in range(4):
            tr = min(proc.t_ratios[j], 4.0)
            bw = int(tr / 4.0 * 120)
            cv2.putText(img, labels[j], (6, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (90, 95, 110), 1)
            cv2.rectangle(img, (70, y), (70 + bw, y + 10), colors[j], -1)
            cv2.rectangle(img, (70, y), (190, y + 10), (40, 45, 55), 1)
            # Threshold line
            tl = int(1.8 / 4.0 * 120)
            cv2.line(img, (70 + tl, y), (70 + tl, y + 10), (0, 0, 180), 1)
            y += 14

        # Power bar
        bw = int(pw * (self.SZ - 20))
        bc = (0, 200, 100) if pw < 0.4 else (0, 200, 255) if pw < 0.7 else (0, 60, 255)
        cv2.rectangle(img, (10, self.SZ - 16), (10 + bw, self.SZ - 6), bc, -1)
        cv2.rectangle(img, (10, self.SZ - 16), (self.SZ - 10, self.SZ - 6), (40, 45, 55), 1)
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
        if inp == 0 and out == 0:
            continue
        tag = "[IN] " if inp > 0 and out == 0 else "[OUT]" if out > 0 and inp == 0 else "[I/O]"
        mk = ""
        if i == sd.default.device[0]: mk += " <<IN"
        if i == sd.default.device[1]: mk += " <<OUT"
        print(f"  {i:3d} {tag} {d['name'][:48]:<48} {int(d['default_samplerate'])}Hz{mk}")
    print("  " + "-" * 70)
    return devs


def auto_find_capture(devs):
    for i, d in enumerate(devs):
        if d['max_input_channels'] >= 2 and d['default_samplerate'] <= 48000:
            name_l = d['name'].lower()
            if 'hdmi' in name_l and ('live streamer' in name_l or 'avermedia' in name_l):
                return i
    for i, d in enumerate(devs):
        if d['max_input_channels'] >= 2:
            for kw in ["avermedia", "gc571", "game capture"]:
                if kw in d['name'].lower():
                    return i
    return None


def find_device_by_name(devs, name, need_input=True):
    if not name:
        return None
    for i, d in enumerate(devs):
        if name in d['name']:
            if need_input and d['max_input_channels'] > 0:
                return i
            if not need_input and d['max_output_channels'] > 0:
                return i
    return None


# ============================================================
# MAIN
# ============================================================
def main():
    if sd is None:
        sys.exit(1)

    parser = argparse.ArgumentParser(description="eRayz Audio — Transient Engine v5")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--input", type=int, default=None)
    parser.add_argument("--output", type=int, default=None)
    parser.add_argument("--no-radar", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    devs = list_devices()

    print()
    print("  " + "=" * 55)
    print("  eRayz Audio — TRANSIENT ENGINE v5")
    print("  Per-Band Transient Detection | Sustained Killer")
    print("  DT 990 Pro | Raw HDMI | 48kHz/256")
    print("  " + "=" * 55)

    if args.list:
        return

    # Device selection
    in_dev = args.input
    if in_dev is None and cfg["input_name"]:
        in_dev = find_device_by_name(devs, cfg["input_name"], need_input=True)
    if in_dev is None and cfg["input_device"] is not None:
        idx = cfg["input_device"]
        if idx < len(devs) and devs[idx]['max_input_channels'] > 0:
            in_dev = idx
    if in_dev is None:
        auto = auto_find_capture(devs)
        if auto is not None:
            in_dev = auto
            print(f"\n  Auto: [{auto}] {devs[auto]['name']}")
        else:
            in_dev = sd.default.device[0]

    out_dev = args.output
    if out_dev is None and cfg["output_name"]:
        out_dev = find_device_by_name(devs, cfg["output_name"], need_input=False)
    if out_dev is None and cfg["output_device"] is not None:
        idx = cfg["output_device"]
        if idx < len(devs) and devs[idx]['max_output_channels'] > 0:
            out_dev = idx
    if out_dev is None:
        out_dev = sd.default.device[1]

    cfg["input_device"] = in_dev
    cfg["output_device"] = out_dev
    cfg["input_name"] = devs[in_dev]['name']
    cfg["output_name"] = devs[out_dev]['name']

    if args.no_radar:
        cfg["show_radar"] = False

    sr = 48000
    cfg["samplerate"] = sr
    ch = min(devs[in_dev]['max_input_channels'], 2)
    cfg["channels"] = ch
    bs = cfg["blocksize"]

    print(f"\n  IN:  [{in_dev}] {devs[in_dev]['name']}")
    print(f"  OUT: [{out_dev}] {devs[out_dev]['name']}")
    print(f"  {sr}Hz | Stereo | {bs} samples (~{bs / sr * 1000:.1f}ms)")
    print(f"\n  Band A (Thump  100-{cfg['xover_ab']}Hz):  {cfg['gain_a_db']:+.0f}dB  Comp {cfg['comp_a_ratio']:.0f}:1")
    print(f"  Band B (Mud  {cfg['xover_ab']}-{cfg['xover_bc']}Hz):   {cfg['gain_b_db']:+.0f}dB  KILL")
    print(f"  Band C (Steps {cfg['xover_bc']}-{cfg['xover_cd']}Hz): {cfg['gain_c_db']:+.0f}dB  Comp {cfg['comp_c_ratio']:.0f}:1")
    print(f"  Band D (Gun   {cfg['xover_cd']}Hz+):    {cfg['gain_d_db']:+.0f}dB  Limit {cfg['comp_d_ratio']:.0f}:1")
    print(f"\n  Transient: {cfg['transient_sensitivity']:.1f}  Sustain Kill: {cfg['sustain_kill']:.1f}")
    print(f"  Spatial: {cfg['spatial_width']:.1f}x  Gain: {cfg['output_gain_db']:+.0f}dB  DT990: {cfg['dt990_db']:.0f}dB")
    print(f"\n  [Q/W]A [E/R]B [A/S]C [D/F]D [T/Z]Trans [J/K]Kill")
    print(f"  [G/H]Gain [X/C]Spa [B]Bypass [M]Mute [V]Radar [ESC]Quit")
    print("  " + "=" * 55)

    proc = TransientEngine(cfg)
    radar = Radar() if (cfg["show_radar"] and cv2) else None

    # Stream
    print("\n  Starte Stream...")
    duplex = None
    audio_ok = False

    def duplex_cb(indata, outdata, frames, ti, status):
        try:
            outdata[:] = proc.process(indata)[:outdata.shape[0], :outdata.shape[1]]
        except Exception:
            outdata[:] = indata

    try:
        duplex = sd.Stream(
            device=(in_dev, out_dev), samplerate=sr, blocksize=bs,
            channels=ch, dtype='float32', callback=duplex_cb, latency=0.02,
        )
        duplex.start()
        audio_ok = True
        print("  DUPLEX OK\n")
    except Exception as e1:
        print(f"  Duplex fail: {e1}")
        print("  Fallback: Ring-Buffer...")

        class _RB:
            def __init__(self, cap, nch):
                self.buf = np.zeros((cap, nch), dtype=np.float32)
                self.cap, self.wp, self.rp, self.lv = cap, 0, 0, 0
                self.lk = threading.Lock()
            def write(self, d):
                n = len(d)
                with self.lk:
                    e = self.wp + n
                    if e <= self.cap: self.buf[self.wp:e] = d
                    else:
                        p = self.cap - self.wp
                        self.buf[self.wp:] = d[:p]; self.buf[:n - p] = d[p:]
                    self.wp = e % self.cap; self.lv = min(self.lv + n, self.cap)
            def read(self, n):
                with self.lk:
                    o = np.zeros((n, self.buf.shape[1]), dtype=np.float32)
                    av = min(self.lv, n)
                    if av > 0:
                        e = self.rp + av
                        if e <= self.cap: o[:av] = self.buf[self.rp:e]
                        else:
                            p = self.cap - self.rp
                            o[:p] = self.buf[self.rp:]; o[p:av] = self.buf[:av - p]
                        self.rp = (self.rp + av) % self.cap; self.lv -= av
                    return o
            def fill(self):
                return self.lv / self.cap

        rb = _RB(int(sr * 0.3), ch)
        try:
            _si = sd.InputStream(device=in_dev, samplerate=sr, blocksize=bs, channels=ch, dtype='float32')
            _so = sd.OutputStream(device=out_dev, samplerate=sr, blocksize=bs, channels=ch, dtype='float32')
            _si.start(); _so.start()
            rb.write(np.zeros((int(sr * 0.12), ch), dtype=np.float32))

            def _ti():
                while True:
                    try: d, _ = _si.read(bs); rb.write(proc.process(d))
                    except: pass
            def _to():
                while True:
                    try:
                        _so.write(rb.read(bs))
                        f = rb.fill()
                        if f > 0.75: rb.read(bs)
                        elif f < 0.2: time.sleep(bs / sr * 0.25)
                    except: pass

            threading.Thread(target=_ti, daemon=True).start()
            threading.Thread(target=_to, daemon=True).start()
            audio_ok = True
            print("  RING-BUFFER OK\n")
        except Exception as e2:
            print(f"  FEHLER: {e2}")
            return

    if not audio_ok:
        return

    # UI Loop
    try:
        while True:
            if radar and cfg["show_radar"] and cv2:
                cv2.imshow("eRayz Radar", radar.draw(proc, cfg))

            key = (cv2.waitKey(33) & 0xFF) if (cv2 and cfg["show_radar"]) else (time.sleep(0.033) or 255)

            if key == 27: break
            elif key == ord('q'): cfg["gain_a_db"] = max(-15, cfg["gain_a_db"] - 1); proc.rebuild(); print(f"  A: {cfg['gain_a_db']:+.0f}dB")
            elif key == ord('w'): cfg["gain_a_db"] = min(22, cfg["gain_a_db"] + 1); proc.rebuild(); print(f"  A: {cfg['gain_a_db']:+.0f}dB")
            elif key == ord('e'): cfg["gain_b_db"] = max(-18, cfg["gain_b_db"] - 1); proc.rebuild(); print(f"  B: {cfg['gain_b_db']:+.0f}dB")
            elif key == ord('r'): cfg["gain_b_db"] = min(10, cfg["gain_b_db"] + 1); proc.rebuild(); print(f"  B: {cfg['gain_b_db']:+.0f}dB")
            elif key == ord('a'): cfg["gain_c_db"] = max(-10, cfg["gain_c_db"] - 1); proc.rebuild(); print(f"  C: {cfg['gain_c_db']:+.0f}dB")
            elif key == ord('s'): cfg["gain_c_db"] = min(25, cfg["gain_c_db"] + 1); proc.rebuild(); print(f"  C: {cfg['gain_c_db']:+.0f}dB")
            elif key == ord('d'): cfg["gain_d_db"] = max(-20, cfg["gain_d_db"] - 1); proc.rebuild(); print(f"  D: {cfg['gain_d_db']:+.0f}dB")
            elif key == ord('f'): cfg["gain_d_db"] = min(5, cfg["gain_d_db"] + 1); proc.rebuild(); print(f"  D: {cfg['gain_d_db']:+.0f}dB")
            elif key == ord('t'): cfg["transient_sensitivity"] = max(0, round(cfg["transient_sensitivity"] - 0.1, 1)); print(f"  Trans: {cfg['transient_sensitivity']:.1f}")
            elif key == ord('z'): cfg["transient_sensitivity"] = min(3.0, round(cfg["transient_sensitivity"] + 0.1, 1)); print(f"  Trans: {cfg['transient_sensitivity']:.1f}")
            elif key == ord('j'): cfg["sustain_kill"] = max(0, round(cfg["sustain_kill"] - 0.05, 2)); print(f"  Kill: {cfg['sustain_kill']:.2f}")
            elif key == ord('k'): cfg["sustain_kill"] = min(1.0, round(cfg["sustain_kill"] + 0.05, 2)); print(f"  Kill: {cfg['sustain_kill']:.2f}")
            elif key == ord('g'): cfg["output_gain_db"] = max(-6, cfg["output_gain_db"] - 1); proc.rebuild(); print(f"  Gain: {cfg['output_gain_db']:+.0f}dB")
            elif key == ord('h'): cfg["output_gain_db"] = min(12, cfg["output_gain_db"] + 1); proc.rebuild(); print(f"  Gain: {cfg['output_gain_db']:+.0f}dB")
            elif key == ord('x'): cfg["spatial_width"] = max(0.5, round(cfg["spatial_width"] - 0.1, 1)); print(f"  Spa: {cfg['spatial_width']:.1f}x")
            elif key == ord('c'): cfg["spatial_width"] = min(3.0, round(cfg["spatial_width"] + 0.1, 1)); print(f"  Spa: {cfg['spatial_width']:.1f}x")
            elif key == ord('b'): proc.bypass = not proc.bypass; print(f"  {'>>> BYPASS <<<' if proc.bypass else '>>> TRANSIENT ENGINE AKTIV <<<'}")
            elif key == ord('m'): proc.muted = not proc.muted; print(f"  {'MUTED' if proc.muted else 'UNMUTED'}")
            elif key == ord('v'):
                cfg["show_radar"] = not cfg["show_radar"]
                if not cfg["show_radar"] and cv2: cv2.destroyAllWindows()
                print(f"  Radar: {'EIN' if cfg['show_radar'] else 'AUS'}")
            elif key == ord('p'):
                print(f"\n  === TRANSIENT ENGINE v5 SETTINGS ===")
                for k, v in sorted(cfg.items()):
                    if k not in ("input_device", "output_device", "input_name", "output_name"):
                        print(f"    {k}: {v}")
                print()

    except KeyboardInterrupt:
        print("\n  Stop.")
    finally:
        if duplex:
            duplex.stop()
            duplex.close()
        save_config(cfg)
        print("  Config gespeichert. GG.")
        if cv2:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
