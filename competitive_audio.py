"""
eRayz Audio — MULTIBAND PRO
=============================
4-Band Frequency Splitting | Per-Band Compression
Transient Enhancement | DT 990 Pro | Raw HDMI

ERSTER START:  python competitive_audio.py --input 18 --output 14
DANACH NUR:    python competitive_audio.py

STEUERUNG:
  Q/W    Band A (Steps Low) Gain -/+
  E/R    Band B (Mud) Gain -/+
  A/S    Band C (Steps Detail) Gain -/+
  D/F    Band D (Gunfire) Gain -/+
  T/Z    Master Compression -/+
  G/H    Output Gain -/+
  X/C    Spatial -/+
  B      BYPASS (A/B Vergleich)
  M      Mute
  V      Radar an/aus
  P      Settings anzeigen
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
    "preset": "multiband_pro",
    # ══════════════════════════════════════════════════════════
    # MULTIBAND PRO — 4-Band Split Processing
    #
    # Band A: 80-300Hz   → Footstep Body (Boost + Compress)
    # Band B: 300-1500Hz → Mud Zone (Cut + Gate)
    # Band C: 1500-5500Hz→ Footstep Detail (Boost + Heavy Compress)
    # Band D: 5500-11kHz → Gunfire/Treble (Cut + Limit)
    # ══════════════════════════════════════════════════════════
    #
    # Crossover Frequencies
    "xover_1": 300, "xover_2": 1500, "xover_3": 5500,
    "highpass_hz": 80, "lowpass_hz": 11000,
    #
    # Band A: Footstep Body
    "band_a_gain_db": 9.0,
    "band_a_comp_ratio": 3.5,
    "band_a_comp_thresh": -18.0,
    #
    # Band B: Mud Zone — Weniger schneiden, Raum-Info behalten fuer Hinter-Dir
    "band_b_gain_db": -2.0,
    "band_b_comp_ratio": 2.0,
    "band_b_comp_thresh": -15.0,
    #
    # Band C: Footstep Detail — Der Geldband. Alles hoerbar machen.
    "band_c_gain_db": 14.0,
    "band_c_comp_ratio": 5.5,
    "band_c_comp_thresh": -22.0,
    #
    # Band D: Gunfire/Treble — Runterziehen + Limiten
    "band_d_gain_db": -8.0,
    "band_d_comp_ratio": 10.0,
    "band_d_comp_thresh": -12.0,
    #
    # DT 990 Pro: Notch bei 8kHz
    "dt990_notch_hz": 8000, "dt990_notch_db": -5.0, "dt990_notch_q": 3.0,
    #
    # Transient Enhancer: Verstaerkt Attack von Steps
    "transient_attack": 0.6,  # 0=aus, 1=maximal
    #
    # Spatial — Breiter fuer Hinter-Dir Wahrnehmung
    "spatial_width": 2.0,
    #
    # Master
    "output_gain_db": 4.0,
    "gate_db": -52.0,
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
    A = 10 ** (db / 40.0)
    w0 = 2 * np.pi * fc / fs
    al = np.sin(w0) / (2 * Q)
    b0, b1, b2 = 1 + al*A, -2*np.cos(w0), 1 - al*A
    a0, a1, a2 = 1 + al/A, -2*np.cos(w0), 1 - al/A
    return np.array([b0/a0, b1/a0, b2/a0]), np.array([1.0, a1/a0, a2/a0])


# ============================================================
# MULTIBAND PROCESSOR
# ============================================================
class MultibandProcessor:
    """Splittet Audio in 4 Baender, verarbeitet jedes einzeln."""

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
        # Per-band compressor envelopes [bandA, bandB, bandC, bandD] x [L, R]
        self.comp_env = [[1e-10]*2 for _ in range(4)]
        # Transient detector state
        self.prev_env_c = [0.0, 0.0]
        self._build()

    def _f(self, hz):
        return max(20.0, min(float(hz), self.ny * 0.95))

    def _build(self):
        sr, ny, c = self.sr, self.ny, self.cfg

        # Highpass + Lowpass
        self.sos_hp = sig.butter(4, self._f(c["highpass_hz"])/ny, 'highpass', output='sos')
        self.sos_lp = sig.butter(3, self._f(c["lowpass_hz"])/ny, 'lowpass', output='sos')

        # Crossover filters (Linkwitz-Riley 4th order = 2x Butterworth 2nd order)
        x1, x2, x3 = self._f(c["xover_1"])/ny, self._f(c["xover_2"])/ny, self._f(c["xover_3"])/ny

        # Band A: HP(80) to LP(xover_1)
        self.sos_a_lp = sig.butter(4, x1, 'lowpass', output='sos')
        # Band B: HP(xover_1) to LP(xover_2)
        self.sos_b_bp = sig.butter(3, [x1, x2], 'bandpass', output='sos')
        # Band C: HP(xover_2) to LP(xover_3)
        self.sos_c_bp = sig.butter(3, [x2, x3], 'bandpass', output='sos')
        # Band D: HP(xover_3) to LP(11k)
        self.sos_d_hp = sig.butter(4, x3, 'highpass', output='sos')

        # Radar detection band
        lo_r, hi_r = self._f(1500)/ny, self._f(5000)/ny
        self.sos_rd = sig.butter(2, [lo_r, hi_r], 'bandpass', output='sos')

        # DT990 notch
        if abs(c["dt990_notch_db"]) > 0.1:
            self.notch_b, self.notch_a = make_peak_eq(
                self._f(c["dt990_notch_hz"]), c["dt990_notch_db"], c["dt990_notch_q"], sr)
            self.has_notch = True
        else:
            self.has_notch = False

        # Gains
        self.g_a = 10 ** (c["band_a_gain_db"] / 20)
        self.g_b = 10 ** (c["band_b_gain_db"] / 20)
        self.g_c = 10 ** (c["band_c_gain_db"] / 20)
        self.g_d = 10 ** (c["band_d_gain_db"] / 20)
        self.g_out = 10 ** (c["output_gain_db"] / 20)
        self.g_gate = 10 ** (c["gate_db"] / 20)

        # Comp thresholds
        self.ct = [10 ** (c[f"band_{b}_comp_thresh"] / 20) for b in "abcd"]
        self.cr = [c[f"band_{b}_comp_ratio"] for b in "abcd"]

        self._zi()

    def _zi(self):
        z = lambda s: np.zeros((s.shape[0], 2))
        self.zi_hp = [z(self.sos_hp), z(self.sos_hp)]
        self.zi_lp = [z(self.sos_lp), z(self.sos_lp)]
        self.zi_a = [z(self.sos_a_lp), z(self.sos_a_lp)]
        self.zi_b = [z(self.sos_b_bp), z(self.sos_b_bp)]
        self.zi_c = [z(self.sos_c_bp), z(self.sos_c_bp)]
        self.zi_d = [z(self.sos_d_hp), z(self.sos_d_hp)]
        self.zi_rd = [z(self.sos_rd), z(self.sos_rd)]
        if self.has_notch:
            self.zi_notch = [sig.lfilter_zi(self.notch_b, self.notch_a)*0,
                             sig.lfilter_zi(self.notch_b, self.notch_a)*0]

    def rebuild(self):
        self._build()

    def _compress_band(self, x, band_idx, ch_idx):
        """Per-Band Kompression."""
        ratio = self.cr[band_idx]
        if ratio <= 1.01:
            return x
        thresh = self.ct[band_idx]
        pk = max(np.max(np.abs(x)), 1e-10)
        env = self.comp_env[band_idx][ch_idx]
        # Fast attack, medium release
        coeff = 0.002 if pk > env else 0.04
        env += coeff * (pk - env)
        env = max(env, 1e-10)
        self.comp_env[band_idx][ch_idx] = env
        if env > thresh:
            over_db = 20 * math.log10(env / thresh)
            reduce_db = over_db * (1 - 1/ratio)
            g = 10 ** (-reduce_db / 20)
            # Makeup gain: halber Reduce zurueck
            makeup = 10 ** (reduce_db * 0.5 / 20)
            return x * g * makeup
        return x

    def _transient_enhance(self, x, ch_idx):
        """Verstaerkt den Attack von Transienten (Schritte knackiger)."""
        amt = self.cfg["transient_attack"]
        if amt < 0.05:
            return x
        env = np.sqrt(np.mean(x * x))
        delta = env - self.prev_env_c[ch_idx]
        self.prev_env_c[ch_idx] = env
        # Positive delta = Attack (Signal wird lauter) → boosten
        if delta > 0.001:
            boost = 1.0 + delta * amt * 15
            boost = min(boost, 1.0 + amt)  # Cap
            return x * boost
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
            a, self.zi_a[i] = sig.sosfilt(self.sos_a_lp, x, zi=self.zi_a[i])
            b, self.zi_b[i] = sig.sosfilt(self.sos_b_bp, x, zi=self.zi_b[i])
            c, self.zi_c[i] = sig.sosfilt(self.sos_c_bp, x, zi=self.zi_c[i])
            d, self.zi_d[i] = sig.sosfilt(self.sos_d_hp, x, zi=self.zi_d[i])

            # Per-band: Gain → Compress
            a = self._compress_band(a * self.g_a, 0, i)
            b = self._compress_band(b * self.g_b, 1, i)
            c = self._compress_band(c * self.g_c, 2, i)
            d = self._compress_band(d * self.g_d, 3, i)

            # Transient Enhancement auf Band C (Step Detail)
            c = self._transient_enhance(c, i)

            # Recombine
            x = a + b + c + d

            # DT990 Notch
            if self.has_notch:
                x, self.zi_notch[i] = sig.lfilter(self.notch_b, self.notch_a, x, zi=self.zi_notch[i])

            # Radar detection
            r, self.zi_rd[i] = sig.sosfilt(self.sos_rd, x, zi=self.zi_rd[i])
            rms = float(np.sqrt(np.mean(r*r)))
            if i == 0: self.step_L = rms
            else: self.step_R = rms

            C[i] = x

        L, R = C

        # Noise Gate
        level = np.sqrt(np.mean(L*L) + np.mean(R*R))
        if level < self.g_gate:
            L *= 0.02; R *= 0.02

        # Spatial
        w = self.cfg["spatial_width"]
        if abs(w - 1) > 0.01:
            m, s = (L+R)*0.5, (L-R)*0.5*w
            L, R = m+s, m-s

        # Output + Soft Limiter
        L = np.tanh(L * self.g_out)
        R = np.tanh(R * self.g_out)

        # Radar data
        t = self.step_L + self.step_R
        if t > 0.0003:
            self.peak_dir = (self.step_R - self.step_L) / t
            self.step_pwr = min(1.0, t * 25)
        else:
            self.step_pwr *= 0.82

        out = np.column_stack([L, R])
        if data.ndim > 1 and data.shape[1] > 2:
            out = np.column_stack([out, np.zeros((len(L), data.shape[1]-2), dtype=np.float32)])
        return out.astype(np.float32)


# ============================================================
# RADAR
# ============================================================
class Radar:
    SZ = 320
    def __init__(self):
        self.cx, self.cy = self.SZ//2, self.SZ//2
        self.r = self.SZ//2 - 35
        self.trail = deque(maxlen=25)

    def draw(self, d, pw, cfg, bypass):
        img = np.zeros((self.SZ, self.SZ, 3), dtype=np.uint8)
        cv2.circle(img, (self.cx, self.cy), self.r, (25,25,30), -1)
        cv2.circle(img, (self.cx, self.cy), self.r, (50,50,60), 2)
        cv2.circle(img, (self.cx, self.cy), self.r//2, (35,35,40), 1)
        cv2.line(img, (self.cx-10,self.cy), (self.cx+10,self.cy), (50,50,60), 1)
        cv2.line(img, (self.cx,self.cy-10), (self.cx,self.cy+10), (50,50,60), 1)
        cv2.putText(img, "L", (6,self.cy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80,80,90), 1)
        cv2.putText(img, "R", (self.SZ-18,self.cy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80,80,90), 1)

        if pw > 0.04:
            self.trail.append((d, pw))
            for i, (dd, pp) in enumerate(self.trail):
                a = (i+1)/len(self.trail)
                cv2.circle(img, (int(self.cx+dd*self.r*0.85),self.cy), int(2+pp*10*a), (0,int(60*a),0), -1)
            px, sz = int(self.cx+d*self.r*0.85), int(4+pw*18)
            col = (0,80,255) if pw>0.7 else (0,200,255) if pw>0.4 else (0,200,120)
            cv2.circle(img, (px,self.cy), sz, col, -1)
            cv2.line(img, (self.cx,self.cy), (px,self.cy), col, 2)

        if bypass:
            cv2.putText(img, ">>> BYPASS <<<", (80,self.SZ//2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        cv2.putText(img, f"[MULTIBAND PRO] A:{cfg['band_a_gain_db']:.0f} B:{cfg['band_b_gain_db']:.0f} C:{cfg['band_c_gain_db']:.0f} D:{cfg['band_d_gain_db']:.0f}",
                    (6,16), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0,180,100), 1)
        cv2.putText(img, f"DT990:{cfg['dt990_notch_db']:.0f} SPA:{cfg['spatial_width']:.1f} TRANS:{cfg['transient_attack']:.1f}",
                    (6,32), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (130,130,140), 1)
        cv2.putText(img, f"COMP A:{cfg['band_a_comp_ratio']:.0f}:1 C:{cfg['band_c_comp_ratio']:.0f}:1 GAIN:{cfg['output_gain_db']:.0f}dB",
                    (6,48), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (130,130,140), 1)

        bw = int(pw*(self.SZ-20))
        bc = (0,180,100) if pw<0.4 else (0,200,255) if pw<0.7 else (0,80,255)
        cv2.rectangle(img, (10,self.SZ-16), (10+bw,self.SZ-6), bc, -1)
        cv2.rectangle(img, (10,self.SZ-16), (self.SZ-10,self.SZ-6), (50,50,60), 1)
        return img


# ============================================================
# DEVICES
# ============================================================
def list_devices():
    devs = sd.query_devices()
    print("\n  Aktive Audio-Geraete:")
    print("  " + "-" * 70)
    for i, d in enumerate(devs):
        inp, out = d['max_input_channels'], d['max_output_channels']
        if inp == 0 and out == 0:
            continue
        tag = "[IN] " if inp>0 and out==0 else "[OUT]" if out>0 and inp==0 else "[I/O]" if inp>0 else "[---]"
        mk = ""
        if i == sd.default.device[0]: mk += " <<IN"
        if i == sd.default.device[1]: mk += " <<OUT"
        print(f"  {i:3d} {tag} {d['name'][:48]:<48} {int(d['default_samplerate'])}Hz i:{inp} o:{out}{mk}")
    print("  " + "-" * 70)
    return devs


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


def auto_find_capture(devs):
    for i, d in enumerate(devs):
        if d['max_input_channels'] >= 2 and d['default_samplerate'] <= 48000:
            name_l = d['name'].lower()
            if 'hdmi' in name_l and ('live streamer' in name_l or 'avermedia' in name_l or 'capture' in name_l):
                return i
    for i, d in enumerate(devs):
        if d['max_input_channels'] >= 2:
            for kw in ["avermedia", "gc571", "game capture", "live gamer"]:
                if kw in d['name'].lower():
                    return i
    return None


# ============================================================
# MAIN
# ============================================================
def main():
    if sd is None:
        sys.exit(1)

    parser = argparse.ArgumentParser(description="eRayz Audio — Multiband Pro")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--input", type=int, default=None)
    parser.add_argument("--output", type=int, default=None)
    parser.add_argument("--no-radar", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    devs = list_devices()

    print()
    print("  " + "=" * 55)
    print("  eRayz Audio — MULTIBAND PRO")
    print("  4-Band Split | Per-Band Compression | Transient Shaper")
    print("  DT 990 Pro | Raw HDMI | 48kHz/256")
    print("  " + "=" * 55)

    if args.list:
        return

    in_dev = args.input
    if in_dev is None and cfg["input_name"]:
        in_dev = find_device_by_name(devs, cfg["input_name"], need_input=True)
        if in_dev is not None:
            print(f"\n  Gespeichertes Input: [{in_dev}] {devs[in_dev]['name']}")
    if in_dev is None and cfg["input_device"] is not None:
        idx = cfg["input_device"]
        if idx < len(devs) and devs[idx]['max_input_channels'] > 0:
            in_dev = idx
    if in_dev is None:
        auto = auto_find_capture(devs)
        if auto is not None:
            in_dev = auto
            print(f"\n  Auto-Erkannt: [{auto}] {devs[auto]['name']}")
        else:
            in_dev = sd.default.device[0]

    out_dev = args.output
    if out_dev is None and cfg["output_name"]:
        out_dev = find_device_by_name(devs, cfg["output_name"], need_input=False)
        if out_dev is not None:
            print(f"  Gespeichertes Output: [{out_dev}] {devs[out_dev]['name']}")
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
    print(f"  {sr}Hz | Stereo | {bs} samples (~{bs/sr*1000:.1f}ms)")
    print(f"\n  [MULTIBAND PRO]")
    print(f"  Band A (Steps Low  80-{cfg['xover_1']}Hz):  Gain:{cfg['band_a_gain_db']:+.0f}dB  Comp:{cfg['band_a_comp_ratio']:.0f}:1")
    print(f"  Band B (Mud    {cfg['xover_1']}-{cfg['xover_2']}Hz): Gain:{cfg['band_b_gain_db']:+.0f}dB  Comp:{cfg['band_b_comp_ratio']:.0f}:1")
    print(f"  Band C (Detail {cfg['xover_2']}-{cfg['xover_3']}Hz): Gain:{cfg['band_c_gain_db']:+.0f}dB  Comp:{cfg['band_c_comp_ratio']:.0f}:1")
    print(f"  Band D (Gun    {cfg['xover_3']}Hz+):     Gain:{cfg['band_d_gain_db']:+.0f}dB  Comp:{cfg['band_d_comp_ratio']:.0f}:1")
    print(f"  Transient: {cfg['transient_attack']:.1f}  Spatial: {cfg['spatial_width']:.1f}x  DT990: {cfg['dt990_notch_db']:.0f}dB")
    print(f"\n  [Q/W]BandA [E/R]BandB [A/S]BandC [D/F]BandD")
    print(f"  [T/Z]CompC [G/H]Gain [X/C]Spatial [B]Bypass [M]Mute [ESC]Quit")
    print("  " + "=" * 55)

    proc = MultibandProcessor(cfg)
    radar = Radar() if (cfg["show_radar"] and cv2) else None

    print("\n  Starte Duplex Stream...")
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
        print(f"  Duplex fehlgeschlagen: {e1}")
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
                        self.buf[self.wp:] = d[:p]; self.buf[:n-p] = d[p:]
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
                            o[:p] = self.buf[self.rp:]; o[p:av] = self.buf[:av-p]
                        self.rp = (self.rp + av) % self.cap; self.lv -= av
                    return o
            def fill(self): return self.lv / self.cap

        rb = _RB(int(sr * 0.3), ch)
        _run = [True]
        try:
            _si = sd.InputStream(device=in_dev, samplerate=sr, blocksize=bs, channels=ch, dtype='float32')
            _so = sd.OutputStream(device=out_dev, samplerate=sr, blocksize=bs, channels=ch, dtype='float32')
            _si.start(); _so.start()
            rb.write(np.zeros((int(sr*0.12), ch), dtype=np.float32))

            def _ti():
                while _run[0]:
                    try: d, _ = _si.read(bs); rb.write(proc.process(d))
                    except: pass
            def _to():
                while _run[0]:
                    try:
                        _so.write(rb.read(bs))
                        f = rb.fill()
                        if f > 0.75: rb.read(bs)
                        elif f < 0.2: time.sleep(bs/sr*0.25)
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

    try:
        while True:
            if radar and cfg["show_radar"] and cv2:
                cv2.imshow("eRayz Radar", radar.draw(proc.peak_dir, proc.step_pwr, cfg, proc.bypass))

            key = (cv2.waitKey(33) & 0xFF) if (cv2 and cfg["show_radar"]) else (time.sleep(0.033) or 255)

            if key == 27: break
            elif key == ord('q'): cfg["band_a_gain_db"] = max(-10, cfg["band_a_gain_db"]-1); proc.rebuild(); print(f"  Band A: {cfg['band_a_gain_db']:.0f}dB")
            elif key == ord('w'): cfg["band_a_gain_db"] = min(20, cfg["band_a_gain_db"]+1); proc.rebuild(); print(f"  Band A: {cfg['band_a_gain_db']:.0f}dB")
            elif key == ord('e'): cfg["band_b_gain_db"] = max(-15, cfg["band_b_gain_db"]-1); proc.rebuild(); print(f"  Band B: {cfg['band_b_gain_db']:.0f}dB")
            elif key == ord('r'): cfg["band_b_gain_db"] = min(10, cfg["band_b_gain_db"]+1); proc.rebuild(); print(f"  Band B: {cfg['band_b_gain_db']:.0f}dB")
            elif key == ord('a'): cfg["band_c_gain_db"] = max(-10, cfg["band_c_gain_db"]-1); proc.rebuild(); print(f"  Band C: {cfg['band_c_gain_db']:.0f}dB")
            elif key == ord('s'): cfg["band_c_gain_db"] = min(20, cfg["band_c_gain_db"]+1); proc.rebuild(); print(f"  Band C: {cfg['band_c_gain_db']:.0f}dB")
            elif key == ord('d'): cfg["band_d_gain_db"] = max(-15, cfg["band_d_gain_db"]-1); proc.rebuild(); print(f"  Band D: {cfg['band_d_gain_db']:.0f}dB")
            elif key == ord('f'): cfg["band_d_gain_db"] = min(5, cfg["band_d_gain_db"]+1); proc.rebuild(); print(f"  Band D: {cfg['band_d_gain_db']:.0f}dB")
            elif key == ord('t'): cfg["band_c_comp_ratio"] = max(1, cfg["band_c_comp_ratio"]-1); proc.rebuild(); print(f"  Comp C: {cfg['band_c_comp_ratio']:.0f}:1")
            elif key == ord('z'): cfg["band_c_comp_ratio"] = min(12, cfg["band_c_comp_ratio"]+1); proc.rebuild(); print(f"  Comp C: {cfg['band_c_comp_ratio']:.0f}:1")
            elif key == ord('g'): cfg["output_gain_db"] = max(-6, cfg["output_gain_db"]-1); proc.rebuild(); print(f"  Gain: {cfg['output_gain_db']:.0f}dB")
            elif key == ord('h'): cfg["output_gain_db"] = min(12, cfg["output_gain_db"]+1); proc.rebuild(); print(f"  Gain: {cfg['output_gain_db']:.0f}dB")
            elif key == ord('x'): cfg["spatial_width"] = max(0.5, round(cfg["spatial_width"]-0.1,1)); print(f"  Spatial: {cfg['spatial_width']:.1f}x")
            elif key == ord('c'): cfg["spatial_width"] = min(3.0, round(cfg["spatial_width"]+0.1,1)); print(f"  Spatial: {cfg['spatial_width']:.1f}x")
            elif key == ord('b'): proc.bypass = not proc.bypass; print(f"  {'>>> BYPASS <<<' if proc.bypass else '>>> MULTIBAND AKTIV <<<'}")
            elif key == ord('m'): proc.muted = not proc.muted; print(f"  {'MUTED' if proc.muted else 'UNMUTED'}")
            elif key == ord('v'):
                cfg["show_radar"] = not cfg["show_radar"]
                if not cfg["show_radar"] and cv2: cv2.destroyAllWindows()
                print(f"  Radar: {'EIN' if cfg['show_radar'] else 'AUS'}")
            elif key == ord('p'):
                print(f"\n  === MULTIBAND PRO SETTINGS ===")
                for k,v in sorted(cfg.items()):
                    if k not in ("input_device","output_device","input_name","output_name"):
                        print(f"    {k}: {v}")
                print()

    except KeyboardInterrupt:
        print("\n  Stop.")
    finally:
        if duplex:
            duplex.stop(); duplex.close()
        save_config(cfg)
        print(f"  Config gespeichert.")
        if cv2: cv2.destroyAllWindows()
        print("  GG.")


if __name__ == "__main__":
    main()
