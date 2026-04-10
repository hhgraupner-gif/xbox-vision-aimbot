"""
eRayz Audio — SURGICAL PRO v6
================================
Research-Based | DT 990 Pro Corrected | Subtle & Effective
Harman Target + Competitive Footstep Enhancement

Based on:
- Harman 2018 Target Curve for DT 990 Pro
- Pro competitive EQ research (SteelSeries, GadgetryTech)
- DT 990 Pro frequency response compensation
- CoD Warzone footstep frequency analysis

ERSTER START:  python competitive_audio.py --input 18 --output 14
DANACH NUR:    python competitive_audio.py

STEUERUNG:
  Q/W    Step Body 250Hz -/+
  E/R    Step Texture 2kHz -/+
  A/S    Step Direction 3.2kHz -/+
  D/F    Step Clarity 4.5kHz -/+
  T/Z    Compression -/+
  G/H    Output Gain -/+
  X/C    Spatial -/+
  U/I    Gunfire Cut -/+
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

# ═══════════════════════════════════════════════════════════════
# SURGICAL PRO v6 — Research-Based Settings
#
# DT 990 Pro Probleme:
#   - Eingebrochene Mitten 1-5kHz (wo Steps sind!)
#   - Riesen-Spike bei 8kHz (Ohrenschmerzen)
#   - Zu viel Sub-Bass (maskiert Steps)
#
# Loesung: Harman-Korrektur + chirurgische Step-Boosts
# ═══════════════════════════════════════════════════════════════
DEFAULT_CFG = {
    "input_device": None, "output_device": None,
    "input_name": "", "output_name": "",
    "samplerate": 48000, "blocksize": 256, "channels": 2,
    #
    # === STEP 1: DT 990 Pro Korrektur (Harman Target) ===
    # Bringt den Kopfhoerer erstmal auf neutral
    "dt990_mid_fill_hz": 1200, "dt990_mid_fill_db": 3.0, "dt990_mid_fill_q": 0.8,
    "dt990_spike_hz": 8200, "dt990_spike_db": -7.0, "dt990_spike_q": 3.0,
    "dt990_sibilance_hz": 6000, "dt990_sibilance_db": -2.0, "dt990_sibilance_q": 2.0,
    #
    # === STEP 2: Competitive Step Enhancement ===
    # Chirurgische Boosts an bewiesenen Step-Frequenzen
    #
    # Band 1: Step Body (250Hz) — Gewicht/Aufprall auf Boden
    "step_body_hz": 250, "step_body_db": 5.0, "step_body_q": 2.2,
    #
    # Band 2: Step Texture (2000Hz) — Material-Sound, Reload
    "step_texture_hz": 2000, "step_texture_db": 4.0, "step_texture_q": 1.5,
    #
    # Band 3: Step Direction (3170Hz) — Richtungs-Cues (HRTF)
    "step_direction_hz": 3170, "step_direction_db": 4.0, "step_direction_q": 4.0,
    #
    # Band 4: Step Clarity (4500Hz) — Schritt-Details
    "step_clarity_hz": 4500, "step_clarity_db": 3.0, "step_clarity_q": 2.0,
    #
    # === STEP 3: Unerwuenschtes entfernen ===
    # Highpass: Sub-Bass weg (maskiert Steps)
    "highpass_hz": 80,
    # Lowpass: Nur das was noetig ist
    "lowpass_hz": 10000,
    # Mud Cut: Ambient raus
    "mud_hz": 500, "mud_db": -3.0, "mud_q": 1.0,
    # Gunfire Reduction
    "gunfire_hz": 5500, "gunfire_db": -5.0, "gunfire_q": 2.0,
    #
    # === STEP 4: Dynamics ===
    # Moderate Compression — leise Steps hoeher, laute Steps nicht unangenehm
    "comp_ratio": 3.5, "comp_thresh_db": -22.0,
    "comp_attack": 0.003, "comp_release": 0.05,
    # Noise Gate
    "gate_db": -55.0,
    #
    # === STEP 5: Spatial & Output ===
    "spatial_width": 1.5,
    "output_gain_db": 3.0,
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


def make_peak(fc, db, Q, fs):
    """Peaking EQ Biquad Filter."""
    fc = max(20.0, min(float(fc), fs / 2.0 * 0.95))
    if abs(db) < 0.05:
        return None
    A = 10 ** (db / 40.0)
    w0 = 2 * np.pi * fc / fs
    al = np.sin(w0) / (2 * Q)
    b = np.array([1 + al * A, -2 * np.cos(w0), 1 - al * A])
    a = np.array([1 + al / A, -2 * np.cos(w0), 1 - al / A])
    return b / a[0], a / a[0]


class SurgicalProcessor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.sr = cfg["samplerate"]
        self.ny = self.sr / 2.0
        self.muted = False
        self.bypass = False
        self.comp_env = 0.0
        self.step_L = 0.0
        self.step_R = 0.0
        self.peak_dir = 0.0
        self.step_pwr = 0.0
        self._build()

    def _build(self):
        c = self.cfg
        ny = self.ny

        # Pre: Highpass + Lowpass
        hp = max(20, min(c["highpass_hz"], ny * 0.95))
        lp = max(20, min(c["lowpass_hz"], ny * 0.95))
        self.sos_hp = sig.butter(4, hp / ny, 'highpass', output='sos')
        self.sos_lp = sig.butter(2, lp / ny, 'lowpass', output='sos')

        # Build EQ chain: DT990 Correction + Step Enhancement + Cuts
        self.eqs = []
        params = [
            # DT 990 Correction
            ("dt990_mid_fill_hz", "dt990_mid_fill_db", "dt990_mid_fill_q"),
            ("dt990_spike_hz", "dt990_spike_db", "dt990_spike_q"),
            ("dt990_sibilance_hz", "dt990_sibilance_db", "dt990_sibilance_q"),
            # Step Enhancement
            ("step_body_hz", "step_body_db", "step_body_q"),
            ("step_texture_hz", "step_texture_db", "step_texture_q"),
            ("step_direction_hz", "step_direction_db", "step_direction_q"),
            ("step_clarity_hz", "step_clarity_db", "step_clarity_q"),
            # Cuts
            ("mud_hz", "mud_db", "mud_q"),
            ("gunfire_hz", "gunfire_db", "gunfire_q"),
        ]
        for fk, dk, qk in params:
            r = make_peak(c[fk], c[dk], c[qk], self.sr)
            if r is not None:
                self.eqs.append(r)

        # Radar detection: 1.5-5kHz
        lo_r = max(20, min(1500, ny * 0.95)) / ny
        hi_r = max(20, min(5000, ny * 0.95)) / ny
        self.sos_rd = sig.butter(2, [lo_r, hi_r], 'bandpass', output='sos')

        # Compression
        self.g_out = 10 ** (c["output_gain_db"] / 20)
        self.g_gate = 10 ** (c["gate_db"] / 20)
        self.g_comp = 10 ** (c["comp_thresh_db"] / 20)

        self._zi()

    def _zi(self):
        z = lambda s: np.zeros((s.shape[0], 2))
        self.zi_hp = [z(self.sos_hp), z(self.sos_hp)]
        self.zi_lp = [z(self.sos_lp), z(self.sos_lp)]
        self.zi_rd = [z(self.sos_rd), z(self.sos_rd)]
        self.zi_eq = [[sig.lfilter_zi(b, a) * 0, sig.lfilter_zi(b, a) * 0]
                      for b, a in self.eqs]

    def rebuild(self):
        self._build()

    def process(self, data):
        if self.muted:
            return np.zeros_like(data)
        if self.bypass:
            return data.copy()

        ch = min(data.shape[1] if data.ndim > 1 else 1, 2)
        C = [data[:, 0].copy(), data[:, 1].copy()] if ch >= 2 else [data.flatten().copy()] * 2

        for i in range(2):
            x = C[i]

            # Highpass
            x, self.zi_hp[i] = sig.sosfilt(self.sos_hp, x, zi=self.zi_hp[i])

            # EQ Chain (9 Bands: 3x DT990 + 4x Steps + Mud + Gunfire)
            for j, (b, a) in enumerate(self.eqs):
                x, self.zi_eq[j][i] = sig.lfilter(b, a, x, zi=self.zi_eq[j][i])

            # Lowpass
            x, self.zi_lp[i] = sig.sosfilt(self.sos_lp, x, zi=self.zi_lp[i])

            # Radar
            r, self.zi_rd[i] = sig.sosfilt(self.sos_rd, x, zi=self.zi_rd[i])
            rms = float(np.sqrt(np.mean(r * r)))
            if i == 0: self.step_L = rms
            else: self.step_R = rms

            C[i] = x

        L, R = C

        # Noise Gate
        level = math.sqrt(float(np.mean(L * L)) + float(np.mean(R * R)))
        if level < self.g_gate:
            L *= 0.02
            R *= 0.02

        # Compression (Soft)
        rat = self.cfg["comp_ratio"]
        if rat > 1.01:
            pk = max(float(np.max(np.abs(L))), float(np.max(np.abs(R))), 1e-10)
            att = self.cfg["comp_attack"]
            rel = self.cfg["comp_release"]
            self.comp_env += (att if pk > self.comp_env else rel) * (pk - self.comp_env)
            self.comp_env = max(self.comp_env, 1e-10)
            if self.comp_env > self.g_comp:
                odb = 20 * math.log10(self.comp_env / self.g_comp)
                rdb = odb * (1 - 1 / rat)
                g = 10 ** (-rdb / 20) * 10 ** (rdb * 0.4 / 20)  # 40% Makeup
            else:
                g = 1.0
            L *= g
            R *= g

        # Spatial
        w = self.cfg["spatial_width"]
        if abs(w - 1) > 0.01:
            m, s = (L + R) * 0.5, (L - R) * 0.5 * w
            L, R = m + s, m - s

        # Output + Limiter
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
            out = np.column_stack([out, np.zeros((len(L), data.shape[1] - 2), dtype=np.float32)])
        return out.astype(np.float32)


class Radar:
    SZ = 320

    def __init__(self):
        self.cx, self.cy = self.SZ // 2, self.SZ // 2
        self.r = self.SZ // 2 - 35
        self.trail = deque(maxlen=25)

    def draw(self, proc, cfg):
        img = np.zeros((self.SZ, self.SZ, 3), dtype=np.uint8)
        cv2.circle(img, (self.cx, self.cy), self.r, (20, 22, 28), -1)
        cv2.circle(img, (self.cx, self.cy), self.r, (45, 50, 60), 2)
        cv2.circle(img, (self.cx, self.cy), self.r // 2, (30, 33, 40), 1)
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

        if proc.bypass:
            cv2.putText(img, "BYPASS", (self.SZ // 2 - 50, self.SZ // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.putText(img, f"[SURGICAL PRO v6]", (6, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 200, 120), 1)
        cv2.putText(img,
                    f"Body:{cfg['step_body_db']:+.0f} Tex:{cfg['step_texture_db']:+.0f} "
                    f"Dir:{cfg['step_direction_db']:+.0f} Clar:{cfg['step_clarity_db']:+.0f}",
                    (6, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (120, 125, 140), 1)
        cv2.putText(img,
                    f"Mud:{cfg['mud_db']:+.0f} Gun:{cfg['gunfire_db']:+.0f} "
                    f"Comp:{cfg['comp_ratio']:.1f}:1 Spa:{cfg['spatial_width']:.1f} "
                    f"Gain:{cfg['output_gain_db']:+.0f}",
                    (6, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (100, 105, 120), 1)

        bw = int(pw * (self.SZ - 20))
        bc = (0, 200, 100) if pw < 0.4 else (0, 200, 255) if pw < 0.7 else (0, 60, 255)
        cv2.rectangle(img, (10, self.SZ - 16), (10 + bw, self.SZ - 6), bc, -1)
        cv2.rectangle(img, (10, self.SZ - 16), (self.SZ - 10, self.SZ - 6), (40, 45, 55), 1)
        return img


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
        if d['max_input_channels'] >= 2:
            name_l = d['name'].lower()
            if 'hdmi' in name_l and ('live streamer' in name_l or 'avermedia' in name_l):
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


def main():
    if sd is None:
        sys.exit(1)

    parser = argparse.ArgumentParser(description="eRayz Audio — Surgical Pro v6")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--input", type=int, default=None)
    parser.add_argument("--output", type=int, default=None)
    parser.add_argument("--no-radar", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    devs = list_devices()

    print()
    print("  " + "=" * 55)
    print("  eRayz Audio — SURGICAL PRO v6")
    print("  Harman-Corrected DT 990 Pro | Competitive Steps")
    print("  9-Band Parametric EQ | 48kHz/256")
    print("  " + "=" * 55)

    if args.list:
        return

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
    print(f"  {sr}Hz | {bs} samples (~{bs / sr * 1000:.1f}ms)")

    print(f"\n  DT 990 Pro Korrektur:")
    print(f"    Mid Fill 1.2kHz: {cfg['dt990_mid_fill_db']:+.0f}dB")
    print(f"    Spike 8.2kHz:    {cfg['dt990_spike_db']:+.0f}dB")
    print(f"    Sibilance 6kHz:  {cfg['dt990_sibilance_db']:+.0f}dB")

    print(f"\n  Competitive Steps:")
    print(f"    Body 250Hz:      {cfg['step_body_db']:+.0f}dB")
    print(f"    Texture 2kHz:    {cfg['step_texture_db']:+.0f}dB")
    print(f"    Direction 3.2kHz:{cfg['step_direction_db']:+.0f}dB")
    print(f"    Clarity 4.5kHz:  {cfg['step_clarity_db']:+.0f}dB")

    print(f"\n  Cuts: Mud {cfg['mud_db']:+.0f}dB | Gunfire {cfg['gunfire_db']:+.0f}dB")
    print(f"  Comp: {cfg['comp_ratio']:.1f}:1 | Spatial: {cfg['spatial_width']:.1f}x | Gain: {cfg['output_gain_db']:+.0f}dB")

    print(f"\n  [Q/W]Body [E/R]Tex [A/S]Dir [D/F]Clar [T/Z]Comp")
    print(f"  [G/H]Gain [X/C]Spa [U/I]Gun [B]Bypass [M]Mute [ESC]Quit")
    print("  " + "=" * 55)

    proc = SurgicalProcessor(cfg)
    radar = Radar() if (cfg["show_radar"] and cv2) else None

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
            def fill(self): return self.lv / self.cap

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

    try:
        while True:
            if radar and cfg["show_radar"] and cv2:
                cv2.imshow("eRayz Radar", radar.draw(proc, cfg))

            key = (cv2.waitKey(33) & 0xFF) if (cv2 and cfg["show_radar"]) else (time.sleep(0.033) or 255)

            if key == 27: break
            elif key == ord('q'): cfg["step_body_db"] = max(0, cfg["step_body_db"] - 1); proc.rebuild(); print(f"  Body: {cfg['step_body_db']:+.0f}dB")
            elif key == ord('w'): cfg["step_body_db"] = min(10, cfg["step_body_db"] + 1); proc.rebuild(); print(f"  Body: {cfg['step_body_db']:+.0f}dB")
            elif key == ord('e'): cfg["step_texture_db"] = max(0, cfg["step_texture_db"] - 1); proc.rebuild(); print(f"  Tex: {cfg['step_texture_db']:+.0f}dB")
            elif key == ord('r'): cfg["step_texture_db"] = min(10, cfg["step_texture_db"] + 1); proc.rebuild(); print(f"  Tex: {cfg['step_texture_db']:+.0f}dB")
            elif key == ord('a'): cfg["step_direction_db"] = max(0, cfg["step_direction_db"] - 1); proc.rebuild(); print(f"  Dir: {cfg['step_direction_db']:+.0f}dB")
            elif key == ord('s'): cfg["step_direction_db"] = min(10, cfg["step_direction_db"] + 1); proc.rebuild(); print(f"  Dir: {cfg['step_direction_db']:+.0f}dB")
            elif key == ord('d'): cfg["step_clarity_db"] = max(0, cfg["step_clarity_db"] - 1); proc.rebuild(); print(f"  Clar: {cfg['step_clarity_db']:+.0f}dB")
            elif key == ord('f'): cfg["step_clarity_db"] = min(10, cfg["step_clarity_db"] + 1); proc.rebuild(); print(f"  Clar: {cfg['step_clarity_db']:+.0f}dB")
            elif key == ord('t'): cfg["comp_ratio"] = max(1.0, round(cfg["comp_ratio"] - 0.5, 1)); proc.rebuild(); print(f"  Comp: {cfg['comp_ratio']:.1f}:1")
            elif key == ord('z'): cfg["comp_ratio"] = min(8.0, round(cfg["comp_ratio"] + 0.5, 1)); proc.rebuild(); print(f"  Comp: {cfg['comp_ratio']:.1f}:1")
            elif key == ord('g'): cfg["output_gain_db"] = max(-6, cfg["output_gain_db"] - 1); proc.rebuild(); print(f"  Gain: {cfg['output_gain_db']:+.0f}dB")
            elif key == ord('h'): cfg["output_gain_db"] = min(12, cfg["output_gain_db"] + 1); proc.rebuild(); print(f"  Gain: {cfg['output_gain_db']:+.0f}dB")
            elif key == ord('x'): cfg["spatial_width"] = max(0.5, round(cfg["spatial_width"] - 0.1, 1)); print(f"  Spa: {cfg['spatial_width']:.1f}x")
            elif key == ord('c'): cfg["spatial_width"] = min(3.0, round(cfg["spatial_width"] + 0.1, 1)); print(f"  Spa: {cfg['spatial_width']:.1f}x")
            elif key == ord('u'): cfg["gunfire_db"] = min(0, cfg["gunfire_db"] + 1); proc.rebuild(); print(f"  Gun: {cfg['gunfire_db']:+.0f}dB")
            elif key == ord('i'): cfg["gunfire_db"] = max(-12, cfg["gunfire_db"] - 1); proc.rebuild(); print(f"  Gun: {cfg['gunfire_db']:+.0f}dB")
            elif key == ord('b'): proc.bypass = not proc.bypass; print(f"  {'>>> BYPASS <<<' if proc.bypass else '>>> SURGICAL PRO AKTIV <<<'}")
            elif key == ord('m'): proc.muted = not proc.muted; print(f"  {'MUTED' if proc.muted else 'UNMUTED'}")
            elif key == ord('v'):
                cfg["show_radar"] = not cfg["show_radar"]
                if not cfg["show_radar"] and cv2: cv2.destroyAllWindows()
                print(f"  Radar: {'EIN' if cfg['show_radar'] else 'AUS'}")
            elif key == ord('p'):
                print(f"\n  === SURGICAL PRO v6 SETTINGS ===")
                for k, v in sorted(cfg.items()):
                    if k not in ("input_device", "output_device", "input_name", "output_name"):
                        print(f"    {k}: {v}")
                print()

    except KeyboardInterrupt:
        print("\n  Stop.")
    finally:
        if duplex:
            duplex.stop(); duplex.close()
        save_config(cfg)
        print("  Config gespeichert. GG.")
        if cv2: cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
