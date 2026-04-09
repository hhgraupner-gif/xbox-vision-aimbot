"""
WARZONE COMPETITIVE AUDIO V3 — FINAL
======================================
Chirurgische Peaking-EQ | DT 990 Pro + GameDAC
Duplex WASAPI Stream | Geraete werden gespeichert

INSTALLATION:  pip install sounddevice numpy scipy opencv-python
ERSTER START:  python competitive_audio.py --input 33 --output 24
DANACH NUR:    python competitive_audio.py

STEUERUNG:
  1/2/3  Preset: Warzone / Multiplayer / Resurgence
  Q/W    Step Body 200Hz -/+
  E/R    Step Texture 2.4kHz -/+
  A/S    Step Edge 4.4kHz -/+
  D/F    Spatial -/+
  T/Z    Compression -/+
  G/H    Output Gain -/+
  U/I    Gunfire Cut -/+
  O      Mud Cut an/aus
  B      BYPASS (A/B Vergleich)
  M      Mute
  V      Radar an/aus
  P      Settings anzeigen
  ESC    Beenden
"""

import sys, os, json, time, argparse, threading
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
    "samplerate": 48000, "blocksize": 512, "channels": 2,
    "preset": "warzone",
    "step_body_hz": 200, "step_body_db": 8.0, "step_body_q": 1.8,
    "mud_cut_hz": 600, "mud_cut_db": -4.0, "mud_cut_q": 1.5, "mud_cut_on": True,
    "step_texture_hz": 2400, "step_texture_db": 8.0, "step_texture_q": 2.5,
    "step_edge_hz": 4400, "step_edge_db": 5.0, "step_edge_q": 1.2,
    "gunfire_hz": 5500, "gunfire_db": -6.0, "gunfire_q": 2.0,
    "treble_hz": 8000, "treble_db": -5.0, "treble_q": 3.0,
    "highpass_hz": 100, "lowpass_hz": 12000,
    "comp_ratio": 4.0, "comp_threshold": -18.0, "comp_attack": 0.003, "comp_release": 0.06,
    "gate_db": -46.0, "spatial_width": 1.6, "output_gain_db": 2.0,
    "show_radar": True,
}

PRESETS = {
    "warzone": {
        "step_body_db": 8.0, "step_texture_db": 8.0, "step_edge_db": 5.0,
        "mud_cut_db": -4.0, "mud_cut_on": True, "gunfire_db": -6.0, "treble_db": -5.0,
        "highpass_hz": 100, "lowpass_hz": 12000,
        "comp_ratio": 4.0, "comp_threshold": -18.0,
        "spatial_width": 1.6, "gate_db": -46.0, "output_gain_db": 2.0,
    },
    "multiplayer": {
        "step_body_db": 6.0, "step_texture_db": 7.0, "step_edge_db": 4.0,
        "mud_cut_db": -3.0, "mud_cut_on": True, "gunfire_db": -4.0, "treble_db": -4.0,
        "highpass_hz": 80, "lowpass_hz": 13000,
        "comp_ratio": 3.0, "comp_threshold": -16.0,
        "spatial_width": 1.3, "gate_db": -44.0, "output_gain_db": 1.0,
    },
    "resurgence": {
        "step_body_db": 10.0, "step_texture_db": 10.0, "step_edge_db": 6.0,
        "mud_cut_db": -5.0, "mud_cut_on": True, "gunfire_db": -8.0, "treble_db": -6.0,
        "highpass_hz": 110, "lowpass_hz": 11000,
        "comp_ratio": 5.0, "comp_threshold": -20.0,
        "spatial_width": 1.8, "gate_db": -48.0, "output_gain_db": 3.0,
    },
    "rebirth": {
        # Rebirth Island: Kleine Map, viele Gebaeude, Metall-Treppen, viel Vertikales
        # Metall-Steps brauchen mehr Edge (4.4kHz), starke Compression fuer Steps durch Waende
        "step_body_db": 10.0, "step_texture_db": 12.0, "step_edge_db": 8.0,
        "mud_cut_db": -5.0, "mud_cut_on": True, "gunfire_db": -8.0, "treble_db": -5.0,
        "highpass_hz": 120, "lowpass_hz": 11000,
        "comp_ratio": 5.5, "comp_threshold": -22.0,
        "spatial_width": 1.5, "gate_db": -50.0, "output_gain_db": 3.0,
    },
    "heavens": {
        # Heavens Hollow: Enge Raeume, Mixed Terrain, viel Vertikales Gameplay
        # Braucht breites Spatial fuer Oben/Unten, maximale Step-Klarheit
        "step_body_db": 12.0, "step_texture_db": 14.0, "step_edge_db": 7.0,
        "mud_cut_db": -6.0, "mud_cut_on": True, "gunfire_db": -9.0, "treble_db": -6.0,
        "highpass_hz": 130, "lowpass_hz": 10500,
        "comp_ratio": 6.0, "comp_threshold": -24.0,
        "spatial_width": 1.9, "gate_db": -52.0, "output_gain_db": 4.0,
    },
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


def apply_preset(cfg, name):
    if name in PRESETS:
        cfg.update(PRESETS[name])
        cfg["preset"] = name


def make_peak_eq(fc, db, Q, fs):
    A = 10 ** (db / 40.0)
    w0 = 2 * np.pi * fc / fs
    al = np.sin(w0) / (2 * Q)
    b0, b1, b2 = 1 + al*A, -2*np.cos(w0), 1 - al*A
    a0, a1, a2 = 1 + al/A, -2*np.cos(w0), 1 - al/A
    return np.array([b0/a0, b1/a0, b2/a0]), np.array([1.0, a1/a0, a2/a0])


# ============================================================
# AUDIO PROCESSOR
# ============================================================
class Processor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.sr = cfg["samplerate"]
        self.nyq = self.sr / 2.0
        self.muted = False
        self.bypass = False
        self.comp_env = 0.0
        self.step_L = 0.0
        self.step_R = 0.0
        self.peak_dir = 0.0
        self.step_pwr = 0.0
        self._build()

    def _f(self, hz):
        return max(20.0, min(float(hz), self.nyq * 0.95))

    def _build(self):
        sr, ny, c = self.sr, self.nyq, self.cfg
        self.sos_hp = sig.butter(5, self._f(c["highpass_hz"])/ny, 'highpass', output='sos')
        self.sos_lp = sig.butter(2, self._f(c["lowpass_hz"])/ny, 'lowpass', output='sos')

        self.eqs = []
        for nm, fk, dk, qk, on in [
            ("body", "step_body_hz", "step_body_db", "step_body_q", True),
            ("mud",  "mud_cut_hz", "mud_cut_db", "mud_cut_q", c["mud_cut_on"]),
            ("tex",  "step_texture_hz", "step_texture_db", "step_texture_q", True),
            ("edge", "step_edge_hz", "step_edge_db", "step_edge_q", True),
            ("gun",  "gunfire_hz", "gunfire_db", "gunfire_q", True),
            ("treb", "treble_hz", "treble_db", "treble_q", True),
        ]:
            if on and abs(c[dk]) > 0.1:
                self.eqs.append(make_peak_eq(self._f(c[fk]), c[dk], c[qk], sr))

        lo, hi = self._f(1500), self._f(4500)
        self.sos_rd = sig.butter(2, [lo/ny, hi/ny], 'bandpass', output='sos')

        self.g_out = 10 ** (c["output_gain_db"]/20)
        self.g_gate = 10 ** (c["gate_db"]/20)
        self.g_comp = 10 ** (c["comp_threshold"]/20)
        self._zi()

    def _zi(self):
        z = lambda s: np.zeros((s.shape[0], 2))
        self.zi_hp = [z(self.sos_hp), z(self.sos_hp)]
        self.zi_lp = [z(self.sos_lp), z(self.sos_lp)]
        self.zi_rd = [z(self.sos_rd), z(self.sos_rd)]
        self.zi_eq = [[sig.lfilter_zi(b, a)*0, sig.lfilter_zi(b, a)*0] for b, a in self.eqs]

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
            x, self.zi_hp[i] = sig.sosfilt(self.sos_hp, x, zi=self.zi_hp[i])
            for j, (b, a) in enumerate(self.eqs):
                x, self.zi_eq[j][i] = sig.lfilter(b, a, x, zi=self.zi_eq[j][i])
            x, self.zi_lp[i] = sig.sosfilt(self.sos_lp, x, zi=self.zi_lp[i])
            r, self.zi_rd[i] = sig.sosfilt(self.sos_rd, x, zi=self.zi_rd[i])
            rms = float(np.sqrt(np.mean(r*r)))
            if i == 0: self.step_L = rms
            else: self.step_R = rms
            C[i] = x

        L, R = C

        # Noise Gate
        if np.sqrt(np.mean(L*L) + np.mean(R*R)) < self.g_gate:
            L *= 0.02; R *= 0.02

        # Compression
        rat = self.cfg["comp_ratio"]
        if rat > 1.01:
            pk = max(np.max(np.abs(L)), np.max(np.abs(R)), 1e-10)
            att, rel = self.cfg["comp_attack"], self.cfg["comp_release"]
            self.comp_env += (att if pk > self.comp_env else rel) * (pk - self.comp_env)
            self.comp_env = max(self.comp_env, 1e-10)
            if self.comp_env > self.g_comp:
                odb = 20 * np.log10(self.comp_env / self.g_comp)
                rdb = odb * (1 - 1/rat)
                g = 10**(-rdb/20) * 10**(rdb*0.5/20)
            else:
                g = 1.0
            L *= g; R *= g

        # Spatial
        w = self.cfg["spatial_width"]
        if abs(w - 1) > 0.01:
            m, s = (L+R)*0.5, (L-R)*0.5*w
            L, R = m+s, m-s

        # Output + Limiter
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

        # Status
        p = cfg["preset"].upper()
        if bypass:
            cv2.putText(img, ">>> BYPASS <<<", (80,self.SZ//2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        cv2.putText(img, f"[{p}] BODY:{cfg['step_body_db']:.0f} TEX:{cfg['step_texture_db']:.0f} EDGE:{cfg['step_edge_db']:.0f}",
                    (6,16), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0,180,100), 1)
        cv2.putText(img, f"MUD:{cfg['mud_cut_db']:.0f} GUN:{cfg['gunfire_db']:.0f} DT990:{cfg['treble_db']:.0f} SPA:{cfg['spatial_width']:.1f}",
                    (6,32), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (130,130,140), 1)
        cv2.putText(img, f"COMP:{cfg['comp_ratio']:.1f}:1 GAIN:{cfg['output_gain_db']:.0f}dB HP:{cfg['highpass_hz']}Hz",
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
    """Findet Geraet per Name (fuer gespeicherte Config)."""
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
        if d['max_input_channels'] >= 2:
            for kw in ["avermedia", "gc571", "capture", "game capture", "live gamer"]:
                if kw in d['name'].lower():
                    return i
    return None


# ============================================================
# MAIN
# ============================================================
def main():
    if sd is None:
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Warzone Competitive Audio V3")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--input", type=int, default=None)
    parser.add_argument("--output", type=int, default=None)
    parser.add_argument("--preset", choices=["warzone", "multiplayer", "resurgence", "rebirth", "heavens"])
    parser.add_argument("--no-radar", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    devs = list_devices()

    print()
    print("  " + "=" * 50)
    print("  WARZONE COMPETITIVE AUDIO V3")
    print("  DT 990 Pro + GameDAC | Duplex WASAPI")
    print("  " + "=" * 50)

    if args.list:
        return

    # --- Geraete bestimmen (Prioritaet: CLI > Config > Auto) ---
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

    # Geraete-Namen speichern fuer naechsten Start
    cfg["input_device"] = in_dev
    cfg["output_device"] = out_dev
    cfg["input_name"] = devs[in_dev]['name']
    cfg["output_name"] = devs[out_dev]['name']

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
    bs = cfg["blocksize"]

    print(f"\n  IN:  [{in_dev}] {devs[in_dev]['name']}")
    print(f"  OUT: [{out_dev}] {devs[out_dev]['name']}")
    print(f"  {sr}Hz | Stereo | {bs} samples (~{bs/sr*1000:.1f}ms)")
    print(f"\n  [{cfg['preset'].upper()}]"
          f" Body:+{cfg['step_body_db']:.0f}dB Tex:+{cfg['step_texture_db']:.0f}dB Edge:+{cfg['step_edge_db']:.0f}dB"
          f" Mud:{cfg['mud_cut_db']:.0f}dB Gun:{cfg['gunfire_db']:.0f}dB")
    print(f"  Comp:{cfg['comp_ratio']:.1f}:1 Spatial:{cfg['spatial_width']:.1f}x"
          f" Gain:{cfg['output_gain_db']:.0f}dB DT990:{cfg['treble_db']:.0f}dB")
    print(f"\n  [1]WZ [2]MP [3]Resurg [4]Rebirth [5]Heavens | [Q/W]Body [E/R]Tex [A/S]Edge")
    print(f"  [D/F]Spa [T/Z]Comp [G/H]Gain [U/I]Gun [O]Mud [B]Bypass [M]Mute [ESC]Quit")
    print("  " + "=" * 50)

    proc = Processor(cfg)
    radar = Radar() if (cfg["show_radar"] and cv2) else None

    # --- DUPLEX STREAM (Fallback: Ring-Buffer) ---
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
            channels=ch, dtype='float32', callback=duplex_cb, latency=0.03,
        )
        duplex.start()
        audio_ok = True
        print("  DUPLEX OK\n")
    except Exception as e1:
        print(f"  Duplex fehlgeschlagen: {e1}")
        print("  Fallback: Separate Streams...")

        # Ring-Buffer Fallback
        from collections import deque as _dq

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

    # --- UI LOOP ---
    try:
        while True:
            if radar and cfg["show_radar"] and cv2:
                cv2.imshow("Step Radar", radar.draw(proc.peak_dir, proc.step_pwr, cfg, proc.bypass))

            key = (cv2.waitKey(33) & 0xFF) if (cv2 and cfg["show_radar"]) else (time.sleep(0.033) or 255)

            if key == 27: break
            elif key == ord('1'): apply_preset(cfg, "warzone"); proc.rebuild(); print("  >> WARZONE")
            elif key == ord('2'): apply_preset(cfg, "multiplayer"); proc.rebuild(); print("  >> MULTIPLAYER")
            elif key == ord('3'): apply_preset(cfg, "resurgence"); proc.rebuild(); print("  >> RESURGENCE")
            elif key == ord('4'): apply_preset(cfg, "rebirth"); proc.rebuild(); print("  >> REBIRTH ISLAND")
            elif key == ord('5'): apply_preset(cfg, "heavens"); proc.rebuild(); print("  >> HEAVENS HOLLOW")
            elif key == ord('q'): cfg["step_body_db"] = max(0, cfg["step_body_db"]-2); proc.rebuild(); print(f"  Body: {cfg['step_body_db']:.0f}dB")
            elif key == ord('w'): cfg["step_body_db"] = min(18, cfg["step_body_db"]+2); proc.rebuild(); print(f"  Body: {cfg['step_body_db']:.0f}dB")
            elif key == ord('e'): cfg["step_texture_db"] = max(0, cfg["step_texture_db"]-2); proc.rebuild(); print(f"  Tex: {cfg['step_texture_db']:.0f}dB")
            elif key == ord('r'): cfg["step_texture_db"] = min(18, cfg["step_texture_db"]+2); proc.rebuild(); print(f"  Tex: {cfg['step_texture_db']:.0f}dB")
            elif key == ord('a'): cfg["step_edge_db"] = max(0, cfg["step_edge_db"]-2); proc.rebuild(); print(f"  Edge: {cfg['step_edge_db']:.0f}dB")
            elif key == ord('s'): cfg["step_edge_db"] = min(18, cfg["step_edge_db"]+2); proc.rebuild(); print(f"  Edge: {cfg['step_edge_db']:.0f}dB")
            elif key == ord('d'): cfg["spatial_width"] = max(0.5, round(cfg["spatial_width"]-0.1,1)); print(f"  Spa: {cfg['spatial_width']:.1f}x")
            elif key == ord('f'): cfg["spatial_width"] = min(3.0, round(cfg["spatial_width"]+0.1,1)); print(f"  Spa: {cfg['spatial_width']:.1f}x")
            elif key == ord('t'): cfg["comp_ratio"] = max(1.0, round(cfg["comp_ratio"]-0.5,1)); proc.rebuild(); print(f"  Comp: {cfg['comp_ratio']:.1f}:1")
            elif key == ord('z'): cfg["comp_ratio"] = min(8.0, round(cfg["comp_ratio"]+0.5,1)); proc.rebuild(); print(f"  Comp: {cfg['comp_ratio']:.1f}:1")
            elif key == ord('g'): cfg["output_gain_db"] = max(-12, cfg["output_gain_db"]-1); proc.rebuild(); print(f"  Gain: {cfg['output_gain_db']:.0f}dB")
            elif key == ord('h'): cfg["output_gain_db"] = min(18, cfg["output_gain_db"]+1); proc.rebuild(); print(f"  Gain: {cfg['output_gain_db']:.0f}dB")
            elif key == ord('u'): cfg["gunfire_db"] = min(0, cfg["gunfire_db"]+1); proc.rebuild(); print(f"  Gun: {cfg['gunfire_db']:.0f}dB")
            elif key == ord('i'): cfg["gunfire_db"] = max(-15, cfg["gunfire_db"]-1); proc.rebuild(); print(f"  Gun: {cfg['gunfire_db']:.0f}dB")
            elif key == ord('o'): cfg["mud_cut_on"] = not cfg["mud_cut_on"]; proc.rebuild(); print(f"  Mud: {'EIN' if cfg['mud_cut_on'] else 'AUS'}")
            elif key == ord('b'): proc.bypass = not proc.bypass; print(f"  {'>>> BYPASS (raw audio) <<<' if proc.bypass else '>>> FILTER AKTIV <<<'}")
            elif key == ord('m'): proc.muted = not proc.muted; print(f"  {'MUTED' if proc.muted else 'UNMUTED'}")
            elif key == ord('v'):
                cfg["show_radar"] = not cfg["show_radar"]
                if not cfg["show_radar"] and cv2: cv2.destroyAllWindows()
                print(f"  Radar: {'EIN' if cfg['show_radar'] else 'AUS'}")
            elif key == ord('p'):
                print(f"\n  === SETTINGS ===")
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
