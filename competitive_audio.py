"""
WARZONE COMPETITIVE AUDIO V2 — DT 990 Pro Edition
====================================================
Chirurgische Peaking-EQ Filter auf exakten Footstep-Frequenzen.

STREAMING:
  Versuch 1: Duplex Stream (synchroner Clock = beste Stabilitaet)
  Versuch 2: Ring-Buffer mit Drift-Korrektur (2 separate Streams)

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
import threading
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


def make_peaking_eq(fc, gain_db, Q, fs):
    A = 10 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * fc / fs
    alpha = np.sin(w0) / (2.0 * Q)
    b0 = 1.0 + alpha * A
    b1 = -2.0 * np.cos(w0)
    b2 = 1.0 - alpha * A
    a0 = 1.0 + alpha / A
    a1 = -2.0 * np.cos(w0)
    a2 = 1.0 - alpha / A
    return np.array([b0/a0, b1/a0, b2/a0]), np.array([1.0, a1/a0, a2/a0])


# ============================================================
# AUDIO PROCESSOR
# ============================================================
class CompetitiveAudioV2:
    def __init__(self, cfg):
        self.cfg = cfg
        self.sr = cfg["samplerate"]
        self.nyq = self.sr / 2.0
        self.muted = False
        self.comp_env = 0.0
        self.step_energy_L = 0.0
        self.step_energy_R = 0.0
        self.peak_dir = 0.0
        self.step_power = 0.0
        self._build()

    def _sf(self, hz):
        return max(20.0, min(float(hz), self.nyq * 0.95))

    def _build(self):
        sr, nyq, c = self.sr, self.nyq, self.cfg
        self.sos_hp = sig.butter(5, self._sf(c["highpass_hz"]) / nyq, btype='highpass', output='sos')
        self.sos_lp = sig.butter(2, self._sf(c["lowpass_hz"]) / nyq, btype='lowpass', output='sos')
        self.eq_bands = []
        for key, hz_k, db_k, q_k, cond in [
            ("body", "step_body_hz", "step_body_db", "step_body_q", True),
            ("mud", "mud_cut_hz", "mud_cut_db", "mud_cut_q", c["mud_cut_on"]),
            ("tex", "step_texture_hz", "step_texture_db", "step_texture_q", True),
            ("edge", "step_edge_hz", "step_edge_db", "step_edge_q", True),
            ("gun", "gunfire_hz", "gunfire_db", "gunfire_q", True),
            ("treb", "treble_hz", "treble_db", "treble_q", True),
        ]:
            if cond and abs(c[db_k]) > 0.1:
                b, a = make_peaking_eq(self._sf(c[hz_k]), c[db_k], c[q_k], sr)
                self.eq_bands.append((key, b, a))
        bp_lo, bp_hi = self._sf(1500), self._sf(4500)
        self.sos_radar = sig.butter(2, [bp_lo/nyq, bp_hi/nyq], btype='bandpass', output='sos')
        self.out_gain = 10 ** (c["output_gain_db"] / 20.0)
        self.gate_lin = 10 ** (c["gate_db"] / 20.0)
        self.comp_thresh = 10 ** (c["comp_threshold"] / 20.0)
        self._init_zi()

    def _init_zi(self):
        z = lambda sos: np.zeros((sos.shape[0], 2))
        self.zi_hp = [z(self.sos_hp), z(self.sos_hp)]
        self.zi_lp = [z(self.sos_lp), z(self.sos_lp)]
        self.zi_radar = [z(self.sos_radar), z(self.sos_radar)]
        self.zi_eq = []
        for _, b, a in self.eq_bands:
            self.zi_eq.append([sig.lfilter_zi(b, a) * 0, sig.lfilter_zi(b, a) * 0])

    def rebuild(self):
        self._build()

    def process(self, data):
        if self.muted:
            return np.zeros_like(data)
        ch = min(data.shape[1] if data.ndim > 1 else 1, 2)
        chs = [data[:, 0].copy(), data[:, 1].copy()] if ch >= 2 else [data.flatten().copy()] * 2
        for i in range(2):
            x = chs[i]
            x, self.zi_hp[i] = sig.sosfilt(self.sos_hp, x, zi=self.zi_hp[i])
            for bi, (_, b, a) in enumerate(self.eq_bands):
                x, self.zi_eq[bi][i] = sig.lfilter(b, a, x, zi=self.zi_eq[bi][i])
            x, self.zi_lp[i] = sig.sosfilt(self.sos_lp, x, zi=self.zi_lp[i])
            rb, self.zi_radar[i] = sig.sosfilt(self.sos_radar, x, zi=self.zi_radar[i])
            rms = float(np.sqrt(np.mean(rb ** 2)))
            if i == 0: self.step_energy_L = rms
            else: self.step_energy_R = rms
            chs[i] = x
        L, R = chs
        if np.sqrt(np.mean(L**2) + np.mean(R**2)) < self.gate_lin:
            L *= 0.02; R *= 0.02
        ratio = self.cfg["comp_ratio"]
        if ratio > 1.01:
            pk = max(np.max(np.abs(L)), np.max(np.abs(R)), 1e-10)
            if pk > self.comp_env: self.comp_env += self.cfg["comp_attack"] * (pk - self.comp_env)
            else: self.comp_env += self.cfg["comp_release"] * (pk - self.comp_env)
            self.comp_env = max(self.comp_env, 1e-10)
            if self.comp_env > self.comp_thresh:
                odb = 20 * np.log10(self.comp_env / self.comp_thresh)
                rdb = odb * (1 - 1/ratio)
                cg = 10 ** (-rdb/20); mu = 10 ** (rdb*0.5/20)
            else: cg, mu = 1.0, 1.0
            L *= cg * mu; R *= cg * mu
        w = self.cfg["spatial_width"]
        if abs(w - 1) > 0.01:
            m = (L + R) * 0.5; s = (L - R) * 0.5 * w
            L = m + s; R = m - s
        L *= self.out_gain; R *= self.out_gain
        L = np.tanh(L); R = np.tanh(R)
        t = self.step_energy_L + self.step_energy_R
        if t > 0.0003:
            self.peak_dir = (self.step_energy_R - self.step_energy_L) / t
            self.step_power = min(1.0, t * 25)
        else: self.step_power *= 0.82
        out = np.column_stack([L, R])
        if data.ndim > 1 and data.shape[1] > 2:
            out = np.column_stack([out, np.zeros((len(L), data.shape[1]-2), dtype=np.float32)])
        return out.astype(np.float32)


# ============================================================
# RADAR
# ============================================================
class StepRadar:
    SZ = 320
    def __init__(self):
        self.cx, self.cy = self.SZ//2, self.SZ//2
        self.r = self.SZ//2 - 35
        self.trail = deque(maxlen=25)
    def draw(self, d, pw, cfg):
        img = np.zeros((self.SZ, self.SZ, 3), dtype=np.uint8)
        cv2.circle(img, (self.cx, self.cy), self.r, (25,25,30), -1)
        cv2.circle(img, (self.cx, self.cy), self.r, (50,50,60), 2)
        cv2.circle(img, (self.cx, self.cy), self.r//2, (35,35,40), 1)
        cv2.line(img, (self.cx-10, self.cy), (self.cx+10, self.cy), (50,50,60), 1)
        cv2.line(img, (self.cx, self.cy-10), (self.cx, self.cy+10), (50,50,60), 1)
        cv2.putText(img, "L", (6, self.cy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80,80,90), 1)
        cv2.putText(img, "R", (self.SZ-18, self.cy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80,80,90), 1)
        if pw > 0.04:
            self.trail.append((d, pw))
            for i2, (dd, pp) in enumerate(self.trail):
                a = (i2+1)/len(self.trail)
                cv2.circle(img, (int(self.cx+dd*self.r*0.85), self.cy), int(2+pp*10*a), (0,int(60*a),0), -1)
            px = int(self.cx + d*self.r*0.85)
            sz = int(4 + pw*18)
            col = (0,80,255) if pw > 0.7 else (0,200,255) if pw > 0.4 else (0,200,120)
            cv2.circle(img, (px, self.cy), sz, col, -1)
            cv2.line(img, (self.cx, self.cy), (px, self.cy), col, 2)
        p = cfg["preset"].upper()
        cv2.putText(img, f"[{p}] BODY:{cfg['step_body_db']:.0f}dB TEX:{cfg['step_texture_db']:.0f}dB EDGE:{cfg['step_edge_db']:.0f}dB",
                    (6,16), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0,180,100), 1)
        cv2.putText(img, f"MUD:{cfg['mud_cut_db']:.0f}dB GUN:{cfg['gunfire_db']:.0f}dB DT990:{cfg['treble_db']:.0f}dB",
                    (6,32), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (130,130,140), 1)
        cv2.putText(img, f"COMP:{cfg['comp_ratio']:.1f}x SPA:{cfg['spatial_width']:.1f}x GAIN:{cfg['output_gain_db']:.0f}dB HP:{cfg['highpass_hz']}Hz",
                    (6,48), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (130,130,140), 1)
        bw = int(pw * (self.SZ-20))
        bc = (0,180,100) if pw < 0.4 else (0,200,255) if pw < 0.7 else (0,80,255)
        cv2.rectangle(img, (10, self.SZ-16), (10+bw, self.SZ-6), bc, -1)
        cv2.rectangle(img, (10, self.SZ-16), (self.SZ-10, self.SZ-6), (50,50,60), 1)
        return img


# ============================================================
# RING BUFFER (Profi-Niveau)
# ============================================================
class RingBuf:
    """Thread-safe Ring-Buffer mit Drift-Korrektur."""
    def __init__(self, cap, nch):
        self.buf = np.zeros((cap, nch), dtype=np.float32)
        self.cap = cap
        self.wp = 0
        self.rp = 0
        self.lock = threading.Lock()
        self.level = 0

    def write(self, data):
        n = len(data)
        with self.lock:
            end = self.wp + n
            if end <= self.cap:
                self.buf[self.wp:end] = data
            else:
                p1 = self.cap - self.wp
                self.buf[self.wp:] = data[:p1]
                self.buf[:n-p1] = data[p1:]
            self.wp = end % self.cap
            self.level = min(self.level + n, self.cap)

    def read(self, n):
        with self.lock:
            out = np.zeros((n, self.buf.shape[1]), dtype=np.float32)
            avail = min(self.level, n)
            if avail > 0:
                end = self.rp + avail
                if end <= self.cap:
                    out[:avail] = self.buf[self.rp:end]
                else:
                    p1 = self.cap - self.rp
                    out[:p1] = self.buf[self.rp:]
                    out[p1:avail] = self.buf[:avail-p1]
                self.rp = (self.rp + avail) % self.cap
                self.level -= avail
            return out

    def fill(self):
        return self.level / self.cap


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
        if i == sd.default.device[0]: mark += " << IN"
        if i == sd.default.device[1]: mark += " << OUT"
        print(f"  {i:3d} {tag} {d['name'][:48]:<48} {int(d['default_samplerate'])}Hz in:{inp} out:{out}{mark}")
    print("  " + "-" * 70)
    return devs


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
    blocksize = cfg["blocksize"]

    print(f"\n  Input:    [{in_dev}] {devs[in_dev]['name']}")
    print(f"  Output:   [{out_dev}] {devs[out_dev]['name']}")
    print(f"  SR: {sr}Hz | CH: {ch} | Block: {blocksize} (~{blocksize/sr*1000:.1f}ms)")
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

    # =================================================================
    # STREAMING: Duplex (best) → Fallback Ring-Buffer
    # =================================================================
    stream_mode = None
    duplex_stream = None
    in_stream = None
    out_stream = None
    audio_running = True
    audio_threads = []

    # --- VERSUCH 1: DUPLEX STREAM ---
    print("\n  Versuche Duplex Stream...")
    try:
        def duplex_cb(indata, outdata, frames, time_info, status):
            try:
                outdata[:] = proc.process(indata)[:outdata.shape[0], :outdata.shape[1]]
            except Exception:
                outdata[:] = indata

        duplex_stream = sd.Stream(
            device=(in_dev, out_dev), samplerate=sr, blocksize=blocksize,
            channels=ch, dtype='float32', callback=duplex_cb,
            latency=0.03,  # 30ms Headroom — verhindert Menu-Stottern
        )
        duplex_stream.start()
        stream_mode = "duplex"
        print(f"  DUPLEX LAEUFT! (synchroner Clock, ~{blocksize/sr*1000:.1f}ms)\n")
    except Exception as e:
        print(f"  Duplex nicht moeglich: {e}")

    # --- VERSUCH 2: RING-BUFFER ---
    if stream_mode != "duplex":
        print("  Starte Ring-Buffer Modus...")
        buf_secs = 0.3
        buf_cap = int(sr * buf_secs)
        rb = RingBuf(buf_cap, ch)

        try:
            in_stream = sd.InputStream(
                device=in_dev, samplerate=sr, blocksize=blocksize,
                channels=ch, dtype='float32',
            )
            out_stream = sd.OutputStream(
                device=out_dev, samplerate=sr, blocksize=blocksize,
                channels=ch, dtype='float32',
            )
            in_stream.start()
            out_stream.start()

            # Pre-Fill 40%
            rb.write(np.zeros((int(buf_cap * 0.4), ch), dtype=np.float32))
            stream_mode = "ringbuf"
            print(f"  RING-BUFFER LAEUFT! (Buffer: {buf_secs}s, ~{blocksize/sr*1000:.1f}ms)\n")
        except Exception as e:
            print(f"  FEHLER: {e}")
            if in_stream: in_stream.close()
            if out_stream: out_stream.close()
            return

        def t_input():
            while audio_running:
                try:
                    data, _ = in_stream.read(blocksize)
                    processed = proc.process(data)
                    rb.write(processed)
                except Exception:
                    pass

        def t_output():
            while audio_running:
                try:
                    data = rb.read(blocksize)
                    out_stream.write(data)
                    f = rb.fill()
                    if f > 0.75:
                        rb.read(blocksize)
                    elif f < 0.2:
                        time.sleep(blocksize / sr * 0.25)
                except Exception:
                    pass

        ti = threading.Thread(target=t_input, daemon=True)
        to = threading.Thread(target=t_output, daemon=True)
        ti.start()
        to.start()
        audio_threads = [ti, to]

    # =================================================================
    # UI LOOP
    # =================================================================
    try:
        while True:
            if radar and cfg["show_radar"] and cv2 is not None:
                img = radar.draw(proc.peak_dir, proc.step_power, cfg)
                # Ring-Buffer Level anzeigen
                if stream_mode == "ringbuf":
                    fl = rb.fill()
                    cv2.putText(img, f"BUF:{fl*100:.0f}%", (240, 16),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.30,
                                (0,200,100) if 0.2 < fl < 0.75 else (0,80,255), 1)
                cv2.imshow("Step Radar", img)

            if cv2 is not None and cfg["show_radar"]:
                key = cv2.waitKey(33) & 0xFF
            else:
                time.sleep(0.033)
                key = 255

            if key == 27: break
            elif key == ord('1'): apply_preset(cfg, "warzone"); proc.rebuild(); print("  >> WARZONE")
            elif key == ord('2'): apply_preset(cfg, "multiplayer"); proc.rebuild(); print("  >> MULTIPLAYER")
            elif key == ord('3'): apply_preset(cfg, "resurgence"); proc.rebuild(); print("  >> RESURGENCE")
            elif key == ord('q'): cfg["step_body_db"] = max(0, cfg["step_body_db"]-2); proc.rebuild(); print(f"  Body: {cfg['step_body_db']:.0f}dB")
            elif key == ord('w'): cfg["step_body_db"] = min(18, cfg["step_body_db"]+2); proc.rebuild(); print(f"  Body: {cfg['step_body_db']:.0f}dB")
            elif key == ord('e'): cfg["step_texture_db"] = max(0, cfg["step_texture_db"]-2); proc.rebuild(); print(f"  Tex: {cfg['step_texture_db']:.0f}dB")
            elif key == ord('r'): cfg["step_texture_db"] = min(18, cfg["step_texture_db"]+2); proc.rebuild(); print(f"  Tex: {cfg['step_texture_db']:.0f}dB")
            elif key == ord('a'): cfg["step_edge_db"] = max(0, cfg["step_edge_db"]-2); proc.rebuild(); print(f"  Edge: {cfg['step_edge_db']:.0f}dB")
            elif key == ord('s'): cfg["step_edge_db"] = min(18, cfg["step_edge_db"]+2); proc.rebuild(); print(f"  Edge: {cfg['step_edge_db']:.0f}dB")
            elif key == ord('d'): cfg["spatial_width"] = max(0.5, round(cfg["spatial_width"]-0.1, 1)); print(f"  Spatial: {cfg['spatial_width']:.1f}x")
            elif key == ord('f'): cfg["spatial_width"] = min(3.0, round(cfg["spatial_width"]+0.1, 1)); print(f"  Spatial: {cfg['spatial_width']:.1f}x")
            elif key == ord('t'): cfg["comp_ratio"] = max(1.0, round(cfg["comp_ratio"]-0.5, 1)); proc.rebuild(); print(f"  Comp: {cfg['comp_ratio']:.1f}:1")
            elif key == ord('z'): cfg["comp_ratio"] = min(8.0, round(cfg["comp_ratio"]+0.5, 1)); proc.rebuild(); print(f"  Comp: {cfg['comp_ratio']:.1f}:1")
            elif key == ord('g'): cfg["output_gain_db"] = max(-12, cfg["output_gain_db"]-1); proc.rebuild(); print(f"  Gain: {cfg['output_gain_db']:.0f}dB")
            elif key == ord('h'): cfg["output_gain_db"] = min(18, cfg["output_gain_db"]+1); proc.rebuild(); print(f"  Gain: {cfg['output_gain_db']:.0f}dB")
            elif key == ord('u'): cfg["gunfire_db"] = min(0, cfg["gunfire_db"]+1); proc.rebuild(); print(f"  Gun: {cfg['gunfire_db']:.0f}dB")
            elif key == ord('i'): cfg["gunfire_db"] = max(-15, cfg["gunfire_db"]-1); proc.rebuild(); print(f"  Gun: {cfg['gunfire_db']:.0f}dB")
            elif key == ord('o'): cfg["mud_cut_on"] = not cfg["mud_cut_on"]; proc.rebuild(); print(f"  Mud: {'EIN' if cfg['mud_cut_on'] else 'AUS'}")
            elif key == ord('m'): proc.muted = not proc.muted; print(f"  {'MUTED' if proc.muted else 'UNMUTED'}")
            elif key == ord('v'):
                cfg["show_radar"] = not cfg["show_radar"]
                if not cfg["show_radar"] and cv2: cv2.destroyAllWindows()
                print(f"  Radar: {'EIN' if cfg['show_radar'] else 'AUS'}")
            elif key == ord('p'):
                print(f"\n  === CONFIG ({stream_mode}) ===")
                for k, v in sorted(cfg.items()):
                    if k not in ("input_device", "output_device"): print(f"    {k}: {v}")
                print()

    except KeyboardInterrupt:
        print("\n  Ctrl+C")
    finally:
        audio_running = False
        for t in audio_threads:
            t.join(timeout=1)
        if duplex_stream:
            duplex_stream.stop()
            duplex_stream.close()
        if in_stream:
            in_stream.stop()
            in_stream.close()
        if out_stream:
            out_stream.stop()
            out_stream.close()
        save_config(cfg)
        print(f"  Config gespeichert: {CONFIG_FILE}")
        if cv2: cv2.destroyAllWindows()
        print("  Beendet.")


if __name__ == "__main__":
    main()
