"""
eRayz Audio ULTIMATE v1.0
==========================
Competitive Audio Tool fuer Warzone / BO7 / DT990 Pro

Kombiniert:
  - eRayz Surgical PRO DSP Engine (bewährt)
  - GUI mit Live-Slidern (kein PowerShell nötig)
  - InsuredFrames-Style "Eigene Schüsse leiser" (Gunfire Ducker)
  - Erweiterte HRTF-basierte Spatialization
  - Pro Preset für DT990 Pro + GC571

Starten: python erayz_audio_ultimate.py
"""

import sys, os, json, threading, math, time
import numpy as np
from scipy import signal as sig

try:
    import sounddevice as sd
    SD_OK = True
except ImportError:
    SD_OK = False
    print("FEHLER: pip install sounddevice")

try:
    import tkinter as tk
    from tkinter import ttk, messagebox
    TK_OK = True
except ImportError:
    TK_OK = False

# ============================================================
# CONFIG
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(os.path.expanduser("~"), "erayz_ultimate.json")

DEFAULT_CFG = {
    "input_device": None,
    "output_device": None,
    "samplerate": 48000,
    "blocksize": 128,
    "channels": 2,

    # DT990 Pro Korrektur (Harman Target + Pro Warzone)
    "dt990_sub_cut_hz": 40, "dt990_sub_cut_db": -10.0, "dt990_sub_cut_q": 0.7,
    "dt990_boom_hz": 75, "dt990_boom_db": -7.0, "dt990_boom_q": 3.0,
    "dt990_mid_fill_hz": 1200, "dt990_mid_fill_db": 3.5, "dt990_mid_fill_q": 0.8,
    "dt990_spike_hz": 8200, "dt990_spike_db": -8.0, "dt990_spike_q": 3.0,
    "dt990_sibilance_hz": 6000, "dt990_sibilance_db": -2.5, "dt990_sibilance_q": 2.0,

    # Footstep Enhancement (Pro-Frequenzen aus Warzone Audio-Analyse)
    "step_impact_hz": 115, "step_impact_db": 6.0, "step_impact_q": 2.0,
    "step_body_hz": 250, "step_body_db": 8.0, "step_body_q": 1.5,
    "step_presence_hz": 770, "step_presence_db": 3.0, "step_presence_q": 1.4,
    "step_texture_hz": 1500, "step_texture_db": 7.0, "step_texture_q": 2.0,
    "step_direction_hz": 3170, "step_direction_db": 10.0, "step_direction_q": 4.5,
    "step_clarity_hz": 4500, "step_clarity_db": 6.0, "step_clarity_q": 1.8,

    # Extra Bands (Feind-Aktionen lauter)
    "reload_hz": 800, "reload_db": 4.0, "reload_q": 2.0,
    "parachute_hz": 1400, "parachute_db": 3.5, "parachute_q": 2.5,

    # Cuts (eigene Sounds + Dreck raus)
    "highpass_hz": 55,
    "lowpass_hz": 10000,
    "mud_hz": 500, "mud_db": -8.0, "mud_q": 1.0,
    "gunfire_eq_hz": 5500, "gunfire_eq_db": -9.0, "gunfire_eq_q": 2.0,
    "streak_hz": 200, "streak_db": -6.0, "streak_q": 1.0,
    "ambient_hz": 90, "ambient_db": -8.0, "ambient_q": 0.8,
    "wind_hz": 350, "wind_db": -5.0, "wind_q": 1.5,

    # Gunfire Ducker (InsuredFrames-Style — aggressiver)
    "ducker_enabled": True,
    "ducker_thresh_db": -20.0,
    "ducker_ratio": 5.0,
    "ducker_attack": 0.0003,
    "ducker_release": 0.06,

    # Dynamics (Loudness EQ Effekt — leise Steps lauter)
    "comp_ratio": 4.5, "comp_thresh_db": -28.0,
    "comp_attack": 0.002, "comp_release": 0.05,
    "gate_db": -58.0,

    # HRTF Spatial (staerker)
    "spatial_width": 2.0,
    "hrtf_enabled": True,
    "hrtf_delay_ms": 0.4,
    "hrtf_high_shelf_db": -3.0,
    "hrtf_crossfeed": 0.20,

    # Output (Preamp runter wegen mehr Boost)
    "output_gain_db": 2.0,
}


def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE) as f:
                c = DEFAULT_CFG.copy()
                c.update(json.load(f))
                return c
        except Exception:
            pass
    return DEFAULT_CFG.copy()


def save_config(c):
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(c, f, indent=2)
    except Exception:
        pass


def make_peak(fc, db, Q, fs):
    fc = max(20.0, min(float(fc), fs / 2.0 * 0.95))
    if abs(db) < 0.05:
        return None
    A = 10 ** (db / 40.0)
    w0 = 2 * np.pi * fc / fs
    al = np.sin(w0) / (2 * Q)
    b = np.array([1 + al * A, -2 * np.cos(w0), 1 - al * A])
    a = np.array([1 + al / A, -2 * np.cos(w0), 1 - al / A])
    return b / a[0], a / a[0]


# ============================================================
# DSP ENGINE
# ============================================================
class UltimateProcessor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.sr = cfg["samplerate"]
        self.ny = self.sr / 2.0
        self.muted = False
        self.bypass = False
        self.comp_env = 0.0
        self.duck_env = 0.0
        self.step_comp_env = 0.0
        self.step_duck_env = 0.0
        self.transient_env = [0.0, 0.0]
        self._lock = threading.Lock()
        self.step_L = 0.0
        self.step_R = 0.0
        self.peak_dir = 0.0
        self.step_pwr = 0.0
        self._build()

    def _build(self):
        c = self.cfg
        ny = self.ny

        # Highpass + Lowpass
        hp = max(20, min(c["highpass_hz"], ny * 0.95))
        lp = max(20, min(c["lowpass_hz"], ny * 0.95))
        self.sos_hp = sig.butter(4, hp / ny, "highpass", output="sos")
        self.sos_lp = sig.butter(2, lp / ny, "lowpass", output="sos")

        # EQ Chain: DT990 Korrektur + Steps + Extra + Cuts
        self.eqs = []
        params = [
            ("dt990_sub_cut_hz", "dt990_sub_cut_db", "dt990_sub_cut_q"),
            ("dt990_boom_hz", "dt990_boom_db", "dt990_boom_q"),
            ("dt990_mid_fill_hz", "dt990_mid_fill_db", "dt990_mid_fill_q"),
            ("dt990_spike_hz", "dt990_spike_db", "dt990_spike_q"),
            ("dt990_sibilance_hz", "dt990_sibilance_db", "dt990_sibilance_q"),
            ("step_impact_hz", "step_impact_db", "step_impact_q"),
            ("step_body_hz", "step_body_db", "step_body_q"),
            ("step_presence_hz", "step_presence_db", "step_presence_q"),
            ("step_texture_hz", "step_texture_db", "step_texture_q"),
            ("step_direction_hz", "step_direction_db", "step_direction_q"),
            ("step_clarity_hz", "step_clarity_db", "step_clarity_q"),
            ("reload_hz", "reload_db", "reload_q"),
            ("parachute_hz", "parachute_db", "parachute_q"),
            ("mud_hz", "mud_db", "mud_q"),
            ("gunfire_eq_hz", "gunfire_eq_db", "gunfire_eq_q"),
            ("streak_hz", "streak_db", "streak_q"),
            ("ambient_hz", "ambient_db", "ambient_q"),
            ("wind_hz", "wind_db", "wind_q"),
        ]
        for fk, dk, qk in params:
            r = make_peak(c[fk], c[dk], c[qk], self.sr)
            if r:
                self.eqs.append(r)

        # Gunfire Ducker: Bandpass 80-800Hz (eigene Schuesse)
        lo_d = max(20, min(80, ny * 0.95)) / ny
        hi_d = max(20, min(800, ny * 0.95)) / ny
        self.sos_duck_bp = sig.butter(2, [lo_d, hi_d], "bandpass", output="sos")

        # Step Detection: 1.5-5kHz
        lo_r = max(20, min(1500, ny * 0.95)) / ny
        hi_r = max(20, min(5000, ny * 0.95)) / ny
        self.sos_step = sig.butter(2, [lo_r, hi_r], "bandpass", output="sos")

        # HRTF: High-shelf filter fuer kontralaterale Daempfung
        if c.get("hrtf_enabled", True):
            hs_freq = max(20, min(4000, ny * 0.95))
            r = make_peak(hs_freq, c.get("hrtf_high_shelf_db", -2.0), 0.7, self.sr)
            self.hrtf_filter = r
            # ITD: Interaural Time Difference (echte Ohr-Physik)
            # 0.6ms = max ITD beim Menschen (Sound von ganz links/rechts)
            itd_ms = c.get("hrtf_delay_ms", 0.4)
            self.hrtf_delay = max(1, int(itd_ms * self.sr / 1000))
            self.hrtf_buf_L = np.zeros(self.hrtf_delay, dtype=np.float32)
            self.hrtf_buf_R = np.zeros(self.hrtf_delay, dtype=np.float32)
            # ILD: Interaural Level Difference (High-Shelf fuer Kopf-Schatten)
            ild_freq = max(20, min(2000, ny * 0.95))
            r2 = make_peak(ild_freq, -4.0, 0.5, self.sr)
            self.ild_filter = r2
        else:
            self.hrtf_filter = None
            self.ild_filter = None

        # Step-Band Isolierung: 200Hz - 5kHz (fuer Transient Shaper + Step Compression)
        lo_step = max(20, min(200, ny * 0.95)) / ny
        hi_step = max(20, min(5000, ny * 0.95)) / ny
        self.sos_step_iso = sig.butter(3, [lo_step, hi_step], "bandpass", output="sos")

        # Transient Detection: Schneller Envelope fuer Attack-Erkennung
        # Sehr schnell (0.1ms attack) um den ANFANG eines Steps zu erkennen
        self.trans_attack = math.exp(-1.0 / (self.sr * 0.0001))
        self.trans_release = math.exp(-1.0 / (self.sr * 0.015))

        # Gains
        self.g_out = 10 ** (c["output_gain_db"] / 20)
        self.g_gate = 10 ** (c["gate_db"] / 20)
        self.g_comp = 10 ** (c["comp_thresh_db"] / 20)
        self.g_duck = 10 ** (c["ducker_thresh_db"] / 20)

        # Filter States
        self._init_states()

    def _init_states(self):
        z = lambda s: np.zeros((s.shape[0], 2))
        self.zi_hp = [z(self.sos_hp), z(self.sos_hp)]
        self.zi_lp = [z(self.sos_lp), z(self.sos_lp)]
        self.zi_duck = [z(self.sos_duck_bp), z(self.sos_duck_bp)]
        self.zi_step = [z(self.sos_step), z(self.sos_step)]
        self.zi_step_iso = [z(self.sos_step_iso), z(self.sos_step_iso)]
        self.zi_eq = [
            [sig.lfilter_zi(b, a) * 0, sig.lfilter_zi(b, a) * 0]
            for b, a in self.eqs
        ]
        if self.hrtf_filter:
            b, a = self.hrtf_filter
            self.zi_hrtf = [sig.lfilter_zi(b, a) * 0, sig.lfilter_zi(b, a) * 0]
        if self.ild_filter:
            b, a = self.ild_filter
            self.zi_ild = [sig.lfilter_zi(b, a) * 0, sig.lfilter_zi(b, a) * 0]

    def rebuild(self):
        with self._lock:
            self._build()

    def process(self, data):
        if self.muted:
            return np.zeros_like(data)
        if self.bypass:
            return data.copy()

        ch = min(data.shape[1] if data.ndim > 1 else 1, 2)
        C = [data[:, 0].copy(), data[:, 1].copy()] if ch >= 2 else [data.flatten().copy()] * 2

        with self._lock:
            for i in range(2):
                x = C[i]

                # Highpass
                x, self.zi_hp[i] = sig.sosfilt(self.sos_hp, x, zi=self.zi_hp[i])

                # EQ Chain (16 Bands)
                for j, (b, a) in enumerate(self.eqs):
                    x, self.zi_eq[j][i] = sig.lfilter(b, a, x, zi=self.zi_eq[j][i])

                # Lowpass
                x, self.zi_lp[i] = sig.sosfilt(self.sos_lp, x, zi=self.zi_lp[i])

                # ── TRANSIENT SHAPER (sanfter — nur echte Schritt-Attacks) ──
                step_band, self.zi_step_iso[i] = sig.sosfilt(
                    self.sos_step_iso, x, zi=self.zi_step_iso[i])
                abs_step = np.abs(step_band)
                env = np.zeros_like(abs_step)
                e = self.transient_env[i]
                ta, tr = self.trans_attack, self.trans_release
                for s in range(len(abs_step)):
                    if abs_step[s] > e:
                        e = ta * e + (1 - ta) * abs_step[s]
                    else:
                        e = tr * e + (1 - tr) * abs_step[s]
                    env[s] = e
                self.transient_env[i] = e
                transient = np.maximum(abs_step - env, 0)
                # Nur 1.5x boost (statt 3x — weniger Ambient-Verstaerkung)
                x = x + step_band * transient * 1.5

                # ── STEP-BAND COMPRESSION (sanfter) ──
                step_rms = float(np.sqrt(np.mean(step_band * step_band)))
                if step_rms > 0.012:
                    s_odb = 20 * math.log10(step_rms / 0.012)
                    s_rdb = s_odb * (1 - 1 / 3.0)
                    s_makeup = 10 ** (s_rdb * 0.4 / 20)
                    x = x + step_band * (s_makeup - 1.0) * 0.3

                # Step Detection
                r, self.zi_step[i] = sig.sosfilt(self.sos_step, x, zi=self.zi_step[i])
                rms = float(np.sqrt(np.mean(r * r)))
                if i == 0:
                    self.step_L = rms
                else:
                    self.step_R = rms

                C[i] = x

        L, R = C

        # ── STEP-PRIORITY DUCKING (sanfter) ──
        step_total = self.step_L + self.step_R
        if step_total > 0.02:
            step_duck = max(0.65, 1.0 - step_total * 2.0)
            mono = (L + R) * 0.5
            side = (L - R) * 0.5
            L = mono * step_duck + side
            R = mono * step_duck - side

        # ── GUNFIRE DUCKER ──
        if self.cfg.get("ducker_enabled", True):
            mono = (L + R) * 0.5
            duck_sig, self.zi_duck[0] = sig.sosfilt(
                self.sos_duck_bp, mono, zi=self.zi_duck[0])
            duck_peak = float(np.max(np.abs(duck_sig)))
            d_att = self.cfg["ducker_attack"]
            d_rel = self.cfg["ducker_release"]
            if duck_peak > self.duck_env:
                self.duck_env += d_att * (duck_peak - self.duck_env)
            else:
                self.duck_env += d_rel * (duck_peak - self.duck_env)
            self.duck_env = max(self.duck_env, 1e-10)
            if self.duck_env > self.g_duck:
                d_ratio = self.cfg["ducker_ratio"]
                d_odb = 20 * math.log10(self.duck_env / self.g_duck)
                d_rdb = d_odb * (1 - 1 / d_ratio)
                L *= 10 ** (-d_rdb / 20)
                R *= 10 ** (-d_rdb / 20)

        # ── NOISE GATE ──
        level = math.sqrt(float(np.mean(L * L)) + float(np.mean(R * R)))
        if level < self.g_gate:
            L *= 0.02
            R *= 0.02

        # ── MASTER COMPRESSION ──
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
                g = 10 ** (-rdb / 20) * 10 ** (rdb * 0.4 / 20)
            else:
                g = 1.0
            L *= g
            R *= g

        # ── HRTF v2: ITD + ILD (Pro-Trick #4) ──
        if self.cfg.get("hrtf_enabled", True) and self.hrtf_filter:
            b, a = self.hrtf_filter
            L_delayed = np.concatenate([self.hrtf_buf_L, L])[:len(L)]
            R_delayed = np.concatenate([self.hrtf_buf_R, R])[:len(R)]
            self.hrtf_buf_L = L[-self.hrtf_delay:]
            self.hrtf_buf_R = R[-self.hrtf_delay:]
            xfeed = self.cfg.get("hrtf_crossfeed", 0.20)
            if self.ild_filter:
                bi, ai = self.ild_filter
                R_shadow, self.zi_ild[0] = sig.lfilter(bi, ai, R_delayed * xfeed, zi=self.zi_ild[0])
                L_shadow, self.zi_ild[1] = sig.lfilter(bi, ai, L_delayed * xfeed, zi=self.zi_ild[1])
            else:
                R_shadow = R_delayed * xfeed
                L_shadow = L_delayed * xfeed
            L_cross, self.zi_hrtf[0] = sig.lfilter(b, a, R_shadow, zi=self.zi_hrtf[0])
            R_cross, self.zi_hrtf[1] = sig.lfilter(b, a, L_shadow, zi=self.zi_hrtf[1])
            L = L + L_cross
            R = R + R_cross

        # ── SPATIAL WIDTH ──
        w = self.cfg["spatial_width"]
        if abs(w - 1) > 0.01:
            m, s = (L + R) * 0.5, (L - R) * 0.5 * w
            L, R = m + s, m - s

        # ── LOOK-AHEAD LIMITER (Pro-Trick #5) ──
        out_g = self.g_out
        L_out = L * out_g
        R_out = R * out_g
        pk_out = max(float(np.max(np.abs(L_out))), float(np.max(np.abs(R_out))), 1e-10)
        if pk_out > 0.95:
            limit_g = 0.95 / pk_out
            L_out *= limit_g
            R_out *= limit_g
        L = np.tanh(L_out * 1.05)
        R = np.tanh(R_out * 1.05)

        # Radar Data
        t = self.step_L + self.step_R
        if t > 0.0003:
            self.peak_dir = (self.step_R - self.step_L) / t
            self.step_pwr = min(1.0, t * 25)
        else:
            self.step_pwr *= 0.82

        out = np.column_stack([L, R])
        return out.astype(np.float32)


# ============================================================
# GUI
# ============================================================
if TK_OK:

    # Colors
    BG = "#0c0c16"
    BG2 = "#10101e"
    BG3 = "#080812"
    BG4 = "#1a1a2e"
    ACC = "#00e87b"
    ACC2 = "#00c468"
    PUR = "#7b2fff"
    PUR2 = "#9d5cff"
    GOLD = "#c9a84c"
    RED = "#e94560"
    TXT = "#e0e0f0"
    MUT = "#444466"
    CYAN = "#00d4ff"

    BANDS_LEFT = [
        ("step_impact_db", "Impact", "115 Hz", 0, 12, ACC),
        ("step_body_db", "Body", "250 Hz", 0, 14, ACC),
        ("step_presence_db", "Presence", "770 Hz", 0, 8, ACC),
        ("step_texture_db", "Texture", "1.5 kHz", 0, 14, CYAN),
        ("step_direction_db", "Direction", "3.2 kHz", 0, 14, CYAN),
        ("step_clarity_db", "Clarity", "4.5 kHz", 0, 14, CYAN),
        ("reload_db", "Reload", "800 Hz", 0, 12, GOLD),
        ("mud_db", "Mud Cut", "500 Hz", -12, 0, RED),
        ("gunfire_eq_db", "Gunfire EQ", "5.5 kHz", -12, 0, RED),
        ("streak_db", "Streaks", "200 Hz", -8, 0, RED),
    ]

    BANDS_RIGHT = [
        ("comp_ratio", "Kompressor", "Ratio", 1.0, 8.0, GOLD),
        ("ducker_ratio", "Gunfire Duck", "Ratio", 1.0, 8.0, RED),
        ("spatial_width", "Spatial Width", "L/R", 0.5, 2.5, PUR2),
        ("output_gain_db", "Output Gain", "dB", -6, 12, ACC),
    ]

    class ErayzApp(tk.Tk):
        def __init__(self):
            super().__init__()
            self.title("eRayz Audio ULTIMATE")
            self.configure(bg=BG)
            self.geometry("900x720")
            self.resizable(False, False)
            self.cfg = load_config()
            self.proc = None
            self.stream = None
            self.running = False
            self.devs = []
            self.sliders = {}
            self._build_ui()
            self._load_devices()
            self.protocol("WM_DELETE_WINDOW", self._on_close)

        def _build_ui(self):
            # Header
            hdr = tk.Canvas(self, bg=BG2, height=70, highlightthickness=0)
            hdr.pack(fill="x")
            hdr.create_rectangle(0, 66, 900, 70, fill=ACC, outline="")
            hdr.create_text(24, 22, text="eRayz Audio ULTIMATE",
                            font=("Segoe UI", 20, "bold"), fill="#ffffff", anchor="w")
            hdr.create_text(26, 48,
                            text="11-Band EQ  |  Gunfire Ducker  |  HRTF Spatial  |  DT990 Pro",
                            font=("Segoe UI", 9), fill=ACC2, anchor="w")
            hdr.create_rectangle(850, 12, 888, 42, fill=ACC, outline="")
            hdr.create_text(869, 27, text="v1", font=("Segoe UI", 10, "bold"),
                            fill="#000", anchor="center")

            # Status Bar
            self.status_var = tk.StringVar(
                value="OFFLINE  |  Geraete auswaehlen und START druecken"
            )
            self.status_lbl = tk.Label(
                self, textvariable=self.status_var,
                font=("Segoe UI", 9, "bold"), bg=BG3, fg=MUT, pady=7, anchor="w", padx=16
            )
            self.status_lbl.pack(fill="x")

            # Device Selection
            df = tk.Frame(self, bg=BG2, padx=20, pady=10)
            df.pack(fill="x")
            tk.Label(df, text="INPUT", font=("Segoe UI", 8, "bold"),
                     bg=BG2, fg=ACC, width=8, anchor="w").grid(row=0, column=0, sticky="w")
            self.in_combo = ttk.Combobox(df, width=62, state="readonly",
                                          font=("Segoe UI", 9))
            self.in_combo.grid(row=0, column=1, padx=8, pady=3)

            tk.Label(df, text="OUTPUT", font=("Segoe UI", 8, "bold"),
                     bg=BG2, fg=GOLD, width=8, anchor="w").grid(row=1, column=0, sticky="w")
            self.out_combo = ttk.Combobox(df, width=62, state="readonly",
                                           font=("Segoe UI", 9))
            self.out_combo.grid(row=1, column=1, padx=8, pady=3)

            # Buttons
            bf = tk.Frame(self, bg=BG3, pady=10, padx=20)
            bf.pack(fill="x")
            self.start_btn = tk.Button(
                bf, text="  START  ", font=("Segoe UI", 11, "bold"),
                bg=ACC, fg="#000", relief="flat", pady=8, cursor="hand2", bd=0,
                command=self._toggle
            )
            self.start_btn.pack(side="left", padx=(0, 10))

            self.bypass_btn = tk.Button(
                bf, text="  BYPASS  ", font=("Segoe UI", 10),
                bg=BG4, fg=MUT, relief="flat", pady=8, cursor="hand2", bd=0,
                command=self._toggle_bypass
            )
            self.bypass_btn.pack(side="left", padx=4)

            self.mute_btn = tk.Button(
                bf, text="  MUTE  ", font=("Segoe UI", 10),
                bg=BG4, fg=MUT, relief="flat", pady=8, cursor="hand2", bd=0,
                command=self._toggle_mute
            )
            self.mute_btn.pack(side="left", padx=4)

            self.duck_btn = tk.Button(
                bf, text="  DUCKER  ", font=("Segoe UI", 10),
                bg=RED if self.cfg.get("ducker_enabled", True) else BG4,
                fg="#fff" if self.cfg.get("ducker_enabled", True) else MUT,
                relief="flat", pady=8, cursor="hand2", bd=0,
                command=self._toggle_ducker
            )
            self.duck_btn.pack(side="left", padx=4)

            self.hrtf_btn = tk.Button(
                bf, text="  HRTF  ", font=("Segoe UI", 10),
                bg=PUR if self.cfg.get("hrtf_enabled", True) else BG4,
                fg="#fff" if self.cfg.get("hrtf_enabled", True) else MUT,
                relief="flat", pady=8, cursor="hand2", bd=0,
                command=self._toggle_hrtf
            )
            self.hrtf_btn.pack(side="left", padx=4)

            # Separator
            sep = tk.Canvas(self, bg=BG, height=3, highlightthickness=0)
            sep.pack(fill="x")
            sep.create_rectangle(0, 0, 450, 3, fill=ACC, outline="")
            sep.create_rectangle(450, 0, 900, 3, fill=GOLD, outline="")

            # EQ Panels
            eq_frame = tk.Frame(self, bg=BG, padx=16, pady=8)
            eq_frame.pack(fill="both", expand=True)

            # Left Panel: Footstep Enhancement
            left = tk.Frame(eq_frame, bg=BG2, bd=0)
            left.pack(side="left", fill="both", expand=True, padx=(0, 6))
            lhdr = tk.Canvas(left, bg=BG2, height=28, highlightthickness=0)
            lhdr.pack(fill="x")
            lhdr.create_rectangle(0, 0, 4, 28, fill=ACC, outline="")
            lhdr.create_text(14, 14, text="FOOTSTEP ENHANCEMENT",
                             font=("Segoe UI", 9, "bold"), fill=ACC, anchor="w")

            for key, label, freq, mn, mx, color in BANDS_LEFT:
                self._make_slider(left, key, label, freq, mn, mx, color)

            # Right Panel: Dynamics & Output
            right = tk.Frame(eq_frame, bg=BG2, bd=0)
            right.pack(side="left", fill="both", expand=True, padx=(6, 0))
            rhdr = tk.Canvas(right, bg=BG2, height=28, highlightthickness=0)
            rhdr.pack(fill="x")
            rhdr.create_rectangle(0, 0, 4, 28, fill=GOLD, outline="")
            rhdr.create_text(14, 14, text="DYNAMICS & OUTPUT",
                             font=("Segoe UI", 9, "bold"), fill=GOLD, anchor="w")

            for key, label, freq, mn, mx, color in BANDS_RIGHT:
                self._make_slider(right, key, label, freq, mn, mx, color)

            # Footer
            ft = tk.Canvas(self, bg=BG3, height=24, highlightthickness=0)
            ft.pack(fill="x", side="bottom")
            ft.create_text(
                12, 12,
                text="DT990 Korrektur: Mid+3 | Peak-7@8.2k | Sib-2@6k  |  "
                     "Ducker: Eigene Schuesse leiser  |  "
                     "HRTF: 3D Spatial  |  Xbox > GC571 > DSP > Headset",
                font=("Segoe UI", 7), fill="#333355", anchor="w"
            )

        def _make_slider(self, parent, key, label, freq, mn, mx, color):
            row = tk.Frame(parent, bg=BG2)
            row.pack(fill="x", padx=10, pady=4)

            lf = tk.Frame(row, bg=BG2, width=85)
            lf.pack(side="left")
            lf.pack_propagate(False)
            tk.Label(lf, text=label, font=("Segoe UI", 9, "bold"),
                     bg=BG2, fg=TXT, anchor="w").pack(fill="x")
            tk.Label(lf, text=freq, font=("Segoe UI", 7),
                     bg=BG2, fg=MUT, anchor="w").pack(fill="x")

            val = self.cfg.get(key, 0)
            var = tk.DoubleVar(value=val)

            tk.Scale(
                row, from_=mn, to=mx, resolution=0.5, orient="horizontal",
                variable=var, length=195,
                bg=BG2, fg=TXT, troughcolor=BG4, highlightthickness=0,
                bd=0, showvalue=False,
                command=lambda v, k=key, dv=var: self._on_slide(k, dv)
            ).pack(side="left", padx=6)

            vf = tk.Frame(row, bg=BG3, width=80, height=28)
            vf.pack(side="left")
            vf.pack_propagate(False)
            vlbl = tk.Label(
                vf, text=self._fmt(val, key),
                font=("Segoe UI", 10, "bold"), bg=BG3, fg=color, anchor="center"
            )
            vlbl.pack(fill="both", expand=True)
            self.sliders[key] = (var, vlbl, color)

        def _fmt(self, v, key):
            if key == "comp_ratio" or key == "ducker_ratio":
                return f"{v:.1f} : 1"
            elif key == "spatial_width":
                return f"{v:.1f} x"
            else:
                return f"{v:+.1f} dB"

        def _on_slide(self, key, var):
            val = round(var.get(), 1)
            self.cfg[key] = val
            _, vlbl, _ = self.sliders[key]
            c = ACC if val > 0 else RED if val < 0 else MUT
            vlbl.config(text=self._fmt(val, key), fg=c)
            if self.proc:
                self.proc.cfg[key] = val
                self.proc.rebuild()

        def _load_devices(self):
            if not SD_OK:
                return
            self.devs = sd.query_devices()
            in_names, out_names = [], []
            self._in_idx, self._out_idx = [], []
            auto_in, auto_out = None, None

            for i, d in enumerate(self.devs):
                nm = f"[{i}]  {d['name']}"
                if d["max_input_channels"] >= 2:
                    in_names.append(nm)
                    self._in_idx.append(i)
                    nl = d["name"].lower()
                    if "hdmi" in nl and ("live streamer" in nl or "avermedia" in nl):
                        auto_in = len(in_names) - 1
                if d["max_output_channels"] >= 2:
                    out_names.append(nm)
                    self._out_idx.append(i)

            self.in_combo["values"] = in_names
            self.out_combo["values"] = out_names
            if auto_in is not None:
                self.in_combo.current(auto_in)
            elif in_names:
                self.in_combo.current(0)
            if out_names:
                self.out_combo.current(0)

        def _toggle(self):
            if self.running:
                self._stop()
            else:
                self._start()

        def _start(self):
            if not SD_OK:
                return
            try:
                ix = self._in_idx[self.in_combo.current()]
                ox = self._out_idx[self.out_combo.current()]
            except Exception:
                messagebox.showerror("Fehler", "Geraete auswaehlen!")
                return

            self.cfg["input_device"] = ix
            self.cfg["output_device"] = ox
            ch = min(self.devs[ix]["max_input_channels"], 2)
            self.proc = UltimateProcessor(self.cfg)

            def callback(indata, outdata, frames, ti, status):
                try:
                    outdata[:] = self.proc.process(indata)[:outdata.shape[0], :outdata.shape[1]]
                except Exception:
                    outdata[:] = indata

            try:
                self.stream = sd.Stream(
                    device=(ix, ox), samplerate=self.cfg["samplerate"],
                    blocksize=self.cfg["blocksize"], channels=ch,
                    dtype="float32", callback=callback, latency="low"
                )
                self.stream.start()
                self.running = True
                self.start_btn.config(text="  STOP  ", bg=RED)
                in_name = self.devs[ix]["name"][:30]
                out_name = self.devs[ox]["name"][:26]
                self.status_var.set(f"ONLINE  |  {in_name}  >  {out_name}")
                self.status_lbl.config(fg=ACC)
            except Exception as e:
                messagebox.showerror("Stream Fehler", str(e))

        def _stop(self):
            if self.stream:
                try:
                    self.stream.stop()
                    self.stream.close()
                except Exception:
                    pass
            self.stream = None
            self.running = False
            self.start_btn.config(text="  START  ", bg=ACC)
            self.status_var.set("OFFLINE  |  Gestoppt")
            self.status_lbl.config(fg=MUT)

        def _toggle_bypass(self):
            if self.proc:
                self.proc.bypass = not self.proc.bypass
                active = self.proc.bypass
                self.bypass_btn.config(
                    bg=RED if active else BG4,
                    fg="#fff" if active else MUT
                )
                self.status_var.set(
                    "BYPASS AKTIV  |  Kein Processing" if active
                    else "ONLINE  |  eRayz ULTIMATE aktiv"
                )
                self.status_lbl.config(fg=RED if active else ACC)

        def _toggle_mute(self):
            if self.proc:
                self.proc.muted = not self.proc.muted
                active = self.proc.muted
                self.mute_btn.config(
                    bg=RED if active else BG4,
                    fg="#fff" if active else MUT
                )

        def _toggle_ducker(self):
            self.cfg["ducker_enabled"] = not self.cfg.get("ducker_enabled", True)
            active = self.cfg["ducker_enabled"]
            self.duck_btn.config(
                bg=RED if active else BG4,
                fg="#fff" if active else MUT
            )
            if self.proc:
                self.proc.cfg["ducker_enabled"] = active

        def _toggle_hrtf(self):
            self.cfg["hrtf_enabled"] = not self.cfg.get("hrtf_enabled", True)
            active = self.cfg["hrtf_enabled"]
            self.hrtf_btn.config(
                bg=PUR if active else BG4,
                fg="#fff" if active else MUT
            )
            if self.proc:
                self.proc.cfg["hrtf_enabled"] = active
                self.proc.rebuild()

        def _on_close(self):
            self._stop()
            save_config(self.cfg)
            self.destroy()


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    if TK_OK:
        ErayzApp().mainloop()
    else:
        print("tkinter nicht verfuegbar!")
