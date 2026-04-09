"""
WARZONE COMPETITIVE AUDIO — DT 990 Pro Edition
=================================================
Echtzeit Audio-Enhancement fuer Warzone Ranked.
Speziell abgestimmt auf Beyerdynamic DT 990 Pro + SteelSeries GameDAC.

FEATURES:
  - DT 990 Pro Korrektur (8kHz Spitze zaehmen, Mitten anheben)
  - Footstep Enhancement (100-300Hz Thuds + 2-4kHz Schritte)
  - Explosion/Gunfire Suppression (Bass-Rumble + Schuss-Peaks daempfen)
  - Dynamic Compression (leise Steps lauter, laute Sounds begrenzen)
  - Stereo Spatial Enhancement (breiteres Stereo fuer Richtungserkennung)
  - Noise Gate (Stille bleibt still, kein Grundrauschen)
  - Visueller Audio-Radar (zeigt Richtung starker Step-Signale)
  - 3 Presets: Warzone (BR), Multiplayer, Resurgence

INSTALLATION:
  pip install sounddevice numpy scipy opencv-python

SETUP:
  1. Xbox Audio muss ueber Capture Card (GC571) auf den PC kommen
  2. Script starten: python competitive_audio.py
  3. Input = Capture Card Audio, Output = GameDAC / Default
  4. SteelSeries Sonar am besten AUS oder auf "Flat" stellen
     (unser Tool macht das besser und gezielter)

STEUERUNG (im Radar-Fenster):
  1/2/3     = Preset: Warzone / Multiplayer / Resurgence
  Q/W       = Footstep Boost -/+
  E/R       = Compression -/+
  A/S       = Bass Cut Frequenz -/+
  D/F       = Spatial Width -/+
  T/Z       = Treble Tame (DT990 8kHz) -/+
  G/H       = Output Gain -/+
  M         = Mute/Unmute
  V         = Radar an/aus
  P         = Alle Einstellungen anzeigen
  ESC       = Beenden (Config wird gespeichert)
"""

import sys
import os
import json
import time
import threading
import argparse
from collections import deque

import numpy as np
from scipy import signal as sig

try:
    import sounddevice as sd
except ImportError:
    sd = None
    print("=" * 55)
    print("  FEHLER: 'sounddevice' nicht installiert!")
    print("  Installiere mit:  pip install sounddevice")
    print("=" * 55)

try:
    import cv2
except ImportError:
    cv2 = None

# ============================================================
# CONFIG
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(SCRIPT_DIR, "audio_config.json")

DEFAULT_CFG = {
    "input_device":   None,    # None = auto-detect / default
    "output_device":  None,    # None = default
    "samplerate":     48000,
    "blocksize":      512,     # ~10ms bei 48kHz
    "channels":       2,

    "preset":         "warzone",

    # --- Footstep Enhancement ---
    "step_boost_db":      12.0,    # dB Boost im Step-Band
    "step_low_hz":        80,      # Untere Grenze (Thuds)
    "step_high_hz":       3500,    # Obere Grenze (Schritte/Scuff)
    "step_mid_boost_db":  6.0,     # Extra Mid-Boost 1-2.5kHz (Klarheit)

    # --- Bass Cut (Explosionen/Fahrzeuge) ---
    "bass_cut_hz":    60,      # Alles darunter wird steil abgeschnitten

    # --- DT 990 Pro Korrektur ---
    "treble_tame_db": -4.0,    # dB Cut bei 8kHz (DT990 Spitze)
    "mid_lift_db":     2.0,    # dB Boost bei 1.5kHz (DT990 Delle)

    # --- Gunfire Suppression ---
    "gunfire_cut_db": -3.0,    # dB Cut bei 4-6kHz (Schuss-Peaks)

    # --- Dynamic Compression ---
    "comp_ratio":      3.5,    # Kompression (1=aus, 6=extrem)
    "comp_threshold": -20.0,   # dB Schwelle
    "comp_attack":     0.005,  # Sekunden
    "comp_release":    0.08,   # Sekunden

    # --- Noise Gate ---
    "gate_db":        -48.0,   # Unter diesem Pegel = Stille

    # --- Spatial ---
    "spatial_width":   1.4,    # Stereo-Breite (1.0=normal)

    # --- Output ---
    "output_gain_db":  0.0,    # Master-Lautstaerke

    # --- Radar ---
    "show_radar":      True,
}

# === Presets ===
PRESETS = {
    "warzone": {
        "step_boost_db":     12.0,
        "step_low_hz":       80,
        "step_high_hz":      3500,
        "step_mid_boost_db": 6.0,
        "bass_cut_hz":       60,
        "treble_tame_db":   -4.0,
        "mid_lift_db":       2.0,
        "gunfire_cut_db":   -3.0,
        "comp_ratio":        3.5,
        "comp_threshold":   -20.0,
        "spatial_width":     1.4,
        "gate_db":          -48.0,
    },
    "multiplayer": {
        "step_boost_db":     10.0,
        "step_low_hz":       100,
        "step_high_hz":      3000,
        "step_mid_boost_db": 5.0,
        "bass_cut_hz":       50,
        "treble_tame_db":   -3.0,
        "mid_lift_db":       1.5,
        "gunfire_cut_db":   -2.0,
        "comp_ratio":        2.5,
        "comp_threshold":   -18.0,
        "spatial_width":     1.2,
        "gate_db":          -45.0,
    },
    "resurgence": {
        "step_boost_db":     14.0,
        "step_low_hz":       70,
        "step_high_hz":      4000,
        "step_mid_boost_db": 7.0,
        "bass_cut_hz":       70,
        "treble_tame_db":   -5.0,
        "mid_lift_db":       2.5,
        "gunfire_cut_db":   -4.0,
        "comp_ratio":        4.0,
        "comp_threshold":   -22.0,
        "spatial_width":     1.5,
        "gate_db":          -50.0,
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
# AUDIO PROCESSOR — DT 990 Pro Optimiert
# ============================================================
class CompetitiveAudio:
    """Echtzeit Audio-Verarbeitung fuer Warzone Competitive.
    
    Signal-Kette:
    1. Highpass (Bass Cut) → Explosionen/Fahrzeuge weg
    2. DT 990 Korrektur → 8kHz Spitze zaehmen + Mitten anheben
    3. Gunfire Notch → 4-6kHz Schuss-Peaks daempfen
    4. Footstep Boost → 80-3500Hz Band verstaerken
    5. Mid-Clarity Boost → 1-2.5kHz extra fuer Schritt-Klarheit
    6. Noise Gate → Stille bleibt still
    7. Dynamic Compression → Leises lauter, Lautes begrenzen
    8. Spatial Enhancement → Breiteres Stereo
    9. Soft Limiter → Kein Clipping
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.sr = cfg["samplerate"]
        self.nyq = self.sr / 2.0
        self.muted = False
        self.comp_env = 0.0

        # Radar-Daten
        self.step_energy_L = 0.0
        self.step_energy_R = 0.0
        self.peak_dir = 0.0      # -1 links, +1 rechts
        self.step_power = 0.0    # 0-1 Gesamtstaerke

        self._build_filters()

    def _build_filters(self):
        """Baut alle IIR-Filter (Butterworth + Peaking)."""
        sr = self.sr
        nyq = self.nyq
        c = self.cfg

        # --- 1. HIGHPASS: Bass Cut ---
        f_hp = np.clip(c["bass_cut_hz"], 20, nyq * 0.9)
        self.sos_hp = sig.butter(5, f_hp / nyq, btype='highpass', output='sos')

        # --- 2. FOOTSTEP BANDPASS (fuer Boost-Berechnung) ---
        f_lo = np.clip(c["step_low_hz"], 20, nyq * 0.8)
        f_hi = np.clip(c["step_high_hz"], f_lo + 50, nyq * 0.9)
        self.sos_step = sig.butter(3, [f_lo / nyq, f_hi / nyq], btype='bandpass', output='sos')

        # --- 3. MID-CLARITY BANDPASS (1-2.5kHz) ---
        mid_lo = np.clip(1000, 20, nyq * 0.8)
        mid_hi = np.clip(2500, mid_lo + 50, nyq * 0.9)
        self.sos_mid = sig.butter(2, [mid_lo / nyq, mid_hi / nyq], btype='bandpass', output='sos')

        # --- 4. DT 990 TREBLE TAME (8kHz Peaking Notch) ---
        f_treble = 8000.0
        if f_treble < nyq:
            b_t, a_t = sig.iirnotch(f_treble, Q=3.0, fs=sr)
            self.ba_treble = (b_t, a_t)
        else:
            self.ba_treble = None

        # --- 5. GUNFIRE CUT (4-6kHz) ---
        gun_lo = np.clip(4000, 20, nyq * 0.8)
        gun_hi = np.clip(6000, gun_lo + 50, nyq * 0.9)
        self.sos_gun = sig.butter(2, [gun_lo / nyq, gun_hi / nyq], btype='bandpass', output='sos')

        # Gain-Werte vorberechnen
        self.step_gain = 10 ** (c["step_boost_db"] / 20.0)
        self.mid_gain = 10 ** (c["step_mid_boost_db"] / 20.0)
        self.treble_mix = 10 ** (c["treble_tame_db"] / 20.0)  # <1.0 = daempfen
        self.mid_lift = 10 ** (c["mid_lift_db"] / 20.0)
        self.gun_cut = 10 ** (c["gunfire_cut_db"] / 20.0)     # <1.0 = daempfen
        self.out_gain = 10 ** (c["output_gain_db"] / 20.0)
        self.gate_lin = 10 ** (c["gate_db"] / 20.0)
        self.comp_thresh = 10 ** (c["comp_threshold"] / 20.0)

        # Filter-Zustaende pro Kanal (L/R)
        self._init_states()

    def _init_states(self):
        """Initialisiert Filter-Zustaende."""
        z = lambda sos: np.zeros((sos.shape[0], 2))
        self.zi_hp = [z(self.sos_hp), z(self.sos_hp)]
        self.zi_step = [z(self.sos_step), z(self.sos_step)]
        self.zi_mid = [z(self.sos_mid), z(self.sos_mid)]
        self.zi_gun = [z(self.sos_gun), z(self.sos_gun)]
        if self.ba_treble:
            b, a = self.ba_treble
            self.zi_treble = [sig.lfilter_zi(b, a) * 0.0, sig.lfilter_zi(b, a) * 0.0]

    def rebuild(self):
        """Config hat sich geaendert → Filter neu bauen."""
        self._build_filters()

    def process(self, data):
        """Verarbeitet einen Stereo-Block [frames, 2] → [frames, 2]."""
        if self.muted:
            return np.zeros_like(data)

        ch = min(data.shape[1] if data.ndim > 1 else 1, 2)
        if ch >= 2:
            channels = [data[:, 0].copy(), data[:, 1].copy()]
        else:
            channels = [data.flatten().copy(), data.flatten().copy()]

        for i in range(2):  # L=0, R=1
            x = channels[i]

            # === 1. HIGHPASS — Bass Cut ===
            x, self.zi_hp[i] = sig.sosfilt(self.sos_hp, x, zi=self.zi_hp[i])

            # === 2. DT 990 TREBLE TAME ===
            if self.ba_treble and self.treble_mix < 0.99:
                b, a = self.ba_treble
                notched, self.zi_treble[i] = sig.lfilter(b, a, x, zi=self.zi_treble[i])
                # Mix: Original + gedaempftes Treble
                x = x * self.treble_mix + notched * (1.0 - self.treble_mix)

            # === 3. GUNFIRE CUT ===
            if self.gun_cut < 0.99:
                gun_band, self.zi_gun[i] = sig.sosfilt(self.sos_gun, x, zi=self.zi_gun[i])
                x = x - gun_band * (1.0 - self.gun_cut)

            # === 4. FOOTSTEP BOOST ===
            step_band, self.zi_step[i] = sig.sosfilt(self.sos_step, x, zi=self.zi_step[i])
            x = x + step_band * (self.step_gain - 1.0)

            # === 5. MID-CLARITY BOOST ===
            mid_band, self.zi_mid[i] = sig.sosfilt(self.sos_mid, x, zi=self.zi_mid[i])
            x = x + mid_band * (self.mid_lift - 1.0)

            # Radar: Step-Energie tracken
            rms = float(np.sqrt(np.mean(step_band ** 2)))
            if i == 0:
                self.step_energy_L = rms
            else:
                self.step_energy_R = rms

            channels[i] = x

        left, right = channels

        # === 6. NOISE GATE ===
        rms_total = np.sqrt(np.mean(left ** 2) + np.mean(right ** 2))
        if rms_total < self.gate_lin:
            gate_factor = 0.02  # Sehr leise statt komplett aus (kein Klick)
            left *= gate_factor
            right *= gate_factor

        # === 7. DYNAMIC COMPRESSION ===
        ratio = self.cfg["comp_ratio"]
        if ratio > 1.01:
            peak = max(np.max(np.abs(left)), np.max(np.abs(right)), 1e-10)

            # Envelope Follower
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
                makeup = 10 ** (reduce_db * 0.45 / 20.0)  # 45% Makeup
            else:
                comp_g = 1.0
                makeup = 1.0

            left *= comp_g * makeup
            right *= comp_g * makeup

        # === 8. SPATIAL ENHANCEMENT ===
        w = self.cfg["spatial_width"]
        if abs(w - 1.0) > 0.01:
            mid = (left + right) * 0.5
            side = (left - right) * 0.5
            side *= w
            left = mid + side
            right = mid - side

        # === 9. OUTPUT GAIN + SOFT LIMITER ===
        left *= self.out_gain
        right *= self.out_gain
        left = np.tanh(left)
        right = np.tanh(right)

        # Radar-Daten aktualisieren
        total = self.step_energy_L + self.step_energy_R
        if total > 0.0005:
            self.peak_dir = (self.step_energy_R - self.step_energy_L) / total
            self.step_power = min(1.0, total * 20.0)
        else:
            self.step_power *= 0.85

        # Output zusammenbauen
        out = np.column_stack([left, right])
        if data.ndim > 1 and data.shape[1] > 2:
            extra = np.zeros((len(left), data.shape[1] - 2), dtype=np.float32)
            out = np.column_stack([out, extra])
        return out.astype(np.float32)


# ============================================================
# VISUELLER AUDIO-RADAR
# ============================================================
class StepRadar:
    """Zeigt Richtung und Staerke erkannter Schritte."""

    SIZE = 320

    def __init__(self):
        self.cx = self.SIZE // 2
        self.cy = self.SIZE // 2
        self.r = self.SIZE // 2 - 35
        self.trail = deque(maxlen=25)

    def draw(self, direction, power, cfg):
        img = np.zeros((self.SIZE, self.SIZE, 3), dtype=np.uint8)

        # Hintergrund
        cv2.circle(img, (self.cx, self.cy), self.r, (25, 25, 30), -1)
        cv2.circle(img, (self.cx, self.cy), self.r, (50, 50, 60), 2)
        cv2.circle(img, (self.cx, self.cy), self.r // 2, (35, 35, 40), 1)

        # Kreuz
        cv2.line(img, (self.cx - 10, self.cy), (self.cx + 10, self.cy), (50, 50, 60), 1)
        cv2.line(img, (self.cx, self.cy - 10), (self.cx, self.cy + 10), (50, 50, 60), 1)

        # L / R
        cv2.putText(img, "L", (6, self.cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 90), 1)
        cv2.putText(img, "R", (self.SIZE - 18, self.cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 90), 1)

        # Richtungspunkt + Trail
        if power > 0.04:
            self.trail.append((direction, power))

            for idx, (d, p) in enumerate(self.trail):
                a = (idx + 1) / len(self.trail)
                px = int(self.cx + d * self.r * 0.85)
                sz = int(2 + p * 10 * a)
                br = int(60 * a)
                cv2.circle(img, (px, self.cy), sz, (0, br, 0), -1)

            # Aktuell
            px = int(self.cx + direction * self.r * 0.85)
            sz = int(4 + power * 18)
            if power > 0.7:
                col = (0, 80, 255)    # Rot-Orange = SEHR nah
            elif power > 0.4:
                col = (0, 200, 255)   # Gelb = mittel
            else:
                col = (0, 200, 120)   # Gruen = leise

            cv2.circle(img, (px, self.cy), sz, col, -1)
            cv2.line(img, (self.cx, self.cy), (px, self.cy), col, 2)

        # Info-Zeilen
        p = cfg["preset"].upper()
        cv2.putText(img, f"[{p}] STEP:{cfg['step_boost_db']:.0f}dB MID:{cfg['step_mid_boost_db']:.0f}dB",
                    (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 180, 100), 1)
        cv2.putText(img, f"COMP:{cfg['comp_ratio']:.1f}x SPA:{cfg['spatial_width']:.1f} BASS<{cfg['bass_cut_hz']}Hz",
                    (6, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (130, 130, 140), 1)
        cv2.putText(img, f"DT990: TREBLE:{cfg['treble_tame_db']:.0f}dB GUN:{cfg['gunfire_cut_db']:.0f}dB",
                    (6, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (130, 130, 140), 1)

        # Energie-Bar
        bw = int(power * (self.SIZE - 20))
        bc = (0, 180, 100) if power < 0.4 else (0, 200, 255) if power < 0.7 else (0, 80, 255)
        cv2.rectangle(img, (10, self.SIZE - 16), (10 + bw, self.SIZE - 6), bc, -1)
        cv2.rectangle(img, (10, self.SIZE - 16), (self.SIZE - 10, self.SIZE - 6), (50, 50, 60), 1)

        return img


# ============================================================
# GERAETE
# ============================================================
def list_devices():
    devs = sd.query_devices()
    print("\n  Audio-Geraete:")
    print("  " + "-" * 65)
    for i, d in enumerate(devs):
        inp, out = d['max_input_channels'], d['max_output_channels']
        tag = "[IN]    " if inp > 0 and out == 0 else "[OUT]   " if out > 0 and inp == 0 else "[IN/OUT]" if inp > 0 else "[---]   "
        mark = ""
        if i == sd.default.device[0]:
            mark += " << DEFAULT INPUT"
        if i == sd.default.device[1]:
            mark += " << DEFAULT OUTPUT"
        sr = int(d['default_samplerate'])
        print(f"  {i:3d} {tag} {d['name'][:45]:<45} {sr}Hz  in:{inp} out:{out}{mark}")
    print("  " + "-" * 65)
    return devs


def auto_find_capture(devs):
    """Versucht die AVerMedia Capture Card zu finden."""
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

    parser = argparse.ArgumentParser(description="Warzone Competitive Audio — DT 990 Pro")
    parser.add_argument("--list", action="store_true", help="Geraete auflisten und beenden")
    parser.add_argument("--input", type=int, default=None, help="Input Device Index")
    parser.add_argument("--output", type=int, default=None, help="Output Device Index")
    parser.add_argument("--preset", choices=["warzone", "multiplayer", "resurgence"])
    parser.add_argument("--no-radar", action="store_true", help="Ohne visuellen Radar starten")
    args = parser.parse_args()

    cfg = load_config()

    print()
    print("  " + "=" * 55)
    print("  WARZONE COMPETITIVE AUDIO")
    print("  DT 990 Pro + GameDAC Edition")
    print("  " + "=" * 55)

    devs = list_devices()
    if args.list:
        return

    # Input bestimmen
    in_dev = args.input if args.input is not None else cfg["input_device"]
    if in_dev is None:
        auto = auto_find_capture(devs)
        if auto is not None:
            in_dev = auto
            print(f"\n  Capture Card erkannt: [{auto}] {devs[auto]['name']}")
        else:
            in_dev = sd.default.device[0]
            print(f"\n  Kein Capture Card erkannt — Default Input: [{in_dev}]")

    out_dev = args.output if args.output is not None else cfg["output_device"]
    if out_dev is None:
        out_dev = sd.default.device[1]

    cfg["input_device"] = in_dev
    cfg["output_device"] = out_dev

    # Preset
    if args.preset:
        apply_preset(cfg, args.preset)
    elif cfg["preset"] in PRESETS:
        apply_preset(cfg, cfg["preset"])

    if args.no_radar:
        cfg["show_radar"] = False

    # Samplerate vom Device
    sr = int(devs[in_dev]['default_samplerate'])
    cfg["samplerate"] = sr
    ch = min(devs[in_dev]['max_input_channels'], 2)
    cfg["channels"] = ch

    lat_ms = cfg["blocksize"] / sr * 1000

    print(f"\n  Input:    [{in_dev}] {devs[in_dev]['name']}")
    print(f"  Output:   [{out_dev}] {devs[out_dev]['name']}")
    print(f"  SR: {sr}Hz | CH: {ch} | Block: {cfg['blocksize']} (~{lat_ms:.1f}ms)")
    print(f"\n  Preset:   {cfg['preset'].upper()}")
    print(f"  Steps:    +{cfg['step_boost_db']}dB ({cfg['step_low_hz']}-{cfg['step_high_hz']}Hz)")
    print(f"  Mid:      +{cfg['step_mid_boost_db']}dB (1-2.5kHz)")
    print(f"  Bass Cut: <{cfg['bass_cut_hz']}Hz")
    print(f"  DT990:    Treble {cfg['treble_tame_db']}dB @ 8kHz")
    print(f"  Gunfire:  {cfg['gunfire_cut_db']}dB @ 4-6kHz")
    print(f"  Comp:     {cfg['comp_ratio']}:1 @ {cfg['comp_threshold']}dB")
    print(f"  Spatial:  {cfg['spatial_width']}x")
    print(f"  Radar:    {'EIN' if cfg['show_radar'] else 'AUS'}")
    print(f"\n  Tasten: 1=WZ 2=MP 3=RES | Q/W=Step E/R=Comp")
    print(f"          A/S=Bass D/F=Spatial T/Z=Treble G/H=Gain")
    print(f"          M=Mute V=Radar P=Print ESC=Quit")
    print("  " + "=" * 55)

    # Audio Processor
    proc = CompetitiveAudio(cfg)

    # Radar
    radar = StepRadar() if (cfg["show_radar"] and cv2 is not None) else None

    def callback(indata, outdata, frames, time_info, status):
        try:
            outdata[:] = proc.process(indata)[:outdata.shape[0], :outdata.shape[1]]
        except Exception:
            outdata[:] = indata

    # Stream starten
    print("\n  Starte Audio-Stream...")
    try:
        stream = sd.Stream(
            device=(in_dev, out_dev),
            samplerate=sr,
            blocksize=cfg["blocksize"],
            channels=ch,
            dtype='float32',
            callback=callback,
            latency='low',
        )
        stream.start()
        print("  LAEUFT!\n")
    except Exception as e:
        print(f"\n  FEHLER: {e}")
        print(f"  Tipp: python competitive_audio.py --list")
        print(f"        python competitive_audio.py --input <NR> --output <NR>")
        return

    # === Hauptschleife ===
    try:
        while True:
            # Radar
            if radar and cfg["show_radar"] and cv2 is not None:
                img = radar.draw(proc.peak_dir, proc.step_power, cfg)
                cv2.imshow("Step Radar", img)

            if cv2 is not None and cfg["show_radar"]:
                key = cv2.waitKey(33) & 0xFF
            else:
                time.sleep(0.033)
                key = 255

            if key == 27:  # ESC
                break

            # --- Presets ---
            elif key == ord('1'):
                apply_preset(cfg, "warzone"); proc.rebuild()
                print("  >> Preset: WARZONE")
            elif key == ord('2'):
                apply_preset(cfg, "multiplayer"); proc.rebuild()
                print("  >> Preset: MULTIPLAYER")
            elif key == ord('3'):
                apply_preset(cfg, "resurgence"); proc.rebuild()
                print("  >> Preset: RESURGENCE")

            # --- Step Boost ---
            elif key == ord('q'):
                cfg["step_boost_db"] = max(0, cfg["step_boost_db"] - 2); proc.rebuild()
                print(f"  Step Boost: {cfg['step_boost_db']:.0f}dB")
            elif key == ord('w'):
                cfg["step_boost_db"] = min(24, cfg["step_boost_db"] + 2); proc.rebuild()
                print(f"  Step Boost: {cfg['step_boost_db']:.0f}dB")

            # --- Compression ---
            elif key == ord('e'):
                cfg["comp_ratio"] = max(1.0, round(cfg["comp_ratio"] - 0.5, 1)); proc.rebuild()
                print(f"  Compression: {cfg['comp_ratio']:.1f}:1")
            elif key == ord('r'):
                cfg["comp_ratio"] = min(8.0, round(cfg["comp_ratio"] + 0.5, 1)); proc.rebuild()
                print(f"  Compression: {cfg['comp_ratio']:.1f}:1")

            # --- Bass Cut ---
            elif key == ord('a'):
                cfg["bass_cut_hz"] = max(20, cfg["bass_cut_hz"] - 10); proc.rebuild()
                print(f"  Bass Cut: <{cfg['bass_cut_hz']}Hz")
            elif key == ord('s'):
                cfg["bass_cut_hz"] = min(300, cfg["bass_cut_hz"] + 10); proc.rebuild()
                print(f"  Bass Cut: <{cfg['bass_cut_hz']}Hz")

            # --- Spatial ---
            elif key == ord('d'):
                cfg["spatial_width"] = max(0.5, round(cfg["spatial_width"] - 0.1, 1))
                print(f"  Spatial: {cfg['spatial_width']:.1f}x")
            elif key == ord('f'):
                cfg["spatial_width"] = min(2.5, round(cfg["spatial_width"] + 0.1, 1))
                print(f"  Spatial: {cfg['spatial_width']:.1f}x")

            # --- Treble Tame ---
            elif key == ord('t'):
                cfg["treble_tame_db"] = max(-12, cfg["treble_tame_db"] - 1); proc.rebuild()
                print(f"  DT990 Treble: {cfg['treble_tame_db']:.0f}dB")
            elif key == ord('z'):
                cfg["treble_tame_db"] = min(0, cfg["treble_tame_db"] + 1); proc.rebuild()
                print(f"  DT990 Treble: {cfg['treble_tame_db']:.0f}dB")

            # --- Output Gain ---
            elif key == ord('g'):
                cfg["output_gain_db"] = max(-12, cfg["output_gain_db"] - 1); proc.rebuild()
                print(f"  Output: {cfg['output_gain_db']:.0f}dB")
            elif key == ord('h'):
                cfg["output_gain_db"] = min(12, cfg["output_gain_db"] + 1); proc.rebuild()
                print(f"  Output: {cfg['output_gain_db']:.0f}dB")

            # --- Mute ---
            elif key == ord('m'):
                proc.muted = not proc.muted
                print(f"  {'MUTED' if proc.muted else 'UNMUTED'}")

            # --- Radar Toggle ---
            elif key == ord('v'):
                cfg["show_radar"] = not cfg["show_radar"]
                if not cfg["show_radar"] and cv2 is not None:
                    cv2.destroyAllWindows()
                print(f"  Radar: {'EIN' if cfg['show_radar'] else 'AUS'}")

            # --- Print Config ---
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
