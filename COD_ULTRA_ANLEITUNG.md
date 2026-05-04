# eRayz COD ULTRA Training — Anleitung

## 🎯 Was dieses Notebook macht

Trainiert ein **CoD/Warzone-spezifisches YOLO-Modell** auf 30k+ Bilder, optimiert für:
- ✅ Zelesis NEO (DML Backend, opset=12)
- ✅ Eigenes `aimbot_direct.py`
- ✅ AMD/Intel/NVIDIA GPU kompatibel

---

## 📋 Vorbereitung (5 Min, einmalig)

### 1. Roboflow API Key holen (falls nicht da)
1. Geh zu [https://app.roboflow.com](https://app.roboflow.com)
2. Login → oben rechts dein Profil → **Settings**
3. Tab **API Keys** → **Private API Key** kopieren
4. Sieht aus wie: `AbCdEf123XyZ456`

### 2. Google Colab vorbereiten
1. Geh zu [https://colab.research.google.com](https://colab.research.google.com)
2. **Datei → Notebook hochladen** → wähle `COD_ULTRA_Training.ipynb`
3. **Laufzeit → Laufzeittyp ändern → T4 GPU**

### 3. Notebook anpassen
- In **Schritt 4** den `RF_API_KEY` ersetzen (Zeile mit `'DEIN_ROBOFLOW_API_KEY_HIER'`)
- Optional: In **Schritt 3** `USE_SUNXDS = True` setzen, wenn du sunxds_0.7.8.pt nutzen willst

---

## ▶️ Ausführen

1. **Schritt 1**: GPU-Check (sollte "Tesla T4" zeigen)
2. **Schritt 2**: Drive-Mount (Bestätigung im Popup → Login)
3. **Schritt 3**: Basis-Modell wählen
4. **Schritt 4**: Datasets ziehen (~10 Min, ~5 GB Download)
5. **Schritt 5**: Datasets mergen (~5 Min)
6. **Schritt 6**: data.yaml schreiben (sofort)
7. **Schritt 7**: **TRAINING** (90–180 Min) ⏳
8. **Schritt 8**: Ergebnisse als Plots anzeigen
9. **Schritt 9**: ONNX Export
10. **Schritt 10**: Download + Drive-Backup

---

## ⚠️ Bei Colab-Disconnect

**Kein Problem!** Auto-Save zu Drive ist aktiv. Wenn die Session abbricht:
1. Notebook neu öffnen
2. Schritte 1–6 wiederholen
3. **Schritt 7** wird automatisch aus letztem Checkpoint weitermachen (über `last.pt`)

---

## 🎮 Modell in Zelesis NEO laden

Sobald `erayz_cod_ultra.onnx` heruntergeladen ist:

1. Datei in beliebigen Ordner legen, z.B. `C:\Users\<Name>\Downloads\`
2. Zelesis öffnen → **KI-Tab**
3. Neben "Modell" → **"+"** klicken
4. `erayz_cod_ultra.onnx` auswählen

### Empfohlene Zelesis-Settings für dieses Modell:
| Setting | Wert |
|---|---|
| KI-Blob-Größe | **640** |
| Konfidenzschwelle | **0.40** |
| NMS-Schwellenwert | **0.50** |
| Max. Erkennungen | **5** |
| Backend | **DML** (AMD/Intel) |

---

## 💡 Tipps für maximale Genauigkeit

### Wenn das erste Modell nicht reicht:
1. **Zweiter Trainings-Lauf**: nutze `erayz_cod_ultra.pt` als Basis statt yolov8m → noch spezifischer
2. **Mehr Daten**: Füge eigene BO7-Screenshots zum Dataset (Auto-Label mit `auto_label_v2.py`)
3. **Größeres Modell**: Wechsle zu `yolov8l.pt` (Achtung: T4 OOM-Risiko, batch auf 8 senken)

### Wenn das Training in Colab zu langsam ist:
- **Colab Pro** (10€/Monat) → A100 GPU = 5× schneller
- **Kaggle** (kostenlos, 30h/Woche) → P100 GPU
- **Lokal mit RTX**: Dieses Notebook läuft auch lokal mit `pip install ultralytics` und einem RTX 3060+

---

## 🔧 Troubleshooting

**Problem: "OOM Error" beim Training**
→ In Schritt 7: `batch=16` auf `batch=8` reduzieren

**Problem: "Roboflow rate limit"**
→ Wechsle in Schritt 4 die Reihenfolge der `versions` (älteste zuerst), oft hilft das

**Problem: "Dataset xy konnte nicht geladen werden"**
→ Dataset wurde vom Owner gelöscht, einfach übersprungen werden

**Problem: ONNX in Zelesis lädt nicht**
→ Prüfe: opset=12 ✓, imgsz=640 ✓, format='onnx' ✓
→ Versuch in Zelesis Backend zu wechseln: DML → CPU (für Test)

**Problem: Drive-Speicher voll**
→ Lösche alte `runs/detect/*` Ordner aus Drive nach dem Training
