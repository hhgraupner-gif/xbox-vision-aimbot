# BO7 Custom Model - GPU Training Anleitung

## Was du brauchst
- Google Account (fuer Colab)
- Die Dateien aus diesem Repo:
  - `BO7_GPU_Training.ipynb` (das Notebook)
  - `labels_only.zip` (die Labels, 1.2 MB)
  - Deine Screenshots (die 5120 resized .jpg Bilder als ZIP)

## Schritt fuer Schritt

### 1. Git Pull
```powershell
cd C:\Users\hhgra\Desktop\xbox-vision-aimbot
git pull
```

### 2. Screenshots als ZIP packen
Falls du deine resized Screenshots noch als losen Ordner hast:
```powershell
Compress-Archive -Path "C:\Users\hhgra\Desktop\xbox-vision-aimbot\screenshots_resized\*" -DestinationPath "C:\Users\hhgra\Desktop\screenshots.zip"
```
(Passe den Pfad an, wo deine resized 320x320 Bilder liegen)

### 3. Google Colab oeffnen
1. Geh zu https://colab.research.google.com
2. Datei -> Notebook hochladen -> waehle `BO7_GPU_Training.ipynb`
3. **WICHTIG**: Laufzeit -> Laufzeittyp aendern -> **T4 GPU** auswaehlen

### 4. Notebook ausfuehren
- Fuehre jede Zelle der Reihe nach aus (Shift+Enter)
- Bei Schritt 2: Lade `labels_only.zip` und deine `screenshots.zip` hoch
- Training dauert ca. **20-40 Minuten** auf T4 GPU
- Am Ende werden die ONNX-Dateien automatisch heruntergeladen

### 5. ONNX-Datei einsetzen
Kopiere die heruntergeladene Datei:
```powershell
copy C:\Users\hhgra\Downloads\bo7_custom_v4_320.onnx C:\Users\hhgra\Desktop\xbox-vision-aimbot\backend\bo7_custom_320.onnx
```

### 6. Aimbot starten
```powershell
cd C:\Users\hhgra\Desktop\xbox-vision-aimbot
python aimbot_direct.py
```
Das Modell wird automatisch als BO7-Custom erkannt (2 Klassen: player + head).

## Was ist besser als vorher?
- **YOLOv8s** statt YOLOv8n (staerkeres Modell)
- **GPU Training** statt CPU (10-20x schneller, bessere Batches)
- **640px Trainings-Aufloesung** (lernt mehr Details)
- **100 Epochen** mit Early Stopping (trainiert bis es optimal ist)
- **Optimierte Augmentation** (bessere Generalisierung)

## Erwartete Ergebnisse
- mAP50: ~0.40-0.60 (vs 0.23 auf CPU)
- Deutlich bessere Spieler-Erkennung in BO7/Warzone
