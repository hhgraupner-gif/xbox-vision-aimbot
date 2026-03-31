"""
AUTO-LABELING SCRIPT
Geht durch alle Screenshots und erstellt YOLO-Labels automatisch
mit dem SunOner FPS-Modell.

Ergebnis: Ein fertiger Datensatz zum Trainieren.

Starten:
  python auto_label.py
"""
import cv2
import numpy as np
import os
import sys
import random
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))
from yolo_onnx import YOLODetector, TARGET_CLASSES

# ============================================================
# EINSTELLUNGEN
# ============================================================
INPUT_FOLDER = "training_data"
OUTPUT_FOLDER = "dataset_bo7"
MODEL_PATH = os.path.join("backend", "sunxds_640.onnx")  # Standard-Modell fuer bessere Labels
CONFIDENCE = 0.30          # Etwas niedriger fuer mehr Erkennungen
IMG_SIZE = 640             # Zielgroesse fuer Training
TRAIN_SPLIT = 0.85         # 85% Training, 15% Validation
# ============================================================


def create_yolo_label(detections, img_w, img_h):
    """Erstellt YOLO-Format Labels (class_id cx cy w h - normalisiert)."""
    lines = []
    # Mapping: Nur Ziel-Klassen labeln
    # 0=player -> 0, 1=bot -> 0 (gleich wie player), 7=head -> 1
    class_map = {
        "player": 0,
        "bot": 0,
        "head": 1,
        "person": 0,  # COCO fallback
    }

    for det in detections:
        cls_name = det["class_name"]
        if cls_name not in class_map:
            continue

        x1, y1, x2, y2 = det["bbox"]
        # Clamp to image bounds
        x1 = max(0, min(x1, img_w))
        y1 = max(0, min(y1, img_h))
        x2 = max(0, min(x2, img_w))
        y2 = max(0, min(y2, img_h))

        bw = x2 - x1
        bh = y2 - y1
        if bw < 5 or bh < 5:
            continue

        # YOLO format: class cx cy w h (normalized 0-1)
        cx = ((x1 + x2) / 2.0) / img_w
        cy = ((y1 + y2) / 2.0) / img_h
        w = bw / img_w
        h = bh / img_h

        class_id = class_map[cls_name]
        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    return "\n".join(lines)


def resize_with_padding(img, target_size):
    """Resize mit Letterbox-Padding auf target_size x target_size."""
    h, w = img.shape[:2]
    scale = min(target_size / h, target_size / w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Padding
    canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas, scale, pad_x, pad_y


def adjust_labels_for_resize(label_text, orig_w, orig_h, target_size, scale, pad_x, pad_y):
    """Passt die Labels an die neue Bildgroesse an."""
    if not label_text.strip():
        return ""

    new_lines = []
    for line in label_text.strip().split("\n"):
        parts = line.split()
        if len(parts) != 5:
            continue
        cls_id = parts[0]
        cx = float(parts[1]) * orig_w
        cy = float(parts[2]) * orig_h
        w = float(parts[3]) * orig_w
        h = float(parts[4]) * orig_h

        # Apply resize + padding
        new_cx = (cx * scale + pad_x) / target_size
        new_cy = (cy * scale + pad_y) / target_size
        new_w = (w * scale) / target_size
        new_h = (h * scale) / target_size

        new_lines.append(f"{cls_id} {new_cx:.6f} {new_cy:.6f} {new_w:.6f} {new_h:.6f}")

    return "\n".join(new_lines)


def main():
    print("=" * 55)
    print("  AUTO-LABELING fuer Black Ops 7 / Warzone")
    print("=" * 55)
    print()

    # Check input
    if not os.path.exists(INPUT_FOLDER):
        print(f"FEHLER: Ordner '{INPUT_FOLDER}' nicht gefunden!")
        sys.exit(1)

    images = sorted([f for f in os.listdir(INPUT_FOLDER)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    print(f"Gefunden: {len(images)} Screenshots in '{INPUT_FOLDER}/'")

    if len(images) == 0:
        print("Keine Bilder gefunden!")
        sys.exit(1)

    # Load model
    if not os.path.exists(MODEL_PATH):
        # Fallback to nano
        alt = os.path.join("backend", "sunxds_nano_320.onnx")
        if os.path.exists(alt):
            model_path = alt
            print(f"Nutze Nano-Modell: {alt}")
        else:
            print("FEHLER: Kein YOLO-Modell gefunden!")
            sys.exit(1)
    else:
        model_path = MODEL_PATH
        print(f"Nutze Standard-Modell: {model_path}")

    print("Lade YOLO-Modell...")
    detector = YOLODetector(model_path, conf_threshold=CONFIDENCE)
    print("Modell geladen!")
    print()

    # Create output structure
    for split in ["train", "val"]:
        os.makedirs(os.path.join(OUTPUT_FOLDER, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_FOLDER, split, "labels"), exist_ok=True)

    # Shuffle and split
    random.seed(42)
    shuffled = images.copy()
    random.shuffle(shuffled)
    split_idx = int(len(shuffled) * TRAIN_SPLIT)
    train_imgs = shuffled[:split_idx]
    val_imgs = shuffled[split_idx:]

    print(f"Split: {len(train_imgs)} Training + {len(val_imgs)} Validation")
    print()

    total_labels = 0
    empty_labels = 0

    for idx, (split_name, img_list) in enumerate([("train", train_imgs), ("val", val_imgs)]):
        print(f"--- {split_name.upper()} ({len(img_list)} Bilder) ---")

        for i, img_name in enumerate(img_list):
            img_path = os.path.join(INPUT_FOLDER, img_name)
            frame = cv2.imread(img_path)
            if frame is None:
                continue

            orig_h, orig_w = frame.shape[:2]

            # Detect with current model
            detections = detector.detect(frame, conf_threshold=CONFIDENCE)
            target_dets = [d for d in detections if d["class_name"] in TARGET_CLASSES or d["class_name"] == "person"]

            # Create label
            label_text = create_yolo_label(target_dets, orig_w, orig_h)

            # Resize image
            resized, scale, pad_x, pad_y = resize_with_padding(frame, IMG_SIZE)

            # Adjust labels for resized image
            if label_text.strip():
                label_text = adjust_labels_for_resize(label_text, orig_w, orig_h, IMG_SIZE, scale, pad_x, pad_y)
                total_labels += label_text.count("\n") + 1
            else:
                empty_labels += 1

            # Save
            base_name = os.path.splitext(img_name)[0]
            out_img = os.path.join(OUTPUT_FOLDER, split_name, "images", f"{base_name}.jpg")
            out_lbl = os.path.join(OUTPUT_FOLDER, split_name, "labels", f"{base_name}.txt")

            cv2.imwrite(out_img, resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
            with open(out_lbl, 'w') as f:
                f.write(label_text)

            if (i + 1) % 50 == 0 or (i + 1) == len(img_list):
                print(f"  [{i+1}/{len(img_list)}] verarbeitet...")

    # Create data.yaml
    yaml_content = f"""# Black Ops 7 / Warzone Custom Dataset
# Auto-labeled mit SunOner FPS-Modell
# {len(images)} Bilder, {total_labels} Labels

path: .
train: train/images
val: val/images

nc: 2
names:
  0: player
  1: head
"""
    yaml_path = os.path.join(OUTPUT_FOLDER, "data.yaml")
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    # Summary
    print()
    print("=" * 55)
    print("  FERTIG!")
    print("=" * 55)
    print(f"  Bilder verarbeitet: {len(images)}")
    print(f"  Labels erstellt:    {total_labels} Objekte")
    print(f"  Leere Labels:      {empty_labels} (keine Gegner erkannt)")
    print(f"  Training:           {len(train_imgs)} Bilder")
    print(f"  Validation:         {len(val_imgs)} Bilder")
    print(f"  Klassen:            0=player, 1=head")
    print(f"  Ausgabe:            {OUTPUT_FOLDER}/")
    print()

    # Check size
    total_size = 0
    for root, dirs, files in os.walk(OUTPUT_FOLDER):
        for f in files:
            total_size += os.path.getsize(os.path.join(root, f))
    size_mb = total_size / (1024 * 1024)
    print(f"  Gesamtgroesse:      {size_mb:.0f} MB")
    print()
    print("  NAECHSTER SCHRITT:")
    print("  Fuehre diese Befehle aus:")
    print()
    print("  cd C:\\Users\\hhgra\\Desktop\\xbox-vision-aimbot")
    print("  git add dataset_bo7/")
    print("  git commit -m \"BO7 training dataset\"")
    print("  git push origin main")
    print()
    print("  Dann sag mir Bescheid und ich trainiere das Modell!")
    print("=" * 55)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAbgebrochen.")
