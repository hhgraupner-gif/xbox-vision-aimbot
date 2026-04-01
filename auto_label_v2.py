"""
AUTO-LABELING v2 - Dual Model + Dead Body Filter
Nutzt SunOner UND BO7 Custom Modell fuer bessere Labels.
Filtert tote Gegner raus (liegende Bounding Boxes).
"""
import cv2
import numpy as np
import os
import sys
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))
from yolo_onnx import YOLODetector, TARGET_CLASSES

INPUT_FOLDER = "training_data"
OUTPUT_FOLDER = "dataset_bo7_v2"
CONFIDENCE = 0.25
IMG_SIZE = 640
TRAIN_SPLIT = 0.85


def is_dead_body(bbox):
    """Erkennt ob eine Bounding Box ein toter Gegner ist (liegt am Boden)."""
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    if h == 0:
        return True
    ratio = w / h
    # Toter Gegner = breiter als hoch (liegt am Boden)
    return ratio > 2.0


def merge_detections(dets1, dets2, iou_threshold=0.5):
    """Merged Erkennungen von zwei Modellen. Entfernt Duplikate via IoU."""
    all_dets = list(dets1)

    for d2 in dets2:
        is_dup = False
        b2 = d2["bbox"]
        for d1 in dets1:
            b1 = d1["bbox"]
            # IoU berechnen
            ix1 = max(b1[0], b2[0])
            iy1 = max(b1[1], b2[1])
            ix2 = min(b1[2], b2[2])
            iy2 = min(b1[3], b2[3])
            iw = max(0, ix2 - ix1)
            ih = max(0, iy2 - iy1)
            inter = iw * ih
            a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
            a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
            union = a1 + a2 - inter
            if union > 0 and inter / union > iou_threshold:
                is_dup = True
                break
        if not is_dup:
            all_dets.append(d2)

    return all_dets


def create_yolo_label(detections, img_w, img_h):
    """YOLO-Format Labels erstellen. Filtert tote Gegner."""
    lines = []
    class_map = {"player": 0, "bot": 0, "head": 1, "person": 0}

    for det in detections:
        cls_name = det["class_name"]
        if cls_name not in class_map:
            continue

        bbox = det["bbox"]
        # Tote Gegner rausfiltern
        if cls_name in ("player", "bot", "person") and is_dead_body(bbox):
            continue

        x1, y1, x2, y2 = bbox
        x1 = max(0, min(x1, img_w))
        y1 = max(0, min(y1, img_h))
        x2 = max(0, min(x2, img_w))
        y2 = max(0, min(y2, img_h))

        bw = x2 - x1
        bh = y2 - y1
        if bw < 5 or bh < 5:
            continue

        cx = ((x1 + x2) / 2.0) / img_w
        cy = ((y1 + y2) / 2.0) / img_h
        w = bw / img_w
        h = bh / img_h

        lines.append(f"{class_map[cls_name]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    return "\n".join(lines)


def resize_with_padding(img, target_size):
    h, w = img.shape[:2]
    scale = min(target_size / h, target_size / w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas, scale, pad_x, pad_y


def adjust_labels_for_resize(label_text, orig_w, orig_h, target_size, scale, pad_x, pad_y):
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
        new_cx = (cx * scale + pad_x) / target_size
        new_cy = (cy * scale + pad_y) / target_size
        new_w = (w * scale) / target_size
        new_h = (h * scale) / target_size
        new_lines.append(f"{cls_id} {new_cx:.6f} {new_cy:.6f} {new_w:.6f} {new_h:.6f}")
    return "\n".join(new_lines)


def main():
    print("=" * 55)
    print("  AUTO-LABELING v2 - Dual Model + Dead Body Filter")
    print("=" * 55)

    if not os.path.exists(INPUT_FOLDER):
        print(f"FEHLER: '{INPUT_FOLDER}' nicht gefunden!")
        sys.exit(1)

    images = sorted([f for f in os.listdir(INPUT_FOLDER)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    print(f"Gefunden: {len(images)} Screenshots")

    # Load both models
    models = []
    model_paths = [
        ("SunOner 640", "backend/sunxds_640.onnx"),
        ("SunOner nano", "backend/sunxds_nano_320.onnx"),
        ("BO7 Custom", "backend/bo7_custom_320.onnx"),
    ]
    for name, path in model_paths:
        if os.path.exists(path):
            print(f"Lade {name}...")
            det = YOLODetector(path, conf_threshold=CONFIDENCE)
            models.append((name, det))
            print(f"  OK!")

    if not models:
        print("FEHLER: Kein Modell gefunden!")
        sys.exit(1)

    print(f"{len(models)} Modelle geladen fuer Dual-Labeling")

    # Create output
    for split in ["train", "val"]:
        os.makedirs(os.path.join(OUTPUT_FOLDER, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_FOLDER, split, "labels"), exist_ok=True)

    random.seed(42)
    shuffled = images.copy()
    random.shuffle(shuffled)
    split_idx = int(len(shuffled) * TRAIN_SPLIT)
    train_imgs = shuffled[:split_idx]
    val_imgs = shuffled[split_idx:]

    print(f"Split: {len(train_imgs)} Train + {len(val_imgs)} Val")
    print()

    total_labels = 0
    total_with_labels = 0
    dead_filtered = 0

    for split_name, img_list in [("train", train_imgs), ("val", val_imgs)]:
        print(f"--- {split_name.upper()} ({len(img_list)} Bilder) ---")

        for i, img_name in enumerate(img_list):
            img_path = os.path.join(INPUT_FOLDER, img_name)
            frame = cv2.imread(img_path)
            if frame is None:
                continue

            orig_h, orig_w = frame.shape[:2]

            # Run all models and merge results
            all_target_dets = []
            for name, det in models:
                dets = det.detect(frame, conf_threshold=CONFIDENCE)
                target_dets = [d for d in dets
                               if d["class_name"] in TARGET_CLASSES or d["class_name"] == "person"]
                # Filter dead_body class from SunOner
                target_dets = [d for d in target_dets if d["class_name"] != "dead_body"]
                all_target_dets = merge_detections(all_target_dets, target_dets)

            # Count dead bodies filtered
            before_dead_filter = len(all_target_dets)
            all_target_dets_alive = [d for d in all_target_dets if not is_dead_body(d["bbox"])]
            dead_filtered += (before_dead_filter - len(all_target_dets_alive))

            label_text = create_yolo_label(all_target_dets, orig_w, orig_h)

            # Resize
            resized, scale, pad_x, pad_y = resize_with_padding(frame, IMG_SIZE)

            if label_text.strip():
                label_text = adjust_labels_for_resize(label_text, orig_w, orig_h, IMG_SIZE, scale, pad_x, pad_y)
                count = label_text.count("\n") + 1
                total_labels += count
                total_with_labels += 1

            base_name = os.path.splitext(img_name)[0]
            out_img = os.path.join(OUTPUT_FOLDER, split_name, "images", f"{base_name}.jpg")
            out_lbl = os.path.join(OUTPUT_FOLDER, split_name, "labels", f"{base_name}.txt")

            cv2.imwrite(out_img, resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
            with open(out_lbl, 'w') as f:
                f.write(label_text)

            if (i + 1) % 100 == 0 or (i + 1) == len(img_list):
                print(f"  [{i+1}/{len(img_list)}] verarbeitet... ({total_with_labels} mit Labels)")

    # data.yaml
    yaml_path = os.path.join(OUTPUT_FOLDER, "data.yaml")
    with open(yaml_path, 'w') as f:
        f.write(f"path: /app/{OUTPUT_FOLDER}\ntrain: train/images\nval: val/images\n\nnc: 2\nnames:\n  0: player\n  1: head\n")

    print()
    print("=" * 55)
    print("  FERTIG!")
    print(f"  Bilder total:       {len(images)}")
    print(f"  Bilder mit Labels:  {total_with_labels} ({100*total_with_labels/len(images):.0f}%)")
    print(f"  Labels total:       {total_labels}")
    print(f"  Tote gefiltert:     {dead_filtered}")
    print(f"  Ausgabe:            {OUTPUT_FOLDER}/")
    print("=" * 55)


if __name__ == "__main__":
    main()
