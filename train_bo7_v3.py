"""Training v3 SAFE - BO7 model with memory management"""
import gc
import os
import shutil

# Force garbage collection before training
gc.collect()

from ultralytics import YOLO
import time

print("=== BO7 MODEL v3 - 5120 Bilder (Safe Mode) ===")
start = time.time()

model = YOLO("yolov8n.pt")

results = model.train(
    data="dataset_bo7_v2/data.yaml",
    epochs=20,
    imgsz=320,
    batch=4,
    device="cpu",
    workers=0,
    patience=10,
    save=True,
    project="runs_v3",
    name="bo7_v3",
    exist_ok=True,
    half=False,
    amp=False,
    cache=False,
    verbose=True,
    mosaic=1.0,
    mixup=0.1,
    scale=0.5,
    fliplr=0.5,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
)

elapsed = time.time() - start
print(f"\nTraining done in {elapsed/60:.1f} min")

# Export ONNX
print("Exporting ONNX...")
best = YOLO("runs_v3/bo7_v3/weights/best.pt")
best.export(format="onnx", imgsz=320, simplify=True, half=True)
src = "runs_v3/bo7_v3/weights/best.onnx"
dst = "backend/bo7_custom_320.onnx"
if os.path.exists(src):
    shutil.copy2(src, dst)
    sz = os.path.getsize(dst) / (1024*1024)
    print(f"Saved: {dst} ({sz:.1f} MB)")
print("FERTIG!")
