"""Training script v2 - higher quality BO7 model"""
from ultralytics import YOLO
import time, os, shutil

print("=== BO7 CUSTOM MODEL v2 (High Quality) ===")
start = time.time()

model = YOLO("yolov8n.pt")

results = model.train(
    data="dataset_bo7/data.yaml",
    epochs=40,
    imgsz=640,
    batch=4,
    device="cpu",
    workers=1,
    patience=15,
    save=True,
    project="runs_v2",
    name="bo7_hq",
    exist_ok=True,
    half=False,
    amp=False,
    cache=False,
    verbose=True,
    lr0=0.005,
    lrf=0.01,
    mosaic=1.0,
    mixup=0.15,
    copy_paste=0.1,
    degrees=5.0,
    translate=0.15,
    scale=0.5,
    flipud=0.0,
    fliplr=0.5,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
)

elapsed = time.time() - start
print(f"\nTraining done in {elapsed/60:.1f} min")

# Export both 640 and 320 ONNX
print("Exporting ONNX models...")
best = YOLO("runs_v2/bo7_hq/weights/best.pt")

# 640 version (accurate)
best.export(format="onnx", imgsz=640, simplify=True, half=True)
src640 = "runs_v2/bo7_hq/weights/best.onnx"
dst640 = "backend/bo7_custom_640.onnx"
if os.path.exists(src640):
    shutil.copy2(src640, dst640)
    print(f"  {dst640} ({os.path.getsize(dst640)/1024/1024:.1f} MB)")

# 320 version (fast)
best.export(format="onnx", imgsz=320, simplify=True, half=True)
src320 = "runs_v2/bo7_hq/weights/best.onnx"
dst320 = "backend/bo7_custom_320.onnx"
if os.path.exists(src320):
    shutil.copy2(src320, dst320)
    print(f"  {dst320} ({os.path.getsize(dst320)/1024/1024:.1f} MB)")

print("FERTIG!")
