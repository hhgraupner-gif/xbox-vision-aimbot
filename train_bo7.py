"""Training script for BO7 custom model - memory optimized"""
from ultralytics import YOLO
import time, os, shutil

print("=== BO7 CUSTOM MODEL TRAINING ===")
start = time.time()

model = YOLO("yolov8n.pt")

results = model.train(
    data="dataset_bo7/data.yaml",
    epochs=20,
    imgsz=320,
    batch=8,
    device="cpu",
    workers=1,
    patience=8,
    save=True,
    project="runs",
    name="bo7_custom",
    exist_ok=True,
    half=False,
    amp=False,
    cache=False,
    verbose=True,
)

elapsed = time.time() - start
print(f"\nTraining fertig in {elapsed/60:.1f} Minuten")

# Export to ONNX
print("Exportiere ONNX...")
best = YOLO("runs/bo7_custom/weights/best.pt")
best.export(format="onnx", imgsz=320, simplify=True, half=True)

# Copy to backend
src = "runs/bo7_custom/weights/best.onnx"
dst = "backend/bo7_custom_320.onnx"
if os.path.exists(src):
    shutil.copy2(src, dst)
    size = os.path.getsize(dst) / (1024*1024)
    print(f"Modell gespeichert: {dst} ({size:.1f} MB)")
print("FERTIG!")
