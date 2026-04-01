"""Resume V3 training from last checkpoint - epoch 8/20"""
import gc
import os
import shutil
import time

gc.collect()

from ultralytics import YOLO

print("=== RESUME BO7 V3 TRAINING (Epoch 9-20) ===")
print(f"Start: {time.strftime('%H:%M:%S')}")
start = time.time()

# Resume from last checkpoint
model = YOLO("runs/detect/runs_v3/bo7_v3/weights/last.pt")

results = model.train(
    resume=True,
)

elapsed = time.time() - start
print(f"\nTraining fertig in {elapsed/60:.1f} min")

# Export to ONNX (fp32 for compatibility)
print("Exportiere ONNX (fp32)...")
best = YOLO("runs/detect/runs_v3/bo7_v3/weights/best.pt")
best.export(format="onnx", imgsz=320, simplify=True, half=False)

src = "runs/detect/runs_v3/bo7_v3/weights/best.onnx"
dst = "backend/bo7_custom_320.onnx"
if os.path.exists(src):
    shutil.copy2(src, dst)
    sz = os.path.getsize(dst) / (1024*1024)
    print(f"ONNX gespeichert: {dst} ({sz:.1f} MB)")
else:
    print(f"FEHLER: {src} nicht gefunden!")

print(f"Ende: {time.strftime('%H:%M:%S')}")
print("=== FERTIG! ===")
