"""
YOLO ONNX Inference Wrapper — Profi v6
=======================================
Unterstuetzt:
  - YOLO11s COCO (80 Klassen, Klasse 0 = "person") → Primaer-Modell
  - SunOner FPS (10 Klassen: player, bot, head, etc.) → Sekundaer
  - Custom BO7 (1-2 Klassen) → Fallback

Autor: Aimbot Vision System
"""

import os
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None

# ============================================================
# KLASSEN-DEFINITIONEN
# ============================================================

COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]

# SunOner FPS-Modell Klassen (10)
FPS_NAMES = {
    0: "player", 1: "bot", 2: "weapon", 3: "outline", 4: "dead_body",
    5: "hideout_target_human", 6: "hideout_target_balls", 7: "head",
    8: "smoke", 9: "fire"
}

# BO7 Custom Klassen (2)
BO7_NAMES = {0: "player", 1: "head"}

# Ziel-Klassen fuer den Aimbot (nur diese werden getrackt)
TARGET_CLASSES = {"player", "bot", "head", "person"}

# Ignorierte Klassen (werden erkannt aber nicht getrackt)
IGNORE_CLASSES = {"weapon", "outline", "dead_body", "hideout_target_human",
                  "hideout_target_balls", "smoke", "fire"}


class YOLODetector:
    """YOLO ONNX Runtime Inference mit automatischer Modell-Erkennung."""

    def __init__(self, onnx_path, input_size=640):
        if ort is None:
            raise ImportError("onnxruntime nicht installiert!")

        self.onnx_path = onnx_path
        self.input_size = input_size
        self.is_fps_model = False
        self.is_coco_model = False
        self.names = {}
        self.num_classes = 0

        self._load_model()

    def _load_model(self):
        """Lade ONNX Modell mit DirectML (AMD GPU) oder CPU Fallback."""
        providers = []
        try:
            available = ort.get_available_providers()
            if 'DmlExecutionProvider' in available:
                providers.append('DmlExecutionProvider')
            if 'CUDAExecutionProvider' in available:
                providers.append('CUDAExecutionProvider')
        except Exception:
            pass
        providers.append('CPUExecutionProvider')

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(self.onnx_path, sess_options=opts, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_shape = self.session.get_outputs()[0].shape

        # Input-Datentyp erkennen (float32 oder float16)
        input_type = self.session.get_inputs()[0].type
        self.use_fp16 = 'float16' in input_type or 'Half' in input_type
        if self.use_fp16:
            print(f"  Datentyp: float16 (Half Precision)")
        else:
            print(f"  Datentyp: float32")

        # Input-Groesse aus Modell lesen (falls statisch)
        if len(self.input_shape) == 4:
            dim = self.input_shape[2]
            if isinstance(dim, int) and dim > 0:
                self.input_size = dim

        # Klassen-Anzahl bestimmen
        self._detect_model_type()

        # Provider Info
        active = self.session.get_providers()
        gpu = "DirectML" if 'DmlExecutionProvider' in active else \
              "CUDA" if 'CUDAExecutionProvider' in active else "CPU"
        print(f"  ONNX Provider: {gpu}")

    def _detect_model_type(self):
        """Erkennt automatisch ob COCO (80), FPS (10), BO7 (2) oder Single-Class (1)."""
        num_classes = -1

        # Methode 1: Statische Output-Shape
        if self.output_shape and len(self.output_shape) == 3:
            dim1 = self.output_shape[1]
            if isinstance(dim1, int) and dim1 > 4:
                num_classes = dim1 - 4

        # Methode 2: Dummy-Inference (bei dynamischen Dimensionen)
        if num_classes <= 0:
            try:
                h = w = self.input_size
                dummy = np.zeros((1, 3, h, w), dtype=np.float32)
                dummy_out = self.session.run(None, {self.input_name: dummy})[0]
                if len(dummy_out.shape) == 3:
                    num_classes = dummy_out.shape[1] - 4
            except Exception:
                pass

        # Methode 3: Dateiname als letzter Fallback
        if num_classes <= 0:
            fname = os.path.basename(self.onnx_path).lower()
            if "yolo11" in fname or "yolov8" in fname or "coco" in fname:
                num_classes = 80
            elif "sunxds" in fname or "fps" in fname:
                num_classes = 10
            elif "bo7" in fname or "custom" in fname:
                num_classes = 2
            else:
                num_classes = 80  # Standard-Fallback

        self.num_classes = num_classes

        # Klassen-Mapping zuweisen
        if num_classes == 80:
            self.is_coco_model = True
            self.is_fps_model = False
            self.names = {i: name for i, name in enumerate(COCO_NAMES)}
            print(f"  Modell: COCO | {self.input_size}x{self.input_size} | 80 Klassen")
            print(f"  → Erkennt 'person' (Klasse 0) fuer Spieler-Tracking")
        elif num_classes == 10:
            self.is_fps_model = True
            self.is_coco_model = False
            self.names = FPS_NAMES
            print(f"  Modell: FPS (SunOner) | {self.input_size}x{self.input_size} | 10 Klassen")
            print(f"  → Erkennt: player, bot, head (ignoriert: waffen, tote, rauch)")
        elif num_classes <= 2:
            self.is_fps_model = True
            self.is_coco_model = False
            self.names = BO7_NAMES if num_classes == 2 else {0: "person"}
            print(f"  Modell: Custom | {self.input_size}x{self.input_size} | {num_classes} Klasse(n)")
        else:
            self.is_coco_model = True
            self.is_fps_model = False
            self.names = {i: COCO_NAMES[i] if i < len(COCO_NAMES) else f"class_{i}" for i in range(num_classes)}
            print(f"  Modell: Unbekannt | {self.input_size}x{self.input_size} | {num_classes} Klassen")

    def detect(self, frame, conf_threshold=0.35):
        """Fuehrt YOLO Inference auf einem Frame aus.

        Args:
            frame: BGR numpy array (beliebige Groesse)
            conf_threshold: Mindest-Confidence (0.0 - 1.0)

        Returns:
            Liste von Detections: [{"bbox": [x1,y1,x2,y2], "confidence": float,
                                     "class_id": int, "class_name": str}, ...]
        """
        orig_h, orig_w = frame.shape[:2]
        blob = self._preprocess(frame)
        output = self.session.run(None, {self.input_name: blob})[0]
        detections = self._postprocess(output, orig_w, orig_h, conf_threshold)

        # Fuer COCO: Nur "person" zurueckgeben (Klasse 0)
        if self.is_coco_model:
            return [d for d in detections if d["class_name"] == "person"]
        return detections

    def _preprocess(self, frame):
        """Resize + Normalize + NCHW Konvertierung. Auto float16/float32."""
        import cv2
        resized = cv2.resize(frame, (self.input_size, self.input_size))
        blob = resized.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)  # HWC → CHW
        blob = np.expand_dims(blob, 0)   # → NCHW
        if self.use_fp16:
            blob = blob.astype(np.float16)
        return blob

    def _postprocess(self, output, orig_w, orig_h, conf_threshold):
        """NMS + Koordinaten-Transformation."""
        predictions = output[0]  # [84, 8400] oder [5, 8400]

        if predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.T  # → [8400, 84]

        boxes_xywh = predictions[:, :4]
        class_scores = predictions[:, 4:]

        max_scores = np.max(class_scores, axis=1)
        class_ids = np.argmax(class_scores, axis=1)

        # Confidence Filter
        mask = max_scores > conf_threshold
        boxes_xywh = boxes_xywh[mask]
        max_scores = max_scores[mask]
        class_ids = class_ids[mask]

        if len(boxes_xywh) == 0:
            return []

        # xywh → xyxy (Modell-Koordinaten: 0-640)
        scale_x = orig_w / self.input_size
        scale_y = orig_h / self.input_size

        x_center = boxes_xywh[:, 0] * scale_x
        y_center = boxes_xywh[:, 1] * scale_y
        w = boxes_xywh[:, 2] * scale_x
        h = boxes_xywh[:, 3] * scale_y

        x1 = (x_center - w / 2).astype(int)
        y1 = (y_center - h / 2).astype(int)
        x2 = (x_center + w / 2).astype(int)
        y2 = (y_center + h / 2).astype(int)

        # NMS
        detections = []
        indices = np.argsort(-max_scores)
        used = set()

        for idx in indices:
            if idx in used:
                continue

            cid = int(class_ids[idx])
            det = {
                "bbox": [int(x1[idx]), int(y1[idx]), int(x2[idx]), int(y2[idx])],
                "confidence": float(max_scores[idx]),
                "class_id": cid,
                "class_name": self.names.get(cid, "unknown"),
            }
            detections.append(det)

            # Suppress overlapping boxes (IoU > 0.5)
            for jdx in indices:
                if jdx in used or jdx == idx:
                    continue
                iou = self._iou(x1[idx], y1[idx], x2[idx], y2[idx],
                                x1[jdx], y1[jdx], x2[jdx], y2[jdx])
                if iou > 0.5:
                    used.add(jdx)

            used.add(idx)

        return detections

    @staticmethod
    def _iou(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2):
        """Intersection over Union."""
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0
