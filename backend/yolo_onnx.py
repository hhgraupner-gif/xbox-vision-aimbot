"""
YOLO ONNX Inference Wrapper — v6.1
===================================
Unterstuetzt:
  - SunOner End2End Format: (1, 300, 6) = [x1,y1,x2,y2,conf,cls] — NMS eingebaut!
  - Standard YOLO Format: (1, num_classes+4, anchors) = [xywh + class_scores]
  - Auto-Detection: Erkennt Format automatisch anhand Output-Shape
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

# SunOner FPS-Modell Klassen (10) — trainiert auf 30.000+ FPS Game-Screenshots
FPS_NAMES = {
    0: "player", 1: "bot", 2: "weapon", 3: "outline", 4: "dead_body",
    5: "hideout_target_human", 6: "hideout_target_balls", 7: "head",
    8: "smoke", 9: "fire"
}

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

# Ziel-Klassen fuer den Aimbot
TARGET_CLASSES = {"player", "bot", "head", "person"}

# Ignorierte Klassen (sunxds_0.7.8 + allgemein)
IGNORE_CLASSES = {"weapon", "outline", "dead_body", "hideout_target_human",
                  "hideout_target_balls", "smoke", "fire", "teammate",
                  "vehicle", "loot", "other"}


class YOLODetector:
    """YOLO ONNX Runtime Inference — Auto-Format-Erkennung."""

    def __init__(self, onnx_path, input_size=640):
        if ort is None:
            raise ImportError("onnxruntime nicht installiert!")

        self.onnx_path = onnx_path
        self.input_size = input_size
        self.is_end2end = False  # SunOner Format (1, 300, 6)
        self.is_fps_model = False
        self.is_coco_model = False
        self.use_fp16 = False
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

        # Float16 erkennen
        input_type = self.session.get_inputs()[0].type
        self.use_fp16 = 'float16' in input_type or 'Half' in input_type

        # Input-Groesse aus Modell lesen
        if len(self.input_shape) == 4:
            dim = self.input_shape[2]
            if isinstance(dim, int) and dim > 0:
                self.input_size = dim

        # Format + Klassen erkennen
        self._detect_format()

        # Provider Info
        active = self.session.get_providers()
        gpu = "DirectML" if 'DmlExecutionProvider' in active else \
              "CUDA" if 'CUDAExecutionProvider' in active else "CPU"
        dtype = "fp16" if self.use_fp16 else "fp32"
        print(f"  Provider: {gpu} | Typ: {dtype}")

    def _detect_format(self):
        """Erkennt automatisch ob End2End (300,6) oder Standard YOLO Format."""
        out = self.output_shape

        # End2End Format: (1, 300, 6) oder (batch, N, 6)
        if len(out) == 3 and isinstance(out[2], int) and out[2] == 6:
            self.is_end2end = True
            self._detect_classes_from_name()
            fmt = "End2End (NMS eingebaut)"
            print(f"  Modell: {fmt} | {self.input_size}x{self.input_size}")
            if self.is_fps_model:
                print(f"  → FPS-Modell: player, bot, head (ignoriert: waffen, tote, rauch)")
            return

        # Standard YOLO Format: (1, num_cls+4, anchors)
        if len(out) == 3:
            dim1 = out[1]
            if isinstance(dim1, int) and dim1 > 4:
                num_cls = dim1 - 4
                self._assign_classes(num_cls)
                return

        # Dummy inference fallback
        try:
            dummy = np.zeros((1, 3, self.input_size, self.input_size), dtype=np.float32)
            if self.use_fp16:
                dummy = dummy.astype(np.float16)
            result = self.session.run(None, {self.input_name: dummy})[0]
            if len(result.shape) == 3 and result.shape[2] == 6:
                self.is_end2end = True
                self._detect_classes_from_name()
                return
            if len(result.shape) == 3:
                num_cls = result.shape[1] - 4
                self._assign_classes(num_cls)
                return
        except Exception:
            pass

        # Fallback
        self._detect_classes_from_name()

    def _detect_classes_from_name(self):
        """Erkennt Klassen anhand des Dateinamens."""
        fname = os.path.basename(self.onnx_path).lower()
        if "sunxds" in fname or "fps" in fname:
            self.is_fps_model = True
            self.names = FPS_NAMES
            self.num_classes = 10
        elif "bo7" in fname or "custom" in fname or "warzone" in fname or "ruje" in fname:
            self.is_fps_model = True
            self.names = {0: "player", 1: "head"}
            self.num_classes = 2
        else:
            self.is_coco_model = True
            self.names = {i: n for i, n in enumerate(COCO_NAMES)}
            self.num_classes = 80

    def _assign_classes(self, num_cls):
        """Weist Klassen basierend auf Anzahl zu."""
        self.num_classes = num_cls
        if num_cls == 80:
            self.is_coco_model = True
            self.names = {i: n for i, n in enumerate(COCO_NAMES)}
            print(f"  Modell: COCO | {self.input_size}x{self.input_size} | 80 Klassen")
        elif num_cls == 10:
            self.is_fps_model = True
            self.names = FPS_NAMES
            print(f"  Modell: FPS (SunOner) | {self.input_size}x{self.input_size} | 10 Klassen")
        elif num_cls <= 5:
            self.is_fps_model = True
            names_map = {0: "player"}
            if num_cls >= 2: names_map[1] = "head"
            if num_cls >= 3: names_map[2] = "bot"
            if num_cls >= 4: names_map[3] = "dead_body"
            if num_cls >= 5: names_map[4] = "weapon"
            self.names = names_map
            print(f"  Modell: Custom FPS | {self.input_size}x{self.input_size} | {num_cls} Klasse(n)")
        elif num_cls <= 12:
            self.is_fps_model = True
            # sunxds_0.7.8 Klassen (11):
            # 0=player, 1=bot, 2=head, 3=outline, 4=dead_body
            # 5=smoke, 6=fire, 7=teammate, 8=vehicle, 9=loot, 10=weapon
            names_map = {0: "player", 1: "bot", 2: "head", 3: "outline", 4: "dead_body",
                         5: "smoke", 6: "fire", 7: "teammate", 8: "vehicle", 9: "loot", 10: "weapon"}
            self.names = {i: names_map.get(i, f"cls_{i}") for i in range(num_cls)}
            print(f"  Modell: FPS Pro (sunxds) | {self.input_size}x{self.input_size} | {num_cls} Klassen")
            print(f"  -> Zielt auf: player, bot, head")
            print(f"  -> Ignoriert: dead_body, smoke, fire, teammate, vehicle, loot, weapon")
        else:
            self.is_coco_model = True
            self.names = {i: COCO_NAMES[i] if i < len(COCO_NAMES) else f"cls_{i}" for i in range(num_cls)}
            print(f"  Modell: Unbekannt | {self.input_size}x{self.input_size} | {num_cls} Klassen")

    def detect(self, frame, conf_threshold=0.30):
        """YOLO Inference auf einem Frame.

        Returns: Liste von Detections [{bbox, confidence, class_id, class_name}, ...]
        """
        orig_h, orig_w = frame.shape[:2]
        blob = self._preprocess(frame)
        output = self.session.run(None, {self.input_name: blob})[0]

        if self.is_end2end:
            detections = self._postprocess_end2end(output, orig_w, orig_h, conf_threshold)
        else:
            detections = self._postprocess_standard(output, orig_w, orig_h, conf_threshold)

        # Fuer COCO: Nur "person"
        if self.is_coco_model:
            return [d for d in detections if d["class_name"] == "person"]
        return detections

    def _preprocess(self, frame):
        """Resize + Normalize + NCHW."""
        import cv2
        resized = cv2.resize(frame, (self.input_size, self.input_size))
        blob = resized.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)
        blob = np.expand_dims(blob, 0)
        if self.use_fp16:
            blob = blob.astype(np.float16)
        return blob

    def _postprocess_end2end(self, output, orig_w, orig_h, conf_threshold):
        """End2End Format: (1, 300, 6) = [x1,y1,x2,y2,conf,cls]
        Koordinaten sind bereits in Modell-Skala (0-640), NMS bereits angewendet.
        """
        preds = output[0]  # (300, 6)
        scale_x = orig_w / self.input_size
        scale_y = orig_h / self.input_size

        detections = []
        for i in range(preds.shape[0]):
            conf = float(preds[i, 4])
            if conf < conf_threshold:
                continue

            x1 = int(preds[i, 0] * scale_x)
            y1 = int(preds[i, 1] * scale_y)
            x2 = int(preds[i, 2] * scale_x)
            y2 = int(preds[i, 3] * scale_y)
            cls_id = int(preds[i, 5])

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
                "class_id": cls_id,
                "class_name": self.names.get(cls_id, "unknown"),
            })

        return detections

    def _postprocess_standard(self, output, orig_w, orig_h, conf_threshold):
        """Standard YOLO Format: (1, num_cls+4, anchors)."""
        predictions = output[0]
        if predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.T

        boxes_xywh = predictions[:, :4]
        class_scores = predictions[:, 4:]
        max_scores = np.max(class_scores, axis=1)
        class_ids = np.argmax(class_scores, axis=1)

        mask = max_scores > conf_threshold
        boxes_xywh = boxes_xywh[mask]
        max_scores = max_scores[mask]
        class_ids = class_ids[mask]

        if len(boxes_xywh) == 0:
            return []

        scale_x = orig_w / self.input_size
        scale_y = orig_h / self.input_size
        x_c = boxes_xywh[:, 0] * scale_x
        y_c = boxes_xywh[:, 1] * scale_y
        w = boxes_xywh[:, 2] * scale_x
        h = boxes_xywh[:, 3] * scale_y
        x1 = (x_c - w / 2).astype(int)
        y1 = (y_c - h / 2).astype(int)
        x2 = (x_c + w / 2).astype(int)
        y2 = (y_c + h / 2).astype(int)

        # Simple NMS
        detections = []
        indices = np.argsort(-max_scores)
        used = set()
        for idx in indices:
            if idx in used:
                continue
            cid = int(class_ids[idx])
            detections.append({
                "bbox": [int(x1[idx]), int(y1[idx]), int(x2[idx]), int(y2[idx])],
                "confidence": float(max_scores[idx]),
                "class_id": cid,
                "class_name": self.names.get(cid, "unknown"),
            })
            for jdx in indices:
                if jdx not in used and jdx != idx:
                    iou = self._iou(x1[idx], y1[idx], x2[idx], y2[idx],
                                    x1[jdx], y1[jdx], x2[jdx], y2[jdx])
                    if iou > 0.5:
                        used.add(jdx)
            used.add(idx)
        return detections

    @staticmethod
    def _iou(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2):
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0
