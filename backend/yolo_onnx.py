"""
YOLOv8 ONNX Inference - DirectML GPU (no PyTorch needed)
Works with AMD GPUs via onnxruntime-directml

Supports both:
  - Generic COCO models (80 classes, fp32, 640x640)
  - SunOner FPS models (10 classes, fp16, 320x320 or 640x640)
"""
import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)

# BO7 Custom model class names (2 classes)
BO7_NAMES = {
    0: "player",
    1: "head",
}

# SunOner FPS-Aimbot class names (sunxds_0.2.1)
FPS_NAMES = {
    0: "player",
    1: "bot",
    2: "weapon",
    3: "outline",
    4: "dead_body",
    5: "hideout_target_human",
    6: "hideout_target_balls",
    7: "head",
    8: "smoke",
    9: "fire",
}

# Target classes for aimbot (only aim at these)
TARGET_CLASSES = {"player", "bot", "head"}

# COCO class names (YOLOv8 default)
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
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]


class YOLODetector:
    """YOLOv8 detector using ONNX Runtime (no PyTorch needed).
    Auto-detects model type (FPS vs COCO) based on output dimensions.
    """

    def __init__(self, onnx_path, conf_threshold=0.5, iou_threshold=0.45):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.session = None
        self.is_fps_model = False
        self.use_fp16 = False
        self.input_size = 640
        self.names = {i: name for i, name in enumerate(COCO_NAMES)}
        self._load_model(onnx_path)

    def _load_model(self, onnx_path):
        import onnxruntime as ort
        providers = ort.get_available_providers()
        logger.info(f"ONNX Runtime providers: {providers}")

        selected = []
        if 'DmlExecutionProvider' in providers:
            selected.append('DmlExecutionProvider')
            logger.info("Using DirectML (AMD GPU) for inference")
        selected.append('CPUExecutionProvider')

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(onnx_path, sess_options=opts, providers=selected)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.input_shape = inp.shape

        # Detect input size from model shape
        if len(self.input_shape) == 4:
            dim = self.input_shape[2]
            if isinstance(dim, int) and dim > 0:
                self.input_size = dim
            # Sonst: Default 640 bleibt (fuer dynamische Modelle)

        # Detect fp16
        self.use_fp16 = 'float16' in inp.type
        dtype_str = "fp16" if self.use_fp16 else "fp32"

        # Detect model type by running dummy inference
        out = self.session.get_outputs()[0]
        out_shape = out.shape
        # YOLOv8 output: [1, 4+num_classes, num_detections]
        # For BO7 custom: [1, 6, N] -> 6-4 = 2 classes
        # For FPS model: [1, 14, N] -> 14-4 = 10 classes
        # For COCO: [1, 84, N] -> 84-4 = 80 classes
        num_classes = -1
        if out_shape and len(out_shape) == 3:
            dim1 = out_shape[1]
            if isinstance(dim1, int):
                num_classes = dim1 - 4

        # If dim1 was dynamic/symbolic, run dummy inference to find actual shape
        if num_classes <= 0:
            try:
                dummy = np.zeros((1, 3, self.input_size, self.input_size), dtype=np.float16 if self.use_fp16 else np.float32)
                dummy_out = self.session.run(None, {self.input_name: dummy})[0]
                num_classes = dummy_out.shape[1] - 4
                logger.info(f"Dummy inference: output shape {dummy_out.shape}, {num_classes} classes")
            except Exception as e:
                logger.warning(f"Dummy inference failed: {e}")
                # Fallback: check filename
                import os
                fname = os.path.basename(onnx_path).lower()
                if "bo7" in fname or "custom" in fname:
                    num_classes = 2
                elif "sunxds" in fname or "fps" in fname:
                    num_classes = 10

        if num_classes == 1:
            self.is_fps_model = True
            self.names = {0: "person"}
            logger.info(f"Single-class model detected (1 class: person/enemy)")
        elif num_classes == 2:
            self.is_fps_model = True
            self.names = BO7_NAMES
            logger.info(f"BO7 custom model detected ({num_classes} classes)")
        elif num_classes == 10:
            self.is_fps_model = True
            self.names = FPS_NAMES
            logger.info(f"FPS aimbot model detected ({num_classes} classes)")
        elif num_classes == 80:
            self.is_fps_model = False
            self.names = {i: name for i, name in enumerate(COCO_NAMES)}
            logger.info(f"COCO model detected ({num_classes} classes)")
        else:
            logger.info(f"Unknown model with {num_classes} classes")

        model_type = "FPS" if self.is_fps_model else "COCO"
        logger.info(f"Model loaded: {model_type} {dtype_str} {self.input_size}x{self.input_size}")
        print(f"      Modell: {model_type} | {dtype_str} | {self.input_size}x{self.input_size}")

    def preprocess(self, frame):
        """Resize + normalize frame for YOLOv8 input."""
        h, w = frame.shape[:2]
        scale = min(self.input_size / h, self.input_size / w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        padded = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        pad_x = (self.input_size - new_w) // 2
        pad_y = (self.input_size - new_h) // 2
        padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        # HWC -> CHW, BGR -> RGB, normalize to 0-1
        blob = padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        if self.use_fp16:
            blob = blob.astype(np.float16)
        blob = np.expand_dims(blob, axis=0)

        return blob, scale, pad_x, pad_y

    def postprocess(self, output, scale, pad_x, pad_y, orig_h, orig_w, conf_threshold=None):
        """Parse YOLOv8 output and apply NMS."""
        if conf_threshold is None:
            conf_threshold = self.conf_threshold

        predictions = output[0]
        # Convert fp16 to fp32 for post-processing
        if predictions.dtype == np.float16:
            predictions = predictions.astype(np.float32)

        if predictions.shape[0] == 1:
            predictions = predictions[0]
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

        # Convert xywh -> xyxy
        boxes_xyxy = np.zeros_like(boxes_xywh)
        boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
        boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
        boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
        boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2

        # Remove padding and rescale
        boxes_xyxy[:, 0] = (boxes_xyxy[:, 0] - pad_x) / scale
        boxes_xyxy[:, 1] = (boxes_xyxy[:, 1] - pad_y) / scale
        boxes_xyxy[:, 2] = (boxes_xyxy[:, 2] - pad_x) / scale
        boxes_xyxy[:, 3] = (boxes_xyxy[:, 3] - pad_y) / scale

        boxes_xyxy[:, 0] = np.clip(boxes_xyxy[:, 0], 0, orig_w)
        boxes_xyxy[:, 1] = np.clip(boxes_xyxy[:, 1], 0, orig_h)
        boxes_xyxy[:, 2] = np.clip(boxes_xyxy[:, 2], 0, orig_w)
        boxes_xyxy[:, 3] = np.clip(boxes_xyxy[:, 3], 0, orig_h)

        indices = self._nms(boxes_xyxy, max_scores, self.iou_threshold)

        results = []
        for i in indices:
            x1, y1, x2, y2 = boxes_xyxy[i].astype(int)
            cid = int(class_ids[i])
            results.append({
                "class_id": cid,
                "class_name": self.names.get(cid, "unknown"),
                "confidence": float(max_scores[i]),
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "center": [int((x1 + x2) // 2), int((y1 + y2) // 2)]
            })
        return results

    def _nms(self, boxes, scores, iou_threshold):
        """Simple NMS implementation."""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        return keep

    def detect(self, frame, conf_threshold=None):
        """Run detection on a frame. Returns list of detections."""
        orig_h, orig_w = frame.shape[:2]
        blob, scale, pad_x, pad_y = self.preprocess(frame)
        output = self.session.run(None, {self.input_name: blob})
        return self.postprocess(output, scale, pad_x, pad_y, orig_h, orig_w, conf_threshold)

    def detect_targets_only(self, frame, conf_threshold=None):
        """Run detection and return only target classes (player, bot, head).
        For FPS models, filters out weapons, dead_bodies, smoke, fire etc.
        For COCO models, returns only 'person' class.
        """
        all_dets = self.detect(frame, conf_threshold)
        if self.is_fps_model:
            return [d for d in all_dets if d["class_name"] in TARGET_CLASSES]
        else:
            return [d for d in all_dets if d["class_name"] == "person"]
