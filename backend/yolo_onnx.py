"""
YOLOv8 ONNX Inference - DirectML GPU (no PyTorch needed)
Works with AMD GPUs via onnxruntime-directml
"""
import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)

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
    """YOLOv8 detector using ONNX Runtime (no PyTorch needed)."""

    def __init__(self, onnx_path, conf_threshold=0.5, iou_threshold=0.45):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_size = 640
        self.session = None
        self.names = {i: name for i, name in enumerate(COCO_NAMES)}
        self._load_model(onnx_path)

    def _load_model(self, onnx_path):
        import onnxruntime as ort
        providers = ort.get_available_providers()
        logger.info(f"ONNX Runtime providers: {providers}")

        # Prefer DirectML (AMD GPU) > CPU
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
        self.input_shape = inp.shape  # e.g. [1, 3, 640, 640]
        logger.info(f"ONNX model loaded: input={inp.name} shape={inp.shape}")

    def preprocess(self, frame):
        """Resize + normalize frame for YOLOv8 input."""
        h, w = frame.shape[:2]
        # Letterbox resize to 640x640 keeping aspect ratio
        scale = min(self.input_size / h, self.input_size / w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad to 640x640
        padded = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        pad_x = (self.input_size - new_w) // 2
        pad_y = (self.input_size - new_h) // 2
        padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        # HWC -> CHW, BGR -> RGB, normalize to 0-1, add batch dim
        blob = padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.expand_dims(blob, axis=0)

        return blob, scale, pad_x, pad_y

    def postprocess(self, output, scale, pad_x, pad_y, orig_h, orig_w, conf_threshold=None):
        """Parse YOLOv8 output and apply NMS."""
        if conf_threshold is None:
            conf_threshold = self.conf_threshold

        # YOLOv8 output shape: [1, 84, 8400] -> transpose to [8400, 84]
        predictions = output[0]
        if predictions.shape[0] == 1:
            predictions = predictions[0]
        if predictions.shape[0] == 84:
            predictions = predictions.T  # [8400, 84]

        # Split into boxes and class scores
        boxes_xywh = predictions[:, :4]
        class_scores = predictions[:, 4:]

        # Get best class per detection
        max_scores = np.max(class_scores, axis=1)
        class_ids = np.argmax(class_scores, axis=1)

        # Filter by confidence
        mask = max_scores > conf_threshold
        boxes_xywh = boxes_xywh[mask]
        max_scores = max_scores[mask]
        class_ids = class_ids[mask]

        if len(boxes_xywh) == 0:
            return []

        # Convert xywh -> xyxy
        boxes_xyxy = np.zeros_like(boxes_xywh)
        boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2  # x1
        boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2  # y1
        boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2  # x2
        boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2  # y2

        # Remove padding and rescale to original image
        boxes_xyxy[:, 0] = (boxes_xyxy[:, 0] - pad_x) / scale
        boxes_xyxy[:, 1] = (boxes_xyxy[:, 1] - pad_y) / scale
        boxes_xyxy[:, 2] = (boxes_xyxy[:, 2] - pad_x) / scale
        boxes_xyxy[:, 3] = (boxes_xyxy[:, 3] - pad_y) / scale

        # Clip to image bounds
        boxes_xyxy[:, 0] = np.clip(boxes_xyxy[:, 0], 0, orig_w)
        boxes_xyxy[:, 1] = np.clip(boxes_xyxy[:, 1], 0, orig_h)
        boxes_xyxy[:, 2] = np.clip(boxes_xyxy[:, 2], 0, orig_w)
        boxes_xyxy[:, 3] = np.clip(boxes_xyxy[:, 3], 0, orig_h)

        # NMS
        indices = self._nms(boxes_xyxy, max_scores, self.iou_threshold)

        results = []
        for i in indices:
            x1, y1, x2, y2 = boxes_xyxy[i].astype(int)
            results.append({
                "class_id": int(class_ids[i]),
                "class_name": self.names.get(int(class_ids[i]), "unknown"),
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
