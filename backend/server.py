from fastapi import FastAPI, APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone
import asyncio
import base64
import json
import cv2
import numpy as np
from PIL import Image
import io
import mss
import mss.tools

# Import pyautogui for mouse control (aimbot - works on Windows with display)
try:
    import pyautogui
    pyautogui.FAILSAFE = False
    pyautogui.PAUSE = 0
    MOUSE_AIMBOT_AVAILABLE = True
    print("PyAutoGUI loaded - Mouse aimbot available")
except (ImportError, KeyError, Exception):
    pyautogui = None
    MOUSE_AIMBOT_AVAILABLE = False
    print("PyAutoGUI not available (normal on headless server)")

# Import KMBox Net for hardware mouse control (XIM Matrix integration)
try:
    import kmbox_net
    KMBOX_AVAILABLE = True
    print("KMBox Net module loaded")
except ImportError:
    kmbox_net = None
    KMBOX_AVAILABLE = False
    print("KMBox Net not available")

AIMBOT_AVAILABLE = MOUSE_AIMBOT_AVAILABLE or KMBOX_AVAILABLE

# Import controller module
from controller import controller, get_scuf_config, DEFAULT_BINDINGS, SCUF_VALOR_PRO_CONFIG

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection (optional - app works without it)
mongo_url = os.environ.get('MONGO_URL', '')
db = None

if mongo_url:
    try:
        client = AsyncIOMotorClient(mongo_url)
        db = client[os.environ.get('DB_NAME', 'xbox_vision')]
        print("✅ MongoDB connected")
    except Exception as e:
        print(f"⚠️ MongoDB not available: {e}")
        db = None
else:
    print("⚠️ Running without MongoDB (settings won't persist)")

# Create the main app
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Game Profiles
GAME_PROFILES = {
    "default": {
        "name": "Default",
        "description": "Standard detection settings",
        "confidence_threshold": 0.7,
        "aim_sensitivity": 0.3,
        "target_classes": ["person"],
        "aim_point_offset": 0.15,
        "box_color": "#FF003C",
        "priority_targeting": "closest"
    },
    "cod_warzone": {
        "name": "Call of Duty: Warzone",
        "description": "Optimized for Warzone - stable aim",
        "confidence_threshold": 0.75,
        "aim_sensitivity": 0.35,
        "target_classes": ["person"],
        "aim_point_offset": 0.12,
        "box_color": "#FF003C",
        "priority_targeting": "closest"
    },
    "cod_bo7": {
        "name": "Call of Duty: Black Ops 7",
        "description": "Optimized for BO7 multiplayer",
        "confidence_threshold": 0.70,
        "aim_sensitivity": 0.4,
        "target_classes": ["person"],
        "aim_point_offset": 0.10,
        "box_color": "#FF2A6D",
        "priority_targeting": "center"
    },
    "cod_zombies": {
        "name": "Call of Duty: Zombies",
        "description": "Optimized for Zombies mode",
        "confidence_threshold": 0.65,
        "aim_sensitivity": 0.45,
        "target_classes": ["person"],
        "aim_point_offset": 0.20,
        "box_color": "#39FF14",
        "priority_targeting": "closest"
    }
}

# Global settings
detection_settings = {
    "confidence_threshold": 0.45,
    "aim_sensitivity": 0.25,
    "target_classes": ["person"],
    "enabled": True,
    "show_boxes": True,
    "show_crosshair": True,
    "aim_assist_enabled": False,
    "aimbot_enabled": False,
    "aimbot_mode": "kmbox",
    "capture_monitor": 1,
    "capture_region": None,
    "active_profile": "default",
    "aim_point_offset": 0.45,
    "priority_targeting": "closest",
    "smoothing": 0.85,
    "min_target_size": 800,
    "deadzone": 150,
    "use_capture_card": False,
    "capture_device": 0,
    "max_move_px": 12,
    "lock_frames_required": 3,
    "kmbox_ip": "192.168.2.188",
    "kmbox_port": "",
    "kmbox_uuid": "",
    "kmbox_connected": False,
    "trigger_mode": "always",
}

# ============================================================
# STABILIZED AIMBOT - Target Tracker with EMA Smoothing
# ============================================================
import time as _time

class TargetTracker:
    """Tracks a single target across frames with smoothing."""
    def __init__(self):
        self.ema_x = None
        self.ema_y = None
        self.frames_seen = 0
        self.last_seen = 0
        self.locked = False
        self.prev_move_x = 0.0
        self.prev_move_y = 0.0

    def update(self, raw_x, raw_y, alpha=0.3):
        """Update tracker with new raw detection. alpha = EMA weight (lower=smoother)."""
        now = _time.monotonic()
        if self.ema_x is None or (now - self.last_seen) > 0.5:
            # First frame or target was lost for >500ms -> reset
            self.ema_x = float(raw_x)
            self.ema_y = float(raw_y)
            self.frames_seen = 1
            self.locked = False
        else:
            self.ema_x = alpha * raw_x + (1 - alpha) * self.ema_x
            self.ema_y = alpha * raw_y + (1 - alpha) * self.ema_y
            self.frames_seen += 1
        self.last_seen = now

    def get_position(self):
        if self.ema_x is None:
            return None
        return (self.ema_x, self.ema_y)

    def is_stable(self, required_frames=3):
        return self.frames_seen >= required_frames

    def reset(self):
        self.ema_x = None
        self.ema_y = None
        self.frames_seen = 0
        self.locked = False
        self.prev_move_x = 0.0
        self.prev_move_y = 0.0

target_tracker = TargetTracker()


def move_mouse_smoothed(target_x, target_y, frame_width, frame_height):
    """Smooth mouse movement via KMBox Net (primary) or PyAutoGUI (fallback)."""
    try:
        sensitivity = detection_settings.get("aim_sensitivity", 0.25)
        smoothing = detection_settings.get("smoothing", 0.85)
        max_move = detection_settings.get("max_move_px", 12)
        deadzone = detection_settings.get("deadzone", 60)

        center_x = frame_width / 2.0
        center_y = frame_height / 2.0

        dx = target_x - center_x
        dy = target_y - center_y

        distance = (dx * dx + dy * dy) ** 0.5
        if distance < deadzone:
            target_tracker.prev_move_x *= 0.5
            target_tracker.prev_move_y *= 0.5
            return False

        move_x = (dx / frame_width) * sensitivity * 200
        move_y = (dy / frame_height) * sensitivity * 200

        move_x = smoothing * target_tracker.prev_move_x + (1 - smoothing) * move_x
        move_y = smoothing * target_tracker.prev_move_y + (1 - smoothing) * move_y

        mag = (move_x * move_x + move_y * move_y) ** 0.5
        if mag > max_move:
            scale = max_move / mag
            move_x *= scale
            move_y *= scale

        target_tracker.prev_move_x = move_x
        target_tracker.prev_move_y = move_y

        ix = int(round(move_x))
        iy = int(round(move_y))

        if abs(ix) >= 1 or abs(iy) >= 1:
            # Try KMBox Net first (for XIM Matrix), fallback to PyAutoGUI
            mode = detection_settings.get("aimbot_mode", "kmbox")
            if mode == "kmbox" and KMBOX_AVAILABLE and kmbox_net.is_connected():
                kmbox_net.move(ix, iy)
                return True
            elif MOUSE_AIMBOT_AVAILABLE and pyautogui:
                pyautogui.moveRel(ix, iy, duration=0)
                return True

        return False
    except Exception as e:
        logger.error(f"Aimbot move error: {e}")
        return False

# YOLO Model - ONNX Runtime (no PyTorch needed!)
from yolo_onnx import YOLODetector

yolo_model = None
_yolo_load_failed = False

def get_yolo_model():
    global yolo_model, _yolo_load_failed
    if _yolo_load_failed:
        return None
    if yolo_model is None:
        try:
            # Try FPS models first, then fallback to COCO
            candidates = [
                str(ROOT_DIR / 'sunxds_nano_320.onnx'),
                str(ROOT_DIR / 'sunxds_640.onnx'),
                str(ROOT_DIR / 'yolov8n.onnx'),
            ]
            for onnx_path in candidates:
                if os.path.exists(onnx_path):
                    yolo_model = YOLODetector(onnx_path)
                    logger.info(f"YOLO model loaded from {onnx_path}")
                    break
            else:
                _yolo_load_failed = True
                logger.error("No YOLO ONNX model found")
                return None
        except Exception as e:
            _yolo_load_failed = True
            logger.error(f"YOLO model failed to load: {e}")
            return None
    return yolo_model


# Models
class StatusCheck(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class StatusCheckCreate(BaseModel):
    client_name: str

class DetectionSettings(BaseModel):
    confidence_threshold: float = Field(default=0.5, ge=0.1, le=1.0)
    aim_sensitivity: float = Field(default=0.8, ge=0.1, le=1.0)
    target_classes: List[str] = Field(default=["person"])
    enabled: bool = True
    show_boxes: bool = True
    show_crosshair: bool = True
    aim_assist_enabled: bool = False
    capture_monitor: int = 1
    capture_region: Optional[dict] = None
    active_profile: str = "default"
    aim_point_offset: float = Field(default=0.15, ge=0.0, le=0.5)
    priority_targeting: str = "closest"

class GameProfile(BaseModel):
    name: str
    description: str
    confidence_threshold: float
    aim_sensitivity: float
    target_classes: List[str]
    aim_point_offset: float
    box_color: str
    priority_targeting: str

class ControllerBindings(BaseModel):
    aim_assist_toggle: str = "LEFT_SHOULDER"
    aim_hold: str = "LEFT_TRIGGER"
    fire: str = "RIGHT_TRIGGER"
    snap_to_target: str = "RIGHT_THUMB"
    cycle_target: str = "RIGHT_SHOULDER"
    toggle_overlay: str = "BACK"

class ControllerConfig(BaseModel):
    enabled: bool = True
    bindings: Dict[str, str] = Field(default_factory=lambda: DEFAULT_BINDINGS.copy())
    trigger_threshold: float = 0.3
    vibration_enabled: bool = True

# Controller settings (global)
controller_config = {
    "enabled": True,
    "bindings": DEFAULT_BINDINGS.copy(),
    "trigger_threshold": 0.3,
    "vibration_enabled": True,
    "scuf_mode": True  # Enable Scuf Valor Pro optimizations
}

class Detection(BaseModel):
    class_name: str
    confidence: float
    bbox: List[int]  # [x1, y1, x2, y2]
    center: List[int]  # [cx, cy]

class DetectionResult(BaseModel):
    detections: List[Detection]
    frame_width: int
    frame_height: int
    aim_target: Optional[List[int]] = None  # [x, y] of best target
    processing_time_ms: float
    timestamp: str


# Routes
@api_router.get("/")
async def root():
    return {"message": "Xbox Vision AI - Computer Vision Aimbot System"}

@api_router.post("/status", response_model=StatusCheck)
async def create_status_check(input: StatusCheckCreate):
    status_dict = input.model_dump()
    status_obj = StatusCheck(**status_dict)
    
    if db is not None:
        doc = status_obj.model_dump()
        doc['timestamp'] = doc['timestamp'].isoformat()
        _ = await db.status_checks.insert_one(doc)
    
    return status_obj

@api_router.get("/status", response_model=List[StatusCheck])
async def get_status_checks():
    if db is None:
        return []
    
    status_checks = await db.status_checks.find({}, {"_id": 0}).to_list(1000)
    for check in status_checks:
        if isinstance(check['timestamp'], str):
            check['timestamp'] = datetime.fromisoformat(check['timestamp'])
    return status_checks


# Detection Settings
@api_router.get("/settings")
async def get_settings():
    return detection_settings

@api_router.put("/settings")
async def update_settings(settings: DetectionSettings):
    global detection_settings
    detection_settings.update(settings.model_dump())
    
    # Save to MongoDB (if available)
    if db is not None:
        await db.settings.update_one(
            {"type": "detection"},
            {"$set": {**settings.model_dump(), "updated_at": datetime.now(timezone.utc).isoformat()}},
            upsert=True
        )
    return detection_settings


# Aimbot Control
@api_router.post("/aimbot/enable")
async def enable_aimbot():
    """Enable the aimbot - mouse will move to targets"""
    global detection_settings
    detection_settings["aimbot_enabled"] = True
    detection_settings["aim_assist_enabled"] = True
    logger.info("🎯 AIMBOT ENABLED")
    return {"status": "enabled", "message": "Aimbot is now ACTIVE - mouse will move to targets"}

@api_router.post("/aimbot/disable")
async def disable_aimbot():
    """Disable the aimbot"""
    global detection_settings
    detection_settings["aimbot_enabled"] = False
    logger.info("🛑 AIMBOT DISABLED")
    return {"status": "disabled", "message": "Aimbot disabled"}

@api_router.get("/aimbot/status")
async def aimbot_status():
    """Get current aimbot status including tracker state"""
    return {
        "aimbot_enabled": detection_settings.get("aimbot_enabled", False),
        "aim_assist_enabled": detection_settings.get("aim_assist_enabled", False),
        "aimbot_mode": detection_settings.get("aimbot_mode", "kmbox"),
        "mouse_available": MOUSE_AIMBOT_AVAILABLE,
        "kmbox_available": KMBOX_AVAILABLE,
        "kmbox_connected": kmbox_net.is_connected() if KMBOX_AVAILABLE else False,
        "sensitivity": detection_settings.get("aim_sensitivity", 0.25),
        "smoothing": detection_settings.get("smoothing", 0.85),
        "deadzone": detection_settings.get("deadzone", 60),
        "max_move_px": detection_settings.get("max_move_px", 12),
        "lock_frames_required": detection_settings.get("lock_frames_required", 3),
        "tracker_locked": target_tracker.locked,
        "tracker_frames": target_tracker.frames_seen,
    }

@api_router.put("/aimbot/sensitivity")
async def set_aimbot_sensitivity(sensitivity: float = 0.25, smoothing: float = 0.85):
    """Adjust aimbot sensitivity and smoothing"""
    global detection_settings
    detection_settings["aim_sensitivity"] = max(0.05, min(1.0, sensitivity))
    detection_settings["smoothing"] = max(0.0, min(0.95, smoothing))
    target_tracker.reset()
    return {
        "sensitivity": detection_settings["aim_sensitivity"],
        "smoothing": detection_settings["smoothing"]
    }

@api_router.put("/aimbot/tuning")
async def set_aimbot_tuning(deadzone: int = 60, max_move: int = 12, lock_frames: int = 3):
    """Fine-tune aimbot parameters"""
    global detection_settings
    detection_settings["deadzone"] = max(10, min(200, deadzone))
    detection_settings["max_move_px"] = max(2, min(50, max_move))
    detection_settings["lock_frames_required"] = max(1, min(10, lock_frames))
    target_tracker.reset()
    return {
        "deadzone": detection_settings["deadzone"],
        "max_move_px": detection_settings["max_move_px"],
        "lock_frames_required": detection_settings["lock_frames_required"]
    }

@api_router.put("/aimbot/mode")
async def set_aimbot_mode(mode: str = "kmbox"):
    """Set aimbot mode: 'kmbox' (KMBox Net → XIM Matrix) or 'mouse' (PyAutoGUI)"""
    global detection_settings
    if mode in ["kmbox", "mouse"]:
        detection_settings["aimbot_mode"] = mode
    return {"mode": detection_settings["aimbot_mode"]}

@api_router.post("/aimbot/reset-tracker")
async def reset_aimbot_tracker():
    """Reset the target tracker (useful if aim gets stuck)"""
    target_tracker.reset()
    return {"status": "tracker_reset"}


# KMBox Net Configuration
@api_router.post("/kmbox/connect")
async def kmbox_connect(ip: str = "192.168.2.188", port: str = "", uuid: str = ""):
    """Connect to KMBox Net device"""
    global detection_settings
    if not KMBOX_AVAILABLE:
        return {"success": False, "error": "KMBox module not available"}
    if not port or not uuid:
        return {"success": False, "error": "Port und UUID werden benoetigt (vom KMBox Display)"}

    detection_settings["kmbox_ip"] = ip
    detection_settings["kmbox_port"] = port
    detection_settings["kmbox_uuid"] = uuid

    ret = kmbox_net.init(ip, port, uuid)
    if ret == 0:
        detection_settings["kmbox_connected"] = True
        detection_settings["aimbot_mode"] = "kmbox"
        # Start monitor to detect physical mouse button presses (ADS trigger)
        kmbox_net.monitor(10000)
        return {"success": True, "message": f"KMBox Net verbunden: {ip}:{port}"}
    else:
        detection_settings["kmbox_connected"] = False
        return {"success": False, "error": "Verbindung fehlgeschlagen"}

@api_router.get("/kmbox/status")
async def kmbox_status():
    """Get KMBox Net connection status"""
    return {
        "available": KMBOX_AVAILABLE,
        "connected": kmbox_net.is_connected() if KMBOX_AVAILABLE else False,
        "ip": detection_settings.get("kmbox_ip", ""),
        "port": detection_settings.get("kmbox_port", ""),
        "uuid": detection_settings.get("kmbox_uuid", ""),
    }

@api_router.post("/kmbox/test-move")
async def kmbox_test_move(x: int = 10, y: int = 0):
    """Test KMBox Net mouse movement"""
    if not KMBOX_AVAILABLE or not kmbox_net.is_connected():
        return {"success": False, "error": "KMBox nicht verbunden"}
    ret = kmbox_net.move(x, y)
    return {"success": ret == 0, "moved": [x, y]}

@api_router.put("/aimbot/trigger-mode")
async def set_trigger_mode(mode: str = "mouse_right"):
    """Set trigger mode: 'always', 'mouse_right', 'mouse_left'"""
    global detection_settings
    if mode in ["always", "mouse_right", "mouse_left"]:
        detection_settings["trigger_mode"] = mode
    return {"trigger_mode": detection_settings["trigger_mode"]}


# Capture Card Settings
@api_router.put("/capture-card/config")
async def configure_capture_card(enabled: bool = False, device_id: int = 0):
    """Enable/disable capture card and set device index"""
    global detection_settings
    old_device = detection_settings.get("capture_device", -1)
    detection_settings["use_capture_card"] = enabled
    detection_settings["capture_device"] = device_id
    # Reset persistent connection if device changed or disabled
    if not enabled or device_id != old_device:
        release_capture_card()
    return {
        "use_capture_card": detection_settings["use_capture_card"],
        "capture_device": detection_settings["capture_device"]
    }

@api_router.get("/capture-card/test")
async def test_capture_card():
    """Test capture card with high resolution"""
    device_id = detection_settings.get("capture_device", 0)
    try:
        if _platform.system() == "Windows":
            cap = cv2.VideoCapture(device_id, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(device_id)
        if not cap.isOpened():
            return {"success": False, "error": f"Device {device_id} konnte nicht geoeffnet werden"}
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ret, frame = cap.read()
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if not ret:
            return {"success": False, "error": "Frame konnte nicht gelesen werden"}
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_b64 = base64.b64encode(buffer).decode('utf-8')
        return {"success": True, "frame": frame_b64, "width": w, "height": h, "device_id": device_id}
    except Exception as e:
        return {"success": False, "error": str(e)}


# Game Profiles
@api_router.get("/profiles")
async def get_profiles():
    """Get all available game profiles"""
    return {
        "profiles": GAME_PROFILES,
        "active_profile": detection_settings.get("active_profile", "default")
    }

@api_router.get("/profiles/{profile_id}")
async def get_profile(profile_id: str):
    """Get a specific game profile"""
    if profile_id not in GAME_PROFILES:
        raise HTTPException(status_code=404, detail="Profile not found")
    return GAME_PROFILES[profile_id]

@api_router.post("/profiles/{profile_id}/activate")
async def activate_profile(profile_id: str):
    """Activate a game profile and apply its settings"""
    global detection_settings
    
    if profile_id not in GAME_PROFILES:
        raise HTTPException(status_code=404, detail="Profile not found")
    
    profile = GAME_PROFILES[profile_id]
    
    # Apply profile settings
    detection_settings["confidence_threshold"] = profile["confidence_threshold"]
    detection_settings["aim_sensitivity"] = profile["aim_sensitivity"]
    detection_settings["target_classes"] = profile["target_classes"]
    detection_settings["aim_point_offset"] = profile["aim_point_offset"]
    detection_settings["priority_targeting"] = profile["priority_targeting"]
    detection_settings["active_profile"] = profile_id
    
    # Save to MongoDB (if available)
    if db is not None:
        await db.settings.update_one(
            {"type": "detection"},
            {"$set": {**detection_settings, "updated_at": datetime.now(timezone.utc).isoformat()}},
            upsert=True
        )
    
    logger.info(f"Activated game profile: {profile['name']}")
    
    return {
        "message": f"Profile '{profile['name']}' activated",
        "settings": detection_settings
    }


# Controller Endpoints
@api_router.get("/controller/status")
async def get_controller_status():
    """Get current controller status and state"""
    return {
        "available": controller.is_available(),
        "connected": controller.is_connected(),
        "state": controller.to_dict(),
        "config": controller_config,
        "scuf_config": SCUF_VALOR_PRO_CONFIG
    }

@api_router.get("/controller/bindings")
async def get_controller_bindings():
    """Get current controller button bindings"""
    return {
        "bindings": controller_config["bindings"],
        "default_bindings": DEFAULT_BINDINGS,
        "scuf_recommended": SCUF_VALOR_PRO_CONFIG["recommended_bindings"]
    }

@api_router.put("/controller/bindings")
async def update_controller_bindings(bindings: ControllerBindings):
    """Update controller button bindings"""
    global controller_config
    
    new_bindings = bindings.model_dump()
    controller_config["bindings"].update(new_bindings)
    
    # Update controller manager
    for action, button in new_bindings.items():
        controller.set_binding(action, button)
    
    # Save to MongoDB (if available)
    if db is not None:
        await db.controller_config.update_one(
            {"type": "bindings"},
            {"$set": {"bindings": controller_config["bindings"], "updated_at": datetime.now(timezone.utc).isoformat()}},
            upsert=True
        )
    
    return {"message": "Bindings updated", "bindings": controller_config["bindings"]}

@api_router.put("/controller/config")
async def update_controller_config(config: ControllerConfig):
    """Update controller configuration"""
    global controller_config
    controller_config.update(config.model_dump())
    
    controller.trigger_threshold = config.trigger_threshold
    
    if db is not None:
        await db.controller_config.update_one(
            {"type": "config"},
            {"$set": {**config.model_dump(), "updated_at": datetime.now(timezone.utc).isoformat()}},
            upsert=True
        )
    
    return {"message": "Controller config updated", "config": controller_config}

@api_router.post("/controller/vibrate")
async def vibrate_controller(left: float = 0.5, right: float = 0.5, duration: int = 200):
    """Test controller vibration"""
    if not controller.is_connected():
        raise HTTPException(status_code=400, detail="Controller not connected")
    
    controller.vibrate(left, right, duration)
    return {"message": "Vibration sent"}

@api_router.get("/controller/input")
async def get_controller_input():
    """Get current controller input state and processed actions"""
    actions = controller.process_input()
    return {
        "actions": actions,
        "raw_state": controller.to_dict()
    }

@api_router.get("/controller/scuf-config")
async def get_scuf_configuration():
    """Get Scuf Valor Pro specific configuration and tips"""
    return get_scuf_config()


# Screen Capture - MIT CAPTURE CARD SUPPORT (PERSISTENT CONNECTION)
import platform as _platform

# Persistent capture card connection (stays open for speed)
_persistent_cap = None
_persistent_cap_device = -1

def get_capture_card():
    """Get or create persistent capture card connection."""
    global _persistent_cap, _persistent_cap_device
    device_id = detection_settings.get("capture_device", 0)

    # If device changed or not open, reconnect
    if _persistent_cap is None or _persistent_cap_device != device_id or not _persistent_cap.isOpened():
        if _persistent_cap is not None:
            _persistent_cap.release()
        if _platform.system() == "Windows":
            _persistent_cap = cv2.VideoCapture(device_id, cv2.CAP_DSHOW)
        else:
            _persistent_cap = cv2.VideoCapture(device_id)
        if _persistent_cap.isOpened():
            _persistent_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            _persistent_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            _persistent_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            _persistent_cap_device = device_id
            logger.info(f"Capture card opened: Device {device_id}")
        else:
            _persistent_cap = None
            _persistent_cap_device = -1
    return _persistent_cap

def release_capture_card():
    """Release persistent capture card."""
    global _persistent_cap, _persistent_cap_device
    if _persistent_cap is not None:
        _persistent_cap.release()
        _persistent_cap = None
        _persistent_cap_device = -1

def capture_screen(monitor_num: int = 1, region: Optional[dict] = None, use_capture_card: bool = False, capture_device: int = 0):
    """Capture screen using persistent capture card OR mss"""

    # CAPTURE CARD MODE
    if use_capture_card or detection_settings.get("use_capture_card", False):
        cap = get_capture_card()
        if cap is not None and cap.isOpened():
            ret, frame = cap.read()
            if ret:
                return frame
        logger.warning("Capture card frame failed, falling back to screen capture")
    
    # SCREEN CAPTURE MODE (Original)
    with mss.mss() as sct:
        if region:
            monitor = region
        elif monitor_num <= len(sct.monitors) - 1:
            monitor = sct.monitors[monitor_num]
        else:
            monitor = sct.monitors[1]
        
        screenshot = sct.grab(monitor)
        img = np.array(screenshot)
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img


def run_detection(frame: np.ndarray) -> tuple:
    """Run YOLO detection on frame with stabilized targeting."""
    start_time = _time.monotonic()

    detector = get_yolo_model()
    frame_height, frame_width = frame.shape[:2]

    # If YOLO not available, return empty results immediately
    if detector is None:
        return [], None, 0.0, frame_width, frame_height

    conf = detection_settings["confidence_threshold"]
    aim_offset = detection_settings.get("aim_point_offset", 0.15)
    min_size = detection_settings.get("min_target_size", 4000)

    # Crop to center 50% when aimbot is active (reduces false positives)
    if detection_settings.get("aimbot_enabled", False):
        crop_x = frame_width // 4
        crop_y = frame_height // 4
        crop_w = frame_width // 2
        crop_h = frame_height // 2
        cropped = frame[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
        raw_dets = detector.detect(cropped, conf_threshold=conf)
        offset_x, offset_y = crop_x, crop_y
    else:
        raw_dets = detector.detect(frame, conf_threshold=conf)
        offset_x, offset_y = 0, 0

    detections = []
    center_x = frame_width / 2.0
    center_y = frame_height / 2.0

    best_target = None
    min_distance = float('inf')

    for det in raw_dets:
        cls_name = det["class_name"]
        if cls_name not in detection_settings["target_classes"]:
            continue

        x1, y1, x2, y2 = det["bbox"]
        x1 += offset_x
        x2 += offset_x
        y1 += offset_y
        y2 += offset_y

        w_box = x2 - x1
        h_box = y2 - y1

        # Filter: minimum area
        if w_box * h_box < min_size:
            continue

        # Filter: humans are taller than wide (ratio 0.8 - 5.0)
        ratio = h_box / max(1, w_box)
        if ratio < 0.8 or ratio > 5.0:
            continue

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        detections.append({
            "class_name": cls_name,
            "confidence": round(det["confidence"], 3),
            "bbox": [x1, y1, x2, y2],
            "center": [cx, cy]
        })

        # Pick target closest to crosshair
        if detection_settings.get("aimbot_enabled", False):
            target_y = y1 + int(h_box * aim_offset)
            dist = ((cx - center_x) ** 2 + (target_y - center_y) ** 2) ** 0.5
            if dist < min_distance:
                min_distance = dist
                best_target = [cx, target_y]

    processing_time = (_time.monotonic() - start_time) * 1000

    # ---- Stabilized Aimbot ----
    if detection_settings.get("aimbot_enabled", False):
        # Check trigger mode - only aim when trigger is active
        trigger_mode = detection_settings.get("trigger_mode", "always")
        trigger_active = True

        if trigger_mode == "mouse_right" and KMBOX_AVAILABLE and kmbox_net.is_connected():
            trigger_active = kmbox_net.is_mouse_right_pressed()
        elif trigger_mode == "mouse_left" and KMBOX_AVAILABLE and kmbox_net.is_connected():
            trigger_active = kmbox_net.is_mouse_left_pressed()

        lock_frames = detection_settings.get("lock_frames_required", 3)
        ema_alpha = max(0.1, 1.0 - detection_settings.get("smoothing", 0.85))

        if best_target and trigger_active:
            target_tracker.update(best_target[0], best_target[1], alpha=ema_alpha)
            if target_tracker.is_stable(lock_frames):
                pos = target_tracker.get_position()
                if pos:
                    target_tracker.locked = True
                    move_mouse_smoothed(pos[0], pos[1], frame_width, frame_height)
                    best_target = [int(pos[0]), int(pos[1])]
        else:
            if target_tracker.frames_seen > 0:
                target_tracker.frames_seen = max(0, target_tracker.frames_seen - 1)
            if target_tracker.frames_seen == 0:
                target_tracker.reset()

    return detections, best_target, processing_time, frame_width, frame_height


def draw_detections(frame: np.ndarray, detections: list, aim_target: Optional[list] = None) -> np.ndarray:
    """Draw bounding boxes and crosshair on frame"""
    frame_height, frame_width = frame.shape[:2]
    center_x, center_y = frame_width // 2, frame_height // 2
    
    # Draw detections
    if detection_settings["show_boxes"]:
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            conf = det["confidence"]
            
            # Red box for enemies
            color = (0, 0, 255)  # BGR Red
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Corner brackets
            bracket_len = 15
            # Top-left
            cv2.line(frame, (x1, y1), (x1 + bracket_len, y1), color, 3)
            cv2.line(frame, (x1, y1), (x1, y1 + bracket_len), color, 3)
            # Top-right
            cv2.line(frame, (x2, y1), (x2 - bracket_len, y1), color, 3)
            cv2.line(frame, (x2, y1), (x2, y1 + bracket_len), color, 3)
            # Bottom-left
            cv2.line(frame, (x1, y2), (x1 + bracket_len, y2), color, 3)
            cv2.line(frame, (x1, y2), (x1, y2 - bracket_len), color, 3)
            # Bottom-right
            cv2.line(frame, (x2, y2), (x2 - bracket_len, y2), color, 3)
            cv2.line(frame, (x2, y2), (x2, y2 - bracket_len), color, 3)
            
            # Label
            label = f"{det['class_name']} {conf:.0%}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Draw crosshair
    if detection_settings["show_crosshair"]:
        crosshair_color = (255, 240, 0)  # Cyan
        crosshair_size = 20
        cv2.line(frame, (center_x - crosshair_size, center_y), (center_x + crosshair_size, center_y), crosshair_color, 2)
        cv2.line(frame, (center_x, center_y - crosshair_size), (center_x, center_y + crosshair_size), crosshair_color, 2)
        cv2.circle(frame, (center_x, center_y), 5, crosshair_color, 2)
    
    # Draw aim line to target
    if aim_target and detection_settings["aim_assist_enabled"]:
        aim_color = (20, 255, 57)  # Green
        cv2.line(frame, (center_x, center_y), tuple(aim_target), aim_color, 2)
        cv2.circle(frame, tuple(aim_target), 10, aim_color, 3)
    
    return frame


@api_router.get("/capture")
async def capture_frame():
    """Capture and return a single frame with detections"""
    try:
        frame = capture_screen(
            detection_settings["capture_monitor"],
            detection_settings["capture_region"]
        )

        detections, aim_target, proc_time = [], None, 0.0
        h, w = frame.shape[:2]

        if detection_settings["enabled"]:
            try:
                detections, aim_target, proc_time, w, h = run_detection(frame)
                frame = draw_detections(frame, detections, aim_target)
            except Exception as det_err:
                logger.warning(f"Detection failed (showing raw frame): {det_err}")
                # Still draw crosshair even without detection
                frame = draw_detections(frame, [], None)

        # Encode frame to JPEG
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_base64 = base64.b64encode(buffer).decode('utf-8')

        return {
            "frame": frame_base64,
            "detections": detections,
            "aim_target": aim_target,
            "processing_time_ms": round(proc_time, 2),
            "frame_width": w,
            "frame_height": h,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Capture error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/detect")
async def detect_only():
    """Run detection and return results without frame"""
    try:
        frame = capture_screen(
            detection_settings["capture_monitor"],
            detection_settings["capture_region"]
        )
        detections, aim_target, proc_time, w, h = run_detection(frame)
        
        return DetectionResult(
            detections=[Detection(**d) for d in detections],
            frame_width=w,
            frame_height=h,
            aim_target=aim_target,
            processing_time_ms=round(proc_time, 2),
            timestamp=datetime.now(timezone.utc).isoformat()
        )
    except Exception as e:
        logger.error(f"Detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/stream")
async def video_stream():
    """MJPEG video stream with detections"""
    async def generate():
        while True:
            try:
                frame = capture_screen(
                    detection_settings["capture_monitor"],
                    detection_settings["capture_region"]
                )
                
                if detection_settings["enabled"]:
                    detections, aim_target, _, _, _ = run_detection(frame)
                    frame = draw_detections(frame, detections, aim_target)
                
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                
                await asyncio.sleep(0.033)  # ~30 FPS
            except Exception as e:
                logger.error(f"Stream error: {e}")
                await asyncio.sleep(0.1)
    
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


# WebSocket for real-time detection data
@api_router.websocket("/ws/detections")
async def websocket_detections(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket client connected")
    
    try:
        while True:
            try:
                frame = capture_screen(
                    detection_settings["capture_monitor"],
                    detection_settings["capture_region"]
                )
                
                if detection_settings["enabled"]:
                    detections, aim_target, proc_time, w, h = run_detection(frame)
                    frame = draw_detections(frame, detections, aim_target)
                else:
                    detections, aim_target, proc_time = [], None, 0
                    h, w = frame.shape[:2]
                
                # Encode frame
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                
                await websocket.send_json({
                    "frame": frame_base64,
                    "detections": detections,
                    "aim_target": aim_target,
                    "processing_time_ms": round(proc_time, 2),
                    "detection_count": len(detections),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                
                await asyncio.sleep(0.05)  # ~20 FPS for WebSocket
                
            except Exception as e:
                logger.error(f"Detection error: {e}")
                await asyncio.sleep(0.1)
                
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")


# YOLO class names endpoint
@api_router.get("/yolo/classes")
async def get_yolo_classes():
    """Get available YOLO classes"""
    model = get_yolo_model()
    if model is not None:
        return {"classes": list(model.names.values())}
    return {"classes": ["person", "car", "truck", "bus", "motorcycle", "bicycle"]}

@api_router.post("/yolo/reload")
async def reload_yolo():
    """Force reload YOLO model"""
    global yolo_model, _yolo_load_failed
    yolo_model = None
    _yolo_load_failed = False
    model = get_yolo_model()
    if model is not None:
        return {"status": "ok", "message": "YOLO ONNX model loaded"}
    return {"status": "error", "message": "YOLO model failed to load - yolov8n.onnx missing?"}

@api_router.get("/system/info")
async def system_info():
    """Get system info for debugging performance"""
    info = {
        "yolo_loaded": yolo_model is not None,
        "yolo_failed": _yolo_load_failed,
        "yolo_type": "ONNX DirectML" if yolo_model else "not loaded",
        "capture_card_open": _persistent_cap is not None and _persistent_cap.isOpened() if _persistent_cap else False,
        "mouse_available": MOUSE_AIMBOT_AVAILABLE,
        "platform": _platform.system(),
    }
    try:
        import onnxruntime
        info["onnxruntime_version"] = onnxruntime.__version__
        info["onnx_providers"] = onnxruntime.get_available_providers()
    except ImportError:
        info["onnxruntime_version"] = "not installed"
    return info


# Demo mode - generate fake detections for testing
@api_router.get("/demo/frame")
async def demo_frame():
    """Generate a demo frame with fake detections for testing"""
    import random
    
    # Create a dark frame
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[:] = (20, 20, 20)  # Dark background
    
    # Add some grid lines
    for i in range(0, 1280, 100):
        cv2.line(frame, (i, 0), (i, 720), (40, 40, 40), 1)
    for i in range(0, 720, 100):
        cv2.line(frame, (0, i), (1280, i), (40, 40, 40), 1)
    
    # Generate random detections
    detections = []
    num_detections = random.randint(1, 4)
    
    for _ in range(num_detections):
        w = random.randint(80, 150)
        h = random.randint(150, 250)
        x1 = random.randint(100, 1100)
        y1 = random.randint(100, 400)
        x2 = x1 + w
        y2 = y1 + h
        
        detections.append({
            "class_name": "person",
            "confidence": round(random.uniform(0.6, 0.95), 3),
            "bbox": [x1, y1, x2, y2],
            "center": [(x1 + x2) // 2, (y1 + y2) // 2]
        })
    
    # Find best target
    center_x, center_y = 640, 360
    best_target = None
    min_dist = float('inf')
    
    for det in detections:
        cx, cy = det["center"]
        target_y = det["bbox"][1] + int((det["bbox"][3] - det["bbox"][1]) * 0.15)
        dist = ((cx - center_x) ** 2 + (target_y - center_y) ** 2) ** 0.5
        if dist < min_dist:
            min_dist = dist
            best_target = [cx, target_y]
    
    # Draw detections
    frame = draw_detections(frame, detections, best_target if detection_settings["aim_assist_enabled"] else None)
    
    # Add demo watermark
    cv2.putText(frame, "DEMO MODE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)
    
    # Encode
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    frame_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return {
        "frame": frame_base64,
        "detections": detections,
        "aim_target": best_target,
        "processing_time_ms": round(random.uniform(15, 35), 2),
        "frame_width": 1280,
        "frame_height": 720,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "demo": True
    }


# File download endpoint for training files
@api_router.get("/download/{filename}")
async def download_file(filename: str):
    """Download training files (notebook, labels, anleitung)"""
    allowed = {
        "BO7_GPU_Training.ipynb": Path("/app/BO7_GPU_Training.ipynb"),
        "labels_only.zip": Path("/app/labels_only.zip"),
        "ANLEITUNG_GPU_TRAINING.md": Path("/app/ANLEITUNG_GPU_TRAINING.md"),
        "test_scuf.py": Path("/app/test_scuf.py"),
        "scuf_passthrough.py": Path("/app/scuf_passthrough.py"),
        "aimbot_direct.py": Path("/app/aimbot_direct.py"),
        "kmbox_net.py": Path("/app/backend/kmbox_net.py"),
    }
    if filename not in allowed:
        raise HTTPException(status_code=404, detail="Datei nicht gefunden")
    fpath = allowed[filename]
    if not fpath.exists():
        raise HTTPException(status_code=404, detail="Datei existiert nicht auf dem Server")
    return FileResponse(str(fpath), filename=filename)


# Include the router
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    logger.info("Starting Xbox Vision AI Server...")
    # Load settings from MongoDB (if available)
    if db is not None:
        try:
            saved_settings = await db.settings.find_one({"type": "detection"}, {"_id": 0})
            if saved_settings:
                detection_settings.update({k: v for k, v in saved_settings.items() if k != "type" and k != "updated_at"})
                logger.info("Loaded settings from database")
        except Exception as e:
            logger.warning(f"Could not load settings from database: {e}")

@app.on_event("shutdown")
async def shutdown_db_client():
    release_capture_card()
    if db is not None:
        try:
            client.close()
        except Exception:
            pass
