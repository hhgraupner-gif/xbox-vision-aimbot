from fastapi import FastAPI, APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import StreamingResponse
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

# Import controller module
from controller import controller, get_scuf_config, DEFAULT_BINDINGS, SCUF_VALOR_PRO_CONFIG

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

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
        "confidence_threshold": 0.5,
        "aim_sensitivity": 0.8,
        "target_classes": ["person"],
        "aim_point_offset": 0.15,  # 15% from top (head)
        "box_color": "#FF003C",
        "priority_targeting": "closest"  # closest, highest_confidence, center
    },
    "cod_warzone": {
        "name": "Call of Duty: Warzone",
        "description": "Optimized for Warzone - fast targets, high precision",
        "confidence_threshold": 0.45,  # Slightly lower for fast movement
        "aim_sensitivity": 0.95,  # Very responsive
        "target_classes": ["person"],
        "aim_point_offset": 0.12,  # Higher headshot line
        "box_color": "#FF003C",
        "priority_targeting": "closest"
    },
    "cod_bo7": {
        "name": "Call of Duty: Black Ops 7",
        "description": "Optimized for BO7 multiplayer - quick reflexes",
        "confidence_threshold": 0.40,  # Lower threshold for fast-paced
        "aim_sensitivity": 1.0,  # Maximum responsiveness
        "target_classes": ["person"],
        "aim_point_offset": 0.10,  # Aggressive headshot targeting
        "box_color": "#FF2A6D",
        "priority_targeting": "center"  # Prioritize center of screen
    },
    "cod_zombies": {
        "name": "Call of Duty: Zombies",
        "description": "Optimized for Zombies mode - multiple targets",
        "confidence_threshold": 0.35,  # Lower for hordes
        "aim_sensitivity": 0.85,
        "target_classes": ["person"],
        "aim_point_offset": 0.20,  # Center mass for zombies
        "box_color": "#39FF14",
        "priority_targeting": "closest"
    }
}

# Global settings
detection_settings = {
    "confidence_threshold": 0.5,
    "aim_sensitivity": 0.8,
    "target_classes": ["person"],
    "enabled": True,
    "show_boxes": True,
    "show_crosshair": True,
    "aim_assist_enabled": False,
    "capture_monitor": 1,
    "capture_region": None,
    "active_profile": "default",
    "aim_point_offset": 0.15,
    "priority_targeting": "closest"
}

# YOLO Model (lazy load)
yolo_model = None

def get_yolo_model():
    global yolo_model
    if yolo_model is None:
        try:
            from ultralytics import YOLO
            # Use YOLOv8n (nano) for fastest inference
            yolo_model = YOLO('yolov8n.pt')
            logger.info("YOLO model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise HTTPException(status_code=500, detail=f"YOLO model failed to load: {str(e)}")
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
    doc = status_obj.model_dump()
    doc['timestamp'] = doc['timestamp'].isoformat()
    _ = await db.status_checks.insert_one(doc)
    return status_obj

@api_router.get("/status", response_model=List[StatusCheck])
async def get_status_checks():
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
    
    # Save to MongoDB
    await db.settings.update_one(
        {"type": "detection"},
        {"$set": {**settings.model_dump(), "updated_at": datetime.now(timezone.utc).isoformat()}},
        upsert=True
    )
    return detection_settings


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
    
    # Save to MongoDB
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
    
    # Save to MongoDB
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


# Screen Capture
def capture_screen(monitor_num: int = 1, region: Optional[dict] = None):
    """Capture screen using mss"""
    with mss.mss() as sct:
        if region:
            monitor = region
        elif monitor_num <= len(sct.monitors) - 1:
            monitor = sct.monitors[monitor_num]
        else:
            monitor = sct.monitors[1]
        
        screenshot = sct.grab(monitor)
        # Convert to numpy array (BGR format for OpenCV)
        img = np.array(screenshot)
        # Remove alpha channel if present
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img


def run_detection(frame: np.ndarray) -> tuple:
    """Run YOLO detection on frame"""
    import time
    start_time = time.time()
    
    model = get_yolo_model()
    
    # Run inference
    results = model(frame, verbose=False, conf=detection_settings["confidence_threshold"])
    
    detections = []
    frame_height, frame_width = frame.shape[:2]
    center_x, center_y = frame_width // 2, frame_height // 2
    
    best_target = None
    min_distance = float('inf')
    max_confidence = 0
    
    # Get targeting mode from settings
    priority_mode = detection_settings.get("priority_targeting", "closest")
    aim_offset = detection_settings.get("aim_point_offset", 0.15)
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])
            
            # Filter by target classes
            if cls_name not in detection_settings["target_classes"]:
                continue
            
            # Get bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            detections.append({
                "class_name": cls_name,
                "confidence": round(conf, 3),
                "bbox": [x1, y1, x2, y2],
                "center": [cx, cy]
            })
            
            # Find best target based on priority mode
            if detection_settings["aim_assist_enabled"]:
                # Calculate aim point (head area based on offset)
                target_y = y1 + int((y2 - y1) * aim_offset)
                distance = ((cx - center_x) ** 2 + (target_y - center_y) ** 2) ** 0.5
                
                if priority_mode == "closest":
                    if distance < min_distance:
                        min_distance = distance
                        best_target = [cx, target_y]
                elif priority_mode == "highest_confidence":
                    if conf > max_confidence:
                        max_confidence = conf
                        best_target = [cx, target_y]
                elif priority_mode == "center":
                    # Weight by both distance and confidence
                    score = distance / (conf + 0.1)
                    if score < min_distance:
                        min_distance = score
                        best_target = [cx, target_y]
    
    processing_time = (time.time() - start_time) * 1000
    
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
        
        if detection_settings["enabled"]:
            detections, aim_target, proc_time, w, h = run_detection(frame)
            frame = draw_detections(frame, detections, aim_target)
        else:
            detections, aim_target, proc_time, w, h = [], None, 0, frame.shape[1], frame.shape[0]
        
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
    try:
        model = get_yolo_model()
        return {"classes": list(model.names.values())}
    except Exception as e:
        return {"classes": ["person", "car", "truck", "bus", "motorcycle", "bicycle"]}


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
    # Load settings from MongoDB
    saved_settings = await db.settings.find_one({"type": "detection"}, {"_id": 0})
    if saved_settings:
        detection_settings.update({k: v for k, v in saved_settings.items() if k != "type" and k != "updated_at"})
        logger.info("Loaded settings from database")

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
