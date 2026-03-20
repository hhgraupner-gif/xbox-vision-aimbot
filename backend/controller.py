"""
Xbox Controller Input Handler for Scuf Valor Pro
Supports: Button mapping, trigger detection, stick input
"""
import logging
import asyncio
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

# Try to import XInput (Windows only)
try:
    import xinput
    XINPUT_AVAILABLE = True
    logger.info("XInput loaded successfully - Controller support enabled")
except ImportError:
    XINPUT_AVAILABLE = False
    logger.warning("XInput not available - Running in simulation mode")


class ControllerButton(Enum):
    """Xbox Controller Button Names"""
    A = "A"
    B = "B"
    X = "X"
    Y = "Y"
    LB = "LEFT_SHOULDER"
    RB = "RIGHT_SHOULDER"
    LT = "LEFT_TRIGGER"  # Analog
    RT = "RIGHT_TRIGGER"  # Analog
    LS = "LEFT_THUMB"  # Stick click
    RS = "RIGHT_THUMB"  # Stick click
    START = "START"
    BACK = "BACK"  # Select/View button
    DPAD_UP = "DPAD_UP"
    DPAD_DOWN = "DPAD_DOWN"
    DPAD_LEFT = "DPAD_LEFT"
    DPAD_RIGHT = "DPAD_RIGHT"
    # Scuf Paddles are mapped to other buttons
    PADDLE_1 = "PADDLE_1"  # Usually mapped to A or Jump
    PADDLE_2 = "PADDLE_2"  # Usually mapped to B or Crouch
    PADDLE_3 = "PADDLE_3"
    PADDLE_4 = "PADDLE_4"


@dataclass
class ControllerState:
    """Current state of the controller"""
    connected: bool = False
    buttons: Dict[str, bool] = None
    left_trigger: float = 0.0
    right_trigger: float = 0.0
    left_stick: tuple = (0.0, 0.0)
    right_stick: tuple = (0.0, 0.0)
    
    def __post_init__(self):
        if self.buttons is None:
            self.buttons = {}


# Default button bindings for Aimbot (Scuf Valor Pro optimized)
DEFAULT_BINDINGS = {
    # Aim Assist Toggle - LB (Left Bumper) - Quick toggle while aiming
    "aim_assist_toggle": "LEFT_SHOULDER",
    
    # Hold to Aim - LT (Left Trigger) - Standard ADS
    "aim_hold": "LEFT_TRIGGER",
    
    # Fire/Shoot - RT (Right Trigger) - Standard fire
    "fire": "RIGHT_TRIGGER",
    
    # Quick Snap to Target - RS Click - Instant aim adjustment
    "snap_to_target": "RIGHT_THUMB",
    
    # Cycle Target - RB (Right Bumper) - Switch between detected targets
    "cycle_target": "RIGHT_SHOULDER",
    
    # Toggle Detection Overlay - BACK/View - Show/Hide boxes
    "toggle_overlay": "BACK",
    
    # Emergency Disable - Both Bumpers - Safety disable all assists
    "emergency_disable": ["LEFT_SHOULDER", "RIGHT_SHOULDER"],
    
    # Scuf Paddle Bindings (if configured)
    # Paddle 1 (left rear) - Quick aim assist pulse
    "paddle_aim_pulse": "PADDLE_1",
    # Paddle 2 (right rear) - Lock target
    "paddle_lock_target": "PADDLE_2",
}


class ControllerManager:
    """Manages Xbox Controller input for aimbot functionality"""
    
    def __init__(self, controller_index: int = 0):
        self.controller_index = controller_index
        self.bindings = DEFAULT_BINDINGS.copy()
        self.callbacks: Dict[str, Callable] = {}
        self.state = ControllerState()
        self.previous_state = ControllerState()
        self.running = False
        self.trigger_threshold = 0.3  # Trigger activation threshold
        self.aim_assist_active = False
        self.target_locked = False
        self.current_target_index = 0
        
    def is_available(self) -> bool:
        """Check if controller support is available"""
        return XINPUT_AVAILABLE
    
    def is_connected(self) -> bool:
        """Check if controller is connected"""
        if not XINPUT_AVAILABLE:
            return False
        try:
            connected = xinput.get_connected()
            return connected[self.controller_index]
        except Exception:
            return False
    
    def get_state(self) -> ControllerState:
        """Get current controller state"""
        if not XINPUT_AVAILABLE:
            return ControllerState(connected=False)
        
        try:
            if not self.is_connected():
                return ControllerState(connected=False)
            
            state = xinput.get_state(self.controller_index)
            buttons = xinput.get_button_values(state)
            triggers = xinput.get_trigger_values(state)
            thumbs = xinput.get_thumb_values(state)
            
            return ControllerState(
                connected=True,
                buttons=buttons,
                left_trigger=triggers[0],
                right_trigger=triggers[1],
                left_stick=thumbs[0],
                right_stick=thumbs[1]
            )
        except Exception as e:
            logger.error(f"Error reading controller state: {e}")
            return ControllerState(connected=False)
    
    def is_button_pressed(self, button_name: str) -> bool:
        """Check if a specific button is currently pressed"""
        if not self.state.connected:
            return False
        return self.state.buttons.get(button_name, 0) == 1
    
    def is_button_just_pressed(self, button_name: str) -> bool:
        """Check if button was just pressed (edge detection)"""
        current = self.state.buttons.get(button_name, 0)
        previous = self.previous_state.buttons.get(button_name, 0)
        return current == 1 and previous == 0
    
    def is_trigger_active(self, trigger: str) -> bool:
        """Check if trigger is pulled past threshold"""
        if trigger == "LEFT_TRIGGER":
            return self.state.left_trigger > self.trigger_threshold
        elif trigger == "RIGHT_TRIGGER":
            return self.state.right_trigger > self.trigger_threshold
        return False
    
    def get_trigger_value(self, trigger: str) -> float:
        """Get analog trigger value (0.0 - 1.0)"""
        if trigger == "LEFT_TRIGGER":
            return self.state.left_trigger
        elif trigger == "RIGHT_TRIGGER":
            return self.state.right_trigger
        return 0.0
    
    def set_binding(self, action: str, button: str):
        """Change button binding for an action"""
        self.bindings[action] = button
    
    def get_bindings(self) -> Dict[str, str]:
        """Get current button bindings"""
        return self.bindings.copy()
    
    def register_callback(self, action: str, callback: Callable):
        """Register callback for an action"""
        self.callbacks[action] = callback
    
    def vibrate(self, left_motor: float = 0.0, right_motor: float = 0.0, duration_ms: int = 200):
        """Vibrate controller (haptic feedback)"""
        if not XINPUT_AVAILABLE or not self.is_connected():
            return
        
        try:
            # Convert 0.0-1.0 to 0-65535
            left = int(left_motor * 65535)
            right = int(right_motor * 65535)
            xinput.set_vibration(self.controller_index, left, right)
            
            # Auto-stop vibration after duration
            async def stop_vibration():
                await asyncio.sleep(duration_ms / 1000)
                xinput.set_vibration(self.controller_index, 0, 0)
            
            asyncio.create_task(stop_vibration())
        except Exception as e:
            logger.error(f"Vibration error: {e}")
    
    def feedback_target_locked(self):
        """Haptic feedback when target is locked"""
        self.vibrate(left_motor=0.3, right_motor=0.8, duration_ms=150)
    
    def feedback_aim_assist_toggle(self):
        """Haptic feedback when aim assist is toggled"""
        self.vibrate(left_motor=0.5, right_motor=0.5, duration_ms=100)
    
    def process_input(self) -> Dict[str, Any]:
        """Process controller input and return action states"""
        self.previous_state = self.state
        self.state = self.get_state()
        
        if not self.state.connected:
            return {"connected": False}
        
        actions = {
            "connected": True,
            "aim_assist_active": False,
            "fire_pressed": False,
            "snap_to_target": False,
            "cycle_target": False,
            "toggle_overlay": False,
            "emergency_disable": False,
            "left_trigger": self.state.left_trigger,
            "right_trigger": self.state.right_trigger,
            "right_stick": self.state.right_stick,
        }
        
        # Check aim hold (LT) - Aim assist active while aiming
        if self.is_trigger_active("LEFT_TRIGGER"):
            actions["aim_assist_active"] = True
        
        # Check fire (RT)
        if self.is_trigger_active("RIGHT_TRIGGER"):
            actions["fire_pressed"] = True
        
        # Check aim assist toggle (LB) - edge detection
        if self.is_button_just_pressed(self.bindings.get("aim_assist_toggle")):
            self.aim_assist_active = not self.aim_assist_active
            actions["aim_assist_toggled"] = True
            self.feedback_aim_assist_toggle()
        
        # Check snap to target (RS click)
        if self.is_button_just_pressed(self.bindings.get("snap_to_target")):
            actions["snap_to_target"] = True
            self.feedback_target_locked()
        
        # Check cycle target (RB)
        if self.is_button_just_pressed(self.bindings.get("cycle_target")):
            actions["cycle_target"] = True
            self.current_target_index += 1
        
        # Check toggle overlay (BACK)
        if self.is_button_just_pressed(self.bindings.get("toggle_overlay")):
            actions["toggle_overlay"] = True
        
        # Check emergency disable (LB + RB)
        emergency_binding = self.bindings.get("emergency_disable")
        if isinstance(emergency_binding, list):
            if all(self.is_button_pressed(b) for b in emergency_binding):
                actions["emergency_disable"] = True
                self.aim_assist_active = False
        
        # Persistent state
        actions["aim_assist_enabled"] = self.aim_assist_active
        actions["current_target_index"] = self.current_target_index
        
        return actions
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert current state to dictionary for API"""
        return {
            "connected": self.state.connected,
            "aim_assist_active": self.aim_assist_active,
            "target_locked": self.target_locked,
            "current_target_index": self.current_target_index,
            "buttons": self.state.buttons if self.state.buttons else {},
            "left_trigger": self.state.left_trigger,
            "right_trigger": self.state.right_trigger,
            "left_stick": self.state.left_stick,
            "right_stick": self.state.right_stick,
            "bindings": self.bindings
        }


# Global controller instance
controller = ControllerManager()


# Scuf Valor Pro specific configuration
SCUF_VALOR_PRO_CONFIG = {
    "name": "Scuf Valor Pro",
    "description": "Premium Xbox controller with 4 rear paddles",
    "paddle_mapping": {
        "PADDLE_1": "A",      # Usually Jump
        "PADDLE_2": "B",      # Usually Crouch/Slide
        "PADDLE_3": "X",      # Reload
        "PADDLE_4": "Y",      # Weapon Switch
    },
    "trigger_stops": True,  # Scuf has trigger stops for faster fire
    "recommended_bindings": {
        "aim_assist_toggle": "LEFT_SHOULDER",
        "aim_hold": "LEFT_TRIGGER",
        "fire": "RIGHT_TRIGGER",
        "snap_to_target": "RIGHT_THUMB",
        "cycle_target": "RIGHT_SHOULDER",
        "toggle_overlay": "BACK",
        # Paddle-specific bindings
        "paddle_aim_pulse": "A",  # Paddle 1 mapped to A
        "paddle_lock_target": "B",  # Paddle 2 mapped to B
    },
    "notes": [
        "Enable trigger stops for faster ADS and fire response",
        "Map Paddle 1 to A for quick aim assist pulse",
        "Map Paddle 2 to B for instant target lock",
        "Recommended stick tension: Medium-High for precise aim",
    ]
}


def get_scuf_config():
    """Get Scuf Valor Pro specific configuration"""
    return SCUF_VALOR_PRO_CONFIG
