#!/usr/bin/env python3
"""Standalone policy deployment using only standard MuJoCo (no mjlab).
This version includes keyboard control support.

This script runs trained policies using exported MJCF, requiring only:
- mujoco
- torch
- numpy
- glfw (for gamepad support)
- pynput (for keyboard support)

Run export_mjcf.py first to generate the scene XML file.

USAGE:
1. Edit the CONFIG section below with your desired settings
2. Run: python run_sim2sim_keyboard.py
"""

import json
import os
import csv
import time
from datetime import datetime
import math
from pathlib import Path

import glfw
import mujoco
import mujoco.viewer as viewer
import numpy as np
import torch

# Try to import pynput library for cross-platform keyboard input (no root required on Linux)
try:
    from pynput import keyboard as pynput_keyboard
    KEYBOARD_AVAILABLE = True
except ImportError:
    KEYBOARD_AVAILABLE = False
    pynput_keyboard = None


# ============================================================================
# CONFIGURATION - Edit these variables to change behavior
# ============================================================================

CONFIG = {
    # Path to exported MJCF model
    "model_xml": "scene.xml",
    
    # Initial policy file (will be overridden by gamepad if connected)
    "policy_path": "policies/policy_shamble.pt",
    
    # Device to run on (cpu only for standalone)
    "device": "cpu",
    
    # Global action scale multiplier
    "action_scale": 0.5,
    
    # Number of simulation substeps per policy step
    "n_substeps": 4,
    
    # Path to JSON file with initial pose (or None to use default)
    "init_pose_json": "poses/default-pose.json",
    
    # === INPUT SETTINGS ===
    
    # Enable gamepad control (requires glfw)
    # If False or gamepad not found, falls back to keyboard control
    "use_gamepad": True,
    
    # Enable keyboard control (always available as fallback)
    "use_keyboard": True,
    
    # Max forward/backward velocity (m/s) - scaled by left stick Y
    "max_lin_vel": 2.0,
    
    # Max lateral velocity (m/s) - scaled by left stick X
    "max_lat_vel": 1.0,
    
    # Max angular velocity (rad/s) - scaled by right stick X  
    "max_ang_vel": 1.0,
    
    # Gain adjustment step size for keyboard (q/a keys)
    "gain_step": 0.1,
    
    # Buttons: 0=A (bottom), 1=B (right), 2=X (left), 3=Y (top)
    # Option 1 (recommended): Use a single button to cycle through policies in order
    "policy_cycle": [
        "policies/policy_shamble.pt",
        "policies/policy_crawl_start.pt",
        "policies/policy_crawl.pt",
        # "policies/policy_shamble_start.pt",
    ],
    "cycle_button": 0,  # A button
    
    # Explicit mapping of policy filenames to which default joint set to use
    # Valid values: "stand", "crawl"
    "policy_defaults": {
        "policies/policy_shamble.pt": "stand",
        "policies/policy_crawl_start.pt": "stand",
        "policies/policy_crawl.pt": "crawl",
        # "policies/policy_shamble_start.pt": "stand",
    },
    
    # Exit button (9 = Start/Menu button on most controllers)
    "exit_button": 6,
    
    # Debug mode - prints all button/axis events
    "gamepad_debug": True,
    
    # Preferred gamepad selection (optional)
    # Use a name substring (e.g., "DualSense", "Xbox") or 1-based index
    "gamepad_preferred_name": None,
    "gamepad_preferred_index": 2,
    
    # === CAMERA SETTINGS ===
    
    # Camera distance from robot (meters)
    "camera_distance": 3.0,
    
    # Camera azimuth angle (degrees, 0=front, 90=side)
    "camera_azimuth": 90,
    
    # Camera elevation angle (degrees, negative=look down)
    "camera_elevation": -20,
    
    # Track robot with camera (follows robot position)
    "camera_tracking": True,
    
    # === GAIN MULTIPLIERS (match deploy_real.py concept) ===
    # Group-based KP/KD multipliers applied to MuJoCo actuators
    # Legs: hips, knees, ankles
    # Upper: waist + shoulders, elbows, wrists
    "kp_multiplier_legs": 1.4,
    "kd_multiplier_legs": 1.4,
    "kp_multiplier_upper": 1.4,
    "kd_multiplier_upper": 1.4,
    
    # Amount to change multipliers per button press
    "gain_step": 0.05,
    # Diagnostics logging (CSV) for parity with real robot
    # Enable to record per-step dt, absolute joint positions, targets, and safety metrics
    "diag_log": True,
    # Optional explicit file path; if empty, a timestamped file will be created under diag_dir
    "diag_file": "",
    # Directory for diagnostics when diag_file not set
    "diag_dir": "outputs/diagnostics",
    # Filename prefix when auto-generating
    "diag_prefix": "sim2sim",
}

# ============================================================================
# END CONFIGURATION
# ============================================================================


# PyTorch model expects joints in this specific order
PYTORCH_JOINT_ORDER = [
    'left_hip_pitch_joint', 'right_hip_pitch_joint', 'waist_yaw_joint', 
    'left_hip_roll_joint', 'right_hip_roll_joint',
    'left_hip_yaw_joint', 'right_hip_yaw_joint', 
    'left_knee_joint', 'right_knee_joint', 
    'left_shoulder_pitch_joint', 'right_shoulder_pitch_joint',
    'left_ankle_pitch_joint', 'right_ankle_pitch_joint', 
    'left_shoulder_roll_joint', 'right_shoulder_roll_joint',
    'left_ankle_roll_joint', 'right_ankle_roll_joint', 
    'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint',
    'left_elbow_joint', 'right_elbow_joint', 
    'left_wrist_roll_joint', 'right_wrist_roll_joint'
]

# Training defaults (in MuJoCo joint order)
TRAINING_DEFAULT_ANGLES = {
    "left_hip_pitch_joint": -1.6796101123595506,
    "left_hip_roll_joint": 2.4180011235955052,
    "left_hip_yaw_joint": 1.2083865168539325,
    "left_knee_joint": 2.1130298764044944,
    "left_ankle_pitch_joint": 0.194143033707865,
    "left_ankle_roll_joint": 0.0,
    "right_hip_pitch_joint": -1.6796101123595506,
    "right_hip_roll_joint": -2.4180011235955052,
    "right_hip_yaw_joint": -1.2083865168539325,
    "right_knee_joint": 2.1130298764044944,
    "right_ankle_pitch_joint": 0.194143033707865,
    "right_ankle_roll_joint": 0.0,
    "waist_yaw_joint": 0.0,
    "left_shoulder_pitch_joint": 1.4578526315789473,
    "left_shoulder_roll_joint": 1.5778684210526317,
    "left_shoulder_yaw_joint": 1.4238245614035088,
    "left_elbow_joint": -0.3124709677419355,
    "left_wrist_roll_joint": 0.0,
    "right_shoulder_pitch_joint": 1.4578526315789473,
    "right_shoulder_roll_joint": -1.5778684210526317,
    "right_shoulder_yaw_joint": -1.4238245614035088,
    "right_elbow_joint": -0.3124709677419355,
    "right_wrist_roll_joint": 0.0,
}

# Stand pose defaults (match G1_STAND_CFG.init_state.joint_pos in source/g1_crawl/.../g1.py)
# Fully expanded per-joint map in MuJoCo joint order
STAND_DEFAULT_ANGLES = {
    # Hips
    "left_hip_pitch_joint": -0.20,
    "right_hip_pitch_joint": -0.20,
    "left_hip_roll_joint": 0.0,
    "right_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    # Knees
    "left_knee_joint": 0.42,
    "right_knee_joint": 0.42,
    # Ankles
    "left_ankle_pitch_joint": -0.23,
    "right_ankle_pitch_joint": -0.23,
    "left_ankle_roll_joint": 0.0,
    "right_ankle_roll_joint": 0.0,
    # Waist
    "waist_yaw_joint": 0.0,
    # Shoulders
    "left_shoulder_pitch_joint": 0.35,
    "right_shoulder_pitch_joint": 0.35,
    "left_shoulder_roll_joint": 0.16,
    "right_shoulder_roll_joint": -0.16,
    "left_shoulder_yaw_joint": 0.0,
    "right_shoulder_yaw_joint": 0.0,
    # Elbows
    "left_elbow_joint": 0.87,
    "right_elbow_joint": 0.87,
    # Wrists
    "left_wrist_roll_joint": 0.0,
    "right_wrist_roll_joint": 0.0,
}

# Map default set names to angle dicts
DEFAULT_SET_MAP: dict[str, dict[str, float]] = {
    "stand": STAND_DEFAULT_ANGLES,
    "crawl": TRAINING_DEFAULT_ANGLES,
}

def resolve_default_joint_positions(policy_path: Path, policy_defaults_map: dict[str, str]) -> dict[str, float]:
    """Return default joint positions for a policy based on explicit mapping.

    Requires an explicit entry in policy_defaults_map; fails loudly if missing.
    """
    key = str(policy_path)
    if key not in policy_defaults_map:
        available = ", ".join(sorted(policy_defaults_map.keys())) if policy_defaults_map else "(none)"
        raise KeyError(f"No default set mapped for policy '{key}'. Available mappings: {available}")
    set_name = str(policy_defaults_map[key])
    if set_name not in DEFAULT_SET_MAP:
        valid = ", ".join(sorted(DEFAULT_SET_MAP.keys()))
        raise KeyError(f"Invalid default set '{set_name}' for policy '{key}'. Valid: {valid}")
    return DEFAULT_SET_MAP[set_name]

# Joint limits (min, max) in MuJoCo joint order - matching RESTRICTED_JOINT_RANGE from deploy_real.py
JOINT_LIMITS = {
    "left_hip_pitch_joint": (-2.5307, 2.8798),
    "left_hip_roll_joint": (-0.5236, 2.9671),
    "left_hip_yaw_joint": (-2.7576, 2.7576),
    "left_knee_joint": (-0.087267, 2.8798),
    "left_ankle_pitch_joint": (-0.87267, 0.5236),
    "left_ankle_roll_joint": (-0.2618, 0.2618),
    "right_hip_pitch_joint": (-2.5307, 2.8798),
    "right_hip_roll_joint": (-2.9671, 0.5236),
    "right_hip_yaw_joint": (-2.7576, 2.7576),
    "right_knee_joint": (-0.087267, 2.8798),
    "right_ankle_pitch_joint": (-0.87267, 0.5236),
    "right_ankle_roll_joint": (-0.2618, 0.2618),
    "waist_yaw_joint": (-2.618, 2.618),
    "left_shoulder_pitch_joint": (-3.0892, 2.6704),
    "left_shoulder_roll_joint": (-1.5882, 2.2515),
    "left_shoulder_yaw_joint": (-2.618, 2.618),
    "left_elbow_joint": (-1.0472, 2.0944),
    "left_wrist_roll_joint": (-1.97222, 1.97222),
    "right_shoulder_pitch_joint": (-3.0892, 2.6704),
    "right_shoulder_roll_joint": (-2.2515, 1.5882),
    "right_shoulder_yaw_joint": (-2.618, 2.618),
    "right_elbow_joint": (-1.0472, 2.0944),
    "right_wrist_roll_joint": (-1.97222, 1.97222),
}


class KeyboardController:
    """Keyboard input handler using pynput library (no root required on Linux)."""
    
    def __init__(
        self,
        max_lin_vel: float,
        max_lat_vel: float,
        max_ang_vel: float,
        policy_cycle: list[str] | None = None,
        initial_policy_path: str | None = None,
        gain_step: float = 0.1,
    ):
        if not KEYBOARD_AVAILABLE:
            raise RuntimeError("pynput library not available. Install with: pip install pynput")
        
        self.max_lin_vel = max_lin_vel
        self.max_lat_vel = max_lat_vel
        self.max_ang_vel = max_ang_vel
        self.policy_cycle = list(policy_cycle) if policy_cycle else []
        self.gain_step = gain_step
        
        # Cycle state
        self._cycle_index = 0
        if self.policy_cycle and initial_policy_path is not None:
            try:
                self._cycle_index = self.policy_cycle.index(initial_policy_path)
            except ValueError:
                pass
        
        # Control mode state - 初始为 policy 模式（站立状态）
        self._system_started = True  # 系统初始已启用
        self.control_mode = "policy"  # 初始模式: policy (站立控制)
        
        # Track key states
        self._pressed_keys = set()
        
        # Initialize key states - 新的键盘映射
        self._key_states = {
            # 方向键：前后左右移动
            'up': False,      # 方向键上 - 前进
            'down': False,    # 方向键下 - 后退
            'left': False,    # 方向键左 - 左平移
            'right': False,   # 方向键右 - 右平移
            # 旋转控制
            'z': False,       # Z - 左转
            'c': False,       # C - 右转
            # 模式切换 (IJK 对应原项目的 D-PAD, 移除L)
            'i': False,       # I - 默认位置模式 (D-PAD UP)
            'j': False,       # J - 阻尼模式 (D-PAD LEFT)
            'k': False,       # K - 爬行位置模式 (D-PAD DOWN)
            # 策略和系统控制
            'space': False,   # SPACE - 循环切换策略
            'esc': False,     # ESC - 退出程序
            # 增益调整
            'q': False,       # Q - 增加增益
            'a': False,       # A - 减少增益
            'h': False,       # H - 显示增益
        }
        
        # Helper function to normalize key to string
        def normalize_key(key):
            if isinstance(key, pynput_keyboard.KeyCode):
                try:
                    return key.char.lower() if key.char else None
                except AttributeError:
                    return None
            elif isinstance(key, pynput_keyboard.Key):
                key_mapping = {
                    pynput_keyboard.Key.space: 'space',
                    pynput_keyboard.Key.esc: 'esc',
                    pynput_keyboard.Key.up: 'up',
                    pynput_keyboard.Key.down: 'down',
                    pynput_keyboard.Key.left: 'left',
                    pynput_keyboard.Key.right: 'right',
                }
                return key_mapping.get(key, None)
            return None
        
        # Create keyboard listener (non-blocking)
        def on_press(key):
            self._pressed_keys.add(key)
            key_str = normalize_key(key)
            if key_str and key_str in self._key_states:
                self._key_states[key_str] = True
        
        def on_release(key):
            self._pressed_keys.discard(key)
            key_str = normalize_key(key)
            if key_str and key_str in self._key_states:
                self._key_states[key_str] = False
        
        self._listener = pynput_keyboard.Listener(
            on_press=on_press,
            on_release=on_release,
            suppress=False
        )
        self._listener.start()
        
        # Initialize previous states for edge detection
        self._prev_space_state = False
        self._prev_esc_state = False
        self._prev_i_state = False
        self._prev_j_state = False
        self._prev_k_state = False
        self._prev_q_state = False
        self._prev_a_state = False
        self._prev_h_state = False
        
        # Print control mapping
        print()
        print("=" * 60)
        print("[KEYBOARD] Controls")
        print("=" * 60)
        print("  Movement (Arrow Keys):")
        print("    ↑/↓: Forward/Backward")
        print("    ←/→: Strafe Left/Right")
        print("    Z/C: Rotate Left/Right")
        print()
        print("  Mode Switching (IJK, matching D-PAD):")
        print("    I: Default Position mode (stand)")
        print("    J: Damped mode (safe stop)")
        print("    K: Crawl Position mode")
        print("    (Note: Robot starts in policy/standing mode)")
        print()
        if self.policy_cycle:
            print(f"  Policy Control:")
            print(f"    SPACE: Cycle through {len(self.policy_cycle)} policies")
            for idx, policy_path in enumerate(self.policy_cycle):
                policy_name = Path(policy_path).stem
                marker = "<- current" if idx == self._cycle_index else ""
                print(f"      [{idx}] {policy_name} {marker}")
        print()
        print("  Gain Adjustment:")
        print(f"    Q: Increase gain multiplier (+{self.gain_step})")
        print(f"    A: Decrease gain multiplier (-{self.gain_step})")
        print("    H: Print current gain multiplier")
        print()
        print("  Exit:")
        print("    ESC: Exit simulation")
        print()
        print("=" * 60)
        print()
    
    def get_velocity_commands(self) -> tuple[float, float, float]:
        """Get velocity commands from keyboard.
        Only active in Neural Network mode.
        
        Returns:
            (lin_vel_z, lin_vel_y, ang_vel_x): Forward velocity, lateral velocity, and angular velocity
        """
        # Only provide velocity commands in policy mode
        if self.control_mode != "policy":
            return 0.0, 0.0, 0.0
        
        lin_vel_z = 0.0
        lin_vel_y = 0.0
        ang_vel_x = 0.0
        
        # Forward/Backward (↑/↓)
        if self._key_states.get('up', False):
            lin_vel_z = self.max_lin_vel
        elif self._key_states.get('down', False):
            lin_vel_z = -self.max_lin_vel
        
        # Strafe Left/Right (←/→)
        if self._key_states.get('left', False):
            lin_vel_y = self.max_lat_vel
        elif self._key_states.get('right', False):
            lin_vel_y = -self.max_lat_vel
        
        # Rotate Left/Right (Z/C)
        if self._key_states.get('z', False):
            ang_vel_x = self.max_ang_vel
        elif self._key_states.get('c', False):
            ang_vel_x = -self.max_ang_vel
        
        return lin_vel_z, lin_vel_y, ang_vel_x
    
    def check_exit(self) -> bool:
        """Check if ESC key was pressed to exit.
        
        Returns:
            True if exit key was pressed, False otherwise
        """
        current_state = self._key_states.get('esc', False)
        previous_state = self._prev_esc_state
        
        if current_state and not previous_state:
            print("\n[KEYBOARD] ESC pressed - shutting down...")
            self._prev_esc_state = current_state
            return True
        
        self._prev_esc_state = current_state
        return False
    
    def check_mode_switch(self) -> str | None:
        """Check if mode switch keys (IJK) were pressed.
        
        Returns:
            Mode name if switched, None otherwise
        """
        # I - Default Position mode
        current_i = self._key_states.get('i', False)
        if current_i and not self._prev_i_state:
            self.control_mode = "default_pos"
            print(f"[KEYBOARD] I pressed -> Switched to DEFAULT POSITION mode")
            self._prev_i_state = current_i
            return "default_pos"
        self._prev_i_state = current_i
        
        # J - Damped mode
        current_j = self._key_states.get('j', False)
        if current_j and not self._prev_j_state:
            self.control_mode = "damped"
            print(f"[KEYBOARD] J pressed -> Switched to DAMPED mode")
            self._prev_j_state = current_j
            return "damped"
        self._prev_j_state = current_j
        
        # K - Crawl Position mode
        current_k = self._key_states.get('k', False)
        if current_k and not self._prev_k_state:
            self.control_mode = "crawl_pos"
            print(f"[KEYBOARD] K pressed -> Switched to CRAWL POSITION mode")
            self._prev_k_state = current_k
            return "crawl_pos"
        self._prev_k_state = current_k
        
        return None
    
    def check_policy_switch(self) -> str | None:
        """Check if SPACE key was pressed to switch policy.
        Only works in policy mode.
        
        Returns:
            Policy path if key pressed, None otherwise
        """
        if not self.policy_cycle or self.control_mode != "policy":
            return None
        
        current_state = self._key_states.get('space', False)
        previous_state = self._prev_space_state
        
        if current_state and not previous_state:
            # Advance cycle index and return next policy
            self._cycle_index = (self._cycle_index + 1) % len(self.policy_cycle)
            next_policy = self.policy_cycle[self._cycle_index]
            policy_name = Path(next_policy).stem
            print(f"[KEYBOARD] SPACE pressed -> cycling to [{self._cycle_index}] {policy_name}")
            self._prev_space_state = current_state
            return next_policy
        
        self._prev_space_state = current_state
        return None
    
    def check_gain_adjustment(self) -> tuple[float | None, bool]:
        """Check if Q/A/H keys were pressed for gain adjustment.
        
        Returns:
            (gain_delta, should_print): gain_delta is None if no change, should_print is True if H was pressed
        """
        # Q - Increase gain
        current_q = self._key_states.get('q', False)
        if current_q and not self._prev_q_state:
            self._prev_q_state = current_q
            return self.gain_step, False
        self._prev_q_state = current_q
        
        # A - Decrease gain
        current_a = self._key_states.get('a', False)
        if current_a and not self._prev_a_state:
            self._prev_a_state = current_a
            return -self.gain_step, False
        self._prev_a_state = current_a
        
        # H - Print gain
        current_h = self._key_states.get('h', False)
        if current_h and not self._prev_h_state:
            self._prev_h_state = current_h
            return None, True
        self._prev_h_state = current_h
        
        return None, False
    
    def cleanup(self):
        """Stop the keyboard listener."""
        if hasattr(self, '_listener'):
            self._listener.stop()


class GamepadController:
    """Gamepad input handler using GLFW."""
    
    def __init__(
        self,
        button_policy_map: dict[int, str],
        max_lin_vel: float,
        max_lat_vel: float,
        max_ang_vel: float,
        exit_button: int = 9,
        debug: bool = False,
        policy_cycle: list[str] | None = None,
        cycle_button: int | None = None,
        initial_policy_path: str | None = None,
        preferred_name: str | None = None,
        preferred_index_1based: int | None = None,
    ):
        # Mode configuration
        self.button_policy_map = button_policy_map or {}
        self.policy_cycle = list(policy_cycle) if policy_cycle else []
        self.cycle_button = cycle_button
        if self.policy_cycle and self.cycle_button is not None:
            self._mode = "cycle"
        elif self.button_policy_map:
            self._mode = "map"
        else:
            raise ValueError("GamepadController requires either a non-empty policy_cycle+cycle_button or a non-empty button_policy_map")
        
        self.max_lin_vel = max_lin_vel
        self.max_lat_vel = max_lat_vel
        self.max_ang_vel = max_ang_vel
        self.exit_button = exit_button
        self.debug = debug
        # Select joystick
        self.joystick_id = self._select_joystick(preferred_name, preferred_index_1based)
        self.last_button_state = [0] * 16  # Track button states for edge detection
        
        # Cycle state
        self._cycle_index = 0
        if self._mode == "cycle" and initial_policy_path is not None:
            # Align cycle index with initial policy if present
            try:
                self._cycle_index = self.policy_cycle.index(initial_policy_path)
            except ValueError:
                # Not found; start at 0, fail loudly in debug
                if self.debug:
                    print(f"[GAMEPAD DEBUG] Initial policy not in policy_cycle: {initial_policy_path}")
        
        # Initialize GLFW
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")
        
        # Check for gamepad
        if not glfw.joystick_present(self.joystick_id):
            glfw.terminate()
            raise RuntimeError("No gamepad found")
        
        gamepad_name = glfw.get_joystick_name(self.joystick_id)
        print(f"[GAMEPAD] Connected: {gamepad_name}")
        
        if glfw.joystick_is_gamepad(self.joystick_id):
            print(f"[GAMEPAD] Using standard mapping: {glfw.get_gamepad_name(self.joystick_id)}")
        else:
            print("[GAMEPAD] Warning: Not a standard gamepad mapping")
        
        # Print control mapping
        print()
        if self._mode == "cycle":
            print("[GAMEPAD] Face Button -> Cycle Policies:")
            button_names = {0: "A (bottom)", 1: "B (right)", 2: "X (left)", 3: "Y (top)"}
            btn_name = button_names.get(self.cycle_button, f"Button {self.cycle_button}")
            print(f"  {btn_name:12} -> cycles through {len(self.policy_cycle)} policies")
            for idx, policy_path in enumerate(self.policy_cycle):
                policy_name = Path(policy_path).stem
                marker = "<- start" if idx == self._cycle_index else ""
                print(f"    [{idx}] {policy_name} {marker}")
        else:
            print("[GAMEPAD] Face Button -> Policy Mapping:")
            button_names = {0: "X (bottom)", 1: "Circle (right)", 2: "Square (left)", 3: "Triangle (top)"}
            for btn_id, policy_path in sorted(self.button_policy_map.items()):
                btn_name = button_names.get(btn_id, f"Button {btn_id}")
                policy_name = Path(policy_path).stem
                print(f"  {btn_name:12} -> {policy_name}")
        
        exit_button_names = {8: "Select/Back", 9: "Start/Menu"}
        exit_btn_name = exit_button_names.get(self.exit_button, f"Button {self.exit_button}")
        print(f"\n[GAMEPAD] Exit Button: {exit_btn_name}")
        
        if self.debug:
            print("[GAMEPAD] Debug mode ENABLED - will print all button presses")
        print()

    @staticmethod
    def _to_str_name(name_obj) -> str:
        try:
            if isinstance(name_obj, (bytes, bytearray)):
                return name_obj.decode(errors="ignore")
            return str(name_obj)
        except Exception:
            return str(name_obj)

    @classmethod
    def _enumerate_devices(cls) -> list[tuple[int, str, bool]]:
        first = getattr(glfw, "JOYSTICK_1", 0)
        last = getattr(glfw, "JOYSTICK_LAST", 15)
        devices: list[tuple[int, str, bool]] = []
        for jid in range(first, last + 1):
            if glfw.joystick_present(jid):
                name = cls._to_str_name(glfw.get_joystick_name(jid))
                is_gamepad = bool(glfw.joystick_is_gamepad(jid))
                devices.append((jid, name, is_gamepad))
        return devices

    @classmethod
    def _select_joystick(cls, preferred_name: str | None, preferred_index_1based: int | None) -> int:
        devices = cls._enumerate_devices()
        if not devices:
            raise RuntimeError("No joysticks detected via GLFW")

        # Prefer exact/substring name match with standard mapping
        if preferred_name:
            substr = preferred_name.lower()
            for jid, name, is_gp in devices:
                if not is_gp:
                    continue
                mapping = cls._to_str_name(glfw.get_gamepad_name(jid))
                if substr in name.lower() or substr in mapping.lower():
                    return jid

        # Prefer explicit index (1-based) with standard mapping
        if preferred_index_1based is not None:
            first = getattr(glfw, "JOYSTICK_1", 0)
            jid = first + (preferred_index_1based - 1)
            if not glfw.joystick_present(jid):
                raise RuntimeError(f"Requested joystick index {preferred_index_1based} not present")
            if not glfw.joystick_is_gamepad(jid):
                raise RuntimeError(f"Requested joystick index {preferred_index_1based} does not have a standard gamepad mapping")
            return jid

        # Fallback: first standard-mapped gamepad
        for jid, name, is_gp in devices:
            if is_gp:
                return jid
        raise RuntimeError("Found joysticks but none with a standard gamepad mapping")
    
    def get_velocity_commands(self) -> tuple[float, float, float]:
        """Get velocity commands from analog sticks.
        
        Returns:
            (lin_vel_z, lin_vel_y, ang_vel_x): Forward velocity, lateral velocity, and angular velocity
        """
        state = glfw.get_gamepad_state(self.joystick_id)
        if not state:
            return 0.0, 0.0, 0.0
        
        # Left stick Y (inverted) for forward/backward
        # Stick axes are -1 (up) to 1 (down), so negate for intuitive control
        left_y = -state.axes[1] if len(state.axes) > 1 else 0.0
        lin_vel_z = left_y * self.max_lin_vel
        
        # Left stick X for lateral movement (strafe left/right) - negated for intuitive control
        left_x = -state.axes[0] if len(state.axes) > 0 else 0.0
        lin_vel_y = left_x * self.max_lat_vel
        
        # Right stick X for rotation - negated for intuitive control
        right_x = -state.axes[2] if len(state.axes) > 2 else 0.0
        ang_vel_x = right_x * self.max_ang_vel
        
        # Apply deadzone
        deadzone = 0.1
        if abs(lin_vel_z) < deadzone:
            lin_vel_z = 0.0
        if abs(lin_vel_y) < deadzone:
            lin_vel_y = 0.0
        if abs(ang_vel_x) < deadzone:
            ang_vel_x = 0.0
        
        return lin_vel_z, lin_vel_y, ang_vel_x
    
    def check_policy_switch(self) -> str | None:
        """Check if a face button was pressed to switch policy.
        
        Returns:
            Policy path if button pressed, None otherwise
        """
        state = glfw.get_gamepad_state(self.joystick_id)
        if not state:
            return None
        
        if self._mode == "cycle":
            btn_id = self.cycle_button if self.cycle_button is not None else -1
            if 0 <= btn_id < len(state.buttons):
                current = state.buttons[btn_id]
                previous = self.last_button_state[btn_id]
                if current == 1 and previous == 0:
                    # Advance cycle index and return next policy
                    self._cycle_index = (self._cycle_index + 1) % len(self.policy_cycle)
                    next_policy = self.policy_cycle[self._cycle_index]
                    self.last_button_state[btn_id] = current
                    if self.debug:
                        btn_names = {0: "A", 1: "B", 2: "X", 3: "Y"}
                        btn_name = btn_names.get(btn_id, f"Button {btn_id}")
                        print(f"[GAMEPAD DEBUG] Button {btn_id} ({btn_name}) pressed -> cycling to [{self._cycle_index}] {Path(next_policy).stem}")
                    return next_policy
                self.last_button_state[btn_id] = current
        else:
            # Check each mapped button for rising edge (0->1)
            for btn_id, policy_path in self.button_policy_map.items():
                if btn_id < len(state.buttons):
                    current = state.buttons[btn_id]
                    previous = self.last_button_state[btn_id]
                    
                    if current == 1 and previous == 0:  # Button just pressed
                        self.last_button_state[btn_id] = current
                        if self.debug:
                            btn_names = {0: "A", 1: "B", 2: "X", 3: "Y"}
                            btn_name = btn_names.get(btn_id, f"Button {btn_id}")
                            print(f"[GAMEPAD DEBUG] Button {btn_id} ({btn_name}) pressed -> switching policy")
                        return policy_path
                    
                    self.last_button_state[btn_id] = current
        
        return None
    
    def check_exit(self) -> bool:
        """Check if exit button was pressed.
        
        Returns:
            True if exit button was pressed, False otherwise
        """
        state = glfw.get_gamepad_state(self.joystick_id)
        if not state:
            return False
        
        # Check exit button for rising edge (0->1)
        if self.exit_button < len(state.buttons):
            current = state.buttons[self.exit_button]
            previous = self.last_button_state[self.exit_button]
            
            if current == 1 and previous == 0:  # Button just pressed
                self.last_button_state[self.exit_button] = current
                if self.debug:
                    print(f"[GAMEPAD DEBUG] Button {self.exit_button} (Exit) pressed")
                print("\n[GAMEPAD] Exit button pressed - shutting down...")
                return True
            
            self.last_button_state[self.exit_button] = current
        
        return False
    
    def print_debug_info(self):
        """Print debug information about button and axis states (if debug enabled).
        
        This should be called BEFORE check_policy_switch and check_exit to catch all buttons.
        """
        if not self.debug:
            return
        
        state = glfw.get_gamepad_state(self.joystick_id)
        if not state:
            return
        
        # Check for button presses (rising edges)
        for btn_id in range(min(len(state.buttons), len(self.last_button_state))):
            current = state.buttons[btn_id]
            previous = self.last_button_state[btn_id]
            
            if current == 1 and previous == 0:  # Button just pressed
                # Skip buttons that are handled by other methods (they print their own debug)
                if btn_id in self.button_policy_map or (self.cycle_button is not None and btn_id == self.cycle_button) or btn_id == self.exit_button:
                    continue
                    
                # Try to identify the button
                btn_names = {
                    0: "A (bottom/cross)", 
                    1: "B (right/circle)", 
                    2: "X (left/square)", 
                    3: "Y (top/triangle)",
                    4: "LB (left bumper)",
                    5: "RB (right bumper)",
                    6: "LT (left trigger)",
                    7: "RT (right trigger)",
                    8: "Select/Back",
                    9: "Start/Menu",
                    10: "L3 (left stick click)",
                    11: "R3 (right stick click)",
                    12: "D-pad Up",
                    13: "D-pad Down",
                    14: "D-pad Left",
                    15: "D-pad Right",
                }
                btn_name = btn_names.get(btn_id, f"Unknown")
                print(f"[GAMEPAD DEBUG] Button {btn_id} pressed ({btn_name}) [unmapped]")
            
            # Update state for this method's tracking (don't interfere with other methods)
            self.last_button_state[btn_id] = current
    
    def cleanup(self, terminate_glfw: bool = True):
        """Clean up GLFW resources.
        
        Args:
            terminate_glfw: If True, call glfw.terminate(). Set to False if GLFW
                          is still in use by other components (e.g., MuJoCo viewer).
        """
        if terminate_glfw:
            try:
                glfw.terminate()
            except Exception:
                # Suppress OpenGL context errors during cleanup (harmless)
                pass


def rpy_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert roll-pitch-yaw to quaternion (w, x, y, z)."""
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    quat = np.array([w, x, y, z], dtype=np.float32)
    return quat / np.linalg.norm(quat)


def compute_projected_gravity(quat: np.ndarray) -> np.ndarray:
    """Project gravity into body frame from quaternion."""
    qw, qx, qy, qz = quat
    
    # Exact formula from training
    gravity_x = 2 * (-qz * qx + qw * qy)
    gravity_y = -2 * (qz * qy + qw * qx)
    gravity_z = 1 - 2 * (qw * qw + qz * qz)
    
    return np.array([gravity_x, gravity_y, gravity_z], dtype=np.float32)


class StandalonePolicyController:
    """Standalone policy controller using only numpy/torch (no mjlab)."""
    
    def __init__(
        self,
        policy_path: Path,
        mj_model: mujoco.MjModel,
        device: str = "cpu",
        action_scale: float = 0.5,
        training_defaults: dict[str, float] | None = None,
        kp_multiplier_legs: float = 1.0,
        kd_multiplier_legs: float = 1.0,
        kp_multiplier_upper: float = 1.0,
        kd_multiplier_upper: float = 1.0,
        policy_defaults_map: dict[str, str] | None = None,
    ):
        self.mj_model = mj_model
        self.device = device
        self.action_scale = action_scale
        self.policy_defaults_map = dict(policy_defaults_map or {})
        
        # Gain multipliers (group-based)
        self.kp_multiplier_legs = float(kp_multiplier_legs)
        self.kd_multiplier_legs = float(kd_multiplier_legs)
        self.kp_multiplier_upper = float(kp_multiplier_upper)
        self.kd_multiplier_upper = float(kd_multiplier_upper)
        
        # Get joint names from MuJoCo model (skip free joint)
        self.joint_names = []
        for i in range(mj_model.njnt):
            name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, i)
            # Skip free joint (floating base)
            if name and name not in ["floating_base_joint", "root", "rootJoint"]:
                self.joint_names.append(name)
        
        self.num_robot_joints = len(self.joint_names)
        self.num_policy_joints = 23
        
        print(f"[INFO] Found {self.num_robot_joints} actuated joints")
        
        # Build remapping indices
        self.mujoco_to_pytorch_idx = []
        for pytorch_joint in PYTORCH_JOINT_ORDER:
            if pytorch_joint in self.joint_names:
                mujoco_idx = self.joint_names.index(pytorch_joint)
                self.mujoco_to_pytorch_idx.append(mujoco_idx)
            else:
                self.mujoco_to_pytorch_idx.append(-1)
        
        # Get default joint positions
        if training_defaults is None:
            training_defaults = TRAINING_DEFAULT_ANGLES
        self._update_default_positions_from_map(training_defaults)
        
        # Get joint limits
        joint_limits_lower = []
        joint_limits_upper = []
        for joint_name in self.joint_names:
            if joint_name in JOINT_LIMITS:
                limits = JOINT_LIMITS[joint_name]
                joint_limits_lower.append(limits[0])
                joint_limits_upper.append(limits[1])
            else:
                print(f"[WARN] Joint {joint_name} not in JOINT_LIMITS, using [-π, π]")
                joint_limits_lower.append(-np.pi)
                joint_limits_upper.append(np.pi)
        
        self.joint_limits_lower = torch.tensor(joint_limits_lower, device=device, dtype=torch.float32)
        self.joint_limits_upper = torch.tensor(joint_limits_upper, device=device, dtype=torch.float32)
        
        # Initialize last actions
        self.last_actions_pytorch = torch.zeros(self.num_policy_joints, device=device, dtype=torch.float32)
        
        # Safety monitoring (matching deploy_real.py thresholds)
        self._prev_joint_pos = None
        self._prev_joint_vel = None
        self._safety_initialized = False
        self._control_dt = 0.02  # 50 Hz control rate
        
        # Safety thresholds (from deploy_real.py)
        self._max_position_jump = 0.3  # rad per timestep (15 rad/s at 20ms)
        self._max_velocity = 25.0  # rad/s
        self._max_acceleration = 1500.0  # rad/s^2
        self._max_action_magnitude = 5.0  # Action output limit
        # Time-aware absolute position rate limit (rad/s) for sampling jitter parity
        self._max_pos_rate = 15.0
        self._last_sample_time = time.time()
        self._last_dt = 0.0
        self._last_max_position_delta = 0.0
        self._last_allowed_jump = 0.0
        self._last_position_jump_joint = -1
        
        # Rolling 1s window for max position jump
        self._rolling_time = 0.0
        self._rolling_max_jump = 0.0
        self._rolling_max_jump_joint = -1
        
        # Rolling 1s window for clamping (max clamp magnitude and joint)
        self._rolling_clamp_max_diff = 0.0
        self._rolling_clamp_joint = -1
        
        # Load policy
        print(f"[INFO] Loading policy from: {policy_path}")
        self._policy = torch.load(policy_path, map_location=device, weights_only=False)
        self._policy.eval()
        self._current_policy_path = policy_path
        print(f"[INFO] Policy loaded successfully")
        
        # Initialize PD gain scaling from model actuators
        self._init_actuator_gain_scaling()
        self._apply_gain_multipliers()

        # Diagnostics logging
        self._diag_enabled = False
        self._diag_writer = None
        self._diag_file = None
        self._diag_header_written = False
        self._diag_started_printed = False

    def enable_diagnostics(self, file_path: str) -> None:
        """Enable CSV diagnostics logging to the specified file."""
        self._diag_enabled = True
        self._diag_file = file_path
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        # Open file and write header lazily on first write
        self._diag_writer = open(file_path, 'w', newline='')
        writer = csv.writer(self._diag_writer)
        # Header: time, dt, policy, metrics, per-joint q/dq/target
        header = [
            't', 'dt', 'policy',
            'max_delta', 'allowed_jump', 'max_delta_joint',
        ]
        # Per-joint columns
        for name in self.joint_names:
            header.append(f"q_{name}")
        for name in self.joint_names:
            header.append(f"dq_{name}")
        for name in self.joint_names:
            header.append(f"target_{name}")
        writer.writerow(header)
        self._diag_writer.flush()
        self._diag_header_written = True

    def _log_step(self, joint_pos: np.ndarray, joint_vel: np.ndarray, targets: np.ndarray) -> None:
        if not self._diag_enabled or self._diag_writer is None:
            return
        writer = csv.writer(self._diag_writer)
        t = time.time()
        row = [
            f"{t:.6f}", f"{self._last_dt:.6f}", str(self._current_policy_path),
            f"{self._last_max_position_delta:.6f}", f"{self._last_allowed_jump:.6f}", str(self._last_position_jump_joint),
        ]
        # Per-joint values
        row.extend([f"{float(v):.6f}" for v in joint_pos])
        row.extend([f"{float(v):.6f}" for v in joint_vel])
        row.extend([f"{float(v):.6f}" for v in targets])
        writer.writerow(row)
        self._diag_writer.flush()
        if not self._diag_started_printed:
            print(f"[DIAG] Logging started (first row written) -> {self._diag_file}")
            self._diag_started_printed = True

    def _update_default_positions_from_map(self, defaults_map: dict[str, float]) -> None:
        """Rebuild default joint position tensors from a name->value map."""
        default_pos_mujoco: list[float] = []
        for joint_name in self.joint_names:
            if joint_name in defaults_map:
                default_pos_mujoco.append(float(defaults_map[joint_name]))
            else:
                print(f"[WARN] Joint {joint_name} not in defaults map, using 0.0")
                default_pos_mujoco.append(0.0)
        self.default_joint_pos_mujoco = torch.tensor(default_pos_mujoco, device=self.device, dtype=torch.float32)
        self.default_joint_pos_pytorch = self._remap_mujoco_to_pytorch(self.default_joint_pos_mujoco)

    def _init_actuator_gain_scaling(self) -> None:
        """Capture base KP/KD from MuJoCo actuators and group into legs/upper."""
        nu = self.mj_model.nu
        self._actuator_names: list[str] = []
        self._actuator_group: list[str] = []  # "legs" or "upper"
        self._actuator_base_kp = np.zeros(nu, dtype=np.float32)
        self._actuator_base_kd = np.zeros(nu, dtype=np.float32)

        # Define grouping by actuator (joint) name
        def is_leg(name: str) -> bool:
            return (
                ("hip_" in name) or ("knee" in name) or ("ankle_" in name)
            )

        def is_upper(name: str) -> bool:
            return (
                (name == "waist_yaw_joint") or ("shoulder_" in name) or ("elbow" in name) or ("wrist_" in name)
            )

        for i in range(nu):
            act_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            self._actuator_names.append(act_name or f"actuator_{i}")

            # Base KP from gainprm[0]; Base KD from -biasprm[2]
            base_kp = float(self.mj_model.actuator_gainprm[i, 0])
            base_kd = float(-self.mj_model.actuator_biasprm[i, 2])
            self._actuator_base_kp[i] = base_kp
            self._actuator_base_kd[i] = base_kd

            if is_leg(self._actuator_names[-1]):
                self._actuator_group.append("legs")
            elif is_upper(self._actuator_names[-1]):
                self._actuator_group.append("upper")
            else:
                # Default any unknowns to upper to be conservative
                self._actuator_group.append("upper")

        print("[GAINS] Captured base KP/KD from model actuators")

    def _apply_gain_multipliers(self) -> None:
        """Apply current group multipliers to MuJoCo actuator PD params."""
        for i, group in enumerate(self._actuator_group):
            base_kp = self._actuator_base_kp[i]
            base_kd = self._actuator_base_kd[i]
            if group == "legs":
                kp = base_kp * self.kp_multiplier_legs
                kd = base_kd * self.kd_multiplier_legs
            else:
                kp = base_kp * self.kp_multiplier_upper
                kd = base_kd * self.kd_multiplier_upper

            # Update actuator parameters in-place
            self.mj_model.actuator_gainprm[i, 0] = kp
            self.mj_model.actuator_biasprm[i, 1] = -kp
            self.mj_model.actuator_biasprm[i, 2] = -kd

        print(
            f"[GAINS] Applied multipliers | legs: kp={self.kp_multiplier_legs:.2f} kd={self.kd_multiplier_legs:.2f}  "
            f"upper: kp={self.kp_multiplier_upper:.2f} kd={self.kd_multiplier_upper:.2f}"
        )

    def set_gain_multipliers(
        self,
        kp_multiplier_legs: float | None = None,
        kd_multiplier_legs: float | None = None,
        kp_multiplier_upper: float | None = None,
        kd_multiplier_upper: float | None = None,
    ) -> None:
        """Update gain multipliers and re-apply to actuators."""
        if kp_multiplier_legs is not None:
            self.kp_multiplier_legs = float(kp_multiplier_legs)
        if kd_multiplier_legs is not None:
            self.kd_multiplier_legs = float(kd_multiplier_legs)
        if kp_multiplier_upper is not None:
            self.kp_multiplier_upper = float(kp_multiplier_upper)
        if kd_multiplier_upper is not None:
            self.kd_multiplier_upper = float(kd_multiplier_upper)
        self._apply_gain_multipliers()
    
    def load_policy(self, policy_path: Path) -> None:
        """Hot-swap to a different policy."""
        if policy_path == self._current_policy_path:
            return  # Already loaded
        
        print(f"[POLICY] Switching to: {policy_path.stem}")
        self._policy = torch.load(policy_path, map_location=self.device, weights_only=False)
        self._policy.eval()
        self._current_policy_path = policy_path
        
        # Update default joint positions according to new policy (explicit mapping)
        new_defaults = resolve_default_joint_positions(policy_path, self.policy_defaults_map)
        self._update_default_positions_from_map(new_defaults)
        
        # Reset last actions to avoid discontinuities
        self.last_actions_pytorch = torch.zeros(self.num_policy_joints, device=self.device, dtype=torch.float32)
        
        # Reset safety monitoring to avoid false positives from policy switch
        self._safety_initialized = False
        self._prev_joint_pos = None
        self._prev_joint_vel = None
    
    def _remap_mujoco_to_pytorch(self, mujoco_data: torch.Tensor) -> torch.Tensor:
        """Remap joint data from MuJoCo order to PyTorch order."""
        pytorch_data = torch.zeros(self.num_policy_joints, device=self.device, dtype=torch.float32)
        for pytorch_idx, mujoco_idx in enumerate(self.mujoco_to_pytorch_idx):
            if mujoco_idx >= 0 and mujoco_idx < len(mujoco_data):
                pytorch_data[pytorch_idx] = mujoco_data[mujoco_idx]
        return pytorch_data
    
    def _remap_pytorch_to_mujoco(self, pytorch_data: torch.Tensor) -> torch.Tensor:
        """Remap joint data from PyTorch order to MuJoCo order."""
        mujoco_data = torch.zeros(self.num_robot_joints, device=self.device, dtype=torch.float32)
        for pytorch_idx in range(min(len(pytorch_data), len(self.mujoco_to_pytorch_idx))):
            mujoco_idx = self.mujoco_to_pytorch_idx[pytorch_idx]
            if mujoco_idx >= 0 and mujoco_idx < self.num_robot_joints:
                mujoco_data[mujoco_idx] = pytorch_data[pytorch_idx]
        return mujoco_data
    
    def get_observation(
        self,
        projected_gravity: np.ndarray,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        lin_vel_z: float,
        lin_vel_y: float,
        ang_vel_x: float,
    ) -> torch.Tensor:
        """Build observation tensor matching training format."""
        # Convert to torch
        proj_grav_t = torch.from_numpy(projected_gravity).to(self.device)
        joint_pos_t = torch.from_numpy(joint_pos).to(self.device)
        joint_vel_t = torch.from_numpy(joint_vel).to(self.device)
        
        velocity_commands = torch.tensor(
            [lin_vel_z, lin_vel_y, ang_vel_x],
            device=self.device,
            dtype=torch.float32
        )
        
        # Remap to PyTorch order
        joint_pos_pytorch = self._remap_mujoco_to_pytorch(joint_pos_t)
        joint_vel_pytorch = self._remap_mujoco_to_pytorch(joint_vel_t)
        
        # Joint positions relative to default
        joint_pos_rel_pytorch = joint_pos_pytorch - self.default_joint_pos_pytorch
        
        # Stack observation
        obs = torch.cat([
            proj_grav_t,                    # (3,)
            velocity_commands,              # (3,)
            joint_pos_rel_pytorch,          # (23,)
            joint_vel_pytorch,              # (23,)
            self.last_actions_pytorch,      # (23,)
        ], dim=-1)
        
        return obs  # (75,)
    
    def check_safety_limits(self, joint_pos: np.ndarray, joint_vel: np.ndarray, pytorch_actions: torch.Tensor) -> bool:
        """
        Check if current state violates safety limits (matching deploy_real.py thresholds).
        Prints LOUD warnings but doesn't stop simulation (just warns).
        
        Returns True if safe, False if safety violation detected.
        """
        # Time-aware dt (wall clock) for parity with deploy_real.py
        now = time.time()
        dt = now - self._last_sample_time
        if dt <= 0:
            dt = 1e-4
        self._last_sample_time = now
        self._last_dt = dt

        if not self._safety_initialized:
            # First iteration - just store current state
            self._prev_joint_pos = joint_pos.copy()
            self._prev_joint_vel = joint_vel.copy()
            self._safety_initialized = True
            return True
        
        # Check 1: Absolute position jumps scaled by actual dt
        position_delta = np.abs(joint_pos - self._prev_joint_pos)
        max_position_delta = float(np.max(position_delta))
        position_jump_joint = int(np.argmax(position_delta))
        allowed_jump = self._max_pos_rate * dt
        self._last_max_position_delta = max_position_delta
        self._last_allowed_jump = allowed_jump
        self._last_position_jump_joint = position_jump_joint
        
        if max_position_delta > allowed_jump:
            joint_name = self.joint_names[position_jump_joint]
            print("\n" + "=" * 60)
            print("⚠️  SAFETY VIOLATION: POSITION JUMP DETECTED (ABSOLUTE)! ⚠️")
            print("=" * 60)
            print(f"{joint_name} (joint {position_jump_joint}) jumped {max_position_delta:.3f} rad over {dt*1000:.1f} ms")
            print(f"(limit: {allowed_jump:.3f} = {self._max_pos_rate:.1f} rad/s * dt)")
            print(f"Current pos: {joint_pos[position_jump_joint]:.3f}, Previous: {self._prev_joint_pos[position_jump_joint]:.3f}")
            print("⚠️  THIS WOULD TRIGGER DAMPED MODE ON REAL ROBOT! ⚠️")
            print("=" * 60 + "\n")
        
        # Track rolling 1-second max position jump
        if max_position_delta > self._rolling_max_jump:
            self._rolling_max_jump = float(max_position_delta)
            self._rolling_max_jump_joint = int(position_jump_joint)
        
        # Check 2: Velocity spikes
        velocity_magnitude = np.abs(joint_vel)
        max_velocity = np.max(velocity_magnitude)
        velocity_spike_joint = np.argmax(velocity_magnitude)
        
        if max_velocity > self._max_velocity:
            joint_name = self.joint_names[velocity_spike_joint]
            print("\n" + "=" * 60)
            print("⚠️  SAFETY VIOLATION: VELOCITY SPIKE DETECTED! ⚠️")
            print("=" * 60)
            print(f"{joint_name} (joint {velocity_spike_joint}) velocity: {joint_vel[velocity_spike_joint]:.3f} rad/s")
            print(f"(threshold: {self._max_velocity:.3f} rad/s)")
            print("⚠️  THIS WOULD TRIGGER DAMPED MODE ON REAL ROBOT! ⚠️")
            print("=" * 60 + "\n")
        
        # Check 3: Acceleration spikes
        acceleration = (joint_vel - self._prev_joint_vel) / self._control_dt
        max_acceleration = np.max(np.abs(acceleration))
        acceleration_spike_joint = np.argmax(np.abs(acceleration))
        
        if max_acceleration > self._max_acceleration:
            joint_name = self.joint_names[acceleration_spike_joint]
            print("\n" + "=" * 60)
            print("⚠️  SAFETY VIOLATION: ACCELERATION SPIKE DETECTED! ⚠️")
            print("=" * 60)
            print(f"{joint_name} (joint {acceleration_spike_joint}) acceleration: {acceleration[acceleration_spike_joint]:.1f} rad/s^2")
            print(f"(threshold: {self._max_acceleration:.1f} rad/s^2)")
            print(f"Velocity changed from {self._prev_joint_vel[acceleration_spike_joint]:.3f} to {joint_vel[acceleration_spike_joint]:.3f} rad/s")
            print("⚠️  THIS WOULD TRIGGER DAMPED MODE ON REAL ROBOT! ⚠️")
            print("=" * 60 + "\n")
        
        # Update previous state for next iteration
        self._prev_joint_pos = joint_pos.copy()
        self._prev_joint_vel = joint_vel.copy()
        
        # Rolling window logging (every ~1s)
        self._rolling_time += self._control_dt
        if self._rolling_time >= 1.0:
            if self._rolling_max_jump_joint >= 0:
                joint_name = self.joint_names[self._rolling_max_jump_joint]
                print(f"[SAFETY] 1s max position jump: {self._rolling_max_jump:.3f} rad @ {joint_name}")
            else:
                print("[SAFETY] 1s max position jump: 0.000 rad")
            if self._rolling_clamp_joint >= 0:
                joint_name = self.joint_names[self._rolling_clamp_joint]
                print(f"[SAFETY] 1s max target clamp: {self._rolling_clamp_max_diff:.3f} rad @ {joint_name}")
            else:
                print("[SAFETY] 1s max target clamp: 0.000 rad")
            # Reset rolling window
            self._rolling_time = 0.0
            self._rolling_max_jump = 0.0
            self._rolling_max_jump_joint = -1
            self._rolling_clamp_max_diff = 0.0
            self._rolling_clamp_joint = -1
        
        # Return True even if violations found (just warn, don't stop sim)
        return True
    
    def get_action(self, obs: torch.Tensor, joint_pos: np.ndarray, joint_vel: np.ndarray) -> np.ndarray:
        """Run policy to get actions and check safety limits."""
        with torch.no_grad():
            actions_pytorch = self._policy(obs)
        
        # Safety check before applying actions
        self.check_safety_limits(joint_pos, joint_vel, actions_pytorch)
        
        self.last_actions_pytorch = actions_pytorch.clone()
        
        # Convert to MuJoCo order
        actions_mujoco = self._remap_pytorch_to_mujoco(actions_pytorch)
        
        # Apply to joint targets (unclamped)
        targets_unclamped = actions_mujoco * self.action_scale + self.default_joint_pos_mujoco
        
        # Clamp to joint limits
        targets = torch.clamp(targets_unclamped, self.joint_limits_lower, self.joint_limits_upper)
        
        # Track clamping for rolling 1s summary (suppress per-step prints)
        clamped_mask = (targets != targets_unclamped)
        if clamped_mask.any():
            diffs = torch.abs(targets - targets_unclamped)
            step_max_diff_val = diffs.max().item()
            step_max_diff_idx = int(torch.argmax(diffs).item())
            if step_max_diff_val > self._rolling_clamp_max_diff:
                self._rolling_clamp_max_diff = float(step_max_diff_val)
                self._rolling_clamp_joint = step_max_diff_idx
        
        targets_np = targets.cpu().numpy()
        # Diagnostics logging
        self._log_step(joint_pos=joint_pos, joint_vel=joint_vel, targets=targets_np)
        return targets_np


def load_initial_pose(json_path: Path, mj_model: mujoco.MjModel, data: mujoco.MjData):
    """Load initial pose from JSON file."""
    with open(json_path, 'r') as f:
        pose_data = json.load(f)
    
    pose = pose_data['poses'][0]
    base_pos = pose['base_pos']
    base_rpy = pose['base_rpy']
    joints = pose['joints']
    
    # Set base position
    data.qpos[0] = base_pos[0]
    data.qpos[1] = base_pos[1]
    data.qpos[2] = base_pos[2]
    
    # Set base orientation (quaternion)
    quat = rpy_to_quat(base_rpy[0], base_rpy[1], base_rpy[2])
    data.qpos[3:7] = quat
    
    # Set joint positions
    for i in range(mj_model.njnt):
        joint_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if joint_name and joint_name in joints:
            # Find joint position index (skip free joint positions)
            if i > 0:  # Free joint is index 0
                qpos_idx = 7 + (i - 1)  # 7 for free joint (3 pos + 4 quat)
                if qpos_idx < len(data.qpos):
                    data.qpos[qpos_idx] = joints[joint_name]
    
    mujoco.mj_forward(mj_model, data)


def main():
    """Main entry point for standalone deployment."""
    
    # Parse config
    model_xml = Path(CONFIG["model_xml"])
    policy_path = Path(CONFIG["policy_path"])
    device = CONFIG["device"]
    action_scale = CONFIG["action_scale"]
    n_substeps = CONFIG["n_substeps"]
    init_pose_json = Path(CONFIG["init_pose_json"]) if CONFIG["init_pose_json"] else None
    
    use_gamepad = CONFIG["use_gamepad"]
    max_lin_vel = CONFIG.get("max_lin_vel", 2.0)
    max_lat_vel = CONFIG.get("max_lat_vel", 1.5)
    max_ang_vel = CONFIG.get("max_ang_vel", 1.5)
    button_policy_map = CONFIG.get("button_policy_map", {})
    policy_cycle = CONFIG.get("policy_cycle", None)
    cycle_button = CONFIG.get("cycle_button", None)
    exit_button = CONFIG.get("exit_button", 9)
    gamepad_debug = CONFIG.get("gamepad_debug", False)
    gamepad_preferred_name = CONFIG.get("gamepad_preferred_name", None)
    gamepad_preferred_index = CONFIG.get("gamepad_preferred_index", None)
    
    # Gain multipliers
    kp_multiplier_legs = CONFIG.get("kp_multiplier_legs", 1.0)
    kd_multiplier_legs = CONFIG.get("kd_multiplier_legs", 1.0)
    kp_multiplier_upper = CONFIG.get("kp_multiplier_upper", 1.0)
    kd_multiplier_upper = CONFIG.get("kd_multiplier_upper", 1.0)
    
    # Camera settings
    camera_distance = CONFIG.get("camera_distance", 3.0)
    camera_azimuth = CONFIG.get("camera_azimuth", 90)
    camera_elevation = CONFIG.get("camera_elevation", -20)
    camera_tracking = CONFIG.get("camera_tracking", True)
    
    print("[INFO] Starting standalone policy deployment")
    print(f"[INFO] Model XML: {model_xml}")
    print(f"[INFO] Policy: {policy_path}")

    # Enumerate available gamepads for visibility (before initialization)
    try:
        glfw.init()
        devices = GamepadController._enumerate_devices()
        print("\n[INFO] Detected gamepads:")
        if devices:
            first = getattr(glfw, "JOYSTICK_1", 0)
            for jid, name, is_gp in devices:
                idx = (jid - first) + 1
                mapping = GamepadController._to_str_name(glfw.get_gamepad_name(jid)) if is_gp else "(no standard mapping)"
                pref_marker = ""
                if gamepad_preferred_index is not None and idx == gamepad_preferred_index:
                    pref_marker = " <- preferred index"
                if gamepad_preferred_name:
                    substr = gamepad_preferred_name.lower()
                    if substr in name.lower() or (is_gp and substr in mapping.lower()):
                        pref_marker = " <- preferred name"
                print(f"  [{idx:2d}] {name} | mapping: {mapping}{pref_marker}")
        else:
            print("  (none)")
        print()
    except Exception as e:
        print(f"[WARN] Gamepad scan error: {e}")
    
    # Load MuJoCo model
    if not model_xml.exists():
        print(f"[ERROR] Model XML not found: {model_xml}")
        print("[INFO] Run export_mjcf.py first to generate the scene XML")
        return
    
    print("[INFO] Loading MuJoCo model...")
    # Use absolute path to ensure mesh files are found relative to XML location
    model_xml_abs = model_xml.absolute()
    mj_model = mujoco.MjModel.from_xml_path(str(model_xml_abs))
    data = mujoco.MjData(mj_model)
    
    print(f"[INFO] Model loaded: nq={mj_model.nq}, nv={mj_model.nv}, nu={mj_model.nu}")
    
    # Set initial pose
    if init_pose_json and init_pose_json.exists():
        print(f"[INFO] Loading initial pose from: {init_pose_json}")
        load_initial_pose(init_pose_json, mj_model, data)
    else:
        mujoco.mj_forward(mj_model, data)
    
    # Initialize gamepad if enabled
    gamepad = None
    keyboard = None
    if use_gamepad:
        try:
            gamepad = GamepadController(
                button_policy_map=button_policy_map,
                max_lin_vel=max_lin_vel,
                max_lat_vel=max_lat_vel,
                max_ang_vel=max_ang_vel,
                exit_button=exit_button,
                debug=gamepad_debug,
                policy_cycle=policy_cycle,
                cycle_button=cycle_button,
                initial_policy_path=str(policy_path),
                preferred_name=gamepad_preferred_name,
                preferred_index_1based=gamepad_preferred_index,
            )
            print("[INFO] Gamepad initialized - using analog sticks for control")
        except RuntimeError as e:
            print(f"[WARN] Gamepad initialization failed: {e}")
            print("[INFO] Falling back to keyboard control")
            use_gamepad = False
    
    # Initialize keyboard if gamepad not available or keyboard explicitly enabled
    use_keyboard = CONFIG.get("use_keyboard", True)
    if (not use_gamepad and use_keyboard) or (use_gamepad and use_keyboard):
        # Will initialize after viewer is created (need window handle)
        pass
    
    # Create controller
    print("[INFO] Creating policy controller...")
    # Load explicit mapping for defaults
    policy_defaults_map = CONFIG.get("policy_defaults", {}) or {}
    if not isinstance(policy_defaults_map, dict):
        raise TypeError("CONFIG['policy_defaults'] must be a dict of {policy_path: default_set}")
    # Validate presence for initial policy and any policies in the cycle
    init_key = str(policy_path)
    if init_key not in policy_defaults_map:
        raise KeyError(f"Initial policy '{init_key}' missing from CONFIG['policy_defaults']")
    for p in policy_cycle or []:
        if str(p) not in policy_defaults_map:
            raise KeyError(f"Policy '{p}' in policy_cycle missing from CONFIG['policy_defaults']")
    
    # Choose per-policy default joint positions (explicit)
    init_defaults = resolve_default_joint_positions(policy_path, policy_defaults_map)
    
    controller = StandalonePolicyController(
        policy_path=policy_path,
        mj_model=mj_model,
        device=device,
        action_scale=action_scale,
        training_defaults=init_defaults,
        kp_multiplier_legs=kp_multiplier_legs,
        kd_multiplier_legs=kd_multiplier_legs,
        kp_multiplier_upper=kp_multiplier_upper,
        kd_multiplier_upper=kd_multiplier_upper,
        policy_defaults_map=policy_defaults_map,
    )

    # Enable diagnostics logging if configured
    if CONFIG.get("diag_log", False):
        diag_file = CONFIG.get("diag_file", "")
        if not diag_file:
            # Build absolute diagnostics directory under project root
            project_root = Path(__file__).resolve().parents[1]
            diag_dir_cfg = CONFIG.get("diag_dir", "outputs/diagnostics")
            diag_dir = Path(diag_dir_cfg)
            if not diag_dir.is_absolute():
                diag_dir = project_root / diag_dir
            os.makedirs(str(diag_dir), exist_ok=True)
            diag_prefix = CONFIG.get("diag_prefix", "sim2sim")
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            diag_file = str(diag_dir / f"{diag_prefix}_{ts}.csv")
        controller.enable_diagnostics(diag_file)
        print(f"[INFO] Diagnostics logging enabled: {diag_file}")
    else:
        print("[INFO] Diagnostics logging disabled")
    
    # Launch viewer
    print("[INFO] Launching viewer...")
    print("\n" + "=" * 60)
    print("SAFETY MONITORING ENABLED")
    print("=" * 60)
    print("Real robot safety thresholds are active:")
    print(f"  - Position jump: Max {controller._max_position_jump:.2f} rad/timestep")
    print(f"  - Velocity spike: Max {controller._max_velocity:.1f} rad/s")
    print(f"  - Acceleration spike: Max {controller._max_acceleration:.1f} rad/s²")
    print("\nLOUD warnings will appear if policy would trigger safety shutoff!")
    print("=" * 60 + "\n")
    
    with viewer.launch_passive(
        model=mj_model,
        data=data,
        show_left_ui=False,
        show_right_ui=False
    ) as v:
        # Set camera from config
        v.cam.distance = camera_distance
        v.cam.azimuth = camera_azimuth
        v.cam.elevation = camera_elevation
        
        # Initialize keyboard controller if needed
        gain_step = CONFIG.get("gain_step", 0.1)
        if (not use_gamepad and use_keyboard) or (use_gamepad and use_keyboard):
            try:
                keyboard = KeyboardController(
                    max_lin_vel=max_lin_vel,
                    max_lat_vel=max_lat_vel,
                    max_ang_vel=max_ang_vel,
                    policy_cycle=policy_cycle,
                    initial_policy_path=str(policy_path),
                    gain_step=gain_step,
                )
                if not use_gamepad:
                    print("[INFO] Keyboard control enabled")
            except RuntimeError as e:
                print(f"[WARN] Keyboard initialization failed: {e}")
                print("[INFO] Install pynput with: pip install pynput")
                keyboard = None
        
        print("[INFO] Viewer launched. Running policy...")
        print(f"[INFO] Camera: distance={camera_distance}m, azimuth={camera_azimuth}°, tracking={camera_tracking}")
        if not use_gamepad and not keyboard:
            print("[INFO] No input device - robot will stand still")
        
        step = 0
        lin_vel_z = 0.0
        lin_vel_y = 0.0
        ang_vel_x = 0.0
        
        # Control mode state (for keyboard) - 初始为 policy 模式（站立状态）
        control_mode = "policy"  # 初始模式: policy (站立控制)
        
        # Helper function to set joint targets from angle dict
        def set_joint_targets_from_dict(angle_dict: dict[str, float], mj_model, data, controller):
            """Set joint targets from a dictionary of joint names to angles."""
            for i, joint_name in enumerate(controller.joint_names):
                if joint_name in angle_dict:
                    # Find the joint index in MuJoCo
                    joint_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                    if joint_id >= 0:
                        # Set control target (qpos index is 7 + i for robot joints)
                        if i < controller.num_robot_joints:
                            data.ctrl[i] = angle_dict[joint_name]
        
        while v.is_running():
            # Poll GLFW events to update gamepad state (keyboard library handles its own polling)
            if gamepad:
                glfw.poll_events()
            
            # Handle keyboard controls
            if keyboard:
                # Check exit (ESC key)
                if keyboard.check_exit():
                    break
                
                # Check mode switch (IJK keys)
                new_mode = keyboard.check_mode_switch()
                if new_mode:
                    control_mode = new_mode
                    keyboard.control_mode = new_mode
                
                # Check gain adjustment (Q/A/H keys)
                gain_delta, should_print_gain = keyboard.check_gain_adjustment()
                if gain_delta is not None:
                    # Adjust gains
                    new_kp_legs = controller.kp_multiplier_legs + gain_delta
                    new_kd_legs = controller.kd_multiplier_legs + gain_delta
                    new_kp_upper = controller.kp_multiplier_upper + gain_delta
                    new_kd_upper = controller.kd_multiplier_upper + gain_delta
                    controller.set_gain_multipliers(
                        kp_multiplier_legs=max(0.0, new_kp_legs),
                        kd_multiplier_legs=max(0.0, new_kd_legs),
                        kp_multiplier_upper=max(0.0, new_kp_upper),
                        kd_multiplier_upper=max(0.0, new_kd_upper),
                    )
                    print(f"[KEYBOARD] Gain multiplier updated: legs={controller.kp_multiplier_legs:.2f}, upper={controller.kp_multiplier_upper:.2f}")
                elif should_print_gain:
                    print(f"[KEYBOARD] Current gain multiplier: legs={controller.kp_multiplier_legs:.2f}, upper={controller.kp_multiplier_upper:.2f}")
                
                # Get velocity commands (only in policy mode)
                if control_mode == "policy":
                    lin_vel_z, lin_vel_y, ang_vel_x = keyboard.get_velocity_commands()
                    
                    # Check for policy switch (only in policy mode)
                    new_policy_path = keyboard.check_policy_switch()
                    if new_policy_path:
                        controller.load_policy(Path(new_policy_path))
                else:
                    lin_vel_z, lin_vel_y, ang_vel_x = 0.0, 0.0, 0.0
            
            # Handle gamepad controls
            elif gamepad:
                lin_vel_z, lin_vel_y, ang_vel_x = gamepad.get_velocity_commands()
                
                # Print debug info (catches unmapped buttons)
                gamepad.print_debug_info()
                
                # Check for policy switch
                new_policy_path = gamepad.check_policy_switch()
                if new_policy_path:
                    controller.load_policy(Path(new_policy_path))
                
                # Check for exit button
                if gamepad.check_exit():
                    # Don't terminate GLFW yet - viewer is still using it
                    # Just mark gamepad as done and break
                    break
            
            # Handle different control modes (keyboard only)
            if keyboard and control_mode != "policy":
                if control_mode == "default_pos":
                    # Set to default standing position
                    set_joint_targets_from_dict(STAND_DEFAULT_ANGLES, mj_model, data, controller)
                elif control_mode == "crawl_pos":
                    # Set to crawl position (use training defaults which is crawl pose)
                    set_joint_targets_from_dict(TRAINING_DEFAULT_ANGLES, mj_model, data, controller)
                elif control_mode == "damped":
                    # Damped mode: set all controls to zero (high damping)
                    data.ctrl[:controller.num_robot_joints] = 0.0
                # policy mode is handled below
            
            # Get robot state
            root_quat = data.qpos[3:7]
            joint_pos = data.qpos[7:7+controller.num_robot_joints]
            joint_vel = data.qvel[6:6+controller.num_robot_joints]  # Skip free joint velocities
            
            # Only run policy in policy mode
            if (keyboard and control_mode == "policy") or (gamepad and not keyboard):
                # Compute projected gravity
                proj_gravity = compute_projected_gravity(root_quat)
                
                # Get observation
                obs = controller.get_observation(
                    projected_gravity=proj_gravity,
                    joint_pos=joint_pos,
                    joint_vel=joint_vel,
                    lin_vel_z=lin_vel_z,
                    lin_vel_y=lin_vel_y,
                    ang_vel_x=ang_vel_x,
                )
                
                # Get action from policy (every n_substeps)
                if step % n_substeps == 0:
                    joint_targets = controller.get_action(obs, joint_pos, joint_vel)
                    data.ctrl[:controller.num_robot_joints] = joint_targets
            
            # Step simulation
            mujoco.mj_step(mj_model, data)
            step += 1
            
            # Update camera to track robot (if enabled)
            if camera_tracking and step % 10 == 0:  # Update every 10 steps for smoothness
                root_pos = data.qpos[0:3]
                v.cam.lookat[:] = root_pos  # Camera follows robot position
            
            # Sync viewer
            if step % 2 == 0:
                v.sync()
            
            # Print status
            # if step % 500 == 0:
            #     root_pos = data.qpos[0:3]
            #     vel_str = f"vel=[{lin_vel_z:.2f}, {ang_vel_x:.2f}]" if use_gamepad else ""
            #     print(f"Step {step}: pos=[{root_pos[0]:.2f}, {root_pos[1]:.2f}, {root_pos[2]:.2f}] {vel_str}")
    
    # Cleanup keyboard listener if it was created
    if keyboard and hasattr(keyboard, 'cleanup'):
        keyboard.cleanup()
    
    # Don't need to explicitly cleanup gamepad - GLFW is shared with viewer
    # and will be cleaned up automatically when the process exits
    
    print("[INFO] Deployment finished.")


if __name__ == "__main__":
    main()

