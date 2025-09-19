from __future__ import annotations

import torch
from typing import TYPE_CHECKING
import math
import random

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import ObservationTermCfg
from isaaclab.sensors import Camera, Imu, RayCaster, RayCasterCamera, TiledCamera
from ..g1 import get_animation

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedEnv

from isaaclab.envs.utils.io_descriptors import (
    generic_io_descriptor,
    record_shape,
)

def compute_animation_phase_and_frame(env: ManagerBasedRLEnv):
    """Compute per-env animation phase and frame index from episode time and phase offset.

    Returns:
        phase: Tensor [num_envs] in [0, 1)
        frame_idx: LongTensor [num_envs] in [0, T-1]
    """
    anim = get_animation()
    T = int(anim["num_frames"])
    if T <= 0:
        raise ValueError("Animation has no frames (num_frames <= 0)")
    anim_dt = float(anim["dt"]) if "dt" in anim else 1.0 / 30.0
    cycle_time = float(T) * anim_dt
    # if not math.isfinite(cycle_time) or cycle_time <= 0.0:
    #     raise ValueError(f"Invalid cycle_time computed: {cycle_time} (T={T}, anim_dt={anim_dt})")
    if not hasattr(env, "_anim_phase_offset"):
        # raise RuntimeError("Missing _anim_phase_offset on env. Ensure reset_from_animation (reset) or init_anim_phase (startup) ran.")
        print("Missing _anim_phase_offset on env. Ensure reset_from_animation (reset) or init_anim_phase (startup) ran.")
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32), torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    # Choose device: prefer env.device, else use phase offset's device
    device = getattr(env, "device", getattr(env._anim_phase_offset, "device"))  # type: ignore[attr-defined]

    # Episode time per env (GPU)
    t_s = env.episode_length_buf.to(device=device, dtype=torch.float32) * float(env.step_dt)
    phase_offset = env._anim_phase_offset.to(device=device, dtype=torch.float32)  # type: ignore[attr-defined]


    phase = (phase_offset + (t_s / cycle_time)) % 1.0


    frame_idx = torch.floor(phase * float(T)).to(dtype=torch.long, device=device)
    return phase, frame_idx


@generic_io_descriptor(dtype=torch.float32, observation_type="RootState", on_inspect=[record_shape])
def animation_phase(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Per-env animation phase in [0,1), as a column vector for observations."""
    phase, _ = compute_animation_phase_and_frame(env)
    return phase.unsqueeze(1)


    
@generic_io_descriptor(dtype=torch.float32, observation_type="Action", on_inspect=[record_shape])
def last_action_with_log(env: ManagerBasedEnv, action_name: str | None = None) -> torch.Tensor:
    """The last input action to the environment.

    The name of the action term for which the action is required. If None, the
    entire action tensor is returned.
    """
    action = env.action_manager.action
    action_max = action.max().item()
    action_min = action.min().item()
    if action_max > 40 or action_min < -40:
        action_max_idx = action.argmax().item()
        action_min_idx = action.argmin().item()
        
        # Convert flat indices to (env_idx, joint_idx)
        num_joints = action.shape[1]
        max_env_idx = action_max_idx // num_joints
        max_joint_idx = action_max_idx % num_joints
        min_env_idx = action_min_idx // num_joints
        min_joint_idx = action_min_idx % num_joints
        
        # Get joint names for debugging
        robot = env.scene["robot"]
        joint_names = robot.data.joint_names
        max_joint_name = joint_names[max_joint_idx] if max_joint_idx < len(joint_names) else f"joint_{max_joint_idx}"
        min_joint_name = joint_names[min_joint_idx] if min_joint_idx < len(joint_names) else f"joint_{min_joint_idx}"
        
        print(f"action_max: {action_max} (env={max_env_idx}, joint={max_joint_name}), action_min: {action_min} (env={min_env_idx}, joint={min_joint_name})")

    if action_name is None:
        return env.action_manager.action
    else:
        return env.action_manager.get_term(action_name).raw_actions

