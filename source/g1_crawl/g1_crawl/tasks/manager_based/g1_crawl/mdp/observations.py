from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import ObservationTermCfg
from isaaclab.sensors import Camera, Imu, RayCaster, RayCasterCamera, TiledCamera
from ..g1 import get_animation

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

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
    if not hasattr(env, "_anim_phase_offset"):
        raise RuntimeError("Missing _anim_phase_offset on env. Ensure reset_from_animation was called.")

    # Choose device: prefer env.device, else use phase offset's device
    device = getattr(env, "device", getattr(env._anim_phase_offset, "device"))  # type: ignore[attr-defined]

    # Episode time per env (GPU)
    t_s = env.episode_length_buf.to(device=device, dtype=torch.float32) * float(env.step_dt)
    phase_offset = env._anim_phase_offset.to(device=device, dtype=torch.float32)  # type: ignore[attr-defined]
    phase = (phase_offset + (t_s / cycle_time)) % 1.0
    frame_idx = torch.floor(phase * float(T)).to(dtype=torch.long, device=device)
    return phase, frame_idx