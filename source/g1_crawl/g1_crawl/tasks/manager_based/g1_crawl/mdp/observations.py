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

    # Shape checks
    # if t_s.ndim != 1 or t_s.shape[0] != env.num_envs:
    #     raise ValueError(f"t_s has invalid shape {tuple(t_s.shape)}; expected ({env.num_envs},)")
    # if phase_offset.ndim != 1 or phase_offset.shape[0] != env.num_envs:
    #     raise ValueError(f"phase_offset has invalid shape {tuple(phase_offset.shape)}; expected ({env.num_envs},)")

    # Finite checks before using values
    # if not torch.isfinite(t_s).all():
    #     num_bad = (~torch.isfinite(t_s)).sum().item()
    #     raise ValueError(f"t_s contains non-finite values (count={num_bad}); step_dt={float(env.step_dt)}")
    # if not torch.isfinite(phase_offset).all():
    #     num_bad = (~torch.isfinite(phase_offset)).sum().item()
    #     raise ValueError(f"phase_offset contains non-finite values (count={num_bad})")

    phase = (phase_offset + (t_s / cycle_time)) % 1.0

    # Phase validity checks
    # if phase.ndim != 1 or phase.shape[0] != env.num_envs:
    #     raise ValueError(f"phase has invalid shape {tuple(phase.shape)}; expected ({env.num_envs},)")
    # if not torch.isfinite(phase).all():
    #     num_bad = (~torch.isfinite(phase)).sum().item()
    #     raise ValueError(f"phase contains non-finite values (count={num_bad})")
    # if not (torch.ge(phase, 0.0).all() and torch.lt(phase, 1.0).all()):
    #     pmin = phase.min().item()
    #     pmax = phase.max().item()
    #     raise ValueError(f"phase out of range [0,1): min={pmin}, max={pmax}")

    # Rare debug print (≈1 in 5000 calls)
    # if random.randint(1, 5000) == 1:
    #     print(
    #         f"[observations] anim dbg: T={T}, anim_dt={anim_dt:.6g}, cycle_time={cycle_time:.6g}, "
    #         f"step_dt={float(env.step_dt):.6g}, num_envs={env.num_envs}, phase.shape={tuple(phase.shape)}, "
    #         f"t_s=[min={t_s.min().item():.6g}, max={t_s.max().item():.6g}], "
    #         f"phase_offset=[min={phase_offset.min().item():.6g}, max={phase_offset.max().item():.6g}], "
    #         f"phase=[min={phase.min().item():.6g}, max={phase.max().item():.6g}]"
    #     )
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
    if not torch.isfinite(action).all():
        num_nan = torch.isnan(action).sum().item()
        num_posinf = (action == float("inf")).sum().item()
        num_neginf = (action == float("-inf")).sum().item()
        raise RuntimeError(f"last_action_with_log: action contains non-finite values: NaN={num_nan}, +Inf={num_posinf}, -Inf={num_neginf}")
    if action_max > 50 or action_min < -50:
        print(action_max, action_min)
    # env.extras["metrics"] = {
    #     "action_max": action_max,
    #     "action_min": action_min,
    # }
    if action_name is None:
        return env.action_manager.action
    else:
        return env.action_manager.get_term(action_name).raw_actions

