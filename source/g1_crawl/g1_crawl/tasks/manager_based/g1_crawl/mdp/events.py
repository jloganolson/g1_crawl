from __future__ import annotations

import math
import torch
import time
import numpy as np
from typing import TYPE_CHECKING, Literal

import carb
import omni.physics.tensors.impl.api as physx
import omni.usd
from isaacsim.core.utils.extensions import enable_extension
from pxr import Gf, Sdf, UsdGeom, Vt


# Try to import debug drawing - gracefully handle headless mode
try:
    import isaacsim.util.debug_draw._debug_draw as omni_debug_draw
    DEBUG_DRAW_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    # In headless mode or when debug drawing is not available
    omni_debug_draw = None
    DEBUG_DRAW_AVAILABLE = False

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, DeformableObject, RigidObject
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from .observations import compute_animation_phase_and_frame
from ..g1 import get_animation

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

# Global variable to track when push lines were last drawn
_push_lines_timestamp = 0.0
_push_lines_drawn = False
# Separate tracker for site point visualization
_site_points_timestamp = 0.0
_site_points_drawn = False
# Separate tracker for base position points visualization
_base_points_timestamp = 0.0
_base_points_drawn = False
# Separate tracker for per-step animation site visualization
_anim_sites_timestamp = 0.0
_anim_sites_drawn = False


def is_visualization_available() -> bool:
    """
    Check if visualization is available (not in headless mode).
    
    Returns:
        bool: True if debug drawing is available, False if in headless mode or not available
    """
    return DEBUG_DRAW_AVAILABLE


def push_by_setting_velocity_with_viz(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Push the asset by setting the root velocity to a random value within the given ranges.

    This creates an effect similar to pushing the asset with a random impulse that changes the asset's velocity.
    It samples the root velocity from the given ranges and sets the velocity into the physics simulation.

    The function takes a dictionary of velocity ranges for each axis and rotation. The keys of the dictionary
    are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form ``(min, max)``.
    If the dictionary does not contain a key, the velocity is set to zero for that axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # velocities
    vel_w = asset.data.root_vel_w[env_ids]
    # sample random velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    sampled_velocities = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], vel_w.shape, device=asset.device)
    vel_w += sampled_velocities
    
    # Add debug visualization for the applied velocities
    visualize_applied_velocities(
        asset_positions=asset.data.root_pos_w[env_ids],
        applied_velocities=sampled_velocities,
        env_ids=env_ids,
        duration=1.0,
        arrow_scale=2.0
    )
    
    # set the velocities into the physics simulation
    asset.write_root_velocity_to_sim(vel_w, env_ids=env_ids)


def visualize_applied_velocities(
    asset_positions: torch.Tensor,
    applied_velocities: torch.Tensor, 
    env_ids: torch.Tensor,
    duration: float = 1.0,
    arrow_scale: float = 2.0,
    height_offset: float = 0.5,
    min_magnitude_threshold: float = 0.01,
    arrowhead_length: float = 0.3,
    arrowhead_angle: float = 25.0,
    arrow_color_override: tuple[float, float, float, float] | None = None,
    print_debug_info: bool = True
):
    """
    Visualize applied velocities as colored arrows using Isaac Sim's debug drawing interface.
    
    This function creates arrow visualizations showing the direction and magnitude of applied velocities.
    Arrows are colored based on magnitude (red = strong, yellow = weak) and include arrowheads for clarity.
    
    Note: Visualization is automatically disabled in headless mode (during training) to prevent errors.
    
    Args:
        asset_positions: Positions of assets in world frame. Shape: [num_envs, 3]
        applied_velocities: Applied velocity vectors. Shape: [num_envs, 6] (linear + angular)
        env_ids: Environment IDs being affected
        duration: How long to show the arrows (seconds)
        arrow_scale: Scaling factor for arrow length visualization
        height_offset: Height offset above asset position to draw arrows
        min_magnitude_threshold: Minimum velocity magnitude to visualize
        arrowhead_length: Length of arrowhead lines
        arrowhead_angle: Angle of arrowhead lines (degrees)
        arrow_color_override: Fixed color for all arrows (R,G,B,A). If None, uses magnitude-based coloring
        print_debug_info: Whether to print debug information
    
    Usage example in other event functions:
        ```python
        # After applying some effect to assets
        visualize_applied_velocities(
            asset_positions=asset.data.root_pos_w[env_ids],
            applied_velocities=sampled_forces,  # or velocities, or any vector
            env_ids=env_ids,
            duration=2.0,  # Show for 2 seconds
            arrow_scale=1.5
        )
        ```
    """
    # Early return if visualization is not available (headless mode)
    if not is_visualization_available():
        if print_debug_info and len(env_ids) <= 4:
            print(f"[DEBUG VIZ] Headless mode - skipping visualization for {len(env_ids)} environments")
        return
    
    global _push_lines_timestamp, _push_lines_drawn
    
    # Get debug drawing interface
    draw_interface = omni_debug_draw.acquire_debug_draw_interface()
    
    # Clear old visualization if duration has passed
    current_time = time.time()
    if _push_lines_drawn and (current_time - _push_lines_timestamp) > duration:
        draw_interface.clear_lines()
        _push_lines_drawn = False

    # Prepare lists for line drawing
    line_start_points = []
    line_end_points = []
    line_colors = []
    line_sizes = []
    
    # Convert angle to radians for calculations
    angle_rad = np.radians(arrowhead_angle)
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)
    
    for i, env_id in enumerate(env_ids):
        # Get the applied velocity for this environment (only linear components)
        applied_vel = applied_velocities[i, :3].cpu().numpy()  # x, y, z components
        robot_pos = asset_positions[i].cpu().numpy()
        
        # Skip if velocity magnitude is too small to visualize
        vel_magnitude = float(torch.norm(applied_velocities[i, :3]))
        if vel_magnitude < min_magnitude_threshold:
            continue
            
        # Calculate arrow start and end points
        arrow_start = [
            float(robot_pos[0]), 
            float(robot_pos[1]), 
            float(robot_pos[2] + height_offset)
        ]
        arrow_end = [
            float(robot_pos[0] + applied_vel[0] * arrow_scale),
            float(robot_pos[1] + applied_vel[1] * arrow_scale), 
            float(robot_pos[2] + height_offset + applied_vel[2] * arrow_scale)
        ]
        
        # Color based on velocity magnitude (red = strong, yellow = weak) or use override
        if arrow_color_override is not None:
            arrow_color = arrow_color_override
        else:
            color_intensity = min(vel_magnitude / 2.0, 1.0)  # Normalize to 0-1
            arrow_color = (1.0, 1.0 - color_intensity, 0.0, 1.0)  # Red to yellow gradient
        
        # Add main arrow line
        line_start_points.append(arrow_start)
        line_end_points.append(arrow_end)
        line_colors.append(arrow_color)
        line_sizes.append(4.0)
        
        # Create arrowhead in horizontal plane for better visibility
        direction_2d = np.array([arrow_end[0] - arrow_start[0], arrow_end[1] - arrow_start[1]])
        if np.linalg.norm(direction_2d) > 0:
            direction_2d = direction_2d / np.linalg.norm(direction_2d)
            
            # Left arrowhead line
            arrowhead_dir1 = np.array([
                -direction_2d[0] * cos_angle + direction_2d[1] * sin_angle,
                -direction_2d[1] * cos_angle - direction_2d[0] * sin_angle
            ])
            arrowhead_end1 = [
                arrow_end[0] + arrowhead_dir1[0] * arrowhead_length,
                arrow_end[1] + arrowhead_dir1[1] * arrowhead_length,
                arrow_end[2]
            ]
            line_start_points.append(arrow_end)
            line_end_points.append(arrowhead_end1)
            line_colors.append(arrow_color)
            line_sizes.append(3.0)
            
            # Right arrowhead line
            arrowhead_dir2 = np.array([
                -direction_2d[0] * cos_angle - direction_2d[1] * sin_angle,
                -direction_2d[1] * cos_angle + direction_2d[0] * sin_angle
            ])
            arrowhead_end2 = [
                arrow_end[0] + arrowhead_dir2[0] * arrowhead_length,
                arrow_end[1] + arrowhead_dir2[1] * arrowhead_length,
                arrow_end[2]
            ]
            line_start_points.append(arrow_end)
            line_end_points.append(arrowhead_end2)
            line_colors.append(arrow_color)
            line_sizes.append(3.0)
    
    # Draw all the arrows if any were created
    if line_start_points:
        # Clear previous lines before drawing new ones
        if _push_lines_drawn:
            draw_interface.clear_lines()
        
        draw_interface.draw_lines(line_start_points, line_end_points, line_colors, line_sizes)
        _push_lines_timestamp = current_time
        _push_lines_drawn = True
        
        # Optional debug info (limited to avoid spam)
        if print_debug_info and len(env_ids) <= 4:
            for i, env_id in enumerate(env_ids):
                vel_mag = float(torch.norm(applied_velocities[i, :3]))
                if vel_mag > min_magnitude_threshold:
                    print(f"[DEBUG VIZ] Applied velocity to env {env_id.item()}: magnitude = {vel_mag:.3f}")


# ===== HOW TO ADD VISUALIZATION TO OTHER EVENT FUNCTIONS =====
"""
To add visualization to any event function, follow this simple pattern:

1. Do your event logic normally
2. Call visualize_applied_velocities() with your data

Example:
```python
def my_event_with_viz(env, env_ids, params, asset_cfg):
    # Your normal event logic
    asset = env.scene[asset_cfg.name]
    sampled_values = sample_something(...)
    apply_changes(asset, sampled_values, env_ids)
    
    # Add visualization (just 4 lines!)
    viz_vectors = torch.zeros(len(env_ids), 6, device=asset.device)
    viz_vectors[:, :3] = sampled_values * scale_factor
    visualize_applied_velocities(asset.data.root_pos_w[env_ids], viz_vectors, env_ids)
```

Note: Visualization automatically detects headless mode and gracefully skips drawing
to prevent import errors during training.
"""

def reset_root_state_to_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the asset root state to a random position and velocity uniformly within the given ranges.

    This function randomizes the root position and velocity of the asset.

    * It samples the root position from the given ranges and adds them to the default root position, before setting
      them into the physics simulation.
    * It samples the root orientation from the given ranges and sets them into the physics simulation.
    * It samples the root velocity from the given ranges and sets them into the physics simulation.

    The function takes a dictionary of pose and velocity ranges for each axis and rotation. The keys of the
    dictionary are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form
    ``(min, max)``. If the dictionary does not contain a key, the position or velocity is set to zero for that axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    # get default root state
    root_states = asset.data.default_root_state[env_ids].clone()

    # poses
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)
    # velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    velocities = root_states[:, 7:13] + rand_samples

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


# ===== Animation-based reset helpers and event =====
from ..g1 import get_animation


def _build_joint_index_map(asset: Articulation, joints_meta, qpos_labels):
    robot_joint_names = asset.data.joint_names
    index_map: list[int] = []
    missing: list[str] = []

    name_to_qposadr: dict[str, int] = {}
    if isinstance(joints_meta, dict):
        for name, qposadr in joints_meta.items():
            try:
                name_to_qposadr[str(name)] = int(qposadr)
            except Exception:
                continue
    elif isinstance(joints_meta, list):
        for item in joints_meta:
            if not isinstance(item, dict):
                continue
            jname = item.get("name")
            jtype = item.get("type")
            adr = item.get("qposadr")
            dim = item.get("qposdim", 1)
            if jname is not None and jtype in ("hinge", "slide") and isinstance(adr, int) and int(dim) == 1:
                name_to_qposadr[str(jname)] = int(adr)

    label_lookup: dict[str, int] = {}
    if qpos_labels is not None:
        for i, lbl in enumerate(qpos_labels):
            s = str(lbl)
            label_lookup[s] = i
            if s.startswith("joint:"):
                label_lookup[s[len("joint:"):]] = i

    for jn in robot_joint_names:
        qidx = -1
        if jn in name_to_qposadr:
            qidx = name_to_qposadr[jn]
        else:
            if jn in label_lookup:
                qidx = label_lookup[jn]
            elif ("joint:" + jn) in label_lookup:
                qidx = label_lookup["joint:" + jn]
        index_map.append(qidx)
        if qidx == -1:
            missing.append(jn)

    if missing:
        print(f"[WARN] Missing qpos indices for {len(missing)} joints (will keep defaults)")
    return index_map


def _build_joint_velocity_index_map(asset: Articulation, joints_meta, qvel_labels, qpos_labels=None):
    """Build per-joint mapping into the animation's qvel vector.

    Preference order per joint:
    1) joints_meta.qveladr if provided
    2) joints_meta.qposadr (hinge/slide 1-DoF often match)
    3) label lookup in qvel_labels
    4) fallback to qpos label lookup
    """
    robot_joint_names = asset.data.joint_names
    index_map: list[int] = []
    missing: list[str] = []

    name_to_qveladr: dict[str, int] = {}
    if isinstance(joints_meta, dict):
        for name, adr in joints_meta.items():
            try:
                # Allow either qveladr directly or legacy qposadr
                if isinstance(adr, dict):
                    if "qveladr" in adr and isinstance(adr["qveladr"], int):
                        name_to_qveladr[str(name)] = int(adr["qveladr"])  # type: ignore[index]
                    elif "qposadr" in adr and isinstance(adr["qposadr"], int):
                        name_to_qveladr[str(name)] = int(adr["qposadr"])  # type: ignore[index]
                else:
                    name_to_qveladr[str(name)] = int(adr)
            except Exception:
                continue
    elif isinstance(joints_meta, list):
        for item in joints_meta:
            if not isinstance(item, dict):
                continue
            jname = item.get("name")
            jtype = item.get("type")
            vadr = item.get("qveladr")
            padr = item.get("qposadr")
            vdim = item.get("qveldim", 1)
            pdim = item.get("qposdim", 1)
            # Only 1-DoF joints are considered here
            if (
                jname is not None
                and jtype in ("hinge", "slide")
                and ((isinstance(vadr, int) and int(vdim) == 1) or (isinstance(padr, int) and int(pdim) == 1))
            ):
                if isinstance(vadr, int) and int(vdim) == 1:
                    name_to_qveladr[str(jname)] = int(vadr)
                elif isinstance(padr, int) and int(pdim) == 1:
                    name_to_qveladr[str(jname)] = int(padr)

    # Label lookup from qvel labels, with fallback to qpos labels
    label_lookup: dict[str, int] = {}
    if qvel_labels is not None:
        for i, lbl in enumerate(qvel_labels):
            s = str(lbl)
            label_lookup[s] = i
            if s.startswith("joint:"):
                label_lookup[s[len("joint:"):]] = i
    if not label_lookup and qpos_labels is not None:
        for i, lbl in enumerate(qpos_labels):
            s = str(lbl)
            label_lookup[s] = i
            if s.startswith("joint:"):
                label_lookup[s[len("joint:"):]] = i

    for jn in robot_joint_names:
        qidx = -1
        if jn in name_to_qveladr:
            qidx = name_to_qveladr[jn]
        else:
            if jn in label_lookup:
                qidx = label_lookup[jn]
            elif ("joint:" + jn) in label_lookup:
                qidx = label_lookup["joint:" + jn]
        index_map.append(qidx)
        if qidx == -1:
            missing.append(jn)

    if missing:
        print(f"[WARN] Missing qvel indices for {len(missing)} joints (will keep default zeros)")
    return index_map


def init_animation_phase_offsets(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None = None,
):
    """Initialize per-env animation phase offsets to zeros on device.

    This ensures observations that use phase can run before any reset-based randomization.
    """
    print("init_animation_phase_offsets")
    asset: Articulation = env.scene["robot"]
    device = asset.device
    num_envs = env.num_envs
    setattr(env, "_anim_phase_offset", torch.zeros(num_envs, device=device, dtype=torch.float32))
 # if not hasattr(env, "_anim_phase_offset"):
    #     setattr(env, "_anim_phase_offset", torch.zeros(env.num_envs, device=asset.device, dtype=torch.float32))
    


def reset_from_animation(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    json_path: str | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset root pose, joint positions, and joint velocities from a random frame of an animation JSON.

    Requires base pose indices and per-frame joint velocities to be present in the animation.
    Raises an error if any required data or joint index mapping is missing.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    device = asset.device

    anim = get_animation(json_path)

    # Build index maps once per call (cheap) — could be cached by joint_names hash if needed
    joint_index_map = _build_joint_index_map(asset, anim.get("joints_meta"), anim.get("qpos_labels"))
    joint_vel_index_map = _build_joint_velocity_index_map(
        asset,
        anim.get("joints_meta"),
        anim.get("qvel_labels"),
        anim.get("qpos_labels"),
    )

    num_envs = len(env_ids)
    num_robot_dofs = asset.data.default_joint_pos.shape[1]

    # Choose random frames per env
    T = int(anim["num_frames"])
    frame_indices = torch.randint(low=0, high=T, size=(num_envs,), device="cpu")

    # Also store per-env phase offsets and global cycle time for downstream use (rewards)
    phase_offsets = (frame_indices.to(device=asset.device, dtype=torch.float32) / float(T))
    # if not hasattr(env, "_anim_phase_offset"):
    #     setattr(env, "_anim_phase_offset", torch.zeros(env.num_envs, device=asset.device, dtype=torch.float32))
    env._anim_phase_offset[env_ids] = phase_offsets  # type: ignore[attr-defined]
    # env.animation_phase_offset[env_ids] = phase_offsets
    # store cycle time in seconds
    # setattr(env, "_anim_cycle_time_s", float(T) * float(anim["dt"]))

    # Prepare root states
    root_state = asset.data.default_root_state[env_ids].clone()
    base_meta = anim.get("base_meta")
    if base_meta is None:
        raise ValueError("Animation JSON is missing base metadata 'base_meta'.")
    pos_idx = base_meta.get("pos_indices", None)
    quat_idx = base_meta.get("quat_indices", None)
    if pos_idx is None or quat_idx is None:
        raise ValueError("Animation base metadata must include 'pos_indices' and 'quat_indices'.")

    # env origins
    env_origins = env.scene.env_origins[env_ids].to(device=device)

    # Apply base pose from animation (required)
    qpos = anim["qpos"]  # GPU tensor [T, nq]
    base_pos = torch.stack([qpos[int(fi), pos_idx] for fi in frame_indices], dim=0).to(device=device)
    wxyz = torch.stack([qpos[int(fi), quat_idx] for fi in frame_indices], dim=0).to(device=device)
    root_state[:, 0:3] = base_pos.to(device) + env_origins
    root_state[:, 3:7] = wxyz.to(device)

    # Zero root velocities (we currently do not map base velocities unless metadata provides explicit indices)
    root_state[:, 7:13] = 0.0

    # Prepare joint states
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = torch.zeros_like(joint_pos, device=joint_pos.device)

    # Apply joint positions from animation
    nq = int(anim["nq"])
    # Required joint velocities from animation
    qvel = anim.get("qvel", None)
    if not isinstance(qvel, torch.Tensor):
        raise ValueError("Animation JSON must include per-frame velocities 'qvel'/'vel_frames'.")
    nv = int(anim.get("nv", 0) or qvel.shape[1])
    if nv <= 0:
        raise ValueError("Animation 'nv' must be > 0 when providing velocities.")

    # Validate joint index maps: crash if any joint is unmapped
    robot_joint_names = asset.data.joint_names
    missing_pos = [jn for jn, idx in zip(robot_joint_names, joint_index_map) if not (isinstance(idx, int) and idx >= 0)]
    if missing_pos:
        raise ValueError(f"Missing qpos indices for joints: {missing_pos}")
    missing_vel = [jn for jn, idx in zip(robot_joint_names, joint_vel_index_map) if not (isinstance(idx, int) and idx >= 0)]
    if missing_vel:
        raise ValueError(f"Missing qvel indices for joints: {missing_vel}")

    for i in range(num_envs):
        fi = int(frame_indices[i])
        qrow = qpos[fi]  # CPU
        for j_idx in range(num_robot_dofs):
            qidx = joint_index_map[j_idx] if j_idx < len(joint_index_map) else -1
            if isinstance(qidx, int) and 0 <= qidx < nq:
                joint_pos[i, j_idx] = qrow[qidx].to(joint_pos.device)
        # Apply joint velocities (required)
        vrow = qvel[fi]
        for j_idx in range(num_robot_dofs):
            vidx = joint_vel_index_map[j_idx] if j_idx < len(joint_vel_index_map) else -1
            if isinstance(vidx, int) and 0 <= vidx < nv:
                joint_vel[i, j_idx] = vrow[vidx].to(joint_vel.device)

    # Write to sim
    asset.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
    asset.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    # # Visualize site world positions for the selected frames (if provided in animation)
    # sites = anim.get("site_positions", None)
    # nsite = int(anim.get("nsite", 0) or 0)
    # if is_visualization_available():
    #     draw_interface = omni_debug_draw.acquire_debug_draw_interface()
    #     # Clear old points if expired
    #     current_time = time.time()
    #     if _site_points_drawn and (current_time - _site_points_timestamp) > 1.0:
    #         draw_interface.clear_points()
    #         # Some versions only support clear_lines; fall back if needed
    #         try:
    #             pass
    #         except Exception:
    #             try:
    #                 draw_interface.clear_lines()
    #             except Exception:
    #                 pass
    #         globals()["_site_points_drawn"] = False

    #     point_list = []
    #     colors = []
    #     sizes = []
    #     # Fixed color and size for all site points
    #     color = (0.1, 0.7, 1.0, 1.0)
    #     size = 8
    #     for i in range(num_envs):
    #         fi = int(frame_indices[i])
    #         pts = sites[fi].detach().cpu()  # [nsite, 3]
    #         # Add env origin offset
    #         origin = env_origins[i].cpu()
    #         for j in range(int(pts.shape[0])):
    #             x = float(pts[j, 0].item() + origin[0].item())
    #             y = float(pts[j, 1].item() + origin[1].item())
    #             z = float(pts[j, 2].item() + origin[2].item())
    #             point_list.append((x, y, z))
    #             colors.append(color)
    #             sizes.append(size)

    #     if point_list:
    #         try:
    #             draw_interface.draw_points(point_list, colors, sizes)
    #             globals()["_site_points_timestamp"] = current_time
    #             globals()["_site_points_drawn"] = True
    #         except Exception as e:
    #             # In cases where draw_points isn't available, skip silently
    #             print(f"[DEBUG VIZ] draw_points unavailable: {e}")

    # Also draw a single point at each env's base world position for 30 seconds to verify visibility
    # if is_visualization_available():
    #     draw_interface = omni_debug_draw.acquire_debug_draw_interface()
    #     current_time = time.time()
    #     # Clear old base points after 30 seconds
    #     if _base_points_drawn and (current_time - _base_points_timestamp) > 30.0:
    #         try:
    #             draw_interface.clear_points()
    #         except Exception:
    #             try:
    #                 draw_interface.clear_lines()
    #             except Exception:
    #                 pass
    #         globals()["_base_points_drawn"] = False

    #     # Build point list at root_state world positions (already includes env origins)
    #     base_points = []
    #     base_colors = []
    #     base_sizes = []
    #     base_color = (1.0, 0.1, 0.1, 1.0)
    #     base_size = 20
    #     rs_cpu = root_state[:, 0:3].detach().cpu()
    #     for i in range(len(env_ids)):
    #         p = rs_cpu[i]
    #         base_points.append((float(p[0].item()), float(p[1].item()), float(p[2].item())))
    #         base_colors.append(base_color)
    #         base_sizes.append(base_size)
    #     if base_points:
    #         try:
    #             draw_interface.draw_points(base_points, base_colors, base_sizes)
    #             globals()["_base_points_timestamp"] = current_time
    #             globals()["_base_points_drawn"] = True
    #         except Exception as e:
    #             print(f"[DEBUG VIZ] draw_points (base) unavailable: {e}")


def viz_animation_sites_step(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    max_envs: int = 4,
    throttle_steps: int = 1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Visualize animation site points each step for a few envs, centered at current base pose.

    - Uses compute_animation_phase_and_frame(env) for consistent phase.
    - Offsets animation sites by env origin and current base world position.
    - Throttled by throttle_steps and limited to max_envs for performance.
    """
    if not is_visualization_available():
        return
    # Throttle by common step counter
    if throttle_steps > 1 and (getattr(env, "common_step_counter", 0) % throttle_steps != 0):
        return

    anim = get_animation()
    sites = anim.get("site_positions", None)
    nsite = int(anim.get("nsite", 0) or 0)
    if sites is None or nsite <= 0:
        return

    asset: Articulation = env.scene[asset_cfg.name]
    # Determine which envs to draw
    draw_env_ids = env_ids[: max_envs]
    if len(draw_env_ids) == 0:
        return

    # Compute frame indices
    _, frame_idx = compute_animation_phase_and_frame(env)
    # Current base positions
    base_pos_w = asset.data.root_pos_w[draw_env_ids]
    env_origins = env.scene.env_origins[draw_env_ids]

    draw_interface = omni_debug_draw.acquire_debug_draw_interface()
    # Overwrite any previous point drawings for a clean per-frame update
    draw_interface.clear_points()
    current_time = time.time()
    global _anim_sites_timestamp, _anim_sites_drawn

    point_list = []
    colors = []
    sizes = []
    color = (0.1, 0.9, 0.2, 1.0)
    size = 6
    # Build points per env
    fi_sel = frame_idx[draw_env_ids].detach().cpu()
    base_pos_cpu = base_pos_w.detach().cpu()
    origins_cpu = env_origins.detach().cpu()
    for i in range(len(draw_env_ids)):
        fi = int(fi_sel[i].item())
        pts = sites[fi].detach().cpu()  # [nsite, 3]
        origin = origins_cpu[i]
        base = base_pos_cpu[i]
        # Center around current base: shift animation sites by (origin + base)
        for j in range(nsite):
            x = float(pts[j, 0].item() + origin[0].item() + base[0].item())
            y = float(pts[j, 1].item() + origin[1].item() + base[1].item())
            z = float(pts[j, 2].item() + origin[2].item() + base[2].item())
            point_list.append((x, y, z))
            colors.append(color)
            sizes.append(size)

    if point_list:
        try:
            draw_interface.draw_points(point_list, colors, sizes)
            _anim_sites_timestamp = current_time
            _anim_sites_drawn = True
        except Exception as e:
            print(f"[DEBUG VIZ] draw_points (anim sites) unavailable: {e}")


# def viz_base_positions_step(
#     env: ManagerBasedEnv,
#     env_ids: torch.Tensor,
#     max_envs: int = 4,
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
# ):
#     """Visualize base world positions each step for a few envs to confirm interval events run.

#     Draws points at current base positions; no clearing to avoid wiping other debug draws.
#     """
#     if not is_visualization_available():
#         return
#     # if throttle_steps > 1 and (getattr(env, "common_step_counter", 0) % throttle_steps != 0):
#     #     return
    
#     asset: Articulation = env.scene[asset_cfg.name]
#     draw_env_ids = env_ids[: max_envs]
#     if len(draw_env_ids) == 0:
#         return
#     draw_interface = omni_debug_draw.acquire_debug_draw_interface()
#     base_pos_w = asset.data.root_pos_w[draw_env_ids].detach().cpu()
#     point_list = []
#     colors = []
#     sizes = []
#     color = (1.0, 0.1, 0.1, 1.0)
#     size = 12
#     for i in range(len(draw_env_ids)):
#         p = base_pos_w[i]
#         point_list.append((float(p[0].item()), float(p[1].item()), float(p[2].item())))
#         colors.append(color)
#         sizes.append(size)
#     draw_interface.draw_points(point_list, colors, sizes)
