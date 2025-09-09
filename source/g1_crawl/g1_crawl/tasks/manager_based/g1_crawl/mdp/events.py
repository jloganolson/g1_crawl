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

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

# Global variable to track when push lines were last drawn
_push_lines_timestamp = 0.0
_push_lines_drawn = False


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
