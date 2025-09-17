# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import  RigidObject
from isaaclab.assets import Articulation

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# Reuse animation helpers
from ..g1 import get_animation, build_joint_index_map
from .observations import compute_animation_phase_and_frame


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def animation_pose_similarity_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Negative L2 pose error between current joint positions and animation frame joints.

    - Uses a per-env advancing animation frame counter stored on the env (initialized on reset).
    - Advances the frame counter each call by step_dt / anim_dt frames.
    - Excludes floating base by relying on the joint index map built from animation metadata.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    device = asset.device

    anim = get_animation()
    qpos: torch.Tensor = anim["qpos"]  # [T, nq] on GPU
    T = int(anim["num_frames"])
    anim_dt = float(anim["dt"]) if "dt" in anim else 1.0 / 30.0

    # Build and cache joint index map on the env
    if not hasattr(env, "_anim_joint_index_map"):
        index_map_list = build_joint_index_map(asset, anim.get("joints_meta"), anim.get("qpos_labels"))
        index_map = torch.tensor(index_map_list, dtype=torch.long, device=torch.device("cpu"))
        setattr(env, "_anim_joint_index_map", index_map)
    index_map_cpu: torch.Tensor = env._anim_joint_index_map  # type: ignore[attr-defined]

    # Derive frame from episode time + phase offset (single source of truth)
    # Get frame indices on device and index GPU qpos directly
    _, frame_idx = compute_animation_phase_and_frame(env)

    # Compute integer frame indices and gather target joints
    target_qpos = qpos[frame_idx]  # [N, nq]
    target_joint = target_qpos.index_select(dim=1, index=index_map_cpu.to(device))  # [N, ndofs]
    target_joint = target_joint.to(dtype=asset.data.joint_pos.dtype)

    # Current joint positions
    current_joint = asset.data.joint_pos
    err = current_joint - target_joint
    # Negative L2 squared per env
    return -torch.sum(err * err, dim=1)


def animation_forward_velocity_similarity_exp(
    env: ManagerBasedRLEnv,
    std: float = 0.2,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Exponential tracking of forward (world +X) velocity to animation metadata target.

    Uses metadata key 'base_forward_velocity_mps' if available; otherwise returns zeros.
    reward = exp(- (vx - vx_target)^2 / std^2)
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    anim = get_animation()
    meta = anim.get("metadata", {}) or {}
    vx_target = float(meta.get("base_forward_velocity_mps", 0.0))
    if vx_target == 0.0 and "base_forward_velocity_mps" not in meta:
        return torch.zeros(asset.data.root_lin_vel_w.shape[0], device=asset.device)
    vx = asset.data.root_lin_vel_w[:, 0]
    err_sq = torch.square(vx - vx_target)
    return torch.exp(-err_sq / (std ** 2))


def track_lin_vel_yz_base_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands in base YZ plane using exponential kernel.

    Compares the commanded [vz, vy] (base frame) to measured base linear velocity [vz, vy].
    """
    asset = env.scene[asset_cfg.name]
    vel_b = asset.data.root_lin_vel_b[:, :3]
    cmd_yz = env.command_manager.get_command(command_name)[:, :2]  # [vz, vy]
    meas_yz = vel_b[:, [2, 1]]  # [vz, vy]
    lin_vel_error = torch.sum(torch.square(cmd_yz - meas_yz), dim=1)
    return torch.exp(-lin_vel_error / (std ** 2))


def track_ang_vel_x_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of roll angular velocity command (ωx) in world frame using exponential kernel."""
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(
        env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 0]
    )
    return torch.exp(-ang_vel_error / (std ** 2))


def both_feet_air(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize when both feet are in the air.

    This function penalizes the agent when both feet are off the ground simultaneously, which helps
    promote stable bipedal locomotion by encouraging at least one foot to maintain ground contact.
    
    Args:
        env: The environment instance.
        sensor_cfg: Configuration for the contact sensor.

    Returns:
        1 if both feet are airborne, 0 otherwise.
    """
    # contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # # Check if feet are in contact (contact force > threshold)
    # in_contact = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    # # Count feet in contact
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    feet_in_air = air_time > 0.0
    both_feet_air = torch.sum(feet_in_air.int(), dim=1) == 2
    return both_feet_air.float()


def lin_vel_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize all base linear velocity using L2 squared kernel.

    This penalizes any movement of the robot's base in x, y, or z directions.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_lin_vel_b[:, :3]), dim=1)


def ang_vel_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize all base angular velocity using L2 squared kernel.

    This penalizes any rotation of the robot's base about x, y, or z axes.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :3]), dim=1)


def align_projected_gravity_to_target_l2(
    env: ManagerBasedRLEnv,
    target: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward alignment of projected gravity with an arbitrary target direction in base frame.

    This uses the projected gravity in base frame ``g_b = asset.data.projected_gravity_b`` and
    a target direction (3,) or per-env targets (num_envs, 3). The target is normalized internally.

    reward = 1 - 0.5 * || g_b - \hat{target} ||^2, yielding values in [-1, 1].

    Args:
        env: RL environment.
        target: Desired gravity direction in base frame. Shape (3,) or (num_envs, 3).
        asset_cfg: Scene entity for the robot asset.

    Returns:
        Tensor of shape (num_envs,) with alignment rewards in [-1, 1].
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    g_b = asset.data.projected_gravity_b  # (num_envs, 3)

    # Prepare target on correct device/dtype and broadcast if needed
    if target.dim() == 1:
        target_b = target.unsqueeze(0).expand(g_b.shape[0], -1)
    elif target.dim() == 2 and target.shape[0] == g_b.shape[0] and target.shape[1] == 3:
        target_b = target
    else:
        raise ValueError("target must have shape (3,) or (num_envs, 3)")
    target_b = target_b.to(dtype=g_b.dtype, device=g_b.device)

    # Normalize target direction per env to avoid scale effects
    target_norm = torch.clamp(torch.norm(target_b, dim=1, keepdim=True), min=1e-6)
    target_b = target_b / target_norm

    dist_sq = torch.sum(torch.square(g_b - target_b), dim=1)  # in [0, 4]
    return 1.0 - 0.5 * dist_sq


def align_projected_gravity_plus_x_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward alignment of projected gravity with +X axis in base frame using L2 mapping.

    reward = 1 - 0.5 * || g_b - [1, 0, 0] ||^2, in [-1, 1].
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    g_b = asset.data.projected_gravity_b  # (num_envs, 3)
    target = torch.tensor([1.0, 0.0, 0.0], dtype=g_b.dtype, device=g_b.device)
    dist_sq = torch.sum(torch.square(g_b - target), dim=1)
    return 1.0 - 0.5 * dist_sq