# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for the velocity-based locomotion task."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

import math
from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, FRAME_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass



class CrawlVelocityCommand(CommandTerm):
    r"""Command generator that generates a velocity command in SE(2) from uniform distribution.

    The command comprises of a linear velocity in z and y direction and an angular velocity around
    the x-axis (roll). It is given in the robot's base frame.

    If the :attr:`cfg.heading_command` flag is set to True, the angular velocity is computed from the heading
    error similar to doing a proportional control on the heading error. The target heading is sampled uniformly
    from the provided range. Otherwise, the angular velocity is sampled uniformly from the provided range.

    Note: When :attr:`cfg.heading_command` is True, the current implementation still computes the error
    using the robot's heading around z (yaw) to remain backward compatible. Set ``heading_command=False``
    to use purely sampled roll rates, or adapt this path to roll targets as needed.
    """

    cfg: CrawlVelocityCommandCfg
    """The configuration of the command generator."""

    def __init__(self, cfg: CrawlVelocityCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.

        Raises:
            ValueError: If the heading command is active but the heading range is not provided.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # check configuration
        if self.cfg.heading_command and self.cfg.ranges.heading is None:
            raise ValueError(
                "The velocity command has heading commands active (heading_command=True) but the `ranges.heading`"
                " parameter is set to None."
            )
        if self.cfg.ranges.heading and not self.cfg.heading_command:
            omni.log.warn(
                f"The velocity command has the 'ranges.heading' attribute set to '{self.cfg.ranges.heading}'"
                " but the heading command is not active. Consider setting the flag for the heading command to True."
            )

        # obtain the robot asset
        # -- robot
        self.robot: Articulation = env.scene[cfg.asset_name]

        # crete buffers to store the command
        # -- command: z vel, y vel, roll vel, heading
        self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_target = torch.zeros(self.num_envs, device=self.device)
        self.is_heading_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_standing_env = torch.zeros_like(self.is_heading_env)
        # -- metrics
        self.metrics["error_vel_yz"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_roll"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "UniformVelocityCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tHeading command: {self.cfg.heading_command}\n"
        if self.cfg.heading_command:
            msg += f"\tHeading probability: {self.cfg.rel_heading_envs}\n"
        msg += f"\tStanding probability: {self.cfg.rel_standing_envs}"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity command in the base frame. Shape is (num_envs, 3)."""
        return self.vel_command_b

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # time for which the command was executed
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        # logs data
        # Linear velocity tracking in base YZ plane: command [vz, vy] vs measured [vz, vy]
        measured_lin_yz = self.robot.data.root_lin_vel_b[:, [2, 1]]
        lin_err = torch.norm(self.vel_command_b[:, :2] - measured_lin_yz, dim=-1) / max_command_step
        self.metrics["error_vel_yz"] += lin_err
        # Angular velocity tracking around base X (roll)
        ang_err = torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_ang_vel_b[:, 0]) / max_command_step
        self.metrics["error_vel_roll"] += ang_err

    def _resample_command(self, env_ids: Sequence[int]):
        # sample velocity commands
        r = torch.empty(len(env_ids), device=self.device)
        ranges = self.cfg.ranges
        lin_vel_z_range = ranges.lin_vel_z
        lin_vel_y_range = ranges.lin_vel_y
        ang_vel_x_range = ranges.ang_vel_x
        # -- linear velocity - z direction (index 0)
        self.vel_command_b[env_ids, 0] = r.uniform_(*lin_vel_z_range)
        # -- linear velocity - y direction (index 1)
        self.vel_command_b[env_ids, 1] = r.uniform_(*lin_vel_y_range)
        # -- angular velocity roll around x (index 2)
        self.vel_command_b[env_ids, 2] = r.uniform_(*ang_vel_x_range)
        # heading target
        if self.cfg.heading_command:
            self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
            # update heading envs
            self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
        # update standing envs
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    def _update_command(self):
        """Post-processes the velocity command.

        This function sets velocity command to zero for standing environments and computes angular
        velocity from heading direction if the heading_command flag is set.
        """
        # Compute angular velocity from heading direction
        if self.cfg.heading_command:
            # resolve indices of heading envs
            env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
            # compute angular velocity from heading error (backward compatible: yaw-based)
            heading_error = math_utils.wrap_to_pi(self.heading_target[env_ids] - self.robot.data.heading_w[env_ids])
            # clamp using roll range
            ranges = self.cfg.ranges
            ang_vel_x_range = ranges.ang_vel_x
            self.vel_command_b[env_ids, 2] = torch.clip(
                self.cfg.heading_control_stiffness * heading_error,
                min=ang_vel_x_range[0],
                max=ang_vel_x_range[1],
            )
        # Enforce standing (i.e., zero velocity command) for standing envs
        # TODO: check if conversion is needed
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[standing_env_ids, :] = 0.0

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first time
            if not hasattr(self, "goal_vel_visualizer"):
                # -- goal
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                # -- current (base alt1 - canonical)
                self.current_vel_base_alt1_visualizer = VisualizationMarkers(self.cfg.current_vel_base_alt1_visualizer_cfg)
            else:
                if not hasattr(self, "current_vel_base_alt1_visualizer"):
                    self.current_vel_base_alt1_visualizer = VisualizationMarkers(self.cfg.current_vel_base_alt1_visualizer_cfg)
            # set their visibility to true
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_base_alt1_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                if hasattr(self, "current_vel_base_alt1_visualizer"):
                    self.current_vel_base_alt1_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # get marker location
        # -- base state
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        # -- resolve the scales and quaternions
        # goal: rotate base [0, vy, vz] to world, then align arrow +X via yaw/pitch
        cmd_xyz_b = torch.zeros((self.num_envs, 3), device=self.device)
        cmd_xyz_b[:, 1] = self.command[:, 1]
        cmd_xyz_b[:, 2] = self.command[:, 0]
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_base_velocity_to_arrow_via_world(cmd_xyz_b)
        # current (canonical: rotate base lin vel to world, align +X via yaw/pitch)
        measured_lin_xyz_b = self.robot.data.root_lin_vel_b
        vel_base_alt1_scale, vel_base_alt1_quat = self._resolve_base_velocity_to_arrow_via_world(measured_lin_xyz_b)
        # display markers
        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_base_alt1_visualizer.visualize(base_pos_w, vel_base_alt1_quat, vel_base_alt1_scale)

    """
    Internal helpers.
    """

    def _resolve_yz_velocity_to_arrow(self, yz_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Deprecated: YZ base-plane arrow. Kept for reference; not used for goal anymore."""
        # obtain default scale of the marker
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(yz_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(yz_velocity, dim=1) * 3.0
        # arrow-direction
        phi = torch.atan2(yz_velocity[:, 1], yz_velocity[:, 0])  # atan2(vy, vz) with ordering [vz, vy]
        zeros = torch.zeros_like(phi)
        pitch_neg_90 = torch.full_like(phi, -math.pi / 2.0)
        align_x_to_z = math_utils.quat_from_euler_xyz(zeros, pitch_neg_90, zeros)
        roll_by_phi = math_utils.quat_from_euler_xyz(phi, zeros, zeros)
        # Apply alignment first, then roll around the aligned X axis: arrow = roll ⊗ align
        arrow_quat = math_utils.quat_mul(roll_by_phi, align_x_to_z)
        # convert everything back from base to world frame
        base_quat_w = self.robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat

    def _resolve_base_velocity_to_arrow_via_world(self, xyz_velocity_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the base XYZ velocity to world, then aligns arrow +X to it (yaw/pitch).

        This should match the world-frame visualization if frame transforms are consistent.
        """
        # scale by 3D norm of base velocity
        default_scale = self.current_vel_base_alt1_visualizer.cfg.markers["arrow"].scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xyz_velocity_b.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xyz_velocity_b, dim=1) * 3.0
        # rotate base velocity into world frame
        base_quat_w = self.robot.data.root_quat_w
        vel_w = math_utils.quat_apply(base_quat_w, xyz_velocity_b)
        # orient arrow like world method
        vx = vel_w[:, 0]
        vy = vel_w[:, 1]
        vz = vel_w[:, 2]
        yaw = torch.atan2(vy, vx)
        horiz = torch.sqrt(vx * vx + vy * vy)
        pitch = torch.atan2(-vz, horiz)
        zeros = torch.zeros_like(yaw)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, pitch, yaw)
        return arrow_scale, arrow_quat



@configclass
class CrawlVelocityCommandCfg(CommandTermCfg):
    """Configuration for the uniform velocity command generator."""

    class_type: type = CrawlVelocityCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    heading_command: bool = False
    """Whether to use heading command or angular velocity command. Defaults to False.

    If True, the angular velocity command is computed from the heading error, where the
    target heading is sampled uniformly from provided range. Otherwise, the angular velocity
    command is sampled uniformly from provided range.
    """

    heading_control_stiffness: float = 1.0
    """Scale factor to convert the heading error to angular velocity command. Defaults to 1.0."""

    rel_standing_envs: float = 0.0
    """The sampled probability of environments that should be standing still. Defaults to 0.0."""

    rel_heading_envs: float = 1.0
    """The sampled probability of environments where the robots follow the heading-based angular velocity command
    (the others follow the sampled angular velocity command). Defaults to 1.0.

    This parameter is only used if :attr:`heading_command` is True.
    """

    @configclass
    class Ranges:
        """Uniform distribution ranges for the velocity commands in base YZ plane with roll about X."""

        lin_vel_z: tuple[float, float] = MISSING
        """Range for the linear-z velocity command (in m/s)."""

        lin_vel_y: tuple[float, float] = MISSING
        """Range for the linear-y velocity command (in m/s)."""

        ang_vel_x: tuple[float, float] = MISSING
        """Range for the angular-x (roll) velocity command (in rad/s)."""

        heading: tuple[float, float] | None = None
        """Range for the heading command (in rad). Defaults to None.

        This parameter is only used if :attr:`~UniformVelocityCommandCfg.heading_command` is True.
        """

    ranges: Ranges = MISSING
    """Distribution ranges for the velocity commands."""

    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )
    """The configuration for the goal velocity visualization marker. Defaults to GREEN_ARROW_X_MARKER_CFG."""

    current_vel_base_alt1_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current_base_alt1"
    )
    """Alternative base-frame visualization: rotate base velocity into world, yaw/pitch align."""

    # Set the scale of the visualization markers to (0.5, 0.5, 0.5)
    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    current_vel_base_alt1_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)

    # Unique colors per visualizer for clarity
    # goal: green (from GREEN_ARROW_X_MARKER_CFG)
    current_vel_base_alt1_visualizer_cfg.markers["arrow"].color = (1.0, 0.6, 0.1)
    

