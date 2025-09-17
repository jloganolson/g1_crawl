# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from .g1 import G1_CFG

##
# Scene definition
##


@configclass
class G1CrawlSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""
  # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAAC_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = MISSING
    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # imu_pelvis = ImuCfg(prim_path="{ENV_REGEX_NS}/Robot/base",offset)
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.CrawlVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.1,
        rel_heading_envs=1.0,
        heading_command=False,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.CrawlVelocityCommandCfg.Ranges(
            heading=(0.0,0.0),
            # Crawling fields used by the command implementation
            lin_vel_z=(0, 1.0),
            lin_vel_y=(0.,0.),
            ang_vel_x=(-1.0, 1.0)
        )
    )


    # base_velocity = mdp.UniformVelocityCommandCfg(
    #     asset_name="robot",
    #     resampling_time_range=(10.0, 10.0),
    #     rel_standing_envs=0.02,
    #     rel_heading_envs=1.0,
    #     heading_command=True,
    #     heading_control_stiffness=0.5,
    #     debug_vis=True,
    #     ranges=mdp.UniformVelocityCommandCfg.Ranges(
    #         lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
    #     ),
    # )



@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: ObsGroup = PolicyCfg()
    critic: ObsGroup = CriticCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    # Floor / foot friction: =U(0.4, 1.0) - matching MuJoCo rand_dynamics
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.4, 1.0),
            "dynamic_friction_range": (0.4, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    # Scale all link masses: *U(0.9, 1.1) - matching MuJoCo rand_dynamics
    randomize_body_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )

    # Add mass to torso: +U(-1.0, 1.0) - matching MuJoCo rand_dynamics
    add_torso_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-1.0, 1.0),
            "operation": "add",
        },
    )

    # Scale static friction: *U(0.5, 2.0) - matching MuJoCo rand_dynamics joint friction
    randomize_joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": (0.5, 2.0),
            "operation": "scale",
        },
    )

    # Scale armature: *U(1.0, 1.05) - matching MuJoCo rand_dynamics
    randomize_joint_armature = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "armature_distribution_params": (1.0, 1.05),
            "operation": "scale",
        },
    )
    # reset
    # Replace uniform base reset with animation-based reset
    reset_base = EventTerm(
        func=mdp.reset_from_animation,
        mode="reset",
        params={},
    )

    # Disable default joint reset since animation sets joints explicitly
    # reset_robot_joints = EventTerm(
    #     func=mdp.reset_joints_by_scale,
    #     mode="reset",
    #     params={
    #         "position_range": (1.0, 1.0),
    #         "velocity_range": (0.0, 0.0),
    #     },
    # )

    # interval
    viz_anim_sites = EventTerm(
        func=mdp.viz_animation_sites_step,
        mode="interval",
        interval_range_s=(0.0,0.0),
        params={
            "max_envs": 32,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    viz_base_step = EventTerm(
        func=mdp.viz_base_positions_step,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={
            "max_envs": 32,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    # push_robot = EventTerm(
    #     func=mdp.push_by_setting_velocity_with_viz,
    #     mode="interval",
    #     interval_range_s=(2.0, 2.0),
    #     params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    # )



@configclass
class RewardsCfg:
    """Reward terms for the MDP."""


    #hold still
    # lin_vel_l2 = RewTerm(func=mdp.lin_vel_l2, weight=-5.0)
    # ang_vel_l2 = RewTerm(func=mdp.ang_vel_l2, weight=-5.0)

    #follow commands (base YZ plane and roll about X)
    track_lin_vel_yz_exp = RewTerm(
        func=mdp.track_lin_vel_yz_base_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_x_exp = RewTerm(
        func=mdp.track_ang_vel_x_world_exp, weight=2.0, params={"command_name": "base_velocity", "std": 0.5}
    )
    flat_orientation_l2 = RewTerm(func=mdp.align_projected_gravity_plus_x_l2, weight=1.0)
    
    
    # termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    
    
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=-.1,
        params={
            "target_height": 0.22,
            "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
        },
    )

    joint_deviation_all = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    
    #limits
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    torque_limits = RewTerm(
        func=mdp.applied_torque_limits,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    #regulatorization
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1e-2)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-4)

    bellyhead_drag_penalty = RewTerm(
        func=mdp.undesired_contacts,
        weight=-5.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names= "torso_link",
            ),
            "threshold": 1.0,  # in Newtons (normal force magnitude)
        },
    )
    slippage = RewTerm(
        func=mdp.feet_slide, 
        weight=-.01,
                params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_ankle_roll_link", ".*_wrist_link"]),
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*_ankle_roll_link", ".*_wrist_link"]),
        },
    )

    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.05,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_ankle_roll_link", ".*_wrist_link"]),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )

    # Animation-tracking rewards (initial small weights)
    anim_pose_similarity = RewTerm(
        func=mdp.animation_pose_similarity_l2,
        weight=0.5,
        params={},
    )
    anim_forward_vel = RewTerm(
        func=mdp.animation_forward_velocity_similarity_exp,
        weight=0.5,
        params={"std": 0.2},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # base_contact = DoneTerm(
    #     func=mdp.illegal_contact,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    # )

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    push_magnitude = CurrTerm(
        func=mdp.modify_event_parameter, params={"num_steps": 8000}
    )

    # self.events.push_robot.params params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}}


@configclass
class G1CrawlEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: G1CrawlSceneCfg = G1CrawlSceneCfg(num_envs=4096, env_spacing=4.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()

    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    # curriculum: CurriculumCfg = CurriculumCfg()


    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

# 
        self.scene.robot = G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"

        # update sensor update periods
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # Set terrain to plane and disable height scanning
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        # self.observations.policy.height_scan = None
        # self.curriculum.terrain_levels = None

        # Randomization

        # self.events.base_external_force_torque.params["asset_cfg"].body_names = ["torso_link"]

        # Rewards
        # self.rewards.track_ang_vel_z_exp.weight = 1.0
        # self.rewards.lin_vel_z_l2.weight = -0.2
        # self.rewards.dof_torques_l2.weight = -1.0e-4
        # self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg(
        #     "robot", joint_names=[".*"]
        # )

        # self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
        #     "robot", joint_names=[".*"]
        # )

        # Commands
        # Crawling fields
        # self.commands.base_velocity.ranges.lin_vel_z = (-1.0, 1.0)
        # self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        # self.commands.base_velocity.ranges.ang_vel_x = (-1.0, 1.0)

        # terminations
        # self.terminations.base_contact.params["sensor_cfg"].body_names = "torso_link"
