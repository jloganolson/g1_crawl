import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
import torch
import os
import json

# taken from https://github.com/HybridRobotics/whole_body_tracking/blob/dcecabd8c24c68f59d143fdf8e3a670f420c972d/source/whole_body_tracking/whole_body_tracking/robots/g1.py
ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2

DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ

G1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"assets/g1_23dof_simple-forearm.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, 
            solver_position_iteration_count=8, 
            solver_velocity_iteration_count=4,
            # fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.22), #.4267 according to 
        rot=(0.721249, -0.000001, 0.692676, -0.000012), # rpy(rad): [-0.000456, 1.530386, -0.000471]
        joint_pos={
            # Hip joints - identical pitch, inverse roll/yaw for left/right
            ".*_hip_pitch_joint": -1.6796101123595506,
            "left_hip_roll_joint": 2.4180011235955052,
            "right_hip_roll_joint": -2.4180011235955052,
            "left_hip_yaw_joint": 1.2083865168539325,
            "right_hip_yaw_joint": -1.2083865168539325,
            
            # Knee joints - identical for both legs
            ".*_knee_joint": 2.1130298764044944,
            
            # Ankle joints - identical pitch, zero roll
            ".*_ankle_pitch_joint": 0.194143033707865,
            ".*_ankle_roll_joint": 0.0,
            
            # Waist
            "waist_yaw_joint": 0.0,
            
            # Shoulder joints - identical pitch, inverse roll/yaw for left/right
            ".*_shoulder_pitch_joint": 1.4578526315789473,
            "left_shoulder_roll_joint": 1.5778684210526317,
            "right_shoulder_roll_joint": -1.5778684210526317,
            "left_shoulder_yaw_joint": 1.4238245614035088,
            "right_shoulder_yaw_joint": -1.4238245614035088,
            
            # Elbow and wrist joints - identical for both arms
            ".*_elbow_joint": -0.3124709677419355,
            ".*_wrist_roll_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
                # "waist_yaw_joint",
            ],
            effort_limit_sim={
                ".*_hip_yaw_joint": 88.0,
                ".*_hip_roll_joint": 139.0,
                ".*_hip_pitch_joint": 88.0,
                ".*_knee_joint": 139.0,
                # "waist_yaw_joint": 88.0,
            },
            velocity_limit_sim={
                ".*_hip_yaw_joint": 32.0,
                ".*_hip_roll_joint": 20.0,
                ".*_hip_pitch_joint": 32.0,
                ".*_knee_joint": 20.0,
            },
            stiffness={
                ".*_hip_pitch_joint": STIFFNESS_7520_14,
                ".*_hip_roll_joint": STIFFNESS_7520_22,
                ".*_hip_yaw_joint": STIFFNESS_7520_14,
                ".*_knee_joint": STIFFNESS_7520_22,
            },
            damping={
                ".*_hip_pitch_joint": DAMPING_7520_14,
                ".*_hip_roll_joint": DAMPING_7520_22,
                ".*_hip_yaw_joint": DAMPING_7520_14,
                ".*_knee_joint": DAMPING_7520_22,
            },
            armature={
                ".*_hip_pitch_joint": ARMATURE_7520_14,
                ".*_hip_roll_joint": ARMATURE_7520_22,
                ".*_hip_yaw_joint": ARMATURE_7520_14,
                ".*_knee_joint": ARMATURE_7520_22,
            },
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit_sim=50.0,
            velocity_limit_sim=37.0,
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=2.0 * STIFFNESS_5020,
            damping=2.0 * DAMPING_5020,
            armature=2.0 * ARMATURE_5020,
        ),
        "waist_yaw": ImplicitActuatorCfg(
            effort_limit_sim=88,
            velocity_limit_sim=32.0,
            joint_names_expr=["waist_yaw_joint"],
            stiffness=STIFFNESS_7520_14,
            damping=DAMPING_7520_14,
            armature=ARMATURE_7520_14,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_roll_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 25.0,
                ".*_shoulder_roll_joint": 25.0,
                ".*_shoulder_yaw_joint": 25.0,
                ".*_elbow_joint": 25.0,
                ".*_wrist_roll_joint": 25.0,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 37.0,
                ".*_shoulder_roll_joint": 37.0,
                ".*_shoulder_yaw_joint": 37.0,
                ".*_elbow_joint": 37.0,
                ".*_wrist_roll_joint": 37.0,
            },
            stiffness={
                ".*_shoulder_pitch_joint": STIFFNESS_5020,
                ".*_shoulder_roll_joint": STIFFNESS_5020,
                ".*_shoulder_yaw_joint": STIFFNESS_5020,
                ".*_elbow_joint": STIFFNESS_5020,
                ".*_wrist_roll_joint": STIFFNESS_5020,
            },
            damping={
                ".*_shoulder_pitch_joint": DAMPING_5020,
                ".*_shoulder_roll_joint": DAMPING_5020,
                ".*_shoulder_yaw_joint": DAMPING_5020,
                ".*_elbow_joint": DAMPING_5020,
                ".*_wrist_roll_joint": DAMPING_5020,
            },
            armature={
                ".*_shoulder_pitch_joint": ARMATURE_5020,
                ".*_shoulder_roll_joint": ARMATURE_5020,
                ".*_shoulder_yaw_joint": ARMATURE_5020,
                ".*_elbow_joint": ARMATURE_5020,
                ".*_wrist_roll_joint": ARMATURE_5020,
            },
        ),
    },
)


def _default_animation_path() -> str:
    # return "assets/animation_rc0.json"
    # """Return the default animation JSON path.

    # Priority:
    # 1) Environment variable `G1_ANIMATION_JSON`
    # 2) Repository path: scripts/experiments/animation_20250915_134944.json
    # """
    # env_path = os.environ.get("G1_ANIMATION_JSON", None)
    # if env_path and os.path.exists(env_path):
    #     return env_path

    # # Compute repo root from this file
    this_dir = os.path.dirname(__file__)
    repo_root = os.path.abspath(os.path.join(this_dir, "../../../../../../"))
    # candidate = os.path.join(repo_root, "scripts/experiments/animation_20250915_134944.json")
    candidate = os.path.join(repo_root, "assets/animation_rc1.json")

    return candidate


def load_animation_json(json_path: str | None = None) -> dict:
    """Load animation JSON containing qpos frames and metadata.

    Returns dict with keys:
    - dt: float
    - nq: int
    - qpos: torch.FloatTensor [T, nq] (CPU tensor)
    - qpos_labels: list[str] | None
    - metadata: dict
    - base_meta: dict | None (with pos_indices, quat_indices)
    - joints_meta: list|dict mapping joint names to qpos indices
    - num_frames: int
    - json_path: str
    """
    path = json_path or _default_animation_path()
    if not os.path.exists(path):
        raise FileNotFoundError(f"Animation JSON not found: {path}")

    with open(path, "r") as f:
        data = json.load(f)

    if "dt" in data:
        dt = float(data["dt"])
    elif "fps" in data and data["fps"]:
        dt = 1.0 / float(data["fps"])
    else:
        dt = 1.0 / 30.0

    nq = int(data.get("nq", 0)) or None

    if "qpos" in data:
        qpos_list = data["qpos"]
    elif "frames" in data:
        qpos_list = data["frames"]
    elif "positions" in data:
        qpos_list = data["positions"]
    else:
        raise KeyError("Animation JSON missing 'qpos' (or 'frames'/'positions') array")

    T = len(qpos_list)
    qpos_tensor = torch.tensor(qpos_list, dtype=torch.float32, device="cpu")
    if nq is not None and qpos_tensor.shape[1] != nq:
        nq = qpos_tensor.shape[1]
    elif nq is None:
        nq = qpos_tensor.shape[1]

    metadata = data.get("metadata", {}) or {}
    qpos_labels = data.get("qpos_labels", None) or metadata.get("qpos_labels", None)
    base_meta = metadata.get("base", None)
    joints_meta = metadata.get("joints", {}) or {}
    # Sites metadata and positions (optional)
    sites_meta = metadata.get("sites", {}) or {}
    nsite = int(data.get("nsite", 0) or sites_meta.get("nsite", 0) or 0)
    site_positions_tensor = None
    if "site_positions" in data and data["site_positions"] is not None:
        # Expecting [T, nsite, 3]
        site_positions_tensor = torch.tensor(data["site_positions"], dtype=torch.float32, device="cpu")
        # Basic sanity: ensure time dimension matches
        if site_positions_tensor.ndim != 3 or site_positions_tensor.shape[0] != T:
            # Try to coerce if possible; otherwise, ignore
            try:
                site_positions_tensor = site_positions_tensor.view(T, -1, 3)
                nsite = int(site_positions_tensor.shape[1])
            except Exception:
                site_positions_tensor = None

    # Normalize base world x/y so the animation starts at the origin.
    # If base position indices are provided, subtract the first frame's x/y from all frames.
    if base_meta is not None:
        pos_idx = base_meta.get("pos_indices", None)
        if isinstance(pos_idx, (list, tuple)) and len(pos_idx) >= 2:
            x_idx = int(pos_idx[0])
            y_idx = int(pos_idx[1])
            x0 = float(qpos_tensor[0, x_idx].item())
            y0 = float(qpos_tensor[0, y_idx].item())
            if x0 != 0.0 or y0 != 0.0:
                qpos_tensor[:, x_idx] -= x0
                qpos_tensor[:, y_idx] -= y0
                # Apply the same shift to site world positions if provided
                if site_positions_tensor is not None:
                    site_positions_tensor[:, :, 0] -= x0
                    site_positions_tensor[:, :, 1] -= y0

    return {
        "dt": float(dt),
        "nq": int(nq),
        "qpos": qpos_tensor,
        "qpos_labels": qpos_labels,
        "metadata": metadata,
        "base_meta": base_meta,
        "joints_meta": joints_meta,
        "nsite": int(nsite),
        "site_positions": site_positions_tensor,
        "sites_meta": sites_meta,
        "num_frames": int(T),
        "json_path": path,
    }