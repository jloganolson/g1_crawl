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
    candidate = os.path.join(repo_root, "assets/animation_rc3.json")

    return candidate


def load_animation_json(json_path: str | None = None) -> dict:
    """Load animation JSON containing qpos frames and metadata.

    Returns dict with keys:
    - dt: float
    - nq: int
        - nv: int
    - qpos: torch.FloatTensor [T, nq] (CPU tensor)
        - qvel: torch.FloatTensor [T, nv] (CPU tensor) if provided
    - qpos_labels: list[str] | None
        - qvel_labels: list[str] | None
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
    nv = int(data.get("nv", 0)) or None

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
    # Optional velocities: accept keys "qvel" or "vel_frames"
    qvel_tensor = None
    if "qvel" in data and data["qvel"] is not None:
        qvel_list = data["qvel"]
    elif "vel_frames" in data and data["vel_frames"] is not None:
        qvel_list = data["vel_frames"]
    else:
        qvel_list = None
    if qvel_list is not None:
        qvel_tensor = torch.tensor(qvel_list, dtype=torch.float32, device="cpu")
        # Coerce shape if needed and reconcile nv
        if qvel_tensor.ndim != 2 or qvel_tensor.shape[0] != T:
            try:
                qvel_tensor = qvel_tensor.view(T, -1)
            except Exception:
                qvel_tensor = None
        if qvel_tensor is not None:
            if nv is not None and qvel_tensor.shape[1] != nv:
                nv = qvel_tensor.shape[1]
            elif nv is None:
                nv = qvel_tensor.shape[1]

    metadata = data.get("metadata", {}) or {}
    qpos_labels = data.get("qpos_labels", None) or metadata.get("qpos_labels", None)
    qvel_labels = data.get("qvel_labels", None) or metadata.get("qvel_labels", None)
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

    # Required per-frame contact flags: [T, K] with order specified in metadata.contact.order
    if "contact_flags" not in data or data["contact_flags"] is None:
        raise RuntimeError("Animation JSON missing required 'contact_flags' array")
    cf_list = data["contact_flags"]
    cf_tensor = torch.tensor(cf_list, dtype=torch.float32, device="cpu")
    if cf_tensor.ndim != 2 or int(cf_tensor.shape[0]) != int(T):
        raise RuntimeError("contact_flags must be a 2D array with shape [T, K] matching frame count T")
    if not isinstance(metadata, dict) or "contact" not in metadata or metadata["contact"] is None:
        raise RuntimeError("Animation JSON missing required 'metadata.contact' block")
    contact_meta = metadata["contact"]
    if not isinstance(contact_meta, dict):
        raise RuntimeError("metadata.contact must be a dict")
    if "order" not in contact_meta or contact_meta["order"] is None:
        raise RuntimeError("metadata.contact.order is required and must list contact labels")
    contact_order = contact_meta["order"]
    if not isinstance(contact_order, (list, tuple)) or len(contact_order) == 0:
        raise RuntimeError("metadata.contact.order must be a non-empty list")
    if int(cf_tensor.shape[1]) != int(len(contact_order)):
        raise RuntimeError(
            f"contact_flags column count {int(cf_tensor.shape[1])} does not match metadata.contact.order length {int(len(contact_order))}"
        )
    if "threshold_m" not in contact_meta or contact_meta["threshold_m"] is None:
        raise RuntimeError("metadata.contact.threshold_m is required")
    try:
        contact_threshold_m = float(contact_meta["threshold_m"])
    except Exception:
        raise RuntimeError("metadata.contact.threshold_m must be a float")
    contact_flags_tensor = cf_tensor

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

    # Move tensors to GPU for runtime use (fail loudly if CUDA not available)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for animation tensors; no GPU available")
    device = torch.device("cuda")
    qpos_tensor = qpos_tensor.to(device)
    if qvel_tensor is not None:
        qvel_tensor = qvel_tensor.to(device)
    if site_positions_tensor is not None:
        site_positions_tensor = site_positions_tensor.to(device)
    if contact_flags_tensor is not None:
        contact_flags_tensor = contact_flags_tensor.to(device)

    return {
        "dt": float(dt),
        "nq": int(nq),
        "nv": int(nv) if nv is not None else int(0),
        "qpos": qpos_tensor,
        "qvel": qvel_tensor if qvel_tensor is not None else None,
        "qpos_labels": qpos_labels,
        "qvel_labels": qvel_labels,
        "metadata": metadata,
        "base_meta": base_meta,
        "joints_meta": joints_meta,
        "nsite": int(nsite),
        "site_positions": site_positions_tensor,
        "sites_meta": sites_meta,
        "contact_flags": contact_flags_tensor if contact_flags_tensor is not None else None,
        "contact_order": list(contact_order) if contact_order is not None else None,
        "contact_threshold_m": float(contact_threshold_m) if contact_threshold_m is not None else None,
        "num_frames": int(T),
        "json_path": path,
    }


# ===== Animation helpers (moved from events.py) =====
from isaaclab.assets import Articulation  # type: ignore  # for type hints

_ANIM_CACHE: dict | None = None


def get_animation(json_path: str | None = None) -> dict:
    """Return cached animation dict; loads once via load_animation_json."""
    global _ANIM_CACHE
    if _ANIM_CACHE is None:
        _ANIM_CACHE = load_animation_json(json_path)
    return _ANIM_CACHE


def build_joint_index_map(asset: Articulation, joints_meta, qpos_labels):
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


def build_joint_velocity_index_map(asset: Articulation, joints_meta, qvel_labels, qpos_labels=None):
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