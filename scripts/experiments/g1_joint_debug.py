from isaaclab.app import AppLauncher

# launch omniverse app
app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

from g1_crawl.tasks.manager_based.g1_crawl.g1 import G1_CFG

from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import AssetBaseCfg
import isaaclab.sim as sim_utils

import torch
import numpy as np
import json
import math

# Add carb and omni for keyboard input handling
import carb
import omni


# Configure which base axis corresponds to "belly-forward" alignment with gravity in base frame.
# For G1, +X (pelvis forward) is desired, so axis='x'. If flipped, set sign=-1.0.
BELLY_TARGET_AXIS = 'x'  # one of: 'x', 'y', 'z'
BELLY_TARGET_SIGN = 1.0


class DebugSceneCfg(InteractiveSceneCfg):
    """Scene configuration for joint debugging."""

    # Ground-plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # robot - keep fix_root_link=True for stable debugging
    Robot = G1_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=G1_CFG.spawn.replace(
            articulation_props=G1_CFG.spawn.articulation_props.replace(
                fix_root_link=True
            )
        )
    )


def get_joint_info(scene):
    """Print detailed joint information for debugging."""
    robot = scene["Robot"]
    joint_names = robot.data.joint_names
    joint_pos = robot.data.joint_pos[0]

    # Base orientation in world frame
    base_quat = robot.data.root_quat_w[0]
    roll, pitch, yaw = _quat_to_euler_rpy(base_quat)

    print("=" * 100)
    print("G1 ROBOT STATE")
    print("=" * 100)
    print("Base orientation:")
    print(f"  Quaternion [w, x, y, z]: {[float(q) for q in base_quat]}")
    print(f"  Euler RPY [rad]: roll={roll:.4f}, pitch={pitch:.4f}, yaw={yaw:.4f}")
    # Orientation penalty from projected gravity
    penalty, g_b = _laying_down_orientation_l2_from_quat(base_quat, robot.device)
    print("")
    print("Laying-down orientation penalty (xy L2 of gravity in base):")
    print(f"  penalty={float(penalty):.6f}, projected_gravity_b=[{float(g_b[0]):.4f}, {float(g_b[1]):.4f}, {float(g_b[2]):.4f}]")
    align_reward = _axis_alignment_reward_from_projected_gravity(g_b, robot.device, axis=BELLY_TARGET_AXIS, sign=BELLY_TARGET_SIGN)
    print(f"  axis_alignment_reward={float(align_reward):.6f} (axis={BELLY_TARGET_AXIS}, sign={BELLY_TARGET_SIGN:+.1f})")

    # Print joints: current and target (if available)
    print(f"Total joints: {len(joint_names)}")
    print("\nJoint Index | Joint Name                   | Current    | Target")
    print("-" * 100)

    # Try to read last commanded targets: fall back to current if not present
    joint_target = None
    # Not all backends expose targets; we keep None-safe printing
    try:
        # Some implementations keep last command accessible via data.joint_pos_target
        joint_target = robot.data.joint_pos_target[0]
    except Exception:
        joint_target = None

    for i, name in enumerate(joint_names):
        curr = float(joint_pos[i])
        targ = float(joint_target[i]) if joint_target is not None else float('nan')
        if joint_target is None:
            print(f"{i:11d} | {name:30s} | {curr:10.4f} |    n/a")
        else:
            print(f"{i:11d} | {name:30s} | {curr:10.4f} | {targ:7.4f}")
    print("=" * 100)


def apply_debug_joint_values(scene):
    """Apply the specific joint values from the screenshot for debugging."""
    
    # Values from the screenshot - updated with precise values
    debug_joint_values = {
        # Hip joints
        "left_hip_pitch_joint": -1.22,
        "left_hip_roll_joint": 0.0827,
        "left_hip_yaw_joint": 0.165,
        "left_knee_joint": 1.06,
        "left_ankle_pitch_joint": 0.0526,
        "left_ankle_roll_joint": 0.0,
        
        "right_hip_pitch_joint": -0.515,
        "right_hip_roll_joint": 0.174,
        "right_hip_yaw_joint": 0.127,
        "right_knee_joint": 1.25,
        "right_ankle_pitch_joint": 0.7,
        "right_ankle_roll_joint": 0.0,
        
        # Waist
        "waist_yaw_joint": 0.385,
        
        # Shoulder and arm joints
        "left_shoulder_pitch_joint": -0.119,
        "left_shoulder_roll_joint": 0.0277,
        "left_shoulder_yaw_joint": -0.0999,
        "left_elbow_joint": 0.396,
        "left_wrist_roll_joint": 0.0348,
        
        "right_shoulder_pitch_joint": 0.0,
        "right_shoulder_roll_joint": -0.0986,
        "right_shoulder_yaw_joint": -0.101,
        "right_elbow_joint": 0.253,
        "right_wrist_roll_joint": -0.0347,
    }
    
    robot = scene["Robot"]
    joint_names = robot.data.joint_names
    
    # Create joint position tensor
    joint_positions = torch.zeros(1, len(joint_names), device=robot.device)
    
    # Apply debug values
    applied_joints = []
    missing_joints = []
    
    for i, joint_name in enumerate(joint_names):
        if joint_name in debug_joint_values:
            joint_positions[0, i] = debug_joint_values[joint_name]
            applied_joints.append((joint_name, debug_joint_values[joint_name]))
        else:
            # Keep default value
            joint_positions[0, i] = robot.data.default_joint_pos[0, i]
            missing_joints.append(joint_name)
    
    # Apply joint positions
    robot.set_joint_position_target(joint_positions)
    robot.write_data_to_sim()
    
    print("\n" + "=" * 80)
    print("APPLIED DEBUG JOINT VALUES")
    print("=" * 80)
    for joint_name, value in applied_joints:
        print(f"{joint_name:30s} = {value:8.4f}")
    
    if missing_joints:
        print(f"\nJoints using default values: {missing_joints}")
    
    print("=" * 80)
    # Print current orientation penalty
    base_quat = robot.data.root_quat_w[0]
    penalty, g_b = _laying_down_orientation_l2_from_quat(base_quat, robot.device)
    print(f"Orientation penalty after applying joints: {float(penalty):.6f}")
    print(f"projected_gravity_b: [{float(g_b[0]):.4f}, {float(g_b[1]):.4f}, {float(g_b[2]):.4f}]")
    align_reward = _axis_alignment_reward_from_projected_gravity(g_b, robot.device, axis=BELLY_TARGET_AXIS, sign=BELLY_TARGET_SIGN)
    print(f"axis_alignment_reward: {float(align_reward):.6f}")


def _euler_rpy_to_quat(roll, pitch, yaw, device):
    """Convert roll, pitch, yaw (radians) to quaternion [w, x, y, z]."""
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return torch.tensor([w, x, y, z], dtype=torch.float32, device=device)


def _quat_to_euler_rpy(quat):
    """Convert quaternion tensor/list [w, x, y, z] to roll, pitch, yaw (radians)."""
    # Ensure we have floats
    if hasattr(quat, "tolist"):
        w, x, y, z = quat.tolist()
    else:
        w, x, y, z = quat
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def _ensure_quat_tensor(quat, device):
    """Ensure quaternion is a 4D torch tensor [w, x, y, z] on given device."""
    if isinstance(quat, torch.Tensor):
        return quat.to(device=device, dtype=torch.float32)
    return torch.tensor(quat, dtype=torch.float32, device=device)


def _quat_to_rotation_matrix(quat, device):
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix (world->body aligns as R)."""
    q = _ensure_quat_tensor(quat, device)
    w, x, y, z = q.unbind(-1)
    # Rotation matrix corresponding to active rotation by q
    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y - z * w)
    r02 = 2 * (x * z + y * w)

    r10 = 2 * (x * y + z * w)
    r11 = 1 - 2 * (x * x + z * z)
    r12 = 2 * (y * z - x * w)

    r20 = 2 * (x * z - y * w)
    r21 = 2 * (y * z + x * w)
    r22 = 1 - 2 * (x * x + y * y)

    return torch.stack(
        [
            torch.stack([r00, r01, r02], dim=-1),
            torch.stack([r10, r11, r12], dim=-1),
            torch.stack([r20, r21, r22], dim=-1),
        ],
        dim=-2,
    )


def _rotate_vector_by_quat_inverse(quat, vec, device):
    """Rotate a 3D vector by the inverse of quaternion (i.e., world->body frame transform)."""
    R = _quat_to_rotation_matrix(quat, device)
    v = vec if isinstance(vec, torch.Tensor) else torch.tensor(vec, dtype=torch.float32, device=device)
    return torch.matmul(R.transpose(-2, -1), v)


def _laying_down_orientation_l2_from_quat(quat, device):
    """Compute sum of squared XY components of projected gravity in base frame.

    gravity (world) is assumed to be [0, 0, -1].
    Returns (penalty_tensor, projected_gravity_b_tensor).
    """
    g_w = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=device)
    g_b = _rotate_vector_by_quat_inverse(quat, g_w, device)
    penalty = g_b[0] * g_b[0] + g_b[1] * g_b[1]
    return penalty, g_b


def _axis_alignment_reward_from_projected_gravity(
    g_b: torch.Tensor,
    device,
    axis: str = 'x',
    sign: float = 1.0,
) -> torch.Tensor:
    """Reward aligning a chosen base axis with gravity in the base frame.

    - g_b: gravity in base frame (3,)
    - axis: which base axis to align with gravity: 'x'|'y'|'z'
    - sign: +1 to align +axis with gravity direction in base, -1 for -axis

    Returns reward in [-1, 1], where 1 is perfect alignment, -1 opposite, ~0 orthogonal.
    This is equivalent to 2*(axis_dot) - 1 after squaring distance mapping.
    """
    if axis == 'x':
        target = torch.tensor([float(sign), 0.0, 0.0], dtype=torch.float32, device=device)
    elif axis == 'y':
        target = torch.tensor([0.0, float(sign), 0.0], dtype=torch.float32, device=device)
    elif axis == 'z':
        target = torch.tensor([0.0, 0.0, float(sign)], dtype=torch.float32, device=device)
    else:
        raise ValueError("axis must be one of 'x', 'y', 'z'")
    dist_sq = torch.sum(torch.square(g_b - target))  # in [0, 4]
    return 1.0 - 0.5 * dist_sq  # maps 0->1, 2->0, 4->-1


def load_poses_from_json(file_path):
    """Load poses list from a JSON file with keys 'base_rpy' and 'joints'."""
    with open(file_path, "r") as f:
        data = json.load(f)
    poses = data.get("poses", [])
    if not isinstance(poses, list):
        raise ValueError("Invalid poses.json format: 'poses' must be a list")
    return poses


def apply_pose(scene: InteractiveScene, pose):
    """Apply a pose dict: set base orientation from base_rpy and joint targets.

    Pose schema:
      {
        "base_rpy": [roll, pitch, yaw],  # radians
        "joints": { joint_name: value, ... }
      }
    """
    robot = scene["Robot"]
    device = robot.device

    # Base pose: start from default, keep position, change orientation from RPY
    root_robot_state = robot.data.default_root_state.clone()
    root_robot_state[:, :3] += scene.env_origins

    base_rpy = pose.get("base_rpy", [0.0, 0.0, 0.0])
    if len(base_rpy) != 3:
        raise ValueError("Pose 'base_rpy' must be a list of length 3")
    roll, pitch, yaw = float(base_rpy[0]), float(base_rpy[1]), float(base_rpy[2])
    base_quat = _euler_rpy_to_quat(roll, pitch, yaw, device)
    root_robot_state[0, 3:7] = base_quat

    # Joints: start from defaults and override provided joints
    joint_names = robot.data.joint_names
    joint_positions = robot.data.default_joint_pos.clone()
    for i, joint_name in enumerate(joint_names):
        if "joints" in pose and joint_name in pose["joints"]:
            joint_positions[0, i] = float(pose["joints"][joint_name])

    # Apply to sim
    robot.write_root_pose_to_sim(root_robot_state[:, :7])
    robot.write_root_velocity_to_sim(root_robot_state[:, 7:])
    robot.set_joint_position_target(joint_positions)
    scene.write_data_to_sim()
    
    # Report orientation penalty for this pose
    base_quat_now = robot.data.root_quat_w[0]
    penalty, g_b = _laying_down_orientation_l2_from_quat(base_quat_now, device)
    print("\n[POSE] Orientation penalty:")
    print(f"  penalty={float(penalty):.6f}, projected_gravity_b=[{float(g_b[0]):.4f}, {float(g_b[1]):.4f}, {float(g_b[2]):.4f}]")
    align_reward = _axis_alignment_reward_from_projected_gravity(g_b, device, axis=BELLY_TARGET_AXIS, sign=BELLY_TARGET_SIGN)
    print(f"  axis_alignment_reward={float(align_reward):.6f}")


def compare_joint_orders():
    """Print comparison of different joint ordering systems for debugging."""
    print("\n" + "=" * 100)
    print("JOINT ORDER COMPARISON (for debugging mapping issues)")
    print("=" * 100)
    
    # Expected joint names in Isaac Lab order
    expected_isaac_joints = [
        "left_hip_pitch_joint",
        "right_hip_pitch_joint", 
        "waist_yaw_joint",
        "left_hip_roll_joint",
        "right_hip_roll_joint",
        "left_hip_yaw_joint",
        "right_hip_yaw_joint",
        "left_knee_joint",
        "right_knee_joint",
        "left_shoulder_pitch_joint",
        "right_shoulder_pitch_joint",
        "left_ankle_pitch_joint",
        "right_ankle_pitch_joint",
        "left_shoulder_roll_joint",
        "right_shoulder_roll_joint",
        "left_ankle_roll_joint",
        "right_ankle_roll_joint",
        "left_shoulder_yaw_joint",
        "right_shoulder_yaw_joint",
        "left_elbow_joint",
        "right_elbow_joint",
        "left_wrist_roll_joint",
        "right_wrist_roll_joint",
    ]
    
    print("Expected Isaac Lab Joint Order:")
    for i, joint in enumerate(expected_isaac_joints):
        print(f"{i:2d}: {joint}")
    
    print("=" * 100)


def run_debug_visualization(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Run the debug visualization with keyboard controls."""
    # Load poses
    # poses_path = "/home/logan/Projects/g1_crawl/scripts/experiments/poses.json"
    poses_path = "/home/logan/Projects/g1_crawl/scripts/experiments/gravity-test.json"
    try:
        poses = load_poses_from_json(poses_path)
        if len(poses) == 0:
            print(f"[WARN] No poses found in {poses_path}. Falling back to debug values.")
            poses = None
        else:
            print(f"[INFO] Loaded {len(poses)} poses from {poses_path}")
    except Exception as e:
        print(f"[ERROR] Failed to load poses from {poses_path}: {e}")
        poses = None

    # Print joint information
    get_joint_info(scene)
    compare_joint_orders()

    # Apply initial pose or fallback
    if poses is not None:
        apply_pose(scene, poses[0])
        current_pose_idx = 0
    else:
        apply_debug_joint_values(scene)
        current_pose_idx = -1
    
    # Set up keyboard input handling
    input_interface = carb.input.acquire_input_interface()
    keyboard = omni.appwindow.get_default_app_window().get_keyboard()
    
    keys_pressed = {
        "R": False,
        "D": False,
        "I": False,
        "N": False,
        "P": False,
        "ESCAPE": False,
    }
    
    def on_keyboard_event(event):
        nonlocal current_pose_idx
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == "R" and not keys_pressed["R"]:
                keys_pressed["R"] = True
                print("\n[DEBUG] Resetting to default pose...")
                # Reset to default robot state
                root_robot_state = scene["Robot"].data.default_root_state.clone()
                root_robot_state[:, :3] += scene.env_origins
                scene["Robot"].write_root_pose_to_sim(root_robot_state[:, :7])
                scene["Robot"].write_root_velocity_to_sim(root_robot_state[:, 7:])
                
                joint_pos = scene["Robot"].data.default_joint_pos.clone()
                joint_vel = scene["Robot"].data.default_joint_vel.clone()
                scene["Robot"].write_joint_state_to_sim(joint_pos, joint_vel)
                scene.write_data_to_sim()
                # Report orientation penalty after reset
                base_quat = scene["Robot"].data.root_quat_w[0]
                penalty, g_b = _laying_down_orientation_l2_from_quat(base_quat, scene["Robot"].device)
                print(f"[DEBUG] Orientation penalty after reset: {float(penalty):.6f}")
                print(f"        projected_gravity_b: [{float(g_b[0]):.4f}, {float(g_b[1]):.4f}, {float(g_b[2]):.4f}]")
                align_reward = _axis_alignment_reward_from_projected_gravity(g_b, scene["Robot"].device, axis=BELLY_TARGET_AXIS, sign=BELLY_TARGET_SIGN)
                print(f"        axis_alignment_reward: {float(align_reward):.6f}")
                
            elif event.input.name == "D" and not keys_pressed["D"]:
                keys_pressed["D"] = True
                print("\n[DEBUG] Applying debug joint values...")
                apply_debug_joint_values(scene)
                
            elif event.input.name == "I" and not keys_pressed["I"]:
                keys_pressed["I"] = True
                print("\n[DEBUG] Printing joint info...")
                get_joint_info(scene)
            elif event.input.name == "N" and not keys_pressed["N"]:
                keys_pressed["N"] = True
                if poses is not None and len(poses) > 0:
                    current_pose_idx = (current_pose_idx + 1) % len(poses)
                    apply_pose(scene, poses[current_pose_idx])
                    print(f"[POSE] Applied next pose {current_pose_idx+1}/{len(poses)}")
                else:
                    print("[WARN] No poses loaded; cannot switch to next pose.")
            elif event.input.name == "P" and not keys_pressed["P"]:
                keys_pressed["P"] = True
                if poses is not None and len(poses) > 0:
                    current_pose_idx = (current_pose_idx - 1) % len(poses)
                    apply_pose(scene, poses[current_pose_idx])
                    print(f"[POSE] Applied previous pose {current_pose_idx+1}/{len(poses)}")
                else:
                    print("[WARN] No poses loaded; cannot switch to previous pose.")
                
            elif event.input.name == "ESCAPE" and not keys_pressed["ESCAPE"]:
                keys_pressed["ESCAPE"] = True
                
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input.name in keys_pressed:
                keys_pressed[event.input.name] = False
    
    # Subscribe to keyboard events
    keyboard_subscription = input_interface.subscribe_to_keyboard_events(keyboard, on_keyboard_event)
    
    print(f"\n" + "=" * 80)
    print("G1 JOINT DEBUG VISUALIZATION")
    print("=" * 80)
    print("Controls:")
    print("  R - Reset to default pose")
    print("  D - Apply debug joint values from screenshot")
    print("  I - Print joint information")
    print("  N - Next pose from poses.json")
    print("  P - Previous pose from poses.json")
    print("  ESC - Exit")
    print("=" * 80)
    
    sim_dt = sim.get_physics_dt()
    
    try:
        while simulation_app.is_running():
            if keys_pressed["ESCAPE"]:
                print("Exiting debug visualization...")
                break
                
            sim.step()
            scene.update(sim_dt)
    
    finally:
        # Clean up keyboard subscription
        if keyboard_subscription:
            input_interface.unsubscribe_to_keyboard_events(keyboard, keyboard_subscription)


def main():
    """Main function."""
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(device="cuda")
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 1.0])
    
    # Create scene
    scene_cfg = DebugSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    
    # Reset simulation
    sim.reset()
    
    print("[INFO]: G1 Joint Debug Visualization starting...")
    
    # Run the debug visualization
    run_debug_visualization(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close() 