
from isaaclab.app import AppLauncher

# launch omniverse app
app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

from g1_23dof_rigor.tasks.manager_based.g1_23dof_rigor.g1 import G1_CFG

from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
import isaaclab.sim as sim_utils

import torch
import numpy as np
import os

import joblib

# Add carb and omni for keyboard input handling
import carb
import omni


class G1MjxJointIndex:
    """Joint indices based on the order in g1_mjx_alt.xml (23 DoF model)."""
    LeftHipPitch = 0
    LeftHipRoll = 1
    LeftHipYaw = 2
    LeftKnee = 3
    LeftAnklePitch = 4
    LeftAnkleRoll = 5
    RightHipPitch = 6
    RightHipRoll = 7
    RightHipYaw = 8
    RightKnee = 9
    RightAnklePitch = 10
    RightAnkleRoll = 11
    WaistYaw = 12
    LeftShoulderPitch = 13
    LeftShoulderRoll = 14
    LeftShoulderYaw = 15
    LeftElbow = 16
    LeftWristRoll = 17
    RightShoulderPitch = 18
    RightShoulderRoll = 19
    RightShoulderYaw = 20
    RightElbow = 21
    RightWristRoll = 22


class G1PyTorchJointIndex:
    """Joint indices based on the order in your PyTorch model."""
    LeftHipPitch = 0       # left_hip_pitch_joint
    RightHipPitch = 1      # right_hip_pitch_joint
    WaistYaw = 2           # waist_yaw_joint
    LeftHipRoll = 3        # left_hip_roll_joint
    RightHipRoll = 4       # right_hip_roll_joint
    LeftHipYaw = 5         # left_hip_yaw_joint
    RightHipYaw = 6        # right_hip_yaw_joint
    LeftKnee = 7           # left_knee_joint
    RightKnee = 8          # right_knee_joint
    LeftShoulderPitch = 9  # left_shoulder_pitch_joint
    RightShoulderPitch = 10 # right_shoulder_pitch_joint
    LeftAnklePitch = 11    # left_ankle_pitch_joint
    RightAnklePitch = 12   # right_ankle_pitch_joint
    LeftShoulderRoll = 13  # left_shoulder_roll_joint
    RightShoulderRoll = 14 # right_shoulder_roll_joint
    LeftAnkleRoll = 15     # left_ankle_roll_joint
    RightAnkleRoll = 16    # right_ankle_roll_joint
    LeftShoulderYaw = 17   # left_shoulder_yaw_joint
    RightShoulderYaw = 18  # right_shoulder_yaw_joint
    LeftElbow = 19         # left_elbow_joint
    RightElbow = 20        # right_elbow_joint
    LeftWristRoll = 21     # left_wrist_roll_joint
    RightWristRoll = 22    # right_wrist_roll_joint


# Mapping from PyTorch model joint order to MuJoCo joint order
pytorch2mujoco_idx = [
    # PyTorch idx -> MuJoCo idx
    G1MjxJointIndex.LeftHipPitch,      # 0: left_hip_pitch_joint -> LeftHipPitch (0)
    G1MjxJointIndex.RightHipPitch,     # 1: right_hip_pitch_joint -> RightHipPitch (6)
    G1MjxJointIndex.WaistYaw,          # 2: waist_yaw_joint -> WaistYaw (12)
    G1MjxJointIndex.LeftHipRoll,       # 3: left_hip_roll_joint -> LeftHipRoll (1)
    G1MjxJointIndex.RightHipRoll,      # 4: right_hip_roll_joint -> RightHipRoll (7)
    G1MjxJointIndex.LeftHipYaw,        # 5: left_hip_yaw_joint -> LeftHipYaw (2)
    G1MjxJointIndex.RightHipYaw,       # 6: right_hip_yaw_joint -> RightHipYaw (8)
    G1MjxJointIndex.LeftKnee,          # 7: left_knee_joint -> LeftKnee (3)
    G1MjxJointIndex.RightKnee,         # 8: right_knee_joint -> RightKnee (9)
    G1MjxJointIndex.LeftShoulderPitch, # 9: left_shoulder_pitch_joint -> LeftShoulderPitch (13)
    G1MjxJointIndex.RightShoulderPitch,# 10: right_shoulder_pitch_joint -> RightShoulderPitch (18)
    G1MjxJointIndex.LeftAnklePitch,    # 11: left_ankle_pitch_joint -> LeftAnklePitch (4)
    G1MjxJointIndex.RightAnklePitch,   # 12: right_ankle_pitch_joint -> RightAnklePitch (10)
    G1MjxJointIndex.LeftShoulderRoll,  # 13: left_shoulder_roll_joint -> LeftShoulderRoll (14)
    G1MjxJointIndex.RightShoulderRoll, # 14: right_shoulder_roll_joint -> RightShoulderRoll (19)
    G1MjxJointIndex.LeftAnkleRoll,     # 15: left_ankle_roll_joint -> LeftAnkleRoll (5)
    G1MjxJointIndex.RightAnkleRoll,    # 16: right_ankle_roll_joint -> RightAnkleRoll (11)
    G1MjxJointIndex.LeftShoulderYaw,   # 17: left_shoulder_yaw_joint -> LeftShoulderYaw (15)
    G1MjxJointIndex.RightShoulderYaw,  # 18: right_shoulder_yaw_joint -> RightShoulderYaw (20)
    G1MjxJointIndex.LeftElbow,         # 19: left_elbow_joint -> LeftElbow (16)
    G1MjxJointIndex.RightElbow,        # 20: right_elbow_joint -> RightElbow (21)
    G1MjxJointIndex.LeftWristRoll,     # 21: left_wrist_roll_joint -> LeftWristRoll (17)
    G1MjxJointIndex.RightWristRoll,    # 22: right_wrist_roll_joint -> RightWristRoll (22)
]

# Create inverse mapping: MuJoCo idx -> PyTorch idx
mujoco2pytorch_idx = [0] * 23  # Initialize with zeros
for pytorch_idx, mujoco_idx in enumerate(pytorch2mujoco_idx):
    mujoco2pytorch_idx[mujoco_idx] = pytorch_idx


def load_mocap_data(device):
    """Load mocap data from pkl file and create joint mapping."""
    # pkl_path = "data/amass_all.pkl"
    pkl_path = "data/0-motion.pkl"
    
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"PKL file not found: {pkl_path}")
    
    print(f"Loading mocap data from: {pkl_path}")
    motion_data = joblib.load(pkl_path)
    
    motion_keys = list(motion_data.keys())
    print(f"Found {len(motion_keys)} animations in PKL file")
    print(f"Animation keys: {motion_keys[:5]}{'...' if len(motion_keys) > 5 else ''}")
    
    # Get info from first animation
    first_motion = motion_data[motion_keys[0]]
    print(f"First animation '{motion_keys[0]}' has {first_motion['dof'].shape[0]} timesteps")
    print(f"DOF shape: {first_motion['dof'].shape}")
    print(f"Root trans offset shape: {first_motion['root_trans_offset'].shape}")
    print(f"Root rot shape: {first_motion['root_rot'].shape}")
    
    # Check SMPL joints if available
    num_smpl_joints = 0
    if 'smpl_joints' in first_motion:
        print(f"SMPL joints shape: {first_motion['smpl_joints'].shape}")
        num_smpl_joints = first_motion['smpl_joints'].shape[1]
        print(f"Number of SMPL joints: {num_smpl_joints}")
    
    # Convert motion data to GPU tensors
    processed_motion_data = {}
    for key, motion in motion_data.items():
        processed_motion_data[key] = {
            'root_trans_offset': torch.from_numpy(motion['root_trans_offset']).float().to(device),
            'root_rot': torch.from_numpy(motion['root_rot']).float().to(device),
            'dof': torch.from_numpy(motion['dof']).float().to(device),
            'smpl_joints': torch.from_numpy(motion['smpl_joints']).float().to(device) if 'smpl_joints' in motion else None
        }
    
    print(f"Converted motion data to torch tensors on {device}")
    
    return processed_motion_data, motion_keys, num_smpl_joints


def create_smpl_spheres_scene_cfg(num_joints):
    """Create a scene configuration with the appropriate number of SMPL joint spheres."""
    
    class DynamicTestSceneCfg(InteractiveSceneCfg):
        """Designs the scene with dynamic number of SMPL spheres."""

        # Ground-plane
        ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

        # lights
        dome_light = AssetBaseCfg(
            prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
        )

        # robot - keep fix_root_link=False so we can see the mocap movement
        Robot = G1_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=G1_CFG.spawn.replace(
                articulation_props=G1_CFG.spawn.articulation_props.replace(
                    fix_root_link=False
                )
            )
        )
    
    # Add individual SMPL joint spheres
    for i in range(num_joints):
        sphere_cfg = RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/SmplSphere_{i:02d}",
            spawn=sim_utils.SphereCfg(
                radius=0.015,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),  # Make kinematic (no physics)
                mass_props=sim_utils.MassPropertiesCfg(mass=0.001),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),  # Disable collisions
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 0.0, 0.0),  # No diffuse
                    emissive_color=(1.0, 0.2, 0.2),  # Bright red emissive (unshaded)
                    metallic=0.0,
                    roughness=1.0,
                    opacity=1.0  # Fully opaque
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -10.0), rot=(1.0, 0.0, 0.0, 0.0)),  # Start below ground
        )
        setattr(DynamicTestSceneCfg, f"smpl_sphere_{i:02d}", sphere_cfg)
    
    return DynamicTestSceneCfg


def scene_reset(scene: InteractiveScene, motion_data=None, motion_keys=None, motion_id=0):
    """Reset the scene to initial state, optionally using first mocap frame."""
    if motion_data is not None and motion_keys is not None:
        # Use first frame of current motion for reset
        curr_motion_key = motion_keys[motion_id]
        curr_motion = motion_data[curr_motion_key]
        apply_mocap_frame(scene, curr_motion, 0)
        print(f"Reset to motion '{curr_motion_key}' frame 0")
    else:
        # Fall back to default robot state
        root_robot_state = scene["Robot"].data.default_root_state.clone()
        root_robot_state[:, :3] += scene.env_origins

        # copy the default root state to the sim
        scene["Robot"].write_root_pose_to_sim(root_robot_state[:, :7])
        scene["Robot"].write_root_velocity_to_sim(root_robot_state[:, 7:])

        # copy the default joint states to the sim
        joint_pos, joint_vel = (
            scene["Robot"].data.default_joint_pos.clone(),
            scene["Robot"].data.default_joint_vel.clone(),
        )
        scene["Robot"].write_joint_state_to_sim(joint_pos, joint_vel)
        
        # Hide SMPL spheres by moving them below ground when no mocap data
        for asset_name in scene._assets:
            if asset_name.startswith("smpl_sphere_"):
                try:
                    sphere_root_state = torch.zeros(1, 13, device=scene[asset_name].device)
                    sphere_root_state[0, :3] = torch.tensor([0.0, 0.0, -10.0], device=scene[asset_name].device)  # Below ground
                    sphere_root_state[0, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=scene[asset_name].device)
                    sphere_root_state[0, 7:13] = 0.0
                    scene[asset_name].write_root_pose_to_sim(sphere_root_state[:, :7])
                    scene[asset_name].write_root_velocity_to_sim(sphere_root_state[:, 7:])
                except KeyError:
                    # Sphere doesn't exist, skip it
                    continue
        
        print("Reset to default robot state")
    
    scene.reset()


def apply_mocap_frame(scene: InteractiveScene, motion, frame_idx):
    """Apply a single mocap frame to the robot and update SMPL joint visualization."""
    # Extract base pose and create base velocity (set to zero for now)
    base_pos = motion['root_trans_offset'][frame_idx]  # [x, y, z]
    
    # Apply inverse of MuJoCo reordering since we're going PKL→Isaac, not PKL→MuJoCo
    # MuJoCo uses [3, 0, 1, 2], so inverse is [1, 2, 3, 0]
    root_rot_pkl = motion['root_rot'][frame_idx]
    base_quat = torch.tensor(root_rot_pkl[[3, 0, 1, 2]], device=scene["Robot"].device)
    
    # Create root state tensor
    new_root_state = torch.zeros(1, 13, device=scene["Robot"].device)
    new_root_state[0, :3] = base_pos
    new_root_state[0, 2] += 0.2  # add .2 to z
    new_root_state[0, 3:7] = base_quat
    new_root_state[0, 7:13] = 0.0  # Set base velocities to zero for now
    
    # Extract joint positions and map from MuJoCo order to PyTorch order
    mujoco_dof = motion['dof'][frame_idx]  # Joint positions in MuJoCo order
    
    joint_positions = torch.zeros(1, 23, device=scene["Robot"].device)
    
    for mujoco_idx, pytorch_idx in enumerate(mujoco2pytorch_idx):
        joint_positions[0, pytorch_idx] = mujoco_dof[mujoco_idx]
    
    # Set joint velocities to zero for now
    joint_velocities = torch.zeros(1, 23, device=scene["Robot"].device)
    
    # Apply to robot
    scene["Robot"].write_root_pose_to_sim(new_root_state[:, :7])
    scene["Robot"].write_root_velocity_to_sim(new_root_state[:, 7:])
    scene["Robot"].set_joint_position_target(joint_positions)
    scene["Robot"].set_joint_velocity_target(joint_velocities)
    
    # Update SMPL joint visualization spheres
    if motion['smpl_joints'] is not None:
        smpl_joint_positions = motion['smpl_joints'][frame_idx]  # Shape: [num_joints, 3]
        
        # Update each sphere position
        for i in range(smpl_joint_positions.shape[0]):
            sphere_name = f"smpl_sphere_{i:02d}"
            try:
                # Create root state for the sphere
                sphere_root_state = torch.zeros(1, 13, device=scene[sphere_name].device)
                sphere_root_state[0, :3] = smpl_joint_positions[i]  # xyz position
                sphere_root_state[0, 2] += 0.2  # add .2 to z like the robot
                sphere_root_state[0, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=scene[sphere_name].device)  # identity quaternion
                sphere_root_state[0, 7:13] = 0.0  # zero velocity
                
                # Apply to sphere
                scene[sphere_name].write_root_pose_to_sim(sphere_root_state[:, :7])
                scene[sphere_name].write_root_velocity_to_sim(sphere_root_state[:, 7:])
            except KeyError:
                # Sphere doesn't exist, skip it
                continue
    
    scene.write_data_to_sim()


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Run the simulator with mocap playback."""
    # Load mocap data
    try:
        motion_data, motion_keys, num_smpl_joints = load_mocap_data(scene["Robot"].device)
        print(f"Loaded motion data with {num_smpl_joints} SMPL joints")
    except Exception as e:
        print(f"Error loading mocap data: {e}")
        print("Falling back to default animation...")
        # Fall back to the original random animation
        run_default_animation(sim, scene)
        return
    
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    
    # Animation state
    motion_id = 0
    frame_idx = 0
    last_mocap_time = 0.0
    mocap_dt = 1.0 / 30.0  # 30 FPS playback
    paused = False
    
    # Reset to first motion
    scene_reset(scene, motion_data, motion_keys, motion_id)
    
    # Set up keyboard input handling
    input_interface = carb.input.acquire_input_interface()
    keyboard = omni.appwindow.get_default_app_window().get_keyboard()
    
    # Keyboard state tracking
    keys_pressed = {
        "R": False,
        "SPACE": False,
        "N": False,
        "P": False,
        "ESCAPE": False,
        "LEFT": False,
        "RIGHT": False,
        "Q": False,
        "E": False,
        "D": False
    }
    
    manual_stepping = False  # Track if we're in manual stepping mode
    
    def on_keyboard_event(event):
        nonlocal paused, frame_idx, last_mocap_time, motion_id, manual_stepping
        
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == "R" and not keys_pressed["R"]:
                keys_pressed["R"] = True
                frame_idx = 0
                last_mocap_time = sim_time
                manual_stepping = False
                scene_reset(scene, motion_data, motion_keys, motion_id)
                print("Animation reset!")
            elif event.input.name == "SPACE" and not keys_pressed["SPACE"]:
                keys_pressed["SPACE"] = True
                paused = not paused
                manual_stepping = False
                print(f"Animation {'paused' if paused else 'resumed'}")
            elif event.input.name == "N" and not keys_pressed["N"]:
                keys_pressed["N"] = True
                motion_id = (motion_id + 1) % len(motion_keys)
                frame_idx = 0
                last_mocap_time = sim_time
                manual_stepping = False
                scene_reset(scene, motion_data, motion_keys, motion_id)
                print(f"Switched to next animation: '{motion_keys[motion_id]}'")
            elif event.input.name == "P" and not keys_pressed["P"]:
                keys_pressed["P"] = True
                motion_id = (motion_id - 1) % len(motion_keys)
                frame_idx = 0
                last_mocap_time = sim_time
                manual_stepping = False
                scene_reset(scene, motion_data, motion_keys, motion_id)
                print(f"Switched to previous animation: '{motion_keys[motion_id]}'")
            elif event.input.name == "LEFT" and not keys_pressed["LEFT"]:
                keys_pressed["LEFT"] = True
                manual_stepping = True
                paused = True
                curr_motion_key = motion_keys[motion_id]
                curr_motion = motion_data[curr_motion_key]
                max_frames = curr_motion['dof'].shape[0]
                
                # Step backward 10 frames
                frame_idx = max(0, frame_idx - 10)
                apply_mocap_frame(scene, curr_motion, frame_idx)
                print(f"[MANUAL STEPPING] Stepped back 10 frames to frame {frame_idx}/{max_frames-1}")
            elif event.input.name == "RIGHT" and not keys_pressed["RIGHT"]:
                keys_pressed["RIGHT"] = True
                manual_stepping = True
                paused = True
                curr_motion_key = motion_keys[motion_id]
                curr_motion = motion_data[curr_motion_key]
                max_frames = curr_motion['dof'].shape[0]
                
                # Step forward 10 frames
                frame_idx = min(max_frames - 1, frame_idx + 10)
                apply_mocap_frame(scene, curr_motion, frame_idx)
                print(f"[MANUAL STEPPING] Stepped forward 10 frames to frame {frame_idx}/{max_frames-1}")
            elif event.input.name == "Q" and not keys_pressed["Q"]:
                keys_pressed["Q"] = True
                manual_stepping = True
                paused = True
                curr_motion_key = motion_keys[motion_id]
                curr_motion = motion_data[curr_motion_key]
                max_frames = curr_motion['dof'].shape[0]
                
                # Step backward 1 frame
                frame_idx = max(0, frame_idx - 1)
                apply_mocap_frame(scene, curr_motion, frame_idx)
                print(f"[MANUAL STEPPING] Stepped back 1 frame to frame {frame_idx}/{max_frames-1}")
            elif event.input.name == "E" and not keys_pressed["E"]:
                keys_pressed["E"] = True
                manual_stepping = True
                paused = True
                curr_motion_key = motion_keys[motion_id]
                curr_motion = motion_data[curr_motion_key]
                max_frames = curr_motion['dof'].shape[0]
                
                # Step forward 1 frame
                frame_idx = min(max_frames - 1, frame_idx + 1)
                apply_mocap_frame(scene, curr_motion, frame_idx)
                print(f"[MANUAL STEPPING] Stepped forward 1 frame to frame {frame_idx}/{max_frames-1}")
            elif event.input.name == "D" and not keys_pressed["D"]:
                keys_pressed["D"] = True
                curr_motion_key = motion_keys[motion_id]
                curr_motion = motion_data[curr_motion_key]
                max_frames = curr_motion['dof'].shape[0]
                current_dof = curr_motion['dof'][frame_idx]
                
                print(f"[DEBUG INFO]")
                print(f"  Current motion: '{curr_motion_key}'")
                print(f"  Frame: {frame_idx}/{max_frames-1}")
                print(f"  DOF values: {current_dof.cpu().numpy()}")
                print(f"  DOF shape: {current_dof.shape}")
            elif event.input.name == "ESCAPE" and not keys_pressed["ESCAPE"]:
                keys_pressed["ESCAPE"] = True
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input.name in keys_pressed:
                keys_pressed[event.input.name] = False
    
    # Subscribe to keyboard events
    keyboard_subscription = input_interface.subscribe_to_keyboard_events(keyboard, on_keyboard_event)
    
    print(f"Starting mocap playback...")
    print(f"Simulation dt: {sim_dt:.4f}s, Mocap dt: {mocap_dt:.4f}s")
    print(f"Current animation: '{motion_keys[motion_id]}'")
    print(f"Controls:")
    print(f"  R - Reset current animation to beginning")
    print(f"  SPACE - Pause/Resume animation (exits manual stepping mode)")
    print(f"  N - Next animation")
    print(f"  P - Previous animation")
    print(f"  LEFT ARROW - Step backward 10 frames (auto-pauses)")
    print(f"  RIGHT ARROW - Step forward 10 frames (auto-pauses)")
    print(f"  Q - Step backward 1 frame (auto-pauses)")
    print(f"  E - Step forward 1 frame (auto-pauses)")
    print(f"  D - Print current frame index and DOF values")
    print(f"  ESC - Exit")
    print(f"Press Ctrl+C or close window to stop")
    
    try:
        while simulation_app.is_running():
            # Check for exit condition
            if keys_pressed["ESCAPE"]:
                print("Exiting...")
                break
            
            # Get current motion data
            curr_motion_key = motion_keys[motion_id]
            curr_motion = motion_data[curr_motion_key]
            
            # Always apply current frame to prevent robot from falling over
            if frame_idx < curr_motion['dof'].shape[0]:
                apply_mocap_frame(scene, curr_motion, frame_idx)
            
            # Advance frame only if not paused and not in manual stepping mode
            if not paused and not manual_stepping and sim_time - last_mocap_time >= mocap_dt:
                if frame_idx < curr_motion['dof'].shape[0] - 1:
                    frame_idx += 1
                    last_mocap_time = sim_time
                else:
                    # Loop the animation
                    frame_idx = 0
                    last_mocap_time = sim_time
                    print(f"Looping animation '{curr_motion_key}'...")
            
            sim.step()
            sim_time += sim_dt
            scene.update(sim_dt)
    
    finally:
        # Clean up keyboard subscription
        if keyboard_subscription:
            input_interface.unsubscribe_to_keyboard_events(keyboard, keyboard_subscription)


def run_default_animation(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Fallback to original random animation if mocap loading fails."""
    sim_dt = sim.get_physics_dt()
    scene_reset(scene)  # Use default reset without mocap data
    count = 0
    
    while simulation_app.is_running():
        count += 1
        
        if count % 100 == 0:
            # Apply random joint movements
            lower = scene["Robot"].data.joint_pos_limits[0, :, 0]
            upper = scene["Robot"].data.joint_pos_limits[0, :, 1]

            mean = (upper + lower) / 2
            std = (upper - lower) / 6
            random_pos = torch.normal(mean, std)
            random_pos = torch.clamp(random_pos, lower, upper)

            scene["Robot"].set_joint_position_target(random_pos)
            scene.write_data_to_sim()
            
        sim.step()
        scene.update(sim_dt)


def main():
    """Main function."""
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(device="cuda")
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])
    
    # Determine number of SMPL joints
    num_smpl_joints = 0
    try:
        # Attempt to load mocap data to determine num_smpl_joints
        _, _, num_smpl_joints = load_mocap_data(sim.device)
        print(f"Detected {num_smpl_joints} SMPL joints in mocap data")
    except Exception as e:
        print(f"Error loading mocap data to determine SMPL joints: {e}")
        print("Falling back to default SMPL joint count (24).")
        num_smpl_joints = 24  # Common SMPL joint count
    
    # Create scene configuration with dynamic spheres
    scene_cfg_class = create_smpl_spheres_scene_cfg(num_smpl_joints)
    scene_cfg = scene_cfg_class(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    
    # Play the simulator
    sim.reset()
    
    # Now we are ready!
    print("[INFO]: Setup complete...")
    print("[INFO]: Loading and playing mocap animation...")
    
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()