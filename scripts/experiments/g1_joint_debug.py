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

# Add carb and omni for keyboard input handling
import carb
import omni


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
    joint_pos = robot.data.joint_pos[0]  # First environment
    
    print("=" * 80)
    print("G1 ROBOT JOINT INFORMATION")
    print("=" * 80)
    print(f"Total joints: {len(joint_names)}")
    print("\nJoint Index | Joint Name | Current Position")
    print("-" * 60)
    for i, (name, pos) in enumerate(zip(joint_names, joint_pos)):
        print(f"{i:11d} | {name:30s} | {pos:8.4f}")
    print("=" * 80)


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
    
    # Print joint information
    get_joint_info(scene)
    compare_joint_orders()
    
    # Apply debug joint values
    apply_debug_joint_values(scene)
    
    # Set up keyboard input handling
    input_interface = carb.input.acquire_input_interface()
    keyboard = omni.appwindow.get_default_app_window().get_keyboard()
    
    keys_pressed = {
        "R": False,
        "D": False,
        "I": False,
        "ESCAPE": False,
    }
    
    def on_keyboard_event(event):
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
                
            elif event.input.name == "D" and not keys_pressed["D"]:
                keys_pressed["D"] = True
                print("\n[DEBUG] Applying debug joint values...")
                apply_debug_joint_values(scene)
                
            elif event.input.name == "I" and not keys_pressed["I"]:
                keys_pressed["I"] = True
                print("\n[DEBUG] Printing joint info...")
                get_joint_info(scene)
                
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