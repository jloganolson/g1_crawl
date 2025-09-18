
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

class TestSceneCfg(InteractiveSceneCfg):
    """Designs the scene."""

    # Ground-plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # robot - programmatically set fix_root_link=True
    Robot = G1_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=G1_CFG.spawn.replace(
            articulation_props=G1_CFG.spawn.articulation_props.replace(
                fix_root_link=True
            )
        )
    )


def scene_reset(scene: InteractiveScene):
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
    scene.reset()


def print_joint_name_index(scene: InteractiveScene):
    """Print a labeled mapping of joint index -> joint name to CLI."""
    robot = scene["Robot"]
    joint_names = robot.data.joint_names
    print("\n=== Robot joints (index -> name) ===")
    for i, name in enumerate(joint_names):
        print(f"[{i:02d}] {str(name)}")
    print(f"Total joints: {len(joint_names)}")


def test_root_movement(scene: InteractiveScene):
    """Test if we can move the robot's root when fix_root_link=True"""
    print("\n=== Testing Root Movement with fix_root_link=True ===")
    
    # Get current root state
    current_pose = scene["Robot"].data.root_pos_w[0].clone()
    current_quat = scene["Robot"].data.root_quat_w[0].clone()
    print(f"Initial root position: {current_pose}")
    print(f"Initial root orientation: {current_quat}")
    
    # Try to move the robot up by 1 meter
    new_pose = current_pose.clone()
    new_pose[2] += 1.0  # Move up 1 meter in Z direction
    
    # Create new root state
    new_root_state = torch.zeros(1, 13, device=scene["Robot"].device)
    new_root_state[0, :3] = new_pose
    new_root_state[0, 3:7] = current_quat
    # Keep velocities at zero
    
    print(f"Attempting to move robot to: {new_pose}")
    
    # Try to write the new pose
    scene["Robot"].write_root_pose_to_sim(new_root_state[:, :7])
    scene["Robot"].write_root_velocity_to_sim(new_root_state[:, 7:])
    
    return new_pose


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    scene_reset(scene)
    count = 0
    
    # Test root movement initially
    target_pos = test_root_movement(scene)
    
    while simulation_app.is_running():
        count += 1
        
        # Every 100 steps, try different root movements
        if count % 100 == 0:
            # Test different types of movement
            current_pose = scene["Robot"].data.root_pos_w[0].clone()
            # print(f"Step {count}: Current robot position: {current_pose}")
            
            # if count % 500 == 0:
            #     # Try to make the robot float up and down
            #     height_offset = 1.0 + 0.5 * torch.sin(torch.tensor(sim_time * 2.0))  # Oscillate between 0.5 and 1.5 meters
            #     new_pose = scene["Robot"].data.default_root_state[0, :3].clone()
            #     new_pose[2] += height_offset
                
            #     new_root_state = torch.zeros(1, 13, device=scene["Robot"].device)
            #     new_root_state[0, :3] = new_pose
            #     new_root_state[0, 3:7] = scene["Robot"].data.root_quat_w[0]
                
            #     print(f"Trying to move robot to height: {new_pose[2]:.2f}")
            #     scene["Robot"].write_root_pose_to_sim(new_root_state[:, :7])
            #     scene["Robot"].write_root_velocity_to_sim(new_root_state[:, 7:])
            
            # Also continue with random joint movements
            lower = scene["Robot"].data.joint_pos_limits[0, :, 0]
            upper = scene["Robot"].data.joint_pos_limits[0, :, 1]

            mean = (upper + lower) / 2
            std = (upper - lower) / 6  # Using 6 sigma to keep most values within bounds
            random_pos = torch.normal(mean, std)
            random_pos = torch.clamp(random_pos, lower, upper)

            # Apply random actions to the robot
            scene["Robot"].set_joint_position_target(random_pos)
            scene.write_data_to_sim()
            
        sim.step()
        sim_time += sim_dt
        scene.update(sim_dt)


def main():
    """Main function."""
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(device="cuda")
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])
    
    # design scene
    scene_cfg = TestSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    
    # Play the simulator
    sim.reset()
    
    # Print joint index/name mapping once at startup
    print_joint_name_index(scene)
    
    # Now we are ready!
    print("[INFO]: Setup complete...")
    print("[INFO]: Testing fix_root_link=True behavior...")
    print("[INFO]: Watch the robot - if fix_root_link works correctly,")
    print("[INFO]: the robot should NOT move when we try to change its root position")
    
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()