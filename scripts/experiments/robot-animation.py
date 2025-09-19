
from isaaclab.app import AppLauncher

# launch omniverse app
app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

from g1_crawl.tasks.manager_based.g1_crawl.g1 import G1_CFG, get_animation, build_joint_index_map

from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import AssetBaseCfg
import isaaclab.sim as sim_utils

import torch

# Add carb and omni for keyboard input handling
import carb
import omni
from pxr import Usd, Sdf
ALLOW_ROOT_WRITES = True
def create_scene_cfg():
    """Create a simple scene with ground, light, and the robot (free root)."""
    class SimpleSceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
        dome_light = AssetBaseCfg(
            prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
        )
        Robot = G1_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=G1_CFG.spawn.replace(
                rigid_props=G1_CFG.spawn.rigid_props.replace(
                    disable_gravity=True
                ),
                articulation_props=G1_CFG.spawn.articulation_props.replace(
                    fix_root_link=False
                )
            )
        )
    return SimpleSceneCfg


def scene_reset(scene: InteractiveScene, anim=None, joint_index_map=None):
    """Reset the scene to initial state, optionally using first animation frame."""
    if anim is not None:
        apply_animation_frame(scene, anim, 0, joint_index_map)
        print("Reset to animation frame 0")
    else:
        root_robot_state = scene["Robot"].data.default_root_state.clone()
        root_robot_state[:, :3] += scene.env_origins
        scene["Robot"].write_root_pose_to_sim(root_robot_state[:, :7])
        scene["Robot"].write_root_velocity_to_sim(root_robot_state[:, 7:])
        joint_pos, joint_vel = (
            scene["Robot"].data.default_joint_pos.clone(),
            scene["Robot"].data.default_joint_vel.clone(),
        )
        scene["Robot"].write_joint_state_to_sim(joint_pos, joint_vel)
        print("Reset to default robot state")
    scene.reset()
def apply_animation_frame(scene: InteractiveScene, anim, frame_idx, joint_index_map=None):
    """Apply one animation frame from the JSON to the robot."""
    qpos_row = anim["qpos"][frame_idx]
    device = scene["Robot"].device

    # Root pose
    base_meta = anim["base_meta"]
    if base_meta is not None:
        pos_idx = base_meta.get("pos_indices", None)
        quat_idx = base_meta.get("quat_indices", None)
        if pos_idx is not None and quat_idx is not None:
            base_pos = qpos_row[pos_idx]
            wxyz = qpos_row[quat_idx]
            # Normalize quaternion to avoid backend rejection
            qwxyz_norm = torch.linalg.norm(wxyz)
            if qwxyz_norm > 0:
                wxyz = wxyz / qwxyz_norm
            new_root_state = torch.zeros(1, 13, device=device)
            origin = scene.env_origins.to(device=device)[0] if hasattr(scene, 'env_origins') else torch.zeros(3, device=device)
            new_root_state[0, :3] = base_pos + origin
            new_root_state[0, 3:7] = wxyz
            new_root_state[0, 7:13] = 0.0
            scene["Robot"].write_root_pose_to_sim(new_root_state[:, :7])
            scene["Robot"].write_root_velocity_to_sim(new_root_state[:, 7:])

    # Joint DOF targets
    num_robot_dofs = scene["Robot"].data.default_joint_pos.shape[1]
    joint_positions = scene["Robot"].data.default_joint_pos.clone()
    joint_velocities = torch.zeros_like(joint_positions)

    if joint_index_map is None:
        joint_index_map = list(range(min(num_robot_dofs, anim["nq"])))

    for j_idx in range(num_robot_dofs):
        qidx = joint_index_map[j_idx] if j_idx < len(joint_index_map) else -1
        if isinstance(qidx, (int,)) and qidx >= 0 and qidx < anim["nq"]:
            joint_positions[0, j_idx] = qpos_row[qidx]

    # Write states directly to sim (no PD targets) for physics-free playback
    scene["Robot"].write_joint_state_to_sim(joint_positions, joint_velocities)
    scene.write_data_to_sim()



def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Run the simulator with JSON animation playback."""
    # Load JSON animation via shared helper (fails loudly if invalid)
    anim = get_animation()

    sim_dt = sim.get_physics_dt()
    print(f"Simulation dt: {sim_dt:.4f}s")
    sim_time = 0.0

    # Build joint index map using shared helper (pass articulation asset)
    joint_index_map = build_joint_index_map(scene["Robot"], anim["joints_meta"], anim.get("qpos_labels"))

    # Animation state
    frame_idx = 0
    last_frame_time = 0.0
    anim_dt = float(anim["dt"]) if anim["dt"] else 1.0 / 30.0
    paused = False

    # Reset to first frame
    scene_reset(scene, anim, joint_index_map)

    
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
        nonlocal paused, frame_idx, last_frame_time, manual_stepping
        
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == "R" and not keys_pressed["R"]:
                keys_pressed["R"] = True
                frame_idx = 0
                last_frame_time = sim_time
                manual_stepping = False
                scene_reset(scene, anim, joint_index_map)
                print("Animation reset!")
            elif event.input.name == "SPACE" and not keys_pressed["SPACE"]:
                keys_pressed["SPACE"] = True
                paused = not paused
                manual_stepping = False
                print(f"Animation {'paused' if paused else 'resumed'}")
            # N/P controls removed (single animation file)
            elif event.input.name == "LEFT" and not keys_pressed["LEFT"]:
                keys_pressed["LEFT"] = True
                manual_stepping = True
                paused = True
                max_frames = anim["num_frames"]
                
                # Step backward 10 frames
                frame_idx = max(0, frame_idx - 10)
                apply_animation_frame(scene, anim, frame_idx, joint_index_map)
                print(f"[MANUAL STEPPING] Stepped back 10 frames to frame {frame_idx}/{max_frames-1}")
            elif event.input.name == "RIGHT" and not keys_pressed["RIGHT"]:
                keys_pressed["RIGHT"] = True
                manual_stepping = True
                paused = True
                max_frames = anim["num_frames"]
                
                # Step forward 10 frames
                frame_idx = min(max_frames - 1, frame_idx + 10)
                apply_animation_frame(scene, anim, frame_idx, joint_index_map)
                print(f"[MANUAL STEPPING] Stepped forward 10 frames to frame {frame_idx}/{max_frames-1}")
            elif event.input.name == "Q" and not keys_pressed["Q"]:
                keys_pressed["Q"] = True
                manual_stepping = True
                paused = True
                max_frames = anim["num_frames"]
                
                # Step backward 1 frame
                frame_idx = max(0, frame_idx - 1)
                apply_animation_frame(scene, anim, frame_idx, joint_index_map)
                print(f"[MANUAL STEPPING] Stepped back 1 frame to frame {frame_idx}/{max_frames-1}")
            elif event.input.name == "E" and not keys_pressed["E"]:
                keys_pressed["E"] = True
                manual_stepping = True
                paused = True
                max_frames = anim["num_frames"]
                
                # Step forward 1 frame
                frame_idx = min(max_frames - 1, frame_idx + 1)
                apply_animation_frame(scene, anim, frame_idx, joint_index_map)
                print(f"[MANUAL STEPPING] Stepped forward 1 frame to frame {frame_idx}/{max_frames-1}")
            elif event.input.name == "D" and not keys_pressed["D"]:
                keys_pressed["D"] = True
                max_frames = anim["num_frames"]
                current_qpos = anim["qpos"][frame_idx]
                
                print(f"[DEBUG INFO]")
                print(f"  Frame: {frame_idx}/{max_frames-1}")
                print(f"  qpos values: {current_qpos.cpu().numpy()}")
                print(f"  qpos shape: {current_qpos.shape}")
            elif event.input.name == "ESCAPE" and not keys_pressed["ESCAPE"]:
                keys_pressed["ESCAPE"] = True
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input.name in keys_pressed:
                keys_pressed[event.input.name] = False
    
    # Subscribe to keyboard events
    keyboard_subscription = input_interface.subscribe_to_keyboard_events(keyboard, on_keyboard_event)
    
    print(f"Starting animation playback...")
    print(f"Simulation dt: {sim_dt:.4f}s, Animation dt: {anim_dt:.4f}s")
    print(f"Controls:")
    print(f"  R - Reset current animation to beginning")
    print(f"  SPACE - Pause/Resume animation (exits manual stepping mode)")
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
            
            # Always apply current frame to keep articulation stable
            if frame_idx < anim["num_frames"]:
                apply_animation_frame(scene, anim, frame_idx, joint_index_map)
            
            # Advance frame only if not paused and not in manual stepping mode
            if not paused and not manual_stepping and sim_time - last_frame_time >= anim_dt:
                if frame_idx < anim["num_frames"] - 1:
                    frame_idx += 1
                    last_frame_time = sim_time
                else:
                    frame_idx = 0
                    last_frame_time = sim_time
                    print("Looping animation...")
            
            sim.step()
            sim_time += sim_dt
            scene.update(sim_dt)
    
    finally:
        # Clean up keyboard subscription
        if keyboard_subscription:
            input_interface.unsubscribe_to_keyboard_events(keyboard, keyboard_subscription)


def run_default_animation(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    raise RuntimeError("Default animation fallback has been disabled. Fix the animation JSON and retry.")


def main():
    """Main function."""
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(device="cuda")
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])

    # Create scene configuration
    scene_cfg_class = create_scene_cfg()
    scene_cfg = scene_cfg_class(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    
    # Play the simulator
    sim.reset()
    
    # Now we are ready!
    print("[INFO]: Setup complete...")
    print("[INFO]: Loading and playing JSON animation...")
    
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()