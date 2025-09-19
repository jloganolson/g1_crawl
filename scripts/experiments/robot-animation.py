
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

# Debug draw (3D overlays) availability
try:
    import isaacsim.util.debug_draw._debug_draw as omni_debug_draw  # type: ignore
    _DEBUG_DRAW_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    omni_debug_draw = None  # type: ignore
    _DEBUG_DRAW_AVAILABLE = False
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
    
    # ===== Contact flags 3D visualization (feet/hands) =====
    def _require_debug_draw():
        if not _DEBUG_DRAW_AVAILABLE:
            raise RuntimeError("Debug draw is not available. Ensure visualization mode and debug_draw extension are enabled.")

    def _find_first_match(name_list, patterns):
        lname = [str(n).lower() for n in name_list]
        for p in patterns:
            pl = p.lower()
            for i, n in enumerate(lname):
                if pl in n:
                    return i
        return None

    def _find_limb_body_indices(asset):
        names = asset.data.body_names
        name_to_idx = {str(n): i for i, n in enumerate(names)}

        # Prefer exact known names from env cfg
        preferred = {
            "left_foot": "left_ankle_roll_link",
            "right_foot": "right_ankle_roll_link",
            "left_hand": "left_wrist_link",
            "right_hand": "right_wrist_link",
        }

        resolved: dict[str, int | None] = {k: None for k in preferred.keys()}
        used_fallback: dict[str, bool] = {k: False for k in preferred.keys()}

        for limb, exact_name in preferred.items():
            if exact_name in name_to_idx:
                resolved[limb] = int(name_to_idx[exact_name])
                print(f"[contacts] {limb} -> '{exact_name}' (index {resolved[limb]}) [exact]")
            else:
                # Heuristic substring fallback
                patterns = []
                if limb == "left_foot":
                    patterns = ["left_ankle", "left_foot", "l_ankle", "l_foot", "left_toe", "left_sole"]
                elif limb == "right_foot":
                    patterns = ["right_ankle", "right_foot", "r_ankle", "r_foot", "right_toe", "right_sole"]
                elif limb == "left_hand":
                    patterns = ["left_wrist", "left_hand", "l_wrist", "l_hand", "left_palm"]
                elif limb == "right_hand":
                    patterns = ["right_wrist", "right_hand", "r_wrist", "r_hand", "right_palm"]
                idx = _find_first_match(names, patterns)
                if idx is not None:
                    used_fallback[limb] = True
                    resolved[limb] = int(idx)
                    print(f"[contacts] {limb} -> '{names[int(idx)]}' (index {idx}) [fallback-substr]")

        missing = [k for k, v in resolved.items() if v is None]
        if missing:
            print(f"[contacts] Available bodies ({len(names)}): {list(names)}")
            raise RuntimeError(f"Could not locate bodies for: {missing}. Prefer exact names from env cfg.")

        return {k: int(v) for k, v in resolved.items() if v is not None}

    def _resolve_contact_label_indices(contact_order):
        # Normalize
        order_raw = list(contact_order)
        order = [str(x).strip().lower() for x in order_raw]

        # First prefer canonical FL/FR/RL/RR
        try_map: dict[str, list[str]] = {
            "left_hand": ["fl", "front_left"],
            "right_hand": ["fr", "front_right"],
            "left_foot": ["rl", "rear_left", "back_left"],
            "right_foot": ["rr", "rear_right", "back_right"],
        }

        indices: dict[str, int | None] = {k: None for k in try_map.keys()}
        for limb, keys in try_map.items():
            for key in keys:
                for i, s in enumerate(order):
                    if s == key:
                        indices[limb] = i
                        break
                if indices[limb] is not None:
                    break

        # Fallback: broader synonyms inside strings
        def find_fuzzy(candidates):
            for i, s in enumerate(order):
                for c in candidates:
                    if c in s:
                        return i
            return None

        if indices["left_hand"] is None:
            indices["left_hand"] = find_fuzzy(["left_hand", "l_hand", "left_wrist", "l_wrist"])
        if indices["right_hand"] is None:
            indices["right_hand"] = find_fuzzy(["right_hand", "r_hand", "right_wrist", "r_wrist"])
        if indices["left_foot"] is None:
            indices["left_foot"] = find_fuzzy(["left_foot", "l_foot", "left_toe", "l_toe", "left_ankle"])
        if indices["right_foot"] is None:
            indices["right_foot"] = find_fuzzy(["right_foot", "r_foot", "right_toe", "r_toe", "right_ankle"])

        missing = [k for k, v in indices.items() if v is None]
        if missing:
            raise RuntimeError(
                f"Animation contact_order missing required entries for {missing}. Found: {order_raw}"
            )

        print(
            "[contacts] contact_order mapping:",
            {
                "left_hand": order_raw[int(indices["left_hand"])],
                "right_hand": order_raw[int(indices["right_hand"])],
                "left_foot": order_raw[int(indices["left_foot"])],
                "right_foot": order_raw[int(indices["right_foot"])],
            },
        )
        return {k: int(v) for k, v in indices.items() if v is not None}

    # Prepare mapping once
    _require_debug_draw()
    contact_order = anim.get("contact_order", None)
    if contact_order is None:
        raise RuntimeError("Animation JSON missing 'metadata.contact.order' for contact visualization")
    label_indices = _resolve_contact_label_indices(contact_order)
    body_indices = _find_limb_body_indices(scene["Robot"])
    print(
        "[contacts] body mapping:",
        {
            k: {
                "name": str(scene["Robot"].data.body_names[v]),
                "index": int(v),
            }
            for k, v in body_indices.items()
        },
    )

    draw_interface = omni_debug_draw.acquire_debug_draw_interface()

    def draw_contact_flags(frame_index: int):
        flags = anim.get("contact_flags", None)
        if flags is None:
            raise RuntimeError("Animation JSON missing 'contact_flags' for contact visualization")
        # Build points and colors
        pts = []
        colors = []
        sizes = []
        # Fetch current world positions (single env index 0)
        body_pos_w = scene["Robot"].data.body_pos_w[0]
        z_offset = 0.03
        for limb_name in ["left_hand", "right_hand", "left_foot", "right_foot"]:
            bidx = body_indices[limb_name]
            lidx = label_indices[limb_name]
            p = body_pos_w[bidx]
            contact_val = float(flags[frame_index, lidx].item())
            color = (0.1, 0.9, 0.1, 1.0) if contact_val >= 0.5 else (0.9, 0.1, 0.1, 1.0)
            pts.append((float(p[0].item()), float(p[1].item()), float(p[2].item() + z_offset)))
            colors.append(color)
            sizes.append(16)
        # Clear and draw
        draw_interface.clear_points()
        draw_interface.draw_points(pts, colors, sizes)

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
                # Draw contact flags at current frame
                draw_contact_flags(frame_idx)
            
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