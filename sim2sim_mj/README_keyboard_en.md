# G1 Crawl Keyboard Control Guide (English)

## Overview

`run_sim2sim_keyboard.py` is a MuJoCo simulation deployment script with keyboard control support. The robot starts in a standing state (policy mode) and can be controlled in real-time via keyboard.

### Demo Animation

![Robot Crawling Demo](docs/images/robot_crawling_demo.gif)

*Note: The above animation shows the robot performing crawling motion in the simulation environment using keyboard control*

---

## 🎯 Enhanced Features in This Version

This version adds complete **keyboard control functionality** to the original project, enabling robot control via keyboard without requiring a gamepad.

### Main Contributions

1. **Complete Keyboard Control Implementation**
   - Implemented cross-platform keyboard input handling using `pynput` library
   - Supports all original gamepad functions including movement control, mode switching, policy switching, etc.

2. **Keyboard Mapping Design**
   - **Arrow Keys**: Intuitive forward/backward and lateral movement control
   - **IJK Keys**: Mode switching corresponding to original D-PAD functions
   - **Z/C Keys**: Rotation control
   - **SPACE Key**: Policy cycling
   - **Q/A/H Keys**: Real-time gain adjustment

3. **Full Feature Parity with Original Project**
   - Completely preserves all original control logic and safety features
   - Supports all control modes (default_pos, crawl_pos, damped, policy)
   - Retains gain adjustment functionality with real-time PD gain modification
   - Maintains complete safety monitoring system

4. **User Experience Improvements**
   - Robot starts in standing state, ready for immediate control
   - Clear keyboard control instructions
   - Comprehensive Chinese and English documentation

### Technical Implementation

- Uses `pynput` library for cross-platform keyboard listening (no root required on Linux)
- Complete key state management and edge detection
- State synchronization for mode switching and policy switching
- Seamless integration with original code architecture

---

## Feature Comparison: Keyboard vs Gamepad

This version provides keyboard control as an alternative to gamepad while preserving all original project functionality.

| Feature | Gamepad Control (run_sim2sim.py) | Keyboard Control (run_sim2sim_keyboard.py) |
|---------|----------------------------------|--------------------------------------------|
| **Movement Control** | Left stick Y/X, Right stick X | Arrow keys ↑↓←→, Z/C rotation |
| **Mode Switching** | D-PAD direction keys | IJK keys (I=stand, J=damped, K=crawl) |
| **Policy Switching** | A button cycles | SPACE key cycles |
| **Gain Adjustment** | Keyboard Q/A/H | Keyboard Q/A/H (same) |
| **System Start** | Requires START button | Pre-enabled (policy mode) |
| **Exit Program** | SELECT button | ESC key |
| **Cross-platform** | Requires GLFW + gamepad | Uses pynput (no root on Linux) |
| **Dependencies** | glfw + gamepad | pynput (lighter) |
| **Use Case** | Users with gamepad | Users without gamepad |

> **Note**: Keyboard and gamepad controls are functionally equivalent, with all control logic and safety features maintained consistently.

---

## Installation

```bash
conda activate g1-crawl
pip install -r requirements.txt
```

## Running the Program

```bash
cd sim2sim_mj
conda activate g1-crawl
python run_sim2sim_keyboard.py
```

## Keyboard Controls

### Quick Reference Table

| Key | Function | Active Mode | Description |
|-----|----------|-------------|-------------|
| **↑** | Forward | policy | Up arrow |
| **↓** | Backward | policy | Down arrow |
| **←** | Strafe left | policy | Left arrow |
| **→** | Strafe right | policy | Right arrow |
| **Z** | Rotate left (CCW) | policy | Rotation control |
| **C** | Rotate right (CW) | policy | Rotation control |
| **I** | Default Position (stand) | All | Corresponds to D-PAD UP |
| **J** | Damped mode (safe stop) | All | Corresponds to D-PAD LEFT |
| **K** | Crawl Position mode | All | Corresponds to D-PAD DOWN |
| **SPACE** | Cycle policies | policy | Corresponds to A button |
| **Q** | Increase gain | All | Step +0.1 |
| **A** | Decrease gain | All | Step -0.1 |
| **H** | Print current gain | All | Prints to terminal |
| **ESC** | Exit program | All | Immediate exit |

> **Note**: The robot starts in **policy mode** (standing control) by default, no key press needed to begin control.

### Detailed Instructions

#### Movement Control (Arrow Keys + Z/C)

In policy mode, use arrow keys to control robot movement:

- **↑ (Up Arrow)**: Forward
- **↓ (Down Arrow)**: Backward
- **← (Left Arrow)**: Strafe left
- **→ (Right Arrow)**: Strafe right
- **Z**: Rotate left (counterclockwise)
- **C**: Rotate right (clockwise)

> **Note**: Movement control is only active in policy mode.

#### Mode Switching (IJK Keys)

Use IJK keys to switch between different control modes (corresponding to D-PAD functions in the original project):

- **I**: Default Position Mode
  - Robot switches to standing pose
  - Smooth transition to preset standing joint angles

- **J**: Damped Mode
  - Safe stop mode
  - All joints enter high damping state, stopping movement
  - Instant switch, no transition time

- **K**: Crawl Position Mode
  - Robot switches to crawling pose
  - Smooth transition to preset crawling joint angles

#### Policy Control

- **SPACE (Spacebar)**: Cycle through loaded policies
  - Only active in policy mode
  - Pressing space cycles through the policy list
  - Default policy order:
    1. `policy_shamble.pt` (Standing/walking policy)
    2. `policy_crawl_start.pt` (Crawl start policy)
    3. `policy_crawl.pt` (Crawling policy)

#### Gain Adjustment

Gain adjustment is available in all modes:

- **Q**: Increase gain multiplier (step: 0.1)
- **A**: Decrease gain multiplier (step: 0.1)
- **H**: Print current gain multiplier

The gain multiplier affects PD gains (KP and KD) for all joints, influencing robot motion rigidity and response speed.

#### Exit Program

- **ESC**: Exit simulation program

## Configuration Options

In the `CONFIG` section of `run_sim2sim_keyboard.py`, you can modify the following settings:

### Basic Configuration

- `model_xml`: MuJoCo scene file path (default: "scene.xml")
- `policy_path`: Initial policy file (default: "policies/policy_shamble.pt")
- `action_scale`: Global action scale multiplier (default: 0.5)
- `n_substeps`: Number of simulation substeps per policy step (default: 4)

### Velocity Limits

- `max_lin_vel`: Maximum forward/backward velocity (m/s, default: 2.0)
- `max_lat_vel`: Maximum lateral velocity (m/s, default: 1.0)
- `max_ang_vel`: Maximum angular velocity (rad/s, default: 1.0)

### Policy Configuration

- `policy_cycle`: Policy cycling list
- `policy_defaults`: Mapping from policies to default joint sets ("stand" or "crawl")
- `gain_step`: Gain adjustment step size (default: 0.1)

### Camera Configuration

- `camera_distance`: Camera distance (meters, default: 3.0)
- `camera_azimuth`: Camera azimuth angle (degrees, default: 90)
- `camera_elevation`: Camera elevation angle (degrees, default: -20)
- `camera_tracking`: Whether to track robot position (default: True)

## Usage Examples

### Basic Operation Flow

1. **Start the program**
   ```bash
   python run_sim2sim_keyboard.py
   ```
   - After startup, the robot starts in standing state (policy mode)
   - MuJoCo viewer window opens automatically

2. **Control robot movement**
   - Use arrow keys to control forward/backward and lateral movement
   - Use Z/C to control rotation
   - Velocity limits are determined by current policy configuration

3. **Switch policies**
   - Press SPACE to cycle through policies
   - Each policy has different velocity limits and default poses

4. **Switch modes**
   - Press I to switch to standing pose
   - Press K to switch to crawling pose
   - Press J to enter safe stop mode

5. **Adjust gains**
   - Press Q to increase gain (more rigid)
   - Press A to decrease gain (more compliant)
   - Press H to view current gain value

6. **Exit program**
   - Press ESC to exit

### Complete Flow: Standing to Crawling

1. Start program (robot starts in standing state)
2. Use arrow keys and Z/C to control robot movement (standing mode)
3. Press SPACE to switch to `policy_crawl_start.pt`
4. Press K to switch to crawl position mode
5. Wait for transition to complete (a few seconds)
6. Press SPACE to switch to `policy_crawl.pt`
7. Use arrow keys to control crawling movement

## Safety Features

The program includes safety monitoring that detects and warns about the following anomalies:

- **Position Jump Detection**: Detects abnormal joint position jumps (threshold: 15.0 rad/s)
- **Velocity Spike Detection**: Detects joint velocities exceeding limits (threshold: 25.0 rad/s)
- **Acceleration Spike Detection**: Detects joint accelerations exceeding limits (threshold: 1500 rad/s²)

> **Note**: In simulation, these warnings do not stop the program, they only alert. On a real robot, these violations could trigger safety protection.

## Troubleshooting

### Robot Not Responding to Controls

- Ensure the MuJoCo viewer window has focus
- Check if in policy mode (initial state is policy mode)
- Verify policy files are loaded

### Window Cannot Be Dragged

- Ensure `suppress=True` is not enabled (disabled by default)
- Try clicking the window title bar before dragging

### Key Conflicts

- MuJoCo viewer shortcuts (e.g., SPACE for play/pause) may conflict with program controls
- When using controls, keep focus on the viewer window

## Technical Specifications

- **Control Frequency**: 50 Hz (20 ms period)
- **Policy Update Frequency**: Updated every 4 simulation steps (configurable via `n_substeps`)
- **Number of Joints**: 23
- **Policy Input Dimension**: 75 (projected gravity 3 + velocity commands 3 + joint positions 23 + joint velocities 23 + last actions 23)
- **Policy Output Dimension**: 23 (joint target positions)

## File Structure

```
sim2sim_mj/
├── run_sim2sim_keyboard.py  # Main program (keyboard control version)
├── run_sim2sim.py            # Original program (gamepad control)
├── scene.xml                 # MuJoCo scene file
├── policies/                 # Policy files directory
│   ├── policy_shamble.pt
│   ├── policy_crawl_start.pt
│   └── policy_crawl.pt
├── poses/                    # Initial pose configurations
│   ├── default-pose.json
│   └── crawl-pose.json
└── requirements.txt          # Dependencies list
```

## Demo Materials

Demo screenshots and videos are located in the `docs/` directory:

```
sim2sim_mj/
└── docs/
    ├── images/
    │   └── robot_crawling_demo.gif  # Robot crawling demo animation
    └── videos/
        └── keyboard_control_demo.mp4  # Keyboard control demo video (optional)
```

## License

Please refer to the LICENSE file in the project root directory.

## Acknowledgments

This keyboard control functionality is based on the original project's gamepad control implementation. While preserving all original features, it provides a keyboard control solution for users without gamepads.

---

*Last updated: 2026.1*
