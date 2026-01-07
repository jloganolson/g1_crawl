## G1 Crawl (Isaac Lab)

Note: I put the final policies in sim2sim_mj/policies but I dont recall the exact reward terms that resulted in them (especially the final crawl), so you might have to do some sweeps to recreate the results seen in video.

### Prerequisites & Setup
- **Isaac Lab environment**: Follow the Isaac Lab documentation to install and create a conda environment. Use that conda env for this repo.
- This project assumes a conda env named `g1-crawl`. Always run commands like: `conda activate g1-crawl && python ...`.
- Optional (for sweep video post-processing): ensure `ffmpeg` is installed and on PATH.

### Train a Policy
- Minimal headless training:
```bash
conda activate g1-crawl && \
python scripts/rsl_rl/train.py --task g1-crawl --headless
```
- Common overrides (examples):
```bash
conda activate g1-crawl && \
python scripts/rsl_rl/train.py --task g1-crawl --headless \
  agent.max_iterations=2500
```
- Logs and checkpoints are written under `logs/rsl_rl/<experiment_name>/<timestamp_UUID>/`.

### Play a Trained Policy
- Play latest checkpoint from a run directory:
```bash
conda activate g1-crawl && \
python scripts/rsl_rl/play.py --task g1-crawl --headless --video --video_length 200 --enable_cameras \
  agent.experiment_name=<your_experiment> \
  agent.load_run=<timestamp_UUID>
```
- Play a specific checkpoint by absolute path:
```bash
conda activate g1-crawl && \
python scripts/rsl_rl/play.py --task g1-crawl --headless --video --video_length 200 --enable_cameras \
  --checkpoint /abs/path/to/model_XXXX.pt
```

### Run a Parameter Sweep (`run_sweep.py`)
1. Open `run_sweep.py` and edit the first section:
   - `SWEEP_QUEUE`: add/adjust entries and parameters to sweep (use lists for values).
   - Use the special string `"__OMIT__"` to test “not setting” a parameter.
   - Set `AUTO_SUSPEND=False` if you do not want the machine to sleep after the sweep.
2. Run the sweep:
```bash
conda activate g1-crawl && \
python run_sweep.py
```
3. What it does:
   - Trains for each combination, then auto-plays and saves videos.
   - Writes training logs to `logs/rsl_rl/<experiment_name>/...`.
   - Writes sweep logs and concatenated videos to `sweep-logs/<experiment_name>_<timestamp>/`.
   - If `ffmpeg` is available, also creates a labeled concatenated video and a segment mapping.

Notes:
- `resume_checkpoint` (when included in a sweep combo) is passed as a CLI flag automatically.
- The script fails loudly on missing run directories or file I/O errors so issues are visible immediately.

### Sim2Sim Standalone Playback (`sim2sim_mj/run_sim2sim.py`)
This runs exported policies in standard MuJoCo (no Isaac Lab runtime needed during playback).

1. Dependencies (within the same conda env):
```bash
conda activate g1-crawl && \
pip install -r sim2sim_mj/requirements.txt
```
2. Assets:
   - Ensure `sim2sim_mj/scene.xml` exists (export your model if needed).
   - Place policy files under `sim2sim_mj/policies/` and update the `CONFIG` in `sim2sim_mj/run_sim2sim.py`:
     - `policy_cycle`: list the `.pt` files you want to cycle through.
     - `policy_defaults`: map each policy path to `"stand"` or `"crawl"` for the default joint set.
3. Run:
```bash
conda activate g1-crawl && \
python sim2sim_mj/run_sim2sim.py
```
4. Controls:
   - A single face button cycles policies (see `CONFIG.cycle_button`).
   - Start/Menu exits (see `CONFIG.exit_button`).
   - Analog sticks provide velocity commands if a standard-mapped gamepad is connected.
   - Optional CSV diagnostics are written under `outputs/diagnostics/` when enabled.

#### Keyboard Control (Optional)
For keyboard-based control without a gamepad, see [KEYBOARD_SIM2SIM.md](KEYBOARD_SIM2SIM.md).

This feature provides an alternative input method using standard keyboard keys (arrow keys, IJK, Z/C, SPACE, Q/A/H, ESC) for controlling the robot in MuJoCo simulation. It's particularly useful for laptops, remote machines, and development setups where a physical gamepad is unavailable.

### Real Robot Deployment (`deployment/`)
- The `deployment/` folder contains scripts/configs intended for running policies on the physical robot.
- These expect robot-side connectivity, drivers, and safety layers (e.g., actuator PD settings, watchdogs).
- Adjust gains, safety thresholds, and device interfaces there to match your hardware before use.
- Use with caution; validate in simulation first.

### Common Paths
- Training logs/checkpoints: `logs/rsl_rl/<experiment_name>/<timestamp_UUID>/`
- Sweep outputs (guides/videos/mapping): `sweep-logs/<experiment_name>_<timestamp>/`
- Sim2Sim policies: `sim2sim_mj/policies/`
- Sim2Sim diagnostics: `outputs/diagnostics/`

### 3D Printed Parts (Onshape links)
- [Head parts (requires heat-set inserts for screws)](https://cad.onshape.com/documents/51f2b1c723880a644b5e4295/w/26b9ae2d16dd6818ee2cba50/e/2c03d03c2d68d17d48cc747c?renderMode=0&uiState=690ba6558c75f7c496f162c5)
- [Custom arms (requires a squash ball)](https://cad.onshape.com/documents/be995b1470263c9f303ec0a3/w/a8114830271f7d8f2d8df84b/e/bdf2f54b49628d7c3c8cd944?renderMode=0&uiState=690ba6ab9ccb00f927e7e3db)
- Corresponding 3MF print files live in `3d-printed-parts/`.