# Keyboard-based Sim2Sim Control (MuJoCo)

This repository provides an **optional keyboard-based controller** for the MuJoCo sim2sim pipeline, enabling interactive control **without requiring an external gamepad**.

This feature is designed for laptops, remote machines, and development environments where a physical controller is unavailable.

---

## Highlights

- ⌨️ **Keyboard-only control** for MuJoCo sim2sim
- 🔄 **Fully optional** — original gamepad workflow remains unchanged
- 🌍 **Cross-platform** (Linux / Windows)
- 📦 **Scoped to `sim2sim_mj`** — no impact on training or core logic
- 🛡️ **Full feature parity** with the original gamepad controller

---

## What's Added

- `run_sim2sim_keyboard.py`: standalone keyboard-controlled sim2sim entry
- Cross-platform keyboard input via `pynput` (no root required)
- Full support for:
  - Movement control
  - Mode switching (stand / damped / crawl / policy)
  - Policy cycling
  - Runtime gain adjustment
  - Existing safety & monitoring logic
- Bilingual documentation (EN / 中文)
- Demo animation

---

## Quick Start (Keyboard)

```bash
conda activate g1-crawl
pip install -r sim2sim_mj/requirements.txt
python sim2sim_mj/run_sim2sim_keyboard.py
```

---

## Default Key Bindings

| Action | Keys |
|--------|------|
| Forward / Backward | ↑ / ↓ |
| Strafe Left / Right | ← / → |
| Rotate Left / Right | Z / C |
| Mode Switch | I (stand) / J (damped) / K (crawl) |
| Policy Switch | SPACE |
| Gain Adjust | Q / A |
| Print Gains | H |
| Exit | ESC |

---

## Documentation

- 📘 [Keyboard Control (English)](sim2sim_mj/README_keyboard_en.md)
- 📙 [键盘控制（中文）](sim2sim_mj/README_keyboard_zh.md)
- ▶️ [Demo: sim2sim_mj/docs/images/robot_crawling_demo.gif](sim2sim_mj/docs/images/robot_crawling_demo.gif)

---

## Note

Keyboard input uses `pynput` and requires an active desktop session (X11 / Wayland on Linux).

For fully headless setups, please continue using the original gamepad-based workflow.
