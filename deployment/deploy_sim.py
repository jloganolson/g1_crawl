# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

#modified by Logan
"""Deploy a PyTorch policy to C MuJoCo and play with it."""

import mujoco
import mujoco.viewer as viewer
import numpy as np
import torch
from keyboard_reader import KeyboardController
from utils import (
    default_angles_config,

    init_joint_mappings,
    remap_pytorch_to_mujoco,
    remap_mujoco_to_pytorch,
)

POLICY_PATH = "policy.pt"


class TorchController:
  """PyTorch controller for the Go-1 robot."""

  def __init__(
      self,
      policy_path: str,
      default_angles: np.ndarray,
      n_substeps: int,
      action_scale: float = 0.5,
      vel_scale_x: float = 1.0,
      vel_scale_y: float = 1.0,
      vel_scale_rot: float = 1.0,
  ):
    self._policy = torch.load(policy_path, weights_only=False)
    self._policy.eval()  # Set to evaluation mode

    self._action_scale = action_scale
    self._default_angles = default_angles
    self._last_action = np.zeros_like(default_angles, dtype=np.float32)  # In MuJoCo order

    self._counter = 0
    self._n_substeps = n_substeps

    self._controller = KeyboardController(
        vel_scale_x=vel_scale_x,
        vel_scale_y=vel_scale_y,
        vel_scale_rot=vel_scale_rot,
    )

    # Initialize joint mappings
    init_joint_mappings()


  def get_obs(self, model, data) -> np.ndarray:
    # Simplified observation: 75 dimensions total
    # projected_gravity (3) + velocity_commands (3) + joint_pos (23) + joint_vel (23) + actions (23)
    
    # Get projected gravity (3 dimensions)
    # imu_xmat = data.site_xmat[model.site("imu_in_pelvis").id].reshape(3, 3)
    # projected_gravity = imu_xmat.T @ np.array([0, 0, -1])
    world_gravity = model.opt.gravity
    world_gravity = world_gravity / np.linalg.norm(world_gravity)  # Normalize
    imu_xmat = data.site_xmat[model.site("imu_in_pelvis").id].reshape(3, 3)
    projected_gravity = imu_xmat.T @ world_gravity
    # Get velocity commands (3 dimensions)
    velocity_commands = self._controller.get_command()
    # velocity_commands = np.array([0.25, 0.0, 0.0])  # Forward velocity command
    
    # Get joint positions and velocities in MuJoCo order, then convert to PyTorch order
    joint_pos_mujoco = data.qpos[7:] - self._default_angles
    joint_vel_mujoco = data.qvel[6:]
    
    # Convert to PyTorch model joint order for the observation
    joint_pos_pytorch = remap_mujoco_to_pytorch(joint_pos_mujoco)
    joint_vel_pytorch = remap_mujoco_to_pytorch(joint_vel_mujoco)
    
    # Last action should also be in PyTorch order for the observation
    # Convert the MuJoCo-ordered last action to PyTorch order
    actions_pytorch = remap_mujoco_to_pytorch(self._last_action)
    
    # Concatenate all observations: 3 + 3 + 23 + 23 + 23 = 75
    obs = np.hstack([
        projected_gravity,
        velocity_commands, 
        joint_pos_pytorch,
        joint_vel_pytorch,
        actions_pytorch,
    ])
    
    return obs.astype(np.float32)

  def get_control(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
    self._counter += 1
    if self._counter % self._n_substeps == 0:
      obs = self.get_obs(model, data)
      
      # Convert to torch tensor and run inference
      obs_tensor = torch.from_numpy(obs).float()
      
      with torch.no_grad():
        action_tensor = self._policy(obs_tensor)
        pytorch_pred = action_tensor.numpy()  # Actions in PyTorch model joint order

      # Zero out arm control similar to training logic (in PyTorch order)
      # In PyTorch order, arm joints are:
      # 9-10: shoulders, 13-14: shoulder rolls, 17-18: shoulder yaws, 19-20: elbows, 21-22: wrists
      # ZERO_ARM_CONTROL = True  # Set this flag as needed
      # if ZERO_ARM_CONTROL:
      #   # Arm joint indices in PyTorch order
      #   arm_indices = [9, 10, 13, 14, 17, 18, 19, 20, 21, 22]  # All arm joints
      #   pytorch_pred[arm_indices] = 0.0

      # Convert actions from PyTorch order to MuJoCo order
      mujoco_pred = remap_pytorch_to_mujoco(pytorch_pred)

      self._last_action = mujoco_pred.copy()  # Store in MuJoCo order
      # data.ctrl[:] =  self._default_angles

      data.ctrl[:] = mujoco_pred * self._action_scale + self._default_angles


def load_callback(model=None, data=None):
  mujoco.set_mjcb_control(None)


  model = mujoco.MjModel.from_xml_path('./g1_description/scene_mjx_alt.xml')
  data = mujoco.MjData(model)

  mujoco.mj_resetDataKeyframe(model, data, 1)

  # ctrl_dt = 0.02
  sim_dt = 0.005
  n_substeps = 4
  model.opt.timestep = sim_dt

  policy = TorchController(
      policy_path=POLICY_PATH,
      default_angles=np.array(default_angles_config),
      n_substeps=n_substeps,
      action_scale=0.5,
      vel_scale_x=1.0,
      vel_scale_y=1.0,
      vel_scale_rot=1.0,
  )

  mujoco.set_mjcb_control(policy.get_control)

  return model, data


if __name__ == "__main__":
  viewer.launch(loader=load_callback)
