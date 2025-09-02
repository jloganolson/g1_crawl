from utils import (
    init_cmd_hg,
    create_damping_cmd,
    create_zero_cmd,
    MotorMode,
    NonBlockingInput,
    joint2motor_idx,
    Kp,
    Kd,
    G1_NUM_MOTOR,
    default_pos,
    default_angles_config,
    get_gravity_orientation,
    action_scale,
    RESTRICTED_JOINT_RANGE,
    G1MjxJointIndex,
    G1PyTorchJointIndex,
    pytorch2mujoco_idx,
    mujoco2pytorch_idx,
    init_joint_mappings,
    remap_pytorch_to_mujoco,
    remap_mujoco_to_pytorch,
)
from unitree_sdk2py.utils.thread import RecurrentThread
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
import time
import numpy as np
import torch
from keyboard_reader import KeyboardController

NETWORK_CARD_NAME = 'enxc8a362b43bfd'
POLICY_PATH = "policy.pt"



class TorchPolicy:
    """PyTorch controller for the G1 robot."""

    def __init__(self, policy_path: str):
        self._policy = torch.load(policy_path, weights_only=False)
        self._policy.eval()  # Set to evaluation mode
        
        # Initialize joint mappings
        init_joint_mappings()

    def get_control(self, obs: np.ndarray) -> np.ndarray:
        """Get control actions from PyTorch policy."""
        # Convert to torch tensor and run inference
        obs_tensor = torch.from_numpy(obs).float()
        
        with torch.no_grad():
            action_tensor = self._policy(obs_tensor)
            pytorch_pred = action_tensor.numpy()  # Actions in PyTorch model joint order

        # Zero out arm control similar to training logic (in PyTorch order)
        # In PyTorch order, arm joints are:
        # 9-10: shoulders, 13-14: shoulder rolls, 17-18: shoulder yaws, 19-20: elbows, 21-22: wrists
        ZERO_ARM_CONTROL = False  # Set this flag as needed
        if ZERO_ARM_CONTROL:
            # Arm joint indices in PyTorch order
            arm_indices = [9, 10, 13, 14, 17, 18, 19, 20, 21, 22]  # All arm joints
            pytorch_pred[arm_indices] = 0.0

        return pytorch_pred


class Controller:
    def __init__(self, policy: TorchPolicy) -> None:
        self.policy = policy

        self.qj = np.zeros(G1_NUM_MOTOR, dtype=np.float32)
        self.dqj = np.zeros(G1_NUM_MOTOR, dtype=np.float32)
        self.action = np.zeros(G1_NUM_MOTOR, dtype=np.float32)  # In MuJoCo order
        self.last_action_pytorch = np.zeros(G1_NUM_MOTOR, dtype=np.float32)  # In PyTorch order
        self.counter = 0

        # Convert joint range tuples to numpy arrays for efficient clamping
        joint_limits = np.array(RESTRICTED_JOINT_RANGE, dtype=np.float32)
        self._joint_lower_bounds = joint_limits[:, 0]
        self._joint_upper_bounds = joint_limits[:, 1]

        self._controller = KeyboardController(
            vel_scale_x=1.0,
            vel_scale_y=1.0,
            vel_scale_rot=1.0,
        )

        self.control_dt = 0.02

        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = unitree_hg_msg_dds__LowState_()
        self.mode_pr_ = MotorMode.PR
        self.mode_machine_ = 0

        self.lowcmd_publisher_ = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher_.Init()

        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.LowStateHandler, 10)
        self.default_pos_array = np.array(default_pos)
        self.default_angles_array = np.array(default_angles_config)

        # wait for the subscriber to receive data
        self.wait_for_low_state()

        # Initialize the command msg
        init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)

    def send_cmd(self, cmd: LowCmd_):
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)

    def zero_torque_state(self):
        print("Enter zero torque state.")
        print("Press Enter to continue...")
        with NonBlockingInput() as nbi:
            while not nbi.check_key('\n'):
                create_zero_cmd(self.low_cmd)
                self.send_cmd(self.low_cmd)
        print("Zero torque state confirmed. Proceeding...")

    def LowStateHandler(self, msg: LowState_):
        self.low_state = msg
        self.mode_machine_ = self.low_state.mode_machine

    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.control_dt)
        print("Successfully connected to the robot.")

    def move_to_default_pos(self):
        print("Moving to default pos.")
        # move time 2s
        total_time = 2
        num_step = int(total_time / self.control_dt)

        init_dof_pos = np.zeros(G1_NUM_MOTOR, dtype=np.float32)
        for i in range(G1_NUM_MOTOR):
            init_dof_pos[i] = self.low_state.motor_state[joint2motor_idx[i]].q

        # move to default pos
        for i in range(num_step):
            alpha = i / num_step
            for j in range(G1_NUM_MOTOR):
                motor_idx = joint2motor_idx[j]
                target_pos = default_pos[j]
                self.low_cmd.motor_cmd[motor_idx].q = init_dof_pos[j] * \
                    (1 - alpha) + target_pos * alpha
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = Kp[j]
                self.low_cmd.motor_cmd[motor_idx].kd = Kd[j]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.control_dt)

    def default_pos_state(self):
        print("Enter default pos state.")
        print("Press Enter to start the controller...")
        with NonBlockingInput() as nbi:
            while not nbi.check_key('\n'):  # Check for Enter key
                # Keep sending default position commands while waiting
                for i in range(len(joint2motor_idx)):
                    motor_idx = joint2motor_idx[i]
                    self.low_cmd.motor_cmd[motor_idx].q = default_pos[i]
                    self.low_cmd.motor_cmd[motor_idx].qd = 0
                    self.low_cmd.motor_cmd[motor_idx].kp = Kp[i]
                    self.low_cmd.motor_cmd[motor_idx].kd = Kd[i]
                    self.low_cmd.motor_cmd[motor_idx].tau = 0

                self.send_cmd(self.low_cmd)
                time.sleep(self.control_dt)
        print("Default position state confirmed. Starting controller...")

    def get_obs(self) -> np.ndarray:
        """Construct observation in the format expected by the PyTorch model."""
        # Get projected gravity (3 dimensions)
        quat = self.low_state.imu_state.quaternion
        gravity = get_gravity_orientation(quat)
        
        # Get velocity commands (3 dimensions)
        velocity_commands = self._controller.get_command()
        
        # Get joint positions and velocities in MuJoCo order, then convert to PyTorch order
        joint_pos_mujoco = self.qj.copy()  # Already relative to default_angles_config
        joint_vel_mujoco = self.dqj.copy()
        
        # Convert to PyTorch model joint order for the observation
        joint_pos_pytorch = remap_mujoco_to_pytorch(joint_pos_mujoco)
        joint_vel_pytorch = remap_mujoco_to_pytorch(joint_vel_mujoco)
        
        # Concatenate all observations: 3 + 3 + 23 + 23 + 23 = 75
        obs = np.hstack([
            gravity,
            velocity_commands, 
            joint_pos_pytorch,
            joint_vel_pytorch,
            self.last_action_pytorch,
        ])
        
        return obs.astype(np.float32)

    def run(self):
        self.counter += 1
        # Get the current joint position and velocity
        for i in range(G1_NUM_MOTOR):
            self.qj[i] = self.low_state.motor_state[joint2motor_idx[i]].q - self.default_angles_array[i]
            self.dqj[i] = self.low_state.motor_state[joint2motor_idx[i]].dq

        # Create observation
        obs = self.get_obs()

        # Get actions from policy (in PyTorch order)
        pytorch_actions = self.policy.get_control(obs)
        
        # Store last action in PyTorch order for next observation
        self.last_action_pytorch = pytorch_actions.copy()
        
        # Convert actions from PyTorch order to MuJoCo order
        mujoco_actions = remap_pytorch_to_mujoco(pytorch_actions)
        
        # Apply action scaling
        action_effect = mujoco_actions * action_scale
        
        # Calculate motor targets
        motor_targets_unclamped = self.default_angles_array + action_effect

        # Clamp motor targets to joint limits and check for clamping
        motor_targets = np.clip(
            motor_targets_unclamped, self._joint_lower_bounds, self._joint_upper_bounds
        )
        clamped_indices = np.where(motor_targets != motor_targets_unclamped)[0]
        if clamped_indices.size > 0:
            print("WARNING: Clamping motor targets for joints:")
            for idx in clamped_indices:
                print(f"  Joint {idx}: {motor_targets_unclamped[idx]:.3f} -> {motor_targets[idx]:.3f} (limits: [{self._joint_lower_bounds[idx]:.3f}, {self._joint_upper_bounds[idx]:.3f}])")

        # Build low cmd
        for i in range(G1_NUM_MOTOR):
            motor_idx = joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = motor_targets[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = Kp[i]
            self.low_cmd.motor_cmd[motor_idx].kd = Kd[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0

        # send the command
        self.send_cmd(self.low_cmd)
        time.sleep(self.control_dt)


if __name__ == "__main__":
    print("Setting up policy...")
    policy = TorchPolicy(POLICY_PATH)
    print("WARNING: Please ensure there are no obstacles around the robot while running this example.")
    # Initial prompt doesn't need non-blocking
    input("Press Enter to acknowledge warning and proceed...")

    ChannelFactoryInitialize(0, NETWORK_CARD_NAME)

    controller = Controller(policy)

    # Enter the zero torque state, press Enter key to continue executing
    controller.zero_torque_state()

    # Move to the default position
    controller.move_to_default_pos()

    # Enter the default position state, press Enter key to continue executing
    controller.default_pos_state()

    print("Controller running. Press 'q' to quit.")
    with NonBlockingInput() as nbi:  # Use context manager for the main loop
        while True:
            controller.run()
            # Check for 'q' key press to exit
            if nbi.check_key('q'):
                print("\n'q' pressed. Exiting loop...")
                break
            # Add a small sleep to prevent busy-waiting if controller.run() is very fast
            time.sleep(0.001)

    print("Entering damping state...")
    create_damping_cmd(controller.low_cmd)
    controller.send_cmd(controller.low_cmd)

    print("Exit") 