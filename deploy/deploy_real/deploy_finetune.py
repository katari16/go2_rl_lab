"""Deploy finetuned low-level locomotion policy on Go2 (no force estimator).

Obs layout (60 dims):
    [0:3]   base_ang_vel
    [3:6]   projected_gravity
    [6:9]   velocity_commands
    [9:21]  joint_pos_rel
    [21:33] joint_vel_rel
    [33:45] last_action
    [45:57] applied_torque * 0.1
    [57:60] force_estimate (zeros — no estimator on real robot)

Usage:
    python deploy_finetune.py <net_interface> <config.yaml>
    python deploy_finetune.py eth0 go2_finetune_j1.yaml
"""

############ LIBRARIES #############
import numpy as np
import time
import torch
import sys
import matplotlib.pyplot as plt
from math import *
from pathlib import Path
####################################

current_dir = Path(__file__).resolve()
project_root = current_dir.parents[1]

sys.path.append('.')
sys.path.append('..')
sys.path.append(project_root / "unitree_sdk2_python")

from unitree_sdk2_python.unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2_python.unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize

from unitree_sdk2_python.unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_

from unitree_sdk2_python.unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2_python.unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo

from unitree_sdk2_python.unitree_sdk2py.utils.crc import CRC
from unitree_sdk2_python.unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2_python.unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2_python.unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_

from common.command_helper import create_damping_cmd, create_zero_cmd, init_cmd_go
from common.rotation_helper import get_gravity_orientation, transform_imu_data
from common.remote_controller import RemoteController, KeyMap


class FinetuneConfig:
    """Extended config for finetuned policies (60-dim obs with torques)."""
    def __init__(self, file_path):
        import yaml
        with open(file_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        self.control_dt = config["control_dt"]
        self.msg_type = config["msg_type"]
        self.imu_type = config["imu_type"]
        self.weak_motor = config.get("weak_motor", [])
        self.lowcmd_topic = config["lowcmd_topic"]
        self.lowstate_topic = config["lowstate_topic"]
        self.policy_path = config["policy_path"]
        self.leg_joint2motor_idx = config["leg_joint2motor_idx"]
        self.kps = config["kps"]
        self.kds = config["kds"]
        self.default_angles = np.array(config["default_angles"], dtype=np.float32)
        self.arm_waist_joint2motor_idx = config.get("arm_waist_joint2motor_idx", [])
        self.arm_waist_kps = config.get("arm_waist_kps", [])
        self.arm_waist_kds = config.get("arm_waist_kds", [])
        self.arm_waist_target = np.array(config.get("arm_waist_target", []), dtype=np.float32)
        self.action_scale = config["action_scale"]
        self.cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
        self.max_cmd = np.array(config["max_cmd"], dtype=np.float32)
        self.num_actions = config["num_actions"]
        self.num_obs = config["num_obs"]


class Controller():
    def __init__(self, config: FinetuneConfig) -> None:
        self.config = config
        self.remote_controller = RemoteController()

        # Load JIT policy
        print("3] ---> LOADING POLICY")
        policy_path = current_dir.parent.parent / "pre_train" / config.policy_path
        self.policy = torch.jit.load(policy_path)
        print(f"         Policy loaded from: {policy_path}")
        print(f"         Obs dims: {config.num_obs}, Action scale: {config.action_scale}")
        print(f"         PD gains: Kp={config.kps[0]}, Kd={config.kds[0]}")

        # Isaac default joint angles (policy order)
        self.defaut_isaac = [0.1, -0.1, 0.1, -0.1, 0.8, 0.8, 1, 1, -1.5, -1.5, -1.5, -1.5]
        self.cmd = np.array([0.0, 0.0, 0.0])
        self.qj = np.zeros(config.num_actions, dtype=np.float32)
        self.dqj = np.zeros(config.num_actions, dtype=np.float32)
        self.action = np.zeros(config.num_actions, dtype=np.float32)
        self.obs = np.zeros(config.num_obs, dtype=np.float32)
        self.debug_log = []
        self.counter = 0

        # Standup sequence
        self.dt = 0.002
        self.startPos = [0.0] * 12
        self.duration_1 = 500
        self.duration_2 = 500
        self.duration_3 = 1000
        self.duration_4 = 900
        self.percent_1 = 0
        self.percent_2 = 0
        self.percent_3 = 0
        self.percent_4 = 0
        self.firstRun = True

        self._targetPos_1 = [0.0, 1.36, -2.65, 0.0, 1.36, -2.65, -0.2, 1.36, -2.65, 0.2, 1.36, -2.65]
        self._targetPos_2 = [-0.1, 0.8, -1.5, 0.1, 0.8, -1.5, -0.1, 1, -1.5, 0.1, 1, -1.5]
        self._targetPos_3 = self._targetPos_2

        # Velocity estimation
        window_size = 20
        self.vx_window = [0] * window_size
        self.vy_window = [0] * window_size
        self.vz_window = [0] * window_size

        # Plot data
        self.L_base_vel_cmd = [[], [], []]

        # Channels
        print("4] ----> INITIALIZING CHANNELS")
        self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdGo)
        self.lowcmd_publisher_.Init()
        self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateGo)
        self.lowstate_subscriber.Init(self.LowStateGoHandler, 10)
        self.sportstate_subscriber = ChannelSubscriber("rt/sportmodestate", SportModeState_)
        self.sportstate_subscriber.Init(self.SportStateMessageHandler, 10)

        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.low_state = unitree_go_msg_dds__LowState_()
        self.wait_for_low_state()
        init_cmd_go(self.low_cmd, weak_motor=self.config.weak_motor)

    def Init(self):
        self.sc = SportClient()
        self.sc.SetTimeout(5.0)
        self.sc.Init()
        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()
        status, result = self.msc.CheckMode()
        while result is not None and result.get('name'):
            self.sc.StandDown()
            self.msc.ReleaseMode()
            print("3] ---> ROBOT IS IN LYING POSITION AND HIGH-LEVEL MODE IS RELEASED -> SWITCHING TO LOW-LEVEL")
            status, result = self.msc.CheckMode()
            time.sleep(1)

    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.config.control_dt)
        print("         Connected to robot")

    def LowStateGoHandler(self, msg: LowStateGo):
        self.low_state = msg
        self.remote_controller.set(self.low_state.wireless_remote)

    def SportStateMessageHandler(self, sport_state_msg):
        self.velocity = sport_state_msg.velocity

    def send_cmd(self, cmd: LowCmdGo):
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)

    def zero_torque_state(self):
        print("5] -----> ZERO TORQUE STATE IS ACTIVE")
        print("          ##################################################")
        print("          # WAITING FOR START BUTTON TO RAISE THE ROBOT     #")
        print("          ##################################################")
        while self.remote_controller.button[KeyMap.start] != 1:
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def move_to_default_pos(self):
        print("6] ------> ROBOT IS MOVING TO DEFAULT POSE")
        dof_idx = self.config.leg_joint2motor_idx
        done = False

        if self.firstRun:
            for i in range(12):
                self.startPos[i] = self.low_state.motor_state[i].q
            self.firstRun = False
        self.count = 0

        while not done:
            self.count += 1
            self.percent_1 += 1.0 / self.duration_1
            self.percent_1 = min(self.percent_1, 1)
            if self.percent_1 < 1:
                for i in range(12):
                    self.low_cmd.motor_cmd[i].q = (1 - self.percent_1) * self.startPos[i] + self.percent_1 * self._targetPos_1[i]
                    self.low_cmd.motor_cmd[i].dq = 0
                    self.low_cmd.motor_cmd[i].kp = 60
                    self.low_cmd.motor_cmd[i].kd = 5
                    self.low_cmd.motor_cmd[i].tau = 0

            if (self.percent_1 == 1) and (self.percent_2 <= 1):
                self.percent_2 += 1.0 / self.duration_2
                self.percent_2 = min(self.percent_2, 1)
                for i in range(12):
                    self.low_cmd.motor_cmd[i].q = (1 - self.percent_2) * self._targetPos_1[i] + self.percent_2 * self._targetPos_2[i]
                    self.low_cmd.motor_cmd[i].dq = 0
                    self.low_cmd.motor_cmd[i].kp = 60
                    self.low_cmd.motor_cmd[i].kd = 5
                    self.low_cmd.motor_cmd[i].tau = 0

            if (self.percent_1 == 1) and (self.percent_2 == 1) and (self.percent_3 < 1):
                self.percent_3 += 1.0 / self.duration_3
                self.percent_3 = min(self.percent_3, 1)
                for i in range(12):
                    self.low_cmd.motor_cmd[i].q = self._targetPos_2[i]
                    self.low_cmd.motor_cmd[i].dq = 0
                    self.low_cmd.motor_cmd[i].kp = 60
                    self.low_cmd.motor_cmd[i].kd = 5
                    self.low_cmd.motor_cmd[i].tau = 0

            if (self.percent_1 == 1) and (self.percent_2 == 1) and (self.percent_3 == 1) and (self.percent_4 <= 1):
                self.percent_4 += 1.0 / self.duration_4
                self.percent_4 = min(self.percent_4, 1)
                for i in range(12):
                    self.low_cmd.motor_cmd[i].q = (1 - self.percent_4) * self._targetPos_2[i] + self.percent_4 * self._targetPos_3[i]
                    self.low_cmd.motor_cmd[i].dq = 0
                    self.low_cmd.motor_cmd[i].kp = 60
                    self.low_cmd.motor_cmd[i].kd = 5
                    self.low_cmd.motor_cmd[i].tau = 0

            self.send_cmd(self.low_cmd)
            if self.percent_4 == 1.0:
                done = True
            time.sleep(0.001)

        print("7] -------> ROBOT IS STANDING")
        print("            ###########################################")
        print("            # PRESS 'A' TO START THE MODEL            #")
        print("            ###########################################")
        while self.remote_controller.button[KeyMap.A] != 1:
            default = self.config.default_angles
            for i in range(12):
                self.low_cmd.motor_cmd[i].q = default[i]
                self.low_cmd.motor_cmd[i].qd = 0
                self.low_cmd.motor_cmd[i].kp = 60
                self.low_cmd.motor_cmd[i].kd = 5
                self.low_cmd.motor_cmd[i].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(0.002)

    def move_to_ground(self):
        percent = 0
        pos_init = []
        for k in range(12):
            pos_init.append(self.low_state.motor_state[k].q)
        while percent != 1:
            percent += 1.0 / 300
            percent = min(percent, 1)
            lying_pos = [0, 1.36, -2.65, 0, 1.36, -2.65, -0.2, 1.36, -2.65, 0.2, 1.36, -2.65]
            for i in range(12):
                self.low_cmd.motor_cmd[i].q = (1 - percent) * pos_init[i] + percent * lying_pos[i]
                self.low_cmd.motor_cmd[i].dq = 0
                self.low_cmd.motor_cmd[i].kp = 60
                self.low_cmd.motor_cmd[i].kd = 5
                self.low_cmd.motor_cmd[i].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(0.002)
        print("9] ---------> ROBOT IS LYING DOWN")

    def run(self):
        self.counter += 1

        # Angular velocity
        ang_vel = np.array(self.low_state.imu_state.gyroscope, dtype=np.float32)

        # Projected gravity
        quat = self.low_state.imu_state.quaternion
        gravity_orientation = get_gravity_orientation(quat)

        # Velocity commands from joystick
        self.cmd[0] = round(self.remote_controller.ly, 1)
        self.cmd[1] = round(self.remote_controller.lx * -1, 1)
        self.cmd[2] = round(self.remote_controller.rx * -1, 1)

        # Joint positions and velocities in policy order
        for i in range(len(self.config.leg_joint2motor_idx)):
            self.qj[i] = self.low_state.motor_state[self.config.leg_joint2motor_idx[i]].q
            self.dqj[i] = self.low_state.motor_state[self.config.leg_joint2motor_idx[i]].dq

        qj_rel = self.qj.copy() - np.array(self.defaut_isaac, dtype=np.float32)
        dqj = self.dqj.copy()

        # Applied torques (read from motor state)
        torques = np.zeros(self.config.num_actions, dtype=np.float32)
        for i in range(len(self.config.leg_joint2motor_idx)):
            motor_idx = self.config.leg_joint2motor_idx[i]
            torques[i] = self.low_state.motor_state[motor_idx].tau_est

        # Build 60-dim observation
        # [0:3]   base_ang_vel
        self.obs[0:3] = ang_vel
        # [3:6]   projected_gravity
        self.obs[3:6] = gravity_orientation
        # [6:9]   velocity_commands
        self.obs[6:9] = self.cmd * self.config.cmd_scale * self.config.max_cmd
        # [9:21]  joint_pos_rel
        self.obs[9:21] = qj_rel
        # [21:33] joint_vel_rel
        self.obs[21:33] = dqj
        # [33:45] last_action
        self.obs[33:45] = self.action
        # [45:57] applied_torque * 0.1
        self.obs[45:57] = torques * 0.1
        # [57:60] force_estimate (zeros — no estimator on real robot)
        self.obs[57:60] = 0.0

        # Run policy
        obs_tensor = torch.from_numpy(self.obs).unsqueeze(0)
        self.action = self.policy(obs_tensor).detach().numpy().squeeze()

        # Debug log
        self.debug_log.append({
            'step': self.counter,
            'obs': self.obs.copy().tolist(),
            'action': self.action.copy().tolist(),
            'target_dof_pos': (self.action * self.config.action_scale + np.array(self.defaut_isaac)).tolist(),
            'joint_pos_isaac': self.qj.copy().tolist(),
            'joint_vel_isaac': self.dqj.copy().tolist(),
            'imu_quat': list(self.low_state.imu_state.quaternion),
            'imu_gyro': list(self.low_state.imu_state.gyroscope),
            'projected_gravity': gravity_orientation.tolist(),
            'velocity_cmd': self.cmd.tolist(),
            'torques': torques.tolist(),
        })

        # Send commands to motors
        target_dof_pos = self.action * self.config.action_scale + np.array(self.defaut_isaac)
        for i in range(len(self.config.leg_joint2motor_idx)):
            motor_idx = self.config.leg_joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = target_dof_pos[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0
        self.send_cmd(self.low_cmd)

        # Store plot data
        self.L_base_vel_cmd[0].append(self.obs[6])
        self.L_base_vel_cmd[1].append(self.obs[7])
        self.L_base_vel_cmd[2].append(self.obs[8])

        return self.obs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Deploy finetuned low-level policy on Go2")
    parser.add_argument("net", type=str, help="network interface")
    parser.add_argument("config", type=str, help="config file name in the configs folder", default="go2_finetune_j1.yaml")
    args = parser.parse_args()

    config_path = Path(__file__).resolve().parent / "configs" / args.config
    config = FinetuneConfig(config_path)
    print("1] -> CONFIG FILE LOADED SUCCESSFULLY")

    ChannelFactoryInitialize(0, args.net)
    print("2] --> CHANNEL FACTORY CREATED")

    controller = Controller(config)
    controller.Init()

    controller.zero_torque_state()
    controller.move_to_default_pos()

    print("8] --------> MODEL IS RUNNING")
    print("             ###############################################")
    print("             # PRESS 'SELECT' TO STOP THE MODEL            #")
    print("             ###############################################")

    while True:
        try:
            obs = controller.run()
            time.sleep(0.02)

            if controller.remote_controller.button[KeyMap.select] == 1:
                controller.move_to_ground()
                break

        except KeyboardInterrupt:
            break

    import json
    with open("debug_log.json", "w") as f:
        json.dump(controller.debug_log, f)
    print(f"Saved {len(controller.debug_log)} steps to debug_log.json")

    # Visualization
    print("10] ----------> DATA VISUALIZATION IN PROGRESS")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    labels = ['Vx', 'Vy', 'Wz']
    for i, ax in enumerate(axes):
        ax.plot(controller.L_base_vel_cmd[i], label=f"cmd_{labels[i]}")
        ax.legend()
        ax.set_title(labels[i])
    plt.tight_layout()
    plt.savefig("deploy_finetune_plot.png")
    plt.show()
