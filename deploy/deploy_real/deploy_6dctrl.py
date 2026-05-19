"""
Real-world deployment: 6Dctrl policy (R1 + commanded roll/pitch/height) with 6D wrench estimator.

Policy obs (66 dims) = 60 raw + 6 wrench estimate.
Raw obs layout (60 dims):
    [0:3]   base_ang_vel
    [3:6]   projected_gravity
    [6:9]   velocity_cmd (vx, vy, wz)
    [9:12]  pose_cmd (roll, pitch, height)
    [12:24] joint_pos_rel
    [24:36] joint_vel_rel
    [36:48] last_action
    [48:60] applied_torque * 0.1

Joystick mapping:
    Left stick     = vx (ly), vy (-lx)
    Right stick X  = wz (-rx)
    L1 / L2        = height -/+ step (0.01 m per press, clamped to cfg range)
    D-pad up/down  = pitch +/- step (0.02 rad per press, clamped)
    D-pad left/right = roll +/- step (0.02 rad per press, clamped)
    A = start policy | SELECT = stop | START = stand up
    B = toggle recording | Y = compliance off/on | X = compliance inverted/normal
    R1 = reset pose command to nominal (roll=0, pitch=0, height=nominal)

Compliance injection (6D wrench → commands):
    velocity_cmd[0:2] += k_xy * F_xy_ema
    velocity_cmd[2]   += k_yaw * τz_ema
    pose_cmd[0]       += k_roll  * τx_ema   (nominally ON, OFF with --height_only)
    pose_cmd[1]       += k_pitch * τy_ema   (nominally ON, OFF with --height_only)
    pose_cmd[2]       += k_height * Fz_ema  (nominally OFF, ON with --height_only)
Roll/pitch/height results are clamped to the pose_cmd_ranges from the yaml so
estimator outliers cannot send the policy out-of-distribution commands.
"""

import argparse
import json
import sys
import time
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))
sys.path.append(str(project_root / "unitree_sdk2_python"))

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_, SportModeState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.go2.sport.sport_client import SportClient

from common.remote_controller import RemoteController, KeyMap

import yaml


def get_gravity_orientation(quat):
    qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]
    gx = 2 * (-qz * qx + qw * qy)
    gy = -2 * (qz * qy + qw * qx)
    gz = -1 + 2 * (qx * qx + qy * qy)
    return np.array([gx, gy, gz])


def init_cmd_go(cmd, weak_motor=None):
    if weak_motor is None:
        weak_motor = []
    cmd.head[0] = 0xFE
    cmd.head[1] = 0xEF
    cmd.level_flag = 0xFF
    cmd.gpio = 0
    for i in range(len(cmd.motor_cmd)):
        cmd.motor_cmd[i].mode = 1 if i in weak_motor else 0x0A
        cmd.motor_cmd[i].q = 2.146e9
        cmd.motor_cmd[i].qd = 16000.0
        cmd.motor_cmd[i].kp = 0
        cmd.motor_cmd[i].kd = 0
        cmd.motor_cmd[i].tau = 0


low_state = unitree_go_msg_dds__LowState_()


class ForceEstimatorDeployment:
    def __init__(self, jit_path, raw_obs_dim, force_dim, temporal_steps, device="cpu"):
        self.device = torch.device(device)
        self.raw_obs_dim = raw_obs_dim
        self.force_dim = force_dim
        self.temporal_steps = temporal_steps
        self.estimator = torch.jit.load(jit_path, map_location=self.device)
        self.estimator.eval()
        print(f"[Estimator] Loaded JIT model from {jit_path}")
        print(f"  Input:  [{temporal_steps * raw_obs_dim}] = {temporal_steps} x {raw_obs_dim}")
        print(f"  Output: [{force_dim}]")
        self._history = torch.zeros(1, temporal_steps * raw_obs_dim, device=self.device)
        self._step = 0

    def reset(self):
        self._history.zero_()
        self._step = 0

    def get_force_estimate(self, raw_obs):
        obs_t = torch.from_numpy(raw_obs).float().unsqueeze(0).to(self.device)
        if self._step < self.temporal_steps:
            start = self._step * self.raw_obs_dim
            self._history[0, start:start + self.raw_obs_dim] = obs_t[0]
            self._step += 1
        else:
            self._history[0, :-self.raw_obs_dim] = self._history[0, self.raw_obs_dim:].clone()
            self._history[0, -self.raw_obs_dim:] = obs_t[0]
        with torch.inference_mode():
            force_hat = self.estimator(self._history)
        return force_hat[0].cpu().numpy()


def _save_recording(buf, index, config_name, control_dt, force_dim, yaw_idx):
    if len(buf) < 2:
        print("[recording] Too short to save, skipping.")
        return
    import csv
    log_dir = Path(__file__).resolve().parent / "logs" / "recordings"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    stem = config_name.replace(".yaml", "")
    base = log_dir / f"{stem}_rec{index:02d}_{ts}"

    with open(str(base) + ".json", "w") as f:
        json.dump({
            "config": config_name,
            "control_dt": control_dt,
            "force_dim": force_dim,
            "has_yaw": yaw_idx is not None,
            "num_steps": len(buf),
            "steps": buf,
        }, f, indent=2)

    force_labels = [f"F{ax}_hat" for ax in ["x", "y", "z"][:min(force_dim, 3)]]
    if force_dim >= 6:
        force_labels += ["tau_x_hat", "tau_y_hat", "tau_z_hat"]
    elif yaw_idx is not None:
        force_labels += ["tau_yaw_hat"]
    fieldnames = ["t", *force_labels,
                  "force_mag", "vx_cmd", "vy_cmd", "wz_cmd",
                  "roll_cmd", "pitch_cmd", "height_cmd", "compliance_mode"]
    with open(str(base) + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in buf:
            r = {"t": row["t"], "force_mag": row["force_mag"],
                 "vx_cmd": row["velocity_cmd"][0], "vy_cmd": row["velocity_cmd"][1],
                 "wz_cmd": row["velocity_cmd"][2],
                 "roll_cmd": row["pose_cmd"][0], "pitch_cmd": row["pose_cmd"][1],
                 "height_cmd": row["pose_cmd"][2],
                 "compliance_mode": row["compliance_mode"]}
            for i, lbl in enumerate(force_labels):
                r[lbl] = row["force_hat"][i]
            w.writerow(r)

    print(f"[recording] Saved {len(buf)} steps → {base}.json / .csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy 6Dctrl policy on real Go2.")
    parser.add_argument("net", type=str)
    parser.add_argument("config", type=str)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument(
        "--height_only",
        action="store_true",
        default=False,
        help="Only inject Fz-based height compliance; disable roll/pitch pose compliance.",
    )
    args = parser.parse_args()

    config_path = Path(__file__).resolve().parent / "configs" / args.config
    with open(config_path, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    num_actions = cfg["num_actions"]
    raw_obs_dim = cfg["raw_obs_dim"]
    force_dim = cfg["force_dim"]
    policy_obs_dim = cfg["policy_obs_dim"]
    action_scale = cfg["action_scale"]
    cmd_scale = np.array(cfg["cmd_scale"], dtype=np.float32)
    max_cmd = np.array(cfg["max_cmd"], dtype=np.float32)
    kps = cfg["kps"]
    kds = cfg["kds"]
    leg_joint2motor_idx = cfg["leg_joint2motor_idx"]
    default_angles_sdk = np.array(cfg["default_angles"], dtype=np.float32)
    control_dt = cfg["control_dt"]
    temporal_steps = cfg["estimator_temporal_steps"]
    compliance_k = cfg.get("compliance_k", 0.0)
    compliance_k_yaw = cfg.get("compliance_k_yaw", 0.0)
    compliance_k_roll = cfg.get("compliance_k_roll", 0.0)
    compliance_k_pitch = cfg.get("compliance_k_pitch", 0.0)
    compliance_k_height = cfg.get("compliance_k_height", 0.0)
    ema_alpha = cfg.get("ema_alpha", 0.1)

    # Pose compliance gating: roll/pitch nominally ON, height ONLY with --height_only.
    pose_roll_enabled = not args.height_only
    pose_pitch_enabled = not args.height_only
    pose_height_enabled = args.height_only
    weak_motor = cfg.get("weak_motor", [])
    force_layout = cfg.get("force_layout", "auto")

    pose_ranges = cfg.get("pose_cmd_ranges", {})
    roll_min, roll_max = pose_ranges.get("roll", [-0.35, 0.35])
    pitch_min, pitch_max = pose_ranges.get("pitch", [-0.35, 0.35])
    height_min, height_max = pose_ranges.get("height", [0.18, 0.38])
    nominal_height = cfg.get("nominal_height", 0.34)

    assert raw_obs_dim == 60, "6Dctrl deploy expects raw_obs_dim=60 (extra pose cmd channels)"

    if force_layout == "xy_yaw":
        yaw_idx = 2
    elif force_dim >= 6:
        yaw_idx = 5
    elif force_dim >= 4:
        yaw_idx = 3
    else:
        yaw_idx = None

    DEFAULT_ANGLES_ISAAC = np.array(
        [0.1, -0.1, 0.1, -0.1, 0.8, 0.8, 1, 1, -1.5, -1.5, -1.5, -1.5],
        dtype=np.float32,
    )
    LYING_POS = [0.0, 1.36, -2.65, 0.0, 1.36, -2.65, -0.2, 1.36, -2.65, 0.2, 1.36, -2.65]

    print(f"[1] Config loaded: {config_path}")

    policy_path = str(project_root / cfg["policy_path"])
    estimator_path = str(project_root / cfg["estimator_path"])

    print(f"[2] Loading policy from {policy_path}")
    policy = torch.jit.load(policy_path, map_location="cpu")
    policy.eval()

    estimator = ForceEstimatorDeployment(
        jit_path=estimator_path,
        raw_obs_dim=raw_obs_dim,
        force_dim=force_dim,
        temporal_steps=temporal_steps,
        device="cpu",
    )

    print(f"[3] Initializing DDS on {args.net}")
    ChannelFactoryInitialize(0, args.net)
    remote_controller = RemoteController()
    low_cmd = unitree_go_msg_dds__LowCmd_()
    pub = ChannelPublisher("rt/lowcmd", LowCmd_)
    pub.Init()

    def low_state_handler(msg):
        global low_state
        low_state = msg
        remote_controller.set(msg.wireless_remote)

    sub_low = ChannelSubscriber("rt/lowstate", LowState_)
    sub_low.Init(low_state_handler, 10)

    crc = CRC()

    print("[4] Waiting for robot state...")
    while low_state.tick == 0:
        time.sleep(control_dt)
    print("    Connected to robot.")

    init_cmd_go(low_cmd, weak_motor=weak_motor)

    def send_cmd():
        low_cmd.crc = crc.Crc(low_cmd)
        pub.Write(low_cmd)

    print("[5] Switching to low-level control...")
    sc = SportClient()
    sc.SetTimeout(5.0)
    sc.Init()
    msc = MotionSwitcherClient()
    msc.SetTimeout(5.0)
    msc.Init()

    status, result = msc.CheckMode()
    while result is not None and result.get('name'):
        sc.StandDown()
        msc.ReleaseMode()
        print("    Releasing high-level mode...")
        status, result = msc.CheckMode()
        time.sleep(1)

    print("\n" + "=" * 60)
    print("  ZERO TORQUE — Press START to stand up")
    print("=" * 60 + "\n")
    while remote_controller.button[KeyMap.start] != 1:
        for i in range(12):
            low_cmd.motor_cmd[i].q = 0.0
            low_cmd.motor_cmd[i].kp = 0.0
            low_cmd.motor_cmd[i].kd = 0.0
            low_cmd.motor_cmd[i].dq = 0.0
            low_cmd.motor_cmd[i].tau = 0.0
        send_cmd()
        time.sleep(control_dt)

    print("[6] Standing up...")
    dt = 0.002
    start_pos = [low_state.motor_state[i].q for i in range(12)]
    for step in range(500):
        phase = min(step / 500, 1.0)
        for i in range(12):
            low_cmd.motor_cmd[i].q = (1 - phase) * start_pos[i] + phase * LYING_POS[i]
            low_cmd.motor_cmd[i].kp = 60.0
            low_cmd.motor_cmd[i].kd = 5.0
            low_cmd.motor_cmd[i].dq = 0.0
            low_cmd.motor_cmd[i].tau = 0.0
        send_cmd()
        time.sleep(dt)
    for step in range(500):
        phase = min(step / 500, 1.0)
        for i in range(12):
            low_cmd.motor_cmd[i].q = (1 - phase) * LYING_POS[i] + phase * default_angles_sdk[i]
            low_cmd.motor_cmd[i].kp = 60.0
            low_cmd.motor_cmd[i].kd = 5.0
            low_cmd.motor_cmd[i].dq = 0.0
            low_cmd.motor_cmd[i].tau = 0.0
        send_cmd()
        time.sleep(dt)
    for step in range(500):
        for i in range(12):
            low_cmd.motor_cmd[i].q = default_angles_sdk[i]
            low_cmd.motor_cmd[i].kp = 60.0
            low_cmd.motor_cmd[i].kd = 5.0
            low_cmd.motor_cmd[i].dq = 0.0
            low_cmd.motor_cmd[i].tau = 0.0
        send_cmd()
        time.sleep(dt)

    gravity_check = get_gravity_orientation(low_state.imu_state.quaternion)
    print(f"    Standing. Gravity={gravity_check.round(3)}, Policy Kp={kps[0]}, Kd={kds[0]}")

    print("\n" + "=" * 60)
    print("  STANDING — Press A to start policy")
    print("  Controls: left stick=vx/vy | right stick X=wz | L1/L2=height -/+ step")
    print("           D-pad up/down=pitch step | left/right=roll step | R1=reset pose")
    print("           B=record | Y=compliance off/on | X=compliance inverted/normal | SELECT=stop")
    print("=" * 60 + "\n")
    while remote_controller.button[KeyMap.A] != 1:
        for i in range(12):
            low_cmd.motor_cmd[i].q = default_angles_sdk[i]
            low_cmd.motor_cmd[i].kp = 60.0
            low_cmd.motor_cmd[i].kd = 5.0
            low_cmd.motor_cmd[i].dq = 0.0
            low_cmd.motor_cmd[i].tau = 0.0
        send_cmd()
        time.sleep(dt)

    print("\n" + "=" * 60)
    print(f"  POLICY RUNNING (50 Hz) — {raw_obs_dim}+{force_dim}={policy_obs_dim} obs dims")
    print(f"  Force estimator: {temporal_steps}x{raw_obs_dim} -> {force_dim}D wrench")
    print(f"  Vel compliance:  k_xy={compliance_k}  k_yaw={compliance_k_yaw}")
    print(f"  Pose compliance: roll={'ON' if pose_roll_enabled else 'OFF'} (k={compliance_k_roll}) "
          f"pitch={'ON' if pose_pitch_enabled else 'OFF'} (k={compliance_k_pitch}) "
          f"height={'ON' if pose_height_enabled else 'OFF'} (k={compliance_k_height})")
    if args.height_only:
        print("  --height_only: roll/pitch pose compliance disabled, height ON")
    print("  Press SELECT to stop")
    print("=" * 60 + "\n")

    action = np.zeros(num_actions, dtype=np.float32)
    raw_obs = np.zeros(raw_obs_dim, dtype=np.float32)
    velocity_cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    pose_cmd = np.array([0.0, 0.0, nominal_height], dtype=np.float32)  # roll, pitch, height
    force_ema = np.zeros(force_dim, dtype=np.float32)

    ROLL_STEP = 0.02
    PITCH_STEP = 0.02
    HEIGHT_STEP = 0.01  # meters per L1/L2 press

    step_count = 0
    debug_log = []
    should_stop = False
    compliance_mode = "normal"
    sim_real_recording = False
    prev_b_button = 0
    prev_y_button = 0
    prev_x_button = 0
    prev_up = 0
    prev_down = 0
    prev_left = 0
    prev_right = 0
    prev_r1 = 0
    prev_l1 = 0
    prev_l2 = 0
    recording_buf = []
    recording_index = 0
    estimator.reset()

    try:
        while not should_stop:
            step_start = time.perf_counter()

            velocity_cmd[0] = round(remote_controller.ly, 1)
            velocity_cmd[1] = round(remote_controller.lx * -1, 1)
            velocity_cmd[2] = round(remote_controller.rx * -1, 1)

            # Roll / pitch: discrete D-pad steps
            up_now = remote_controller.button[KeyMap.up]
            down_now = remote_controller.button[KeyMap.down]
            left_now = remote_controller.button[KeyMap.left]
            right_now = remote_controller.button[KeyMap.right]
            if up_now == 1 and prev_up == 0:
                pose_cmd[1] = float(np.clip(pose_cmd[1] + PITCH_STEP, pitch_min, pitch_max))
            if down_now == 1 and prev_down == 0:
                pose_cmd[1] = float(np.clip(pose_cmd[1] - PITCH_STEP, pitch_min, pitch_max))
            if left_now == 1 and prev_left == 0:
                pose_cmd[0] = float(np.clip(pose_cmd[0] - ROLL_STEP, roll_min, roll_max))
            if right_now == 1 and prev_right == 0:
                pose_cmd[0] = float(np.clip(pose_cmd[0] + ROLL_STEP, roll_min, roll_max))
            prev_up, prev_down, prev_left, prev_right = up_now, down_now, left_now, right_now

            # Height: L1 = step down, L2 = step up (discrete, edge-triggered)
            l1_now = remote_controller.button[KeyMap.L1]
            l2_now = remote_controller.button[KeyMap.L2]
            if l1_now == 1 and prev_l1 == 0:
                pose_cmd[2] = float(np.clip(pose_cmd[2] - HEIGHT_STEP, height_min, height_max))
            if l2_now == 1 and prev_l2 == 0:
                pose_cmd[2] = float(np.clip(pose_cmd[2] + HEIGHT_STEP, height_min, height_max))
            prev_l1, prev_l2 = l1_now, l2_now

            # R1: reset pose command to nominal
            r1_now = remote_controller.button[KeyMap.R1]
            if r1_now == 1 and prev_r1 == 0:
                pose_cmd[:] = [0.0, 0.0, nominal_height]
                print(f"[step {step_count}] *** Pose cmd reset to nominal ***", flush=True)
            prev_r1 = r1_now

            # ── Build raw observation (60 dims) ───────────────────────────
            raw_obs[0] = low_state.imu_state.gyroscope[0]
            raw_obs[1] = low_state.imu_state.gyroscope[1]
            raw_obs[2] = low_state.imu_state.gyroscope[2]
            raw_obs[3:6] = get_gravity_orientation(low_state.imu_state.quaternion)
            raw_obs[6:9] = velocity_cmd * cmd_scale * max_cmd
            raw_obs[9:12] = pose_cmd
            for i in range(num_actions):
                motor_idx = leg_joint2motor_idx[i]
                raw_obs[12 + i] = low_state.motor_state[motor_idx].q - DEFAULT_ANGLES_ISAAC[i]
            for i in range(num_actions):
                motor_idx = leg_joint2motor_idx[i]
                raw_obs[24 + i] = low_state.motor_state[motor_idx].dq
            raw_obs[36:48] = action
            for i in range(num_actions):
                motor_idx = leg_joint2motor_idx[i]
                raw_obs[48 + i] = low_state.motor_state[motor_idx].tau_est * 0.1

            force_hat = estimator.get_force_estimate(raw_obs)
            force_ema = ema_alpha * force_hat + (1.0 - ema_alpha) * force_ema

            obs_for_policy = raw_obs.copy()
            # Velocity-command compliance (XY force + yaw torque)
            if compliance_k > 0.0 and compliance_mode == "normal":
                obs_for_policy[6] += compliance_k * force_ema[0]
                obs_for_policy[7] += compliance_k * force_ema[1]
                if compliance_k_yaw > 0.0 and yaw_idx is not None:
                    obs_for_policy[8] += compliance_k_yaw * force_ema[yaw_idx]
            elif compliance_k > 0.0 and compliance_mode == "inverted":
                obs_for_policy[6] -= compliance_k * force_ema[0]
                obs_for_policy[7] -= compliance_k * force_ema[1]
                if compliance_k_yaw > 0.0 and yaw_idx is not None:
                    obs_for_policy[8] -= compliance_k_yaw * force_ema[yaw_idx]

            # Pose-command compliance (τx → roll, τy → pitch, Fz → height).
            # Only active in a 6D layout. Roll/pitch gated by --height_only
            # absence; height gated by --height_only presence. Always clamped
            # to training ranges so estimator outliers cannot send OOD poses.
            if force_dim >= 6 and compliance_mode != "off":
                sign = 1.0 if compliance_mode == "normal" else -1.0
                roll_cmd = float(obs_for_policy[9])
                pitch_cmd = float(obs_for_policy[10])
                height_cmd = float(obs_for_policy[11])
                if pose_roll_enabled and compliance_k_roll > 0.0:
                    roll_cmd += sign * compliance_k_roll * float(force_ema[3])
                if pose_pitch_enabled and compliance_k_pitch > 0.0:
                    pitch_cmd += sign * compliance_k_pitch * float(force_ema[4])
                if pose_height_enabled and compliance_k_height > 0.0:
                    height_cmd += sign * compliance_k_height * float(force_ema[2])
                obs_for_policy[9] = float(np.clip(roll_cmd, roll_min, roll_max))
                obs_for_policy[10] = float(np.clip(pitch_cmd, pitch_min, pitch_max))
                obs_for_policy[11] = float(np.clip(height_cmd, height_min, height_max))

            if compliance_mode == "inverted":
                full_obs = np.concatenate([obs_for_policy, -force_hat])
            else:
                full_obs = np.concatenate([obs_for_policy, force_hat])

            obs_tensor = torch.from_numpy(full_obs).float().unsqueeze(0)
            with torch.inference_mode():
                action_tensor = policy(obs_tensor)
            action = action_tensor.squeeze(0).numpy()

            if np.any(np.isnan(action)):
                print(f"[step {step_count}] WARNING: NaN in action, zeroing out")
                action = np.zeros(num_actions, dtype=np.float32)
            action = np.clip(action, -10.0, 10.0)

            target_dof_pos = action * action_scale + DEFAULT_ANGLES_ISAAC
            for i in range(num_actions):
                motor_idx = leg_joint2motor_idx[i]
                low_cmd.motor_cmd[motor_idx].q = float(target_dof_pos[i])
                low_cmd.motor_cmd[motor_idx].dq = 0.0
                low_cmd.motor_cmd[motor_idx].kp = kps[i]
                low_cmd.motor_cmd[motor_idx].kd = kds[i]
                low_cmd.motor_cmd[motor_idx].tau = 0.0
            send_cmd()

            if sim_real_recording:
                recording_buf.append({
                    "t": len(recording_buf) * control_dt,
                    "force_hat": force_hat.tolist(),
                    "force_ema": force_ema.tolist(),
                    "force_mag": float(np.linalg.norm(force_hat[:min(force_dim, 3)])),
                    "velocity_cmd": velocity_cmd.tolist(),
                    "pose_cmd": pose_cmd.tolist(),
                    "compliance_mode": compliance_mode,
                })

            b_now = remote_controller.button[KeyMap.B]
            if b_now == 1 and prev_b_button == 0:
                sim_real_recording = not sim_real_recording
                if sim_real_recording:
                    recording_buf = []
                    print(f"[step {step_count}] *** Recording ON ***", flush=True)
                else:
                    _save_recording(recording_buf, recording_index, args.config, control_dt, force_dim, yaw_idx)
                    recording_index += 1
                    print(f"[step {step_count}] *** Recording OFF — saved segment {recording_index} ({len(recording_buf)} steps) ***", flush=True)
                    recording_buf = []
            prev_b_button = b_now

            y_now = remote_controller.button[KeyMap.Y]
            if y_now == 1 and prev_y_button == 0:
                compliance_mode = "normal" if compliance_mode == "off" else "off"
                print(f"[step {step_count}] *** Compliance mode: {compliance_mode} ***", flush=True)
            prev_y_button = y_now

            x_now = remote_controller.button[KeyMap.X]
            if x_now == 1 and prev_x_button == 0:
                compliance_mode = "normal" if compliance_mode == "inverted" else "inverted"
                print(f"[step {step_count}] *** Compliance mode: {compliance_mode} ***", flush=True)
            prev_x_button = x_now

            step_count += 1
            if args.debug:
                debug_log.append({
                    'step': step_count,
                    'wall_time': time.time(),
                    'compliance_mode': compliance_mode,
                    'raw_obs': raw_obs.copy().tolist(),
                    'force_hat': force_hat.tolist(),
                    'force_ema': force_ema.tolist(),
                    'action': action.tolist(),
                    'target_dof_pos': target_dof_pos.tolist(),
                    'velocity_cmd': velocity_cmd.tolist(),
                    'pose_cmd': pose_cmd.tolist(),
                })

            do_print = (step_count <= 5 or (step_count <= 50 and step_count % 10 == 0) or step_count % 50 == 0)
            if do_print:
                n_lin = min(force_dim, 3)
                f_str = ",".join(f"{force_hat[i]:+.1f}" for i in range(n_lin))
                tq_str = ""
                if force_dim >= 6:
                    tq_str = f"  τ=[{force_hat[3]:+.2f},{force_hat[4]:+.2f},{force_hat[5]:+.2f}]"
                print(f"[step {step_count}] v=[{velocity_cmd[0]:.1f},{velocity_cmd[1]:.1f},{velocity_cmd[2]:.1f}]"
                      f" pose=[r={pose_cmd[0]:+.2f},p={pose_cmd[1]:+.2f},h={pose_cmd[2]:.2f}]"
                      f" F=[{f_str}]{tq_str}"
                      f" mode={compliance_mode}")

            if remote_controller.button[KeyMap.select] == 1:
                should_stop = True

            elapsed = time.perf_counter() - step_start
            if control_dt - elapsed > 0:
                time.sleep(control_dt - elapsed)

    except KeyboardInterrupt:
        print("\nCtrl+C received.")

    if recording_buf:
        _save_recording(recording_buf, recording_index, args.config, control_dt, force_dim, yaw_idx)

    print("[9] Lying down...")
    lie_pos = [low_state.motor_state[i].q for i in range(12)]
    for step in range(300):
        phase = min(step / 300, 1.0)
        for i in range(12):
            low_cmd.motor_cmd[i].q = (1 - phase) * lie_pos[i] + phase * LYING_POS[i]
            low_cmd.motor_cmd[i].kp = 60.0
            low_cmd.motor_cmd[i].kd = 5.0
            low_cmd.motor_cmd[i].dq = 0.0
            low_cmd.motor_cmd[i].tau = 0.0
        send_cmd()
        time.sleep(dt)
    print("    Robot is lying down.")

    if args.debug and len(debug_log) > 10:
        log_dir = Path(__file__).resolve().parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        json_path = log_dir / f"real_6dctrl_debug_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump({
                "source": "real_robot",
                "timestamp": timestamp,
                "control_dt": control_dt,
                "num_steps": len(debug_log),
                "log": debug_log,
            }, f)
        print(f"[debug] JSON saved: {json_path}")

    print("EXIT")
