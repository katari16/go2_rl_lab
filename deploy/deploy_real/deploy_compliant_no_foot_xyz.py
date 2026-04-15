"""
Real-world deployment: Compliant locomotion policy (no foot contacts, XYZ force estimator)
on Unitree Go2 with joystick control.

The policy expects 60-dim obs = 57 raw proprioceptive + 3 force estimate (XYZ).
The force estimator maintains a 20-step history buffer internally.
Forces are detected from proprioception — no explicit force input needed.

Usage:
    python deploy_real/deploy_compliant_no_foot_xyz.py enp3s0 go2_compliant_no_foot_xyz.yaml

    enp3s0 = network interface connected to the robot
    go2_compliant_no_foot_xyz.yaml = config file in deploy_real/configs/

Controls (joystick):
    START  = switch to low-level, stand up
    A      = start policy
    SELECT = stop policy, lie down
    Left stick  = vx (ly), vy (-lx)
    Right stick = wz (-rx)
"""

import argparse
import json
import sys
import time
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).resolve().parents[2]  # go2_rl_lab/
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
    """Project gravity into body frame from quaternion [w, x, y, z]."""
    qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]
    gx = 2 * (-qz * qx + qw * qy)
    gy = -2 * (qz * qy + qw * qx)
    gz = -1 + 2 * (qx * qx + qy * qy)
    return np.array([gx, gy, gz])


def init_cmd_go(cmd, weak_motor=None):
    """Initialize low-level command for Go2."""
    if weak_motor is None:
        weak_motor = []
    cmd.head[0] = 0xFE
    cmd.head[1] = 0xEF
    cmd.level_flag = 0xFF
    cmd.gpio = 0
    for i in range(len(cmd.motor_cmd)):
        if i in weak_motor:
            cmd.motor_cmd[i].mode = 1
        else:
            cmd.motor_cmd[i].mode = 0x0A
        cmd.motor_cmd[i].q = 2.146e9
        cmd.motor_cmd[i].qd = 16000.0
        cmd.motor_cmd[i].kp = 0
        cmd.motor_cmd[i].kd = 0
        cmd.motor_cmd[i].tau = 0

# Global state
low_state = unitree_go_msg_dds__LowState_()


# ── Force Estimator Wrapper (JIT) ─────────────────────────────────────────────

class ForceEstimatorDeployment:
    """Lightweight wrapper around a JIT-exported force estimator."""

    def __init__(self, jit_path: str, raw_obs_dim: int, force_dim: int,
                 temporal_steps: int, device: str = "cpu"):
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

    def get_force_estimate(self, raw_obs: np.ndarray) -> np.ndarray:
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


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy compliant policy on real Go2.")
    parser.add_argument("net", type=str, help="Network interface (e.g. enp3s0)")
    parser.add_argument("config", type=str, help="Config file name in deploy_real/configs/",
                        default="go2_compliant_no_foot_xyz.yaml")
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Save obs/action logs as JSON + PDF after run.")
    args = parser.parse_args()

    # Load config
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
    ema_alpha = cfg.get("ema_alpha", 0.1)
    weak_motor = cfg.get("weak_motor", [])
    force_layout = cfg.get("force_layout", "auto")

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

    # Load JIT models
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

    # DDS setup
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

    # Wait for state
    print("[4] Waiting for robot state...")
    while low_state.tick == 0:
        time.sleep(control_dt)
    print("    Connected to robot.")

    # Initialize low cmd
    init_cmd_go(low_cmd, weak_motor=weak_motor)

    def send_cmd():
        low_cmd.crc = crc.Crc(low_cmd)
        pub.Write(low_cmd)

    # ══════════════════════════════════════════════════════════════════════
    # Switch to low-level mode
    # ══════════════════════════════════════════════════════════════════════
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

    # ══════════════════════════════════════════════════════════════════════
    # FSM STATE 1: ZERO TORQUE
    # ══════════════════════════════════════════════════════════════════════
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

    # ══════════════════════════════════════════════════════════════════════
    # FSM STATE 2: STAND UP
    # ══════════════════════════════════════════════════════════════════════
    print("[6] Standing up...")
    dt = 0.002
    start_pos = [low_state.motor_state[i].q for i in range(12)]

    # Phase 1: current -> crouch
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

    # Phase 2: crouch -> standing
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

    # Phase 3: hold
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

    # ══════════════════════════════════════════════════════════════════════
    # FSM STATE 3: WAIT FOR A (hold at stiff gains so robot doesn't collapse)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  STANDING — Press A to start policy")
    print("  Controls: B=toggle recording, Y=compliance off/on, X=compliance inverted/normal, SELECT=stop")
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

    # ══════════════════════════════════════════════════════════════════════
    # FSM STATE 4: RUN POLICY (50 Hz)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print(f"  POLICY RUNNING (50 Hz) — {raw_obs_dim}+{force_dim}={policy_obs_dim} obs dims")
    print(f"  Force estimator: {temporal_steps}x{raw_obs_dim} -> {force_dim}D force")
    if compliance_k > 0:
        print(f"  Compliance: k={compliance_k:.4f}  EMA alpha={ema_alpha}")
    print("  Press SELECT to stop")
    print("=" * 60 + "\n")

    action = np.zeros(num_actions, dtype=np.float32)
    raw_obs = np.zeros(raw_obs_dim, dtype=np.float32)
    velocity_cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    force_ema = np.zeros(force_dim, dtype=np.float32)
    step_count = 0
    debug_log = []
    should_stop = False
    compliance_mode = "normal"
    sim_real_recording = False
    prev_b_button = 0
    prev_y_button = 0
    prev_x_button = 0
    estimator.reset()

    try:
        while not should_stop:
            step_start = time.perf_counter()

            # ── Velocity commands from joystick ───────────────────────
            velocity_cmd[0] = round(remote_controller.ly, 1)
            velocity_cmd[1] = round(remote_controller.lx * -1, 1)
            velocity_cmd[2] = round(remote_controller.rx * -1, 1)

            # ── Build raw observation (57 dims, NO foot contacts) ─────
            # [0:3] base_ang_vel
            raw_obs[0] = low_state.imu_state.gyroscope[0]
            raw_obs[1] = low_state.imu_state.gyroscope[1]
            raw_obs[2] = low_state.imu_state.gyroscope[2]

            # [3:6] projected_gravity
            raw_obs[3:6] = get_gravity_orientation(low_state.imu_state.quaternion)

            # [6:9] velocity_commands
            raw_obs[6:9] = velocity_cmd * cmd_scale * max_cmd

            # [9:21] joint_pos_rel (Isaac convention)
            for i in range(num_actions):
                motor_idx = leg_joint2motor_idx[i]
                raw_obs[9 + i] = low_state.motor_state[motor_idx].q - DEFAULT_ANGLES_ISAAC[i]

            # [21:33] joint_vel_rel (Isaac convention)
            for i in range(num_actions):
                motor_idx = leg_joint2motor_idx[i]
                raw_obs[21 + i] = low_state.motor_state[motor_idx].dq

            # [33:45] last_action
            raw_obs[33:45] = action

            # [45:57] applied_torque * 0.1
            for i in range(num_actions):
                motor_idx = leg_joint2motor_idx[i]
                raw_obs[45 + i] = low_state.motor_state[motor_idx].tau_est * 0.1

            # ── Run force estimator ───────────────────────────────────
            force_hat = estimator.get_force_estimate(raw_obs)

            # EMA filter
            force_ema = ema_alpha * force_hat + (1.0 - ema_alpha) * force_ema

            # ── Compliance modulation (XY force + yaw torque) ────────
            obs_for_policy = raw_obs.copy()
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

            # ── Build full policy input (57 raw + force estimate) ─────
            if compliance_mode == "inverted":
                full_obs = np.concatenate([obs_for_policy, -force_hat])
            else:
                full_obs = np.concatenate([obs_for_policy, force_hat])

            # ── Policy inference ──────────────────────────────────────
            obs_tensor = torch.from_numpy(full_obs).float().unsqueeze(0)
            with torch.inference_mode():
                action_tensor = policy(obs_tensor)
            action = action_tensor.squeeze(0).numpy()

            # NaN safety
            if np.any(np.isnan(action)):
                print(f"[step {step_count}] WARNING: NaN in action, zeroing out")
                action = np.zeros(num_actions, dtype=np.float32)

            action = np.clip(action, -10.0, 10.0)

            # ── Apply action ──────────────────────────────────────────
            target_dof_pos = action * action_scale + DEFAULT_ANGLES_ISAAC

            for i in range(num_actions):
                motor_idx = leg_joint2motor_idx[i]
                low_cmd.motor_cmd[motor_idx].q = float(target_dof_pos[i])
                low_cmd.motor_cmd[motor_idx].dq = 0.0
                low_cmd.motor_cmd[motor_idx].kp = kps[i]
                low_cmd.motor_cmd[motor_idx].kd = kds[i]
                low_cmd.motor_cmd[motor_idx].tau = 0.0

            send_cmd()

            # ── B button: toggle recording marker ─────────────────────
            b_now = remote_controller.button[KeyMap.B]
            if b_now == 1 and prev_b_button == 0:
                sim_real_recording = not sim_real_recording
                tag = "ON" if sim_real_recording else "OFF"
                print(f"[step {step_count}] *** Recording marker {tag} ***", flush=True)
            prev_b_button = b_now

            # ── Y button: toggle compliance OFF / normal ──────────────
            y_now = remote_controller.button[KeyMap.Y]
            if y_now == 1 and prev_y_button == 0:
                if compliance_mode == "off":
                    compliance_mode = "normal"
                else:
                    compliance_mode = "off"
                print(f"[step {step_count}] *** Compliance mode: {compliance_mode} ***", flush=True)
            prev_y_button = y_now

            # ── X button: toggle compliance INVERTED / normal ─────────
            x_now = remote_controller.button[KeyMap.X]
            if x_now == 1 and prev_x_button == 0:
                if compliance_mode == "inverted":
                    compliance_mode = "normal"
                else:
                    compliance_mode = "inverted"
                print(f"[step {step_count}] *** Compliance mode: {compliance_mode} ***", flush=True)
            prev_x_button = x_now

            # ── Debug logging ─────────────────────────────────────────
            step_count += 1
            if args.debug:
                debug_log.append({
                    'step': step_count,
                    'wall_time': time.time(),
                    'sim_real_recording': sim_real_recording,
                    'compliance_mode': compliance_mode,
                    'raw_obs': raw_obs.copy().tolist(),
                    'force_hat': force_hat.tolist(),
                    'force_ema': force_ema.tolist(),
                    'action': action.tolist(),
                    'target_dof_pos': target_dof_pos.tolist(),
                    'velocity_cmd': velocity_cmd.tolist(),
                })

            do_print = (step_count <= 5
                        or (step_count <= 50 and step_count % 10 == 0)
                        or step_count % 50 == 0)
            if do_print:
                n_lin = 2 if force_layout == "xy_yaw" else min(force_dim, 3)
                f_str = ",".join(f"{force_hat[i]:+.1f}" for i in range(n_lin))
                extra = f"  τ_yaw={force_hat[yaw_idx]:+.2f}" if yaw_idx is not None else ""
                print(f"[step {step_count}] cmd=[{velocity_cmd[0]:.1f},{velocity_cmd[1]:.1f},{velocity_cmd[2]:.1f}]"
                      f"  F_hat=[{f_str}]  |F|={np.linalg.norm(force_hat[:n_lin]):.1f}N{extra}"
                      f"  gravity={raw_obs[3:6].round(3)}"
                      f"  action_norm={np.linalg.norm(action):.3f}")

            # ── Check stop ────────────────────────────────────────────
            if remote_controller.button[KeyMap.select] == 1:
                should_stop = True

            elapsed = time.perf_counter() - step_start
            if control_dt - elapsed > 0:
                time.sleep(control_dt - elapsed)

    except KeyboardInterrupt:
        print("\nCtrl+C received.")

    # ══════════════════════════════════════════════════════════════════════
    # FSM STATE 5: LIE DOWN
    # ══════════════════════════════════════════════════════════════════════
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

    # ── Save debug logs ───────────────────────────────────────────────────
    if args.debug and len(debug_log) > 10:
        log_dir = Path(__file__).resolve().parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        raw_obs_arr = np.array([s['raw_obs'] for s in debug_log])
        actions_arr = np.array([s['action'] for s in debug_log])
        force_hat_arr = np.array([s['force_hat'] for s in debug_log])
        force_ema_arr = np.array([s['force_ema'] for s in debug_log])
        velocity_cmd_arr = np.array([s['velocity_cmd'] for s in debug_log])
        target_dof_arr = np.array([s['target_dof_pos'] for s in debug_log])
        N = len(debug_log)
        timestamps = (np.arange(N) * control_dt).tolist()

        obs_labels = (
            ["ang_vel_x", "ang_vel_y", "ang_vel_z"]
            + ["grav_x", "grav_y", "grav_z"]
            + ["cmd_vx", "cmd_vy", "cmd_wz"]
            + [f"jpos_{i}" for i in range(12)]
            + [f"jvel_{i}" for i in range(12)]
            + [f"last_act_{i}" for i in range(12)]
            + [f"torque_{i}" for i in range(12)]
        )

        log_json = {
            "source": "real_robot",
            "timestamp": timestamp,
            "control_dt": control_dt,
            "num_steps": N,
            "compliance_k": compliance_k,
            "ema_alpha": ema_alpha,
            "obs_labels": obs_labels,
            "time_s": timestamps,
            "raw_obs": raw_obs_arr.tolist(),
            "actions": actions_arr.tolist(),
            "force_hat": force_hat_arr.tolist(),
            "force_ema": force_ema_arr.tolist(),
            "velocity_cmd": velocity_cmd_arr.tolist(),
            "target_dof_pos": target_dof_arr.tolist(),
        }
        json_path = log_dir / f"real_deploy_debug_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump(log_json, f)
        print(f"[debug] JSON saved: {json_path}")

        # Generate PDF
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        pdf_path = log_dir / f"real_deploy_debug_{timestamp}.pdf"
        t = np.array(timestamps)

        with PdfPages(str(pdf_path)) as pdf:
            obs_groups = [
                ("Angular Velocity", [0, 1, 2], ["x", "y", "z"]),
                ("Projected Gravity", [3, 4, 5], ["x", "y", "z"]),
                ("Velocity Command", [6, 7, 8], ["vx", "vy", "wz"]),
                ("Joint Positions (rel)", list(range(9, 21)), [f"j{i}" for i in range(12)]),
                ("Joint Velocities (rel)", list(range(21, 33)), [f"j{i}" for i in range(12)]),
                ("Last Action", list(range(33, 45)), [f"j{i}" for i in range(12)]),
                ("Applied Torque (x0.1)", list(range(45, 57)), [f"j{i}" for i in range(12)]),
            ]
            for title, idxs, labels in obs_groups:
                fig, ax = plt.subplots(figsize=(14, 4))
                for i, idx in enumerate(idxs):
                    ax.plot(t, raw_obs_arr[:, idx], linewidth=0.6, alpha=0.8, label=labels[i])
                ax.set_title(f"Obs: {title}")
                ax.set_xlabel("Time (s)")
                ax.legend(loc="upper right", fontsize=7, ncol=min(len(idxs), 6))
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

            fig, ax = plt.subplots(figsize=(14, 5))
            for i in range(num_actions):
                ax.plot(t, actions_arr[:, i], linewidth=0.6, alpha=0.8, label=f"a{i}")
            ax.set_title("Actions")
            ax.set_xlabel("Time (s)")
            ax.legend(loc="upper right", fontsize=7, ncol=6)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(14, 4))
            if force_layout == "xy_yaw":
                force_labels = ["Fx", "Fy", "τ_yaw"]
            elif force_dim == 6:
                force_labels = ["Fx", "Fy", "Fz", "τx", "τy", "τz"]
            elif force_dim == 4:
                force_labels = ["Fx", "Fy", "Fz", "τ_yaw"]
            else:
                force_labels = ["Fx", "Fy", "Fz"][:force_dim]
            for i in range(force_dim):
                ax.plot(t, force_hat_arr[:, i], linewidth=0.8, alpha=0.7, label=f"hat {force_labels[i]}")
                ax.plot(t, force_ema_arr[:, i], linewidth=1.2, alpha=0.9, linestyle="--", label=f"ema {force_labels[i]}")
            ax.set_title("Force Estimate (raw vs EMA)")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Force (N)")
            ax.legend(loc="upper right", fontsize=9)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        print(f"[debug] PDF saved: {pdf_path}")

    print("EXIT")
