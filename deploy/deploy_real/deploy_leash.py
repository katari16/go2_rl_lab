"""
Real-world deployment with align-then-follow leash shaping.

Same as deploy_compliant_no_foot_xyz.py but adds a "leash" deploy-time
command shaping (see docs/deploy_command_shaping.md):

    phi = atan2(F_ema_y, F_ema_x)      # base-frame force azimuth
    |F_xy| > F_min  + |phi| > phi_enter -> ALIGN: wz = k_yaw_align * phi, vx=vy=0
    else                                -> FOLLOW: existing compliance mapping

Leash mode is DISABLED by default. Press R1 to toggle on/off at any time.

Usage:
    python deploy_real/deploy_leash.py enp6s0 go2_ablation_p3.yaml

Controls (joystick):
    START  = switch to low-level, stand up
    A      = start policy
    SELECT = stop policy, lie down
    Left stick  = vx (ly), vy (-lx)
    Right stick = wz (-rx)
    B      = toggle recording
    Y      = compliance off / normal
    X      = compliance inverted / normal
    R1     = LEASH off / on  (align-then-follow command shaping)
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))
sys.path.append(str(project_root / "unitree_sdk2_python"))

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.go2.sport.sport_client import SportClient

from common.remote_controller import RemoteController, KeyMap


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


low_state = None


class ForceEstimatorDeployment:
    def __init__(self, jit_path, raw_obs_dim, force_dim, temporal_steps, device="cpu"):
        self.device = torch.device(device)
        self.raw_obs_dim = raw_obs_dim
        self.force_dim = force_dim
        self.temporal_steps = temporal_steps
        self.estimator = torch.jit.load(jit_path, map_location=self.device)
        self.estimator.eval()
        print(f"[Estimator] Loaded JIT model from {jit_path}")
        print(f"  Input: [{temporal_steps * raw_obs_dim}]   Output: [{force_dim}]")
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
    stem = config_name.replace(".yaml", "") + "_leash"
    base = log_dir / f"{stem}_rec{index:02d}_{ts}"

    with open(str(base) + ".json", "w") as f:
        json.dump({
            "config": config_name,
            "script": "deploy_leash",
            "control_dt": control_dt,
            "force_dim": force_dim,
            "has_yaw": yaw_idx is not None,
            "num_steps": len(buf),
            "steps": buf,
        }, f, indent=2)

    force_labels = [f"F{ax}_hat" for ax in ["x", "y", "z"][:min(force_dim, 3)]]
    if yaw_idx is not None:
        force_labels += ["tau_yaw_hat", "tau_yaw_ema"]
    fieldnames = ["t", *force_labels, "force_mag", "force_mag_ema", "wz_obs",
                  "vx_cmd", "vy_cmd", "wz_cmd", "compliance_mode",
                  "leash_enabled", "leash_state", "phi_deg"]
    with open(str(base) + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in buf:
            r = {
                "t": row["t"], "force_mag": row["force_mag"],
                "force_mag_ema": row["force_mag_ema"], "wz_obs": row["wz_obs"],
                "vx_cmd": row["velocity_cmd"][0], "vy_cmd": row["velocity_cmd"][1],
                "wz_cmd": row["velocity_cmd"][2],
                "compliance_mode": row["compliance_mode"],
                "leash_enabled": row["leash_enabled"],
                "leash_state": row["leash_state"],
                "phi_deg": row["phi_deg"],
            }
            for i, lbl in enumerate([f"F{ax}_hat" for ax in ["x", "y", "z"][:min(force_dim, 3)]]):
                r[lbl] = row["force_hat"][i]
            if yaw_idx is not None:
                r["tau_yaw_hat"] = row["tau_yaw_hat"]
                r["tau_yaw_ema"] = row["tau_yaw_ema"]
            w.writerow(r)

    print(f"[recording] Saved {len(buf)} steps → {base}.json / .csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy compliant policy + leash shaping on Go2.")
    parser.add_argument("net", type=str)
    parser.add_argument("config", type=str)
    parser.add_argument("--debug", action="store_true", default=False)
    # Leash parameters (see docs/deploy_command_shaping.md defaults)
    parser.add_argument("--phi_enter_deg", type=float, default=30.0)
    parser.add_argument("--phi_exit_deg", type=float, default=20.0)
    parser.add_argument("--f_min", type=float, default=5.0, help="Magnitude floor (N) to engage leash.")
    parser.add_argument("--k_yaw_align", type=float, default=1.2, help="wz = k_yaw_align * phi (rad/s per rad).")
    parser.add_argument("--wz_clip", type=float, default=0.7, help="Saturation on leash-generated wz (rad/s).")
    parser.add_argument("--leash_on", action="store_true",
                        help="Start with leash ENABLED (default: disabled; toggle with R1).")
    args = parser.parse_args()

    phi_enter = np.deg2rad(args.phi_enter_deg)
    phi_exit = np.deg2rad(args.phi_exit_deg)

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
        [0.1, -0.1, 0.1, -0.1, 0.8, 0.8, 1, 1, -1.5, -1.5, -1.5, -1.5], dtype=np.float32,
    )
    LYING_POS = [0.0, 1.36, -2.65, 0.0, 1.36, -2.65, -0.2, 1.36, -2.65, 0.2, 1.36, -2.65]

    print(f"[1] Config loaded: {config_path}")
    print(f"[1] Leash params: phi_enter={args.phi_enter_deg}°  phi_exit={args.phi_exit_deg}°  "
          f"F_min={args.f_min:.1f}N  k_yaw_align={args.k_yaw_align:.2f}  wz_clip={args.wz_clip:.2f}")

    policy_path = str(project_root / cfg["policy_path"])
    estimator_path = str(project_root / cfg["estimator_path"])

    print(f"[2] Loading policy from {policy_path}")
    policy = torch.jit.load(policy_path, map_location="cpu")
    policy.eval()

    estimator = ForceEstimatorDeployment(
        jit_path=estimator_path, raw_obs_dim=raw_obs_dim, force_dim=force_dim,
        temporal_steps=temporal_steps, device="cpu",
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
    while low_state is None or low_state.tick == 0:
        time.sleep(control_dt)
    print("    Connected to robot.")

    init_cmd_go(low_cmd, weak_motor=weak_motor)

    def send_cmd():
        low_cmd.crc = crc.Crc(low_cmd)
        pub.Write(low_cmd)

    print("[5] Switching to low-level control...")
    sc = SportClient(); sc.SetTimeout(5.0); sc.Init()
    msc = MotionSwitcherClient(); msc.SetTimeout(5.0); msc.Init()
    status, result = msc.CheckMode()
    while result is not None and result.get('name'):
        sc.StandDown(); msc.ReleaseMode()
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
    dt_hold = 0.002
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
        time.sleep(dt_hold)
    for step in range(500):
        phase = min(step / 500, 1.0)
        for i in range(12):
            low_cmd.motor_cmd[i].q = (1 - phase) * LYING_POS[i] + phase * default_angles_sdk[i]
            low_cmd.motor_cmd[i].kp = 60.0
            low_cmd.motor_cmd[i].kd = 5.0
            low_cmd.motor_cmd[i].dq = 0.0
            low_cmd.motor_cmd[i].tau = 0.0
        send_cmd()
        time.sleep(dt_hold)
    for step in range(500):
        for i in range(12):
            low_cmd.motor_cmd[i].q = default_angles_sdk[i]
            low_cmd.motor_cmd[i].kp = 60.0
            low_cmd.motor_cmd[i].kd = 5.0
            low_cmd.motor_cmd[i].dq = 0.0
            low_cmd.motor_cmd[i].tau = 0.0
        send_cmd()
        time.sleep(dt_hold)

    gravity_check = get_gravity_orientation(low_state.imu_state.quaternion)
    print(f"    Standing. Gravity={gravity_check.round(3)}, Kp={kps[0]}, Kd={kds[0]}")

    print("\n" + "=" * 60)
    print("  STANDING — Press A to start policy")
    print("  Controls: R1=leash on/off  B=record  Y=compl off/on  X=compl inv/norm  SELECT=stop")
    print("=" * 60 + "\n")
    while remote_controller.button[KeyMap.A] != 1:
        for i in range(12):
            low_cmd.motor_cmd[i].q = default_angles_sdk[i]
            low_cmd.motor_cmd[i].kp = 60.0
            low_cmd.motor_cmd[i].kd = 5.0
            low_cmd.motor_cmd[i].dq = 0.0
            low_cmd.motor_cmd[i].tau = 0.0
        send_cmd()
        time.sleep(dt_hold)

    print("\n" + "=" * 60)
    print(f"  POLICY RUNNING (50 Hz) — {raw_obs_dim}+{force_dim}={policy_obs_dim} obs dims")
    print(f"  Leash starts: {'ON' if args.leash_on else 'OFF'} (toggle with R1)")
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
    prev_r1_button = 0
    recording_buf = []
    recording_index = 0

    # Leash state
    leash_enabled = bool(args.leash_on)
    leash_state = "follow"

    estimator.reset()

    try:
        while not should_stop:
            step_start = time.perf_counter()

            # ── Joystick velocity command (user input) ────────────────
            velocity_cmd[0] = round(remote_controller.ly, 1)
            velocity_cmd[1] = round(remote_controller.lx * -1, 1)
            velocity_cmd[2] = round(remote_controller.rx * -1, 1)

            # ── Raw obs ───────────────────────────────────────────────
            raw_obs[0] = low_state.imu_state.gyroscope[0]
            raw_obs[1] = low_state.imu_state.gyroscope[1]
            raw_obs[2] = low_state.imu_state.gyroscope[2]
            raw_obs[3:6] = get_gravity_orientation(low_state.imu_state.quaternion)
            raw_obs[6:9] = velocity_cmd * cmd_scale * max_cmd
            for i in range(num_actions):
                m = leg_joint2motor_idx[i]
                raw_obs[9 + i] = low_state.motor_state[m].q - DEFAULT_ANGLES_ISAAC[i]
            for i in range(num_actions):
                m = leg_joint2motor_idx[i]
                raw_obs[21 + i] = low_state.motor_state[m].dq
            raw_obs[33:45] = action
            for i in range(num_actions):
                m = leg_joint2motor_idx[i]
                raw_obs[45 + i] = low_state.motor_state[m].tau_est * 0.1

            # ── Estimator + EMA ───────────────────────────────────────
            force_hat = estimator.get_force_estimate(raw_obs)
            force_ema = ema_alpha * force_hat + (1.0 - ema_alpha) * force_ema

            # ── Leash FSM (uses EMA force, base frame) ────────────────
            Fxy_ema = force_ema[:2]
            F_mag_ema = float(np.linalg.norm(Fxy_ema))
            phi = float(np.arctan2(Fxy_ema[1], Fxy_ema[0]))

            if leash_enabled:
                if F_mag_ema < args.f_min:
                    leash_state = "follow"
                elif leash_state == "follow" and abs(phi) > phi_enter:
                    leash_state = "align"
                elif leash_state == "align" and abs(phi) < phi_exit:
                    leash_state = "follow"
            else:
                leash_state = "follow"

            # ── Compliance + leash command shaping ────────────────────
            obs_for_policy = raw_obs.copy()

            if leash_enabled and leash_state == "align":
                # Override velocity commands: zero linear, yaw to align.
                wz_align = float(np.clip(args.k_yaw_align * phi, -args.wz_clip, args.wz_clip))
                obs_for_policy[6] = 0.0
                obs_for_policy[7] = 0.0
                obs_for_policy[8] = wz_align
            else:
                # Existing compliance mapping (unchanged).
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
                m = leg_joint2motor_idx[i]
                low_cmd.motor_cmd[m].q = float(target_dof_pos[i])
                low_cmd.motor_cmd[m].dq = 0.0
                low_cmd.motor_cmd[m].kp = kps[i]
                low_cmd.motor_cmd[m].kd = kds[i]
                low_cmd.motor_cmd[m].tau = 0.0
            send_cmd()

            # ── Recording buffer ──────────────────────────────────────
            if sim_real_recording:
                recording_buf.append({
                    "t": len(recording_buf) * control_dt,
                    "force_hat": force_hat.tolist(),
                    "force_ema": force_ema.tolist(),
                    "force_mag": float(np.linalg.norm(force_hat[:min(force_dim, 3)])),
                    "force_mag_ema": F_mag_ema,
                    "tau_yaw_hat": float(force_hat[yaw_idx]) if yaw_idx is not None else 0.0,
                    "tau_yaw_ema": float(force_ema[yaw_idx]) if yaw_idx is not None else 0.0,
                    "wz_obs": float(obs_for_policy[8]),
                    "velocity_cmd": velocity_cmd.tolist(),
                    "compliance_mode": compliance_mode,
                    "leash_enabled": bool(leash_enabled),
                    "leash_state": leash_state,
                    "phi_deg": float(np.rad2deg(phi)),
                })

            # ── Button handling ───────────────────────────────────────
            b_now = remote_controller.button[KeyMap.B]
            if b_now == 1 and prev_b_button == 0:
                sim_real_recording = not sim_real_recording
                if sim_real_recording:
                    recording_buf = []
                    print(f"[step {step_count}] *** Recording ON ***", flush=True)
                else:
                    _save_recording(recording_buf, recording_index, args.config,
                                    control_dt, force_dim, yaw_idx)
                    recording_index += 1
                    print(f"[step {step_count}] *** Recording OFF — saved segment {recording_index} "
                          f"({len(recording_buf)} steps) ***", flush=True)
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

            r1_now = remote_controller.button[KeyMap.R1]
            if r1_now == 1 and prev_r1_button == 0:
                leash_enabled = not leash_enabled
                if not leash_enabled:
                    leash_state = "follow"
                print(f"[step {step_count}] *** LEASH: {'ON' if leash_enabled else 'OFF'} ***", flush=True)
            prev_r1_button = r1_now

            step_count += 1

            if args.debug:
                debug_log.append({
                    "step": step_count, "wall_time": time.time(),
                    "sim_real_recording": sim_real_recording,
                    "compliance_mode": compliance_mode,
                    "leash_enabled": bool(leash_enabled),
                    "leash_state": leash_state,
                    "phi_deg": float(np.rad2deg(phi)),
                    "raw_obs": raw_obs.copy().tolist(),
                    "force_hat": force_hat.tolist(),
                    "force_ema": force_ema.tolist(),
                    "action": action.tolist(),
                    "target_dof_pos": target_dof_pos.tolist(),
                    "velocity_cmd": velocity_cmd.tolist(),
                })

            do_print = (step_count <= 5
                        or (step_count <= 50 and step_count % 10 == 0)
                        or step_count % 50 == 0)
            if do_print:
                n_lin = 2 if force_layout == "xy_yaw" else min(force_dim, 3)
                f_str = ",".join(f"{force_hat[i]:+.1f}" for i in range(n_lin))
                leash_str = (f"  LEASH={leash_state.upper()} phi={np.rad2deg(phi):+.0f}°"
                             if leash_enabled else "  leash=off")
                print(f"[step {step_count}] cmd=[{velocity_cmd[0]:.1f},{velocity_cmd[1]:.1f},{velocity_cmd[2]:.1f}]"
                      f"  F_hat=[{f_str}]  |F_ema|={F_mag_ema:.1f}N{leash_str}"
                      f"  wz_obs={obs_for_policy[8]:+.3f}  mode={compliance_mode}")

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
        time.sleep(dt_hold)
    print("    Robot is lying down.")

    if args.debug and len(debug_log) > 10:
        log_dir = Path(__file__).resolve().parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        json_path = log_dir / f"real_deploy_leash_debug_{ts}.json"
        with open(json_path, "w") as f:
            json.dump({
                "source": "real_robot", "script": "deploy_leash",
                "timestamp": ts, "control_dt": control_dt,
                "num_steps": len(debug_log),
                "leash_params": {
                    "phi_enter_deg": args.phi_enter_deg,
                    "phi_exit_deg": args.phi_exit_deg,
                    "f_min": args.f_min,
                    "k_yaw_align": args.k_yaw_align,
                    "wz_clip": args.wz_clip,
                },
                "steps": debug_log,
            }, f)
        print(f"[debug] JSON saved: {json_path}")

    print("EXIT")
