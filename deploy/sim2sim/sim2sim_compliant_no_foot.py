"""
Sim2Sim deployment for compliant locomotion policy (no foot contacts).

Runs the trained compliant policy + force estimator in unitree_mujoco with
joystick/keyboard control and compliance-based velocity modulation.

The force estimator processes proprioceptive history (20x57=1140 dims) and
outputs a 2D XY force estimate. The compliance mapping modulates the velocity
command: v* = v_cmd + k(f) * EMA(f_hat), where k is piecewise:
  k = 0         if |f_hat| < alpha  (deadzone)
  k = 1/beta    if |f_hat| >= alpha

Usage:
  1. Start unitree_mujoco:  cd ~/unitree_mujoco/simulate_python && python3 unitree_mujoco.py
  2. Run this script:       python3 sim2sim_compliant_no_foot.py [--checkpoint PATH]

Controls (keyboard):
  Enter  = START (begin stand-up)
  Space  = A     (start policy)
  Esc    = SELECT (stop policy, lie down)
  W/S    = forward/backward
  A/D    = strafe left/right
  Q/E    = turn left/right

Controls (joystick via unitree_mujoco bridge):
  START  = begin stand-up
  A      = start policy
  SELECT = stop policy
  Left stick  = vx (ly), vy (-lx)
  Right stick = wz (-rx)
"""

import struct
import sys
import time
import numpy as np
import torch
import yaml
import threading
from pathlib import Path
from datetime import datetime

# ── Try to import pynput for keyboard control ─────────────────────────────────
try:
    from pynput import keyboard as pynput_keyboard
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False
    print("WARNING: pynput not installed. Keyboard control disabled (joystick only).")
    print("         Install with: pip install pynput")

from unitree_sdk2py.core.channel import (
    ChannelPublisher,
    ChannelSubscriber,
    ChannelFactoryInitialize,
)
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_, SportModeState_
from unitree_sdk2py.utils.crc import CRC


# ── Self-contained helpers ────────────────────────────────────────────────────

def get_gravity_orientation(quat):
    """Project gravity into body frame from quaternion [w, x, y, z]."""
    qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]
    gx = 2 * (-qz * qx + qw * qy)
    gy = -2 * (qz * qy + qw * qx)
    gz = -1 + 2 * (qx * qx + qy * qy)
    return np.array([gx, gy, gz])


class KeyMap:
    R1 = 0
    L1 = 1
    start = 2
    select = 3
    R2 = 4
    L2 = 5
    F1 = 6
    F2 = 7
    A = 8
    B = 9
    X = 10
    Y = 11
    up = 12
    right = 13
    down = 14
    left = 15


class RemoteController:
    def __init__(self):
        self.lx = 0.0
        self.ly = 0.0
        self.rx = 0.0
        self.ry = 0.0
        self.button = [0] * 16

    def set(self, data):
        keys = struct.unpack("H", data[2:4])[0]
        for i in range(16):
            self.button[i] = (keys & (1 << i)) >> i
        self.lx = struct.unpack("f", data[4:8])[0]
        self.rx = struct.unpack("f", data[8:12])[0]
        self.ry = struct.unpack("f", data[12:16])[0]
        self.ly = struct.unpack("f", data[20:24])[0]


class KeyboardController:
    """Non-blocking keyboard input using pynput."""

    def __init__(self):
        self.pressed_keys = set()
        self._lock = threading.Lock()
        self._listener = pynput_keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.daemon = True
        self._listener.start()

    def _on_press(self, key):
        with self._lock:
            try:
                self.pressed_keys.add(key.char.lower())
            except AttributeError:
                self.pressed_keys.add(key)

    def _on_release(self, key):
        with self._lock:
            try:
                self.pressed_keys.discard(key.char.lower())
            except AttributeError:
                self.pressed_keys.discard(key)

    def is_pressed(self, key):
        with self._lock:
            return key in self.pressed_keys

    def is_char_pressed(self, char):
        with self._lock:
            return char.lower() in self.pressed_keys

    @property
    def start_pressed(self):
        return self.is_pressed(pynput_keyboard.Key.enter)

    @property
    def a_pressed(self):
        return self.is_pressed(pynput_keyboard.Key.space)

    @property
    def select_pressed(self):
        return self.is_pressed(pynput_keyboard.Key.esc)

    @property
    def l2_pressed(self):
        return self.is_pressed(pynput_keyboard.Key.shift_l) or self.is_pressed(pynput_keyboard.Key.shift)

    def get_velocity_commands(self):
        """Return (vx, vy, wz) from WASD + QE keys."""
        vx = vy = wz = 0.0
        if self.is_char_pressed('w'):
            vx += 1.0
        if self.is_char_pressed('s'):
            vx -= 1.0
        if self.is_char_pressed('a'):
            vy += 1.0
        if self.is_char_pressed('d'):
            vy -= 1.0
        if self.is_char_pressed('q'):
            wz += 1.0
        if self.is_char_pressed('e'):
            wz -= 1.0
        return vx, vy, wz

    def stop(self):
        self._listener.stop()


def load_config(path):
    with open(path, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


# ── Observation History Buffer (numpy version for deployment) ─────────────────

class ObsHistoryBufferNp:
    """Rolling observation history buffer (numpy, single env)."""

    def __init__(self, temporal_steps: int, obs_dim: int):
        self.temporal_steps = temporal_steps
        self.obs_dim = obs_dim
        self.buffer = np.zeros((temporal_steps, obs_dim), dtype=np.float32)

    def insert(self, obs: np.ndarray):
        self.buffer = np.roll(self.buffer, shift=-1, axis=0)
        self.buffer[-1, :] = obs

    def reset(self):
        self.buffer[:] = 0.0

    def get_flattened(self) -> np.ndarray:
        return self.buffer.reshape(-1)


# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_ANGLES_ISAAC = np.array(
    [0.1, -0.1, 0.1, -0.1, 0.8, 0.8, 1, 1, -1.5, -1.5, -1.5, -1.5],
    dtype=np.float32,
)

LYING_POS = [0.0, 1.36, -2.65, 0.0, 1.36, -2.65, -0.2, 1.36, -2.65, 0.2, 1.36, -2.65]

STANDUP_KP = 60.0
STANDUP_KD = 5.0


# ── Global state ──────────────────────────────────────────────────────────────
low_state = None
sport_state = None
remote_controller = RemoteController()


def low_state_handler(msg):
    global low_state
    low_state = msg
    remote_controller.set(msg.wireless_remote)


def sport_state_handler(msg):
    global sport_state
    sport_state = msg


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sim2Sim compliant deployment (no foot contacts)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (.pt) containing policy + estimator weights")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config (default: deploy_real/configs/go2_compliant_no_foot.yaml)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]

    # Load config
    if args.config:
        cfg_path = Path(args.config)
    else:
        cfg_path = project_root / "deploy_real" / "configs" / "go2_compliant_no_foot.yaml"
    cfg = load_config(cfg_path)
    print(f"Config: {cfg_path}")

    num_actions = cfg["num_actions"]
    num_obs = cfg["num_obs"]
    num_raw_obs = cfg["num_raw_obs"]
    action_scale = cfg["action_scale"]
    cmd_scale = np.array(cfg["cmd_scale"], dtype=np.float32)
    max_cmd = np.array(cfg["max_cmd"], dtype=np.float32)
    kps = cfg["kps"]
    kds = cfg["kds"]
    leg_joint2motor_idx = cfg["leg_joint2motor_idx"]
    default_angles_sdk = np.array(cfg["default_angles"], dtype=np.float32)
    control_dt = cfg["control_dt"]
    torque_scale = cfg.get("torque_scale", 0.1)

    # Estimator config
    est_cfg = cfg.get("estimator", {})
    temporal_steps = est_cfg.get("temporal_steps", 20)
    force_dim = est_cfg.get("force_dim", 2)

    # Compliance config
    comp_cfg = cfg.get("compliance", {})
    compliance_alpha = comp_cfg.get("alpha", 5.0)
    compliance_beta = comp_cfg.get("beta", 50.0)
    ema_alpha = comp_cfg.get("ema_alpha", 0.1)
    compliance_k = 1.0 / compliance_beta

    # ── Load checkpoint ───────────────────────────────────────────────────
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        ckpt_path = project_root / "pre_train" / cfg["policy_path"]
    print(f"Loading checkpoint from {ckpt_path}")

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    # Load JIT policy
    if "model_state_dict" in ckpt:
        # Non-JIT checkpoint — need to build the policy and load weights
        # For sim2sim we expect a JIT-traced policy. If you have a non-JIT
        # checkpoint, export it first with torch.jit.trace.
        print("ERROR: Non-JIT checkpoint detected. Please export a JIT-traced policy first.")
        print("       Use: torch.jit.save(torch.jit.trace(policy, dummy_input), 'policy.pt')")
        sys.exit(1)

    # Try loading as JIT model directly
    try:
        policy = torch.jit.load(str(ckpt_path))
        policy.eval()
        print("Loaded JIT policy.")
    except Exception:
        print("ERROR: Could not load policy. Ensure checkpoint is a JIT-traced model.")
        sys.exit(1)

    # Load force estimator from the same checkpoint
    from go2_rl_lab.estimator.force_estimator import ForceEstimator

    estimator = ForceEstimator(
        temporal_steps=temporal_steps,
        num_one_step_obs=num_raw_obs,
        enc_hidden_dims=est_cfg.get("enc_hidden_dims", [128, 64]),
        f_head_dims=est_cfg.get("f_head_dims", [32, 16]),
        force_dim=force_dim,
        dec_hidden_dims=est_cfg.get("dec_hidden_dims", [256, 128]),
        activation=est_cfg.get("activation", "elu"),
    )

    # Load estimator weights from checkpoint
    ckpt_for_est = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if "force_estimator_state_dict" in ckpt_for_est:
        estimator.load_state_dict(ckpt_for_est["force_estimator_state_dict"])
        print("Loaded force estimator weights from checkpoint.")
    else:
        print("WARNING: No force_estimator_state_dict in checkpoint. Estimator has random weights.")
    estimator.eval()

    # ── History buffer ────────────────────────────────────────────────────
    history_buffer = ObsHistoryBufferNp(temporal_steps, num_raw_obs)

    # Keyboard controller
    kb = None
    if PYNPUT_AVAILABLE:
        kb = KeyboardController()
        print("Keyboard control enabled (WASD+QE for movement, Enter/Space/Esc for FSM)")

    # ── DDS setup ─────────────────────────────────────────────────────────
    ChannelFactoryInitialize(1, "lo")

    pub = ChannelPublisher("rt/lowcmd", LowCmd_)
    pub.Init()

    sub_low = ChannelSubscriber("rt/lowstate", LowState_)
    sub_low.Init(low_state_handler, 10)

    sub_sport = ChannelSubscriber("rt/sportmodestate", SportModeState_)
    sub_sport.Init(sport_state_handler, 10)

    crc = CRC()

    # Wait for simulator
    print("Waiting for simulator state...")
    while low_state is None:
        time.sleep(0.01)
    print("Connected to simulator.")

    # ── Create cmd once and reuse ─────────────────────────────────────────
    cmd = unitree_go_msg_dds__LowCmd_()
    cmd.head[0] = 0xFE
    cmd.head[1] = 0xEF
    cmd.level_flag = 0xFF
    cmd.gpio = 0
    for i in range(20):
        cmd.motor_cmd[i].mode = 0x01
        cmd.motor_cmd[i].q = 0.0
        cmd.motor_cmd[i].kp = 0.0
        cmd.motor_cmd[i].dq = 0.0
        cmd.motor_cmd[i].kd = 0.0
        cmd.motor_cmd[i].tau = 0.0

    # ══════════════════════════════════════════════════════════════════════
    # FSM STATE 1: ZERO TORQUE
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  ZERO TORQUE STATE")
    print("  Press START (joystick) or Enter (keyboard) to stand up")
    print("=" * 60 + "\n")

    while True:
        for i in range(12):
            cmd.motor_cmd[i].q = 0.0
            cmd.motor_cmd[i].kp = 0.0
            cmd.motor_cmd[i].kd = 0.0
            cmd.motor_cmd[i].dq = 0.0
            cmd.motor_cmd[i].tau = 0.0
        cmd.crc = crc.Crc(cmd)
        pub.Write(cmd)

        if remote_controller.button[KeyMap.start] == 1:
            break
        if kb is not None and kb.start_pressed:
            break
        time.sleep(control_dt)

    # ══════════════════════════════════════════════════════════════════════
    # FSM STATE 2: STAND UP (multi-phase)
    # ══════════════════════════════════════════════════════════════════════
    print("Standing up...")
    dt = 0.002
    start_pos = [low_state.motor_state[i].q for i in range(12)]

    # Phase 1: current -> crouch (1s)
    duration = 1.0
    t = 0.0
    while t < duration:
        step_start = time.perf_counter()
        phase = min(t / duration, 1.0)
        for i in range(12):
            cmd.motor_cmd[i].q = (1 - phase) * start_pos[i] + phase * LYING_POS[i]
            cmd.motor_cmd[i].kp = STANDUP_KP
            cmd.motor_cmd[i].kd = STANDUP_KD
            cmd.motor_cmd[i].dq = 0.0
            cmd.motor_cmd[i].tau = 0.0
        cmd.crc = crc.Crc(cmd)
        pub.Write(cmd)
        t += dt
        elapsed = time.perf_counter() - step_start
        if dt - elapsed > 0:
            time.sleep(dt - elapsed)

    # Phase 2: crouch -> standing (1s)
    duration = 1.0
    t = 0.0
    while t < duration:
        step_start = time.perf_counter()
        phase = min(t / duration, 1.0)
        for i in range(12):
            cmd.motor_cmd[i].q = (1 - phase) * LYING_POS[i] + phase * default_angles_sdk[i]
            cmd.motor_cmd[i].kp = STANDUP_KP
            cmd.motor_cmd[i].kd = STANDUP_KD
            cmd.motor_cmd[i].dq = 0.0
            cmd.motor_cmd[i].tau = 0.0
        cmd.crc = crc.Crc(cmd)
        pub.Write(cmd)
        t += dt
        elapsed = time.perf_counter() - step_start
        if dt - elapsed > 0:
            time.sleep(dt - elapsed)

    # Phase 3: hold standing (1s)
    print("Holding standing pose...")
    duration = 1.0
    t = 0.0
    while t < duration:
        step_start = time.perf_counter()
        for i in range(12):
            cmd.motor_cmd[i].q = default_angles_sdk[i]
            cmd.motor_cmd[i].kp = STANDUP_KP
            cmd.motor_cmd[i].kd = STANDUP_KD
            cmd.motor_cmd[i].dq = 0.0
            cmd.motor_cmd[i].tau = 0.0
        cmd.crc = crc.Crc(cmd)
        pub.Write(cmd)
        t += dt
        elapsed = time.perf_counter() - step_start
        if dt - elapsed > 0:
            time.sleep(dt - elapsed)

    # Phase 4: ramp standup gains -> policy gains (2s)
    print("Ramping to policy gains...")
    duration = 2.0
    t = 0.0
    while t < duration:
        step_start = time.perf_counter()
        alpha = min(t / duration, 1.0)
        test_kp = (1 - alpha) * STANDUP_KP + alpha * kps[0]
        test_kd = (1 - alpha) * STANDUP_KD + alpha * kds[0]
        for i in range(12):
            cmd.motor_cmd[i].q = default_angles_sdk[i]
            cmd.motor_cmd[i].kp = test_kp
            cmd.motor_cmd[i].kd = test_kd
            cmd.motor_cmd[i].dq = 0.0
            cmd.motor_cmd[i].tau = 0.0
        cmd.crc = crc.Crc(cmd)
        pub.Write(cmd)
        t += dt
        elapsed = time.perf_counter() - step_start
        if dt - elapsed > 0:
            time.sleep(dt - elapsed)

    gravity_check = get_gravity_orientation(low_state.imu_state.quaternion)
    print(f"After gain ramp: gravity={gravity_check.round(3)}, Kp={kps[0]}, Kd={kds[0]}")
    if gravity_check[2] > -0.7:
        print("WARNING: Robot may have collapsed during gain ramp!")

    print("Robot is standing.")

    # ══════════════════════════════════════════════════════════════════════
    # FSM STATE 3: WAIT FOR A
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  STANDING — WAITING FOR POLICY START")
    print("  Press A (joystick) or Space (keyboard) to start policy")
    print("=" * 60 + "\n")

    while True:
        step_start = time.perf_counter()
        for i in range(12):
            cmd.motor_cmd[i].q = default_angles_sdk[i]
            cmd.motor_cmd[i].kp = kps[0]
            cmd.motor_cmd[i].kd = kds[0]
            cmd.motor_cmd[i].dq = 0.0
            cmd.motor_cmd[i].tau = 0.0
        cmd.crc = crc.Crc(cmd)
        pub.Write(cmd)

        if remote_controller.button[KeyMap.A] == 1:
            break
        if kb is not None and kb.a_pressed:
            break

        elapsed = time.perf_counter() - step_start
        if dt - elapsed > 0:
            time.sleep(dt - elapsed)

    # ══════════════════════════════════════════════════════════════════════
    # FSM STATE 4: POLICY RUNNING (50 Hz) with force estimator + compliance
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  COMPLIANT POLICY RUNNING (50 Hz)")
    print(f"  Compliance: alpha={compliance_alpha:.1f}N, beta={compliance_beta:.1f}, k={compliance_k:.4f}")
    print(f"  EMA alpha={ema_alpha:.2f}")
    print("  Movement: W/S (fwd/back), A/D (strafe), Q/E (turn)")
    print("  Stop: SELECT (joystick) or Esc (keyboard)")
    print("=" * 70 + "\n")

    action = np.zeros(num_actions, dtype=np.float32)
    raw_obs = np.zeros(num_raw_obs, dtype=np.float32)
    obs = np.zeros(num_obs, dtype=np.float32)
    velocity_cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    force_ema = np.zeros(force_dim, dtype=np.float32)
    prev_target_dof_pos = DEFAULT_ANGLES_ISAAC.copy()
    policy_dt = control_dt
    step_count = 0
    debug_log = []
    should_stop = False

    # Joint name debug
    ISAAC_JOINT_NAMES = [
        "FL_hip", "FR_hip", "RL_hip", "RR_hip",
        "FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh",
        "FL_calf", "FR_calf", "RL_calf", "RR_calf",
    ]
    SDK_JOINT_NAMES = [
        "FR_hip", "FR_thigh", "FR_calf",
        "FL_hip", "FL_thigh", "FL_calf",
        "RR_hip", "RR_thigh", "RR_calf",
        "RL_hip", "RL_thigh", "RL_calf",
    ]
    print("\n=== JOINT MAPPING DEBUG (standing pose, before policy) ===")
    print(f"{'Isaac idx':<10} {'Isaac name':<12} {'motor_idx':<10} {'SDK name':<12} {'raw_q':>8} {'default':>8} {'q-def':>8}")
    for i in range(num_actions):
        mi = leg_joint2motor_idx[i]
        raw_q = low_state.motor_state[mi].q
        def_q = DEFAULT_ANGLES_ISAAC[i]
        print(f"{i:<10} {ISAAC_JOINT_NAMES[i]:<12} {mi:<10} {SDK_JOINT_NAMES[mi]:<12} {raw_q:>8.3f} {def_q:>8.3f} {raw_q - def_q:>8.3f}")
    print("=" * 78 + "\n")

    # Reset history buffer
    history_buffer.reset()

    try:
        while not should_stop:
            step_start = time.perf_counter()

            # ── Velocity commands from joystick + keyboard ────────────
            joy_vx = round(remote_controller.ly, 1)
            joy_vy = round(remote_controller.lx * -1, 1)
            joy_wz = round(remote_controller.rx * -1, 1)

            kb_vx, kb_vy, kb_wz = 0.0, 0.0, 0.0
            if kb is not None:
                kb_vx, kb_vy, kb_wz = kb.get_velocity_commands()

            if abs(kb_vx) > 0.01 or abs(kb_vy) > 0.01 or abs(kb_wz) > 0.01:
                velocity_cmd[0] = kb_vx
                velocity_cmd[1] = kb_vy
                velocity_cmd[2] = kb_wz
            else:
                velocity_cmd[0] = joy_vx
                velocity_cmd[1] = joy_vy
                velocity_cmd[2] = joy_wz

            # ── Build raw observation (57 dims, no force estimate) ────
            # [0:3] base_ang_vel
            raw_obs[0] = low_state.imu_state.gyroscope[0]
            raw_obs[1] = low_state.imu_state.gyroscope[1]
            raw_obs[2] = low_state.imu_state.gyroscope[2]

            # [3:6] projected_gravity
            raw_obs[3:6] = get_gravity_orientation(low_state.imu_state.quaternion)

            # [6:9] velocity_commands (will be overwritten with adjusted vel below)
            raw_obs[6:9] = velocity_cmd * cmd_scale * max_cmd

            # [9:21] joint_pos - default (Isaac convention)
            for i in range(num_actions):
                motor_idx = leg_joint2motor_idx[i]
                raw_obs[9 + i] = low_state.motor_state[motor_idx].q - DEFAULT_ANGLES_ISAAC[i]

            # [21:33] joint_vel (Isaac convention)
            for i in range(num_actions):
                motor_idx = leg_joint2motor_idx[i]
                raw_obs[21 + i] = low_state.motor_state[motor_idx].dq

            # [33:45] last_action
            raw_obs[33:45] = action

            # [45:57] applied_torque * torque_scale (Isaac convention)
            # Compute applied torque from PD control: tau = kp*(target - current) + kd*(0 - dq)
            for i in range(num_actions):
                motor_idx = leg_joint2motor_idx[i]
                q = low_state.motor_state[motor_idx].q
                dq = low_state.motor_state[motor_idx].dq
                target_q = prev_target_dof_pos[i]
                tau = kps[i] * (target_q - q) + kds[i] * (0.0 - dq)
                raw_obs[45 + i] = tau * torque_scale

            # ── Update history buffer and run estimator ───────────────
            history_buffer.insert(raw_obs)
            history_flat = history_buffer.get_flattened()
            history_tensor = torch.from_numpy(history_flat).unsqueeze(0)

            with torch.no_grad():
                force_hat_tensor, _ = estimator.get_latent(history_tensor)
            force_hat = force_hat_tensor.squeeze(0).numpy()

            # EMA filter
            force_ema = ema_alpha * force_hat + (1.0 - ema_alpha) * force_ema

            # ── Apply compliance mapping ──────────────────────────────
            # k(f) = 0 if |f_hat_ema| < alpha, else 1/beta
            force_mag = np.linalg.norm(force_ema)
            if force_mag >= compliance_alpha:
                k = compliance_k
            else:
                k = 0.0

            # v* = v_cmd + k * f_hat_ema
            adjusted_vel = velocity_cmd.copy()
            adjusted_vel[0] += k * force_ema[0]
            adjusted_vel[1] += k * force_ema[1]

            # ── Build full policy observation (59 dims) ───────────────
            obs[:num_raw_obs] = raw_obs
            # Override velocity commands with adjusted velocity
            obs[6:9] = adjusted_vel * cmd_scale * max_cmd
            # Append force estimate
            obs[num_raw_obs:num_obs] = force_hat

            # ── Policy inference ──────────────────────────────────────
            obs_tensor = torch.from_numpy(obs).unsqueeze(0)
            action = policy(obs_tensor).detach().numpy().squeeze()

            # NaN safety
            if np.any(np.isnan(action)):
                print(f"[step {step_count}] WARNING: NaN in action, zeroing out")
                action = np.zeros(num_actions, dtype=np.float32)

            action = np.clip(action, -10.0, 10.0)

            # ── Apply action ──────────────────────────────────────────
            target_dof_pos = action * action_scale + DEFAULT_ANGLES_ISAAC
            prev_target_dof_pos = target_dof_pos.copy()

            for i in range(num_actions):
                motor_idx = leg_joint2motor_idx[i]
                cmd.motor_cmd[motor_idx].q = float(target_dof_pos[i])
                cmd.motor_cmd[motor_idx].dq = 0.0
                cmd.motor_cmd[motor_idx].kp = kps[i]
                cmd.motor_cmd[motor_idx].kd = kds[i]
                cmd.motor_cmd[motor_idx].tau = 0.0

            cmd.crc = crc.Crc(cmd)
            pub.Write(cmd)

            # ── Debug logging ─────────────────────────────────────────
            step_count += 1
            debug_log.append({
                'step': step_count,
                'obs': obs.copy(),
                'raw_obs': raw_obs.copy(),
                'action': action.copy(),
                'target_dof_pos': target_dof_pos.copy(),
                'velocity_cmd': velocity_cmd.copy(),
                'adjusted_vel': adjusted_vel.copy(),
                'force_hat': force_hat.copy(),
                'force_ema': force_ema.copy(),
                'force_mag': force_mag,
                'compliance_k_active': k,
                'imu_quat': np.array(low_state.imu_state.quaternion),
                'imu_gyro': np.array(low_state.imu_state.gyroscope),
                'imu_rpy': np.array(low_state.imu_state.rpy),
            })

            # Debug print
            do_print = (step_count <= 5
                        or (step_count <= 50 and step_count % 10 == 0)
                        or step_count % 50 == 0)
            if do_print:
                print(f"[step {step_count}] cmd=[{velocity_cmd[0]:.1f},{velocity_cmd[1]:.1f},{velocity_cmd[2]:.1f}]"
                      f"  v*=[{adjusted_vel[0]:+.2f},{adjusted_vel[1]:+.2f}]"
                      f"  f_hat=[{force_hat[0]:+.1f},{force_hat[1]:+.1f}]"
                      f"  |f|={force_mag:.1f}  k={'ON' if k > 0 else 'off'}"
                      f"  gravity={raw_obs[3:6].round(3)}"
                      f"  action_norm={np.linalg.norm(action):.3f}")

            # ── Check stop condition ──────────────────────────────────
            if remote_controller.button[KeyMap.select] == 1:
                should_stop = True
            if kb is not None and kb.select_pressed:
                should_stop = True

            elapsed = time.perf_counter() - step_start
            if policy_dt - elapsed > 0:
                time.sleep(policy_dt - elapsed)

    except KeyboardInterrupt:
        print("\nCtrl+C received.")

    # ══════════════════════════════════════════════════════════════════════
    # FSM STATE 5: LIE DOWN
    # ══════════════════════════════════════════════════════════════════════
    print("Lying down...")
    lie_pos = [low_state.motor_state[i].q for i in range(12)]
    lie_duration = 0.6
    t = 0.0
    while t < lie_duration:
        step_start = time.perf_counter()
        phase = min(t / lie_duration, 1.0)
        for i in range(12):
            cmd.motor_cmd[i].q = (1 - phase) * lie_pos[i] + phase * LYING_POS[i]
            cmd.motor_cmd[i].kp = STANDUP_KP
            cmd.motor_cmd[i].kd = STANDUP_KD
            cmd.motor_cmd[i].dq = 0.0
            cmd.motor_cmd[i].tau = 0.0
        cmd.crc = crc.Crc(cmd)
        pub.Write(cmd)
        t += dt
        elapsed = time.perf_counter() - step_start
        if dt - elapsed > 0:
            time.sleep(dt - elapsed)
    print("Robot is lying down.")

    # ── Save logs ─────────────────────────────────────────────────────────
    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = log_dir / f"sim2sim_compliant_{timestamp}.npz"

    N = len(debug_log)
    if N > 0:
        np.savez(log_path,
                 observations=np.array([s['obs'] for s in debug_log]),
                 raw_observations=np.array([s['raw_obs'] for s in debug_log]),
                 actions=np.array([s['action'] for s in debug_log]),
                 target_dof_pos=np.array([s['target_dof_pos'] for s in debug_log]),
                 velocity_cmd=np.array([s['velocity_cmd'] for s in debug_log]),
                 adjusted_vel=np.array([s['adjusted_vel'] for s in debug_log]),
                 force_hat=np.array([s['force_hat'] for s in debug_log]),
                 force_ema=np.array([s['force_ema'] for s in debug_log]),
                 force_mag=np.array([s['force_mag'] for s in debug_log]),
                 compliance_k_active=np.array([s['compliance_k_active'] for s in debug_log]),
                 imu_quat=np.array([s['imu_quat'] for s in debug_log]),
                 imu_gyro=np.array([s['imu_gyro'] for s in debug_log]),
                 imu_rpy=np.array([s['imu_rpy'] for s in debug_log]),
                 steps=np.array([s['step'] for s in debug_log]),
                 timestamps=np.arange(N) * control_dt,
                 control_dt=control_dt,
                 action_scale=action_scale,
                 compliance_alpha=compliance_alpha,
                 compliance_beta=compliance_beta,
                 ema_alpha=ema_alpha,
                 )
        print(f"Saved {N} steps to {log_path}")
    else:
        print("No data to save.")

    if kb is not None:
        kb.stop()

    print("EXIT")
