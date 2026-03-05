"""
Sim2Sim deployment: Compliant locomotion policy (no foot contacts, XYZ force estimator)
in unitree_mujoco with full FSM and keyboard + joystick control.

The policy expects 60-dim obs = 57 raw proprioceptive + 3 force estimate (XYZ).
The force estimator maintains a 20-step history buffer internally.

External forces: Press Y (gamepad) or F (keyboard) to toggle random forces
applied to the robot base via UDP -> unitree_mujoco. Forces are sampled from
the training distribution ([0, 20]N per XY axis, random sign, fz scaled 0.6).
Re-randomized every 3-5 seconds while active.

Usage:
  1. Start unitree_mujoco:  cd ~/unitree_mujoco/simulate_python && python3 unitree_mujoco.py
  2. Run this script:       cd ~/go2_rl_lab/deploy/sim2sim && python3 sim2sim_compliant_no_foot_xyz.py

Controls (keyboard):
  Enter  = START (begin stand-up)
  Space  = A     (start policy)
  Esc    = SELECT (stop policy, lie down)
  W/S    = forward/backward
  A/D    = strafe left/right
  Q/E    = turn left/right
  F      = toggle external force (same as Y on gamepad)
"""

import json
import socket
import struct
import sys
import time
import numpy as np
import torch
import yaml
import threading
from pathlib import Path
from datetime import datetime
from collections import deque

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


# ── Self-contained helpers ───────────────────────────────────────────────────

def get_gravity_orientation(quat):
    """Project gravity into body frame from quaternion [w, x, y, z]."""
    qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]
    gx = 2 * (-qz * qx + qw * qy)
    gy = -2 * (qz * qy + qw * qx)
    gz = -1 + 2 * (qx * qx + qy * qy)
    return np.array([gx, gy, gz])


class KeyMap:
    R1 = 0; L1 = 1; start = 2; select = 3; R2 = 4; L2 = 5
    F1 = 6; F2 = 7; A = 8; B = 9; X = 10; Y = 11
    up = 12; right = 13; down = 14; left = 15


class RemoteController:
    def __init__(self):
        self.lx = 0.0; self.ly = 0.0; self.rx = 0.0; self.ry = 0.0
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
            on_press=self._on_press, on_release=self._on_release,
        )
        self._listener.daemon = True
        self._listener.start()

    def _on_press(self, key):
        with self._lock:
            try: self.pressed_keys.add(key.char.lower())
            except AttributeError: self.pressed_keys.add(key)

    def _on_release(self, key):
        with self._lock:
            try: self.pressed_keys.discard(key.char.lower())
            except AttributeError: self.pressed_keys.discard(key)

    def is_pressed(self, key):
        with self._lock: return key in self.pressed_keys

    def is_char_pressed(self, char):
        with self._lock: return char.lower() in self.pressed_keys

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
    def f_pressed(self):
        """F key = toggle external force."""
        return self.is_char_pressed('f')

    def get_velocity_commands(self):
        vx = vy = wz = 0.0
        if self.is_char_pressed('w'): vx += 1.0
        if self.is_char_pressed('s'): vx -= 1.0
        if self.is_char_pressed('a'): vy += 1.0
        if self.is_char_pressed('d'): vy -= 1.0
        if self.is_char_pressed('q'): wz += 1.0
        if self.is_char_pressed('e'): wz -= 1.0
        return vx, vy, wz

    def stop(self):
        self._listener.stop()


def load_config(path):
    with open(path, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


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


# ── Force Estimator Wrapper (JIT) ─────────────────────────────────────────────

class ForceEstimatorDeployment:
    """Lightweight wrapper around a JIT-exported force estimator.

    Maintains an observation history buffer and runs the estimator network.
    """

    def __init__(self, jit_path: str, raw_obs_dim: int, force_dim: int,
                 temporal_steps: int, device: str = "cpu"):
        self.device = torch.device(device)
        self.raw_obs_dim = raw_obs_dim
        self.force_dim = force_dim
        self.temporal_steps = temporal_steps

        # Load JIT-exported estimator
        self.estimator = torch.jit.load(jit_path, map_location=self.device)
        self.estimator.eval()
        print(f"[Estimator] Loaded JIT model from {jit_path}")
        print(f"  Input:  [{temporal_steps * raw_obs_dim}] = {temporal_steps} x {raw_obs_dim}")
        print(f"  Output: [{force_dim}]")

        # History buffer: [1, temporal_steps * raw_obs_dim]
        self._history = torch.zeros(1, temporal_steps * raw_obs_dim, device=self.device)
        self._step = 0

    def reset(self):
        self._history.zero_()
        self._step = 0

    def get_force_estimate(self, raw_obs: np.ndarray) -> np.ndarray:
        """Insert obs into history and return force estimate.

        Args:
            raw_obs: [raw_obs_dim] numpy array

        Returns:
            force_hat: [force_dim] numpy array
        """
        obs_t = torch.from_numpy(raw_obs).float().unsqueeze(0).to(self.device)

        # Shift history left by one step and insert new obs at the end
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


# ── External Force Sender ─────────────────────────────────────────────────────

class ExternalForceSender:
    """Sends force commands to unitree_mujoco via UDP.

    Samples forces from the training distribution:
      XY: uniform [force_min, force_max] with random sign
      Z:  same range scaled by fz_scale
    Re-randomizes every rerandom_interval seconds.
    """

    UDP_PORT = 9870

    def __init__(self, force_min: float = 0.0, force_max: float = 20.0,
                 fz_scale: float = 0.6, rerandom_interval: tuple = (3.0, 5.0)):
        self.force_min = force_min
        self.force_max = force_max
        self.fz_scale = fz_scale
        self.rerandom_interval = rerandom_interval
        self.active = False
        self.current_force = np.zeros(3, dtype=np.float64)
        self._next_rerandom = 0.0
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def toggle(self):
        """Toggle force on/off."""
        self.active = not self.active
        if self.active:
            self._sample_force()
            print(f"  [ExtForce] ON: [{self.current_force[0]:+.1f}, {self.current_force[1]:+.1f}, {self.current_force[2]:+.1f}] N")
        else:
            self.current_force[:] = 0.0
            self._send(self.current_force)
            print("  [ExtForce] OFF")

    def _sample_force(self):
        """Sample a new random force from training distribution."""
        for i in range(2):  # XY
            mag = np.random.uniform(self.force_min, self.force_max)
            sign = np.random.choice([-1.0, 1.0])
            self.current_force[i] = mag * sign
        # Z
        mag_z = np.random.uniform(self.force_min, self.force_max) * self.fz_scale
        sign_z = np.random.choice([-1.0, 1.0])
        self.current_force[2] = mag_z * sign_z
        # Schedule next re-randomization
        self._next_rerandom = time.time() + np.random.uniform(*self.rerandom_interval)

    def update(self):
        """Call every step. Re-randomizes if interval elapsed. Sends force via UDP."""
        if not self.active:
            return
        if time.time() >= self._next_rerandom:
            self._sample_force()
            print(f"  [ExtForce] Re-randomized: [{self.current_force[0]:+.1f}, {self.current_force[1]:+.1f}, {self.current_force[2]:+.1f}] N")
        self._send(self.current_force)

    def _send(self, force: np.ndarray):
        """Send force as JSON via UDP."""
        msg = json.dumps({"fx": force[0], "fy": force[1], "fz": force[2]}).encode()
        self._sock.sendto(msg, ("127.0.0.1", self.UDP_PORT))

    def stop(self):
        """Zero out force and close socket."""
        self.current_force[:] = 0.0
        self._send(self.current_force)
        self._sock.close()


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    cfg = load_config(project_root / "deploy_real" / "configs" / "go2_compliant_no_foot_xyz.yaml")

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
    ema_alpha = cfg.get("ema_alpha", 0.1)

    # Load JIT-exported policy
    policy_path = str(project_root / cfg["policy_path"])
    print(f"Loading policy from {policy_path}")
    policy = torch.jit.load(policy_path, map_location="cpu")
    policy.eval()
    print(f"[Policy] Loaded JIT model: {policy_obs_dim}-dim input -> {num_actions}-dim output")

    # Load JIT-exported force estimator
    estimator_path = str(project_root / cfg["estimator_path"])
    estimator = ForceEstimatorDeployment(
        jit_path=estimator_path,
        raw_obs_dim=raw_obs_dim,
        force_dim=force_dim,
        temporal_steps=temporal_steps,
        device="cpu",
    )

    # EMA state for compliance
    force_ema = np.zeros(force_dim, dtype=np.float32)

    # External force sender (Y button / F key)
    ext_force = ExternalForceSender(
        force_min=0.0, force_max=20.0, fz_scale=0.6, rerandom_interval=(3.0, 5.0),
    )

    # Keyboard controller
    kb = None
    if PYNPUT_AVAILABLE:
        kb = KeyboardController()
        print("Keyboard control enabled (WASD+QE for movement, Enter/Space/Esc for FSM, F for force)")

    # ── DDS setup ─────────────────────────────────────────────────────────
    ChannelFactoryInitialize(1, "lo")

    pub = ChannelPublisher("rt/lowcmd", LowCmd_)
    pub.Init()

    sub_low = ChannelSubscriber("rt/lowstate", LowState_)
    sub_low.Init(low_state_handler, 10)

    sub_sport = ChannelSubscriber("rt/sportmodestate", SportModeState_)
    sub_sport.Init(sport_state_handler, 10)

    crc = CRC()

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
    # FSM STATE 2: STAND UP
    # ══════════════════════════════════════════════════════════════════════
    print("Standing up...")
    dt = 0.002
    start_pos = [low_state.motor_state[i].q for i in range(12)]

    # Phase 1: current -> crouch
    duration_1 = 1.0
    t = 0.0
    while t < duration_1:
        step_start = time.perf_counter()
        phase = min(t / duration_1, 1.0)
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

    # Phase 2: crouch -> standing
    duration_2 = 1.0
    t = 0.0
    while t < duration_2:
        step_start = time.perf_counter()
        phase = min(t / duration_2, 1.0)
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

    # Phase 3: hold
    print("Holding standing pose...")
    hold_duration = 1.0
    t = 0.0
    while t < hold_duration:
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

    # Phase 4: ramp gains
    print("Ramping to policy gains...")
    ramp_duration = 2.0
    t = 0.0
    while t < ramp_duration:
        step_start = time.perf_counter()
        alpha = min(t / ramp_duration, 1.0)
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
    # FSM STATE 4: RUN POLICY (50 Hz)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print(f"  POLICY RUNNING (50 Hz) — {raw_obs_dim}+{force_dim}={policy_obs_dim} obs dims")
    print(f"  Force estimator: {temporal_steps}x{raw_obs_dim} history -> {force_dim}D force")
    if compliance_k > 0:
        print(f"  Compliance: k={compliance_k:.4f}  EMA alpha={ema_alpha}")
    print("  Movement: W/S (fwd/back), A/D (strafe), Q/E (turn)")
    print("  Force: Y (gamepad) or F (keyboard) — toggle random external force")
    print("  Stop: SELECT (joystick) or Esc (keyboard)")
    print("=" * 60 + "\n")

    action = np.zeros(num_actions, dtype=np.float32)
    raw_obs = np.zeros(raw_obs_dim, dtype=np.float32)
    velocity_cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    policy_dt = control_dt
    step_count = 0
    debug_log = []
    should_stop = False
    estimator.reset()

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

            # [45:57] applied_torque * 0.1 (Isaac convention)
            for i in range(num_actions):
                motor_idx = leg_joint2motor_idx[i]
                raw_obs[45 + i] = low_state.motor_state[motor_idx].tau_est * 0.1

            # ── Run force estimator ───────────────────────────────────
            force_hat = estimator.get_force_estimate(raw_obs)

            # EMA filter
            force_ema = ema_alpha * force_hat + (1.0 - ema_alpha) * force_ema

            # ── Compliance modulation (XY only) ──────────────────────
            obs_for_policy = raw_obs.copy()
            if compliance_k > 0.0:
                obs_for_policy[6] += compliance_k * force_ema[0]  # vx += k * F_hat_x
                obs_for_policy[7] += compliance_k * force_ema[1]  # vy += k * F_hat_y

            # ── Build full policy input (57 raw + 3 force estimate) ───
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
                'raw_obs': raw_obs.copy(),
                'force_hat': force_hat.copy(),
                'force_ema': force_ema.copy(),
                'action': action.copy(),
                'target_dof_pos': target_dof_pos.copy(),
                'velocity_cmd': velocity_cmd.copy(),
            })

            do_print = (step_count <= 5
                        or (step_count <= 50 and step_count % 10 == 0)
                        or step_count % 50 == 0)
            if do_print:
                print(f"[step {step_count}] cmd=[{velocity_cmd[0]:.1f},{velocity_cmd[1]:.1f},{velocity_cmd[2]:.1f}]"
                      f"  F_hat=[{force_hat[0]:+.1f},{force_hat[1]:+.1f},{force_hat[2]:+.1f}]"
                      f"  gravity={raw_obs[3:6].round(3)}"
                      f"  action_norm={np.linalg.norm(action):.3f}")

            # ── External force toggle (Y button / F key) ─────────────
            y_pressed = remote_controller.button[KeyMap.Y] == 1
            f_key_pressed = kb is not None and kb.f_pressed
            if (y_pressed or f_key_pressed) and not getattr(ext_force, '_debounce', False):
                ext_force.toggle()
                ext_force._debounce = True
            elif not y_pressed and not f_key_pressed:
                ext_force._debounce = False

            # Update force sender (re-randomize + send via UDP)
            ext_force.update()

            # ── Check stop ────────────────────────────────────────────
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
    log_path = log_dir / f"sim2sim_compliant_xyz_{timestamp}.npz"

    N = len(debug_log)
    if N > 0:
        np.savez(log_path,
                 raw_obs=np.array([s['raw_obs'] for s in debug_log]),
                 force_hat=np.array([s['force_hat'] for s in debug_log]),
                 force_ema=np.array([s['force_ema'] for s in debug_log]),
                 actions=np.array([s['action'] for s in debug_log]),
                 target_dof_pos=np.array([s['target_dof_pos'] for s in debug_log]),
                 velocity_cmd=np.array([s['velocity_cmd'] for s in debug_log]),
                 steps=np.array([s['step'] for s in debug_log]),
                 timestamps=np.arange(N) * control_dt,
                 control_dt=control_dt,
                 action_scale=action_scale,
                 compliance_k=compliance_k,
                 ema_alpha=ema_alpha,
                 )
        print(f"Saved {N} steps to {log_path}")
    else:
        print("No data to save.")

    ext_force.stop()
    if kb is not None:
        kb.stop()

    print("EXIT")
