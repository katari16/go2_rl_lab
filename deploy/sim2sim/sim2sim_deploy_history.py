"""
Sim2Sim deployment: Run trained IsaacLab proprioceptive policy (with observation
history) in unitree_mujoco with full FSM and keyboard + joystick control.

The policy expects 5 stacked observation frames (225 dims = 5 x 45).
Observation scales (ang_vel * 0.25, dof_vel * 0.05) are applied to match training.

Usage:
  1. Start unitree_mujoco:  cd ~/unitree_mujoco/simulate_python && python3 unitree_mujoco.py
  2. Run this script:       cd ~/go2_rl_lab/deploy/sim2sim && python3 sim2sim_deploy_history.py

Controls (keyboard):
  Enter  = START (begin stand-up)
  Space  = A     (start policy)
  Esc    = SELECT (stop policy, lie down)
  W/S    = forward/backward
  A/D    = strafe left/right
  Q/E    = turn left/right
  LShift = L2 (reserved for gait switching)

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


# ── Self-contained helpers (no deploy_real imports) ───────────────────────────

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
        """Enter key = START."""
        return self.is_pressed(pynput_keyboard.Key.enter)

    @property
    def a_pressed(self):
        """Space key = A button."""
        return self.is_pressed(pynput_keyboard.Key.space)

    @property
    def select_pressed(self):
        """Esc key = SELECT."""
        return self.is_pressed(pynput_keyboard.Key.esc)

    @property
    def l2_pressed(self):
        """Left Shift = L2."""
        return self.is_pressed(pynput_keyboard.Key.shift_l) or self.is_pressed(pynput_keyboard.Key.shift)

    def get_velocity_commands(self):
        """Return (vx, vy, wz) from WASD + QE keys."""
        vx = 0.0
        vy = 0.0
        wz = 0.0
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
    """Load go2 yaml config."""
    with open(path, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


# ── Config ────────────────────────────────────────────────────────────────────

# Default standing pose (Isaac convention)
DEFAULT_ANGLES_ISAAC = np.array(
    [0.1, -0.1, 0.1, -0.1, 0.8, 0.8, 1, 1, -1.5, -1.5, -1.5, -1.5],
    dtype=np.float32,
)

# Crouch / lying position (SDK ordering)
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
    project_root = Path(__file__).resolve().parents[1]
    cfg = load_config(project_root / "deploy_real" / "configs" / "go2_isaaclab_history.yaml")

    num_actions = cfg["num_actions"]
    num_obs = cfg["num_obs"]
    obs_size_single = cfg.get("obs_size_single", num_obs)
    history_length = cfg.get("history_length", 1)
    action_scale = cfg["action_scale"]
    ang_vel_scale = cfg.get("ang_vel_scale", 1.0)
    dof_pos_scale = cfg.get("dof_pos_scale", 1.0)
    dof_vel_scale = cfg.get("dof_vel_scale", 1.0)
    cmd_scale = np.array(cfg["cmd_scale"], dtype=np.float32)
    max_cmd = np.array(cfg["max_cmd"], dtype=np.float32)
    kps = cfg["kps"]
    kds = cfg["kds"]
    leg_joint2motor_idx = cfg["leg_joint2motor_idx"]
    default_angles_sdk = np.array(cfg["default_angles"], dtype=np.float32)
    control_dt = cfg["control_dt"]

    # Load policy
    policy_path = project_root / "pre_train" / cfg["policy_path"]
    print(f"Loading policy from {policy_path}")
    policy = torch.jit.load(str(policy_path))
    policy.eval()

    # Keyboard controller
    kb = None
    if PYNPUT_AVAILABLE:
        kb = KeyboardController()
        print("Keyboard control enabled (WASD+QE for movement, Enter/Space/Esc for FSM)")

    # ── DDS setup (domain 1, loopback — matches unitree_mujoco config) ────
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
    # FSM STATE 1: ZERO TORQUE — wait for START (joystick) or Enter (keyboard)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  ZERO TORQUE STATE")
    print("  Press START (joystick) or Enter (keyboard) to stand up")
    print("=" * 60 + "\n")

    while True:
        # Send zero commands
        for i in range(12):
            cmd.motor_cmd[i].q = 0.0
            cmd.motor_cmd[i].kp = 0.0
            cmd.motor_cmd[i].kd = 0.0
            cmd.motor_cmd[i].dq = 0.0
            cmd.motor_cmd[i].tau = 0.0
        cmd.crc = crc.Crc(cmd)
        pub.Write(cmd)

        # Check joystick START
        if remote_controller.button[KeyMap.start] == 1:
            break
        # Check keyboard Enter
        if kb is not None and kb.start_pressed:
            break
        time.sleep(control_dt)

    # ══════════════════════════════════════════════════════════════════════
    # FSM STATE 2: MOVE TO DEFAULT POSITION (multi-phase stand-up)
    # ══════════════════════════════════════════════════════════════════════
    print("Standing up...")
    dt = 0.002
    start_pos = [low_state.motor_state[i].q for i in range(12)]

    # Phase 1: current position -> crouch (1 second)
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

    # Phase 2: crouch -> default standing pose (1 second)
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

    # Phase 3: hold standing pose (1 second)
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

    # Phase 4: ramp from standup gains to policy gains (2 seconds)
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

    # Check stability
    gravity_check = get_gravity_orientation(low_state.imu_state.quaternion)
    print(f"After gain ramp: gravity={gravity_check.round(3)}, Kp={kps[0]}, Kd={kds[0]}")
    if gravity_check[2] > -0.7:
        print("WARNING: Robot may have collapsed during gain ramp!")

    print("Robot is standing.")

    # ══════════════════════════════════════════════════════════════════════
    # FSM STATE 3: WAIT FOR A — hold default pose, wait for A or Space
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

        # Check joystick A
        if remote_controller.button[KeyMap.A] == 1:
            break
        # Check keyboard Space
        if kb is not None and kb.a_pressed:
            break

        elapsed = time.perf_counter() - step_start
        if dt - elapsed > 0:
            time.sleep(dt - elapsed)

    # ══════════════════════════════════════════════════════════════════════
    # FSM STATE 4: RUN — policy loop at 50 Hz
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print(f"  POLICY RUNNING (50 Hz) — {history_length}x{obs_size_single} = {num_obs} obs dims")
    print("  Movement: W/S (fwd/back), A/D (strafe), Q/E (turn)")
    print("  Stop: SELECT (joystick) or Esc (keyboard)")
    print("=" * 60 + "\n")

    action = np.zeros(num_actions, dtype=np.float32)
    obs = np.zeros(obs_size_single, dtype=np.float32)
    obs_history = np.zeros((history_length, obs_size_single), dtype=np.float32)
    velocity_cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    policy_dt = control_dt  # 0.02 = 50 Hz
    step_count = 0
    debug_log = []
    should_stop = False

    try:
        while not should_stop:
            step_start = time.perf_counter()

            # ── Velocity commands from joystick + keyboard ────────────
            # Joystick input (same mapping as real deploy)
            joy_vx = round(remote_controller.ly, 1)
            joy_vy = round(remote_controller.lx * -1, 1)
            joy_wz = round(remote_controller.rx * -1, 1)

            # Keyboard input
            kb_vx, kb_vy, kb_wz = 0.0, 0.0, 0.0
            if kb is not None:
                kb_vx, kb_vy, kb_wz = kb.get_velocity_commands()

            # Combine: keyboard overrides if active, otherwise use joystick
            if abs(kb_vx) > 0.01 or abs(kb_vy) > 0.01 or abs(kb_wz) > 0.01:
                velocity_cmd[0] = kb_vx
                velocity_cmd[1] = kb_vy
                velocity_cmd[2] = kb_wz
            else:
                velocity_cmd[0] = joy_vx
                velocity_cmd[1] = joy_vy
                velocity_cmd[2] = joy_wz

            # ── Build observation (45-dim single frame, with scaling) ──
            # obs[0:3] = base_ang_vel * ang_vel_scale
            obs[0] = low_state.imu_state.gyroscope[0] * ang_vel_scale
            obs[1] = low_state.imu_state.gyroscope[1] * ang_vel_scale
            obs[2] = low_state.imu_state.gyroscope[2] * ang_vel_scale

            # obs[3:6] = projected_gravity
            obs[3:6] = get_gravity_orientation(low_state.imu_state.quaternion)

            # obs[6:9] = velocity_commands * cmd_scale * max_cmd
            obs[6:9] = velocity_cmd * cmd_scale * max_cmd

            # obs[9:21] = (joint_pos - default_joint) * dof_pos_scale (Isaac convention)
            for i in range(num_actions):
                motor_idx = leg_joint2motor_idx[i]
                obs[9 + i] = (low_state.motor_state[motor_idx].q - DEFAULT_ANGLES_ISAAC[i]) * dof_pos_scale

            # obs[21:33] = joint_vel * dof_vel_scale (Isaac convention)
            for i in range(num_actions):
                motor_idx = leg_joint2motor_idx[i]
                obs[21 + i] = low_state.motor_state[motor_idx].dq * dof_vel_scale

            # obs[33:45] = last_action
            obs[33:45] = action

            # ── Policy inference (stacked history) ─────────────────────
            # Shift history: drop oldest, append current frame
            obs_history = np.roll(obs_history, -1, axis=0)
            obs_history[-1] = obs
            # Flatten to (1, 225) for the policy — oldest first, newest last
            obs_tensor = torch.from_numpy(obs_history.flatten()).unsqueeze(0)
            action = policy(obs_tensor).detach().numpy().squeeze()

            # NaN safety
            if np.any(np.isnan(action)):
                print(f"[step {step_count}] WARNING: NaN in action, zeroing out")
                action = np.zeros(num_actions, dtype=np.float32)

            # Clip action
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
                'obs': obs.copy(),
                'action': action.copy(),
                'target_dof_pos': target_dof_pos.copy(),
                'velocity_cmd': velocity_cmd.copy(),
                'imu_quat': np.array(low_state.imu_state.quaternion),
                'imu_gyro': np.array(low_state.imu_state.gyroscope),
                'imu_rpy': np.array(low_state.imu_state.rpy),
            })

            # Debug print
            do_print = (step_count <= 5
                        or (step_count <= 50 and step_count % 10 == 0)
                        or step_count % 50 == 0)
            if do_print:
                print(f"[step {step_count}] cmd=[{velocity_cmd[0]:.1f}, {velocity_cmd[1]:.1f}, {velocity_cmd[2]:.1f}]"
                      f"  gravity={obs[3:6].round(3)}"
                      f"  action_norm={np.linalg.norm(action):.3f}")

            # ── Check stop condition ──────────────────────────────────
            if remote_controller.button[KeyMap.select] == 1:
                should_stop = True
            if kb is not None and kb.select_pressed:
                should_stop = True

            # Timing
            elapsed = time.perf_counter() - step_start
            if policy_dt - elapsed > 0:
                time.sleep(policy_dt - elapsed)

    except KeyboardInterrupt:
        print("\nCtrl+C received.")

    # ══════════════════════════════════════════════════════════════════════
    # FSM STATE 5: MOVE TO GROUND — interpolate to lying position
    # ══════════════════════════════════════════════════════════════════════
    print("Lying down...")
    lie_pos = [low_state.motor_state[i].q for i in range(12)]
    lie_duration = 0.6  # 300 steps * 0.002
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
    log_path = log_dir / f"sim2sim_{timestamp}.npz"

    N = len(debug_log)
    if N > 0:
        observations = np.array([s['obs'] for s in debug_log])
        actions = np.array([s['action'] for s in debug_log])
        target_dof_pos_log = np.array([s['target_dof_pos'] for s in debug_log])
        velocity_cmd_log = np.array([s['velocity_cmd'] for s in debug_log])
        imu_quat_log = np.array([s['imu_quat'] for s in debug_log])
        imu_gyro_log = np.array([s['imu_gyro'] for s in debug_log])
        imu_rpy_log = np.array([s['imu_rpy'] for s in debug_log])
        steps = np.array([s['step'] for s in debug_log])

        np.savez(log_path,
                 observations=observations,
                 actions=actions,
                 target_dof_pos=target_dof_pos_log,
                 velocity_cmd=velocity_cmd_log,
                 imu_quat=imu_quat_log,
                 imu_gyro=imu_gyro_log,
                 imu_rpy=imu_rpy_log,
                 steps=steps,
                 timestamps=steps * control_dt,
                 control_dt=control_dt,
                 action_scale=action_scale,
                 )
        print(f"Saved {N} steps to {log_path}")
    else:
        print("No data to save.")

    # Cleanup
    if kb is not None:
        kb.stop()

    print("EXIT")
