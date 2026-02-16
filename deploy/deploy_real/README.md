# Deploy Real - IsaacLab Policy Deployment on Unitree Go2

Python-based deployment pipeline for running IsaacLab-trained RL locomotion policies on the real Unitree Go2 robot, with Kalman filter-based velocity estimation via an external InEKF.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                       deploy_real_isaaclab.py                   │
│                                                                 │
│  ┌──────────┐   ┌──────────────┐   ┌────────────────────────┐  │
│  │  Config   │   │   Policy     │   │   Observation Builder  │  │
│  │ (go2.yaml)│   │ (.pt model)  │   │   (52-dim vector)      │  │
│  └──────────┘   └──────────────┘   └────────────────────────┘  │
│                                                                 │
│  ┌──────────────────────┐   ┌───────────────────────────────┐  │
│  │  FSM State Machine   │   │   Motor Command Publisher     │  │
│  │  (gamepad-driven)    │   │   (rt/lowcmd via DDS)         │  │
│  └──────────────────────┘   └───────────────────────────────┘  │
└────────────────┬────────────────────────────┬───────────────────┘
                 │                            │
        DDS (CycloneDDS)              ROS2 (rclpy)
                 │                            │
    ┌────────────▼──────────┐    ┌────────────▼──────────────┐
    │   Unitree Go2 Robot   │    │   node_kalman.py           │
    │   rt/lowstate         │    │   (KalmanOdomListener)     │
    │   rt/lowcmd           │    │   subscribes to            │
    │   rt/sportmodestate   │    │   /odometry/filtered       │
    └───────────────────────┘    └────────────▲──────────────┘
                                              │
                                    ┌─────────┴─────────┐
                                    │  External InEKF   │
                                    │  State Estimator  │
                                    │  (ROS2 node)      │
                                    └───────────────────┘
```

## File Structure

```
deploy_real/
├── deploy_real_isaaclab.py      # Main deployment script (Controller class + main loop)
├── node_kalman.py               # ROS2 node bridging Kalman filter to deployment
├── configs/
│   ├── config.py                # Config class that parses YAML files
│   ├── go2.yaml                 # Main config for real Go2 deployment
│   ├── go2_sim.yaml             # Config for simulation testing
│   └── go2_old.yaml             # Legacy config backup
├── common/
│   ├── remote_controller.py     # Gamepad button/joystick parser (from wireless_remote bytes)
│   ├── command_helper.py        # Low-level motor command helpers (zero, damping, init)
│   └── rotation_helper.py       # Quaternion to gravity projection + IMU transforms
└── README.md                    # This file
```

## How to Run

### Prerequisites
- Unitree Go2 robot powered on and connected via ethernet
- External InEKF state estimator running and publishing on `/odometry/filtered` (ROS2 topic)
- ROS2 Humble environment sourced
- `unitree_sdk2_python` available at `../unitree_sdk2_python` relative to the project root
- Trained policy exported as `.pt` (TorchScript) file in `../pre_train/`

### Launch Command

```bash
cd /path/to/sim_to_real_go2
python -m deploy_real.deploy_real_isaaclab <network_interface> <config_file>
```

Example:
```bash
python -m deploy_real.deploy_real_isaaclab eth0 go2.yaml
```

- `<network_interface>`: The network interface connected to the Go2 (e.g., `eth0`, `enp3s0`)
- `<config_file>`: YAML config filename from the `configs/` folder (e.g., `go2.yaml`)

### Startup Sequence (Numbered Steps in Console Output)

The script prints numbered steps so you can follow along:

| Step | Console Message | What Happens |
|------|----------------|-------------|
| 1 | `CONFIG FILE LOADED SUCCESSFULLY` | YAML config parsed |
| 2 | `CHANNEL FACTORY CREATED` | DDS communication initialized on the given network interface |
| 3 | `LOADING POLICY` | TorchScript policy loaded from `pre_train/<policy_path>` |
| 3 | `ROBOT IS IN LYING POSITION...` | Robot stands down and high-level mode is released (switches to low-level control) |
| 4 | `INITIALIZING CHANNELS` | DDS publishers/subscribers created for `rt/lowcmd`, `rt/lowstate`, `rt/sportmodestate` |
| 5 | `ZERO TORQUE STATE IS ACTIVE` | Motors receive zero commands. **Press START on the gamepad** to proceed. |
| 6 | `ROBOT IS MOVING TO DEFAULT POSE` | Robot gradually lifts to standing position through 4 interpolation phases |
| 7 | `ROBOT IS STANDING` | Robot holds the default standing pose. **Press A on the gamepad** to start the policy. |
| 8 | `MODEL IS RUNNING` | Policy inference loop is active at 50 Hz (20 ms per step). **Press SELECT** to stop. |
| 9 | `ROBOT IS LYING DOWN` | Robot smoothly lowers to a crouched position |
| 10 | `DATA VISUALIZATION IN PROGRESS` | Plots of velocities and commands are saved to `analyse_robot.pdf` |

## Finite State Machine (FSM)

```
                  ┌──────────────┐
                  │  zero_torque │ ◄── Initial state (motors off)
                  └──────┬───────┘
                         │ [START button]
                  ┌──────▼───────────────┐
                  │  move_to_default_pos │ ◄── 4-phase interpolation to standing
                  └──────┬───────────────┘
                         │ [A button]
                  ┌──────▼───────┐
                  │  run (policy)│ ◄── Main control loop at 50 Hz
                  └──────┬───────┘
                         │ [SELECT button]
                  ┌──────▼───────────┐
                  │  move_to_ground  │ ◄── Smooth interpolation to lying
                  └──────┬───────────┘
                         │
                  ┌──────▼───────┐
                  │  Plots + Exit│
                  └──────────────┘
```

## Configuration (go2.yaml)

```yaml
msg_type: "go"                    # Message type ("go" for Go2)
imu_type: "pelvis"                # IMU location

lowcmd_topic: "rt/lowcmd"        # DDS topic to publish motor commands
lowstate_topic: "rt/lowstate"    # DDS topic to receive robot state

policy_path: "policy_pace_decimation.pt"  # Path to TorchScript policy (relative to pre_train/)

# Joint index mapping: IsaacLab convention -> real Go2 motor indices
# IsaacLab order: [FL_hip, FR_hip, RL_hip, RR_hip, FL_thigh, FR_thigh, RL_thigh, RR_thigh, FL_calf, FR_calf, RL_calf, RR_calf]
# Go2 motor order: [FR_hip(0), FR_thigh(1), FR_calf(2), FL_hip(3), FL_thigh(4), FL_calf(5), RR_hip(6), RR_thigh(7), RR_calf(8), RL_hip(9), RL_thigh(10), RL_calf(11)]
leg_joint2motor_idx: [3,0,9,6,4,1,10,7,5,2,11,8]

# Default standing joint angles (in Go2 motor order)
default_angles: [-0.1, 0.8, -1.5, 0.1, 0.8, -1.5, -0.1, 1, -1.5, 0.1, 1, -1.5]

# PD gains for the motors
kps: [25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0]
kds: [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]

# Policy parameters (must match training config)
control_dt: 0.02          # 50 Hz control loop
action_scale: 0.25        # Multiplier on raw policy output
cmd_scale: [1.0, 1.0, 1.0]
max_cmd: [1, 0.4, 1]      # [lin_vel_x, lin_vel_y, ang_vel_z] max values
num_actions: 12            # 12 joints
num_obs: 52                # Total observation dimension
```

## Observation Vector Layout (52 dimensions)

The observation vector fed to the policy is constructed in `run()` as follows:

| Index | Size | Name | Source | Scale |
|-------|------|------|--------|-------|
| 0-3 | 4 | `foot_contact_forces` | `low_state.foot_force[{1,0,3,2}] / 100` | 0.01 |
| 4-6 | 3 | `base_lin_vel` | Kalman filter via `node_kalman.base_lin_vel_input[0:3]` | 1.0 |
| 7-9 | 3 | `base_ang_vel` | `low_state.imu_state.gyroscope` | 1.0 |
| 10-12 | 3 | `projected_gravity` | Computed from `low_state.imu_state.quaternion` | 1.0 |
| 13-15 | 3 | `velocity_commands` | Gamepad joysticks `[ly, -lx, -rx]` * `cmd_scale` * `max_cmd` | 1.0 |
| 16-27 | 12 | `joint_pos_rel` | `motor_state[leg_joint2motor_idx[i]].q - default_joint_pos[i]` | 1.0 |
| 28-39 | 12 | `joint_vel_rel` | `motor_state[leg_joint2motor_idx[i]].dq` | 1.0 |
| 40-51 | 12 | `last_action` | Previous policy output (raw, before scaling) | 1.0 |

**Important**: The foot contact force indices are swapped (`[1,0,3,2]` instead of `[0,1,2,3]`) to match the IsaacLab joint ordering convention.

## DDS Topics

All robot communication uses CycloneDDS (via `unitree_sdk2_python`):

| Topic | Direction | Message Type | Description |
|-------|-----------|-------------|-------------|
| `rt/lowcmd` | Publish | `LowCmd_` | Motor position/velocity/torque commands with PD gains |
| `rt/lowstate` | Subscribe | `LowState_` | Joint positions, velocities, IMU, foot forces, wireless remote |
| `rt/sportmodestate` | Subscribe | `SportModeState_` | High-level sport mode velocity (used during Init) |

The `ChannelFactoryInitialize(0, args.net)` call initializes DDS on domain 0 using the specified network interface.

## Kalman Filter Integration

### How It Works

1. **External InEKF node** (not part of this repo) runs as a separate ROS2 process. It subscribes to IMU + joint data and publishes filtered odometry on the ROS2 topic `/odometry/filtered`.

2. **`node_kalman.py`** runs a `KalmanOdomListener` ROS2 node in a background thread inside the deployment script. It:
   - Subscribes to `/odometry/filtered` and extracts `twist.twist.linear.{x,y,z}` and `twist.twist.angular.z`
   - Stores these in the global variable `base_lin_vel_input = [vx, vy, vz, wz]`
   - Also republishes `low_state` data as a ROS2 `LowState` message on `/inekf_lowstate` for the InEKF to consume

3. **`deploy_real_isaaclab.py`** reads `node_kalman.base_lin_vel_input[0:3]` at each control step to populate the `base_lin_vel` observation (indices 4-6).

### Data Flow

```
Go2 Robot (rt/lowstate)
     │
     ▼
deploy_real_isaaclab.py
     │
     ├──► node_kalman.py publishes on /inekf_lowstate (ROS2)
     │         │
     │         ▼
     │    External InEKF subscribes to /inekf_lowstate
     │         │
     │         ▼
     │    External InEKF publishes /odometry/filtered (ROS2)
     │         │
     │         ▼
     │    node_kalman.py subscribes to /odometry/filtered
     │         │
     │         ▼
     │    node_kalman.base_lin_vel_input = [vx, vy, vz, wz]
     │
     ├──► deploy_real_isaaclab.py reads base_lin_vel_input into obs[4:7]
     │
     ▼
Policy inference → motor commands → rt/lowcmd
```

### Backup Velocity Estimator

If the Kalman filter is not available, there is a built-in kinematics-based velocity estimator in `compute_velocity()`. It uses:
- Joint angles (hip, thigh, calf) for each of the 4 legs
- Joint velocities
- Foot contact forces (threshold > 20 to determine if a foot is on the ground)
- Forward kinematics with link lengths `l1=0.21m`, `l2=0.23m`

To use the backup estimator instead of the Kalman filter, comment out line 483 and uncomment line 482 in `deploy_real_isaaclab.py`:
```python
self.obs[4:7]= [vx*2,vy*2,vz*2]  # Backup: kinematics velocity estimation
#self.obs[4:7]= [node_kalman.base_lin_vel_input[0],node_kalman.base_lin_vel_input[1],node_kalman.base_lin_vel_input[2]]
```

The kinematic estimator also applies a moving-average smoothing window (default size 20).

## Policy Parsing

The policy is loaded as a TorchScript model (`.pt` file):

```python
policy_path = project_root / "pre_train" / config.policy_path
self.policy = torch.jit.load(policy_path)
```

At each control step in `run()`:
1. The 52-dim observation vector is assembled
2. Converted to a PyTorch tensor: `obs_tensor = torch.from_numpy(self.obs).unsqueeze(0)`
3. Forward pass through the policy: `self.action = self.policy(obs_tensor).detach().numpy().squeeze()`
4. Actions (12 values) are scaled and offset to get target joint positions:
   ```python
   target_dof_pos = self.action * action_scale + default_joint_pos
   ```
5. Target positions are sent to motors via PD control:
   ```python
   self.low_cmd.motor_cmd[motor_idx].q = target_dof_pos[i]
   self.low_cmd.motor_cmd[motor_idx].kp = kps[i]   # default: 25.0
   self.low_cmd.motor_cmd[motor_idx].kd = kds[i]    # default: 10.0
   ```

## Joint Mapping

IsaacLab uses a different joint ordering than the Go2 hardware:

```
IsaacLab index → Go2 motor index
    0 (FL_hip)       → 3
    1 (FR_hip)       → 0
    2 (RL_hip)       → 9
    3 (RR_hip)       → 6
    4 (FL_thigh)     → 4
    5 (FR_thigh)     → 1
    6 (RL_thigh)     → 10
    7 (RR_thigh)     → 7
    8 (FL_calf)      → 5
    9 (FR_calf)      → 2
    10 (RL_calf)     → 11
    11 (RR_calf)     → 8
```

This mapping is defined in `leg_joint2motor_idx: [3,0,9,6,4,1,10,7,5,2,11,8]` in the config.

## Gamepad Controls

The Go2's built-in wireless remote (or BT gamepad) is read from `low_state.wireless_remote`:

| Button | Action |
|--------|--------|
| **START** | Transition from zero-torque to standing |
| **A** | Start running the RL policy |
| **SELECT** | Stop the policy and lay the robot down |
| **Left joystick Y** (`ly`) | Forward/backward velocity command |
| **Left joystick X** (`lx`) | Left/right velocity command (inverted) |
| **Right joystick X** (`rx`) | Yaw rotation command (inverted) |

## Velocity Command Mapping

Joystick inputs are mapped to velocity commands:
```python
cmd[0] = round(ly, 1)          # lin_vel_x: forward/backward
cmd[1] = round(lx * -1, 1)     # lin_vel_y: left/right (inverted)
cmd[2] = round(rx * -1, 1)     # ang_vel_z: yaw (inverted)
```

These are then scaled by `cmd_scale * max_cmd` before entering the observation vector. With the default config (`max_cmd: [1, 0.4, 1]`), this gives ranges of:
- `lin_vel_x`: [-1.0, 1.0] m/s
- `lin_vel_y`: [-0.4, 0.4] m/s
- `ang_vel_z`: [-1.0, 1.0] rad/s

## Move-to-Default-Pose Interpolation

The `move_to_default_pos()` function lifts the robot through 4 phases:

1. **Phase 1** (500 steps): Interpolate from current position to `_targetPos_1` (crouched, legs gathered)
2. **Phase 2** (500 steps): Interpolate from `_targetPos_1` to `_targetPos_2` (semi-standing)
3. **Phase 3** (1000 steps): Hold at `_targetPos_2` (stabilization period)
4. **Phase 4** (900 steps): Interpolate from `_targetPos_2` to `_targetPos_3` (final standing pose)

Each phase uses linear interpolation: `q = (1 - percent) * start + percent * target`

## Debug Output

The script logs every control step to `debug_log.json`:
```json
[
  {"step": 1, "obs": [52 floats], "action": [12 floats]},
  {"step": 2, "obs": [52 floats], "action": [12 floats]},
  ...
]
```

After stopping, it also generates `analyse_robot.pdf` with 4 plots:
- **Vx**: Commanded vs estimated (kinematics) vs Kalman filter
- **Vy**: Commanded vs estimated vs Kalman filter
- **Vz**: Estimated vs Kalman filter
- **Wz**: Commanded vs angular velocity vs Kalman filter

## Troubleshooting

### `base_lin_vel` is always [0, 0, 0]
The Kalman filter is not publishing. Check:
1. Is the InEKF node running?
2. Is it publishing on `/odometry/filtered`? (`ros2 topic echo /odometry/filtered`)
3. Is `node_kalman.py` receiving data? Add a print in `listener_callback`
4. Fallback: switch to the kinematics velocity estimator (see Backup Velocity Estimator section)

### Robot falls over immediately
- Check that `kps` and `kds` in the config match what the policy was trained with
- Verify the policy was trained with the same observation space (52 dims with foot forces + base_lin_vel)
- Ensure `action_scale` matches training (typically 0.25)
- Check that `default_angles` match the training default joint positions

### DDS connection issues
- Verify the network interface: `ip addr show`
- Make sure no ROS2 environment variables are polluting CycloneDDS: `unset ROS_DISTRO AMENT_PREFIX_PATH`
- Exception: ROS2 env is needed for `rclpy` (the Kalman filter), so source ROS2 *before* running

### Motors not responding
- Ensure the robot's high-level mode was released (the script does this in `Init()`)
- The `init_cmd_go()` sets motor mode to `0x0A` (low-level position mode)
- CRC is computed for every command via `CRC().Crc(cmd)` before publishing
