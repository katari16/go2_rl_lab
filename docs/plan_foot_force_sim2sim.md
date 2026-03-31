# Plan: Add Foot Force Touch Sensors to Go2 MuJoCo Sim2Sim

## Context

The force-only policy (`Go2-Force-Only-v0`) uses 61-dim observations that include `foot_contact_force_norms` (4 dims, scaled ×0.01) and `applied_torque` (12 dims, scaled ×0.1). For sim2sim deployment via unitree_mujoco, the DDS bridge currently does NOT populate `foot_force[4]` in `LowState_`. The goal is to:

1. Add MuJoCo touch sensors to a new Go2 XML model (don't break the original)
2. Modify the Python bridge to read touch sensors and populate `foot_force[4]`
3. Create a sim2sim deploy script for the force-only policy (61-dim obs + estimator + compliance)
4. Log foot force readings so the user can verify they're in the same order of magnitude as training

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `unitree_mujoco/unitree_robots/go2/go2_force_sensor.xml` | **CREATE** | Copy of go2.xml + 4 touch sensors |
| `unitree_mujoco/simulate_python/unitree_sdk2py_bridge.py` | **MODIFY** | Read touch sensors → `foot_force[4]` |
| `unitree_mujoco/simulate_python/config.py` | **MODIFY** | Add config for which XML to load |
| `deploy/sim2sim/sim2sim_force_only_deploy.py` | **CREATE** | Sim2sim for 61-dim force-only policy + estimator |

---

## Step 1: New Go2 XML with Touch Sensors

**File**: `unitree_mujoco/unitree_robots/go2/go2_force_sensor.xml`

Copy `go2.xml` and add:

### 1a. Sites on foot bodies

The foot bodies (`FL_foot`, `FR_foot`, `RL_foot`, `RR_foot`) already exist but have no sites. Add a site to each, co-located with the foot geom. The touch sensor needs a site that references the foot geom:

```xml
<!-- Inside FL_calf body, after the foot geom -->
<body name="FL_foot" pos="0 0 -0.213">
  <site name="FL_foot_site" size="0.022" type="sphere"/>
</body>
```

Same for FR, RL, RR foot bodies. The site `size` matches the foot geom size (0.022m sphere).

### 1b. Touch sensors at end of `<sensor>` block

After `<framelinvel name="frame_vel" .../>`, add:

```xml
<!-- Foot force touch sensors (normal contact force in Newtons) -->
<touch name="FL_foot_touch" site="FL_foot_site"/>
<touch name="FR_foot_touch" site="FR_foot_site"/>
<touch name="RL_foot_touch" site="RL_foot_site"/>
<touch name="RR_foot_touch" site="RR_foot_site"/>
```

This adds 4 new sensor values to `data.sensordata`, at indices **52-55** (after the existing 52 sensor dims: 12 pos + 12 vel + 12 torque + 4 quat + 3 gyro + 3 accel + 3 framepos + 3 framevel).

**MuJoCo touch sensor output**: Non-negative scalar = sum of all contact normal forces on the site's geom (in Newtons). This matches IsaacLab's `net_forces_w` norm used in training.

---

## Step 2: Modify Bridge to Read Touch Sensors

**File**: `unitree_mujoco/simulate_python/unitree_sdk2py_bridge.py`

### 2a. Detect touch sensors during `__init__`

In the sensor detection loop (lines 50-57), add detection for touch sensors:

```python
self.have_foot_touch_ = False
self.foot_touch_start_idx_ = -1
for i in range(self.dim_motor_sensor, self.mj_model.nsensor):
    name = mujoco.mj_id2name(self.mj_model, mujoco._enums.mjtObj.mjOBJ_SENSOR, i)
    if name == "imu_quat":
        self.have_imu_ = True
    if name == "frame_pos":
        self.have_frame_sensor_ = True
    if name == "FL_foot_touch":
        self.have_foot_touch_ = True
        # Touch sensors are at the end: get the sensordata address
        self.foot_touch_start_idx_ = self.mj_model.sensor_adr[i]
```

### 2b. Read touch sensors in `PublishLowState()`

After the IMU reading block, add:

```python
if self.have_foot_touch_:
    idx = self.foot_touch_start_idx_
    # MuJoCo touch output is float (Newtons). foot_force is int16.
    # Store as int16 (truncate to integer Newtons — sufficient resolution).
    self.low_state.foot_force[0] = int(self.mj_data.sensordata[idx + 0])  # FL
    self.low_state.foot_force[1] = int(self.mj_data.sensordata[idx + 1])  # FR
    self.low_state.foot_force[2] = int(self.mj_data.sensordata[idx + 2])  # RL
    self.low_state.foot_force[3] = int(self.mj_data.sensordata[idx + 3])  # RR
```

**Foot order**: FL=0, FR=1, RL=2, RR=3. This matches the IsaacLab contact sensor body ordering (`".*_foot"` regex sorts alphabetically: FL_foot, FR_foot, RL_foot, RR_foot).

**Units**: MuJoCo outputs Newtons (float). Cast to int16 for the `foot_force` field. Typical standing force per leg: ~25N (10kg robot / 4 legs × 9.81). During walking: 0-100N range. int16 has plenty of range.

---

## Step 3: Config for XML Selection

**File**: `unitree_mujoco/simulate_python/config.py`

Check if there's a config entry for the XML model path. If so, the user can switch between `go2.xml` and `go2_force_sensor.xml`. If the path is hardcoded in `unitree_mujoco.py`, document how to change it.

---

## Step 4: Sim2Sim Deploy Script for Force-Only Policy

**File**: `deploy/sim2sim/sim2sim_force_only_deploy.py`

Based on the existing `sim2sim_deploy.py` pattern (DDS communication, same FSM, same keyboard controls), but extended for the 61-dim force-only policy with estimator.

### 4a. Observation Building (61 dims)

```python
# obs[0:3] = base_ang_vel (from IMU gyro)
obs[0:3] = low_state.imu_state.gyroscope

# obs[3:6] = projected_gravity (from IMU quat)
obs[3:6] = get_gravity_orientation(low_state.imu_state.quaternion)

# obs[6:9] = velocity_commands
obs[6:9] = velocity_cmd * cmd_scale * max_cmd

# obs[9:21] = joint_pos - default (Isaac convention via leg_joint2motor_idx)
for i in range(12):
    motor_idx = leg_joint2motor_idx[i]
    obs[9 + i] = low_state.motor_state[motor_idx].q - DEFAULT_ANGLES_ISAAC[i]

# obs[21:33] = joint_vel (Isaac convention)
for i in range(12):
    motor_idx = leg_joint2motor_idx[i]
    obs[21 + i] = low_state.motor_state[motor_idx].dq

# obs[33:45] = last_action
obs[33:45] = action

# obs[45:57] = applied_torque * 0.1 (Isaac convention)
for i in range(12):
    motor_idx = leg_joint2motor_idx[i]
    obs[45 + i] = low_state.motor_state[motor_idx].tau_est * 0.1

# obs[57:61] = foot_contact_force_norms * 0.01
# foot_force is int16 (Newtons). Scale by 0.01.
obs[57] = low_state.foot_force[0] * 0.01  # FL
obs[58] = low_state.foot_force[1] * 0.01  # FR
obs[59] = low_state.foot_force[2] * 0.01  # RL
obs[60] = low_state.foot_force[3] * 0.01  # RR
```

### 4b. Force Estimator Loading and Inference

Load from the training checkpoint (`.pt` file contains both policy and estimator):

```python
# 1. Load checkpoint
ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

# 2. Reconstruct policy actor (MLP: 127 → 512 → 256 → 128 → 12)
#    Use the exported JIT policy if available at exported/policy.pt
#    The JIT policy expects 127-dim input (already augmented)
policy = torch.jit.load(exported_policy_path)

# 3. Reconstruct force estimator from checkpoint
#    Architecture: encoder [1220→128→64], f_head [64→32→16→2]
#    Input: flattened history [20 * 61 = 1220]
#    Output: force_hat [2], latent [66] = concat(force_hat, z_t)
from go2_rl_lab.estimator.force_estimator import ForceEstimator
estimator = ForceEstimator(config)
estimator.load_state_dict(ckpt["force_estimator_state_dict"])
estimator.eval()
```

### 4c. Temporal History Buffer

```python
TEMPORAL_STEPS = 20
OBS_DIM = 61
history_buffer = np.zeros((TEMPORAL_STEPS, OBS_DIM), dtype=np.float32)

# Each step: shift buffer and insert new obs
history_buffer = np.roll(history_buffer, -1, axis=0)
history_buffer[-1] = obs

# Flatten for estimator input
history_flat = history_buffer.reshape(1, -1)  # [1, 1220]
```

### 4d. Augmented Observation Assembly

```python
# Run estimator
with torch.inference_mode():
    history_tensor = torch.from_numpy(history_flat)
    force_hat, latent = estimator.get_latent(history_tensor)
    # force_hat: [1, 2], latent: [1, 66]

# Optional compliance modulation
if compliance_k > 0.0:
    force_filtered = ema_alpha * force_hat + (1 - ema_alpha) * force_filtered
    f_mag = torch.norm(force_filtered)
    if f_mag > alpha:
        k = (1.0 / beta) * (1.0 + alpha / f_mag)
        obs[6] += k * force_filtered[0, 0].item()
        obs[7] += k * force_filtered[0, 1].item()

# Build augmented observation
obs_tensor = torch.from_numpy(obs).unsqueeze(0)  # [1, 61]
augmented_obs = torch.cat([obs_tensor, latent], dim=-1)  # [1, 127]

# Policy inference
action = policy(augmented_obs).detach().numpy().squeeze()
```

### 4e. CLI Arguments

```
--checkpoint     Path to force-only training checkpoint (model_XXXX.pt)
--compliance_k   Compliance gain (default 0.0 = disabled)
--alpha          Force threshold for compliance (default 10.0 N)
--beta           Virtual impedance (default 10.0)
--ema_alpha      EMA smoothing for force estimate (default 0.1)
```

### 4f. Foot Force Logging

Log foot forces every step for magnitude verification:

```python
debug_log.append({
    'step': step_count,
    'foot_force_raw': [low_state.foot_force[i] for i in range(4)],  # int16 Newtons
    'foot_force_obs': obs[57:61].copy(),  # After 0.01 scaling
    'force_estimate': force_hat.numpy().squeeze().copy(),  # Estimator output
    ...
})
```

Print periodic summary:

```
[step 50] foot_force_raw=[FL:45, FR:42, RL:38, RR:40] (N)
          foot_force_obs=[0.45, 0.42, 0.38, 0.40]
          force_est=[Fx:0.3, Fy:-0.1]
```

Save to `.npz` at exit (same pattern as existing script) with foot force arrays included.

### 4g. Foot Order Verification

**Critical**: The foot order in the bridge (FL=0, FR=1, RL=2, RR=3) must match the IsaacLab training order. The training env uses `body_names=".*_foot"` which gives alphabetical order: FL_foot, FR_foot, RL_foot, RR_foot. The touch sensors in the XML are defined in the same order (FL, FR, RL, RR). **Match confirmed.**

---

## Step 5: Verification

1. **Start unitree_mujoco** with the new XML:
   ```bash
   # Update config or XML path to use go2_force_sensor.xml
   cd ~/unitree_mujoco/simulate_python && python3 unitree_mujoco.py
   ```

2. **Run sim2sim deploy** with force-only policy:
   ```bash
   cd ~/go2_rl_lab/deploy/sim2sim
   python3 sim2sim_force_only_deploy.py \
       --checkpoint ~/go2_rl_lab/logs/rsl_rl/go2_force_only/<run>/model_XXXX.pt
   ```

3. **Check foot force magnitudes** in the logs:
   - Standing: ~25N per foot (10kg / 4 legs × 9.81)
   - Walking: 0-100N range, alternating per gait cycle
   - Compare with IsaacLab training values (similar range expected)

4. **Test compliance** (optional):
   ```bash
   python3 sim2sim_force_only_deploy.py \
       --checkpoint <path> --compliance_k 0.01 --alpha 10.0 --beta 10.0
   ```

---

## Expected Foot Force Magnitudes

| Scenario | Per-foot force (N) | Obs value (×0.01) |
|----------|-------------------|-------------------|
| Standing (4 legs) | ~25N | ~0.25 |
| Walking (stance leg) | 40-80N | 0.40-0.80 |
| Walking (swing leg) | 0N | 0.0 |
| Landing impact | 100-200N | 1.0-2.0 |

These should match IsaacLab training magnitudes. If they differ significantly, a scaling factor in the bridge or deploy script may be needed.
