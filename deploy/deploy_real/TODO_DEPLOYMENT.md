# Deployment Changes Required

These changes must be applied to `deploy_isaac_config_propioceptive.py` and `configs/go2_isaaclab.yaml`
to match the updated training configuration.

---

## 1. YAML: Update observation scales

**File:** `configs/go2_isaaclab.yaml`

```yaml
# OLD
ang_vel_scale: 1.0
dof_vel_scale: 1.0
num_obs: 45

# NEW
ang_vel_scale: 0.25
dof_vel_scale: 0.05
num_obs: 225
```

---

## 2. Python: Apply observation scales in obs assembly

**File:** `deploy_isaac_config_propioceptive.py`, inside `run()` method (around line 460)

Currently the obs are assembled without scales:
```python
self.obs[:3] = ang_vel                                              # line 460
self.obs[9 + num_actions : 9 + num_actions * 2] = dqj_obs          # line 464
```

Must become:
```python
self.obs[:3] = ang_vel * self.config.ang_vel_scale                  # apply 0.25
self.obs[9 + num_actions : 9 + num_actions * 2] = dqj_obs * self.config.dof_vel_scale  # apply 0.05
```

---

## 3. Python: Add observation history buffer (history_length=5)

The policy now expects 5 stacked observations (225 dims = 5 x 45) instead of a single
45-dim observation. The deployment code must maintain a ring buffer.

### 3a. In `__init__()`, after `self.obs = np.zeros(...)` (around line 65):

Add a history buffer. The single-frame obs size is 45, history length is 5:
```python
self.obs_size_single = 45
self.history_length = 5
self.obs_history = np.zeros((self.history_length, self.obs_size_single), dtype=np.float32)
```

Note: `self.obs` stays as `np.zeros(45)` — it is still used to assemble the current frame.

### 3b. In `run()`, AFTER building `self.obs` (after line 465) and BEFORE feeding the policy (line 470):

Shift the history buffer and append the current observation:
```python
# Shift history: drop oldest, append current
self.obs_history = np.roll(self.obs_history, -1, axis=0)
self.obs_history[-1] = self.obs

# Flatten to (1, 225) for the policy — oldest first, newest last
obs_tensor = torch.from_numpy(self.obs_history.flatten()).unsqueeze(0)
```

This replaces the current lines:
```python
# OLD (lines 470-471)
obs_tensor = torch.from_numpy(self.obs).unsqueeze(0)
self.action = self.policy(obs_tensor).detach().numpy().squeeze()
```

With:
```python
# NEW
self.obs_history = np.roll(self.obs_history, -1, axis=0)
self.obs_history[-1] = self.obs
obs_tensor = torch.from_numpy(self.obs_history.flatten()).unsqueeze(0)
self.action = self.policy(obs_tensor).detach().numpy().squeeze()
```

### 3c. Verify history ordering

IsaacLab's observation manager stacks history as [oldest, ..., newest] in the
concatenated tensor. The `np.roll(..., -1)` + assign to `[-1]` pattern matches this:
- `obs_history[0]` = oldest (t-4)
- `obs_history[4]` = newest (t)
- flattened = [t-4 obs (45 dims), t-3 obs (45 dims), ..., t obs (45 dims)] = 225 dims

---

## 4. YAML comment: Update obs description

Update the comment in the yaml to reflect the new format:

```yaml
# OLD
# Obs order: base_ang_vel(3), projected_gravity(3), velocity_commands(3), joint_pos_rel(12), joint_vel_rel(12), last_action(12) = 45

# NEW
# Single frame (45 dims): base_ang_vel(3)*0.25, projected_gravity(3), velocity_commands(3), joint_pos_rel(12), joint_vel_rel(12)*0.05, last_action(12)
# Policy input: 5 stacked frames = 225 dims [oldest_frame(45) ... newest_frame(45)]
```

---

## Summary of all changes

| What | Where | Change |
|------|-------|--------|
| `ang_vel_scale` | yaml | `1.0` -> `0.25` |
| `dof_vel_scale` | yaml | `1.0` -> `0.05` |
| `num_obs` | yaml | `45` -> `225` |
| Apply ang_vel scale | Python line ~460 | `ang_vel` -> `ang_vel * config.ang_vel_scale` |
| Apply dof_vel scale | Python line ~464 | `dqj_obs` -> `dqj_obs * config.dof_vel_scale` |
| Add history buffer | Python `__init__` | `self.obs_history = np.zeros((5, 45))` |
| Stack observations | Python `run()` | Roll buffer, flatten to 225 dims before policy |
| Update obs comment | yaml | Document new format |
