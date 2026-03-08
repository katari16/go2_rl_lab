# Low-Level Locomotion Finetuning Experiments

Systematic hyperparameter sweep for improving gait quality of the stage 1 low-level policy.
Based on insights from Kyle Morgenstein's deployment-ready RL talk and community best practices.

## Baseline

- **Checkpoint**: `logs/rsl_rl/go2_compliant_no_foot_xyz/2026-03-05_07-58-04/model_16400.pt`
- **Task**: `Go2-Compliant-NoFoot-XYZ-v0`
- **Current PD gains**: Kp=25, Kd=0.5 (uniform, all joints)
- **Action scale**: 0.25
- **Torque limit**: 23.5 Nm

---

## 1. PD Gains Analysis

Kyle's heuristic: `Kp = torque_limit / RoM` (low) or `Kp = torque_limit / (RoM/2)` (high), `Kd = Kp / 20`.

| Joint     | RoM (rad) | Kp_low (τ/RoM) | Kp_high (τ/(RoM/2)) | Current Kp | Status           |
|-----------|-----------|-----------------|----------------------|------------|------------------|
| Hip       | 1.885     | 12.5            | 24.9                 | 25         | At high end      |
| F. Thigh  | 4.555     | 5.2             | 10.3                 | 25         | **Way too high** |
| R. Thigh  | 4.555     | 5.2             | 10.3                 | 25         | **Way too high** |
| Calf      | 1.697     | 13.9            | 27.7                 | 25         | At high end      |

Thigh joints at Kp=25 with RoM=4.555: a 1 rad deviation = 25 Nm > torque limit (23.5 Nm). Bang-bang territory.

Community: "for RL you want gains so soft it can't support the robot's weight under gravity."
Kp=8 reportedly gives better tracking, smoother accel, less torque, better foot lifting, 3x more compliance.

### PD Gain Presets

| Preset | Hip Kp/Kd | Thigh Kp/Kd | Calf Kp/Kd | Notes |
|--------|-----------|-------------|-------------|-------|
| **A: Uniform low** | 8 / 0.4 | 8 / 0.4 | 8 / 0.4 | Community sweet spot, max exploration |
| **B: Per-joint** | 12 / 0.6 | 8 / 0.4 | 14 / 0.7 | Kyle's τ/RoM heuristic |
| **C: Baseline** | 25 / 0.5 | 25 / 0.5 | 25 / 0.5 | Current (control group) |

### Action Scale

With lower Kp, effective torque per unit action is lower. May need to compensate:
- **0.25** — current (conservative)
- **0.5** — double exploration bandwidth

---

## 2. Reward Analysis: Current vs Literature

### Current Reward Set (15 terms)

| # | Reward | Function | Weight | Category |
|---|--------|----------|--------|----------|
| 1 | `track_lin_vel_xy_exp` | compliant_track_lin_vel_xy_exp | +1.5 | Task |
| 2 | `track_ang_vel_z_exp` | mdp.track_ang_vel_z_exp | +0.75 | Task |
| 3 | `lin_vel_z_l2` | mdp.lin_vel_z_l2 | -2.0 | Penalty |
| 4 | `ang_vel_xy_l2` | mdp.ang_vel_xy_l2 | -0.05 | Penalty |
| 5 | `flat_orientation_l2` | mdp.flat_orientation_l2 | -2.5 | Penalty |
| 6 | `dof_acc_l2` | mdp.joint_acc_l2 | -2.5e-7 | Penalty |
| 7 | `feet_air_time` | mdp.feet_air_time (thresh=0.5) | +0.25 | Gait |
| 8 | `energy` | energy (|τ·q̇|) | -2e-5 | Penalty |
| 9 | `undesired_contacts` | mdp.undesired_contacts | -1.0 | Penalty |
| 10 | `feet_slide` | mdp.feet_slide | -0.1 | Gait |
| 11 | `action_smoothness_2` | action_smoothness_2 (2nd order) | -0.01 | Penalty |
| 12 | `pose_similarity` | pose_similarity (joint→default) | -0.1 | Penalty |
| 13 | `feet_clearance` | feet_clearence_dense (target=0.10m) | -0.5 | Gait |
| 14 | `feet_too_near` | feet_too_near (thresh=0.20m) | -1.0 | Gait |
| 15 | `soft_landing` | soft_landing | -1e-3 | Gait |
| 16 | `dof_pos_limits` | mdp.joint_pos_limits | -10.0 | Penalty |

### What's Missing (from both reference papers)

| Reward | Paper 1 | Deep Compliant | Impact | Priority |
|--------|---------|---------------|--------|----------|
| **Trunk height** | exp(-‖h*-h‖²/0.01) w=0.5 | included in pose | Prevents crouching, stabilizes CoM | HIGH |
| **Action rate-1** (1st order) | ‖a_{t-1}-a_t‖² w=-0.01 | — | Smooths joint transitions | HIGH |
| **Torque L2** | ‖τ‖² w=-1e-4 | ‖τ‖² w=-0.0015 | Complementary to energy, reduces peak torques | MEDIUM |
| **Stable stride** | V(β)/E(β) w=1.0 | — | Consistent gait timing | MEDIUM |
| **Power distribution** | V(τq̇) per leg group w=-1e-10 | — | Even power across joints | LOW |

### What Could Be Adjusted

| Current Reward | Issue | Proposed Change | Rationale |
|---------------|-------|----------------|-----------|
| `feet_air_time` w=0.25 | Very low vs paper (1.0) | Increase to 0.5-1.0 | Stronger incentive for proper swing phases |
| `feet_clearance` w=-0.5 (penalty) | Paper uses positive reward form | Try `foot_clearance_reward` w=+0.3 (exp kernel) | Positive reward encourages active stepping vs. penalty just avoiding deviation |
| `flat_orientation_l2` w=-2.5 | Higher than paper (w=-1.0) | Try w=-1.5 | Current may over-constrain, limiting agility |
| `pose_similarity` w=-0.1 | Always active, biases toward stance | Try w=-0.05 or condition on zero-cmd only | May limit expressiveness of gait at speed |
| `feet_too_near` w=-1.0 | Quite aggressive | Try w=-0.5 | May be preventing natural gait patterns |
| `energy` w=-2e-5 | Very small | Try w=-5e-5 | Stronger energy minimization for efficiency |

### What Could Be Replaced/Removed

| Current Reward | Consider | Rationale |
|---------------|----------|-----------|
| `feet_clearance` (dense penalty) | Replace with `foot_clearance_reward` (positive exp) | Positive reward = more active stepping. Both exist in codebase. |
| `pose_similarity` (always on) | Replace with `stand_still` (cmd-gated) | Only penalize joint deviation when standing, not while walking |
| `soft_landing` w=-1e-3 | Increase to w=-0.01 or replace with `foot_height_sparse` | Current weight is negligible. Sparse foot height penalty directly shapes swing peak. |

### Available Reward Functions (already in codebase, not currently used)

These are implemented and ready to use:

| Function | File | Description |
|----------|------|-------------|
| `base_height_l2` | isaaclab mdp | L2 penalty on height deviation from target |
| `base_pose_penalty` | custom rewards.py | φ² + ψ² + 10·(y-y_des)² (Deep Compliant style) |
| `action_rate_l2` | isaaclab mdp | 1st-order action smoothness |
| `joint_torques_l2` | isaaclab mdp | L2 torque penalty |
| `foot_clearance_reward` | custom rewards.py | Positive exp kernel for swing clearance |
| `foot_height_sparse` | custom rewards.py | (p_peak/p_des - 1)² at touchdown |
| `air_time_variance_penalty` | custom rewards.py | Penalize uneven air/contact times across legs |
| `feet_gait` | custom rewards.py | Phase-based gait reward (trot, pace, etc.) |
| `joint_mirror` | custom rewards.py | Penalize asymmetry between left/right legs |
| `stand_still` | custom rewards.py | Joint-to-default penalty only when cmd≈0 |
| `body_lin_acc_l2` | isaaclab mdp | Penalize body linear acceleration (smoothness) |
| `joint_vel_limits` | isaaclab mdp | Soft penalty near velocity limits |
| `is_alive` | isaaclab mdp | Constant +1 bonus for surviving (alive bonus) |

---

## 3. Proposed Reward Presets

### Preset R1: "Baseline" (current, no changes)
Keep all 16 rewards as-is. Control group.

### Preset R2: "Enhanced Smoothness"
Focus: smoother gaits via 1st-order action rate + torque penalty + height tracking.

Changes from baseline:
```
ADD    base_height_l2          w=-0.5   target_height=0.34
ADD    action_rate_l2          w=-0.01
ADD    joint_torques_l2        w=-1e-4
CHANGE feet_air_time           w=0.25 -> 0.75
```

### Preset R3: "Active Gait"
Focus: encourage active, well-defined stepping patterns.

Changes from baseline:
```
ADD    base_height_l2              w=-0.5   target_height=0.34
ADD    action_rate_l2              w=-0.01
ADD    air_time_variance_penalty   w=-0.5   (penalize uneven stride timing)
CHANGE feet_air_time               w=0.25 -> 1.0
REPLACE feet_clearance (penalty -0.5) WITH foot_clearance_reward (positive +0.3, std=0.02, tanh=5.0)
CHANGE flat_orientation_l2         w=-2.5 -> -1.5
CHANGE pose_similarity             w=-0.1 -> -0.05
```

### Preset R4: "Deep Compliant Style"
Focus: mimic the Deep Compliant Control paper reward structure.

Changes from baseline:
```
ADD    base_pose_penalty       w=-2.0   desired_height=0.34  (replaces flat_orientation_l2)
ADD    action_rate_l2          w=-0.01
ADD    joint_torques_l2        w=-1.5e-3  (paper uses -0.0015)
REPLACE feet_clearance WITH foot_height_sparse  w=-0.7  target_height=0.10
REMOVE flat_orientation_l2     (subsumed by base_pose_penalty)
CHANGE energy                  w=-2e-5 -> -1.5e-2  (paper uses -0.015)
CHANGE track_lin_vel weight    w=1.5 -> 0.8  (paper uses 0.8)
CHANGE feet_air_time           w=0.25 -> 0.5
```

---

## 4. Experiment Matrix (8 jobs)

Crossing 3 gain presets x 4 reward presets = 12 combinations is too many.
Select the most informative 8:

| Job | Gains | Action Scale | Rewards | What it tests |
|-----|-------|-------------|---------|---------------|
| 1 | A: Kp=8 uniform | 0.25 | R1: Baseline | Pure low-gains effect |
| 2 | A: Kp=8 uniform | 0.5 | R1: Baseline | Low gains + wider exploration |
| 3 | B: Per-joint | 0.5 | R1: Baseline | Principled gains |
| 4 | A: Kp=8 uniform | 0.5 | R2: Enhanced Smooth | Low gains + smoothness rewards |
| 5 | B: Per-joint | 0.5 | R3: Active Gait | Best gains + active stepping |
| 6 | C: Kp=25 baseline | 0.25 | R2: Enhanced Smooth | Control: only reward changes |
| 7 | A: Kp=8 uniform | 0.5 | R3: Active Gait | Low gains + active gait |
| 8 | A: Kp=8 uniform | 0.5 | R4: Deep Compliant | Low gains + paper-style rewards |

### Priority if limited to 6 jobs: drop jobs 3 and 6.

---

## 5. Implementation Strategy

**Recommended: Option A — Separate env_cfg files per variation.**

Each job gets its own `go2_finetune_j{N}_env_cfg.py` registered as `Go2-Finetune-J{N}-v0`.
Training command: `python train.py --task Go2-Finetune-J{N}-v0 --max_iterations 25000`

### Robot Asset Variants (3 files in `assets/`)

```python
# assets/unitree.py additions:

# Variant A: Uniform low gains
UNITREE_GO2_LOW_GAIN_CFG = UNITREE_GO2_CFG.replace(
    actuators={
        "base_legs": DCMotorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit=23.5, saturation_effort=23.5, velocity_limit=30.0,
            stiffness=8.0, damping=0.4, friction=0.0,
        ),
    },
)

# Variant B: Per-joint heuristic gains
UNITREE_GO2_PERJOINT_CFG = UNITREE_GO2_CFG.replace(
    actuators={
        "hips": DCMotorCfg(
            joint_names_expr=[".*_hip_joint"],
            effort_limit=23.5, saturation_effort=23.5, velocity_limit=30.0,
            stiffness=12.0, damping=0.6, friction=0.0,
        ),
        "thighs": DCMotorCfg(
            joint_names_expr=[".*_thigh_joint"],
            effort_limit=23.5, saturation_effort=23.5, velocity_limit=30.0,
            stiffness=8.0, damping=0.4, friction=0.0,
        ),
        "calves": DCMotorCfg(
            joint_names_expr=[".*_calf_joint"],
            effort_limit=23.5, saturation_effort=23.5, velocity_limit=30.0,
            stiffness=14.0, damping=0.7, friction=0.0,
        ),
    },
)

# Variant C: Baseline (existing UNITREE_GO2_CFG, Kp=25, Kd=0.5)
```

### Env Config Pattern

Each env_cfg inherits from the baseline and overrides only what changes:

```python
# Example: go2_finetune_j4_env_cfg.py (Job 4: Kp=8, scale=0.5, R2 rewards)

from .go2_compliant_no_foot_xyz_env_cfg import *  # inherit everything
from go2_rl_lab.assets.unitree import UNITREE_GO2_LOW_GAIN_CFG

class UnitreeGo2FinetuneJ4EnvCfg(UnitreeGo2CompliantNoFootXyzEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Override robot asset (low gains)
        self.scene.robot = UNITREE_GO2_LOW_GAIN_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Override action scale
        self.actions.joint_pos.scale = 0.5

        # Override/add rewards (R2: Enhanced Smoothness)
        self.rewards.base_height_l2 = RewTerm(
            func=mdp.base_height_l2, weight=-0.5,
            params={"target_height": 0.34},
        )
        self.rewards.action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
        self.rewards.joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1e-4)
        self.rewards.feet_air_time.weight = 0.75
```

---

## 6. Training Notes

- Train from scratch for PD gain changes (jobs 1-5, 7-8) — policy learned Kp=25 dynamics
- Can resume from baseline checkpoint for reward-only changes (job 6)
- Train for **25,000 iterations** minimum (Kyle: 2-5x longer than apparent convergence)
- Current best was 16,400 iters — likely undertrained for regularizer convergence
- **Log actor std deviation** — most critical metric. Must be decreasing and converging.
- Keep net reward positive (add `is_alive` w=+0.5 if needed, especially for R4 with high penalties)
- Use 4096 envs (current default)

---

## 7. Evaluation Criteria

After training, compare policies on:

1. **Velocity tracking error** (lin + ang) — primary task metric
2. **Joint acceleration RMS** — smoothness indicator
3. **Torque RMS** — energy efficiency
4. **Foot clearance** — swing height quality
5. **Stride regularity** — variance in step timing
6. **Survival rate under force** — robustness (use existing eval scripts)
7. **Actor std convergence** — training quality indicator
8. **Sim2sim transfer** — does it look good in MuJoCo?

Use existing eval scripts:
- `scripts/rsl_rl/static_eval.py` — static force evaluation
- `scripts/rsl_rl/dynamic_eval.py` — dynamic evaluation
- `scripts/rsl_rl/eval/push_recovery.py` — push recovery
- `deploy/sim2sim/sim2sim_compliant_no_foot_xyz.py` — MuJoCo transfer check
