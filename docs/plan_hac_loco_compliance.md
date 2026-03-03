# Plan: HAC-LOCO Freeze+Extend Compliance Training

## Context

The force-only policy (`Go2-Force-Only-v0`) and its force estimator are trained in stage 1 on flat ground (and optionally terrain via `Go2-Force-Only-Terrain-v0`). The policy learns robust locomotion under persistent XY forces, and the estimator learns to predict body-frame XY forces from proprioceptive history. Currently, compliance is achieved via a simple linear velocity command modulation (`v* = v + k·F̂`), which is runtime-tunable but fixed in form and subject to noisy force estimates.

The HAC-LOCO paper proposes a **two-stage hierarchical architecture** where:
- **Stage 1** (already done): Train a low-level locomotion policy + force estimator
- **Stage 2** (this plan): **Freeze** the low-level policy and estimator, then train a lightweight **high-level compliance policy** that outputs residual velocity commands `a' = [Δv_x, Δv_y, Δω_z]`

This avoids catastrophic forgetting (the low-level policy never changes) and allows retraining the compliance module with different α/β parameters without touching the base policy.

## Branch

`feature/compliant-rl-finetuning` (to be created from `feature/force-velocity-modulation`)

---

## Architecture Overview

```
                    ┌─────────────────────────────────┐
                    │    High-Level Compliance Policy  │
                    │    π'(o'_t) → a' = [Δvx,Δvy,Δωz]│
                    │    MLP: obs_dim' → [128,64] → 3 │
                    └──────────┬──────────────────────┘
                               │ a' (residual vel cmd)
                               ▼
                    c* = c + a'  (modified velocity command)
                               │
                    ┌──────────▼──────────────────────┐
                    │    Low-Level Policy (FROZEN)     │
                    │    π(o_t, l_t) → 12 joint actions│
                    │    MLP: 127 → [512,256,128] → 12│
                    └──────────┬──────────────────────┘
                               │ joint position targets
                               ▼
                          PD Controller → Robot
```

**High-level policy observation** (from HAC-LOCO eq. 3):
```
o'_t = [s_t, c_t, l_{t-1}, a'_{t-1}, a_{t-1}]
```
Where:
- `s_t` = proprioceptive feedback [ang_vel(3), proj_gravity(3), joint_pos(12), joint_vel(12), applied_torque(12), foot_forces(4)] = **46 dims**
- `c_t` = velocity commands [vx, vy, ωz] = **3 dims**
- `l_{t-1}` = previous estimator latent [f̂(2) + z_t(64)] = **66 dims**
- `a'_{t-1}` = previous high-level action [Δvx, Δvy, Δωz] = **3 dims**
- `a_{t-1}` = previous low-level action [12 joint positions] = **12 dims**
- **Total: 130 dims**

**High-level policy action**: `a' = [Δv_x, Δv_y, Δω_z]` → **3 dims**

**Compliance reward** (HAC-LOCO eq. 7):
```
r_comp = {
  -||a'||²                                                    if ||f|| ≤ α
  -||a'_xy - f_xy/β · (1 + α/|f_xy|)||² - ||a'_z||²         if ||f|| > α
}
```

---

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `estimator/compliance_runner.py` | **CREATE** | Stage 2 runner: loads frozen stage-1 checkpoint, trains high-level compliance policy |
| `estimator/compliance_env_wrapper.py` | **CREATE** | Wraps env with frozen low-level policy; builds high-level obs; applies `c* = c + a'` |
| `estimator/compliance_ppo.py` | **CREATE** | PPO with compliance reward computed per mini-batch |
| `tasks/.../go2_compliance_env_cfg.py` | **CREATE** | Env config for stage 2 (same obs/scene as force-only, but with compliance reward) |
| `tasks/.../agents/rsl_rl_compliance_cfg.py` | **CREATE** | Agent config for compliance runner |
| `tasks/.../mdp/rewards.py` | **MODIFY** | Add `compliance_reward` function |
| `tasks/.../__init__.py` | **MODIFY** | Register `Go2-Compliance-v0` |
| `scripts/rsl_rl/play.py` | **MODIFY** | Add `ComplianceOnPolicyRunner` to runner dispatch |

---

## Step 1: Compliance Environment Wrapper

**File**: `source/go2_rl_lab/go2_rl_lab/estimator/compliance_env_wrapper.py`

This wrapper makes the frozen low-level policy + estimator part of the environment. The high-level policy sees a different observation space and action space than the original env.

### Key Design

```python
class ComplianceEnvWrapper:
    """Wraps env with frozen low-level policy and estimator.

    The high-level policy sees:
        obs:     o'_t = [s_t, c_t, l_{t-1}, a'_{t-1}, a_{t-1}]  (130 dims)
        action:  a' = [Δvx, Δvy, Δωz]  (3 dims)

    On step(a'):
        1. Modify velocity command: c* = c + a'
        2. Build augmented low-level obs (with c* injected)
        3. Run frozen low-level policy: a = π(o_t_augmented)
        4. Step the underlying env with a
        5. Compute compliance reward
        6. Build next high-level obs o'_{t+1}
    """
```

### Implementation Details

**__init__**:
```python
def __init__(self, env, frozen_policy, frozen_estimator,
             history_buffer, alpha, beta, device):
    self._env = env
    self.frozen_policy = frozen_policy   # ActorCritic (frozen, eval mode)
    self.frozen_estimator = frozen_estimator  # ForceEstimator (frozen)
    self.history_buffer = history_buffer  # ObsHistoryBuffer
    self.alpha = alpha       # Force threshold (N)
    self.beta = beta         # Virtual impedance
    self.device = device

    # High-level action/obs dims
    self.num_actions = 3     # [Δvx, Δvy, Δωz]
    # s_t(46) + c_t(3) + l_{t-1}(66) + a'_{t-1}(3) + a_{t-1}(12) = 130
    self.num_obs = 130
    self.num_privileged_obs = None  # Critic obs set separately

    # State buffers
    self._last_latent = None         # l_{t-1}: [num_envs, 66]
    self._last_high_action = None    # a'_{t-1}: [num_envs, 3]
    self._last_low_action = None     # a_{t-1}: [num_envs, 12]
    self._gt_force = None            # For compliance reward: [num_envs, 2]
```

**step(high_level_action)**:
```python
def step(self, high_level_action):  # a': [num_envs, 3]
    # 1. Get raw env obs (un-augmented)
    raw_obs = self._env.unwrapped.observation_manager.compute()
    raw_policy = raw_obs["policy"]  # [num_envs, 61]

    # 2. Modify velocity commands: c* = c + a'
    #    raw_policy[6:9] = velocity commands
    modified_policy = raw_policy.clone()
    modified_policy[:, 6:9] = modified_policy[:, 6:9] + high_level_action

    # 3. Update history buffer with modified obs and get estimator latent
    self.history_buffer.insert(modified_policy)
    force_hat, latent = self.frozen_estimator.get_latent(
        self.history_buffer.get_flattened()
    )

    # 4. Build augmented low-level obs: [modified_policy, latent]
    low_level_obs = torch.cat([modified_policy, latent], dim=-1)  # [num_envs, 127]

    # 5. Run frozen low-level policy
    with torch.inference_mode():
        low_level_action = self.frozen_policy.act_inference(low_level_obs)

    # 6. Step underlying env with low-level actions
    obs, rewards, dones, extras = self._env.step(low_level_action)

    # 7. Extract GT force from critic obs for compliance reward
    critic_obs = obs.get("critic", raw_obs.get("critic"))
    self._gt_force = critic_obs[:, 64:66]  # base_applied_force_xy

    # 8. Compute compliance reward
    compliance_rew = self._compute_compliance_reward(
        high_level_action, self._gt_force
    )
    # Blend with base locomotion reward
    total_reward = rewards + compliance_reward_weight * compliance_rew

    # 9. Store state for next high-level obs
    self._last_latent = latent.detach()
    self._last_high_action = high_level_action.detach()
    self._last_low_action = low_level_action.detach()

    # 10. Reset buffers for terminated envs
    done_ids = dones.nonzero(as_tuple=False).flatten()
    self.history_buffer.reset(done_ids)
    if len(done_ids) > 0:
        self._last_latent[done_ids] = 0.0
        self._last_high_action[done_ids] = 0.0
        self._last_low_action[done_ids] = 0.0

    # 11. Build next high-level obs
    high_obs = self._build_high_level_obs(obs)

    return high_obs, total_reward, dones, extras
```

**_compute_compliance_reward**:
```python
def _compute_compliance_reward(self, a_prime, gt_force):
    """HAC-LOCO compliance reward (eq. 7).

    Args:
        a_prime: [num_envs, 3] — high-level action [Δvx, Δvy, Δωz]
        gt_force: [num_envs, 2] — GT XY force (Newtons)
    """
    f_norm = gt_force.norm(dim=-1)  # [num_envs]

    # Case 1: ||f|| ≤ α → minimize adjustments
    small_force = f_norm <= self.alpha
    r_small = -(a_prime ** 2).sum(dim=-1)  # -||a'||²

    # Case 2: ||f|| > α → comply proportionally
    f_xy = gt_force  # [num_envs, 2]
    f_xy_norm = f_xy.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    target_a_xy = (f_xy / self.beta) * (1.0 + self.alpha / f_xy_norm)
    r_large_xy = -((a_prime[:, :2] - target_a_xy) ** 2).sum(dim=-1)
    r_large_z = -(a_prime[:, 2] ** 2)
    r_large = r_large_xy + r_large_z

    r_comp = torch.where(small_force, r_small, r_large)
    return r_comp
```

**_build_high_level_obs**:
```python
def _build_high_level_obs(self, raw_obs):
    """Build o'_t = [s_t, c_t, l_{t-1}, a'_{t-1}, a_{t-1}]."""
    raw_policy = raw_obs["policy"]  # [num_envs, 61]

    # s_t: proprioceptive (exclude velocity commands and last_action from raw)
    # raw_policy layout: [ang_vel(3), gravity(3), vel_cmd(3), jpos(12), jvel(12), last_act(12), torque(12), foot(4)]
    s_t = torch.cat([
        raw_policy[:, 0:6],    # ang_vel + gravity (6)
        raw_policy[:, 9:33],   # joint_pos + joint_vel (24)
        raw_policy[:, 45:61],  # applied_torque + foot_forces (16)
    ], dim=-1)  # 46 dims

    c_t = raw_policy[:, 6:9]  # velocity commands (3 dims)

    high_obs = torch.cat([
        s_t,                        # 46
        c_t,                        # 3
        self._last_latent,          # 66
        self._last_high_action,     # 3
        self._last_low_action,      # 12
    ], dim=-1)  # Total: 130

    return TensorDict({"policy": high_obs}, batch_size=...)
```

### Critical: VecEnv Interface Compatibility

The wrapper must expose `num_obs`, `num_actions`, `num_envs`, `num_privileged_obs`, `get_observations()`, `step()`, `reset()` — matching the rsl_rl VecEnv interface so that OnPolicyRunner can use it directly.

For the critic, two options:
- **Option A (simpler)**: Same obs as actor (130 dims). The critic doesn't need GT force because the compliance reward is already shaped by it.
- **Option B (privileged, like HAC-LOCO)**: Critic gets `[o'_t, v_t, d_t, f_H, h_t]` with privileged info. This requires a separate critic obs group.

**Recommendation**: Start with Option A for simplicity. The high-level policy is small and trains fast (~2M samples); privileged critic can be added later if needed.

---

## Step 2: Compliance PPO

**File**: `source/go2_rl_lab/go2_rl_lab/estimator/compliance_ppo.py`

This is a thin extension of standard PPO. Since the compliance reward is computed inside the env wrapper (step 1), the PPO algorithm itself doesn't need modifications. We can use the standard `PPO` class from rsl_rl.

However, if we want to log compliance-specific metrics (force magnitude, compliance reward breakdown), we need a thin wrapper:

```python
class CompliancePPO(PPO):
    """Standard PPO with compliance reward logging."""

    def __init__(self, alpha, beta, **ppo_kwargs):
        super().__init__(**ppo_kwargs)
        self.alpha = alpha
        self.beta = beta

    # update() is standard PPO — no changes needed
    # Logging handled by the runner
```

**Alternatively**: Just use standard `PPO` directly and handle logging in the runner. This is simpler and avoids a new file.

**Decision**: Use standard PPO. No `compliance_ppo.py` needed. One less file.

---

## Step 3: Compliance Runner

**File**: `source/go2_rl_lab/go2_rl_lab/estimator/compliance_runner.py`

The runner orchestrates stage 2 training:
1. Loads a stage-1 checkpoint (frozen low-level policy + estimator)
2. Creates the ComplianceEnvWrapper
3. Creates a small ActorCritic for the high-level policy
4. Runs standard PPO training

### Key Design

```python
class ComplianceOnPolicyRunner:
    """Stage 2 runner: trains high-level compliance policy with frozen low-level.

    Config keys (under "compliance"):
        stage1_checkpoint:    Path to stage-1 .pt file
        alpha:                Force threshold (N) for compliance reward
        beta:                 Virtual impedance for compliance reward
        compliance_reward_weight:  Weight for r_comp in total reward
        high_level_hidden_dims:    MLP dims for high-level policy (default: [128, 64])
    """
```

**__init__**:
```python
def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
    comp_cfg = train_cfg.get("compliance", {})

    # 1. Load stage-1 checkpoint
    stage1_path = comp_cfg["stage1_checkpoint"]
    ckpt = torch.load(stage1_path, weights_only=False, map_location=device)

    # 2. Reconstruct frozen low-level policy (ActorCritic)
    #    Need to know the obs dims from stage 1: 127 actor input, 66 critic input, 12 actions
    frozen_policy = ActorCritic(
        num_actor_obs=127,   # 61 raw + 66 latent
        num_critic_obs=66,
        num_actions=12,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    ).to(device)
    frozen_policy.load_state_dict(ckpt["model_state_dict"])
    frozen_policy.eval()
    for p in frozen_policy.parameters():
        p.requires_grad = False

    # 3. Reconstruct frozen estimator
    est_cfg = train_cfg.get("estimator", {})
    frozen_estimator = ForceEstimator(
        temporal_steps=est_cfg.get("temporal_steps", 20),
        num_one_step_obs=61,
        enc_hidden_dims=est_cfg.get("enc_hidden_dims", [128, 64]),
        f_head_dims=est_cfg.get("f_head_dims", [32, 16]),
        force_dim=est_cfg.get("force_dim", 2),
        dec_hidden_dims=est_cfg.get("dec_hidden_dims", [256, 128]),
    ).to(device)
    frozen_estimator.load_state_dict(ckpt["force_estimator_state_dict"])
    frozen_estimator.eval()
    for p in frozen_estimator.parameters():
        p.requires_grad = False

    # 4. Create history buffer
    history_buffer = ObsHistoryBuffer(
        num_envs=env.num_envs,
        temporal_steps=est_cfg.get("temporal_steps", 20),
        obs_dim=61,
        device=device,
    )

    # 5. Wrap env with frozen policy
    self._wrapped_env = ComplianceEnvWrapper(
        env=env,
        frozen_policy=frozen_policy,
        frozen_estimator=frozen_estimator,
        history_buffer=history_buffer,
        alpha=comp_cfg.get("alpha", 10.0),
        beta=comp_cfg.get("beta", 10.0),
        compliance_reward_weight=comp_cfg.get("compliance_reward_weight", 1.0),
        device=device,
    )

    # 6. Create small high-level ActorCritic
    #    obs: 130 dims, actions: 3 dims
    hl_hidden = comp_cfg.get("high_level_hidden_dims", [128, 64])
    self.high_level_policy = ActorCritic(
        num_actor_obs=130,
        num_critic_obs=130,  # Option A: same as actor
        num_actions=3,
        actor_hidden_dims=hl_hidden,
        critic_hidden_dims=hl_hidden,
        activation="elu",
        init_noise_std=0.5,  # Lower than stage 1 — smaller action space
    ).to(device)

    # 7. Create standard PPO
    self.alg = PPO(
        policy=self.high_level_policy,
        device=device,
        **train_cfg["algorithm"],
    )

    # 8. Init storage
    self.alg.init_storage(
        env.num_envs,
        train_cfg["runner"]["num_steps_per_env"],
        [130],   # obs dims
        [130],   # critic obs dims
        [3],     # action dims
    )
```

**learn()**: Standard OnPolicyRunner learn loop (collect rollouts → compute returns → PPO update). Can extend OnPolicyRunner or implement directly.

**save/load**: Save only the high-level policy weights + stage1 checkpoint path for reproducibility.

### Important: Velocity Command Modification

The high-level action `a'` modifies the velocity command **before** the low-level policy sees it. The key question is: should we modify the obs or the command manager?

**Approach**: Modify the raw obs tensor directly. The velocity command is at indices `[6:9]` in the raw policy obs. This is the simplest approach — no need to touch the command manager.

```python
# In ComplianceEnvWrapper.step():
modified_policy_obs = raw_policy_obs.clone()
modified_policy_obs[:, 6] += a_prime[:, 0]  # Δvx
modified_policy_obs[:, 7] += a_prime[:, 1]  # Δvy
modified_policy_obs[:, 8] += a_prime[:, 2]  # Δωz
```

### Important: Reward Structure for Stage 2

From HAC-LOCO, the high-level policy uses **modified velocity tracking rewards** (eqs. 5-6):
```
r_lin_vel = exp(-4 · ||v_xy_cmd + a'_xy - v_xy_actual||²)
r_ang_vel = exp(-4 · ||ω_z_cmd + a'_z - ω_z_actual||²)
```

These replace the standard velocity tracking rewards. The low-level policy handles the tracking — the high-level's modified tracking reward measures whether the **combined** command `c + a'` is tracked.

Plus the compliance reward `r_comp` from above.

**Total stage-2 reward**:
```
r_total = r_base_locomotion + w_comp * r_comp
```

Where `r_base_locomotion` comes from the existing env rewards (already includes modified velocity tracking through the low-level policy tracking `c*`).

---

## Step 4: Environment Config

**File**: `source/go2_rl_lab/go2_rl_lab/tasks/manager_based/go2_rl_lab/go2_compliance_env_cfg.py`

This can be identical to `go2_force_only_env_cfg.py` (or the terrain variant). The compliance reward and frozen policy are handled by the wrapper, not the env config. The env config just needs to:
1. Have the same observation structure (61 policy obs, 66 critic obs with GT force)
2. Apply persistent XY forces (always active, not gated — stage 2 assumes forces are present)
3. Include all standard locomotion rewards

Key changes from `go2_force_only_env_cfg.py`:
- `persistent_xy_force` starts active: `force_range: (0.0, 20.0)` — no activation gate needed
- Force magnitude set to match stage-1 training: same `max_force`

**Alternatively**: Reuse `go2_force_only_env_cfg.py` directly and configure force activation in the runner. This avoids a new env config file.

**Decision**: Reuse the existing `UnitreeGo2ForceOnlyEnvCfg` (or terrain variant). Set force_range to (0.0, max_force) in the runner's __init__. No new env config file needed.

---

## Step 5: Agent Config

**File**: `source/go2_rl_lab/go2_rl_lab/tasks/manager_based/go2_rl_lab/agents/rsl_rl_compliance_cfg.py`

```python
@configclass
class ComplianceRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Stage 2: compliance policy training with frozen low-level."""

    class_name: str = "ComplianceOnPolicyRunner"

    num_steps_per_env: int = 24
    max_iterations: int = 5000   # ~2M samples at 4096 envs
    save_interval: int = 100
    experiment_name: str = "go2_compliance"

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,           # Smaller — 3D action space
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[128, 64],  # Lightweight
        critic_hidden_dims=[128, 64],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

    # Estimator config (for reconstructing frozen estimator from stage-1 ckpt)
    estimator: dict = {
        "temporal_steps": 20,
        "enc_hidden_dims": [128, 64],
        "f_head_dims": [32, 16],
        "force_dim": 2,
        "dec_hidden_dims": [256, 128],
    }

    # Compliance-specific config
    compliance: dict = {
        "stage1_checkpoint": "",  # MUST be set by user at runtime
        "alpha": 10.0,           # Force threshold (N)
        "beta": 10.0,            # Virtual impedance
        "compliance_reward_weight": 0.5,  # Weight for r_comp
        "high_level_hidden_dims": [128, 64],
    }
```

---

## Step 6: Gym Registration

**File**: `source/go2_rl_lab/go2_rl_lab/tasks/manager_based/go2_rl_lab/__init__.py`

Add:
```python
gym.register(
    id="Go2-Compliance-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_force_only_env_cfg:UnitreeGo2ForceOnlyEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_compliance_cfg:ComplianceRunnerCfg",
    },
)
```

Uses the same env config as `Go2-Force-Only-v0` — the compliance wrapper handles the rest.

---

## Step 7: Play Script Update

**File**: `scripts/rsl_rl/play.py`

Add to the runner dispatch:
```python
elif agent_cfg.class_name == "ComplianceOnPolicyRunner":
    from go2_rl_lab.estimator.compliance_runner import ComplianceOnPolicyRunner
    runner = ComplianceOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
```

---

## Final File Summary

| File | Action | Lines (est.) |
|------|--------|-------------|
| `estimator/compliance_env_wrapper.py` | **CREATE** | ~200 |
| `estimator/compliance_runner.py` | **CREATE** | ~300 |
| `tasks/.../agents/rsl_rl_compliance_cfg.py` | **CREATE** | ~60 |
| `tasks/.../__init__.py` | **MODIFY** | +10 lines |
| `scripts/rsl_rl/play.py` | **MODIFY** | +3 lines |
| **Total** | 3 new, 2 modified | ~570 |

No new env config, no new PPO class, no new reward mdp function. The compliance reward lives inside the env wrapper.

---

## Existing Code to Reuse

| Component | File | Reuse |
|-----------|------|-------|
| `ForceEstimator` | `estimator/force_estimator.py` | Load frozen from checkpoint |
| `ObsHistoryBuffer` | `estimator/obs_history_buffer.py` | Same buffer class |
| `ActorCritic` | `rsl_rl.modules.ActorCritic` | Standard rsl_rl class for both frozen low-level and new high-level |
| `PPO` | `rsl_rl.algorithms.PPO` | Standard PPO for high-level training |
| `OnPolicyRunner` | `rsl_rl.runners.OnPolicyRunner` | Base class for ComplianceOnPolicyRunner (extend or replicate learn loop) |
| `UnitreeGo2ForceOnlyEnvCfg` | `tasks/.../go2_force_only_env_cfg.py` | Reuse directly as the underlying env |

---

## Training Workflow

```bash
# Stage 1: Train base policy + estimator (already done)
python train.py --task Go2-Force-Only-v0

# Stage 2: Train compliance policy (this plan)
python train.py --task Go2-Compliance-v0 \
    --compliance.stage1_checkpoint /path/to/model_XXXX.pt \
    --compliance.alpha 10.0 \
    --compliance.beta 10.0

# Play with compliance
python play.py --task Go2-Compliance-v0 \
    --checkpoint /path/to/compliance_model_XXXX.pt
```

Expected convergence: ~2M samples (~5000 iterations × 4096 envs × 24 steps / 100k). Should take <1 hour on a single GPU.

---

## Verification

1. **Sanity check**: Stage 2 with `alpha=9999` (compliance disabled) should reproduce stage-1 behavior — the high-level learns to output near-zero `a'`.

2. **Compliance behavior**: With reasonable `alpha=10, beta=10`:
   - Small forces (< α): Robot resists normally, `a' ≈ 0`
   - Large forces (> α): Robot yields, velocity increases in force direction
   - The `a'_z` (yaw) component should stay small (penalized in r_comp)

3. **Compare with direct modulation**: Same force scenario, compare:
   - Stage-1 + `k·F̂` modulation (current branch)
   - Stage-2 HAC-LOCO compliance policy (this plan)
   - Metrics: energy consumption, tracking error, smoothness, reaction time

4. **Tensorboard logging**: Track compliance reward, high-level action magnitudes, force estimates, and base locomotion metrics.

---

## Key Design Decisions

1. **No privileged critic for high-level** (start simple) — can upgrade later
2. **Compliance reward inside wrapper, not env config** — keeps env config reusable
3. **Standard PPO** (no custom PPO class) — compliance is handled via reward shaping
4. **Reuse existing env config** — no new `go2_compliance_env_cfg.py`
5. **Force always active in stage 2** — no activation gate needed
6. **Small high-level network** [128, 64] — only 3 output dims, trains fast
