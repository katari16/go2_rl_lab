# State Estimation Network — Design Document

> **Branch:** `feature/state-estimation-network`
> **Status:** Implemented (velocity estimation). Force estimation planned for future work.

---

## 1. Motivation

A real Unitree Go2 has no direct access to:
- **Base linear velocity** — the IMU gives angular rates, not body-frame translation velocity.
- **External forces** — disturbances, payload, terrain reaction.

During sim training these quantities are available as ground truth from the physics engine.
The goal of this module is to train a neural network that learns to estimate these hidden
states from the robot's readily observable proprioception (joint angles, joint velocities,
IMU measurements, last actions).

The estimation network is trained **concurrently** with the locomotion policy during Stage 1
PPO training. At inference time, the estimator's latent representation is appended to the
raw proprioceptive obs and fed to the policy, so the policy implicitly uses the velocity
estimate without needing separate post-processing.

---

## 2. Reference Architecture — HAC-LOCO / HIM-LOCO

This implementation adapts the **Stage 1** architecture from:

- **HAC-LOCO** — Zhou et al. (2025) [arXiv:2507.02447]
  Two-stage hierarchical framework: (1) concurrent estimator + PPO, (2) distillation into
  a teacher policy that conditions on estimated states.
- **HIM Loco** — Long et al. (2024) [arXiv:2312.11460]
  Similar idea with a contrastive history-matching variant.
- Reference code: [`references/HIMLoco/`](../references/HIMLoco/) (cloned from
  https://github.com/InternRobotics/HIMLoco)

---

## 3. Network Architecture

```
Observation History (last H steps of proprioception)
  o_t^H  ∈  ℝ^{H × D_obs}         H=10, D_obs=45  ⟹  450-dim flattened input

       │
       ▼
  ┌─────────────────────────────┐
  │  Encoder MLP                │   [450 → 128 → 64]   activation: ELU
  │  (obs history → z_t)        │
  └─────────────────────────────┘
       │
       ▼
  z_t  ∈  ℝ^{64}                    compact latent / world model

       │
       ├──────────────────────────► v_head MLP  [64 → 32 → 16 → 3]
       │                                │
       │                                ▼
       │                          v̂_t  ∈  ℝ^{3}   estimated base linear velocity (vx, vy, vz)
       │
       ▼
  l_t = concat(v̂_t, z_t)  ∈  ℝ^{67}     "rich latent" handed to policy

       │
       ▼
  ┌─────────────────────────────┐
  │  Decoder MLP                │   [67 → 512 → 256 → 128 → 45]   activation: ELU
  │  (latent → ô_{t+1})         │   reconstruction regulariser
  └─────────────────────────────┘
       │
       ▼
  ô_{t+1}  ∈  ℝ^{D_obs}               predicted next proprioceptive observation
```

### Policy input (augmented)

```
policy_input = concat(o_t, l_t)  ∈  ℝ^{45 + 67} = ℝ^{112}
```

The actor MLP sees 112-dimensional input automatically because `EstimatorEnvWrapper`
augments the `"policy"` obs group before rsl_rl's `_construct_algorithm` builds the actor.

### Critic input (unchanged)

The critic still receives the original 48-dim privileged obs
(`base_lin_vel`, `base_ang_vel`, `projected_gravity`, `velocity_commands`, `joint_pos`,
`joint_vel`, `last_actions`) without any augmentation — it has direct access to ground truth.

---

## 4. Training Losses

Two supervised losses are applied every PPO iteration, after the PPO update:

| Loss       | Formula                                        | Purpose                                         |
|------------|------------------------------------------------|-------------------------------------------------|
| `L_vel`    | MSE(v̂_t,  v_gt)                              | Directly supervise velocity estimation          |
| `L_rec`    | MSE(ô_{t+1}, o_{t+1})                         | Reconstruction regularises the encoder          |
| `L_total`  | ω₁ · L_vel  +  ω₃ · L_rec                     | Weighted sum (defaults: ω₁=1.0, ω₃=1.0)        |

**Ground-truth velocity source:** `v_gt` is extracted from the critic obs stored in the PPO
rollout buffer — `critic_obs[:, :, 0:3]` — because `base_lin_vel` is the first obs term in
the critic group (see `go2_test_bench_env_cfg.py` CriticCfg).

---

## 5. File Structure

```
source/go2_rl_lab/go2_rl_lab/estimator/
├── __init__.py                    Exports: VelocityEstimator, ObsHistoryBuffer,
│                                  EstimatorEnvWrapper, EstimatorOnPolicyRunner
├── velocity_estimator.py          Neural network (encoder, v_head, decoder) + optimizer
├── obs_history_buffer.py          Rolling window [num_envs, H, D_obs]; resets on done
├── estimator_env_wrapper.py       rsl_rl VecEnv wrapper — augments "policy" obs with l_t
└── estimator_runner.py            EstimatorOnPolicyRunner(OnPolicyRunner)

tasks/manager_based/go2_rl_lab/agents/
└── rsl_rl_estimator_cfg.py        EstimatorRunnerCfg — all hyperparameters in one place

tasks/manager_based/go2_rl_lab/
└── __init__.py                    Gym registration for Go2-Testbench-Estimator-v0

scripts/rsl_rl/
└── train.py                       3-line elif branch for EstimatorOnPolicyRunner

docs/
├── state_estimation_network.md    This document
├── hac_loco.md                    HAC-LOCO paper (converted from PDF)
└── todos.md                       Previous design notes

references/
└── HIMLoco/                       Cloned reference implementation (git-ignored)
```

---

## 6. Component Details

### 6.1 `ObsHistoryBuffer`

```python
buffer: Tensor[num_envs, temporal_steps, obs_dim]
```

- `insert(obs)` — rolls buffer left (drops oldest) and appends new obs at position `[-1]`.
- `reset(env_ids)` — zeroes the history for terminated episodes (avoids cross-episode leakage).
- `get_flattened()` — returns `[num_envs, H * D_obs]` for the encoder.

### 6.2 `VelocityEstimator`

Three MLP sub-networks sharing one Adam optimizer:

| Sub-net     | Input         | Output       |
|-------------|---------------|--------------|
| `encoder`   | H × D_obs (450) | z_t (64-dim) |
| `v_head`    | z_t (64)      | v̂_t (3-dim) |
| `decoder`   | l_t (67)      | ô_{t+1} (45) |

`get_latent(obs_history)` — inference only, `@torch.no_grad()`, returns `(vel_hat, latent)`.
`update(obs_history, gt_velocity, next_obs)` — one gradient step, returns `(vel_loss, rec_loss, total_loss)`.

### 6.3 `EstimatorEnvWrapper`

Wraps an rsl_rl `RslRlVecEnvWrapper` without subclassing it:

```python
get_observations() → augments "policy" obs with l_t from estimator
step(actions)      → env.step, resets history for dones, stores raw next obs
__getattr__        → delegates all other calls to the underlying env
```

The wrapper is created **before** `super().__init__()` inside `EstimatorOnPolicyRunner`, so
when rsl_rl's parent class calls `env.get_observations()` to determine the actor input dim,
it already sees 112-dimensional policy obs.

### 6.4 `EstimatorOnPolicyRunner`

Extends `OnPolicyRunner`. Key overrides:

| Method              | What it does                                                    |
|---------------------|-----------------------------------------------------------------|
| `__init__`          | Creates estimator, history buffer, wrapper; calls `super().__init__(wrapped_env)` |
| `learn()`           | Full training loop — rollout → PPO update → estimator update   |
| `_update_estimator` | Runs mini-batch supervised updates on the just-collected rollout |
| `_log_with_estimator` | Calls `self.log()` then appends estimator stats to terminal  |
| `save()` / `load()` | Policy + estimator weights in the same `.pt` checkpoint         |
| `train_mode()` / `eval_mode()` | Propagates to estimator as well                  |

**Rollout storage (estimator-specific):**

```python
_est_obs_history:  Tensor[num_steps, num_envs, H * D_obs]   # obs history at each step
_est_next_raw_obs: Tensor[num_steps, num_envs, D_obs]        # next raw obs (decoder target)
```

These plain tensors are populated during the rollout loop and consumed by `_update_estimator`.

---

## 7. Running Estimator Training

```bash
python scripts/rsl_rl/train.py \
    --task Go2-Testbench-Estimator-v0 \
    --headless
```

That's it. The `Go2-Testbench-Estimator-v0` gym registration points to:
- **Env:** `UnitreeGo2EnvCfg` (same as Go2-Testbench-v0 — rewards/obs/events unchanged)
- **Agent cfg:** `EstimatorRunnerCfg` with `class_name = "EstimatorOnPolicyRunner"`

`train.py` detects the class name and instantiates `EstimatorOnPolicyRunner` via a three-line
`elif` branch — no other modifications to the training script.

---

## 8. Terminal Output

During training the terminal shows the standard RSL-RL PPO stats block (mean reward,
mean episode length, FPS, PPO losses), followed immediately by an estimator block:

```
────────────────────────────────────────────────────────────────────────────────
                        [Estimator] vel_loss=0.04123  rec_loss=0.01847  |  smooth_vel=0.03891
```

**What to watch:**
- `vel_loss` should decrease from ~1.0–2.0 (random init) towards <0.05 over training.
  At <0.01 the velocity estimate is near-perfect.
- `rec_loss` should also decrease; it regularises the encoder. If it is much larger than
  `vel_loss` consider reducing `rec_loss_weight`.
- `smooth_vel` is a 20-iteration moving average — use this to judge convergence trend.

---

## 9. TensorBoard Logs

Estimator metrics are written under the `Estimator/` prefix:

| Tag                          | Meaning                                     |
|------------------------------|---------------------------------------------|
| `Estimator/vel_loss`         | Velocity MSE (per iteration)                |
| `Estimator/rec_loss`         | Reconstruction MSE (per iteration)          |
| `Estimator/vel_loss_smooth`  | 20-iteration moving average of vel_loss     |

Launch TensorBoard:

```bash
tensorboard --logdir logs/rsl_rl/go2_estimator
```

---

## 10. Checkpoint Format

The `.pt` checkpoint produced by `EstimatorOnPolicyRunner.save()` contains everything in
the standard rsl_rl checkpoint plus two extra keys:

```python
{
    # --- standard rsl_rl keys ---
    "model_state_dict":     ...,   # actor-critic weights
    "optimizer_state_dict": ...,
    "iter":                 ...,

    # --- estimator additions ---
    "estimator_state_dict":           ...,   # encoder + v_head + decoder weights
    "estimator_optimizer_state_dict": ...,
}
```

Loading resumes both policy and estimator training from the same file transparently.

---

## 11. Key Hyperparameters (`rsl_rl_estimator_cfg.py`)

| Parameter                  | Default    | Effect                                                  |
|----------------------------|------------|---------------------------------------------------------|
| `temporal_steps`           | 10         | History window (~0.2s at 50Hz). Sufficient for velocity estimation. |
| `enc_hidden_dims`          | [128,64]   | Encoder capacity. `enc_hidden_dims[-1]` = latent dim.  |
| `v_head_dims`              | [32,16]    | Small head sufficient; velocity is a simple projection. |
| `dec_hidden_dims`          | [512,256,128] | Decoder capacity for reconstruction.                 |
| `learning_rate`            | 1e-3       | Estimator Adam LR (separate from PPO LR).               |
| `vel_loss_weight`          | 1.0        | ω₁ — increase if vel_loss dominates.                   |
| `rec_loss_weight`          | 1.0        | ω₃ — reduce if reconstruction noise hurts encoder.     |
| `max_grad_norm`            | 10.0       | Gradient clipping for estimator.                        |
| `gt_vel_obs_start_idx`     | 0          | Slice start in critic obs for gt_velocity (base_lin_vel = [0:3]). |
| `num_estimator_mini_batches` | 4        | Mini-batches per estimator update pass.                 |

---

## 12. Obs Dim Reference

| Group    | Dims | Contents                                                                      |
|----------|------|-------------------------------------------------------------------------------|
| Policy   | 45   | base_ang_vel(3), projected_gravity(3), velocity_commands(3), joint_pos(12), joint_vel(12), last_action(12) |
| Critic   | 48   | base_lin_vel(3), base_ang_vel(3), projected_gravity(3), velocity_commands(3), joint_pos(12), joint_vel(12), last_action(12) |
| Latent   | 67   | v̂_t(3) + z_t(64) from estimator                                              |
| Policy+Latent | 112 | concat(policy_obs, latent) — what the actor sees                         |

The policy group deliberately **excludes** `base_lin_vel` because that is the quantity being
estimated. The critic retains `base_lin_vel` at index [0:3] so it can supervise the estimator.

---

## 13. Future Extensions

### 13.1 External Force Estimation

Adding a force estimation head follows the same pattern as `v_head`:

```python
# In VelocityEstimator.__init__:
self.f_head = _build_mlp([enc_latent_dim] + f_head_dims + [force_dim], act_fn)

# In get_latent():
force_hat = self.f_head(z_t)
latent = torch.cat([vel_hat, force_hat, z_t], dim=-1)
```

Ground-truth force can be extracted from contact sensor data and stored in the critic obs,
or logged separately as a privileged obs group.

### 13.2 Stage 2 — Teacher Policy Distillation

In HAC-LOCO Stage 2, a *teacher policy* is trained with access to the full latent output of
a **frozen** estimator. The teacher then distills into a student that runs on real hardware.
This would use rsl_rl's existing `DistillationRunner` as a starting point.

### 13.3 Contrastive / HIM-LOCO Variant

Replace the decoder reconstruction loss with a contrastive objective that pulls together
the latent representations of similar robot states across environments. This can be more
robust when the reconstruction loss is hard to balance.

---

## 14. Design Decisions

**Why wrap the env rather than modify rsl_rl internals?**
The `EstimatorEnvWrapper` approach requires zero changes to rsl_rl. The parent
`OnPolicyRunner.__init__` calls `env.get_observations()` to determine network input dims;
by wrapping before calling `super().__init__()`, the actor is built with the correct 112-dim
input size automatically. This keeps the codebase compatible with future rsl_rl upgrades.

**Why store estimator rollout data as plain tensors?**
rsl_rl's `RolloutStorage` is designed for PPO-specific data (obs, actions, rewards,
advantages). Adding estimator-specific tensors (`obs_history`, `next_raw_obs`) to the same
storage would couple the estimator to rsl_rl internals. Plain tensors in the runner are
simpler and easier to extend.

**Why extract gt_velocity from critic obs rather than a separate obs group?**
The critic already receives `base_lin_vel` in its first three dims. Reusing the existing
storage avoids adding another obs group to the env config and keeps the env configs clean
(they were explicitly not to be modified for this feature).

**Why train the estimator after the PPO update (not before)?**
The PPO update uses critic obs that include `base_lin_vel` — this is the ground truth for
the estimator. Training the estimator on the same rollout data, after the PPO update has
consumed it, reuses stored data without requiring an extra forward pass or separate replay
buffer.
