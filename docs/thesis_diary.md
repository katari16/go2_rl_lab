# Thesis Diary — Design Decisions & Trade-offs (2026-03-02)

## 1. Compliance Architecture: High-Level Policy vs. Velocity Command Modulation

**Date**: 2026-03-02
**Context**: The force estimator is trained and produces body-frame XY force estimates from proprioceptive history. The next step is making the robot compliant — walking in the direction of applied forces instead of resisting them.

### Option A: Two-Level Hierarchy (HAC-LOCO)

A high-level policy outputs velocity command residuals `a' = [Δv_x, Δv_y, Δω_z]` that are added to the user command. The low-level policy (already trained) tracks the modified command. A dedicated compliance reward shapes `a'`:

- If `||f|| ≤ α`: `r_comp = -||a'||²` (minimize adjustments for small forces)
- If `||f|| > α`: `r_comp = -||a'_xy - f_xy/β · (1 + α/|f_xy|)||² - ||a'_z||²` (comply proportionally)

**Pros**:
- The high-level policy implicitly learns smooth outputs — acts as a learned filter with zero lag.
- Tunable impedance via α (force threshold) and β (stiffness) without retraining the low-level policy.
- Can learn complex compliance behaviors beyond linear proportionality.

**Cons**:
- Requires training a second policy on top of the locomotion policy.
- More complex architecture: two networks, two training stages, two inference calls.
- The low-level policy must already be robust to arbitrary velocity command changes.
- Harder to debug — compliance issues could originate from either level.

### Option B: Direct Velocity Command Modulation (SAC-Loco style)

A single policy tracks velocity commands. At inference/training time, the velocity command is directly modified using the force estimate:

`v*_x = v_x + k(f) · F̂_x`, where `k(f) = 1/β · (1 + α/|f|)` when `||f|| > α`, else `k = 0`.

The existing velocity tracking reward handles compliance — tracking the modulated command IS compliant behavior.

**Pros**:
- Single policy, no architectural changes, no new reward functions.
- The existing `track_lin_vel_xy_exp` reward does all the work.
- α and β are runtime-tunable without retraining.
- Simple to deploy: one policy + one estimator + a two-line modulation formula.
- Easy to test incrementally (static_eval.py already has this working).

**Cons**:
- Noisy force estimates → noisy velocity commands → jittery robot motion.
- No learned smoothing — the policy sees raw noise in the command and may produce jerky behavior.
- The compliance response is fixed to the modulation formula; cannot learn non-linear behaviors.

### The Noise Problem with Option B

The force estimator produces noisy outputs (MAE ~11° angle, ~2.7N magnitude from eval plots). With direct modulation, this noise propagates straight into the velocity command. Possible mitigations that don't add lag:

1. **Smoothness loss on the estimator**: Add `L_smooth = ||f̂_t - f̂_{t-1}||²` to the estimator training loss. The estimator already sees 20 steps of history — it can learn temporally consistent outputs. Fixes noise at the source, zero lag.
2. **Action rate penalty**: A strong `||a_t - a_{t-1}||²` penalty makes the policy act as an implicit low-pass filter. It naturally tracks the smooth trend of noisy commands. Already standard in locomotion RL.
3. **Velocity acceleration penalty**: Penalize `||v_t - 2·v_{t-1} + v_{t-2}||²` to directly discourage jerky motion.
4. **Slew rate limiter**: Cap the maximum change in the modulated command per step. Zero lag in steady state, only limits transient rates. Not a filter — converges to the target exactly.

### Decision

**Chosen: Option B (direct modulation) with smoothness loss on the estimator (#1) and action rate penalty (#2).**

Rationale:
- Option B is already implemented and working in static_eval.py and the EstimatorEnvWrapper.
- The compliance behavior from simple modulation + velocity tracking reward is functionally equivalent to what HAC-LOCO achieves, without the two-level complexity.
- The noise problem is best solved at the source (estimator smoothness loss) rather than with a separate policy or post-hoc filtering.
- If this approach proves insufficient after real-robot testing, upgrading to a two-level hierarchy remains possible — the force estimator and low-level policy are reusable.

### Implementation Status

- [x] Force estimator trained and evaluated (static_eval.py with arrow visualization + plots)
- [x] EMA filter for force estimate visualization (--ema_alpha)
- [x] Linear compliance modulation in EstimatorEnvWrapper (v* = v + k·F̂)
- [x] Compliance gating on angle_err_median < 7° with k-ramp in ForceOnPolicyRunner
- [x] Compliance args in play.py (--compliance_k, --ema_alpha) for inference
- [x] Compliance in static_eval.py (--compliance_k) for evaluation
- [ ] α/β-aware modulation formula (replace constant k with k(f) = 1/β·(1+α/|f|))
- [ ] Estimator smoothness loss
- [ ] Real robot deployment and sensor validation

---

## 2. Compliance Training Strategy: Freeze+Extend vs. Single-Policy Finetuning

**Date**: 2026-03-03
**Context**: After the base locomotion policy and force estimator are trained (Stage 1), how should compliance behavior be introduced?

### Option A: HAC-LOCO Freeze+Extend (True Hierarchy)

As described in HAC-LOCO (Section III.B.2): *"The high-level policy is trained in the second stage after the low-level policy is complete. During this phase, the low-level policy is frozen and treated as a part of the environment."*

Two separate neural networks:
- **Low-level** π(o_t, l_t) → joint positions (trained in stage 1, then **frozen**)
- **High-level** π'(o'_t) → residual velocity commands a' = [Δv_x, Δv_y, Δω_z] (trained in stage 2)

The high-level takes `[s_t, c_t, l_{t-1}, a'_{t-1}, a_{t-1}]` as input and is lightweight (~2M samples to converge vs ~200M for low-level). The compliance reward `r_comp` shapes the residual commands with α/β parameters.

**Pros**:
- No catastrophic forgetting — the low-level policy never changes.
- The compliance module can be retrained with different α/β without touching the base policy.
- Clean separation of concerns: locomotion vs. compliance.
- HAC-LOCO paper validated this on real hardware (Unitree Go1).

**Cons**:
- Two networks at inference time (though high-level is small).
- The low-level policy must handle arbitrary velocity command perturbations gracefully.
- More complex training pipeline: stage 1 → freeze → stage 2.
- Need to implement a new runner that wraps the frozen policy.

### Option B: Single-Policy Finetuning (Curriculum Extension)

Keep one policy. After stage 1 converges, introduce the compliance reward and continue training. The same policy learns to be compliant through modified rewards. Optionally feed force estimates into the critic to improve the value function.

**Pros**:
- Simplest architecture — one policy, one training run (with curriculum).
- No need for a separate compliance network or modified runner.
- The policy can jointly optimize locomotion + compliance end-to-end.

**Cons**:
- **Catastrophic forgetting risk**: Stage 1 teaches "forces are disturbances, resist them." Stage 2 says "comply with forces." The value function becomes wrong, policy gradients may destabilize.
- Mitigation exists (low LR, freeze actor briefly, small compliance reward weight) but success is not guaranteed.
- Harder to isolate compliance behavior — debugging whether issues come from locomotion or compliance.
- Cannot swap compliance parameters without retraining.

### Key Difference

HAC-LOCO explicitly avoids the forgetting problem by **never modifying the base policy**. The compliance module is a separate network that only modulates velocity commands — the low-level policy just tracks whatever command it receives. The single-policy approach risks losing the robust gait learned in stage 1.

### Decision

**Open — needs experimental evaluation.** Both approaches should be tried:
- Branch `feature/force-velocity-modulation`: Option B-lite (direct k·F̂ modulation, no new reward, no finetuning — simplest baseline)
- Branch `feature/multistage-force-finetuning`: Could implement either Option A (freeze+extend) or Option B (finetuning). The HAC-LOCO freeze+extend is the safer choice if a new reward is introduced.

### Stage 1 Readiness Criteria (for either approach)
- Force estimator activates when mean episode reward > 30.0
- Force estimator considered "learned" when `angle_err_median_deg < 6.75°`
- Only after both conditions are met should stage 2 (compliance training) begin

---

## 3. Sim2Sim Deployment: Foot Force Sensor Availability in MuJoCo

**Date**: 2026-03-03
**Context**: The force-only policy uses 61-dim observations including `applied_torque` (12 dims) and `foot_contact_force_norms` (4 dims). For sim2sim deployment via unitree_mujoco, we need these sensor readings.

### Current State

- **Applied torques**: Available via `motor_state[i].tau_est` in the unitree_mujoco bridge.
- **Foot contact forces**: **NOT populated** by the unitree_mujoco bridge. The `foot_force[4]` field in `LowState_` exists but the bridge code never writes to it. The mujoco_menagerie Go2 model also has no `<touch>` sensor definitions.

### Solutions Investigated

**Approach 1: Add MuJoCo touch sensors to the Go2 XML**

MuJoCo natively supports touch sensors via `<sensor><touch site="..."/></sensor>`. These measure the sum of all contact normal forces on a site's geom. The Go2 foot geoms already have `condim="6"` contact properties. Steps:
1. Add `<site>` elements to each foot body
2. Add `<touch>` sensors referencing those sites
3. Read `data.sensordata[sensor_id]` in the bridge code
4. Map to `foot_force[4]` in the `LowState_` message

This is the clean solution and widely used in MuJoCo quadruped research.

**Approach 2: Use `data.cfrc_ext[body_id]`**

MuJoCo computes external contact forces per body in `cfrc_ext`. These can be read directly without modifying the XML, but require calling `mj_rnePostConstraint()` explicitly (MuJoCo 2.0+ no longer calls it by default unless force sensors are present in the model).

**Approach 3: Feed zeros for foot forces**

The simplest — just pass zeros for the 4 foot force dims. The estimator has 57 other observation dims (including torques, which implicitly encode ground contact). Worth testing as a baseline before investing in sensor modifications.

### Decision

**Open — start with Approach 3 (zeros) to see if the estimator still works. If it degrades significantly, implement Approach 1 (touch sensors in XML).**

---

## 4. Compliance Training: Failed Approaches and Final Architecture

**Date**: 2026-03-03

### What was tried and failed

#### Attempt 1: Frozen estimator + HAC-LOCO r_comp reward (penalty form)

Loaded a pre-trained force estimator as a frozen observation term (2 extra dims in policy obs). Added the HAC-LOCO resistance-compliance reward:
- `|f| <= α`: penalty `||a'||²` (resist)
- `|f| > α`: penalty `||a'_xy - f_xy/β·(1+α/|f_xy|)||²` (comply)

With negative weight (-1.0), this was a pure stick (no carrot). After activation at reward > 30, the `r_comp` penalty plateaued at -0.14 for 2000+ iterations — the policy completely ignored it. The tracking reward dominated because:
- Tracking gives strong positive signal (exponential, weight 1.5)
- r_comp gives weak negative signal (quadratic penalty, weight -1.0)
- They fight each other: tracking says "stay on command", r_comp says "deviate toward force"

#### Attempt 2: Exponential r_comp + tracking weight reduction

Switched r_comp to exponential form `exp(-error/std)` with positive weight 1.5 (matching tracking weight). Reduced tracking weights by 50% on activation. The robot became unstable — couldn't maintain a stable gait. The two competing positive rewards (track command vs. comply with force) confused the policy.

#### Root cause: frozen estimator overfits to one gait

The force estimator was trained alongside the force-only policy (stage 1). It learned to read force from THAT policy's specific proprioceptive patterns. When a new policy (compliant training) walks differently, the estimator's accuracy degraded from 6° to 17-18° median angular error. The policy couldn't learn compliance from noisy estimates.

Even as training progressed, the policy implicitly adapted its gait to be more "readable" by the frozen estimator (angular error dropped to ~10°), but this co-adaptation was slow and the compliance reward never improved.

### Key insight: the estimator must co-train with the policy

A frozen estimator cannot generalize across gaits. The estimator and policy must adapt together. This is exactly what the force-only env does (ForceOnPolicyRunner trains the estimator per PPO mini-batch alongside the policy). The compliance training should do the same.

### Final architecture: Three-phase single-env training

**CompliantOnPolicyRunner** — extends OnPolicyRunner with estimator training + three phase gates.

**Phase 1** (reward < 30): Walk.
- Forces = 0. Estimator runs inference from checkpoint but doesn't train (no GT force to learn from).
- Policy obs: 63 dims (61 proprioceptive + 2 force estimate ≈ 0).
- Standard velocity tracking reward.

**Phase 2** (reward >= 30): Forces activate.
- Persistent XY force activates (0-20N, re-randomized every 3-5s).
- Estimator starts training on the new policy's gait (supervised: force MSE + angular loss + reconstruction).
- Estimator is warm-started from the stage-1 checkpoint → converges fast.
- Standard velocity tracking reward (no compliance yet).

**Phase 3** (median angular error < 7°): Linear mapping activates.
- Velocity command is modulated: `v* = v_cmd + k(f)·f_hat`
- Piecewise k: `k = 0 if |f_hat| < α else 1/β`
- α (force threshold) = 5N, β (impedance) = 50
- The standard tracking reward now rewards following `v*`, which IS compliant behavior.
- The policy fine-tunes its gait for stable walking under forces + compliance.

**Why this works:**
1. **No competing rewards** — there's only one tracking reward. When mapping activates, the target velocity shifts, and the policy naturally learns to comply.
2. **No frozen estimator** — the estimator adapts to the current policy's gait, maintaining accuracy.
3. **Warm start** — the estimator starts from a good checkpoint, so Phase 2 → Phase 3 transition is fast.
4. **Piecewise k** — small forces (< α) are rejected (k=0, track original command). Large forces trigger compliance (k=1/β). This gives the HAC-LOCO α/β impedance behavior without a separate reward.
5. **Linear mapping feeds back into the policy** — the policy is trained with the shifted command, so it learns a gait that is stable under forces AND naturally compliant. This is the "fine-tuning" that was missing in the frozen estimator approach.

### Comparison with literature

| Aspect | HAC-LOCO | SAC-Loco | This work |
|--------|----------|----------|-----------|
| Architecture | 2 policies (frozen low + high) | Teacher-student distillation | 1 policy + trainable estimator |
| Force estimation | Concurrent encoder + f_head | Teacher: privileged GT force, Student: implicit | Concurrent encoder + f_head (warm-started) |
| Compliance mechanism | High-level NN outputs Δv | Teacher uses `v* = v+k·F_gt`, student imitates | Linear mapping `v* = v+k(f)·f_hat` |
| Compliance reward | r_comp (piecewise penalty) | Standard tracking on v* | Standard tracking on v* |
| α/β tuning | Retrain high-level only | k is command input | k(f) = piecewise function of α, β |
| Estimator robustness | Trained with one gait | No explicit estimator | Co-trains with policy, warm-started |

### Implementation

- **Runner**: `go2_rl_lab/estimator/compliant_on_policy_runner.py`
- **Env config**: `go2_compliant_env_cfg.py` (force-only + ForceEstimateObsTerm passthrough + compliant tracking)
- **Agent config**: `agents/rsl_rl_compliant_cfg.py` (estimator arch + phase thresholds + α/β)
- **Reward**: `compliant_track_lin_vel_xy_exp` in `mdp/rewards.py`

### Open questions

- What are good values for α and β for the Go2? (Currently α=5N, β=50 → k=0.02 when complying)
- Should the tracking weight be reduced when mapping activates, or is the shifted command sufficient?
- Can the estimator be further fine-tuned on real robot using reconstruction loss only (no GT force)?
- Will this generalize to stair terrain? The estimator sees diverse gaits during co-training, which helps.

---
