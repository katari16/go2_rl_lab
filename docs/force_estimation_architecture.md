# Force Estimation Architecture — Design Notes

## Problem Statement

The current force estimator uses a shared encoder feeding three heads (v_head, f_head, decoder). Velocity estimation converges quickly (~0.008 loss) but force estimation stalls (~19.0 loss at 20N). Two papers inform a better approach:

- **Wrench estimation via proprioceptive data** (Gu et al., arXiv:2503.10401): model-based force estimation using joint angles, velocities, and torques through the rigid body dynamics equation.
- **Residual acceleration for force estimation** (Gu et al., IEEE 2025, 10893695): F_ext = M * a_actual - tau + eta(q, q_dot), and the impulse theorem F*dt = delta(mv) for velocity mapping.

Both papers converge on the same physics: **external force is the residual in the dynamics equation**, and the key inputs are joint torques (tau), positions, velocities, accelerations, and foot contact forces.

---

## Why the Current Architecture Struggles

### Velocity vs Force Estimation are Fundamentally Different Tasks

| Property | Velocity Estimation | Force Estimation |
|---|---|---|
| Nature | Kinematic (geometry) | Dynamic (F=ma) |
| Key observations | joint_vel, ang_vel, joint_pos | torques, foot forces, joint_acc |
| Temporal scale | Short (5-10 steps) | Medium (20-50 steps, force persistence) |
| Convergence | Fast (~100 iterations) | Slow (needs sufficient force magnitude) |
| Loss magnitude | ~0.008 | ~19.0 |

### Shared Encoder Gradient Conflict

With a shared encoder, velocity loss dominates gradients (converges first, smaller magnitude). The encoder features lock in for velocity estimation, and the force head is left trying to extract force information from an encoder optimized to ignore it.

### Missing Observations for Force Estimation

From the model-based equation: `F_ext = -(J(q)^T)^dagger * S^T * [tau - eta(q, q_dot)]`

The current observation stack is missing:
1. **Joint torques (tau)** — the most critical missing piece. Torques tell you what the motors are *trying* to do; force is what makes reality differ from that intent.
2. **Foot contact forces** — ground reaction forces change when external forces are applied (load redistribution across feet).

`last_action` is a rough proxy for torques but lossy — PD dynamics, joint friction, and contact state sit between the action and the actual torque.

---

## Proposed Routes

### Route A: Add Torques + Foot Forces to Shared Architecture (Quick Diagnostic)

Add `applied_torque` and `foot_contact_force_norms` to the current shared-encoder setup. If f_head loss drops significantly, it confirms the observation gap as the bottleneck.

**Pros:** Quick to test, minimal code changes.
**Cons:** Still has shared encoder gradient conflict.

### Route B: Two Independent Estimator Networks (Architecturally Clean)

Split into completely separate networks:

**VelocityEstimator** (keep as-is, proven to work):
- Obs: base_ang_vel, projected_gravity, joint_pos_rel, joint_vel_rel, last_action
- History: 10-20 steps
- Produces: vel_latent = concat(v_hat, z_vel)

**ForceEstimator** (new, specialized):
- Obs: applied_torque, joint_acc, joint_pos_rel, joint_vel_rel, foot_contact_force_norms, projected_gravity, base_ang_vel
- History: 20-50 steps (longer for force persistence patterns)
- Produces: force_latent = concat(f_hat, z_force)

Policy input: `concat(raw_obs, vel_latent, force_latent)`

**Pros:** No gradient interference, optimal obs per task, independent training schedules/LRs.
**Cons:** More parameters, two networks to maintain.

### Route C: Physics-Informed Hybrid

Compute a model-based force estimate using the dynamics equation (Isaac Lab provides all terms), then train a small correction network on the residual.

```
F_model = M * a_measured - tau_measured + gravity_terms
F_corrected = F_model + network(obs_history)
```

**Pros:** Bakes in physics prior, very sample-efficient, small network.
**Cons:** Requires accurate mass/inertia model, less flexible, harder sim-to-real.

### Route D: Impulse-Based Temporal Approach

Estimate force from momentum changes over a time window: `F = delta(mv) / dt`.
The network observes base velocity changes and foot contact patterns, working with integrated quantities instead of noisy instantaneous derivatives.

**Pros:** Sidesteps noisy acceleration derivatives, physically grounded.
**Cons:** Temporal averaging introduces latency, less responsive to rapid force changes.

---

## Observation Stack Design for Force Estimation

### Essential (directly in the dynamics equation)

| Observation | Dimension | Source (Isaac Lab) | Real Hardware |
|---|---|---|---|
| Joint torques | 12 | `asset.data.applied_torque` | Motor controller feedback |
| Joint accelerations | 12 | `asset.data.joint_acc` (scaled by 0.01) | Finite diff of encoders |
| Joint positions | 12 | `asset.data.joint_pos` | Joint encoders |
| Joint velocities | 12 | `asset.data.joint_vel` | Joint encoders |

### Highly Informative

| Observation | Dimension | Source (Isaac Lab) | Real Hardware |
|---|---|---|---|
| Foot contact force norms | 4 | `contact_sensor.data.net_forces_w` norm | `low_state.foot_force[0-3]` (scalar, noisy) |
| Projected gravity | 3 | IMU-derived | IMU |
| Base angular velocity | 3 | IMU | IMU |

### Foot Forces: Sim-to-Real Considerations

**In simulation (Isaac Lab):** `ContactSensorCfg` provides clean 3D `net_forces_w` vectors per body. The `foot_contact_force_norms()` function already exists in observations.py and computes scalar norms per foot — [num_envs, 4].

**On real Go2 hardware:** 4 piezoelectric foot force sensors provide scalar normal force values (`low_state.foot_force[0-3]`). These are:
- Scalar (normal force only, no XY shear)
- Noisy (used for contact detection with ~20N threshold, not precision measurement)
- Already divided by 100 in the deploy code

**Bridging the gap:**
- Use **scalar norms** in simulation (not 3D vectors) to match what hardware provides
- Add **uniform noise** (e.g., +/-10-20N equivalent) in sim to match real sensor characteristics
- The critical information (load distribution shift across feet when pushed) survives this noise level
- Consider training with **occasional foot sensor dropout** for robustness

### Joint Torques: Sim-to-Real Considerations

**In simulation:** `asset.data.applied_torque` gives clean per-joint torques.

**On real Go2:** Motor controllers provide torque feedback, but it's filtered and may have calibration offsets. The deploy code doesn't currently use torques.

**Bridging the gap:**
- Add noise to simulated torques to match real motor feedback quality
- `last_action` (joint position targets) can serve as a backup proxy since the PD controller maps actions to torques deterministically given joint state

---

## Key Insights from the Papers

### The Residual Acceleration Framework (Paper 2)

The external force is estimated by isolating the disturbance in the Euler-Lagrange equations:

```
F_ext = M * a_actual - tau_commanded + eta(q, q_dot)
```

In RL terms: the temporal encoder learns to distill this residual as a latent variable from proprioceptive history. Adding joint torques makes this distillation dramatically easier because the network can directly compute the residual instead of inferring tau from indirect effects.

### The Impulse Theorem for Velocity Mapping (Paper 2)

Once F_ext is identified, it maps to velocity changes: `v_target += (F_ext * dt) / m`. This connects force estimation to the compliance controller — the estimated force directly informs how the robot should adjust its velocity to yield or resist.

### HAC-LOCO's Approach (Reference Paper)

HAC-LOCO uses the SAME shared encoder for both velocity and force estimation, but with key differences from our setup:
- Observation includes a **clock signal** (sin/cos gait phase)
- History length H=10 (shorter than our 20)
- Max force magnitude 50N during training
- Force curriculum built into the training (magnitude determined by curriculum)
- Losses are part of the total PPO loss (not a separate optimizer)
- PD gains are Kp=30, Kd=0.75 (we should verify ours)

### Critical Questions

1. **The Derivative Problem:** Joint acceleration from finite differencing amplifies noise. The temporal encoder acts as an implicit low-pass filter, but history length sets the cutoff. Separate networks allow the force estimator to use a longer history (30-50 steps) for better filtering without penalizing velocity estimation responsiveness.

2. **Credit Assignment:** With separate networks, velocity estimation has no force-related interference. Force estimation gets a clean supervised loss (MSE to ground truth). The policy reward remains about locomotion — the force latent is additional context for gait adaptation.

3. **Sim-to-Real for Foot Forces:** Real Go2 foot sensors are scalar and noisy. Training in sim with scalar norms + noise injection should bridge the gap. The load redistribution signal (which feet bear more weight when pushed) is robust to noise.

4. **The Command Conflict:** When zero velocity is commanded but a strong force is applied, the compliance module (HAC-LOCO's Stage 2) handles this. The force estimator doesn't need to resolve this — it just provides the signal. The policy or a higher-level module decides compliance vs resistance.

---

## Recommended Implementation Order

1. **Quick test (Route A):** Add `applied_torque` and `foot_contact_force_norms` to the current shared architecture. Run training to see if f_head loss improves. This validates that observations are the bottleneck.

2. **Clean split (Route B):** If Route A confirms the observation gap, split into two independent networks with specialized obs stacks. This is the long-term architecture.

3. **Compliance module:** Once force estimation works, implement the HAC-LOCO Stage 2 high-level compliance policy that adjusts velocity commands based on estimated forces.
