# Deep Compliant Control — Presentation #3 (Midterm)
### 18 March 2026 | 4-minute presentation | ~8 slides

---

## Slide 1: Title

**Deep Compliant Control for Quadruped Locomotion**

Towards Quadrupeds as Barrows: Force-Aware Compliance on the Unitree Go2

Date: 18 March 2026

---

## Slide 2: Problem Statement

**Why compliance matters**

- Current RL locomotion controllers are trained to **reject disturbances** — they maximise velocity tracking and treat any external force as a perturbation to resist.
- This robustness-first design causes **high joint stiffness**: when an unexpected force persists, the robot fights it with high-frequency corrective torques that can exceed motor limits and risk hardware damage.
- Interaction with the robot is limited to **joystick velocity commands** — there is no mechanism for a human to physically guide the robot or for the robot to yield to sustained contact.
- For applications like **payload transport on uneven terrain**, the robot must simultaneously handle human guidance forces, payload dynamics (shifted centre of mass, inertia), and terrain disturbances — none of which current controllers distinguish from one another.

**In short:** Robust is not enough. A quadruped operating as a barrow must sense forces, distinguish their source, and decide when to resist and when to yield.

---

## Slide 3: Project Goal

**Making the case for quadrupeds as barrows**

The thesis addresses four progressive challenges:

1. **Compliant control** through proprioceptive force estimation and adjustable velocity commands — no force/torque sensor required.
2. **Uneven terrain and inclines** — studying the influence of gravity on force estimation and ensuring compliance generalises beyond flat ground.
3. **Payload effects** — accounting for shifted centre of mass, added inertia, and gravity-induced forces that the estimator must distinguish from human-applied forces.
4. **Terrain advantage** — demonstrating scenarios (inclines, gravel, stairs) where a payload-carrying quadruped offers more versatility than a wheeled barrow.

The compliance level is adjustable at deployment via alpha (force threshold) and beta (impedance gain) — no retraining required.

**Why learning-based:** modelling coupled dynamics of payload, human contact, and rough terrain analytically is intractable. A learned estimator handles arbitrary force profiles where model-based methods break down.

---

## Slide 4: Literature & Positioning

| Method | Approach | Limitations |
|---|---|---|
| Kang et al. (model-based) | IMU+vision force estimation + MPC | Small perturbations only (<50N), pre-defined gaits |
| Hartmann et al. (RL) | Multi-stage episodic reward shaping | Transient pushes only; fails under persistent force |
| Li et al. (RL, torque-based) | Direct torque output for compliance | Fixed compliance; cannot track velocity commands under force |
| **HAC-LOCO** (Zhou et al.) | Hierarchical RL: frozen low-level + high-level compliance planner | Force estimator trained jointly with low-level — high-level planner arguably redundant |
| SAC-Loco (Zhang et al.) | Compliant policy + safety critic + recovery controller | Success rate drops above 500N; single-policy limits |
| Beyond Robustness (Chang et al.) | Load characteristics estimation via teacher-student | Focused on payload dynamics, not human-applied force compliance |
| FACET (impedance tracking) | Force-adaptive control via impedance reference | Impedance-based; not hierarchical, no payload consideration |

**This thesis:** Builds on HAC-LOCO's hierarchical architecture with adjustable compliance (SAC-Loco formulation), and extends toward payload-aware estimation (Beyond Robustness path). Deployed on the real Unitree Go2 with sim-to-real transfer. The key distinction from concurrent work (e.g. Cao et al.) is the focus on human-applied force compliance as the primary goal, with payload as an extension — rather than the reverse.

---

## Slide 5: Methodology — Architecture

**Two-stage hierarchical framework (adapted from HAC-LOCO)**

**Stage 1 — Locomotion + Force Estimation (complete, deployed)**
- Low-level policy: 60-dim obs → [512, 256, 128] → 12 joint actions (ELU, PPO)
- Autoencoder estimator trained concurrently:
  - Encoder: 1140-dim (20-step history x 57 obs) → [128, 64] → z_t
  - Force head: z_t → [32, 16] → 3D force estimate (f_hat)
  - Decoder: l_t = concat(f_hat, z_t) = 67-dim → [256, 128] → reconstructed obs
- Critic receives privileged info: base linear velocity, foot contacts, ground-truth force

**Stage 2 — Compliance (complete, deployed on real robot)**
- Three-phase curriculum: (1) locomotion only → (2) forces activate, estimator co-trains → (3) compliance mapping activates
- Compliance rule: v* = v_cmd + k(f) * f_hat, where k is piecewise on alpha/beta
- Adjustable at deployment: alpha (force threshold), beta (impedance gain)

**Next: Payload extension**
- Add payload randomisation in simulation (mass, CoM offset, inertia)
- Study gravity component separation in force estimates on inclines
- Potential: add load estimation head to the existing autoencoder

*Include architecture diagram (HAC-LOCO framework figure)*

---

## Slide 6: Results So Far

**Stage 1 — Force Estimation**
- Median angular error: ~6-7 degrees on XY plane
- Magnitude MAE: ~2.7N across [0, 20]N force range per axis
- Fz estimation working but with +5N bias in MuJoCo sim2sim
- Deployed and running on real Go2 hardware

**Stage 2 — Compliance (working on real robot)**
- 8 parallel training runs completed; full 5000-iteration runs converged
- Deployed on real Go2: **working on grass, small gravel, and slight inclines**
- Robot resists small disturbances, yields to sustained human pulls
- Compliance coefficient tunable at deployment (tested k=0.06)

**Current challenge: gravity and payloads**
- On inclines, gravity projects onto the force estimate — the estimator confuses slope with applied force
- Adding a payload shifts the CoM and changes the gravity component further
- Open question: how to decompose estimated force into human-applied vs. gravity vs. payload-induced

**Deployment pipeline**
- Sim2sim: MuJoCo via unitree_mujoco bridge, JIT policy + JIT estimator
- Real robot: Unitree SDK2, Kalman filter for velocity, joystick FSM control

---

## Slide 7: Timeline & Next Steps

| Phase | Weeks | Status | Focus |
|---|---|---|---|
| Setup | 1-3 (Feb 11-Mar 3) | DONE | Training pipeline, baseline locomotion, external forces in sim |
| Estimator | 4-5 (Mar 4-Mar 16) | DONE | Stage 1: autoencoder with force heads, sim2sim + real deployment |
| Compliance | 5-6 (Mar 10-Mar 18) | DONE | Stage 2: compliance training, real robot validation on grass/gravel/inclines |
| Gravity + Payload | 6-9 (Mar 18-Apr 13) | NOW | Gravity influence on inclines, payload randomisation, force decomposition |
| Evaluation + Write | 10-13 (Apr 14-May 11) | UPCOMING | Quantitative evaluation, terrain comparison, thesis writing |

**Immediate next steps:**
- Characterise gravity influence on force estimation across incline angles
- Add payload randomisation in Isaac Sim (mass, CoM offset)
- Investigate force decomposition: separating human force from gravity/payload components
- Evaluate whether the existing estimator architecture can accommodate a load estimation head (Beyond Robustness approach)

---

## Slide 8: Summary

- **Problem:** RL controllers reject all forces — unsafe for human-robot interaction and insufficient for payload transport on uneven terrain
- **Approach:** Hierarchical framework with learned force estimation and adjustable compliance, no external sensors
- **Progress:** Stages 1 and 2 complete and deployed on real Go2 — compliance working on grass, gravel, and slight inclines
- **Current frontier:** Handling payloads and gravity effects on inclines — how to decompose estimated forces into human-applied vs. environment-induced
- **Thesis vision:** Make the case for quadrupeds as barrows — compliant, terrain-capable, payload-aware, guided by human contact
