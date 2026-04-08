# Report Context: Proprioceptive Force and Wrench Estimation for Compliant Quadrupedal Locomotion

> This document provides the full technical context for generating the bachelor thesis report in LaTeX.
> It follows the ETH RSL thesis format (see reference theses by Leuthard 2023 and Fritsche/Feingold 2023).
> The author will provide a pipeline figure separately.

---

## Metadata

- **Type:** Bachelor Thesis
- **Title:** Proprioceptive Force and Wrench Estimation for Compliant Quadrupedal Locomotion
- **Term:** Spring Term 2026
- **Author:** Hans Baumann-Ortiz
- **Supervisors:** William Hartmann, Filip (surname TBD)
- **Supervising Lecturer:** Prof. Dr. Marco Hutter
- **Institution:** Robotic Systems Lab (RSL), ETH Zurich
- **Robot platform:** Unitree Go2 (12 DoF quadruped)
- **Simulation:** NVIDIA Isaac Lab (Isaac Sim), GPU-parallelized (4096 envs)
- **Algorithm:** PPO (Proximal Policy Optimization) via rsl_rl
- **Deployment:** Sim-to-sim (MuJoCo), sim-to-real (Unitree Go2 hardware)

---

## Report Structure

Follow the ETH RSL bachelor thesis template. Expected length: 35--50 pages main body + appendices.

```
Title page (ETH + RSL logos, Prof. Dr. Marco Hutter)
Declaration of Originality
Intellectual Property Agreement
Abstract (1 page)
Notation (symbol table)

1  Introduction
   1.1  Motivation
   1.2  Problem Definition
        1.2.1  Goals
        1.2.2  Requirements
   1.3  Related Work
        1.3.1  RL-Based Locomotion for Quadrupeds
        1.3.2  External Force Estimation
        1.3.3  Compliant Locomotion Control

2  System Overview
   2.1  The Unitree Go2 Robot
        2.1.1  Mechanical Design and Dimensions
        2.1.2  Actuators and Sensors
   2.2  Training Pipeline Overview
        (reference to pipeline figure provided by the author)

3  Training Setup
   3.1  Simulation Environment
        3.1.1  Isaac Lab Configuration
        3.1.2  Domain Randomization
   3.2  Observation and Action Spaces
        3.2.1  Policy Observations (57 + force estimate)
        3.2.2  Critic Observations (privileged)
        3.2.3  Action Space (12 joint positions)
   3.3  Reward Design
   3.4  Curriculum
        3.4.1  Three-Phase Training (Locomotion, Estimation, Mapping)
        3.4.2  Force Magnitude Curriculum

4  Force and Wrench Estimation
   4.1  Estimator Architecture
        4.1.1  History Buffer
        4.1.2  Temporal Encoder
        4.1.3  Force Head
        4.1.4  Reconstruction Decoder (Forward Dynamics)
   4.2  Training Losses
        4.2.1  Force MSE Loss
        4.2.2  Angular Loss (Force Direction)
        4.2.3  Reconstruction Loss
        4.2.4  Torque Angle Loss (for 6D wrench)
        4.2.5  Yaw Loss (for 6D wrench)
   4.3  Compliance Injection
        4.3.1  Linear Mapping (Force Estimate to Velocity Commands)
        4.3.2  EMA Smoothing

5  Ablation Studies
   5.1  Experimental Design
        5.1.1  Estimation Dimension Sweep (2D, 3D, 2D wrench, 6D wrench)
        5.1.2  History Length
        5.1.3  Network Capacity
        5.1.4  Reconstruction Loss (with vs. without)
        5.1.5  Compliance Rewards
   5.2  Evaluation Methodology
        5.2.1  Static 360-Degree Force Sweep
        5.2.2  Dynamic Compliance Evaluation
        5.2.3  Estimator Quality Metrics (MSE, RMSE, Angular Error)
        5.2.4  Effective Compliance Metric
   5.3  Results
        (tables and plots from batch 1 and batch 2 ablations)

6  Sim-to-Real Transfer
   6.1  Sim-to-Sim Validation (MuJoCo)
   6.2  Real-World Deployment
        6.2.1  Deployment Pipeline (JIT export)
        6.2.2  Compliance Modes (Normal, Off, Inverted)
        6.2.3  Real-World Observations

7  Conclusion

8  Outlook
   8.1  Higher Force Training (100 N)
   8.2  Estimator-Aware Locomotion (Force MSE in Policy Reward)
   8.3  Slope and Payload Evaluation
   8.4  Hierarchical Compliance Control (HAC-LOCO)

Bibliography
List of Figures
List of Tables
Appendix A: Training Hyperparameters
Appendix B: Full Ablation Results
Appendix C: Observation and Reward Tables
```

---

## Chapter Content Guide

### Abstract (~200 words)

Quadrupedal robots operating alongside humans or in unstructured environments must be able to detect and respond to external contact forces. We present a proprioceptive force and wrench estimation framework for the Unitree Go2 quadruped, trained entirely in simulation using deep reinforcement learning. The estimator processes a temporal history of proprioceptive observations---angular velocities, projected gravity, joint positions, joint velocities, and joint torques---through a learned encoder to predict external forces and torques applied to the robot base, without requiring any dedicated force sensor.

We conduct a systematic ablation study over the estimator design space, varying the number of estimated dimensions (2D force, 3D force, 2D wrench, 6D wrench), temporal history length, network capacity, and auxiliary training losses. We evaluate estimation accuracy using force magnitude MSE, angular error, and a novel effective compliance metric that measures the robot's velocity response per unit applied force.

To achieve compliant behavior, the estimated force is mapped to velocity command modulations via a linear gain, allowing the robot to yield in the direction of applied forces without retraining the locomotion policy. We validate the approach in simulation with a 360-degree force sweep evaluation and demonstrate sim-to-real transfer on the physical Go2 robot, confirming that the proprioceptive estimator generalizes to real-world contact.

---

### 1. Introduction

#### 1.1 Motivation

Legged robots are increasingly deployed in environments shared with humans---inspection, logistics, domestic assistance. In such settings, the robot inevitably experiences physical contact: a human pushing it aside, bumping into obstacles, or carrying payloads. Conventional RL-trained locomotion policies treat such forces as disturbances to reject. Compliant locomotion instead requires the robot to detect the applied force and yield appropriately, matching its velocity to the external perturbation.

Force/torque sensors mounted at the robot base are expensive, add weight, and introduce another failure point. An alternative is proprioceptive force estimation: inferring the applied wrench from quantities the robot already measures---joint positions, velocities, and torques. If such an estimator can be trained in simulation and transferred to the real robot, it provides a zero-cost, sensor-free pathway to contact-aware locomotion.

#### 1.2 Problem Definition

**Goals:**
1. Train a proprioceptive force/wrench estimator that predicts external forces (and optionally torques) on the robot base from joint-level measurements alone.
2. Systematically study the estimator design space: estimated dimensions, temporal history, network architecture, and auxiliary losses.
3. Achieve compliant locomotion behavior by mapping the estimated force to velocity command modulations, without retraining the base locomotion policy.
4. Validate the full pipeline in sim-to-real transfer on the Unitree Go2.

**Requirements:**
- The estimator must run at 50 Hz (control frequency) on the robot's onboard compute.
- The locomotion policy must remain stable under forces up to 100 N.
- The system must generalize to forces from arbitrary directions (360 degrees).
- No additional sensors beyond the robot's standard IMU and joint encoders.

#### 1.3 Related Work

**RL-Based Locomotion:** PPO-based locomotion for quadrupeds has been demonstrated extensively (Rudin et al., Hwangbo et al., Lee et al.). The standard teacher-student paradigm trains a privileged teacher with access to ground-truth state, then distills into a student that relies on proprioception alone.

**External Force Estimation:** Hartmann et al. (HAC-LOCO, 2024) propose a hierarchical architecture where a frozen low-level locomotion policy is augmented with a high-level compliance controller that modulates velocity commands based on estimated external forces. The force estimator uses a temporal history of proprioceptive observations. PAINT (Ji et al., 2024) uses a teacher-student framework where the teacher has access to ground-truth forces via a reward signal, and the student learns to estimate forces through an intent estimator trained with supervised regression.

**Compliant Locomotion:** Traditional impedance/admittance control achieves compliance through explicit force feedback. RL-based approaches encode compliance either through reward shaping (rewarding velocity alignment with applied force) or through hierarchical architectures that separate locomotion from compliance.

---

### 2. System Overview

#### 2.1 The Unitree Go2 Robot

The Unitree Go2 is a quadrupedal robot with 12 actuated degrees of freedom (3 per leg: hip abduction/adduction, hip flexion/extension, knee flexion/extension). It weighs approximately 15 kg and measures roughly 0.7 m in length. Each joint is driven by a proprietary actuator providing position, velocity, and torque feedback. The onboard IMU provides angular velocity and orientation estimates.

#### 2.2 Training Pipeline

> [AUTHOR: Insert pipeline figure here showing the full training and deployment flow]

The pipeline consists of three stages:

**Stage 1 -- Locomotion:** A standard PPO policy is trained to track velocity commands (vx, vy, omega_z) using a reward function combining velocity tracking, regularization (action rate, torque penalties), and termination conditions (base contact, joint limits). The policy observes 57 proprioceptive dimensions and outputs 12 target joint positions. External forces are applied during training as domain randomization, but the policy does not explicitly estimate them.

**Stage 2 -- Force Estimation:** A force estimator module is co-trained alongside the locomotion policy. The estimator takes a temporal history buffer of the 57-dimensional proprioceptive observation and predicts the external force (or wrench) applied to the robot base. The estimator is activated only after the locomotion policy reaches a reward threshold, ensuring stable gaits before attempting force estimation. Training losses include force MSE, angular direction loss, and optionally a reconstruction (forward dynamics) loss.

**Stage 3 -- Compliance Mapping:** Once the estimator achieves sufficient angular accuracy (median error below a threshold), a linear mapping from estimated force to velocity command modulation is activated. This injects compliance behavior without retraining the locomotion policy: the robot's velocity commands are augmented by k * EMA(F_hat), where k is a compliance gain and EMA denotes exponential moving average smoothing.

---

### 3. Training Setup

#### 3.1 Simulation Environment

Training is performed in NVIDIA Isaac Lab using the Isaac Sim physics engine. We simulate 4096 parallel environments on a single GPU. Each episode lasts 20 seconds. The simulation runs at dt = 0.005 s with a policy decimation of 4, yielding a control frequency of 50 Hz (control_dt = 0.02 s).

**Domain randomization** includes friction coefficients, added mass, motor strength scaling, and---critically---persistent external forces applied to the robot base at random magnitudes and directions, re-sampled every 3--5 seconds.

#### 3.2 Observation and Action Spaces

**Policy observations (60 dimensions):**

| Quantity | Dimensions | Notes |
|----------|-----------|-------|
| Angular velocity (body frame) | 3 | From IMU |
| Projected gravity | 3 | Orientation proxy |
| Velocity commands (vx, vy, wz) | 3 | From command generator |
| Joint positions (relative) | 12 | |
| Joint velocities | 12 | |
| Last actions | 12 | |
| Joint torques (scaled by 0.1) | 12 | |
| Force estimate | 2--6 | From estimator (dim varies by config) |

**Critic observations (privileged, 67--73 dimensions):**

All policy observations plus: base linear velocity (3), foot contact forces (4), ground-truth applied force/wrench (3--6).

**Actions:** 12 target joint positions, processed through a PD controller (Kp=25, Kd=0.5) to produce joint torques.

#### 3.3 Reward Design

The reward function follows the standard RSL locomotion reward structure:

| Reward Term | Weight | Description |
|------------|--------|-------------|
| Linear velocity tracking (XY) | 1.0 | exp(-error^2 / sigma) |
| Angular velocity tracking (yaw) | 0.5 | exp(-error^2 / sigma) |
| Linear velocity penalty (Z) | -2.0 | Penalize vertical bouncing |
| Angular velocity penalty (XY) | -0.05 | Penalize roll/pitch rates |
| Joint torque penalty | -0.0002 | Energy efficiency |
| Joint acceleration penalty | -2.5e-7 | Smooth motion |
| Action rate penalty | -0.01 | Smooth commands |
| Feet air time bonus | 0.125 | Encourage trotting gait |
| Undesired contacts | -1.0 | Penalize body/thigh ground contact |
| Flat orientation bonus | -0.5 | Keep base level |

#### 3.4 Curriculum

**Three-phase training with automatic gating:**

1. **Phase 1 -- Locomotion only:** Force range starts at (0, 0) N. Policy learns to walk. Estimator exists but is not yet trusted.
2. **Phase 2 -- Force estimation:** When mean episode reward exceeds a threshold (30), the force curriculum activates, linearly increasing the maximum applied force up to max_force (20 N in batch 1, 100 N in batch 2). The estimator is trained with supervised regression against ground-truth forces.
3. **Phase 3 -- Compliance mapping:** When the estimator's median angular error drops below a threshold (6 degrees), the linear mapping from force estimate to velocity commands is activated.

---

### 4. Force and Wrench Estimation

#### 4.1 Architecture

The estimator is a feedforward neural network operating on a sliding window of proprioceptive history:

**Input:** History buffer of shape [temporal_steps x 57], containing the last H proprioceptive observations (excluding the force estimate itself).

**Encoder:** MLP mapping the flattened history to a latent representation z_t.
- Default: [128, 64] hidden dims
- Bigger variant: [256, 128] hidden dims

**Force head:** MLP mapping z_t to the force/wrench estimate F_hat.
- Default: [32, 16] hidden dims
- Bigger variant: [64, 32] hidden dims
- Output dimension: 2 (Fx, Fy), 3 (Fx, Fy, Fz), 3 (Fx, Fy, tau_yaw), or 6 (Fx, Fy, Fz, tau_roll, tau_pitch, tau_yaw)

**Reconstruction decoder (optional):** MLP mapping z_t back to the next proprioceptive observation (forward dynamics prediction). This auxiliary task encourages the encoder to learn a representation that captures the full state dynamics, not just force-related features.
- Default: [256, 128] hidden dims

#### 4.2 Training Losses

All losses are computed per-step and averaged over the minibatch:

1. **Force MSE loss:** L_force = ||F_hat - F_gt||^2, weighted by force_loss_weight (1.0)

2. **Angular loss:** Penalizes directional error in the XY force plane:
   L_angle = (atan2(sin(theta_gt - theta_hat), cos(theta_gt - theta_hat)))^2
   Only applied when ||F_gt_xy|| > angle_min_force (1.0 N). Weighted by angle_loss_weight (3.0).

3. **Reconstruction loss:** L_rec = ||o_hat_{t+1} - o_{t+1}||^2, weighted by rec_loss_weight (1.0 or 0.0 for ablation).

4. **Torque angle loss (6D only):** Same angular formulation but for the roll-pitch torque plane (tau_roll, tau_pitch). Weighted by torque_angle_loss_weight (3.0 in B4/B5).

5. **Yaw loss (6D only):** Separate MSE on the yaw torque component. Weighted by yaw_loss_weight (3.0 in B4/B5).

Total: L = w_f * L_force + w_a * L_angle + w_r * L_rec + w_ta * L_torque_angle + w_y * L_yaw

#### 4.3 Compliance Injection

The estimated force is smoothed with an exponential moving average (alpha = 0.1) and linearly mapped to velocity command modulations:

```
force_ema = alpha * F_hat + (1 - alpha) * force_ema_prev
obs["policy"][:, 6:8] += compliance_k * force_ema[:, :2]
```

Where compliance_k = 0.06 (default). This modifies the velocity commands (vx, vy) seen by the policy, causing the robot to walk in the direction of the applied force---yielding compliant behavior without any change to the underlying locomotion policy.

---

### 5. Ablation Studies

#### 5.1 Batch 1: Estimator Design Space (force range 0--20 N)

| ID | Estimate dims | History | Reconstruction | Network | Notes |
|----|--------------|---------|---------------|---------|-------|
| A1 | 3D (Fx,Fy,Fz) | h=10 | Yes | [128,64]+[32,16] | Short history baseline |
| A2 | 3D | h=40 | Yes | Default | Long history |
| B1 | 6D wrench | h=10 | Yes | Default | Short history, full wrench |
| B2 | 6D wrench | h=40 | Yes | Default | Long history, full wrench |
| B3 | 6D wrench | h=40 | Yes | [256,128]+[64,32] | 2x network capacity |
| B4 | 6D wrench | h=40 | Yes | Bigger + torque losses | +torque angle + yaw loss |
| B5 | 6D wrench | h=60 | Yes | Bigger + torque losses | Longer history + 10 Nm torque range |
| C1 | 3D | h=10 | No | Default | No reconstruction |
| C2 | 3D | h=40 | No | Default | No reconstruction |
| E1 | 3D | h=20 | Yes | Default | + compliance force reward |
| E2 | 4D (Fx,Fy,Fz,tau_yaw) | h=20 | Yes | Default | + compliance force+torque reward |

#### 5.2 Batch 2 (planned): Dimension Sweep at 100 N

| ID | Estimate | History | Network | Notes |
|----|----------|---------|---------|-------|
| G1 | 2D (Fx, Fy) | h=20 | Default | Minimal estimation |
| G2 | 2D | h=40 | Default | |
| G3 | 3D (Fx, Fy, Fz) | h=20 | Default | Baseline |
| G4 | 3D | h=40 | Bigger | Best from batch 1 |
| G5 | 2D wrench (Fx, Fy, tau_yaw) | h=20 | Default | Practical wrench subset |
| G6 | 2D wrench | h=40 | Default | |
| G7 | 6D wrench | h=20 | Default | Full wrench |
| G8 | 6D wrench | h=40 | Bigger | |

#### 5.3 Evaluation Methodology

**Static 360-degree force sweep:**
- Robot stands still (zero velocity commands).
- Forces applied from 10 equally spaced directions (0, 36, 72, ..., 324 degrees) at magnitudes [5, 10, 15, 20, 25] N.
- 20 trials per (magnitude, direction) combination = 1000 parallel environments.
- Force held for 4 seconds after 3-second warmup.
- Metrics recorded per step: position, velocity, force estimate.
- Evaluated with and without compliance mapping (mapping vs. nomapping).

**Metrics:**
- **Peak displacement:** Maximum Euclidean distance from start position during force application.
- **Effective compliance:** C = (1/T) * sum((v - v') . F / (F . F)), units s/kg. Measures velocity response per unit force.
- **Estimator RMSE:** sqrt(mean(||F_hat - F_gt||^2)) over the force application period.
- **Angular error:** Median angular deviation between estimated and ground-truth force direction.
- **Fall rate:** Fraction of trials where the robot fell during force application.

**Dynamic compliance evaluation:**
- Robot walks along a straight line with PI controller for lateral tracking.
- Forces applied perpendicular to walking direction at varying magnitudes.
- Measures lateral deviation, velocity decay after force removal, return-to-path time.

#### 5.4 Key Findings from Batch 1

- Increasing network capacity (B3 vs B2) reduces the importance of the reconstruction (forward dynamics) loss---the larger network can directly learn force features without the auxiliary task.
- Doubling history from h=10 to h=40 combined with the bigger network yields ~7 degree median angular error (B3).
- 6D wrench estimation is feasible: torque MAE of ~0.5--0.7 Nm at 0--5 Nm applied range (~10--14% relative), with yaw torque being hardest to estimate.
- Removing reconstruction loss (C1, C2) degrades angular accuracy, especially at shorter history lengths.
- Compliance reward (E1, E2) adds an explicit incentive for the policy to walk in the force direction, showing improved velocity alignment but not necessarily better estimation accuracy.

---

### 6. Sim-to-Real Transfer

#### 6.1 Sim-to-Sim Validation

Policies and estimators are exported as TorchScript JIT models. A MuJoCo simulation of the Go2 serves as an intermediate validation step. External forces are applied via a UDP interface on port 9870. Known issues: observation discrepancies between Isaac Sim and MuJoCo (joint velocity noise, torque estimation differences) cause Fz bias (~+5 N with no external force).

#### 6.2 Real-World Deployment

The deployment script runs at 50 Hz on the Go2's onboard Jetson. The system provides three compliance modes selectable via a gamepad toggle:

- **Normal mapping:** obs[:, 6:8] += k * force_ema (robot yields in force direction)
- **Off (no mapping):** Force estimation runs but does not modulate velocity commands
- **Inverted mapping:** obs[:, 6:8] -= k * force_ema (robot resists the force)

Real-world recording sessions capture per-step: raw observations, actions, force estimates, force EMA, velocity commands, compliance mode, and a recording toggle for marking segments of interest.

Analysis of a 12.7-minute real-world session (38,087 steps) identified 6 recording segments across the three modes. The force estimator produces coherent estimates when the robot is pushed, with force magnitude increasing during contact and returning to near-zero after release.

---

### 7. Conclusion

(To be written after batch 2 results are available. Should summarize: proprioceptive force estimation works, ablation identifies key design choices, compliance mapping provides sensor-free yielding behavior, sim-to-real transfer demonstrated.)

---

### 8. Outlook

- **Higher force training (100 N):** Current training at 20 N is out-of-distribution for realistic human pushes. Batch 2 retrains at 100 N.
- **Estimator-aware locomotion:** Adding force estimation MSE as a policy reward term so the locomotion controller learns gaits that expose more informative proprioceptive signals for force estimation.
- **Slope and payload evaluation:** Testing on inclined surfaces and with added mass to evaluate robustness.
- **Hierarchical compliance (HAC-LOCO):** A trainable high-level controller that outputs residual velocity commands based on the estimated force, replacing the fixed linear mapping with a learned compliance behavior.
- **PAINT teacher-student distillation:** Training a student policy that implicitly learns compliance from a privileged teacher, eliminating the need for an explicit compliance mapping stage.

---

## Key References (to cite)

- Rudin et al., "Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning," CoRL 2022
- Hwangbo et al., "Learning Agile and Dynamic Motor Skills for Legged Robots," Science Robotics 2019
- Lee et al., "Learning Quadrupedal Locomotion over Challenging Terrain," Science Robotics 2020
- Hartmann et al. (HAC-LOCO), "Hierarchical Admittance Control for Compliant Locomotion," 2024
- Ji et al. (PAINT), "Proprioceptive Active Inference for Nimble Terrestrial locomotion," 2024
- Schulman et al., "Proximal Policy Optimization Algorithms," 2017
- Mittal et al., "Orbit: A Unified Simulation Framework for Interactive Robot Learning Environments," RA-L 2023

---

## Notation Table

| Symbol | Description |
|--------|-------------|
| o_t | Proprioceptive observation at time t (57 dims) |
| o_t^H | History buffer [H x 57] |
| z_t | Latent encoding from temporal encoder |
| F_hat | Estimated force/wrench vector |
| F_gt | Ground-truth applied force/wrench |
| H | Temporal history length (number of past steps) |
| k | Compliance gain (default 0.06) |
| alpha | EMA smoothing factor (default 0.1) |
| B | Virtual impedance for compliance reward (N*s/m) |
| sigma | Kernel width for compliance reward |
| C | Effective compliance (s/kg) |
| pi | Locomotion policy: o_t -> a_t |
| a_t | Joint position targets (12 dims) |
| v_cmd | Velocity commands [vx, vy, omega_z] |
