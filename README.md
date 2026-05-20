# RoboBarrow: Compliant Force-Based Control for Quadrupedal Robots

**Proprioceptive Force and Wrench Estimation for Compliant Quadrupedal Locomotion**

Bachelor Thesis, Spring Term 2026 — Robotic Systems Lab (RSL), ETH Zurich

*Author:* Hans Baumann-Ortiz | *Supervisors:* Filip Bjelonic, William Talbot | *Lecturer:* Prof. Dr. Marco Hutter

<p align="center">
  <img src="docs/main_figure_robobarrow.png" alt="RoboBarrow real-world deployment" width="600"/>
  <br>
  <em>Figure 1.1: Real-world force-compliant locomotion on the Unitree Go2. (A) Guided walking via leash force, (B) outdoor terrain, (C) gravel, (D) payload transport (3 kg), (E) compliance mode switching, (F) slope traversal (13°). All behaviors driven by proprioceptive force estimation without dedicated force sensors.</em>
</p>

## Abstract

Legged robots offer terrain versatility that wheeled platforms cannot match, yet they currently lack the intuitive force-based interaction that makes wheelbarrows and hand carts easy to use. This work combines the agility of a quadruped with the ease of use of a wheeled barrow by enabling compliant force-based control on the Unitree Go2 without dedicated force sensors.

Two contributions are presented. First, an admittance-based compliant control framework for legged locomotion: a proprioceptive force estimator trained concurrently with a velocity tracking policy estimates external forces and torques from joint-level observations, and a model-based admittance controller maps the estimated wrench to velocity command modulations, providing tunable compliance through a single gain constant adjustable at deployment without retraining. Second, an extensive evaluation of the framework in simulation across architectural ablations, domain randomization, and observability conditions, with qualitative validation on the physical Unitree Go2 over unstructured outdoor terrain.

The force estimator achieves 3 N MAE on horizontal forces, 4.1° median directional accuracy, and 0.57 Nm yaw torque MAE in simulation. The deployed system demonstrates compliant navigation of outdoor terrain including gravel, grass, and slopes up to 13°, as well as payload transport of 3 kg through a parkour environment. Together, these results establish a general-purpose recipe for learning compliant force-based control on quadrupedal robots, enabling intuitive human guidance while carrying static payloads across diverse environments.

## Contributions

This work presents two main contributions. First, we develop an admittance-based learning framework for compliant legged locomotion. A velocity tracking locomotion policy is trained in simulation for the Unitree Go2 quadruped and transferred to the real robot for deployment on unstructured outdoor terrain. A proprioceptive force estimator, trained concurrently with the policy, estimates external forces and torques applied to the robot's base from joint-level observations alone. A first-order admittance controller maps the estimated wrench to velocity command modulations, enabling tunable compliance through a single gain constant *k* without retraining.

Second, we present an extensive evaluation of the force estimator and the compliant control framework. The estimator is characterised in simulation across force magnitudes, directions, and temporal profiles. An ablation study across 15 architectural configurations identifies key design choices for proprioceptive force estimation. An observability analysis quantifies the error introduced by domain randomization and establishes the estimation floor imposed by the proprioceptive observation set. The deployed system is validated on the physical robot over gravel, grass, and slopes up to 13°, and during payload transport of 3 kg through a parkour environment.

<p align="center">
  <img src="docs/training_pipeline.png" alt="Training and deployment pipeline" width="700"/>
  <br>
  <em>Figure 4.1: Training pipeline. The force estimator processes a proprioceptive history buffer and outputs a 4D wrench estimate. The compliance module modulates the velocity commands via admittance control. The locomotion policy produces joint position targets tracked by the on-board PD controllers at 50 Hz.</em>
</p>

## Results

Deployed estimator per-axis accuracy (training-regime rollout, 4096 environments, 20 s):

| Component | MAE ± std | Relative error |
|---|---|---|
| F_x | 3.00 ± 0.61 N | 36% |
| F_y | 2.57 ± 0.47 N | |
| F_z | 6.24 ± 2.81 N | |
| τ_yaw | 0.57 ± 0.12 Nm | 9.8% |
| **Angular error (median)** | **4.1 deg** | |

Comparison under training-regime and OU (continuously varying) disturbance protocols:

| Metric | Training-regime | OU disturbance |
|---|---|---|
| F_x MAE | 3.00 N | 4.41 N |
| F_y MAE | 2.57 N | 4.00 N |
| F_z MAE | 6.24 N | 8.85 N |
| τ_yaw MAE | 0.57 Nm | 0.98 Nm |
| Angular error (median) | 4.1 deg | 10.4 deg |
| Relative error (force) | 36% | 55% |

Effect of domain randomization and privileged observations (TCN architecture, H=30, 6D output):

| Configuration | Privileged inputs | F_x | F_y | F_z | Force MAE | Ang. |
|---|---|---|---|---|---|---|
| Baseline (deployed) | none | 3.00 | 2.93 | 6.88 | 4.27 | 4.3° |
| No mass randomization | none | 2.35 | 2.67 | 2.22 | 2.41 | 3.8° |
| No randomization | none | 2.28 | 2.50 | 2.99 | 2.59 | 3.3° |
| All privileged | mass, vel., contacts | 1.81 | 2.02 | 2.00 | 1.94 | 2.7° |
| All privileged, no rand. | mass, vel., contacts | 1.82 | 1.76 | 1.92 | 1.83 | 2.6° |
| Velocity only | base lin. vel. | 2.04 | 2.12 | 5.92 | 3.36 | 3.0° |
| Contacts only | foot forces | 3.10 | 3.11 | 7.02 | 4.41 | 4.6° |

The architectural floor with all privileged inputs is 1.83 N force MAE and 2.6° angular error, indicating that approximately half of the deployed horizontal error originates from information unavailable in the proprioceptive stream. Real-world validation on the physical Go2 confirms sim-to-real transfer: the estimator tracks a 20.5 N static pull with a mean bias of −0.3 N across four repeated trials.

## Repository Structure

```
go2_rl_lab/
├── source/go2_rl_lab/          # Isaac Lab extension (env definitions, estimator, runner)
├── scripts/
│   └── rsl_rl/                 # Training, evaluation, and export scripts
├── deploy/
│   ├── deploy_real/            # Real robot deployment (Unitree SDK2)
│   ├── sim2sim/                # MuJoCo sim-to-sim validation
│   └── pre_train/              # Exported JIT policy + estimator checkpoints
└── docs/                       # Architecture docs and ablation logs
```

## Environment and Task Definitions

All environments are registered in `source/go2_rl_lab/go2_rl_lab/tasks/manager_based/go2_rl_lab/__init__.py`.

### Core Configurations

| Task ID | Description | Config file |
|---|---|---|
| `Go2-LowLevel-v0` | Base locomotion + 3D force estimation | `go2_lowlevel_env_cfg.py` |
| `Go2-Ablation-6Dctrl-Total50-v0` | Deployed config: 6D wrench, TCN, H=30 | `go2_6dctrl_env_cfg.py` |
| `Go2-LowLevel-Payload-v0` | Payload transport (1–3 kg randomized) | `go2_payload_env_cfg.py` |

### Ablation Configurations

All ablation variants are defined in `go2_ablation_env_cfgs.py` and `rsl_rl_ablation_cfg.py`. The study axes correspond to report sections:

| Ablation axis (Report Section) | Task IDs | What varies |
|---|---|---|
| History length (§A.2) | P1–P4 | H ∈ {10, 20, 30, 40} steps |
| Network capacity (§A.3) | P5, P3, P6 | Encoder width: half / baseline / double |
| TCN preprocessor (§A.4) | J3, J5 | MLP encoder vs TCN temporal convolution |
| Reconstruction loss (§A.5) | P3 vs P11, J3 vs J6 | With / without auxiliary reconstruction |
| Wrench dimensionality (§A.6) | P13, P14, P3, P16, P17 | 2D / 3D / 4D / 6D output |
| PD gains | P3, P20 | Kp=8 (low) vs Kp=25 (default) |
| Domain randomization (§5.1.6) | R1, R3, R4 | Full / no mass / no randomization |
| Force curriculum (§5.1.6) | R1, R6, R8 | Hard gate / linear ramp / bucketed |
| Privileged observations (§5.1.6) | R9–R12 | All / velocity / contacts / none |

## Force Estimator and Runner

```
source/go2_rl_lab/go2_rl_lab/estimator/
├── force_estimator.py              # TCN-based force estimator network
├── obs_history_buffer.py           # Sliding window history buffer
└── compliant_on_policy_runner.py   # Main training runner (policy + estimator jointly)
```

The runner (`compliant_on_policy_runner.py`) trains the locomotion policy and force estimator jointly. The estimator is activated after the policy reaches a reward threshold, and force application begins after directional accuracy meets a gate condition.

## Training

```bash
python scripts/rsl_rl/train.py --task Go2-Ablation-6Dctrl-Total50-v0 --num_envs 4096 --max_iterations 10000
```

## Evaluation

Three evaluation protocols are implemented:

| Script | Protocol | Description |
|--------|----------|-------------|
| `scripts/rsl_rl/static_eval.py` | Training-regime rollout | Constant forces, 12 envs, time-series data |
| `scripts/rsl_rl/eval/ou_force_eval.py` | OU disturbance | Smooth, continuously varying forces |
| `scripts/rsl_rl/eval/rollout_estimator_eval.py` | Training-regime rollout | 4096 envs, 20s, aggregate metrics |

A unified evaluation script runs all three protocols for any configuration:

```bash
# Single task
./scripts/rsl_rl/run_eval.sh --task Go2-Ablation-R1-v0 --checkpoint <path_to_model.pt>

# Predefined groups (matching report sections)
./scripts/rsl_rl/run_eval.sh --group architecture
./scripts/rsl_rl/run_eval.sh --group randomization
./scripts/rsl_rl/run_eval.sh --group curriculum
./scripts/rsl_rl/run_eval.sh --group observability
./scripts/rsl_rl/run_eval.sh --group deployed
```

## Pretrained Policies

Exported JIT checkpoints (policy.pt + estimator.pt) for report configurations:

```
deploy/pre_train/
├── ablation_6dctrl_total50/    # Deployed configuration (6D, TCN, H=30)
├── ablation_p1/ ... p6/        # History length and network capacity sweeps
├── ablation_j3/, ablation_j5/  # TCN preprocessor variants
├── ablation_p18/               # Payload transport policy
└── payload_3kg/                # Payload transport (standalone)
```

## Sim-to-Sim Deployment (MuJoCo)

Validate trained policies in MuJoCo before hardware deployment:

```bash
python deploy/sim2sim/sim2sim_compliant_no_foot_xyz.py
```

Supports joystick control and UDP-based force application for testing.

## Real Robot Deployment

Deploy on the physical Unitree Go2 via the Unitree SDK2:

```bash
python deploy/deploy_real/deploy_6dctrl.py <network_interface> go2_ablation_6dctrl_total50.yaml
```

Configuration files in `deploy/deploy_real/configs/` specify the policy path, estimator path, observation dimensions, and compliance gain.

## Key Design Choices

- **PD gains**: Kp=8, Kd=0.4 (lower than Unitree defaults to improve exploration and force observability)
- **Estimator**: TCN preprocessor, H=30 history steps, 6D wrench output
- **Force curriculum**: Hard gate after reward threshold, forces up to 30 N per axis
- **Domain randomization**: Base mass (-1.0 to +3.0 kg), observation noise, random pushes
- **Admittance control**: First-order, gain k adjustable at deployment

## Dependencies

- NVIDIA Isaac Lab (Isaac Sim 4.x)
- RSL-RL (PPO implementation)
- PyTorch
- MuJoCo (for sim2sim)
- Unitree SDK2 Python (for real deployment)
