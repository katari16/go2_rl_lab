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

Quadrupedal robots operating alongside humans or in unstructured environments must be able to detect and respond to external contact forces. We present a proprioceptive force and wrench estimation framework for the Unitree Go2 quadruped, trained entirely in simulation using deep reinforcement learning. The estimator processes a temporal history of proprioceptive observations — angular velocities, projected gravity, joint positions, joint velocities, and joint torques — through a learned encoder to predict external forces and torques applied to the robot base, without requiring any dedicated force sensor.

We conduct a systematic ablation study over the estimator design space, varying the number of estimated dimensions (2D force, 3D force, 2D wrench, 6D wrench), temporal history length, network capacity, and auxiliary training losses. We evaluate estimation accuracy using force magnitude MSE, angular error, and a novel effective compliance metric that measures the robot's velocity response per unit applied force.

To achieve compliant behavior, the estimated force is mapped to velocity command modulations via a linear gain, allowing the robot to yield in the direction of applied forces without retraining the locomotion policy. We validate the approach in simulation with a 360-degree force sweep evaluation and demonstrate sim-to-real transfer on the physical Go2 robot, confirming that the proprioceptive estimator generalizes to real-world contact.

## Contributions

1. **Proprioceptive force/wrench estimator** — A temporal encoder trained end-to-end with the locomotion policy that predicts up to 6D external wrench (Fx, Fy, Fz, τ_roll, τ_pitch, τ_yaw) from joint-level measurements alone, running at 50 Hz on the robot's onboard compute.
2. **Systematic ablation study** — Comprehensive evaluation of the estimator design space across 7 axes: history length, network capacity, estimated dimensions (2D→6D), reconstruction loss, TCN temporal preprocessing, estimation-accuracy reward, and PD gain selection.
3. **Sensor-free compliance mapping** — A first-order admittance controller that maps the estimated force to velocity command modulations via a single tunable gain, enabling compliant behavior without retraining the base locomotion policy.
4. **Sim-to-real validation** — Full deployment pipeline from Isaac Lab training through MuJoCo sim-to-sim validation to real-world operation on the physical Unitree Go2, with three runtime-switchable compliance modes (yield, off, resist).

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
│   ├── rsl_rl/                 # Training, evaluation, and export scripts
│   └── cluster/                # SLURM training scripts for ETH Euler cluster
├── deploy/
│   ├── deploy_real/            # Real robot deployment (Unitree SDK2)
│   ├── sim2sim/                # MuJoCo sim-to-sim validation
│   └── pre_train/              # Exported JIT policy + estimator checkpoints
└── docs/                       # Architecture docs and ablation logs
```

## Environment and Task Definitions

All environments are defined in:

```
source/go2_rl_lab/go2_rl_lab/tasks/manager_based/go2_rl_lab/
├── go2_lowlevel_env_cfg.py     # Base locomotion + force estimation (deployed config)
├── go2_6dctrl_env_cfg.py       # 6D wrench estimation (Fx, Fy, Fz, τ_roll, τ_pitch, τ_yaw)
├── go2_ablation_env_cfgs.py    # All ablation variants (P-series, J-series, R-series)
├── go2_payload_env_cfg.py      # Payload transport (1-3 kg randomized)
└── mdp/                        # Observations, rewards, events, curricula
```

Agent configs (PPO hyperparameters, network sizes):

```
source/go2_rl_lab/go2_rl_lab/tasks/manager_based/go2_rl_lab/agents/
├── rsl_rl_ppo_cfg.py           # Base PPO config
├── rsl_rl_lowlevel_cfg.py      # Low-level locomotion training
└── rsl_rl_ablation_cfg.py      # Ablation sweep configs
```

## Force Estimator and Runner

```
source/go2_rl_lab/go2_rl_lab/estimator/
├── force_estimator.py              # TCN-based force estimator network
├── obs_history_buffer.py           # Sliding window history buffer
├── compliant_on_policy_runner.py   # Main training runner (policy + estimator jointly)
└── frozen_policy_estimator_runner.py  # Train estimator with frozen policy
```

The runner (`compliant_on_policy_runner.py`) trains the locomotion policy and force estimator jointly. The estimator is activated after the policy reaches a reward threshold, and force application begins after directional accuracy meets a gate condition.

## Training

```bash
python scripts/rsl_rl/train.py --task Go2-Lowlevel-v0 --num_envs 4096 --max_iterations 10000
```

Ablation runs are launched via cluster scripts in `scripts/cluster/`. Each script corresponds to a specific configuration documented in the report appendix.

## Evaluation

Three evaluation protocols are implemented:

| Script | Protocol | Description |
|--------|----------|-------------|
| `scripts/rsl_rl/static_eval.py` | Training-regime rollout | Constant forces, 4096 envs, 20s |
| `scripts/rsl_rl/eval/ou_force_eval.py` | OU disturbance | Smooth, continuously varying forces |
| `scripts/rsl_rl/eval/static_360_eval.py` | Directional sweep | Fixed 20 N from 10 azimuth directions |

Batch evaluation scripts for all report ablations:

```bash
scripts/rsl_rl/run_evals_report_ablations.sh   # P-series (architecture ablation)
scripts/rsl_rl/run_evals_r1_r3_r4.sh           # Randomization ablation
scripts/rsl_rl/run_evals_r6_r8.sh              # Force curriculum ablation
scripts/rsl_rl/run_evals_r9_r12.sh             # Privileged observation ablation
```

## Pretrained Policies

Exported JIT checkpoints (policy.pt + estimator.pt) for all report configurations:

```
deploy/pre_train/
├── ablation_6dctrl_total50/    # Deployed configuration (6D, TCN H=30)
├── ablation_p1/ ... p20/       # Architecture ablation (history, network size)
├── ablation_j3/, ablation_j5/  # Joint-level variants
└── payload_3kg/                # Payload transport policy
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
