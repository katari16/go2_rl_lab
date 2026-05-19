# RoboBarrow: Compliant Force-Based Control for Quadrupedal Robots

**Proprioceptive Force and Wrench Estimation for Compliant Quadrupedal Locomotion**

Bachelor Thesis, Spring Term 2026 — Robotic Systems Lab (RSL), ETH Zurich

*Author:* Hans Baumann-Ortiz | *Supervisors:* William Hartmann, Filip Janovsky | *Lecturer:* Prof. Dr. Marco Hutter

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

## Key Results

Best estimator performance across the ablation study (evaluated on 4096 parallel environments, 0–30 N force range):

| Configuration | Dims | History | Force MAE (N) | Ang. Error (°) | Torque MAE (Nm) | Rel. Error (%) |
|---|---|---|---|---|---|---|
| **J5 (deployed, TCN)** | 4D | H=40 | **3.94** | **4.14** | 0.57 | 35.6 |
| J3 (est-acc reward) | 4D | H=40 | 4.68 | 4.97 | 0.83 | 39.5 |
| P4 (long history) | 4D | H=40 | 5.05 | 5.38 | 0.74 | 43.5 |
| P17 (6D big net) | 6D | H=30 | 5.15 | 5.34 | 0.71 | 43.6 |
| P16 (6D default) | 6D | H=30 | 4.95 | 5.67 | 0.81 | 39.4 |
| P1 (short history) | 4D | H=10 | 6.10 | 10.25 | 0.79 | 49.8 |
| P20 (high PD gains) | 4D | H=30 | 5.41 | 6.52 | 0.80 | 45.3 |

Key findings:
- **TCN preprocessing** yields the single largest improvement: 16% lower force MAE and 17% tighter angular accuracy vs. MLP-only (J5 vs J3).
- **Low PD gains** (Kp=8) outperform conventional gains (Kp=25) by amplifying joint deflection under external forces, producing a richer proprioceptive signal.
- **6D wrench estimation** is feasible with torque MAE of ~0.7 Nm (10–14% relative at 0–5 Nm range), though XY force accuracy degrades slightly compared to 4D.
- **History length** H=40 saturates accuracy; H=10 is insufficient (2× angular error).

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
