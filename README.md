# RoboBarrow: Compliant Force-Based Control for Quadrupedal Robots

Admittance-based compliant locomotion on the Unitree Go2 without dedicated force sensors. Built on NVIDIA Isaac Lab.

<p align="center">
  <img src="docs/compliance_demo.gif" alt="Force-compliant locomotion demo" width="600"/>
</p>

## TL;DR

A proprioceptive force estimator trained concurrently with a velocity-tracking locomotion policy estimates external forces and torques from joint-level observations. A first-order admittance controller maps the estimated wrench to velocity command modulations, enabling tunable compliance through a single gain constant adjustable at deployment without retraining. The robot can be pushed, pulled, and steered through applied forces and yaw torques across unstructured outdoor terrain.

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
