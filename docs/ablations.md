# Ablation Study — Task Configurations

All ablation tasks are registered in `source/go2_rl_lab/go2_rl_lab/tasks/manager_based/go2_rl_lab/__init__.py` following the pattern `Go2-Est-<Axis>-<Variant>-v0`. Each task pairs an **environment config** (what the robot sees and experiences) with a **runner config** (PPO hyperparameters + estimator architecture).

## How tasks are composed

```
Task ID (e.g. Go2-Est-History-H10-v0)
  = Environment config (scene, observations, rewards, force events, randomization)
  + Runner config (PPO network dims, estimator architecture, loss weights, training gates)
```

- **Environment configs** live in `tasks/manager_based/go2_rl_lab/go2_*.py`
- **Runner configs** live in `tasks/manager_based/go2_rl_lab/agents/rsl_rl_*_cfg.py`
- Ablations modify either the env (e.g., different PD gains, force profile, privileged obs) or the runner (e.g., history length, network size, force_dim), or both.

## Deployed and Base Configurations

| Task ID | Env config | Runner config | Description |
|---|---|---|---|
| `Go2-Est-Deploy-v0` | `Go26DctrlEnvCfg` | `Ablation6DctrlTotal50Cfg` | Deployed: 6D wrench, TCN, H=30, big net, total reward gate @ 50 |
| `Go2-Est-Payload-v0` | `LowLevelPayloadEnvCfg` | `AblationP18Cfg` | Payload transport (1–3 kg randomized mass) |
| `Go2-LowLevel-v0` | `LowLevelEnvCfg` | `LowLevelRunnerCfg` | Base locomotion + 3D force estimation |
| `Go2-LowLevel-NoEst-v0` | `LowLevelNoEstEnvCfg` | `LowLevelNoEstRunnerCfg` | Base locomotion without force estimator |

## Estimator Architecture Ablations (Report Chapter 5)

### History Length

| Task ID | What changes | Runner param |
|---|---|---|
| `Go2-Est-History-H10-v0` | H=10 timesteps | `temporal_steps=10` |
| `Go2-Est-History-H20-v0` | H=20 timesteps | `temporal_steps=20` |
| `Go2-Est-History-H30-v0` | H=30 timesteps (baseline) | `temporal_steps=30` |
| `Go2-Est-History-H40-v0` | H=40 timesteps | `temporal_steps=40` |

### TCN Preprocessor

| Task ID | What changes | Runner param |
|---|---|---|
| `Go2-Est-TCN-None-v0` | MLP encoder only | `use_tcn=False` |
| `Go2-Est-TCN-Pre-v0` | Temporal conv before MLP | `use_tcn=True` |

### Network Capacity

| Task ID | What changes | Runner params |
|---|---|---|
| `Go2-Est-NetSize-Half-v0` | Half width | Encoder [64,32], head [16,8] |
| `Go2-Est-NetSize-Default-v0` | Default width (baseline) | Encoder [128,64], head [32,16] |
| `Go2-Est-NetSize-Double-v0` | Double width | Encoder [256,128], head [64,32] |

### Reconstruction Loss

| Task ID | What changes | Runner/Env param |
|---|---|---|
| `Go2-Est-RecLoss-With-v0` | Auxiliary reconstruction decoder active | `rec_loss_weight=1.0` |
| `Go2-Est-RecLoss-None-v0` | No reconstruction decoder | `rec_loss_weight=0` |
| `Go2-Est-RecLoss-NoneEstAcc-v0` | No decoder + est-accuracy reward | `rec_loss_weight=0` + env reward term |

### Wrench Dimensionality

| Task ID | What changes | Runner param |
|---|---|---|
| `Go2-Est-Dim-2D-v0` | Fx, Fy only | `force_dim=2` |
| `Go2-Est-Dim-3DxyYaw-v0` | Fx, Fy, τ_yaw | `force_dim=3` |
| `Go2-Est-Dim-4D-v0` | Fx, Fy, Fz, τ_yaw | `force_dim=4` |
| `Go2-Est-Dim-6D-v0` | Full 6D wrench, default net | `force_dim=6` |
| `Go2-Est-Dim-6DBig-v0` | Full 6D wrench, bigger encoder | `force_dim=6`, enc [256,128] |

### PD Gains

| Task ID | What changes | Env param |
|---|---|---|
| `Go2-Est-PD-Low-v0` | Kp=8, Kd=0.4 (baseline) | Lower gains improve force observability |
| `Go2-Est-PD-Default-v0` | Kp=25, Kd=0.5 | Unitree factory defaults |

## Domain Randomization and Observability (Report Section 5.1.6)

### Domain Randomization

| Task ID | What changes (env-level) |
|---|---|
| `Go2-Est-DomRand-Full-v0` | Full randomization: mass ±[−1,+3] kg, obs noise, random pushes |
| `Go2-Est-DomRand-NoMass-v0` | Removes mass randomization only |
| `Go2-Est-DomRand-None-v0` | No randomization (clean simulation) |

### Force Curriculum

| Task ID | What changes (runner-level) |
|---|---|
| `Go2-Est-Curriculum-HardGate-v0` | Hard step function: forces activate at reward threshold |
| `Go2-Est-Curriculum-LinearRamp-v0` | Linear ramp from 10→30 N over 2500 iterations |
| `Go2-Est-Curriculum-Bucketed-v0` | Bucketed: 10/20/30 N at iterations 0/1000/2000 |

### Privileged Observations

| Task ID | What changes (env-level) |
|---|---|
| `Go2-Est-Priv-All-v0` | All privileged inputs: mass, base velocity, foot contacts |
| `Go2-Est-Priv-AllNoRand-v0` | All privileged + no randomization (estimation floor) |
| `Go2-Est-Priv-Velocity-v0` | Base linear velocity only |
| `Go2-Est-Priv-Contacts-v0` | Foot contact forces only |

## Training an ablation

```bash
python scripts/rsl_rl/train.py --task Go2-Est-History-H10-v0 --num_envs 4096 --headless
```

## Evaluating an ablation

```bash
./scripts/rsl_rl/run_eval.sh --task Go2-Est-History-H10-v0 --checkpoint logs/rsl_rl/<experiment>/model_9500.pt
```
