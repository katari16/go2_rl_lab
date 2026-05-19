# P-series & PX-series Ablation Runs (2026-04-18)

## Overview

Two new ablation sweeps testing force estimation with **constant persistent wrench** (composer-based) and **direct PhysX wrench** application.

### Key changes from previous batches
- **Force profile**: constant wrench with uniform U(0, max_force) sampling per axis (no stratified buckets by default)
- **Force range**: 30N, fz_scale=0.8, torque 10Nm
- **5% zero-wrench probability** per resample interval (not fixed envs)
- Trapezoid and stratified buckets are now explicit ablations (P21, P22) rather than defaults

## P-series (branch: `feature/estimator-ablation`)

All use `apply_persistent_wrench` (composer-based, interval 3-5s resample).

| ID | Ablation | force_dim | h | Network | Key difference |
|----|----------|-----------|---|---------|----------------|
| P1 | history | 4D | 10 | default | short history |
| P2 | history | 4D | 20 | default | |
| **P3** | **baseline** | **4D** | **30** | **default** | **reference run** |
| P4 | history | 4D | 40 | default | long history |
| P5 | network | 4D | 30 | half [64,32]+[16,8] | smaller net |
| P6 | network | 4D | 30 | double [256,128]+[64,32] | bigger net |
| P7 | reward | 4D | 30 | default | est accuracy reward w=50 |
| P8 | reward | 4D | 30 | default | compliance w=0.5 |
| P9 | reward | 4D | 30 | default | compliance w=1.0 |
| P10 | reward | 4D | 30 | default | compliance w=5.0 |
| P11 | loss | 4D | 30 | default | no reconstruction loss |
| P12 | encoder | 4D | 30 | default+TCN | TCN encoder |
| P13 | dim | 2D | 30 | default | Fx, Fy only |
| P14 | dim | 3D xy_yaw | 30 | default | Fx, Fy, τ_yaw |
| P15 | dim | 4D | 30 | default | =P3 (control) |
| P16 | dim | 6D | 30 | default | full wrench |
| P17 | dim | 6D | 30 | bigger | full wrench, bigger net |
| P18 | domain | 4D | 30 | default | payload 0-4kg |
| P19 | torque | 4D | 30 | default | torque 5Nm (vs 10Nm) |
| P20 | actuator | 4D | 30 | default | default PD Kp=25/Kd=0.5 |
| P21 | force profile | 4D | 30 | default | trapezoid ramp (ramp/hold/ramp) |
| P22 | sampling | 4D | 30 | default | stratified 4-bucket force sampling |

## PX-series (branch: `feature/direct-physx-wrench`)

Both use `apply_paint_wrench` (direct PhysX `apply_forces_and_torques_at_position`, fires every 0.02s).

| ID | Force profile | fz_scale | Buckets | ramp_fraction |
|----|---------------|----------|---------|---------------|
| PX1 | constant | 0.8 | uniform (0,1) | 0.0 |
| PX2 | trapezoid | 0.8 | uniform (0,1) | 0.1 |

## What we expect

- **P3 vs PX1**: Isolates composer vs direct PhysX force application (both constant, same params). Should be similar if both methods are equivalent.
- **P21 vs PX2**: Same comparison for trapezoid profile.
- **P21 vs P3**: Does trapezoid ramp help or hurt estimation vs constant force?
- **P22 vs P3**: Does stratified bucket sampling improve generalization across force magnitudes?
- **P13/P14/P16/P17**: Dimension sweep — how many wrench components can we estimate jointly?
- **P1-P4**: History length sweet spot for 4D estimation.
- **P5/P6**: Network capacity sensitivity.
- **P11**: Whether reconstruction loss matters for composer-based training.

## Cluster submission

```
# P-series (estimator-ablation branch)
sbatch scripts/cluster/ablation_p_series_sweep.sh

# PX-series (physx branch)
sbatch scripts/cluster/ablation_px_series_sweep.sh
```
