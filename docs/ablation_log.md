# Force Estimator Ablation Log

Tracks all ablation runs, their configs, known bugs, and key results.
Each entry records what was *actually* trained, not just what was intended.

## Shared defaults

All runs inherit from `LowLevelRunnerCfg` unless noted otherwise.

- **Policy:** [512, 256, 128] MLP, 12 joint actions
- **Obs:** 57 raw dims (no foot contacts) + force_dim estimate = policy input
- **Critic:** raw obs + base_lin_vel(3) + foot_contacts(4) + GT_force(3+)
- **Training:** 4096 envs, decimation=4, dt=0.005, control_dt=0.02 (50Hz), 20s episodes
- **Estimator losses (default):** force MSE (w=1.0), angle MSE (w=3.0), rec MSE (w=1.0)
- **Force curriculum:** activated when mean episode reward > threshold, ramps linearly
- **Mapping curriculum:** activated when angular error median < threshold

## Batch 1 (20N, branch: `feature/force-estimation-xyz`)

Max force: 20N per axis, fz_scale=0.6 (Fz effective 0-12N).

| ID | Log dir | Dims | h | Rec | Network | Env | Notes |
|----|---------|------|---|-----|---------|-----|-------|
| A1 | `ablation_A1_h10_3d_rec` | 3D | 10 | yes | default [128,64]+[32,16] | baseline (force only) | Short history baseline |
| A2 | `ablation_A2_h40_3d_rec` | 3D | 40 | yes | default | baseline | Long history baseline |
| B1 | `ablation_B1_h10_6d_rec` | 6D | 10 | yes | default | wrench | 6D short history |
| B2 | `ablation_B2_h40_6d_rec` | 6D | 40 | yes | default | wrench | 6D long history |
| B3 | `ablation_B3_h40_6d_rec_big` | 6D | 40 | yes | bigger [256,128]+[64,32] | wrench | Bigger network |
| B4 | `ablation_B4_h40_6d_rec_big_tqloss` | 6D | 40 | yes | bigger | wrench | **Bug:** torque_angle + yaw loss configured but never passed to estimator (runner bug, fixed in commit 60d4026). Ran without these losses. |
| B5 | `ablation_B5_h60_6d_rec_big_tqloss` | 6D | 60 | yes | bigger | wrench (high torque 0-10Nm) | **Same bug as B4.** Torque losses not applied. |
| C1 | `ablation_C1_h10_3d_norec` | 3D | 10 | no | default | baseline | No reconstruction loss |
| C2 | `ablation_C2_h40_3d_norec` | 3D | 40 | no | default | baseline | No reconstruction loss |
| E1 | `ablation_E1_h20_3d_compliance` | 3D | 20 | yes | default | compliance | Compliance reward: walk in direction of force |
| E2 | `ablation_E2_h20_4d_compliance` | 4D | 20 | yes | default | wrench + compliance | Compliance with force + torque tracking |

### Batch 1 key findings

- A2 (h=40) outperforms A1 (h=10): longer history helps 3D estimation
- B3 (bigger net) helps 6D over B2 (default net) at h=40
- C1 vs A1, C2 vs A2: reconstruction loss provides mild regularization benefit
- Relative error ~21% (MAE 3N / 14N mean force)


## Batch 2 — H-series (50N / 100N, branch: `feature/estimator-ablation`)

Max force: 50N or 100N per axis. Each config has both variants.

**Known bug (all H-series initial runs):** `torque_angle_loss_weight` and `yaw_loss_weight` were configured in `_est()` but the runner (`compliant_on_policy_runner.py:62-76`) did not pass them through to `ForceEstimator.__init__`. They were silently ignored. **Fixed in commit 60d4026.** All H-series runs that completed before this fix ran *without* yaw/torque_angle losses despite the config specifying them.

| ID | Log dir | Dims | h | Network | Intended losses | Actually applied losses | Est reward | Env | TCN |
|----|---------|------|---|---------|-----------------|------------------------|------------|-----|-----|
| H1 | `ablation_H1_3d_h30_{50,100}N` | 3D | 30 | default | force+angle+rec | force+angle+rec | no | baseline | none |
| H3a | `ablation_H3a_6d_h40_big_yaw_{50,100}N` | 6D | 40 | bigger | force+angle+rec+yaw+tq_angle | force+angle+rec (bug) | no | wrench | none |
| H3b | `ablation_H3b_6d_h30_big_yaw_{50,100}N` | 6D | 30 | bigger | force+angle+rec+yaw+tq_angle | force+angle+rec (bug) | no | wrench | none |
| H3c | `ablation_H3c_6d_h30_def_yaw_{50,100}N` | 6D | 30 | default | force+angle+rec+yaw+tq_angle | force+angle+rec (bug) | no | wrench | none |
| H5 | `ablation_H5_4d_h30_yaw_{50,100}N` | 4D | 30 | default | force+angle+rec+yaw | force+angle+rec (bug) | no | wrench | none |
| H6 | `ablation_H6_4d_h30_{50,100}N` | 4D | 30 | default | force+angle+rec | force+angle+rec | no | wrench | none |
| H7 | `ablation_H7_4d_h30_estrew_{50,100}N` | 4D | 30 | default | force+angle+rec | force+angle+rec | yes (exp kernel, w=0.5, sigma=1.0) | wrench+est_acc | none |
| H8 | `ablation_H8_3d_h30_estrew_{50,100}N` | 3D | 30 | default | force+angle+rec | force+angle+rec | yes (exp kernel, w=0.5, sigma=1.0) | baseline+est_acc | none |
| H9 | `ablation_H9_2d_h30_{50,100}N` | 2D | 30 | default | force+angle+rec | force+angle+rec | no | baseline | none |

### Batch 2 key findings (in progress)

- **100N runs:** Robot struggles to learn locomotion — 100N is extreme for a ~15kg robot
- **50N runs:** Locomotion learns well, force estimation converges
- **Relative error improved:** MAE 4N / 30N mean force = 13% (vs 21% in batch 1)
- **Angular error:** H5 (4D) shows significantly lower median angular deviation (~5 deg vs ~10 deg), likely from estimating tau_yaw (not from yaw loss, which was bugged)
- **Est accuracy reward (H7, H8):** exp(-MSE/sigma) with sigma=1.0 is too tight — reward ~1e-5 (effectively zero gradient). Needs larger sigma or different formulation.
- **H3a observation:** Despite yaw loss bug, 6D estimation with bigger network shows strong performance — the architectural choices matter more than the auxiliary loss here


## Batch 3 — TCN (50N only, branch: `feature/estimator-ablation`)

Based on H3a (6D, h=40, bigger net, yaw+tq_angle losses). These runs use the fixed runner (commit 60d4026) so yaw/torque_angle losses are actually applied.

| ID | Log dir | Dims | h | Network | TCN mode | TCN config | Baseline |
|----|---------|------|---|---------|----------|------------|----------|
| ID | Log dir | Dims | h | Network | TCN mode | TCN config | Est reward | Baseline |
|----|---------|------|---|---------|----------|------------|------------|----------|
| H12a | `ablation_H12a_6d_h40_tcnpre_50N` | 6D | 40 | bigger | preprocessor | ch=[64,64], k=3, dil=[1,2] | no | H3a-50N |
| H12b | `ablation_H12b_6d_h40_tcnrep_50N` | 6D | 40 | bigger | replacement | ch=[64,64], k=3, dil=[1,2] | no | H3a-50N |
| H13a | `ablation_H13a_4d_h30_tcnpre_50N` | 4D | 30 | default | preprocessor | ch=[64,64], k=3, dil=[1,2] | yes (exp, w=0.5, sigma=1.0) | H7-50N |
| H13b | `ablation_H13b_4d_h30_tcnrep_50N` | 4D | 30 | default | replacement | ch=[64,64], k=3, dil=[1,2] | yes (exp, w=0.5, sigma=1.0) | H7-50N |

- **H12a/H13a (preprocessor):** TCN enriches obs history with temporal context → flatten → MLP encoder → force head. Same input shape to MLP, but features are temporally aware.
- **H12b/H13b (replacement):** TCN processes obs history → global avg pool over time → linear projection to z_t → force head. No MLP encoder.

### Ablation axes

- **H12a vs H3a:** Effect of temporal preprocessing on 6D estimation
- **H12b vs H3a:** Can a TCN fully replace the MLP encoder for 6D?
- **H13a vs H7:** Effect of temporal preprocessing on 4D estimation (with est reward)
- **H13b vs H7:** Can a TCN fully replace the MLP encoder for 4D?
- **H12a vs H12b / H13a vs H13b:** Preprocessor (hybrid) vs replacement (pure TCN)


## Batch 4 — Trapezoid force profile (50N only, branch: `feature/estimator-ablation`)

PAINT-style piecewise-linear force envelope with stratified magnitude buckets.
Addresses two training distribution gaps: (1) magnitude clustering around 35-45N,
(2) constant force profile (no ramps/transitions in estimator history).

**Known bug (all runs before this batch):** `torque_range` was set to `(0, 5Nm)` from env init, not gated by the curriculum. Torques were applied during phase 1 (walking-only) before the estimator started training. **Fixed:** `torque_range` now starts at `(0, 0)` and is activated alongside `force_range` by the runner's curriculum gate.

| ID | Log dir | Dims | h | Network | Force profile | Buckets | Baseline |
|----|---------|------|---|---------|---------------|---------|----------|
| H14 | `ablation_H14_6d_h40_big_yaw_trap_50N` | 6D | 40 | bigger | trapezoid (ramp 0.2-0.8s, hold 2-5s, zero 0.5-2s, p_zero=0.02) | [0, 0-10, 10-25, 25-50]N | H3a-50N |

### Force profile details

Envelope s(t) per cycle:
```
ramp_up (0.2-0.8s) → hold (2.0-5.0s) → ramp_down (0.2-0.8s) → zero (0.5-2.0s)
```
Target force sampled once per cycle within the env's assigned magnitude bucket.
2% of cycles are zero-wrench (null case training). Ramp durations randomized per cycle.

### Magnitude buckets (4096 envs ÷ 4)

| Bucket | Envs | Force range | Purpose |
|--------|------|-------------|---------|
| 0 | 0-1023 | 0 N | Null case, prevent baseline bias |
| 1 | 1024-2047 | 0-10 N | Payload-level forces |
| 2 | 2048-3071 | 10-25 N | Mid-range |
| 3 | 3072-4095 | 25-50 N | High forces |

### Ablation axes

- **H14 vs H3a:** Effect of trapezoid force profile + stratified buckets on 6D estimation


## Environment configs reference

| Env config | Force type | Torque range | Extra rewards |
|------------|-----------|--------------|---------------|
| `LowLevelEnvCfg` | XYZ force only (persistent_xyz_force) | N/A | baseline |
| `LowLevelWrenchEnvCfg` | 6D wrench (persistent_wrench) | 0-5 Nm | baseline |
| `LowLevelWrenchHighTorqueEnvCfg` | 6D wrench | 0-10 Nm | baseline |
| `LowLevelEstAccuracyEnvCfg` | XYZ force only | N/A | + force_est_accuracy (w=0.5) |
| `LowLevelWrenchEstAccuracyEnvCfg` | 6D wrench | 0-5 Nm | + force_est_accuracy (w=0.5) |
| `LowLevelWrenchTrapezoidEnvCfg` | 6D trapezoid wrench (stratified buckets) | 0-5 Nm | baseline |

## Network architecture reference

| Label | Encoder | Force head |
|-------|---------|------------|
| default | [128, 64] | [32, 16] |
| bigger | [256, 128] | [64, 32] |

Decoder (reconstruction): always [256, 128] → obs_dim
