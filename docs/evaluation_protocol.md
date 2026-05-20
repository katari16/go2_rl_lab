# Evaluation Protocol — Run Guide

Practical companion to `docs/report/chapters/evaluation.tex`. For every
experiment it lists which runs are involved, which script to call, and
which table / figure in the chapter the output populates.

The chapter itself is the source of truth for **what** is being measured
and why. This file is the source of truth for **how** to reproduce every
number and figure in it.

---

## 0. One-time setup

All four evaluation tests share the same runner-dispatch helpers and the
same data-storage format. The scripts live under `scripts/rsl_rl/eval/`:

| Script | Purpose | Populates |
|---|---|---|
| `static_360_eval.py`   | Test 1 — static 360° sweep standing | §4 master table, §5.1–5.7 findings |
| `step_response_eval.py` | Test 2 — step response | Rise-time column, §5.6 |
| `zero_force_eval.py` | Test 3 — zero-force standing (flat + slope) | Zero-force columns |
| `joystick_zero_force_eval.py` | Test 4 — zero-force joystick trajectory | §5 self-motion table |

Each script writes to

```
data/eval/<experiment_name>/<test_name>_<timestamp>/
    config.json
    raw_data.json
    report.pdf
```

The `raw_data.json` schema is fixed so a single aggregation script
(`scripts/rsl_rl/eval/build_master_table.py`) can crawl all the
eval directories and emit `tab:eval_master` as CSV / LaTeX.

---

## 1. Run roster

All checkpoints live under `logs/rsl_rl/<experiment_name>/<run_timestamp>/`.
Pick `model_<final>.pt` unless otherwise noted.

### 1.1 Group A/B/C/E — 20 N baselines (joint PPO + estimator)

| Label | `experiment_name` | Notes |
|---|---|---|
| A1 | `ablation_A1_h10_3d_rec`       | h=10, 3D force, recon loss |
| A2 | `ablation_A2_h40_3d_rec`       | h=40, 3D force, recon loss |
| B2 | `ablation_B2_h40_6d_rec`       | h=40, 6D wrench, default net |
| B3 | `ablation_B3_h40_6d_rec_big`   | h=40, 6D wrench, big net |
| B4 | `ablation_B4_h40_6d_rec_big_tqloss` | B3 + yaw / torque-angle loss |
| C2 | `ablation_C2_h40_3d_norec`     | A2 without recon loss |
| E1 | `ablation_E1_h20_3d_compliance`| + compliance-force reward |
| E2 | `ablation_E2_h20_4d_compliance`| + compliance force+torque reward |

### 1.2 Group H — 50 / 100 N joint-trained (partial list)

Full list lives in `rsl_rl_ablation_cfg.py`. Start with:

| Label | `experiment_name` |
|---|---|
| H1-50N  | `ablation_H1_3d_h30_50N`  |
| H1-100N | `ablation_H1_3d_h30_100N` |
| H3a-50N | `ablation_H3a_6d_h40_big_yaw_50N`  |
| H3a-100N| `ablation_H3a_6d_h40_big_yaw_100N` |
| H9-50N  | `ablation_H9_2d_h30_50N`  |

Others (H3b, H3c, H5, H6, H7, H8, H12a/b, H13a/b, H14-H19) are included
only in the per-axis subsections where they are directly compared.

### 1.3 Group S — frozen-policy 50 N (S1-S9)

All live under `logs/rsl_rl/ablation_S*_frozen/`. These load
`model_11000.pt` from `go2_lowlevel_no_est/2026-04-13_23-59-10/` as the
frozen base walker.

---

## 2. Test 1 — Static 360° force sweep

Populates:
- `tab:eval_master` columns: Rel. MAE, Ang. err., MAE[0,5]N
- §5.1–5.7 subsection tables and per-axis figures
- `tab:eval_training_distribution` (via the magnitude-binned readout)

### 2.1 Command template

```bash
python scripts/rsl_rl/eval/static_360_eval.py \
  --task <TASK> \
  --checkpoint <PATH_TO_MODEL_PT> \
  --num_envs 128 \
  --force_magnitudes 5 10 20 30 40 50 \
  --num_directions 12 \
  --num_trials 20 \
  --force_hold_s 4.0
```

`<TASK>` must match the run's task registration. Use the `Go2-Est-*`
task IDs (e.g., `Go2-Est-History-H10-v0`, `Go2-Est-Deploy-v0`).
See `__init__.py` for the full registry.

### 2.2 Runs to execute for the master table

All runs in §1. One invocation each. Total: ~30 runs × ~5 min =
~2.5 h on one GPU.

### 2.3 Output → chapter mapping

| JSON field | Goes into |
|---|---|
| `results.summary.rel_mae` | `tab:eval_master` col 1 |
| `results.summary.angular_err_median_deg` | col 2 |
| `results.binned_mae["0-5"]` | col 4 |
| `results.per_direction_mae` | §5.x polar plots |
| `results.binned_mae` (all bins) | `tab:eval_training_distribution`, `fig:eval_binned_mae` |

---

## 3. Test 2 — Step response

Populates: rise-time column of `tab:eval_master`, §5.6.

### 3.1 Command template

```bash
python scripts/rsl_rl/eval/step_response_eval.py \
  --task <TASK> \
  --checkpoint <PATH_TO_MODEL_PT> \
  --num_envs 16 \
  --force_magnitude 20.0 \
  --directions 0 90 180 270 \
  --hold_s 2.0 \
  --num_trials 10
```

### 3.2 Runs to execute

Only the runs whose rows in §5.1 and §5.6 require it:
A1, A2, S6, S7, S8, S9 (and the corresponding H-series baselines if
temporal decay / TCN findings need cross-comparison).

### 3.3 Output → chapter mapping

| JSON field | Goes into |
|---|---|
| `results.summary.rise_time_90_ms` | `tab:eval_master` col 3 |
| `results.per_direction_rise_time` | §5.6 figure |

---

## 4. Test 3 — Zero-force standing

Populates: zero-force columns of `tab:eval_master`.

### 4.1 Command template

```bash
python scripts/rsl_rl/eval/zero_force_eval.py \
  --task <TASK> \
  --checkpoint <PATH_TO_MODEL_PT> \
  --num_envs 64 \
  --duration_s 10 \
  --terrain flat slope15
```

### 4.2 Runs to execute

Every run in §1 (same roster as Test 1).

### 4.3 Output → chapter mapping

| JSON field | Goes into |
|---|---|
| `results.flat.mean_norm`  | `tab:eval_master` col 5 |
| `results.slope.mean_norm` | `tab:eval_master` col 6 |

---

## 5. Test 4 — Zero-force joystick trajectory

Populates: col 7 of `tab:eval_master`, all of `tab:eval_self_motion`,
`fig:eval_joystick_boxplot`.

### 5.1 Command template

```bash
python scripts/rsl_rl/eval/joystick_zero_force_eval.py \
  --task <TASK> \
  --checkpoint <PATH_TO_MODEL_PT> \
  --num_envs 16 \
  --segments stand walk_fwd walk_back turn strafe_left strafe_right \
  --segment_duration_s 30 \
  --terrain flat rough
```

The script drives the robot with a pre-scripted command trajectory
(hard-coded per segment) with no external force applied.

### 5.2 Runs to execute

All runs in §1. This is the test §5 of the chapter is built around; it
is also the cheapest test to run so there is no reason to subset.

### 5.3 Output → chapter mapping

| JSON field | Goes into |
|---|---|
| `results.per_segment.<segment>.mean_norm` | `tab:eval_self_motion` mean columns |
| `results.per_segment.<segment>.p95_norm`  | `tab:eval_self_motion` P95 columns |
| `results.overall.p95_norm`                | `tab:eval_master` col 7 |

---

## 6. Per-subsection run checklist

A compact map from §5.x of the chapter to the runs that need to be
evaluated for that subsection.

| Chapter subsection | Runs needed | Tests required |
|---|---|---|
| §5.1 History length | A1, A2 | Test 1, Test 2 |
| §5.2 Force dimension | A2, B3 | Test 1 |
| §5.3 Network size | B2, B3 | Test 1 |
| §5.4 Reconstruction loss | A2, C2 | Test 1, Test 3 |
| §5.5 Loss weighting | B3, B4 | Test 1 |
| §5.6 Temporal decay / TCN | S6, S7, S8, S9 | Test 1, Test 2 |
| §5.7 Force-layout choice | S2, S3 | Test 1 |
| §5.8 Training force distribution | A2, H1-50N, H1-100N (+B3, H3a-50N, H3a-100N for 6D) | Test 1, Test 3, Test 4 |
| §5.9 Auxiliary rewards | H6, H7 (est-reward); A2, E1 (compliance reward) | Test 1 |
| §5.10 Frozen vs joint | H3a-50N, S6 | Test 1, Test 3, Test 4 |
| §6 Self-motion FPs | A2, B3, H3a-50N, H3a-100N, S6 | Test 4 |

---

## 7. Master aggregation

After all tests have been run, produce the master table and the
supporting CSVs with:

```bash
python scripts/rsl_rl/eval/build_master_table.py \
  --eval_root data/eval \
  --out docs/report/tables/
```

This crawls `data/eval/*/` for `raw_data.json` files, assembles the
per-run rows, and writes:

- `docs/report/tables/master.tex` → included into `tab:eval_master`
- `docs/report/tables/self_motion.tex` → `tab:eval_self_motion`
- `docs/report/tables/binned_mae.tex` → `tab:eval_training_distribution`
- `docs/report/figures/eval_master_heatmap.pdf`
- `docs/report/figures/eval_binned_mae.pdf`
- `docs/report/figures/eval_joystick_boxplot.pdf`

The figure filenames match the `\includegraphics` paths in
`evaluation.tex` so the chapter picks them up automatically.

---

## 8. Real-robot subset

Pick from the completed sim master table:

1. Two runs with the lowest simulated relative MAE.
2. Two runs with the lowest simulated joystick-FPR $P_{95}$.
3. One run that is a clear outlier in sim as a negative control.

On the robot run:

- Test 1 — static pulls at `{10, 20, 30}` N × 8 directions, by hand with
  a calibrated spring scale.
- Test 3 — zero-force standing on a physical ramp (~15°, 10 s each).
- Test 4 — zero-force joystick trajectory (same scripted segments as sim).

Populate `tab:eval_real_runs` and the corresponding subsection of
`§7 Real-world validation`.

---

## 9. TODO — scripts to create

The four eval scripts and the master aggregator do not yet exist.
Implementation order is:

1. `eval_utils.py` — shared runner dispatch, metric computation, JSON writer.
2. `static_360_eval.py` — Test 1. Largest code surface; the others reuse
   its helpers.
3. `zero_force_eval.py` — Test 3. Small; only the standing loop differs
   from Test 1.
4. `joystick_zero_force_eval.py` — Test 4. New bit is the scripted
   command trajectory; measurement code is Test 3 with segment labels.
5. `step_response_eval.py` — Test 2. Smallest; a subset of Test 1 with
   a single magnitude and impulse-style force apply.
6. `build_master_table.py` — aggregator; only runs after all evals.
