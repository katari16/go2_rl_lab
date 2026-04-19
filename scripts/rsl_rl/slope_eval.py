"""Slope evaluation for ablation runs (CompliantOnPolicyRunner).

Robot walks uphill then holds position on a pyramid slope. Tests whether
the force estimator correctly handles the gravity component on slopes
(which projects into XY body frame and looks like a persistent force).

Phases:
  WALK — forward velocity command for --walk_duration seconds
  HOLD — zero velocity command for --hold_duration seconds

Arrows:
  RED       = GT applied external force (body frame)
  BLUE      = force estimator output F_hat  (--show_est)
  YELLOW    = compliance adjustment k*F_hat  (--show_adj)

The key question: does the estimator produce a spurious XY force on the
slope (gravity artefact), and does the compliance mapping compensate for
it? On flat ground with no applied force the estimate should be ~0.
On a slope it will drift — this eval makes that visible.

Usage:
    python scripts/rsl_rl/slope_eval.py --task Go2-Ablation-A2-v0 \\
        --checkpoint logs/rsl_rl/ablation_A2_h40_3d_rec/.../model_10000.pt \\
        --slope_deg 12 --compliance_k 0.06 --show_est --show_adj --real-time
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import math
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Slope eval for CompliantOnPolicyRunner ablation runs.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--slope_deg", type=float, default=10.0,
                    help="Slope inclination in degrees (training max ~17°).")
parser.add_argument("--terrain", type=str, default="pyramid", choices=["pyramid", "valley", "flat", "rough"],
                    help="Terrain type: pyramid (uphill slope), valley (inverted pyramid), flat, or rough.")
parser.add_argument("--walk_speed", type=float, default=0.5,
                    help="Forward velocity command during walk phase (m/s).")
parser.add_argument("--walk_duration", type=float, default=8.0,
                    help="Seconds to walk uphill. With --walk_only this is the total run duration.")
parser.add_argument("--hold_duration", type=float, default=20.0,
                    help="Seconds to hold position on slope (ignored with --walk_only).")
parser.add_argument("--pause_duration", type=float, default=0.0,
                    help="Seconds to pause (zero velocity cmd) after initial walk phase.")
parser.add_argument("--resume_duration", type=float, default=0.0,
                    help="Seconds to resume walking after pause phase.")
parser.add_argument("--payload_mass", type=float, default=None,
                    help="Override payload mass (kg). Set via PhysX set_masses at startup.")
parser.add_argument("--spawn_yaw", type=float, default=0.0,
                    help="Initial robot yaw orientation in degrees.")
parser.add_argument("--compliance_k", type=float, default=0.0,
                    help="Linear compliance gain k for v*=v_cmd+k*F_hat (0=off).")
parser.add_argument("--compliance_deadzone", type=float, default=0.0,
                    help="Force magnitude threshold (N). Only apply compliance if |F_hat| > threshold.")
parser.add_argument("--force_min", type=float, default=0.0,
                    help="Optional external force minimum (N). 0=no external force.")
parser.add_argument("--force_max", type=float, default=0.0,
                    help="Optional external force maximum (N). 0=no external force.")
parser.add_argument("--no_force", action="store_true", default=False,
                    help="Explicitly disable all external forces (overrides force_min/max).")
parser.add_argument("--walk_only", action="store_true", default=False,
                    help="Skip hold phase — robot walks forward for full --walk_duration seconds.")
parser.add_argument("--show_est", action="store_true", default=False,
                    help="Show force estimate arrow (blue).")
parser.add_argument("--show_cmd", action="store_true", default=False,
                    help="Show velocity command arrow (green).")
parser.add_argument("--show_adj", action="store_true", default=False,
                    help="Show compliance adjustment arrow (yellow).")
parser.add_argument("--real-time", action="store_true", default=False)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import time

import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.terrains.height_field.hf_terrains_cfg import HfPyramidSlopedTerrainCfg, HfInvertedPyramidSlopedTerrainCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, retrieve_file_path
from isaaclab.utils.math import quat_apply, quat_conjugate, quat_from_matrix
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import go2_rl_lab.tasks  # noqa: F401


# ── Arrow helpers ─────────────────────────────────────────────────────────────

def _create_arrow_markers(prim_path: str, color: tuple) -> VisualizationMarkers:
    cfg = VisualizationMarkersCfg(
        prim_path=prim_path,
        markers={
            "arrow": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=(1.0, 0.5, 0.5),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            ),
        },
    )
    return VisualizationMarkers(cfg)


def _force_to_quat(force_world: torch.Tensor, device: torch.device) -> torch.Tensor:
    n = force_world.shape[0]
    quats = torch.zeros(n, 4, device=device)
    quats[:, 0] = 1.0
    for i in range(n):
        mag = force_world[i].norm()
        if mag < 0.1:
            continue
        x_ax = force_world[i] / mag
        up = torch.tensor(
            [0.0, 0.0, 1.0] if abs(x_ax[2].item()) < 0.9 else [1.0, 0.0, 0.0],
            device=device,
        )
        z_ax = torch.linalg.cross(x_ax, up)
        z_ax = z_ax / z_ax.norm()
        y_ax = torch.linalg.cross(z_ax, x_ax)
        rot_mat = torch.stack([x_ax, y_ax, z_ax], dim=1)
        quats[i] = quat_from_matrix(rot_mat)
    return quats


# ── Plotting ──────────────────────────────────────────────────────────────────

def generate_plots(log: dict, args, checkpoint_path: str) -> None:
    import matplotlib.pyplot as plt
    from datetime import datetime

    t = np.array(log["time_s"])
    phases = np.array(log["phase"])
    t_switch = args.walk_duration

    force_dim = log["force_dim"]
    fd3 = min(force_dim, 3)

    n_panels = 3 + (1 if force_dim >= 6 else 0)
    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 3.5 * n_panels), sharex=True)

    ext_str = f" + ext [{args.force_min:.0f}-{args.force_max:.0f}N]" if args.force_max > 0 else " (no ext force)"
    k_str = f"k={args.compliance_k:.4f}" if args.compliance_k > 0 else "k=0 (off)"
    fig.suptitle(
        f"Slope Eval — {args.slope_deg:.0f}° slope{ext_str} — {k_str}",
        fontsize=14, fontweight="bold",
    )

    def _shade_phases(ax):
        ax.axvline(t_switch, color="gray", linestyle="--", alpha=0.6, label="WALK→HOLD")
        ax.axvspan(0, t_switch, alpha=0.04, color="blue")
        ax.axvspan(t_switch, t[-1], alpha=0.04, color="orange")

    use_payload_gt = "payload_gt_fx" in log
    gt_fx = log["payload_gt_fx"] if use_payload_gt else log["gt_fx"]
    gt_fy = log["payload_gt_fy"] if use_payload_gt else log["gt_fy"]
    gt_fz = log["payload_gt_fz"] if use_payload_gt else log.get("gt_fz", [0.0] * len(t))
    gt_label = "Payload grav GT" if use_payload_gt else "GT"

    panel = 0

    # Panel 1: Fx
    ax = axes[panel]
    ax.plot(t, gt_fx,         color="tab:red", lw=1.2, label=f"{gt_label} Fx")
    ax.plot(t, log["est_fx"], color="tab:red", lw=0.9, ls="--", alpha=0.6, label="Est Fx")
    ax.axhline(0, color="black", lw=0.5, alpha=0.4)
    _shade_phases(ax)
    ax.set_ylabel("Force (N)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title("Fx: GT (solid) vs Estimator (dashed)", fontsize=11)
    panel += 1

    # Panel 2: Fy
    ax = axes[panel]
    ax.plot(t, gt_fy,         color="darkred", lw=1.2, label=f"{gt_label} Fy")
    ax.plot(t, log["est_fy"], color="darkred", lw=0.9, ls="--", alpha=0.6, label="Est Fy")
    ax.axhline(0, color="black", lw=0.5, alpha=0.4)
    _shade_phases(ax)
    ax.set_ylabel("Force (N)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title("Fy: GT (solid) vs Estimator (dashed)", fontsize=11)
    panel += 1

    # Panel 3: Fz
    ax = axes[panel]
    est_fz = log.get("est_fz", [0.0] * len(t))
    ax.plot(t, gt_fz,  color="tab:purple", lw=1.2, label=f"{gt_label} Fz")
    ax.plot(t, est_fz, color="purple",     lw=0.9, ls="--", alpha=0.6, label="Est Fz")
    ax.axhline(0, color="black", lw=0.5, alpha=0.4)
    _shade_phases(ax)
    ax.set_ylabel("Force (N)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title("Fz: GT (solid) vs Estimator (dashed)", fontsize=11)
    panel += 1

    axes[panel - 1].set_xlabel("Time (s)")

    # Panel 4: Roll/pitch torque estimates (6D only)
    if force_dim >= 6:
        ax = axes[panel]
        ax.plot(t, log["gt_troll"],  color="tab:orange", lw=1.2, label="GT τ_roll")
        ax.plot(t, log["gt_tpitch"], color="darkorange",  lw=1.2, label="GT τ_pitch")
        ax.plot(t, log["est_troll"],  color="tab:orange", lw=0.9, ls="--", alpha=0.6, label="Est τ_roll")
        ax.plot(t, log["est_tpitch"], color="darkorange",  lw=0.9, ls="--", alpha=0.6, label="Est τ_pitch")
        ax.axhline(0, color="black", lw=0.5, alpha=0.4)
        _shade_phases(ax)
        ax.set_ylabel("Torque (Nm)")
        ax.set_xlabel("Time (s)")
        ax.legend(loc="upper right", fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_title("Roll/Pitch Torque: GT (solid) vs Estimator (dashed)", fontsize=11)
        panel += 1

    plt.tight_layout()

    eval_dir = os.path.join(os.path.dirname(checkpoint_path), "slope_eval")
    os.makedirs(eval_dir, exist_ok=True)
    from datetime import datetime
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    fig_path = os.path.join(eval_dir, f"slope_{args.slope_deg:.0f}deg_k{args.compliance_k:.4f}_{ts}.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[slope_eval] Plot saved: {fig_path}")


def generate_payload_plots(log: dict, args, checkpoint_path: str) -> None:
    import matplotlib.pyplot as plt
    from datetime import datetime

    t = np.array(log["time_s"])
    t_switch = args.walk_duration
    delta_mass = args.payload_mass - 3.0

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(
        f"Payload Gravity GT vs Estimator — Δm={delta_mass:.1f}kg ({args.payload_mass:.1f}kg loaded, 3.0kg trained)"
        f" — {args.slope_deg:.0f}° slope",
        fontsize=13, fontweight="bold",
    )

    def _shade(ax):
        ax.axvline(t_switch, color="gray", linestyle="--", alpha=0.6, label="WALK→HOLD")
        ax.axvspan(0, t_switch, alpha=0.04, color="blue")
        ax.axvspan(t_switch, t[-1], alpha=0.04, color="orange")
        ax.axhline(0, color="black", lw=0.5, alpha=0.4)

    labels = [("Fx", "payload_gt_fx", "est_fx", "tab:red"),
              ("Fy", "payload_gt_fy", "est_fy", "tab:blue"),
              ("Fz", "payload_gt_fz", "est_fz", "tab:purple")]

    for ax, (name, gt_key, est_key, color) in zip(axes, labels):
        ax.plot(t, log[gt_key],  color=color, lw=1.4, label=f"GT {name} (Δm·g·proj_grav)")
        ax.plot(t, log[est_key], color=color, lw=0.9, ls="--", alpha=0.7, label=f"Est {name}")
        _shade(ax)
        ax.set_ylabel("Force (N)")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{name}: payload gravity GT vs estimator", fontsize=11)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()

    eval_dir = os.path.join(os.path.dirname(checkpoint_path), "slope_eval")
    os.makedirs(eval_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    fig_path = os.path.join(eval_dir, f"payload_gt_{args.slope_deg:.0f}deg_dm{delta_mass:.1f}kg_{ts}.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[slope_eval] Payload plot saved: {fig_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

@hydra_task_config(args_cli.task, args_cli.agent)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # ── Spawn orientation ─────────────────────────────────────────────────
    if args_cli.spawn_yaw != 0.0:
        import math
        yaw_rad = math.radians(args_cli.spawn_yaw)
        cos_half = math.cos(yaw_rad / 2.0)
        sin_half = math.sin(yaw_rad / 2.0)
        env_cfg.scene.robot.init_state.rot = (cos_half, 0.0, 0.0, sin_half)

    # ── Terrain configuration ─────────────────────────────────────────────
    if args_cli.terrain == "flat":
        env_cfg.scene.terrain.terrain_type = "plane"
    elif args_cli.terrain == "rough":
        env_cfg.scene.terrain.terrain_type = "generator"
        # Use existing training terrain config (rough + obstacles)
    else:
        # pyramid or valley slopes
        slope_rad = math.radians(args_cli.slope_deg)
        terrain_cfg = HfInvertedPyramidSlopedTerrainCfg if args_cli.terrain == "valley" else HfPyramidSlopedTerrainCfg
        env_cfg.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=TerrainGeneratorCfg(
                size=(12.0, 12.0),
                border_width=0.0,
                num_rows=1,
                num_cols=1,
                horizontal_scale=0.1,
                vertical_scale=0.005,
                slope_threshold=0.75,
                use_cache=False,
                sub_terrains={
                    "slope": terrain_cfg(
                        proportion=1.0,
                        slope_range=(slope_rad, slope_rad),
                        platform_width=1.5,
                        border_width=0.25,
                    ),
                },
            ),
            max_init_terrain_level=0,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
        )

    if hasattr(env_cfg, "curriculum"):
        env_cfg.curriculum = None

    # ── Velocity commands ─────────────────────────────────────────────────
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (args_cli.walk_speed, args_cli.walk_speed)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.heading = (0.0, 0.0)
    env_cfg.commands.base_velocity.heading_command = False
    env_cfg.commands.base_velocity.resampling_time_range = (1000.0, 1000.0)
    env_cfg.commands.base_velocity.debug_vis = False

    # ── External forces (optional) ────────────────────────────────────────
    force_event_name = agent_cfg.to_dict().get("force_event_term_name", "persistent_xyz_force")
    force_event = getattr(env_cfg.events, force_event_name, None)
    apply_forces = args_cli.force_max > 0 and not args_cli.no_force
    if force_event is not None:
        if apply_forces:
            force_event.params["force_range"] = (args_cli.force_min, args_cli.force_max)
            force_event.interval_range_s = (3.0, 5.0)
        else:
            force_event.params["force_range"] = (0.0, 0.0)
            if "torque_range" in force_event.params:
                force_event.params["torque_range"] = (0.0, 0.0)

    # Calculate total duration based on phases
    if args_cli.walk_only:
        total_duration = args_cli.walk_duration
    else:
        total_duration = args_cli.walk_duration + args_cli.hold_duration
        if args_cli.pause_duration > 0:
            total_duration += args_cli.pause_duration
        if args_cli.resume_duration > 0:
            total_duration += args_cli.resume_duration
    env_cfg.episode_length_s = total_duration + 5.0

    # ── Checkpoint ────────────────────────────────────────────────────────
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume_path = (
        retrieve_file_path(args_cli.checkpoint)
        if args_cli.checkpoint
        else get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    )
    print(f"[slope_eval] Checkpoint: {resume_path}")

    # ── Runner ────────────────────────────────────────────────────────────
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner_class_name = agent_cfg.class_name
    if runner_class_name != "CompliantOnPolicyRunner":
        raise ValueError(
            f"slope_eval.py only supports CompliantOnPolicyRunner, got: {runner_class_name}. "
            f"For FrozenPolicyEstimatorRunner use slope_eval_stage2.py."
        )

    from go2_rl_lab.estimator.compliant_on_policy_runner import CompliantOnPolicyRunner
    train_cfg = agent_cfg.to_dict()
    if args_cli.checkpoint:
        train_cfg["estimator_checkpoint"] = args_cli.checkpoint
    runner = CompliantOnPolicyRunner(env, train_cfg, log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    runner.eval_mode()

    # Re-apply force override after load() — checkpoint restores force_active
    # which overwrites force_range. Must re-zero if no external forces wanted.
    if force_event is not None and not apply_forces:
        isaac_env_tmp = env.unwrapped
        try:
            ev_cfg = isaac_env_tmp.event_manager.get_term_cfg(force_event_name)
            ev_cfg.params["force_range"] = (0.0, 0.0)
            if "torque_range" in ev_cfg.params:
                ev_cfg.params["torque_range"] = (0.0, 0.0)
        except Exception:
            pass

    force_dim = getattr(runner, "_force_dim", 2)
    fd3 = min(force_dim, 3)
    raw_obs_dim = runner._num_one_step_obs
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    # ── Scene access ──────────────────────────────────────────────────────
    isaac_env = env.unwrapped
    device = isaac_env.device
    asset = isaac_env.scene["robot"]
    base_body_ids, _ = asset.find_bodies("base")
    base_idx = base_body_ids[0] if isinstance(base_body_ids, (list, tuple)) else int(base_body_ids)
    n = args_cli.num_envs

    # ── Payload mass override ─────────────────────────────────────────────
    if args_cli.payload_mass is not None:
        payload_body_ids, _ = asset.find_bodies("payload_link")
        if len(payload_body_ids) > 0:
            payload_idx = payload_body_ids[0] if isinstance(payload_body_ids, (list, tuple)) else int(payload_body_ids)
            masses = asset.root_physx_view.get_masses()
            print(f"[slope_eval] Overriding payload_link mass: {masses[0, payload_idx]:.3f} → {args_cli.payload_mass:.3f} kg")
            masses[:, payload_idx] = args_cli.payload_mass
            asset.root_physx_view.set_masses(masses, torch.arange(n, device="cpu"))
        else:
            print(f"[slope_eval] WARNING: --payload_mass specified but payload_link body not found")

    has_payload_gt = args_cli.payload_mass is not None and args_cli.payload_mass > 3.0

    # ── Arrows ────────────────────────────────────────────────────────────
    # GT force arrow only makes sense when external forces are applied
    gt_markers         = _create_arrow_markers("/World/Visuals/GTForce",      (1.0, 0.0, 0.0)) if apply_forces else None
    est_markers        = _create_arrow_markers("/World/Visuals/EstForce",     (0.2, 0.4, 1.0))  if (args_cli.show_est or has_payload_gt) else None
    payload_gt_markers = _create_arrow_markers("/World/Visuals/PayloadGTForce", (1.0, 0.5, 0.0)) if has_payload_gt else None
    cmd_markers = _create_arrow_markers("/World/Visuals/CmdVel",   (0.0, 0.8, 0.2))  if args_cli.show_cmd else None
    adj_markers = _create_arrow_markers("/World/Visuals/AdjVel",   (1.0, 0.85, 0.0)) if args_cli.show_adj else None

    # ── Loop setup ────────────────────────────────────────────────────────
    dt = isaac_env.step_dt
    walk_steps  = int(args_cli.walk_duration / dt)
    pause_steps = int((args_cli.walk_duration + args_cli.pause_duration) / dt) if args_cli.pause_duration > 0 else 0
    resume_steps = int((args_cli.walk_duration + args_cli.pause_duration + args_cli.resume_duration) / dt) if args_cli.resume_duration > 0 else 0
    hold_steps = int((args_cli.walk_duration + (args_cli.pause_duration if args_cli.pause_duration > 0 else 0) + (args_cli.resume_duration if args_cli.resume_duration > 0 else 0)) / dt)
    total_steps = int(total_duration / dt)

    isaac_env._force_estimate_xy = torch.zeros(n, force_dim, device=device)
    obs = env.get_observations()
    step_count = 0
    phase = "WALK"

    log: dict = {
        "time_s": [], "phase": [],
        "gt_fx": [], "gt_fy": [],
        "est_fx": [], "est_fy": [],
        "base_vx": [], "base_vy": [],
        "cmd_vx": [], "cmd_vy": [],
        "adj_vx": [], "adj_vy": [],
        "force_dim": force_dim,
    }
    if force_dim >= 3:
        log["gt_fz"] = []
        log["est_fz"] = []
    if has_payload_gt:
        log["payload_gt_fx"] = []
        log["payload_gt_fy"] = []
        log["payload_gt_fz"] = []
    if force_dim >= 6:
        log["gt_troll"] = []
        log["gt_tpitch"] = []
        log["est_troll"] = []
        log["est_tpitch"] = []

    print(f"\n{'=' * 70}")
    print(f"  Task        : {args_cli.task}")
    print(f"  Runner      : CompliantOnPolicyRunner")
    print(f"  Slope       : {args_cli.slope_deg:.1f}°  (training max ~17°)")
    print(f"  Force dim   : {force_dim}D")
    print(f"  WALK phase  : {args_cli.walk_speed:.1f} m/s for {args_cli.walk_duration:.0f}s")
    if args_cli.walk_only:
        print(f"  HOLD phase  : SKIPPED (--walk_only)")
    else:
        print(f"  HOLD phase  : zero cmd for {args_cli.hold_duration:.0f}s")
    if apply_forces:
        print(f"  Ext forces  : [{args_cli.force_min:.0f}, {args_cli.force_max:.0f}] N")
    else:
        print(f"  Ext forces  : NONE — pure slope gravity test")
    print(f"  Compliance  : k={args_cli.compliance_k:.4f}" if args_cli.compliance_k > 0 else "  Compliance  : OFF")
    arrows = (["RED=GT"] if apply_forces else []) + (["BLUE=est"] if args_cli.show_est else []) + (["GREEN=cmd"] if args_cli.show_cmd else []) + (["YELLOW=adj"] if args_cli.show_adj else [])
    print(f"  Arrows      : {' '.join(arrows) if arrows else 'none'}")
    print(f"{'=' * 70}\n")

    try:
        while simulation_app.is_running():
            start_time = time.time()

            with torch.inference_mode():
                # Phase transitions
                if step_count == walk_steps and phase == "WALK":
                    if args_cli.pause_duration > 0:
                        phase = "PAUSE"
                        print(f"\n\n  >>> PAUSE phase — zero velocity command <<<\n")
                    elif not args_cli.walk_only:
                        phase = "HOLD"
                        print(f"\n\n  >>> HOLD phase — zero velocity command <<<\n")

                if pause_steps > 0 and step_count == pause_steps and phase == "PAUSE":
                    if args_cli.resume_duration > 0:
                        phase = "RESUME"
                        print(f"\n\n  >>> RESUME phase — forward velocity command <<<\n")
                    else:
                        phase = "HOLD"
                        print(f"\n\n  >>> HOLD phase — zero velocity command <<<\n")

                if resume_steps > 0 and step_count == resume_steps and phase == "RESUME":
                    phase = "HOLD"
                    print(f"\n\n  >>> HOLD phase — zero velocity command <<<\n")

                # Apply velocity commands based on phase
                if phase in ["PAUSE", "HOLD"]:
                    cmd = isaac_env.command_manager.get_command("base_velocity")
                    cmd[:] = 0.0
                elif phase == "RESUME":
                    cmd = isaac_env.command_manager.get_command("base_velocity")
                    cmd[:, 0] = args_cli.walk_speed  # forward velocity
                    cmd[:, 1] = 0.0  # lateral velocity
                    cmd[:, 2] = 0.0  # angular velocity

                # Update history + estimator
                raw_obs = obs["policy"][:, :raw_obs_dim]
                runner._history_buffer.insert(raw_obs)
                force_hat, _ = runner.estimator.get_latent(runner._history_buffer.get_flattened())
                force_hat = force_hat[:n]
                isaac_env._force_estimate_xy = force_hat

                # Inject compliance into obs velocity command slots
                if args_cli.compliance_k > 0.0:
                    # Apply deadzone: only comply if force magnitude exceeds threshold
                    force_xy_mag = torch.sqrt(force_hat[:, 0]**2 + force_hat[:, 1]**2)
                    apply_compliance = force_xy_mag > args_cli.compliance_deadzone
                    obs["policy"][:, 6] = obs["policy"][:, 6] + args_cli.compliance_k * force_hat[:, 0] * apply_compliance
                    obs["policy"][:, 7] = obs["policy"][:, 7] + args_cli.compliance_k * force_hat[:, 1] * apply_compliance

                # Policy step
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
                policy_nn.reset(dones)

                done_ids = (dones > 0).nonzero(as_tuple=False).squeeze(-1)
                if len(done_ids) > 0:
                    runner._history_buffer.reset(done_ids)

                # GT force (body frame)
                gt_force_body = asset.permanent_wrench_composer.composed_force_as_torch[:n, base_idx, :fd3]
                if gt_force_body.dim() == 3:
                    gt_force_body = gt_force_body.squeeze(1)

                # GT torque (body frame) — only used if force_dim >= 6
                if force_dim >= 6:
                    gt_torque_body = asset.permanent_wrench_composer.composed_torque_as_torch[:n, base_idx, :]
                    if gt_torque_body.dim() == 3:
                        gt_torque_body = gt_torque_body.squeeze(1)

                # Compliance-adjusted velocity command
                cmd_vel = isaac_env.command_manager.get_command("base_velocity")[:n]
                adj_vx = cmd_vel[:, 0] + args_cli.compliance_k * force_hat[:, 0]
                adj_vy = cmd_vel[:, 1] + args_cli.compliance_k * force_hat[:, 1]

                # Arrow rendering
                base_pos  = asset.data.root_pos_w[:n]
                base_quat = asset.data.root_quat_w[:n]

                # Payload gravity GT: (payload_mass - 3kg) * g * projected_gravity in base frame
                if has_payload_gt:
                    delta_mass = args_cli.payload_mass - 3.0
                    grav_world = torch.zeros(n, 3, device=device)
                    grav_world[:, 2] = -9.81 * delta_mass
                    payload_grav_base = quat_apply(quat_conjugate(base_quat), grav_world)
                force_scale = 0.05
                vel_scale   = 1.0

                gt_3d = torch.zeros(n, 3, device=device)
                gt_3d[:, :fd3] = gt_force_body
                gt_world = quat_apply(base_quat, gt_3d)
                if gt_markers is not None:
                    gt_pos = base_pos.clone(); gt_pos[:, 2] += 0.55
                    gt_scales = torch.full((n, 3), 0.3, device=device)
                    gt_scales[:, 0:1] = (gt_world.norm(dim=-1, keepdim=True) * force_scale).clamp(min=0.05)
                    gt_markers.visualize(gt_pos, _force_to_quat(gt_world, device), gt_scales)

                if est_markers is not None:
                    est_3d = torch.zeros(n, 3, device=device)
                    est_3d[:, :fd3] = force_hat[:, :fd3]
                    est_pos = base_pos.clone(); est_pos[:, 2] += 0.42
                    est_scales = torch.full((n, 3), 0.3, device=device)
                    est_scales[:, 0:1] = (est_3d.norm(dim=-1, keepdim=True) * force_scale).clamp(min=0.05)
                    est_markers.visualize(est_pos, _force_to_quat(est_3d, device), est_scales)

                if payload_gt_markers is not None:
                    pg_xy = payload_grav_base.clone(); pg_xy[:, 2] = 0.0
                    pg_pos = base_pos.clone(); pg_pos[:, 2] += 0.68
                    pg_scales = torch.full((n, 3), 0.3, device=device)
                    pg_scales[:, 0:1] = (pg_xy.norm(dim=-1, keepdim=True) * force_scale).clamp(min=0.05)
                    payload_gt_markers.visualize(pg_pos, _force_to_quat(pg_xy, device), pg_scales)

                if cmd_markers is not None:
                    cmd_3d = torch.zeros(n, 3, device=device)
                    cmd_3d[:, 0] = cmd_vel[:, 0]
                    cmd_3d[:, 1] = cmd_vel[:, 1]
                    cmd_world = quat_apply(base_quat, cmd_3d)
                    cmd_pos = base_pos.clone(); cmd_pos[:, 2] += 0.29
                    cmd_scales = torch.full((n, 3), 0.3, device=device)
                    cmd_scales[:, 0:1] = (cmd_world.norm(dim=-1, keepdim=True) * vel_scale).clamp(min=0.05)
                    cmd_markers.visualize(cmd_pos, _force_to_quat(cmd_world, device), cmd_scales)

                if adj_markers is not None:
                    adj_3d = torch.zeros(n, 3, device=device)
                    adj_3d[:, 0] = adj_vx
                    adj_3d[:, 1] = adj_vy
                    adj_world = quat_apply(base_quat, adj_3d)
                    adj_pos = base_pos.clone(); adj_pos[:, 2] += 0.16
                    adj_scales = torch.full((n, 3), 0.3, device=device)
                    adj_scales[:, 0:1] = (adj_world.norm(dim=-1, keepdim=True) * vel_scale).clamp(min=0.05)
                    adj_markers.visualize(adj_pos, _force_to_quat(adj_world, device), adj_scales)

                # Log
                g0 = gt_force_body[0]
                e0 = force_hat[0]
                base_vel = asset.data.root_lin_vel_b[:n]
                log["time_s"].append(step_count * dt)
                log["phase"].append(phase)
                log["gt_fx"].append(g0[0].item())
                log["gt_fy"].append(g0[1].item())
                log["est_fx"].append(e0[0].item())
                log["est_fy"].append(e0[1].item())
                log["base_vx"].append(base_vel[0, 0].item())
                log["base_vy"].append(base_vel[0, 1].item())
                log["cmd_vx"].append(cmd_vel[0, 0].item())
                log["cmd_vy"].append(cmd_vel[0, 1].item())
                log["adj_vx"].append(adj_vx[0].item())
                log["adj_vy"].append(adj_vy[0].item())
                if force_dim >= 3:
                    log["gt_fz"].append(g0[2].item() if g0.shape[0] > 2 else 0.0)
                    log["est_fz"].append(e0[2].item() if e0.shape[0] > 2 else 0.0)
                if has_payload_gt:
                    pg = payload_grav_base[0]
                    log["payload_gt_fx"].append(pg[0].item())
                    log["payload_gt_fy"].append(pg[1].item())
                    log["payload_gt_fz"].append(pg[2].item())
                if force_dim >= 6:
                    t0 = gt_torque_body[0]
                    log["gt_troll"].append(t0[0].item())
                    log["gt_tpitch"].append(t0[1].item())
                    log["est_troll"].append(e0[3].item() if e0.shape[0] > 3 else 0.0)
                    log["est_tpitch"].append(e0[4].item() if e0.shape[0] > 4 else 0.0)

                # Terminal readout
                step_count += 1
                if step_count % 10 == 0:
                    pct = step_count / total_steps
                    bar = "\u2588" * int(20 * pct) + "\u2591" * (20 - int(20 * pct))

                    # Build GT and Est strings based on force_dim
                    if force_dim == 2:
                        gt_str = f"GT:[{g0[0]:+5.1f},{g0[1]:+5.1f}]N"
                        est_str = f"Est:[{e0[0]:+5.1f},{e0[1]:+5.1f}]N"
                    elif force_dim == 3:
                        gt_str = f"GT:[{g0[0]:+5.1f},{g0[1]:+5.1f},{g0[2]:+5.1f}]N"
                        est_str = f"Est:[{e0[0]:+5.1f},{e0[1]:+5.1f},{e0[2]:+5.1f}]N"
                    elif force_dim == 4:
                        gt_str = f"GT:[{g0[0]:+5.1f},{g0[1]:+5.1f},{g0[2]:+5.1f}]N τ:[{g0[3]:+5.1f}]Nm"
                        est_str = f"Est:[{e0[0]:+5.1f},{e0[1]:+5.1f},{e0[2]:+5.1f}]N τ:[{e0[3]:+5.1f}]Nm"
                    else:  # 6D
                        gt_str = f"GT:[{g0[0]:+4.1f},{g0[1]:+4.1f},{g0[2]:+4.1f}]N τ:[{g0[3]:+4.1f},{g0[4]:+4.1f},{g0[5]:+4.1f}]Nm"
                        est_str = f"Est:[{e0[0]:+4.1f},{e0[1]:+4.1f},{e0[2]:+4.1f}]N τ:[{e0[3]:+4.1f},{e0[4]:+4.1f},{e0[5]:+4.1f}]Nm"

                    print(
                        f"\r  [{bar}] {step_count * dt:.0f}/{total_duration:.0f}s [{phase:4s}]  "
                        f"{gt_str}  {est_str}  adj:[{adj_vx[0]:+5.2f},{adj_vy[0]:+5.2f}]m/s",
                        end="", flush=True,
                    )

                if step_count >= total_steps:
                    break

            if args_cli.real_time:
                sleep_time = dt - (time.time() - start_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)

    except (KeyboardInterrupt, SystemExit, Exception) as exc:
        print(f"\n\n[slope_eval] Stopped ({type(exc).__name__}).")

    print()
    env.close()

    from time import sleep
    sleep(1)

    if len(log["time_s"]) > 10:
        generate_plots(log, args_cli, resume_path)
        if has_payload_gt:
            generate_payload_plots(log, args_cli, resume_path)
    else:
        print("[slope_eval] Too few steps, skipping plots.")


if __name__ == "__main__":
    main()
    simulation_app.close()
