"""Custom event functions for locomotion tasks."""
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import RED_ARROW_X_MARKER_CFG
from isaaclab.utils import math as math_utils
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def apply_persistent_xy_force(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    force_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="base"),
) -> None:
    """Apply persistent external XY force to the robot base (Z component zeroed).

    Intended for use with ``mode="interval"`` and ``interval_range_s=(3.0, 5.0)``
    so forces are re-randomized every ~4 seconds.  Uses the permanent wrench
    composer API (same as ``apply_external_force_torque``).

    The force magnitude is sampled uniformly in ``[min, max]`` with random sign
    independently for X and Y, with Z = 0.

    Args:
        env: The environment instance.
        env_ids: Environment indices to randomize.
        force_range: (min_abs, max_abs) magnitude range for each XY axis.
        asset_cfg: Asset and body to apply force to.
    """
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    num = len(env_ids)
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # Resolve number of bodies
    num_bodies = len(asset_cfg.body_ids) if isinstance(asset_cfg.body_ids, list) else asset.num_bodies

    lo, hi = float(force_range[0]), float(force_range[1])
    if hi < 1e-6:
        # No force — set zeros
        forces = torch.zeros(num, num_bodies, 3, device=asset.device)
        torques = torch.zeros(num, num_bodies, 3, device=asset.device)
    else:
        # Sample magnitude with random sign for X and Y, Z=0
        mag = torch.empty(num, 2, device=asset.device).uniform_(lo, hi)
        sign = torch.sign(torch.empty(num, 2, device=asset.device).uniform_(-1, 1))
        sign[sign == 0] = 1.0
        xy_force = mag * sign

        forces = torch.zeros(num, num_bodies, 3, device=asset.device)
        forces[:, :, 0] = xy_force[:, 0:1]
        forces[:, :, 1] = xy_force[:, 1:2]
        torques = torch.zeros(num, num_bodies, 3, device=asset.device)

    asset.permanent_wrench_composer.set_forces_and_torques(
        forces=forces,
        torques=torques,
        body_ids=asset_cfg.body_ids,
        env_ids=env_ids,
    )


def apply_persistent_xyz_force(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    force_range: tuple[float, float],
    fz_scale: float = 0.6,
    force_free_fraction: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="base"),
) -> None:
    """Apply persistent external XYZ force to the robot base (body frame).

    Same as ``apply_persistent_xy_force`` but also samples a Z component.
    The Z magnitude range is scaled by ``fz_scale`` relative to XY to avoid
    instability from large vertical forces.

    Args:
        env: The environment instance.
        env_ids: Environment indices to randomize.
        force_range: (min_abs, max_abs) magnitude range for each XY axis.
        fz_scale: Scale factor for Z magnitude range relative to XY (default: 0.6).
        force_free_fraction: Fraction of envs that get zero force (default: 0.0).
        asset_cfg: Asset and body to apply force to.
    """
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    num = len(env_ids)
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    num_bodies = len(asset_cfg.body_ids) if isinstance(asset_cfg.body_ids, list) else asset.num_bodies

    lo, hi = float(force_range[0]), float(force_range[1])
    if hi < 1e-6:
        forces = torch.zeros(num, num_bodies, 3, device=asset.device)
        torques = torch.zeros(num, num_bodies, 3, device=asset.device)
    else:
        # Sample XY magnitude with random sign
        mag_xy = torch.empty(num, 2, device=asset.device).uniform_(lo, hi)
        sign_xy = torch.sign(torch.empty(num, 2, device=asset.device).uniform_(-1, 1))
        sign_xy[sign_xy == 0] = 1.0
        xy_force = mag_xy * sign_xy

        # Sample Z magnitude with scaled range and random sign
        lo_z, hi_z = lo * fz_scale, hi * fz_scale
        mag_z = torch.empty(num, 1, device=asset.device).uniform_(lo_z, hi_z)
        sign_z = torch.sign(torch.empty(num, 1, device=asset.device).uniform_(-1, 1))
        sign_z[sign_z == 0] = 1.0
        z_force = mag_z * sign_z

        forces = torch.zeros(num, num_bodies, 3, device=asset.device)
        forces[:, :, 0] = xy_force[:, 0:1]
        forces[:, :, 1] = xy_force[:, 1:2]
        forces[:, :, 2] = z_force[:, 0:1]
        torques = torch.zeros(num, num_bodies, 3, device=asset.device)

        # Zero out forces for a fraction of envs
        if force_free_fraction > 0.0:
            n_free = max(1, int(num * force_free_fraction))
            free_idx = torch.randperm(num, device=asset.device)[:n_free]
            forces[free_idx] = 0.0

    asset.permanent_wrench_composer.set_forces_and_torques(
        forces=forces,
        torques=torques,
        body_ids=asset_cfg.body_ids,
        env_ids=env_ids,
    )


def apply_persistent_wrench(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    force_range: tuple[float, float],
    fz_scale: float = 0.6,
    torque_range: tuple[float, float] = (0.0, 5.0),
    force_free_fraction: float = 0.0,
    bucket_fracs: tuple[tuple[float, float], ...] | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="base"),
) -> None:
    """Apply persistent external wrench (force + torque) to the robot base.

    Extends ``apply_persistent_xyz_force`` to also sample roll/pitch/yaw torques.
    When ``bucket_fracs`` is provided, envs are divided equally among buckets by
    index (same scheme as ``apply_trapezoid_wrench``).  Each bucket defines a
    (lo_frac, hi_frac) range of ``force_range[1]`` so the training distribution
    is stratified across the full force range.

    Args:
        env: The environment instance.
        env_ids: Environment indices to randomize.
        force_range: (min_abs, max_abs) magnitude range for each XY force axis.
        fz_scale: Scale factor for Z force magnitude relative to XY (default: 0.6).
        torque_range: (min_abs, max_abs) magnitude range for each torque axis (default: 0-5 Nm).
        force_free_fraction: Per-interval probability that an env gets zero wrench (default: 0.0).
        bucket_fracs: Per-bucket (lo_frac, hi_frac) of max force. Envs are
            divided equally among buckets by index. None = uniform sampling.
        asset_cfg: Asset and body to apply wrench to.
    """
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    num = len(env_ids)
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    num_bodies = len(asset_cfg.body_ids) if isinstance(asset_cfg.body_ids, list) else asset.num_bodies

    f_hi = float(force_range[1])
    t_lo, t_hi = float(torque_range[0]), float(torque_range[1])

    if f_hi < 1e-6 and t_hi < 1e-6:
        forces = torch.zeros(num, num_bodies, 3, device=asset.device)
        torques = torch.zeros(num, num_bodies, 3, device=asset.device)
    else:
        # Bucket-stratified per-env force range
        if bucket_fracs is not None:
            if not hasattr(env, "_pw_buckets"):
                N = env.scene.num_envs
                nb = len(bucket_fracs)
                bsz = N // nb
                b_lo = torch.zeros(N, device=asset.device)
                b_hi = torch.zeros(N, device=asset.device)
                for b, (lf, hf) in enumerate(bucket_fracs):
                    start = b * bsz
                    end = start + bsz if b < nb - 1 else N
                    b_lo[start:end] = lf
                    b_hi[start:end] = hf
                env._pw_buckets = {"lo": b_lo, "hi": b_hi}

            f_lo_per = env._pw_buckets["lo"][env_ids] * f_hi  # (num,)
            f_hi_per = env._pw_buckets["hi"][env_ids] * f_hi
        else:
            f_lo = float(force_range[0])
            f_lo_per = torch.full((num,), f_lo, device=asset.device)
            f_hi_per = torch.full((num,), f_hi, device=asset.device)

        # Sample XY force per-env within bucket range
        u_xy = torch.rand(num, 2, device=asset.device)
        range_xy = (f_hi_per - f_lo_per).unsqueeze(-1)
        mag_xy = u_xy * range_xy + f_lo_per.unsqueeze(-1)
        sign_xy = torch.sign(torch.empty(num, 2, device=asset.device).uniform_(-1, 1))
        sign_xy[sign_xy == 0] = 1.0
        xy_force = mag_xy * sign_xy

        # Sample Z force with scaled range
        fz_lo_per = f_lo_per * fz_scale
        fz_hi_per = f_hi_per * fz_scale
        u_z = torch.rand(num, 1, device=asset.device)
        range_z = (fz_hi_per - fz_lo_per).unsqueeze(-1)
        mag_z = u_z * range_z + fz_lo_per.unsqueeze(-1)
        sign_z = torch.sign(torch.empty(num, 1, device=asset.device).uniform_(-1, 1))
        sign_z[sign_z == 0] = 1.0
        z_force = mag_z * sign_z

        forces = torch.zeros(num, num_bodies, 3, device=asset.device)
        forces[:, :, 0] = xy_force[:, 0:1]
        forces[:, :, 1] = xy_force[:, 1:2]
        forces[:, :, 2] = z_force[:, 0:1]

        # Sample roll/pitch/yaw torques with random sign
        mag_t = torch.empty(num, 3, device=asset.device).uniform_(t_lo, max(t_lo, t_hi))
        sign_t = torch.sign(torch.empty(num, 3, device=asset.device).uniform_(-1, 1))
        sign_t[sign_t == 0] = 1.0
        torques = torch.zeros(num, num_bodies, 3, device=asset.device)
        torque_vals = mag_t * sign_t
        torques[:, :, 0] = torque_vals[:, 0:1]
        torques[:, :, 1] = torque_vals[:, 1:2]
        torques[:, :, 2] = torque_vals[:, 2:3]

        # Per-interval zero-wrench probability
        if force_free_fraction > 0.0:
            zero_mask = torch.rand(num, device=asset.device) < force_free_fraction
            forces[zero_mask] = 0.0
            torques[zero_mask] = 0.0

    asset.permanent_wrench_composer.set_forces_and_torques(
        forces=forces,
        torques=torques,
        body_ids=asset_cfg.body_ids,
        env_ids=env_ids,
    )


# ── Trapezoid force profile (PAINT-style) ────────────────────────────────────
# Phase constants for the piecewise-linear envelope
_TRAP_RAMP_UP = 0
_TRAP_HOLD = 1
_TRAP_RAMP_DOWN = 2
_TRAP_ZERO = 3


def apply_trapezoid_wrench(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    force_range: tuple[float, float],
    fz_scale: float = 0.6,
    torque_range: tuple[float, float] = (0.0, 0.0),
    ramp_s_range: tuple[float, float] = (0.2, 0.8),
    hold_s_range: tuple[float, float] = (2.0, 5.0),
    zero_s_range: tuple[float, float] = (0.5, 2.0),
    zero_prob: float = 0.02,
    bucket_fracs: tuple[tuple[float, float], ...] = (
        (0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0),
    ),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="base"),
) -> None:
    """Apply PAINT-style trapezoid wrench with stratified magnitude buckets.

    Must be used with ``interval_range_s`` matching the control dt (e.g. 0.02)
    so it fires every step. The function manages its own cycle internally:
    each env goes through ramp_up → hold → ramp_down → zero → ramp_up → ...

    On each new cycle, a target force+torque is sampled within the env's
    magnitude bucket. The applied wrench is ``target * s(t)`` where s(t) is
    a piecewise-linear envelope (Eq. 12 from PAINT).

    Args:
        env: The environment instance.
        env_ids: Environment indices (all envs every step).
        force_range: (min, max) — max is set by the curriculum. Bucket ranges
            are fractions of force_range[1].
        fz_scale: Scale for Z force relative to XY.
        torque_range: (min, max) for each torque axis.
        ramp_s_range: Duration range for ramp up/down phases (seconds).
        hold_s_range: Duration range for hold phase (seconds).
        zero_s_range: Duration range for zero-force gap (seconds).
        zero_prob: Probability of sampling a full-zero cycle.
        bucket_fracs: Per-bucket (lo_frac, hi_frac) of max force. Envs are
            divided equally among buckets by index.
        asset_cfg: Asset and body to apply wrench to.
    """
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    device = asset.device
    N = env.scene.num_envs
    dt = env.step_dt
    num_bodies = len(asset_cfg.body_ids) if isinstance(asset_cfg.body_ids, list) else asset.num_bodies
    f_max = float(force_range[1])
    t_max = float(torque_range[1])

    # ── First-call init: allocate per-env state ──────────────────────────
    if not hasattr(env, "_trap"):
        num_buckets = len(bucket_fracs)
        bucket_size = N // num_buckets
        bucket_lo = torch.zeros(N, device=device)
        bucket_hi = torch.zeros(N, device=device)
        for b, (lo_f, hi_f) in enumerate(bucket_fracs):
            start = b * bucket_size
            end = start + bucket_size if b < num_buckets - 1 else N
            bucket_lo[start:end] = lo_f
            bucket_hi[start:end] = hi_f

        env._trap = {
            "target_f": torch.zeros(N, num_bodies, 3, device=device),
            "target_t": torch.zeros(N, num_bodies, 3, device=device),
            "phase": torch.full((N,), _TRAP_ZERO, dtype=torch.long, device=device),
            "timer": torch.zeros(N, device=device),
            "duration": torch.ones(N, device=device) * 0.02,
            "bucket_lo": bucket_lo,
            "bucket_hi": bucket_hi,
        }

    s = env._trap

    # ── Tick timers for fired envs ───────────────────────────────────────
    s["timer"][env_ids] -= dt

    # ── Transition envs whose phase expired ──────────────────────────────
    expired_mask = s["timer"][env_ids] <= 0
    if expired_mask.any():
        exp_ids = env_ids[expired_mask]
        _trap_transition(s, exp_ids, f_max, fz_scale, t_max,
                         ramp_s_range, hold_s_range, zero_s_range,
                         zero_prob, num_bodies, device)

    # ── Compute envelope s(t) ∈ [0, 1] ──────────────────────────────────
    alpha = torch.zeros(N, device=device)
    dur = s["duration"].clamp(min=1e-6)

    ramp_up = s["phase"] == _TRAP_RAMP_UP
    alpha[ramp_up] = 1.0 - s["timer"][ramp_up] / dur[ramp_up]

    alpha[s["phase"] == _TRAP_HOLD] = 1.0

    ramp_dn = s["phase"] == _TRAP_RAMP_DOWN
    alpha[ramp_dn] = s["timer"][ramp_dn] / dur[ramp_dn]

    # ZERO phase: alpha stays 0

    alpha = alpha.clamp(0.0, 1.0)

    # ── Apply scaled forces to fired envs ────────────────────────────────
    a = alpha[env_ids].unsqueeze(-1).unsqueeze(-1)  # [len(env_ids), 1, 1]
    forces = s["target_f"][env_ids] * a
    torques = s["target_t"][env_ids] * a

    asset.permanent_wrench_composer.set_forces_and_torques(
        forces=forces,
        torques=torques,
        body_ids=asset_cfg.body_ids,
        env_ids=env_ids,
    )


def apply_paint_trapezoid_wrench(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    force_range: tuple[float, float],
    fz_scale: float = 0.8,
    torque_range: tuple[float, float] = (0.0, 0.0),
    zero_prob: float = 0.02,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="base"),
) -> None:
    """PAINT-style single-episode trapezoid wrench.

    One ramp-up/hold/ramp-down envelope per episode (Eq. 12 from PAINT paper).
    Force is sampled once per episode at reset using Cartesian per-axis sampling
    U(-max, max) independently for each axis — no polar/magnitude bucketing.

    Envelope: ramp up 10% of episode, hold 80%, ramp down 10%.
    zero_prob: probability of a zero-wrench episode (default 0.02, same as PAINT).

    Must be used with interval_range_s=(0.02, 0.02) so it fires every step.
    """
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    device = asset.device
    N = env.scene.num_envs
    dt = env.step_dt
    T = env.max_episode_length * dt

    num_bodies = len(asset_cfg.body_ids) if isinstance(asset_cfg.body_ids, list) else asset.num_bodies
    f_max = float(force_range[1])
    t_max = float(torque_range[1])

    if not hasattr(env, "_paint_trap"):
        env._paint_trap = {
            "target_f": torch.zeros(N, num_bodies, 3, device=device),
            "target_t": torch.zeros(N, num_bodies, 3, device=device),
        }

    s = env._paint_trap

    # Resample at episode start (episode_length_buf == 0 immediately after reset)
    just_reset = env_ids[env.episode_length_buf[env_ids] == 0]
    if len(just_reset) > 0 and f_max > 1e-6:
        nr = len(just_reset)
        is_zero = torch.rand(nr, device=device) < zero_prob

        fx = (torch.rand(nr, device=device) * 2 - 1) * f_max
        fy = (torch.rand(nr, device=device) * 2 - 1) * f_max
        fz = (torch.rand(nr, device=device) * 2 - 1) * f_max * fz_scale

        new_f = torch.zeros(nr, num_bodies, 3, device=device)
        new_f[:, :, 0] = fx.unsqueeze(-1)
        new_f[:, :, 1] = fy.unsqueeze(-1)
        new_f[:, :, 2] = fz.unsqueeze(-1)

        new_t = torch.zeros(nr, num_bodies, 3, device=device)
        if t_max > 1e-6:
            tau_yaw = (torch.rand(nr, device=device) * 2 - 1) * t_max
            new_t[:, :, 2] = tau_yaw.unsqueeze(-1)

        new_f[is_zero] = 0.0
        new_t[is_zero] = 0.0

        s["target_f"][just_reset] = new_f
        s["target_t"][just_reset] = new_t

    # Piecewise-linear envelope s(t)
    t_ep = env.episode_length_buf[env_ids].float() * dt
    t_up = 0.1 * T
    t_hold_end = 0.9 * T

    alpha = torch.zeros(len(env_ids), device=device)
    in_up = t_ep < t_up
    in_hold = (t_ep >= t_up) & (t_ep < t_hold_end)
    in_dn = t_ep >= t_hold_end
    alpha[in_up] = (t_ep[in_up] / t_up).clamp(0.0, 1.0)
    alpha[in_hold] = 1.0
    alpha[in_dn] = ((T - t_ep[in_dn]) / (0.1 * T)).clamp(0.0, 1.0)

    a = alpha.view(-1, 1, 1)
    asset.permanent_wrench_composer.set_forces_and_torques(
        forces=s["target_f"][env_ids] * a,
        torques=s["target_t"][env_ids] * a,
        body_ids=asset_cfg.body_ids,
        env_ids=env_ids,
    )


def _trap_transition(
    s: dict, exp_ids: torch.Tensor, f_max: float, fz_scale: float,
    t_max: float, ramp_s_range: tuple, hold_s_range: tuple,
    zero_s_range: tuple, zero_prob: float, num_bodies: int, device: torch.device,
) -> None:
    """Advance expired envs to the next phase, resample on new cycle."""
    cur_phase = s["phase"][exp_ids]
    n = len(exp_ids)

    # Next phase: 0→1→2→3→0
    next_phase = (cur_phase + 1) % 4
    s["phase"][exp_ids] = next_phase

    # Sample durations for the new phase
    dur = torch.zeros(n, device=device)
    is_ramp = (next_phase == _TRAP_RAMP_UP) | (next_phase == _TRAP_RAMP_DOWN)
    is_hold = next_phase == _TRAP_HOLD
    is_zero = next_phase == _TRAP_ZERO

    if is_ramp.any():
        dur[is_ramp] = torch.empty(is_ramp.sum(), device=device).uniform_(*ramp_s_range)
    if is_hold.any():
        dur[is_hold] = torch.empty(is_hold.sum(), device=device).uniform_(*hold_s_range)
    if is_zero.any():
        dur[is_zero] = torch.empty(is_zero.sum(), device=device).uniform_(*zero_s_range)

    s["timer"][exp_ids] = dur
    s["duration"][exp_ids] = dur

    # ── Resample target force on new cycle (entering RAMP_UP) ────────────
    new_cycle = next_phase == _TRAP_RAMP_UP
    if new_cycle.any():
        cycle_ids = exp_ids[new_cycle]
        nc = len(cycle_ids)

        # Zero-wrench episodes
        is_zero_ep = torch.rand(nc, device=device) < zero_prob

        # Bucket-scaled force range
        b_lo = s["bucket_lo"][cycle_ids] * f_max
        b_hi = s["bucket_hi"][cycle_ids] * f_max

        # Sample per-axis XY magnitude within bucket, random sign
        mag_xy = torch.rand(nc, 2, device=device) * (b_hi - b_lo).unsqueeze(-1) + b_lo.unsqueeze(-1)
        sign_xy = torch.sign(torch.empty(nc, 2, device=device).uniform_(-1, 1))
        sign_xy[sign_xy == 0] = 1.0
        xy = mag_xy * sign_xy

        # Z force
        bz_lo = b_lo * fz_scale
        bz_hi = b_hi * fz_scale
        mag_z = torch.rand(nc, 1, device=device) * (bz_hi - bz_lo).unsqueeze(-1) + bz_lo.unsqueeze(-1)
        sign_z = torch.sign(torch.empty(nc, 1, device=device).uniform_(-1, 1))
        sign_z[sign_z == 0] = 1.0
        z = mag_z * sign_z

        forces = torch.zeros(nc, num_bodies, 3, device=device)
        forces[:, :, 0] = xy[:, 0:1]
        forces[:, :, 1] = xy[:, 1:2]
        forces[:, :, 2] = z[:, 0:1]

        # Torques
        torques = torch.zeros(nc, num_bodies, 3, device=device)
        if t_max > 1e-6:
            mag_t = torch.empty(nc, 3, device=device).uniform_(0, t_max)
            sign_t = torch.sign(torch.empty(nc, 3, device=device).uniform_(-1, 1))
            sign_t[sign_t == 0] = 1.0
            tv = mag_t * sign_t
            torques[:, :, 0] = tv[:, 0:1]
            torques[:, :, 1] = tv[:, 1:2]
            torques[:, :, 2] = tv[:, 2:3]

        # Zero out for zero-wrench episodes
        forces[is_zero_ep] = 0.0
        torques[is_zero_ep] = 0.0

        s["target_f"][cycle_ids] = forces
        s["target_t"][cycle_ids] = torques


def push_by_setting_velocity_with_return(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Push the asset by setting the root velocity and return the sampled velocity delta.

    Same as isaaclab.envs.mdp.events.push_by_setting_velocity but returns the sampled
    velocity delta for visualization purposes.

    Args:
        env: The environment instance.
        env_ids: The environment indices to apply the push to.
        velocity_range: Dictionary with velocity ranges for each axis.
            Keys: "x", "y", "z", "roll", "pitch", "yaw". Values: (min, max) tuples.
        asset_cfg: The asset configuration to apply the push to.

    Returns:
        The sampled velocity delta tensor of shape (num_env_ids, 6) containing
        [x, y, z, roll, pitch, yaw] velocities.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # velocities
    vel_w = asset.data.root_vel_w[env_ids]

    # sample random velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    sampled_vel = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], vel_w.shape, device=asset.device)

    # apply the velocity
    vel_w = vel_w + sampled_vel

    # set the velocities into the physics simulation
    asset.write_root_velocity_to_sim(vel_w, env_ids=env_ids)

    # return the sampled velocity delta
    return sampled_vel


def push_with_visualization(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Push the asset and visualize the push direction with arrows.

    This event is intended for play/evaluation mode to test compliant behavior.
    It applies a velocity push and shows red arrows indicating push direction.

    Args:
        env: The environment instance.
        env_ids: The environment indices to apply the push to.
        velocity_range: Dictionary with velocity ranges for each axis.
        asset_cfg: The asset configuration to apply the push to.
    """
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # Initialize visualization markers on first call
    if not hasattr(env, "_push_visualizer"):
        marker_cfg = RED_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/Events/push_velocity")
        marker_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
        env._push_visualizer = VisualizationMarkers(marker_cfg)
        env._push_vel_storage = torch.zeros(env.num_envs, 2, device=env.device)
        env._push_active = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._push_timer = torch.zeros(env.num_envs, device=env.device)

    # Apply push
    vel_w = asset.data.root_vel_w[env_ids]
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    sampled_vel = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], vel_w.shape, device=asset.device)
    vel_w = vel_w + sampled_vel
    asset.write_root_velocity_to_sim(vel_w, env_ids=env_ids)

    # Store push velocity for visualization (x, y components)
    env._push_vel_storage[env_ids, 0] = sampled_vel[:, 0]
    env._push_vel_storage[env_ids, 1] = sampled_vel[:, 1]
    env._push_active[env_ids] = True
    env._push_timer[env_ids] = 1.0  # Show arrow for 1 second

    # Update visualization for all active pushes
    _update_push_visualization(env, asset)


def _update_push_visualization(env: ManagerBasedEnv, asset: RigidObject | Articulation):
    """Update push visualization markers."""
    if not hasattr(env, "_push_visualizer"):
        return

    # Decay timer
    env._push_timer = (env._push_timer - env.step_dt).clamp(min=0)
    env._push_active = env._push_timer > 0

    active_envs = env._push_active.nonzero(as_tuple=True)[0]

    if len(active_envs) == 0:
        env._push_visualizer.set_visibility(False)
        return

    env._push_visualizer.set_visibility(True)

    # Get positions and orientations for active envs
    base_pos_w = asset.data.root_pos_w[active_envs].clone()
    base_pos_w[:, 2] += 0.5  # Offset above robot
    base_quat_w = asset.data.root_quat_w[active_envs]

    # Convert push velocity to arrow orientation
    push_vel_xy = env._push_vel_storage[active_envs]
    arrow_scale, arrow_quat = _velocity_to_arrow(push_vel_xy, base_quat_w, env.device)

    env._push_visualizer.visualize(base_pos_w, arrow_quat, arrow_scale)


def _velocity_to_arrow(
    xy_velocity: torch.Tensor, base_quat_w: torch.Tensor, device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert XY velocity to arrow scale and quaternion."""
    num_envs = xy_velocity.shape[0]

    # Default arrow scale
    arrow_scale = torch.tensor([0.5, 0.5, 0.5], device=device).repeat(num_envs, 1)
    vel_magnitude = torch.linalg.norm(xy_velocity, dim=1)
    arrow_scale[:, 0] *= vel_magnitude * 3.0

    # Heading angle from velocity
    heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
    zeros = torch.zeros_like(heading_angle)
    arrow_quat = quat_from_euler_xyz(zeros, zeros, heading_angle)
    arrow_quat = quat_mul(base_quat_w, arrow_quat)

    return arrow_scale, arrow_quat


def randomize_payload_mass(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    mass_range: tuple[float, float],
) -> None:
    """Randomize the mass of the last body (payload_link) uniformly in [min, max].

    Args:
        env: The environment instance.
        env_ids: Environment indices to randomize.
        asset_cfg: Asset configuration (must have root_physx_view).
        mass_range: (min_mass, max_mass) in kg.
    """
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    num_envs = len(env_ids)
    num_bodies = asset.num_bodies

    random_masses = mass_range[0] + (mass_range[1] - mass_range[0]) * torch.rand(num_envs, 1, device=env.device)

    masses = asset.root_physx_view.get_masses().clone()

    payload_body_idx = num_bodies - 1
    for i, env_id in enumerate(env_ids):
        masses[env_id, payload_body_idx] = random_masses[i, 0]

    asset.root_physx_view.set_masses(masses, env_ids.cpu())