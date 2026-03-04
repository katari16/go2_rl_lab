from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def force_activation_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    event_term_name: str = "persistent_xy_force",
    reward_threshold: float = 30.0,
    max_force: float = 20.0,
) -> torch.Tensor:
    """Curriculum that activates forces and r_comp once walking is learned.

    Estimates mean episode reward from long-running envs' per-step reward
    rate, scaled to a full episode. Same threshold semantics as the
    ForceOnPolicyRunner (total episode reward >= threshold).

    Args:
        env: Environment instance.
        env_ids: Env indices (unused — curriculum is global).
        event_term_name: Name of the interval event term for persistent force.
        reward_threshold: Mean **total** episode reward to trigger activation.
        max_force: Force magnitude (N) once activated.

    Returns:
        Scalar tensor (0 or 1) for TensorBoard logging.
    """
    if not hasattr(env, "_fac_active"):
        env._fac_active = False
        env._rcomp_active = False
        env._fac_last_step = -1
        env._fac_print_counter = 0

    step = env.common_step_counter
    if step == env._fac_last_step:
        return torch.tensor(float(env._fac_active), device=env.device)
    env._fac_last_step = step
    env._fac_print_counter += 1

    # ── After activation: log force magnitude + r_comp ────────────────
    if env._fac_active:
        env.extras["Curriculum/force_active"] = 1.0
        env.extras["Curriculum/reward_progress_pct"] = 100.0
        asset = env.scene["robot"]
        f_xy = asset.permanent_wrench_composer.composed_force_as_torch[:, 0, :2]
        f_mags = f_xy.norm(dim=1)
        f_mean = f_mags.mean().item()
        f_max = f_mags.max().item()
        env.extras["Curriculum/force_magnitude_mean"] = f_mean
        env.extras["Curriculum/force_magnitude_max"] = f_max
        env.extras["Curriculum/num_envs_with_force"] = (f_mags > 0.5).sum().float().item()
        if env._fac_print_counter % 24 == 0:
            rcomp_val = env.extras.get("r_comp/penalty_mean", 0.0)
            a_prime = env.extras.get("r_comp/a_prime_xy_norm_mean", 0.0)
            n_above = int(env.extras.get("r_comp/num_envs_above_alpha", 0))
            print(
                f"  [Compliant] ACTIVE  |f|={f_mean:.1f}N (max {f_max:.1f}N)  "
                f"r_comp={rcomp_val:.4f}  |a'|={a_prime:.3f}  "
                f"envs>α={n_above}"
            )
        return torch.tensor(1.0, device=env.device)

    # ── Estimate mean episode reward from long-running envs ─────────────
    # Only use envs with enough steps for a stable per-step estimate
    min_steps = 200
    long_running = env.episode_length_buf >= min_steps
    n_long = long_running.sum().item()

    if n_long < 100:
        mean_ep_reward = 0.0
    else:
        total_reward = torch.zeros(env.num_envs, device=env.device)
        for term_name in env.reward_manager._episode_sums:
            total_reward += env.reward_manager._episode_sums[term_name]
        ep_lens = env.episode_length_buf[long_running].float()
        per_step = total_reward[long_running] / ep_lens
        mean_ep_reward = (per_step.mean() * env.max_episode_length).item()

    # ── Log ──────────────────────────────────────────────────────────────
    pct = min(100.0, mean_ep_reward / reward_threshold * 100.0) if reward_threshold > 0 else 0.0
    env.extras["Curriculum/mean_ep_reward"] = mean_ep_reward
    env.extras["Curriculum/reward_progress_pct"] = pct
    env.extras["Curriculum/force_active"] = 0.0
    env.extras["Curriculum/force_magnitude_mean"] = 0.0

    if env._fac_print_counter % 24 == 0:
        print(
            f"  [Compliant] WAITING  reward={mean_ep_reward:.1f}/{reward_threshold:.1f} "
            f"({pct:.0f}%)  long_running_envs={n_long}"
        )

    # ── Check threshold ─────────────────────────────────────────────────
    if n_long >= 100 and mean_ep_reward >= reward_threshold:
        env._fac_active = True
        env._rcomp_active = True

        event_cfg = env.event_manager.get_term_cfg(event_term_name)
        event_cfg.params["force_range"] = (0.0, max_force)

        env.extras["Curriculum/force_active"] = 1.0
        print(
            f"\n{'=' * 70}\n"
            f"  [ForceActivationCurriculum] Mean episode reward "
            f"{mean_ep_reward:.1f} >= {reward_threshold:.1f}\n"
            f"  ACTIVATING: forces={max_force:.0f}N + r_comp  (step {step})\n"
            f"{'=' * 70}"
        )

    return torch.tensor(float(env._fac_active), device=env.device)


def lin_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.lin_vel_x = torch.clamp(
                torch.tensor(ranges.lin_vel_x, device=env.device) + delta_command,
                limit_ranges.lin_vel_x[0],
                limit_ranges.lin_vel_x[1],
            ).tolist()
            ranges.lin_vel_y = torch.clamp(
                torch.tensor(ranges.lin_vel_y, device=env.device) + delta_command,
                limit_ranges.lin_vel_y[0],
                limit_ranges.lin_vel_y[1],
            ).tolist()

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def ang_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_ang_vel_z",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.ang_vel_z = torch.clamp(
                torch.tensor(ranges.ang_vel_z, device=env.device) + delta_command,
                limit_ranges.ang_vel_z[0],
                limit_ranges.ang_vel_z[1],
            ).tolist()

    return torch.tensor(ranges.ang_vel_z[1], device=env.device)
