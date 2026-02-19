"""Reward-gated force magnitude curriculum for force estimation training.

Monitors the ``track_lin_vel_xy_exp`` reward rate and only starts applying
external forces once the robot has learned to walk well (rate > threshold).
Force magnitude then ramps linearly from 0 to ``max_force`` over curriculum
steps.

Usage in env cfg::

    force_curriculum = CurrTerm(
        func=force_magnitude_curriculum,
        params={
            "reward_term_name": "track_lin_vel_xy_exp",
            "reward_threshold_frac": 0.85,
            "event_term_name": "persistent_xy_force",
            "max_force": 50.0,
            "ramp_step": 2.0,
        },
    )
"""
from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def force_magnitude_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy_exp",
    reward_threshold_frac: float = 0.85,
    event_term_name: str = "persistent_xy_force",
    max_force: float = 50.0,
    ramp_step: float = 2.0,
) -> torch.Tensor:
    """Curriculum that ramps external force magnitude based on tracking reward.

    Args:
        env: Environment instance.
        env_ids: Env indices (unused — curriculum is global).
        reward_term_name: Reward term to monitor for gating.
        reward_threshold_frac: Fraction of the reward weight that must be
            achieved before forces start ramping.
        event_term_name: Name of the event term whose ``force_range`` param
            will be updated.
        max_force: Maximum force magnitude (N) to ramp up to.
        ramp_step: How much to increase the max force per curriculum step (N).

    Returns:
        Current force magnitude (scalar tensor) for TensorBoard logging.
    """
    # Get current reward rate
    reward_cfg = env.reward_manager.get_term_cfg(reward_term_name)
    episode_sum = torch.mean(
        env.reward_manager._episode_sums[reward_term_name][env_ids]
    )
    reward_rate = episode_sum / env.max_episode_length_s

    # Threshold = fraction × weight (weight is the max possible rate)
    threshold = reward_threshold_frac * reward_cfg.weight

    # Get event term cfg to modify its force_range param
    event_cfg = env.event_manager.get_term_cfg(event_term_name)
    current_range = event_cfg.params.get("force_range", (0.0, 0.0))
    current_max = current_range[1]

    # Only check at episode boundaries to avoid jitter
    if env.common_step_counter % env.max_episode_length == 0:
        if reward_rate > threshold and current_max < max_force:
            new_max = min(current_max + ramp_step, max_force)
            event_cfg.params["force_range"] = (0.0, new_max)
            current_max = new_max

    # Log to extras for TensorBoard
    env.extras["Curriculum/force_magnitude"] = current_max

    return torch.tensor(current_max, device=env.device)
