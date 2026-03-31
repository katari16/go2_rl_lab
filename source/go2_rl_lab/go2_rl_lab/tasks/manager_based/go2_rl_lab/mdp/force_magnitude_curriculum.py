"""Plateau-based force magnitude curriculum for force estimator training.

Reads ``estimator_force_loss_smooth`` from ``env.extras`` (written by the
EstimatorOnPolicyRunner) and ramps force magnitude when the f_head loss
plateaus — i.e. stops improving for ``patience`` iterations.

This uses the same logic as PyTorch's ``ReduceLROnPlateau`` scheduler:
track the best loss seen, and if no improvement (beyond ``min_delta``)
occurs for ``patience`` iterations, take action (ramp force instead of
reducing LR).

This naturally handles oscillating losses: a loss that oscillates around
an equilibrium is not improving, so patience runs out and force ramps.

Usage in env cfg::

    force_curriculum = CurrTerm(
        func=force_magnitude_curriculum,
        params={
            "event_term_name": "persistent_xy_force",
            "max_force": 20.0,
            "ramp_step": 2.0,
            "patience": 200,
            "min_delta": 0.01,
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
    event_term_name: str = "persistent_xy_force",
    max_force: float = 20.0,
    start_force: float = 0.0,
    ramp_step: float = 2.0,
    patience: int = 200,
    min_delta: float = 0.01,
) -> torch.Tensor:
    """Curriculum that ramps external force magnitude on loss plateau.

    Uses ReduceLROnPlateau-style logic: tracks the best force_loss seen
    since the last ramp and counts iterations without improvement.  When
    ``patience`` iterations pass without the loss improving by at least
    ``min_delta`` (relative), the force magnitude is increased.

    Args:
        env: Environment instance.
        env_ids: Env indices (unused — curriculum is global).
        event_term_name: Name of the event term whose ``force_range`` param
            will be updated.
        max_force: Maximum force magnitude (N) to ramp up to.
        start_force: Initial force magnitude (N) on first activation.
        ramp_step: How much to increase the max force per curriculum step (N).
        patience: Number of iterations without improvement before ramping.
        min_delta: Minimum relative improvement to reset the patience counter
            (0.01 = loss must drop by at least 1% to count as improvement).

    Returns:
        Current force magnitude (scalar tensor) for TensorBoard logging.
    """
    # ── Lazy-init persistent state on env ───────────────────────────────
    if not hasattr(env, "_fc_active"):
        env._fc_active = False
        env._fc_best_loss = float("inf")
        env._fc_wait_count = 0
        env._fc_last_seen_loss = None

    # ── Read smoothed force loss from extras (written by runner) ────────
    force_loss = env.extras.get("estimator_force_loss_smooth", None)

    # Get current force range from event term
    event_cfg = env.event_manager.get_term_cfg(event_term_name)
    current_range = event_cfg.params.get("force_range", (0.0, 0.0))
    current_max = current_range[1]

    # Skip if estimator hasn't reported yet
    if force_loss is None or force_loss == 0:
        env.extras["Curriculum/force_magnitude"] = current_max
        return torch.tensor(current_max, device=env.device)

    # Deduplicate: curriculum is called 24x per iteration (once per env.step)
    # but force_loss_smooth only updates once. Skip duplicate calls.
    if force_loss == env._fc_last_seen_loss:
        env.extras["Curriculum/force_magnitude"] = current_max
        return torch.tensor(current_max, device=env.device)
    env._fc_last_seen_loss = force_loss

    # ── First activation: apply initial small force ─────────────────────
    if not env._fc_active:
        new_max = min(start_force if start_force > 0 else ramp_step, max_force)
        event_cfg.params["force_range"] = (0.0, new_max)
        current_max = new_max
        env._fc_active = True
        env._fc_best_loss = float("inf")
        env._fc_wait_count = 0

    # ── Plateau detection and ramp ──────────────────────────────────────
    elif current_max < max_force:
        # Check for improvement (relative threshold, like ReduceLROnPlateau)
        if force_loss < env._fc_best_loss * (1 - min_delta):
            env._fc_best_loss = force_loss
            env._fc_wait_count = 0
        else:
            env._fc_wait_count += 1

        # Patience exhausted — ramp force
        if env._fc_wait_count >= patience:
            new_max = min(current_max + ramp_step, max_force)
            event_cfg.params["force_range"] = (0.0, new_max)
            current_max = new_max
            env._fc_best_loss = float("inf")
            env._fc_wait_count = 0

    # ── Log to extras for TensorBoard ───────────────────────────────────
    env.extras["Curriculum/force_magnitude"] = current_max
    env.extras["Curriculum/force_plateau_count"] = env._fc_wait_count

    return torch.tensor(current_max, device=env.device)
