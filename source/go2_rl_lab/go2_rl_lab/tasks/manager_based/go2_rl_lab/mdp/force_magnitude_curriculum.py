"""Adaptive force magnitude curriculum driven by estimator loss convergence.

Reads ``estimator_force_loss_smooth`` from ``env.extras`` (written by the
EstimatorOnPolicyRunner) and only ramps up force magnitude when the f_head
has converged at the current level — i.e. the smoothed loss has dropped by
a configurable fraction relative to the loss recorded at the last ramp.

Usage in env cfg::

    force_curriculum = CurrTerm(
        func=force_magnitude_curriculum,
        params={
            "event_term_name": "persistent_xy_force",
            "max_force": 20.0,
            "ramp_step": 2.0,
            "convergence_ratio": 0.6,
            "min_wait": 100,
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
    ramp_step: float = 2.0,
    convergence_ratio: float = 0.8,
    min_wait: int = 100,
) -> torch.Tensor:
    """Curriculum that ramps external force magnitude based on estimator loss convergence.

    The force only increases when the force estimator's smoothed loss has
    dropped to ``convergence_ratio`` of its value at the last ramp, ensuring
    the f_head has learned at the current force level before advancing.

    Args:
        env: Environment instance.
        env_ids: Env indices (unused — curriculum is global).
        event_term_name: Name of the event term whose ``force_range`` param
            will be updated.
        max_force: Maximum force magnitude (N) to ramp up to.
        ramp_step: How much to increase the max force per curriculum step (N).
        convergence_ratio: Loss must drop to this fraction of the baseline
            recorded at the last ramp before ramping again (0.8 = 20% drop).
        min_wait: Minimum curriculum calls between ramps to prevent
            reacting to transient loss dips.

    Returns:
        Current force magnitude (scalar tensor) for TensorBoard logging.
    """
    # ── Lazy-init persistent state on env ───────────────────────────────
    if not hasattr(env, "_fc_active"):
        env._fc_active = False
        env._fc_loss_at_last_ramp = float("inf")
        env._fc_call_count = 0

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

    # ── First activation: apply initial small force ─────────────────────
    if not env._fc_active:
        new_max = min(ramp_step, max_force)
        event_cfg.params["force_range"] = (0.0, new_max)
        current_max = new_max
        env._fc_active = True
        env._fc_loss_at_last_ramp = force_loss
        env._fc_call_count = 0

    # ── Check for convergence and ramp ──────────────────────────────────
    elif current_max < max_force:
        env._fc_call_count += 1
        if (
            env._fc_call_count >= min_wait
            and force_loss < convergence_ratio * env._fc_loss_at_last_ramp
        ):
            new_max = min(current_max + ramp_step, max_force)
            event_cfg.params["force_range"] = (0.0, new_max)
            current_max = new_max
            env._fc_loss_at_last_ramp = force_loss
            env._fc_call_count = 0

    # ── Log to extras for TensorBoard ───────────────────────────────────
    env.extras["Curriculum/force_magnitude"] = current_max

    return torch.tensor(current_max, device=env.device)
