"""Intent Estimator Network — PAINT-style (RA-L 2026).

A simple MLP that maps proprioceptive history to estimated interaction wrench.
Trained with supervised regression loss against GT wrench (PAINT Eq. 8).

    o_t^H  ──► MLP [128, 64, 32] ──► β_est (force_dim)

The teacher-student distillation in PAINT is at the POLICY level
(KL between action distributions), not the estimator level.
This network is just the intent estimator that augments the student
policy's observations with estimated wrench.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def _get_activation(name: str) -> nn.Module:
    activations = {
        "elu": nn.ELU(),
        "relu": nn.ReLU(),
        "silu": nn.SiLU(),
        "tanh": nn.Tanh(),
    }
    if name not in activations:
        raise ValueError(f"Unknown activation '{name}'. Choose from {list(activations.keys())}")
    return activations[name]


def _build_mlp(layer_dims: list[int], act_fn: nn.Module, output_activation: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(layer_dims) - 1):
        layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
        if i < len(layer_dims) - 2 or output_activation:
            layers.append(act_fn)
    return nn.Sequential(*layers)


class IntentEstimator(nn.Module):
    """Intent estimator: proprioceptive history → estimated wrench.

    PAINT Eq. 7-8: β_est = [F̂_xy, Θ̂_yaw] from o_trans history.
    Trained with L_est = ||F̂ - F_gt||² + ||Θ̂ - Θ_gt||².

    Args:
        temporal_steps:   Number of historical timesteps (H).
        num_one_step_obs: Dimension of a single proprioceptive step.
        force_dim:        Output dimension (3 for XYZ, 4 for XYZ+τ_yaw).
        hidden_dims:      MLP hidden widths.
        activation:       Activation name.
        learning_rate:    Optimizer LR.
        max_grad_norm:    Gradient-clipping norm.
    """

    def __init__(
        self,
        temporal_steps: int,
        num_one_step_obs: int,
        force_dim: int = 3,
        hidden_dims: list[int] | None = None,
        activation: str = "elu",
        learning_rate: float = 1e-3,
        max_grad_norm: float = 10.0,
    ) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        self.temporal_steps = temporal_steps
        self.num_one_step_obs = num_one_step_obs
        self.force_dim = force_dim
        self.max_grad_norm = max_grad_norm

        act_fn = _get_activation(activation)
        input_dim = temporal_steps * num_one_step_obs
        self.net = _build_mlp([input_dim] + hidden_dims + [force_dim], act_fn)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    @torch.no_grad()
    def predict(self, obs_history: torch.Tensor) -> torch.Tensor:
        """Inference: obs_history [N, H*obs_dim] → β_est [N, force_dim]."""
        return self.net(obs_history)

    def update(self, obs_history: torch.Tensor, gt_wrench: torch.Tensor) -> dict[str, float]:
        """One supervised gradient step. PAINT Eq. 8.

        Args:
            obs_history: [batch, H * obs_dim]
            gt_wrench:   [batch, force_dim] — GT interaction wrench

        Returns:
            Dict with loss and MAE stats.
        """
        pred = self.net(obs_history)
        loss = F.mse_loss(pred, gt_wrench)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()

        with torch.no_grad():
            error = (pred - gt_wrench).abs()
            stats = {
                "est_loss": loss.item(),
                "mae_total": error.mean().item(),
                "mae_x": error[:, 0].mean().item(),
                "mae_y": error[:, 1].mean().item(),
            }
            if self.force_dim >= 3:
                stats["mae_z"] = error[:, 2].mean().item()
            if self.force_dim >= 4:
                stats["mae_tau_yaw"] = error[:, 3].mean().item()

            # Angular error on XY plane
            gt_angle = torch.atan2(gt_wrench[:, 1], gt_wrench[:, 0])
            pred_angle = torch.atan2(pred[:, 1], pred[:, 0])
            angle_diff = torch.atan2(
                torch.sin(gt_angle - pred_angle),
                torch.cos(gt_angle - pred_angle),
            )
            gt_mag_xy = gt_wrench[:, :2].norm(dim=-1)
            mask = gt_mag_xy > 1.0
            if mask.any():
                stats["angle_err_mean_deg"] = (angle_diff[mask].abs() * 180.0 / torch.pi).mean().item()
            else:
                stats["angle_err_mean_deg"] = 0.0

        return stats
