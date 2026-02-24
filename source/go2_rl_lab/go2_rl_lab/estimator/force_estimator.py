"""Force-Only State Estimator Network — HAC-LOCO Style.

Architecture:

    o_t^H  ──► Encoder [128, 64] ──► z_t (64-dim)
                                         ├── f_head [32, 16] ──► f̂_t (2-dim, XY)
                                         │
                              l_t = concat(f̂_t, z_t)   (66 dims)
                                         │
                              Decoder [256, 128] ──► ô_{t+1}

Training losses (called per PPO mini-batch, own optimizer):
    L_force = MSE(f̂_t,  f_gt_xy)      — supervised force estimation
    L_rec   = MSE(ô_{t+1}, o_{t+1})    — reconstruction regularises encoder

The latent l_t is concatenated with the current proprioceptive obs o_t and
fed to the locomotion policy: policy_input = concat(o_t, l_t).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .velocity_estimator import _build_mlp, _get_activation


class ForceEstimator(nn.Module):
    """Force-only estimator with autoencoder regularisation.

    Produces ``l_t = concat(f̂_t, z_t)`` — no velocity head.

    Args:
        temporal_steps:     Number of historical timesteps stacked (H).
        num_one_step_obs:   Dimension of a single proprioceptive step.
        enc_hidden_dims:    Encoder MLP hidden widths.
        f_head_dims:        Force-head hidden widths.
        force_dim:          Force output dimension (default: 2 for XY).
        dec_hidden_dims:    Decoder MLP hidden widths.
        activation:         Activation name (default: 'elu').
        learning_rate:      Estimator optimizer LR.
        force_loss_weight:  Weight for force MSE loss.
        rec_loss_weight:    Weight for reconstruction MSE loss.
        max_grad_norm:      Gradient-clipping norm.
    """

    def __init__(
        self,
        temporal_steps: int,
        num_one_step_obs: int,
        enc_hidden_dims: list[int] | None = None,
        f_head_dims: list[int] | None = None,
        force_dim: int = 2,
        dec_hidden_dims: list[int] | None = None,
        activation: str = "elu",
        learning_rate: float = 1e-3,
        force_loss_weight: float = 1.0,
        rec_loss_weight: float = 1.0,
        max_grad_norm: float = 10.0,
        **kwargs,
    ) -> None:
        if kwargs:
            print(f"[ForceEstimator] Ignoring unexpected kwargs: {list(kwargs.keys())}")
        super().__init__()

        if enc_hidden_dims is None:
            enc_hidden_dims = [128, 64]
        if f_head_dims is None:
            f_head_dims = [32, 16]
        if dec_hidden_dims is None:
            dec_hidden_dims = [256, 128]

        act_fn = _get_activation(activation)

        self.temporal_steps = temporal_steps
        self.num_one_step_obs = num_one_step_obs
        self.enc_latent_dim = enc_hidden_dims[-1]  # z_t dimensionality
        self.force_dim = force_dim

        # l_t = concat(f̂_t, z_t)
        self.latent_dim = self.force_dim + self.enc_latent_dim

        self.force_loss_weight = force_loss_weight
        self.rec_loss_weight = rec_loss_weight
        self.max_grad_norm = max_grad_norm
        self.learning_rate = learning_rate

        # ── Encoder: o_t^H → z_t ─────────────────────────────────────────
        enc_input = temporal_steps * num_one_step_obs
        self.encoder = _build_mlp([enc_input] + enc_hidden_dims, act_fn)

        # ── Force head: z_t → f̂_t ────────────────────────────────────────
        self.f_head = _build_mlp([self.enc_latent_dim] + f_head_dims + [force_dim], act_fn)

        # ── Decoder: l_t → ô_{t+1} ───────────────────────────────────────
        self.decoder = _build_mlp([self.latent_dim] + dec_hidden_dims + [num_one_step_obs], act_fn)

        # ── Optimizer ─────────────────────────────────────────────────────
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    # ── Internal forward (shared by inference + training) ────────────────

    def _forward(self, obs_history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Shared forward pass returning (z_t, force_hat)."""
        z_t = self.encoder(obs_history)
        force_hat = self.f_head(z_t)
        return z_t, force_hat

    def _build_latent(self, force_hat: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        """Build l_t = concat(f̂_t, z_t)."""
        return torch.cat([force_hat, z_t], dim=-1)

    # ── Inference (no gradients back to policy) ──────────────────────────

    @torch.no_grad()
    def get_latent(self, obs_history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run a forward pass and return detached outputs for policy input.

        Args:
            obs_history: [num_envs, temporal_steps * num_one_step_obs]

        Returns:
            force_hat: [num_envs, force_dim]    — estimated XY force (detached)
            latent:    [num_envs, latent_dim]    — l_t = concat(f̂_t, z_t) (detached)
        """
        z_t, force_hat = self._forward(obs_history)
        latent = self._build_latent(force_hat, z_t)
        return force_hat, latent

    # ── Supervised training step ─────────────────────────────────────────

    def update(
        self,
        obs_history: torch.Tensor,
        gt_force: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> dict[str, float]:
        """One gradient step on the combined supervised loss.

        Called per PPO mini-batch (HAC-LOCO pattern).
        Uses its own fixed learning rate (decoupled from PPO).

        Args:
            obs_history:  [batch, temporal_steps * num_one_step_obs]
            gt_force:     [batch, force_dim]       — ground-truth XY force from critic obs
            next_obs:     [batch, num_one_step_obs] — target for reconstruction

        Returns:
            Dict with loss values and diagnostics.
        """
        z_t, force_hat = self._forward(obs_history)
        latent = self._build_latent(force_hat, z_t)
        next_obs_hat = self.decoder(latent)

        force_loss = F.mse_loss(force_hat, gt_force)
        rec_loss = F.mse_loss(next_obs_hat, next_obs)
        total_loss = self.force_loss_weight * force_loss + self.rec_loss_weight * rec_loss

        self.optimizer.zero_grad()
        total_loss.backward()

        # Collect gradient norms BEFORE clipping
        grad_norm_encoder = _grad_norm(self.encoder)
        grad_norm_f_head = _grad_norm(self.f_head)
        grad_norm_decoder = _grad_norm(self.decoder)

        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # ── Diagnostics (detached, no grad) ─────────────────────────────
        with torch.no_grad():
            gt_mag = gt_force.norm(dim=-1)           # [batch]
            pred_mag = force_hat.norm(dim=-1)         # [batch]
            error = (force_hat - gt_force).abs()      # [batch, 2]

        return {
            "force_loss": force_loss.item(),
            "rec_loss": rec_loss.item(),
            "total_loss": total_loss.item(),
            "estimator_lr": self.optimizer.param_groups[0]["lr"],
            # GT force stats
            "gt_force_mean_mag": gt_mag.mean().item(),
            "gt_force_max_mag": gt_mag.max().item(),
            "gt_force_std_mag": gt_mag.std().item(),
            # Prediction stats
            "pred_force_mean_mag": pred_mag.mean().item(),
            # Per-component MAE
            "mae_x": error[:, 0].mean().item(),
            "mae_y": error[:, 1].mean().item(),
            "mae_total": error.mean().item(),
            # Gradient norms (before clipping)
            "grad_norm_encoder": grad_norm_encoder,
            "grad_norm_f_head": grad_norm_f_head,
            "grad_norm_decoder": grad_norm_decoder,
        }


def _grad_norm(module: nn.Module) -> float:
    """Compute total L2 gradient norm of a module's parameters."""
    total = 0.0
    for p in module.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return total ** 0.5
