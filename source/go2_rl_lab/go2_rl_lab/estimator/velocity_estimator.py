"""State Estimator Network — HAC-LOCO Style (velocity + optional force estimation).

Architecture (from HAC-LOCO Zhou et al. 2025):

    o_t^H  ──► Encoder [128, 64] ──► z_t (64-dim)
                                         ├── v_head [32, 16] ──► v̂_t (3-dim)
                                         └── f_head [32, 16] ──► f̂_t (2-dim, XY only, optional)
                                         │
                              l_t = concat(v̂_t, [f̂_t], z_t)   (67 or 69 dims)
                                         │
                              Decoder [512, 256, 128] ──► ô_{t+1}

Training losses (run concurrently with PPO):
    L_vel = MSE(v̂_t,  v_gt)          — supervised velocity estimation
    L_frc = MSE(f̂_t,  f_gt_xy)      — supervised force estimation (optional)
    L_rec = MSE(ô_{t+1}, o_{t+1})    — reconstruction regularises encoder

The latent l_t is concatenated with the current proprioceptive obs o_t and
fed to the locomotion policy: policy_input = concat(o_t, l_t).

References:
    HAC-LOCO: Zhou et al. (2025) arXiv:2507.02447
    HIM Loco: Long et al. (2024) arXiv:2312.11460  (contrastive variant)
    Ji et al. (2022) IEEE RA-L — concurrent policy + estimator training
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def _build_mlp(layer_dims: list[int], act_fn: nn.Module, output_activation: bool = False) -> nn.Sequential:
    """Build a fully-connected MLP from a list of layer widths."""
    layers: list[nn.Module] = []
    for i in range(len(layer_dims) - 1):
        layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
        if i < len(layer_dims) - 2 or output_activation:
            layers.append(act_fn)
    return nn.Sequential(*layers)


def _get_activation(name: str) -> nn.Module:
    activations = {
        "elu": nn.ELU(),
        "selu": nn.SELU(),
        "relu": nn.ReLU(),
        "silu": nn.SiLU(),
        "lrelu": nn.LeakyReLU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
    }
    if name not in activations:
        raise ValueError(f"Unknown activation '{name}'. Choose from {list(activations.keys())}")
    return activations[name]


class VelocityEstimator(nn.Module):
    """HAC-LOCO-style state estimator with autoencoder regularisation.

    Supports velocity estimation (always) and optional force estimation via an
    additional ``f_head``.  When ``f_head_dims`` is provided and non-empty the
    network produces ``l_t = concat(v̂_t, f̂_t, z_t)``; otherwise the original
    ``l_t = concat(v̂_t, z_t)`` is returned (fully backward compatible).

    Args:
        temporal_steps:     Number of historical timesteps stacked (H).
        num_one_step_obs:   Dimension of a single proprioceptive step (e.g. 45).
        enc_hidden_dims:    Encoder MLP hidden widths (default: [256, 128, 64]).
        v_head_dims:        Velocity-head hidden widths (default: [32, 16]).
        f_head_dims:        Force-head hidden widths (default: None — no force head).
        force_dim:          Force output dimension (default: 2 for XY).
        dec_hidden_dims:    Decoder MLP hidden widths (default: [512, 256, 128]).
        activation:         Activation name (default: 'elu').
        learning_rate:      Estimator optimizer LR.
        vel_loss_weight:    Weight for velocity MSE loss (ω₁).
        force_loss_weight:  Weight for force MSE loss (ω₂).
        rec_loss_weight:    Weight for reconstruction MSE loss (ω₃).
        max_grad_norm:      Gradient-clipping norm.
    """

    def __init__(
        self,
        temporal_steps: int,
        num_one_step_obs: int,
        enc_hidden_dims: list[int] | None = None,
        v_head_dims: list[int] | None = None,
        f_head_dims: list[int] | None = None,
        force_dim: int = 2,
        dec_hidden_dims: list[int] | None = None,
        activation: str = "elu",
        learning_rate: float = 1e-3,
        vel_loss_weight: float = 1.0,
        force_loss_weight: float = 1.0,
        rec_loss_weight: float = 1.0,
        max_grad_norm: float = 10.0,
        **kwargs,
    ) -> None:
        if kwargs:
            print(f"[VelocityEstimator] Ignoring unexpected kwargs: {list(kwargs.keys())}")
        super().__init__()

        if enc_hidden_dims is None:
            enc_hidden_dims = [256, 128, 64]
        if v_head_dims is None:
            v_head_dims = [32, 16]
        if dec_hidden_dims is None:
            dec_hidden_dims = [512, 256, 128]

        act_fn = _get_activation(activation)

        self.temporal_steps = temporal_steps
        self.num_one_step_obs = num_one_step_obs
        self.enc_latent_dim = enc_hidden_dims[-1]   # z_t dimensionality
        self.vel_dim = 3                             # vx, vy, vz

        # ── Force head config ────────────────────────────────────────────
        self.has_force_head = f_head_dims is not None and len(f_head_dims) > 0
        self.force_dim = force_dim if self.has_force_head else 0

        # l_t = concat(v̂_t, [f̂_t], z_t)
        self.latent_dim = self.vel_dim + self.force_dim + self.enc_latent_dim

        self.vel_loss_weight = vel_loss_weight
        self.force_loss_weight = force_loss_weight
        self.rec_loss_weight = rec_loss_weight
        self.max_grad_norm = max_grad_norm
        self.learning_rate = learning_rate

        # ── Encoder: o_t^H → z_t ─────────────────────────────────────────
        enc_input = temporal_steps * num_one_step_obs
        self.encoder = _build_mlp([enc_input] + enc_hidden_dims, act_fn)

        # ── Velocity head: z_t → v̂_t ──────────────────────────────────────
        self.v_head = _build_mlp([self.enc_latent_dim] + v_head_dims + [self.vel_dim], act_fn)

        # ── Force head: z_t → f̂_t (optional) ─────────────────────────────
        if self.has_force_head:
            self.f_head = _build_mlp([self.enc_latent_dim] + f_head_dims + [force_dim], act_fn)

        # ── Decoder: l_t → ô_{t+1} ───────────────────────────────────────
        self.decoder = _build_mlp([self.latent_dim] + dec_hidden_dims + [num_one_step_obs], act_fn)

        # ── Optimizer ─────────────────────────────────────────────────────
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    # ── Internal forward (shared by inference + training) ────────────────

    def _forward(self, obs_history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Shared forward pass returning (z_t, vel_hat, force_hat_or_None)."""
        z_t = self.encoder(obs_history)
        vel_hat = self.v_head(z_t)
        force_hat = self.f_head(z_t) if self.has_force_head else None
        return z_t, vel_hat, force_hat

    def _build_latent(
        self, vel_hat: torch.Tensor, force_hat: torch.Tensor | None, z_t: torch.Tensor
    ) -> torch.Tensor:
        """Build l_t = concat(v̂_t, [f̂_t], z_t)."""
        parts = [vel_hat]
        if force_hat is not None:
            parts.append(force_hat)
        parts.append(z_t)
        return torch.cat(parts, dim=-1)

    # ── Inference (no gradients back to policy) ──────────────────────────

    @torch.no_grad()
    def get_latent(self, obs_history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run a forward pass and return detached outputs for policy input.

        Args:
            obs_history: [num_envs, temporal_steps * num_one_step_obs]

        Returns:
            vel_hat: [num_envs, 3]          — estimated base velocity (detached)
            latent:  [num_envs, latent_dim] — l_t = concat(v̂_t, [f̂_t], z_t) (detached)
        """
        z_t, vel_hat, force_hat = self._forward(obs_history)
        latent = self._build_latent(vel_hat, force_hat, z_t)
        return vel_hat, latent

    # ── Supervised training step ─────────────────────────────────────────

    def update(
        self,
        obs_history: torch.Tensor,
        gt_velocity: torch.Tensor,
        next_obs: torch.Tensor,
        gt_force: torch.Tensor | None = None,
        lr: float | None = None,
    ) -> dict[str, float]:
        """One gradient step on the combined supervised loss.

        Args:
            obs_history:  [batch, temporal_steps * num_one_step_obs]
            gt_velocity:  [batch, 3]              — ground-truth base velocity from sim
            next_obs:     [batch, num_one_step_obs] — target for reconstruction
            gt_force:     [batch, force_dim]       — ground-truth XY force (optional)

        Returns:
            Dict with keys: vel_loss, rec_loss, total_loss, and optionally force_loss.
        """
        if lr is not None:
            self.learning_rate = lr
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

        z_t, vel_hat, force_hat = self._forward(obs_history)
        latent = self._build_latent(vel_hat, force_hat, z_t)
        next_obs_hat = self.decoder(latent)

        vel_loss = F.mse_loss(vel_hat, gt_velocity)
        rec_loss = F.mse_loss(next_obs_hat, next_obs)
        total_loss = self.vel_loss_weight * vel_loss + self.rec_loss_weight * rec_loss

        losses = {
            "vel_loss": vel_loss.item(),
            "rec_loss": rec_loss.item(),
        }

        if self.has_force_head and gt_force is not None:
            force_loss = F.mse_loss(force_hat, gt_force)
            total_loss = total_loss + self.force_loss_weight * force_loss
            losses["force_loss"] = force_loss.item()

        losses["total_loss"] = total_loss.item()

        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return losses
