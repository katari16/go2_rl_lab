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

class TemporalConvBlock(nn.Module):
    """Single TCN block: dilated causal conv → activation → residual."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 dilation: int, activation: nn.Module):
        super().__init__()
        padding = (kernel_size - 1) * dilation  # causal padding
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              dilation=dilation, padding=padding)
        self.activation = activation
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.causal_trim = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, channels, time]
        out = self.conv(x)
        if self.causal_trim > 0:
            out = out[:, :, :-self.causal_trim]
        out = self.activation(out)
        return out + self.residual(x)


class TCN(nn.Module):
    """Temporal Convolutional Network: stacked dilated causal convolutions.

    Input:  [batch, time, features]
    Output: [batch, time, features]  (same shape — preprocessor mode)
    """

    def __init__(self, num_features: int, num_channels: list[int],
                 kernel_size: int = 3, dilations: list[int] | None = None,
                 activation: str = "elu"):
        super().__init__()
        act_fn = _get_activation(activation)
        if dilations is None:
            dilations = [2 ** i for i in range(len(num_channels))]

        layers: list[nn.Module] = []
        in_ch = num_features
        for i, out_ch in enumerate(num_channels):
            layers.append(TemporalConvBlock(in_ch, out_ch, kernel_size, dilations[i], act_fn))
            in_ch = out_ch
        self.network = nn.Sequential(*layers)
        # Project back to original feature dim if needed
        self.proj = nn.Linear(in_ch, num_features) if in_ch != num_features else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, time, features]
        out = x.permute(0, 2, 1)  # [batch, features, time]
        out = self.network(out)
        out = out.permute(0, 2, 1)  # [batch, time, features]
        out = self.proj(out)
        return out


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


def _build_mlp(layer_dims: list[int], act_fn: nn.Module, output_activation: bool = False) -> nn.Sequential:
    """Build a fully-connected MLP from a list of layer widths."""
    layers: list[nn.Module] = []
    for i in range(len(layer_dims) - 1):
        layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
        if i < len(layer_dims) - 2 or output_activation:
            layers.append(act_fn)
    return nn.Sequential(*layers)


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
        angle_loss_weight:  Weight for angular MSE loss (default: 1.0).
        rec_loss_weight:    Weight for reconstruction MSE loss.
        angle_min_force:    Min GT force magnitude to apply angular loss (default: 1.0 N).
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
        angle_loss_weight: float = 1.0,
        rec_loss_weight: float = 1.0,
        angle_min_force: float = 1.0,
        max_grad_norm: float = 10.0,
        torque_angle_loss_weight: float = 0.0,
        torque_angle_min: float = 0.3,
        yaw_loss_weight: float = 0.0,
        tcn_mode: str = "none",
        tcn_channels: list[int] | None = None,
        tcn_kernel_size: int = 3,
        tcn_dilations: list[int] | None = None,
        temporal_decay: str = "none",
        force_layout: str = "auto",
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
        self.angle_loss_weight = angle_loss_weight
        self.rec_loss_weight = rec_loss_weight
        self.angle_min_force = angle_min_force
        self.max_grad_norm = max_grad_norm
        self.learning_rate = learning_rate
        self.torque_angle_loss_weight = torque_angle_loss_weight
        self.torque_angle_min = torque_angle_min
        self.yaw_loss_weight = yaw_loss_weight

        # ── Optional TCN preprocessor/replacement ─────────────────────────
        self.tcn_mode = tcn_mode  # "none", "preprocessor", "replacement"
        self.tcn = None
        if tcn_mode in ("preprocessor", "replacement"):
            if tcn_channels is None:
                tcn_channels = [64, 64]
            self.tcn = TCN(
                num_features=num_one_step_obs,
                num_channels=tcn_channels,
                kernel_size=tcn_kernel_size,
                dilations=tcn_dilations,
                activation=activation,
            )
            print(f"[ForceEstimator] TCN mode={tcn_mode}, channels={tcn_channels}, "
                  f"kernel={tcn_kernel_size}, dilations={tcn_dilations}")

        # ── Temporal decay weighting ──────────────────────────────────────
        self.temporal_decay = temporal_decay
        if temporal_decay == "linear":
            w = torch.linspace(1.0 / temporal_steps, 1.0, temporal_steps)
            self.register_buffer("_decay_weights", w.view(1, temporal_steps, 1))
            print(f"[ForceEstimator] Linear temporal decay: oldest={1.0/temporal_steps:.3f}, newest=1.0")

        # ── Force layout ─────────────────────────────────────────────────
        self.force_layout = force_layout

        # ── Encoder: o_t^H → z_t ─────────────────────────────────────────
        enc_input = temporal_steps * num_one_step_obs
        if tcn_mode == "replacement":
            # TCN → global avg pool over time → project to enc_latent_dim
            self.encoder = None
            self.tcn_pool_proj = nn.Linear(num_one_step_obs, self.enc_latent_dim)
        else:
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
        if self.tcn is not None:
            x = obs_history.view(-1, self.temporal_steps, self.num_one_step_obs)
            if self.temporal_decay == "linear":
                x = x * self._decay_weights
            x = self.tcn(x)
            if self.tcn_mode == "replacement":
                z_t = x.mean(dim=1)
                z_t = self.tcn_pool_proj(z_t)
            else:
                z_t = self.encoder(x.reshape(-1, self.temporal_steps * self.num_one_step_obs))
        elif self.temporal_decay == "linear":
            x = obs_history.view(-1, self.temporal_steps, self.num_one_step_obs)
            x = x * self._decay_weights
            z_t = self.encoder(x.reshape(-1, self.temporal_steps * self.num_one_step_obs))
        else:
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

        Supports both 2D (XY) and 3D (XYZ) force estimation:
        - Force MSE loss: computed over all force components (2 or 3).
        - Angle loss: always computed on the XY plane projection using
          atan2(fy, fx). The mask uses XY magnitude so the angle is only
          penalized when the horizontal force is significant.
        - Reconstruction loss: unchanged.

        Args:
            obs_history:  [batch, temporal_steps * num_one_step_obs]
            gt_force:     [batch, force_dim]       — ground-truth force (2D or 3D)
            next_obs:     [batch, num_one_step_obs] — target for reconstruction

        Returns:
            Dict with loss values and diagnostics.
        """
        z_t, force_hat = self._forward(obs_history)

        force_loss = F.mse_loss(force_hat, gt_force)

        if self.rec_loss_weight > 0:
            if self.tcn_mode == "preprocessor":
                latent = self._build_latent(force_hat.detach(), z_t.detach())
            else:
                latent = self._build_latent(force_hat, z_t)
            next_obs_hat = self.decoder(latent)
            rec_loss = F.mse_loss(next_obs_hat, next_obs)
        else:
            rec_loss = torch.tensor(0.0, device=obs_history.device)

        # ── Angular loss: MSE on wrapped angle difference ────────────
        # Always computed on the XY plane (first 2 components).
        # Mask uses XY magnitude — angle is meaningless when horizontal
        # force is near zero, regardless of fz.
        gt_angle = torch.atan2(gt_force[:, 1], gt_force[:, 0])       # [batch]
        pred_angle = torch.atan2(force_hat[:, 1], force_hat[:, 0])   # [batch]
        angle_diff = gt_angle - pred_angle
        # Wrap to [-pi, pi]
        angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))

        gt_mag_xy = gt_force[:, :2].norm(dim=-1)  # [batch] — XY magnitude only
        mask = gt_mag_xy > self.angle_min_force
        if mask.any():
            angle_loss = (angle_diff[mask] ** 2).mean()
        else:
            angle_loss = torch.tensor(0.0, device=obs_history.device)

        # ── Torque angle loss: direction of torque in roll-pitch plane ───
        # For 6D wrench: indices [3]=τ_roll, [4]=τ_pitch, [5]=τ_yaw
        # Analogous to force angle loss but for the torque XY (roll/pitch) plane
        torque_angle_loss = torch.tensor(0.0, device=obs_history.device)
        if self.torque_angle_loss_weight > 0 and self.force_dim >= 6:
            gt_tau_angle = torch.atan2(gt_force[:, 4], gt_force[:, 3])
            pred_tau_angle = torch.atan2(force_hat[:, 4], force_hat[:, 3])
            tau_angle_diff = torch.atan2(
                torch.sin(gt_tau_angle - pred_tau_angle),
                torch.cos(gt_tau_angle - pred_tau_angle),
            )
            gt_tau_mag_rp = gt_force[:, 3:5].norm(dim=-1)
            tau_mask = gt_tau_mag_rp > self.torque_angle_min
            if tau_mask.any():
                torque_angle_loss = (tau_angle_diff[tau_mask] ** 2).mean()

        # ── Yaw torque loss: separate weighted MSE on yaw component ─────
        yaw_loss = torch.tensor(0.0, device=obs_history.device)
        if self.yaw_loss_weight > 0:
            if self.force_layout == "xy_yaw":
                yaw_idx = 2
            elif self.force_dim >= 6:
                yaw_idx = 5
            elif self.force_dim >= 4:
                yaw_idx = 3
            else:
                yaw_idx = None
            if yaw_idx is not None:
                yaw_loss = F.mse_loss(force_hat[:, yaw_idx], gt_force[:, yaw_idx])

        total_loss = (
            self.force_loss_weight * force_loss
            + self.angle_loss_weight * angle_loss
            + self.rec_loss_weight * rec_loss
            + self.torque_angle_loss_weight * torque_angle_loss
            + self.yaw_loss_weight * yaw_loss
        )

        self.optimizer.zero_grad()
        total_loss.backward()

        # Collect gradient norms BEFORE clipping
        grad_norm_encoder = _grad_norm(self.encoder) if self.encoder is not None else 0.0
        grad_norm_f_head = _grad_norm(self.f_head)
        grad_norm_decoder = _grad_norm(self.decoder)

        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # ── Diagnostics (detached, no grad) ─────────────────────────────
        with torch.no_grad():
            gt_mag = gt_force.norm(dim=-1)            # [batch] — full magnitude
            pred_mag = force_hat.norm(dim=-1)          # [batch]
            error = (force_hat - gt_force).abs()       # [batch, force_dim]
            # Angular error in degrees for interpretability (XY plane)
            angle_err_deg = angle_diff.abs() * (180.0 / torch.pi)
            if mask.any():
                mean_angle_err_deg = angle_err_deg[mask].mean().item()
                median_angle_err_deg = angle_err_deg[mask].median().item()
            else:
                mean_angle_err_deg = 0.0
                median_angle_err_deg = 0.0

        stats = {
            "force_loss": force_loss.item(),
            "angle_loss": angle_loss.item(),
            "rec_loss": rec_loss.item(),
            "torque_angle_loss": torque_angle_loss.item(),
            "yaw_loss": yaw_loss.item(),
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
            # Angular error (only for samples with |f_gt_xy| > threshold)
            "angle_err_mean_deg": mean_angle_err_deg,
            "angle_err_median_deg": median_angle_err_deg,
            # Gradient norms (before clipping)
            "grad_norm_encoder": grad_norm_encoder,
            "grad_norm_f_head": grad_norm_f_head,
            "grad_norm_decoder": grad_norm_decoder,
            "grad_norm_tcn": _grad_norm(self.tcn) if self.tcn is not None else 0.0,
        }

        if self.force_layout == "xy_yaw":
            stats["mae_tau_yaw"] = error[:, 2].mean().item()
        else:
            if self.force_dim >= 3:
                stats["mae_z"] = error[:, 2].mean().item()
            if self.force_dim >= 4:
                stats["mae_tau_yaw"] = error[:, 3].mean().item()
            if self.force_dim >= 6:
                stats["mae_tau_roll"] = error[:, 3].mean().item()
                stats["mae_tau_pitch"] = error[:, 4].mean().item()
                stats["mae_tau_yaw"] = error[:, 5].mean().item()

        return stats


def _grad_norm(module: nn.Module) -> float:
    """Compute total L2 gradient norm of a module's parameters."""
    total = 0.0
    for p in module.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return total ** 0.5
