"""Teacher-Student Force Estimator — PAINT-style distillation.

Architecture (PAINT, RA-L 2026):

    Teacher (privileged):
        [o_t, f_gt] ──► MLP [128, 64] ──► z_teacher (64-dim)
                                                ├── f_head [32, 16] ──► f̂_t
                                                └── decoder [256, 128] ──► ô_{t+1}

    Student (proprioceptive only):
        o_t^H ──► Encoder [128, 64] ──► z_student (64-dim)
                                              ├── f_head [32, 16] ──► f̂_t  (shared with teacher)
                                              └── KL distillation to z_teacher

Training:
    Phase A: Train teacher with GT force (supervised, same as ForceEstimator).
    Phase B: Freeze teacher. Train student encoder via:
        L_student = L_force(student) + λ_kl * KL(z_teacher || z_student)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from go2_rl_lab.estimator.force_estimator import _get_activation, _build_mlp, _grad_norm


class TeacherStudentEstimator(nn.Module):
    """Teacher-student force estimator with KL distillation.

    The teacher sees [o_t, f_gt] and learns a privileged latent.
    The student sees obs history and distills from the teacher's latent.
    The f_head is shared: both teacher and student latents are decoded
    through the same force prediction head.

    Args:
        temporal_steps:     Number of historical timesteps for student (H).
        num_one_step_obs:   Dimension of a single proprioceptive step.
        force_dim:          Force output dimension (3 for XYZ, 4 for XYZ+τ_yaw).
        enc_hidden_dims:    Encoder hidden widths (used for both teacher and student).
        f_head_dims:        Shared force head hidden widths.
        dec_hidden_dims:    Teacher decoder hidden widths.
        activation:         Activation name.
        teacher_lr:         Teacher optimizer LR.
        student_lr:         Student optimizer LR.
        force_loss_weight:  Weight for force MSE loss.
        angle_loss_weight:  Weight for angular MSE loss.
        rec_loss_weight:    Weight for teacher reconstruction loss.
        kl_loss_weight:     Weight for KL distillation loss.
        angle_min_force:    Min GT force magnitude to apply angular loss.
        max_grad_norm:      Gradient-clipping norm.
    """

    def __init__(
        self,
        temporal_steps: int,
        num_one_step_obs: int,
        force_dim: int = 3,
        enc_hidden_dims: list[int] | None = None,
        f_head_dims: list[int] | None = None,
        dec_hidden_dims: list[int] | None = None,
        activation: str = "elu",
        teacher_lr: float = 1e-3,
        student_lr: float = 1e-3,
        force_loss_weight: float = 1.0,
        angle_loss_weight: float = 3.0,
        rec_loss_weight: float = 1.0,
        kl_loss_weight: float = 1.0,
        angle_min_force: float = 1.0,
        max_grad_norm: float = 10.0,
        **kwargs,
    ) -> None:
        if kwargs:
            print(f"[TeacherStudentEstimator] Ignoring unexpected kwargs: {list(kwargs.keys())}")
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
        self.force_dim = force_dim
        self.enc_latent_dim = enc_hidden_dims[-1]
        self.latent_dim = self.force_dim + self.enc_latent_dim

        # Loss weights
        self.force_loss_weight = force_loss_weight
        self.angle_loss_weight = angle_loss_weight
        self.rec_loss_weight = rec_loss_weight
        self.kl_loss_weight = kl_loss_weight
        self.angle_min_force = angle_min_force
        self.max_grad_norm = max_grad_norm

        # ── Teacher encoder: [o_t, f_gt] → z_teacher ────────────────────
        teacher_input_dim = num_one_step_obs + force_dim
        self.teacher_encoder = _build_mlp(
            [teacher_input_dim] + enc_hidden_dims, act_fn
        )

        # ── Student encoder: o_t^H → z_student ──────────────────────────
        student_input_dim = temporal_steps * num_one_step_obs
        self.student_encoder = _build_mlp(
            [student_input_dim] + enc_hidden_dims, act_fn
        )

        # ── Shared force head: z → f̂ ────────────────────────────────────
        self.f_head = _build_mlp(
            [self.enc_latent_dim] + f_head_dims + [force_dim], act_fn
        )

        # ── Teacher decoder: l_t → ô_{t+1} ──────────────────────────────
        self.decoder = _build_mlp(
            [self.latent_dim] + dec_hidden_dims + [num_one_step_obs], act_fn
        )

        # ── Optimizers ───────────────────────────────────────────────────
        # Teacher: encoder + f_head + decoder
        self.teacher_optimizer = optim.Adam(
            list(self.teacher_encoder.parameters())
            + list(self.f_head.parameters())
            + list(self.decoder.parameters()),
            lr=teacher_lr,
        )
        # Student: student encoder only (f_head frozen during distillation)
        self.student_optimizer = optim.Adam(
            self.student_encoder.parameters(),
            lr=student_lr,
        )

        self._training_phase = "teacher"  # "teacher" or "student"

    @property
    def training_phase(self) -> str:
        return self._training_phase

    def set_student_phase(self) -> None:
        """Switch to student distillation phase. Freezes teacher."""
        self._training_phase = "student"
        # Freeze teacher encoder and decoder
        for p in self.teacher_encoder.parameters():
            p.requires_grad = False
        for p in self.decoder.parameters():
            p.requires_grad = False
        # Freeze f_head — student only learns the encoder, not the head
        for p in self.f_head.parameters():
            p.requires_grad = False
        print("[TeacherStudentEstimator] Switched to STUDENT phase — teacher frozen.")

    # ── Inference ────────────────────────────────────────────────────────

    @torch.no_grad()
    def get_latent(self, obs_history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Student inference: obs_history → (force_hat, latent).

        Always uses the student encoder at inference time.
        """
        z_student = self.student_encoder(obs_history)
        force_hat = self.f_head(z_student)
        latent = torch.cat([force_hat, z_student], dim=-1)
        return force_hat, latent

    @torch.no_grad()
    def get_teacher_latent(
        self, obs: torch.Tensor, gt_force: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Teacher inference: [obs, gt_force] → (force_hat, latent)."""
        teacher_input = torch.cat([obs, gt_force], dim=-1)
        z_teacher = self.teacher_encoder(teacher_input)
        force_hat = self.f_head(z_teacher)
        latent = torch.cat([force_hat, z_teacher], dim=-1)
        return force_hat, latent

    # ── Teacher training step ────────────────────────────────────────────

    def update_teacher(
        self,
        obs: torch.Tensor,
        gt_force: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> dict[str, float]:
        """One gradient step for the teacher (supervised).

        Args:
            obs:       [batch, num_one_step_obs]  — current proprioceptive obs
            gt_force:  [batch, force_dim]          — ground-truth force
            next_obs:  [batch, num_one_step_obs]   — target for reconstruction
        """
        teacher_input = torch.cat([obs, gt_force], dim=-1)
        z_teacher = self.teacher_encoder(teacher_input)
        force_hat = self.f_head(z_teacher)

        force_loss = F.mse_loss(force_hat, gt_force)

        # Reconstruction
        latent = torch.cat([force_hat, z_teacher], dim=-1)
        next_obs_hat = self.decoder(latent)
        rec_loss = F.mse_loss(next_obs_hat, next_obs)

        # Angular loss (XY plane)
        angle_loss, angle_diff, mask = self._angular_loss(force_hat, gt_force, obs.device)

        total_loss = (
            self.force_loss_weight * force_loss
            + self.angle_loss_weight * angle_loss
            + self.rec_loss_weight * rec_loss
        )

        self.teacher_optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.teacher_encoder.parameters())
            + list(self.f_head.parameters())
            + list(self.decoder.parameters()),
            self.max_grad_norm,
        )
        self.teacher_optimizer.step()

        return self._build_stats(
            force_hat, gt_force, force_loss, angle_loss, rec_loss, total_loss,
            angle_diff, mask, prefix="teacher",
        )

    # ── Student distillation step ────────────────────────────────────────

    def update_student(
        self,
        obs_history: torch.Tensor,
        obs: torch.Tensor,
        gt_force: torch.Tensor,
    ) -> dict[str, float]:
        """One gradient step for the student (distillation from frozen teacher).

        Args:
            obs_history: [batch, temporal_steps * num_one_step_obs]
            obs:         [batch, num_one_step_obs]  — for teacher input
            gt_force:    [batch, force_dim]          — for teacher input + force loss
        """
        # Teacher forward (no grad — frozen)
        with torch.no_grad():
            teacher_input = torch.cat([obs, gt_force], dim=-1)
            z_teacher = self.teacher_encoder(teacher_input)

        # Student forward
        z_student = self.student_encoder(obs_history)
        force_hat = self.f_head(z_student)

        # Force loss (student prediction vs GT)
        force_loss = F.mse_loss(force_hat, gt_force)

        # KL distillation: MSE between latent representations
        # Using MSE as a simpler proxy for KL (both are deterministic encoders)
        kl_loss = F.mse_loss(z_student, z_teacher)

        # Angular loss
        angle_loss, angle_diff, mask = self._angular_loss(force_hat, gt_force, obs_history.device)

        total_loss = (
            self.force_loss_weight * force_loss
            + self.angle_loss_weight * angle_loss
            + self.kl_loss_weight * kl_loss
        )

        self.student_optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.student_encoder.parameters(), self.max_grad_norm)
        self.student_optimizer.step()

        stats = self._build_stats(
            force_hat, gt_force, force_loss, angle_loss,
            torch.tensor(0.0), total_loss,
            angle_diff, mask, prefix="student",
        )
        stats["kl_loss"] = kl_loss.item()
        stats["grad_norm_student_encoder"] = _grad_norm(self.student_encoder)
        return stats

    # ── Helpers ──────────────────────────────────────────────────────────

    def _angular_loss(
        self, force_hat: torch.Tensor, gt_force: torch.Tensor, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute angular loss on XY plane. Returns (loss, angle_diff, mask)."""
        gt_angle = torch.atan2(gt_force[:, 1], gt_force[:, 0])
        pred_angle = torch.atan2(force_hat[:, 1], force_hat[:, 0])
        angle_diff = torch.atan2(
            torch.sin(gt_angle - pred_angle),
            torch.cos(gt_angle - pred_angle),
        )
        gt_mag_xy = gt_force[:, :2].norm(dim=-1)
        mask = gt_mag_xy > self.angle_min_force
        if mask.any():
            loss = (angle_diff[mask] ** 2).mean()
        else:
            loss = torch.tensor(0.0, device=device)
        return loss, angle_diff, mask

    def _build_stats(
        self,
        force_hat: torch.Tensor,
        gt_force: torch.Tensor,
        force_loss: torch.Tensor,
        angle_loss: torch.Tensor,
        rec_loss: torch.Tensor,
        total_loss: torch.Tensor,
        angle_diff: torch.Tensor,
        mask: torch.Tensor,
        prefix: str = "",
    ) -> dict[str, float]:
        with torch.no_grad():
            gt_mag = gt_force.norm(dim=-1)
            pred_mag = force_hat.norm(dim=-1)
            error = (force_hat - gt_force).abs()
            angle_err_deg = angle_diff.abs() * (180.0 / torch.pi)
            if mask.any():
                mean_angle = angle_err_deg[mask].mean().item()
                median_angle = angle_err_deg[mask].median().item()
            else:
                mean_angle = median_angle = 0.0

        stats = {
            "force_loss": force_loss.item(),
            "angle_loss": angle_loss.item(),
            "rec_loss": rec_loss.item(),
            "total_loss": total_loss.item(),
            "gt_force_mean_mag": gt_mag.mean().item(),
            "pred_force_mean_mag": pred_mag.mean().item(),
            "mae_x": error[:, 0].mean().item(),
            "mae_y": error[:, 1].mean().item(),
            "mae_total": error.mean().item(),
            "angle_err_mean_deg": mean_angle,
            "angle_err_median_deg": median_angle,
        }
        if self.force_dim >= 3:
            stats["mae_z"] = error[:, 2].mean().item()
        if self.force_dim >= 4:
            stats["mae_tau_yaw"] = error[:, 3].mean().item()
        return stats
