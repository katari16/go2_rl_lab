"""Velocity + pose command generator for the 6Dctrl ablation.

Extends the standard 3-dim velocity command (vx, vy, ωz) with three pose channels
(roll, pitch, height) so the policy can be trained to track an arbitrary commanded
posture. Used at deploy time by mapping the wrench estimator's output to these
three extra channels (τ̂_roll → roll_cmd, τ̂_pitch → pitch_cmd, F̂_z → height_cmd).
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class UniformVelocityPoseCommand(UniformVelocityCommand):
    """Velocity command extended with absolute roll/pitch/height pose channels.

    Layout: [vx, vy, ωz, roll, pitch, height]  shape (num_envs, 6).
    """

    cfg: "UniformVelocityPoseCommandCfg"

    def __init__(self, cfg: "UniformVelocityPoseCommandCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        self.pose_command = torch.zeros(self.num_envs, 3, device=self.device)
        self.pose_command[:, 2] = cfg.nominal_height

    @property
    def command(self) -> torch.Tensor:
        return torch.cat([self.vel_command_b, self.pose_command], dim=-1)

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        r = torch.empty(len(env_ids), device=self.device)

        roll = r.uniform_(*self.cfg.ranges.roll)
        roll = torch.where(
            torch.empty_like(roll).uniform_(0.0, 1.0) < self.cfg.rel_nominal_roll,
            torch.zeros_like(roll), roll,
        )
        self.pose_command[env_ids, 0] = roll

        pitch = r.uniform_(*self.cfg.ranges.pitch)
        pitch = torch.where(
            torch.empty_like(pitch).uniform_(0.0, 1.0) < self.cfg.rel_nominal_pitch,
            torch.zeros_like(pitch), pitch,
        )
        self.pose_command[env_ids, 1] = pitch

        height = r.uniform_(*self.cfg.ranges.height)
        height = torch.where(
            torch.empty_like(height).uniform_(0.0, 1.0) < self.cfg.rel_nominal_height,
            torch.full_like(height, self.cfg.nominal_height), height,
        )
        self.pose_command[env_ids, 2] = height

    def _update_command(self):
        super()._update_command()
        # Standing envs hold nominal pose so the standing prior stays consistent.
        standing_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        if standing_ids.numel() > 0:
            self.pose_command[standing_ids, 0] = 0.0
            self.pose_command[standing_ids, 1] = 0.0
            self.pose_command[standing_ids, 2] = self.cfg.nominal_height


@configclass
class UniformVelocityPoseCommandCfg(UniformVelocityCommandCfg):
    """Config for :class:`UniformVelocityPoseCommand` — adds roll/pitch/height ranges."""

    class_type: type = UniformVelocityPoseCommand

    nominal_height: float = 0.34
    rel_nominal_roll: float = 0.20
    rel_nominal_pitch: float = 0.20
    rel_nominal_height: float = 0.20

    @configclass
    class Ranges(UniformVelocityCommandCfg.Ranges):
        roll: tuple[float, float] = MISSING
        pitch: tuple[float, float] = MISSING
        height: tuple[float, float] = MISSING

    ranges: Ranges = MISSING
