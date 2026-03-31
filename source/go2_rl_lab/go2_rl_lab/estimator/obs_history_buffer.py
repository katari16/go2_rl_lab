"""Rolling observation history buffer for the state estimation network."""

from __future__ import annotations

import torch


class ObsHistoryBuffer:
    """Fixed-length rolling window of single-step observations.

    Stores the last `temporal_steps` observations for every environment.
    Oldest observations are dropped when a new one is inserted.

    Shape conventions:
        buffer:             [num_envs, temporal_steps, obs_dim]
        get_flattened():    [num_envs, temporal_steps * obs_dim]
    """

    def __init__(self, num_envs: int, temporal_steps: int, obs_dim: int, device: str | torch.device) -> None:
        self.num_envs = num_envs
        self.temporal_steps = temporal_steps
        self.obs_dim = obs_dim
        self.device = device
        self.buffer = torch.zeros(num_envs, temporal_steps, obs_dim, device=device)

    def insert(self, obs: torch.Tensor) -> None:
        """Push a new single-step observation [num_envs, obs_dim] into the buffer."""
        self.buffer = torch.roll(self.buffer, shifts=-1, dims=1)
        self.buffer[:, -1, :] = obs.to(self.device)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Zero-out the history for terminated/reset environments."""
        if len(env_ids) > 0:
            self.buffer[env_ids] = 0.0

    def get_flattened(self) -> torch.Tensor:
        """Return [num_envs, temporal_steps * obs_dim] — input for the estimator encoder."""
        return self.buffer.reshape(self.num_envs, -1)
