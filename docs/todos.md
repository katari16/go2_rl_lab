# Velocity Estimation Network for Robo-Barrow: Implementation Guide

## 1. Goal & Motivation

We are building a **learned base velocity estimator** for a wheeled-legged robot (Robo-Barrow) trained in Isaac Lab using reinforcement learning. The estimator replaces a traditional Kalman filter for base velocity estimation and is the **first step** toward a full HAC-LOCO-style hierarchical compliance controller.

### The Problem

On real robots, the IMU provides angular velocity and linear acceleration, and joint encoders provide joint positions and velocities. However, **base linear velocity is not directly measurable**. Traditionally, a Kalman filter fuses IMU acceleration with leg kinematics to estimate this. A learned estimator can be more robust — especially during wheel slippage, uneven terrain, or dynamic contact changes — because it learns from diverse simulated conditions.

### The Approach: Teacher-Student with Explicit State Estimation

During **training in simulation**, the simulator provides ground-truth base velocity (privileged information). We use this to train an estimation network via supervised learning, concurrently with the RL policy training. During **deployment on the real robot**, only the trained estimator and policy are used — no simulator, no Kalman filter.

The architecture follows the HAC-LOCO paper (Zhou et al., 2025, arXiv:2507.02447) Stage 1 design, simplified to velocity-only estimation (force estimation will be added later).

### Roadmap

1. **Now:** Train a velocity estimation network (this document)
2. **Next:** Extend estimator to also predict external forces (HAC-LOCO f_head)
3. **Later:** Add high-level compliance action module (HAC-LOCO Stage 2)

---

## 2. Architecture Overview

### Data Flow During Training (Simulation)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ISAAC LAB SIMULATION                                │
│                                                                             │
│  Privileged (sim-only):         Proprioceptive (available on real robot):  │
│  ┌──────────────────────┐       ┌────────────────────────────────────────┐  │
│  │ gt_base_velocity (3) │       │ angular_velocity (3)    ← IMU gyro    │  │
│  │ gt_contact_forces    │       │ projected_gravity (3)   ← IMU accel   │  │
│  │ terrain_friction     │       │ velocity_commands (3)   ← user input  │  │
│  │ terrain_heights      │       │ joint_positions (12)    ← encoders    │  │
│  │ ...                  │       │ joint_velocities (12)   ← encoders    │  │
│  └──────────┬───────────┘       │ last_actions (12)       ← from policy │  │
│             │                   └───────────────┬────────────────────────┘  │
│             │                                   │                           │
│             │                          o_t (single step, ~45 dims)          │
│             │                                   │                           │
│             │                    ┌──────────────▼──────────────────┐        │
│             │                    │  Observation History Buffer     │        │
│             │                    │  o_t^H = [o_{t-H+1}, ..., o_t] │        │
│             │                    │  flattened: H × 45 dims         │        │
│             │                    └──────────────┬─────────────────┘         │
│             │                                   │                           │
│             │                    ┌──────────────▼──────────────────┐        │
│             │                    │       ESTIMATOR NETWORK         │        │
│             │                    │                                  │        │
│             │                    │  Encoder [256,128,64]            │        │
│             │                    │       ↓                          │        │
│             │                    │      z_t (64 dims)              │        │
│             │                    │       ↓                          │        │
│             │                    │  v_head [32,16] → v̂_t (3 dims) │        │
│             │                    │       ↓                          │        │
│             │                    │  l_t = concat(v̂_t, z_t) = 67d  │        │
│             │                    │       ↓                          │        │
│             │                    │  Decoder [512,256,128] → ô_{t+1}│        │
│             │                    └──────┬────────────┬─────────────┘        │
│             │                           │            │                      │
│             │                    ┌──────▼────┐ ┌─────▼──────────┐           │
│             │                    │  vel_hat  │ │   latent l_t   │           │
│             │                    └──────┬────┘ └─────┬──────────┘           │
│             │                           │            │                      │
│  ┌──────────▼──────────────┐     SUPERVISED LOSSES:  │                     │
│  │ Supervised Learning     │     vel_loss = MSE(v̂_t, gt_vel)              │
│  │ gt_vel is used as       │     rec_loss = MSE(ô_{t+1}, o_{t+1})         │
│  │ regression target for   │            │                                   │
│  │ the estimator           │            │                                   │
│  └─────────────────────────┘            │                                   │
│                                         │                                   │
│  ┌──────────────────────────────────────▼───────────────────────────┐       │
│  │                    LOCOMOTION POLICY [512,256,128]                │       │
│  │                                                                   │       │
│  │  Input: concat(o_t, l_t) = concat(45, 67) = 112 dims            │       │
│  │  Output: a_t (joint position targets)                             │       │
│  └───────────────────────────────────┬───────────────────────────────┘       │
│                                      │                                      │
│  ┌───────────────────────────────────▼───────────────────────────────┐      │
│  │              CRITIC (uses privileged obs o_t^p)                    │      │
│  │  Has access to gt_velocity, terrain info, contacts, etc.          │      │
│  │  Used only for PPO value estimation during training               │      │
│  └───────────────────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow During Deployment (Real Robot)

```
  IMU + Encoders → o_t → History Buffer → o_t^H
                    │                        │
                    │                 Estimator → vel_hat, l_t
                    │                                │
                    └──────── concat ─────────────────┘
                                    │
                              Policy → a_t → Robot
  
  No simulator. No Kalman filter. No critic. No privileged info.
```

### Key Design Decisions

1. **Two parallel paths, not serial:** The current observation o_t goes directly to the policy untouched. The estimator runs in parallel on the history and produces l_t which is concatenated alongside o_t. The policy sees `[o_t, l_t]`.

2. **No lag from the estimator:** The history buffer is a rolling window updated every timestep. At timestep t, o_t is already in the buffer. The estimator MLP runs in the same forward pass as the policy (microseconds on GPU).

3. **Gradient isolation:** The estimator returns `.detach()` outputs to the policy. The policy cannot push gradients back into the estimator. They train concurrently but independently — the estimator via supervised learning, the policy via PPO.

4. **Supervised target from privileged/critic observations:** During training, ground-truth base velocity is available from the simulator. This is the same data the critic has access to. The estimator's `update()` method pulls `gt_velocity` from the critic observation buffer and uses it as the MSE regression target.

---

## 3. Gradient Flow During Estimator Training

When `estimator.update()` is called, `total_loss = vel_loss + rec_loss` is backpropagated. Gradients flow through:

```
total_loss = vel_loss + rec_loss
                │            │
           ┌────▼────┐  ┌───▼──────┐
           │ vel_hat  │  │next_obs_hat│
           └────┬────┘  └───┬──────┘
                │            │
           ┌────▼────┐  ┌───▼──────┐
 grads ←── │ v_head  │  │ Decoder  │  ← grads
           └────┬────┘  └───┬──────┘
                │            │
                │       l_t = cat(vel_hat, z_t)
                │           /           \
                │      vel_hat          z_t
                │          │             │
                └────┬─────┘             │
                     │                   │
                ┌────▼───────────────────▼────┐
  grads ←────── │         Encoder             │
                └────┬────────────────────────┘
                     │
                obs_history (input, no gradients beyond here)
```

**vel_loss** updates: `v_head` and `encoder` (to produce z_t from which velocity can be extracted)

**rec_loss** updates: `decoder`, `v_head` (since vel_hat is part of l_t), and `encoder` (since z_t is part of l_t)

The reconstruction loss acts as a **regularizer** on the encoder — it forces z_t to contain meaningful dynamics information beyond just velocity, because z_t must also enable prediction of the next observation. This prevents the encoder from collapsing to a trivial representation.

---

## 4. Reference Implementation: HIM Loco

The HIM Loco paper (Long et al., 2024, "Hybrid Internal Model: Learning Agile Legged Locomotion with Simulated Robot Response") provides an open-source codebase that implements a similar architecture. Their code is at: **https://github.com/OpenRobotLab/HIMLoco**

The key file is `rsl_rl/rsl_rl/algorithms/him_ppo.py` which contains the `HIMEstimator` class. Their approach differs from ours in one key way: instead of using a decoder for reconstruction, they use **SwAV-style contrastive learning** (prototypes + Sinkhorn-Knopp normalization) to train the implicit latent z_t. The explicit velocity estimation part is identical.

### HIM Loco Original Estimator Code

Below is the original `HIMEstimator` class from the HIM Loco codebase. This is the reference we are adapting from:

```python
# ══════════════════════════════════════════════════════════════════════════════
# SOURCE: HIMLoco / rsl_rl/rsl_rl/algorithms/him_ppo.py
# REPO:   https://github.com/OpenRobotLab/HIMLoco
# ══════════════════════════════════════════════════════════════════════════════

# PASTE THE ORIGINAL HIMEstimator CLASS HERE:

import copy
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as torchd
from torch.distributions import Normal, Categorical


class HIMEstimator(nn.Module):
    def __init__(self,
                 temporal_steps,
                 num_one_step_obs,
                 enc_hidden_dims=[128, 64, 16],
                 tar_hidden_dims=[128, 64],
                 activation='elu',
                 learning_rate=1e-3,
                 max_grad_norm=10.0,
                 num_prototype=32,
                 temperature=3.0,
                 **kwargs):
        if kwargs:
            print("Estimator_CL.__init__ got unexpected arguments, which will be ignored: " + str(
                [key for key in kwargs.keys()]))
        super(HIMEstimator, self).__init__()
        activation = get_activation(activation)

        self.temporal_steps = temporal_steps
        self.num_one_step_obs = num_one_step_obs
        self.num_latent = enc_hidden_dims[-1]
        self.max_grad_norm = max_grad_norm
        self.temperature = temperature

        # Encoder
        enc_input_dim = self.temporal_steps * self.num_one_step_obs
        enc_layers = []
        for l in range(len(enc_hidden_dims) - 1):
            enc_layers += [nn.Linear(enc_input_dim, enc_hidden_dims[l]), activation]
            enc_input_dim = enc_hidden_dims[l]
        enc_layers += [nn.Linear(enc_input_dim, enc_hidden_dims[-1] + 3)]
        self.encoder = nn.Sequential(*enc_layers)

        # Target
        tar_input_dim = self.num_one_step_obs
        tar_layers = []
        for l in range(len(tar_hidden_dims)):
            tar_layers += [nn.Linear(tar_input_dim, tar_hidden_dims[l]), activation]
            tar_input_dim = tar_hidden_dims[l]
        tar_layers += [nn.Linear(tar_input_dim, enc_hidden_dims[-1])]
        self.target = nn.Sequential(*tar_layers)

        # Prototype
        self.proto = nn.Embedding(num_prototype, enc_hidden_dims[-1])

        # Optimizer
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def get_latent(self, obs_history):
        vel, z = self.encode(obs_history)
        return vel.detach(), z.detach()

    def forward(self, obs_history):
        parts = self.encoder(obs_history.detach())
        vel, z = parts[..., :3], parts[..., 3:]
        z = F.normalize(z, dim=-1, p=2)
        return vel.detach(), z.detach()

    def encode(self, obs_history):
        parts = self.encoder(obs_history.detach())
        vel, z = parts[..., :3], parts[..., 3:]
        z = F.normalize(z, dim=-1, p=2)
        return vel, z

    def update(self, obs_history, next_critic_obs, lr=None):
        if lr is not None:
            self.learning_rate = lr
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.learning_rate
                
        vel = next_critic_obs[:, self.num_one_step_obs:self.num_one_step_obs+3].detach()
        next_obs = next_critic_obs.detach()[:, 3:self.num_one_step_obs+3]

        z_s = self.encoder(obs_history)
        z_t = self.target(next_obs)
        pred_vel, z_s = z_s[..., :3], z_s[..., 3:]

        z_s = F.normalize(z_s, dim=-1, p=2)
        z_t = F.normalize(z_t, dim=-1, p=2)

        with torch.no_grad():
            w = self.proto.weight.data.clone()
            w = F.normalize(w, dim=-1, p=2)
            self.proto.weight.copy_(w)

        score_s = z_s @ self.proto.weight.T
        score_t = z_t @ self.proto.weight.T

        with torch.no_grad():
            q_s = sinkhorn(score_s)
            q_t = sinkhorn(score_t)

        log_p_s = F.log_softmax(score_s / self.temperature, dim=-1)
        log_p_t = F.log_softmax(score_t / self.temperature, dim=-1)

        swap_loss = -0.5 * (q_s * log_p_t + q_t * log_p_s).mean()
        estimation_loss = F.mse_loss(pred_vel, vel)
        losses = estimation_loss + swap_loss

        self.optimizer.zero_grad()
        losses.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return estimation_loss.item(), swap_loss.item()


@torch.no_grad()
def sinkhorn(out, eps=0.05, iters=3):
    Q = torch.exp(out / eps).T
    K, B = Q.shape[0], Q.shape[1]
    Q /= Q.sum()

    for it in range(iters):
        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= K
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B
    return (Q * B).T


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "silu":
        return nn.SiLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
```

### What the HIM Loco Code Does (Component Breakdown)

| Component | Purpose | Architecture |
|-----------|---------|-------------|
| `self.encoder` | Compresses proprioceptive history into latent z_t + velocity | MLP: `(H×obs_dim)` → `[128, 64]` → `(16 + 3)` |
| Output split `[:3]` / `[3:]` | Separates explicit velocity (3D) from implicit latent (16D) | Last layer outputs 19 dims, manually split |
| `F.normalize(z, ...)` | L2-normalizes latent to unit hypersphere | Required for contrastive learning |
| `self.target` | Encodes NEXT single-step obs into same latent space | MLP: `obs_dim` → `[128, 64]` → `16` |
| `self.proto` | Learnable prototype vectors for SwAV contrastive loss | Embedding: `32 × 16` |
| `sinkhorn()` | Computes balanced soft cluster assignments | Sinkhorn-Knopp algorithm |
| `swap_loss` | Contrastive: encoder latent should predict target's cluster assignment and vice versa | Cross-entropy on swapped assignments |
| `estimation_loss` | Supervised: predicted velocity should match ground truth | MSE(v̂_t, gt_velocity) |

---

## 5. Our Proposed Changes: HAC-LOCO-Style Velocity Estimator

We replace HIM's contrastive learning (target network + prototypes + Sinkhorn) with HAC-LOCO's **autoencoder reconstruction** approach. This is simpler, requires less hyperparameter tuning, and the decoder provides a clear regularization signal.

### Key Differences from HIM Loco

| Aspect | HIM Loco | Our Version (HAC-LOCO style) |
|--------|----------|------------------------------|
| **Encoder output** | Single MLP outputs vel + z concatenated (19D) | Encoder outputs z_t (64D), separate v_head outputs vel (3D) |
| **Latent z_t training** | Contrastive learning (SwAV prototypes + Sinkhorn) | Reconstruction loss (decoder predicts next obs) |
| **Extra components** | Target network, prototype embeddings | Decoder network |
| **Latent normalization** | L2-normalized (unit sphere) | No normalization needed |
| **z_t dimensionality** | 16 | 64 (richer representation) |
| **Encoder dims** | [128, 64] → 19 | [256, 128, 64] (deeper) |
| **Loss** | MSE(vel) + SwAV contrastive | MSE(vel) + MSE(reconstruction) |

### Proposed Code

```python
"""
Velocity Estimator Network — HAC-LOCO Style (Velocity-Only)

Architecture (from HAC-LOCO Stage 1, adapted for velocity-only):
    o_t^H → Encoder [256,128,64] → z_t → v_head [32,16] → v̂_t (3D)
                                     ↓
                                    l_t = [v̂_t, z_t]
                                     ↓
                                  Decoder [512,256,128] → ô_{t+1}

Training losses:
    1. Velocity MSE:       L_vel = MSE(v̂_t, v_gt)
    2. Reconstruction MSE: L_rec = MSE(ô_{t+1}, o_{t+1})

The latent l_t is fed to the locomotion policy alongside proprioception o_t.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class VelocityEstimator(nn.Module):
    """
    HAC-LOCO-style estimator for base velocity prediction.

    Inputs:
        obs_history: [batch, temporal_steps * num_one_step_obs]
            Flattened proprioceptive history

    Outputs (inference):
        vel_hat: [batch, 3]           — estimated base linear velocity (vx, vy, vz)
        latent:  [batch, latent_dim]  — l_t = concat(vel_hat, z_t), fed to policy

    Args:
        temporal_steps:    Number of historical timesteps stacked
        num_one_step_obs:  Dimension of single-step proprioceptive observation
        enc_hidden_dims:   Encoder MLP hidden dims (default: [256, 128, 64])
        v_head_dims:       Velocity head dims (default: [32, 16])
        dec_hidden_dims:   Decoder MLP hidden dims (default: [512, 256, 128])
        activation:        Activation function name (default: 'elu')
        learning_rate:     Optimizer learning rate
        vel_loss_weight:   Weight for velocity estimation loss
        rec_loss_weight:   Weight for reconstruction loss
        max_grad_norm:     Gradient clipping norm
    """

    def __init__(
        self,
        temporal_steps: int,
        num_one_step_obs: int,
        enc_hidden_dims: list = [256, 128, 64],
        v_head_dims: list = [32, 16],
        dec_hidden_dims: list = [512, 256, 128],
        activation: str = "elu",
        learning_rate: float = 1e-3,
        vel_loss_weight: float = 1.0,
        rec_loss_weight: float = 1.0,
        max_grad_norm: float = 10.0,
        **kwargs,
    ):
        if kwargs:
            print(
                f"VelocityEstimator.__init__ got unexpected arguments "
                f"(ignored): {list(kwargs.keys())}"
            )
        super().__init__()

        act_fn = _get_activation(activation)

        self.temporal_steps = temporal_steps
        self.num_one_step_obs = num_one_step_obs
        self.enc_latent_dim = enc_hidden_dims[-1]  # z_t dimension
        self.vel_dim = 3  # vx, vy, vz
        self.latent_dim = self.vel_dim + self.enc_latent_dim  # l_t dimension
        self.vel_loss_weight = vel_loss_weight
        self.rec_loss_weight = rec_loss_weight
        self.max_grad_norm = max_grad_norm
        self.learning_rate = learning_rate

        # ── Encoder: o_t^H → z_t ──────────────────────────────────────
        # Input:  temporal_steps * num_one_step_obs
        # Output: enc_latent_dim (e.g. 64)
        enc_input_dim = temporal_steps * num_one_step_obs
        enc_layers = []
        in_dim = enc_input_dim
        for h in enc_hidden_dims:
            enc_layers += [nn.Linear(in_dim, h), act_fn]
            in_dim = h
        self.encoder = nn.Sequential(*enc_layers)

        # ── Velocity Head: z_t → v̂_t ──────────────────────────────────
        # Input:  enc_latent_dim (64)
        # Output: 3 (vx, vy, vz)
        v_layers = []
        in_dim = self.enc_latent_dim
        for h in v_head_dims:
            v_layers += [nn.Linear(in_dim, h), act_fn]
            in_dim = h
        v_layers += [nn.Linear(in_dim, self.vel_dim)]  # no activation on output
        self.v_head = nn.Sequential(*v_layers)

        # ── Decoder: l_t → ô_{t+1} ────────────────────────────────────
        # Input:  latent_dim = vel_dim + enc_latent_dim (3 + 64 = 67)
        # Output: num_one_step_obs (reconstruct next single-step obs)
        dec_layers = []
        in_dim = self.latent_dim
        for h in dec_hidden_dims:
            dec_layers += [nn.Linear(in_dim, h), act_fn]
            in_dim = h
        dec_layers += [nn.Linear(in_dim, num_one_step_obs)]  # no activation
        self.decoder = nn.Sequential(*dec_layers)

        # ── Optimizer ──────────────────────────────────────────────────
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    # ─── Inference (called during RL rollouts) ─────────────────────────

    def forward(self, obs_history: torch.Tensor):
        """
        Forward pass for inference (no gradients to policy).

        Returns:
            vel_hat: [batch, 3]           — estimated velocity (detached)
            latent:  [batch, latent_dim]  — l_t for policy input (detached)
        """
        z_t = self.encoder(obs_history.detach())
        vel_hat = self.v_head(z_t)
        latent = torch.cat([vel_hat, z_t], dim=-1)
        return vel_hat.detach(), latent.detach()

    def get_latent(self, obs_history: torch.Tensor):
        """Alias for forward — returns (vel_hat, latent), both detached."""
        return self.forward(obs_history)

    # ─── Training (called every N PPO iterations) ──────────────────────

    def update(
        self,
        obs_history: torch.Tensor,
        gt_velocity: torch.Tensor,
        next_obs: torch.Tensor,
        lr: float = None,
    ):
        """
        Single training step with supervised losses.

        Args:
            obs_history:  [batch, temporal_steps * num_one_step_obs]
            gt_velocity:  [batch, 3]  — ground-truth base velocity from sim
            next_obs:     [batch, num_one_step_obs] — next single-step obs

        Returns:
            vel_loss, rec_loss, total_loss (all floats)
        """
        if lr is not None:
            self.learning_rate = lr
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

        # Forward through encoder + heads
        z_t = self.encoder(obs_history.detach())
        vel_hat = self.v_head(z_t)
        latent = torch.cat([vel_hat, z_t], dim=-1)
        next_obs_hat = self.decoder(latent)

        # Losses
        vel_loss = F.mse_loss(vel_hat, gt_velocity.detach())
        rec_loss = F.mse_loss(next_obs_hat, next_obs.detach())
        total_loss = (
            self.vel_loss_weight * vel_loss + self.rec_loss_weight * rec_loss
        )

        # Backward
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return vel_loss.item(), rec_loss.item(), total_loss.item()


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
        raise ValueError(
            f"Unknown activation: {name}. Choose from {list(activations.keys())}"
        )
    return activations[name]
```

---

## 6. Integration into Isaac Lab / rsl_rl Training Loop

### Where the Estimator Fits in the Training Loop

The estimator must be integrated into the PPO training loop (typically in an `on_policy_runner.py` or similar). The pseudocode for one training iteration:

```python
# ═══ Inside the PPO training loop ═══

# 1. Collect rollouts
for step in range(num_steps_per_update):
    # Get current obs and obs history
    obs = env.get_observations()            # o_t, shape [num_envs, obs_dim]
    obs_history = env.get_obs_history()      # o_t^H, shape [num_envs, H * obs_dim]
    
    # Run estimator (no gradients — detached)
    vel_hat, latent = estimator.get_latent(obs_history)
    
    # Build policy input by concatenating
    policy_input = torch.cat([obs, latent], dim=-1)  # [num_envs, obs_dim + latent_dim]
    
    # Run policy
    actions = policy(policy_input)
    
    # Step environment
    next_obs, rewards, dones, infos = env.step(actions)
    
    # Store transition in buffer (including obs_history for estimator training)
    buffer.store(obs, obs_history, actions, rewards, dones, 
                 critic_obs=env.get_privileged_observations())

# 2. Update policy via PPO (standard)
policy_loss = ppo_update(buffer)

# 3. Update estimator via supervised learning
for batch in buffer.get_batches():
    # Extract ground-truth velocity from privileged/critic observations
    gt_velocity = batch.critic_obs[:, VEL_START:VEL_START+3]
    
    # Extract next observation for reconstruction target
    next_obs = batch.next_obs  # single-step, shape [batch, obs_dim]
    
    vel_loss, rec_loss, total = estimator.update(
        batch.obs_history, gt_velocity, next_obs
    )
```

### Things That Need to Be Adapted to the Specific Training Environment

The following items are **environment-specific** and must be matched to the actual Robo-Barrow Isaac Lab setup:

#### 6.1 Observation Dimensions

```python
# MUST MATCH your environment's observation space
num_one_step_obs = ???  # e.g., 45 for a standard quadruped:
                        #   angular_vel(3) + projected_gravity(3) + commands(3)
                        #   + joint_pos(12) + joint_vel(12) + last_actions(12)
                        #
                        # For Robo-Barrow this will be different depending on:
                        #   - Number of joints (legs + wheels)
                        #   - Whether wheel velocities are in obs
                        #   - Whether barrow handle force is in obs
                        #   - Any additional sensors
```

#### 6.2 History Length

```python
temporal_steps = ???  # How many past observations to stack
                      # HAC-LOCO uses ~50
                      # HIM Loco uses 50
                      # Start with 10-20 and increase if estimation is poor
                      # More history = better estimation but larger input
```

#### 6.3 Ground-Truth Velocity Location in Critic Observations

```python
# The ground-truth base velocity is somewhere in the critic/privileged observations.
# You need to know the exact index range. In HIM Loco's code:
vel = next_critic_obs[:, num_one_step_obs:num_one_step_obs+3]

# In your environment it might be at a different index.
# Check your environment config's privileged observation group.
# Common patterns:
#   critic_obs = [proprioceptive_obs, base_lin_vel, base_ang_vel, ...]
#   or
#   critic_obs = [base_lin_vel, base_ang_vel, proprioceptive_obs, ...]
```

#### 6.4 Observation History Buffer

```python
# Isaac Lab / rsl_rl may or may not provide an observation history buffer.
# If not, you need to implement one. Simple rolling buffer:

class ObsHistoryBuffer:
    def __init__(self, num_envs, temporal_steps, obs_dim, device):
        self.buffer = torch.zeros(
            num_envs, temporal_steps, obs_dim, device=device
        )
    
    def insert(self, obs):
        """Push new obs, drop oldest."""
        self.buffer = torch.roll(self.buffer, shifts=-1, dims=1)
        self.buffer[:, -1, :] = obs
    
    def get_flattened(self):
        """Return [num_envs, temporal_steps * obs_dim]."""
        return self.buffer.reshape(self.buffer.shape[0], -1)
    
    def reset(self, env_ids):
        """Reset buffer for specific environments that were reset."""
        self.buffer[env_ids] = 0.0
```

#### 6.5 Policy Network Input Dimension

```python
# The policy's input dimension must be increased to accommodate the latent:
# 
# BEFORE (no estimator):
#   policy_input_dim = num_one_step_obs  (e.g., 45)
#
# AFTER (with estimator):
#   policy_input_dim = num_one_step_obs + estimator.latent_dim  (e.g., 45 + 67 = 112)
#
# This means the policy network's first layer must be updated in the config.
# In rsl_rl, this is typically set in the environment config or agent config.
```

#### 6.6 Next Observation for Reconstruction Target

```python
# The decoder needs the NEXT single-step observation as the reconstruction target.
# During rollout collection, you need to store:
#   - obs_history at time t
#   - next_obs at time t+1 (the single-step obs AFTER taking action a_t)
#
# Make sure next_obs is the PROPRIOCEPTIVE obs (not privileged).
# It should have the same dimension as num_one_step_obs.
```

#### 6.7 When to Call estimator.update()

```python
# Option A: Every PPO update iteration (recommended, matches HIM Loco)
#   - After collecting rollouts and before/after PPO update
#   - Uses the same batch of data as PPO
#
# Option B: Every K PPO iterations (saves compute)
#   - Less frequent updates, estimator lags behind policy slightly
#
# Option C: After policy training is done (Ji et al. showed this works too)
#   - Train policy first with ground-truth velocity
#   - Then train estimator on collected data
#   - Simpler but less elegant
```

---

## 7. Future Extension: Adding Force Estimation (HAC-LOCO Stage 1 Complete)

When ready to add force estimation, the changes are minimal:

```python
# Add f_head alongside v_head:
self.f_head = nn.Sequential(
    nn.Linear(enc_latent_dim, 32), nn.ELU(),
    nn.Linear(32, 16), nn.ELU(),
    nn.Linear(16, 3),  # fx, fy, fz
)

# Expand latent:
# l_t = concat(vel_hat, force_hat, z_t) = 3 + 3 + 64 = 70 dims

# Add force loss in update():
# force_loss = F.mse_loss(force_hat, gt_force.detach())
# total_loss = vel_loss + force_loss + rec_loss

# Ground-truth forces come from the simulator's contact force reporting.
```

After that, HAC-LOCO Stage 2 adds a high-level compliance module that takes the estimated forces and modulates velocity commands — making the robot yield to sustained external forces (like a human pushing the barrow) while resisting transient disturbances.

---

## 8. References

1. **Ji et al. (2022)** — "Concurrent Training of a Control Policy and a State Estimator for Dynamic and Robust Legged Locomotion" — IEEE RA-L — [arXiv:2202.05481](https://arxiv.org/abs/2202.05481) — The foundational paper on concurrent policy + velocity estimator training.

2. **Long et al. (2024)** — "Hybrid Internal Model: Learning Agile Legged Locomotion with Simulated Robot Response" — ICLR 2024 — [arXiv:2312.11460](https://arxiv.org/abs/2312.11460) — Code: [github.com/OpenRobotLab/HIMLoco](https://github.com/OpenRobotLab/HIMLoco) — Our reference codebase. Uses contrastive learning for implicit latent.

3. **Zhou et al. (2025)** — "HAC-LOCO: Learning Hierarchical Active Compliance Control for Quadruped Locomotion under Continuous External Disturbances" — [arXiv:2507.02447](https://arxiv.org/abs/2507.02447) — The target architecture. Uses encoder + separate v_head/f_head + decoder.

4. **Isaac Lab Sim-to-Real Documentation** — Teacher-student distillation pipeline — [Isaac Lab Docs](https://isaac-sim.github.io/IsaacLab/main/source/experimental-features/newton-physics-integration/sim-to-real.html)

5. **CTS (2024)** — "Concurrent Teacher-Student Reinforcement Learning for Legged Locomotion" — [arXiv:2405.10830](https://arxiv.org/abs/2405.10830) — Compares EstimatorNet vs teacher-student distillation.

6. **SLR (2024)** — "Learning Quadruped Locomotion without Privileged Information" — CoRL 2024 — [arXiv:2406.04835](https://arxiv.org/abs/2406.04835) — Code: [github.com/11chens/SLR-master](https://github.com/11chens/SLR-master) — Benchmarks multiple estimation approaches.