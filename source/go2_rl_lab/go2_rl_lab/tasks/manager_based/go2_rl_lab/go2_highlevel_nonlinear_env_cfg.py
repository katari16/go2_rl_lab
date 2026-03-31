"""High-level non-linear compliance env config.

Inherits directly from LowLevelEnvCfg (the V3 config) so the env matches
the frozen low-level policy: Kp=8, Kd=0.4, action_scale=0.5, rough terrain,
R2+pose rewards.

Since LowLevelEnvCfg already has rough terrain with no stairs and terrain
curriculum enabled, this config adds nothing — it exists as a semantic
alias to clarify that this is the high-level compliance training env.
"""

from isaaclab.utils import configclass

from .go2_lowlevel_env_cfg import LowLevelEnvCfg


@configclass
class HighLevelNonLinearEnvCfg(LowLevelEnvCfg):
    """High-level non-linear compliance env — same as the low-level V3 training env.

    The frozen low-level was trained with this exact config (gains, terrain,
    rewards). The high-level compliance policy trains on top.
    """
    pass
