"""Payload compensation env config.

Inherits from LowLevelEnvCfg (V3) and swaps the robot to Go2 with a 3kg
payload link fixed-jointed to the base. Everything else (observations,
rewards, terrain, PD gains) is identical to the standard low-level config.

The payload link creates an asymmetric torque on the base due to its offset
from the CoM, which the policy must learn to compensate for.
"""

from isaaclab.utils import configclass

from .go2_lowlevel_env_cfg import LowLevelEnvCfg
from go2_rl_lab.assets.unitree import UNITREE_GO2_PAYLOAD_CFG


@configclass
class LowLevelPayloadEnvCfg(LowLevelEnvCfg):
    """Go2 low-level locomotion with 3kg payload on base.

    Same as LowLevelEnvCfg but with UNITREE_GO2_PAYLOAD_CFG robot
    (Go2 + 3kg cube payload link attached via fixed joint).
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = UNITREE_GO2_PAYLOAD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
