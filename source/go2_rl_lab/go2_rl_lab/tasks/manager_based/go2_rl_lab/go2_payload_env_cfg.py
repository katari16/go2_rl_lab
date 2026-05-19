"""Payload compensation env configs.

Two variants:
- LowLevelPayload3DEnvCfg: Simple 3D force (70-dim critic) for basic payload training
- LowLevelPayloadEnvCfg: 6D wrench + randomized mass (73-dim critic) for P18
"""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from . import mdp
from .go2_lowlevel_env_cfg import LowLevelEnvCfg
from .go2_ablation_env_cfgs import PSeriesWrenchEnvCfg, PSeriesWrenchEventCfg
from go2_rl_lab.assets.unitree import UNITREE_GO2_PAYLOAD_CFG


@configclass
class LowLevelPayload3DEnvCfg(LowLevelEnvCfg):
    """Go2 low-level locomotion with 3kg payload on base (3D force, 70-dim critic).

    Same as LowLevelEnvCfg but with UNITREE_GO2_PAYLOAD_CFG robot
    (Go2 + 3kg cube payload link attached via fixed joint).
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = UNITREE_GO2_PAYLOAD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class PayloadEventCfg(PSeriesWrenchEventCfg):
    """P-series wrench event + randomized payload mass (0-4kg per episode)."""

    randomize_payload_mass = EventTerm(
        func=mdp.randomize_payload_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mass_range": (0.0, 4.0),
        },
    )


@configclass
class LowLevelPayloadEnvCfg(PSeriesWrenchEnvCfg):
    """Go2 low-level locomotion with P-series wrench + randomized 0-4kg payload (6D wrench, 73-dim critic).

    Changes from PSeriesWrenchEnvCfg:
    - Robot: UNITREE_GO2_PAYLOAD_CFG (Go2 + 3kg baseline payload link)
    - Event: randomize_payload_mass (0-4kg uniform per reset)

    Used by P18 ablation run.
    """

    events: PayloadEventCfg = PayloadEventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = UNITREE_GO2_PAYLOAD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
