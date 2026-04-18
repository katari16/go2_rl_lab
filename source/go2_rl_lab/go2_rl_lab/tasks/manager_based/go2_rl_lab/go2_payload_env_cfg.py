"""Payload compensation env config with randomized mass.

Inherits from LowLevelWrenchTrapezoidEnvCfg (PAINT-style force profile) and swaps
the robot to Go2 with a 3kg payload link fixed-jointed to the base. Adds a mass
randomization event that samples uniform 0-4kg per episode reset.

Used by P18 ablation run.
"""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .go2_ablation_env_cfgs import LowLevelWrenchTrapezoidEnvCfg, TrapezoidWrenchEventCfg
from go2_rl_lab.assets.unitree import UNITREE_GO2_PAYLOAD_CFG


@configclass
class PayloadEventCfg(TrapezoidWrenchEventCfg):
    """PAINT wrench event + randomized payload mass (0-4kg per episode)."""

    randomize_payload_mass = EventTerm(
        func=lambda env, env_ids, asset_cfg, mass_range: setattr(
            env.scene[asset_cfg.name].root_physx_view,
            "masses",
            env.scene[asset_cfg.name].root_physx_view.get_masses().clone().scatter_(
                1,
                env_ids.unsqueeze(1) * env.scene[asset_cfg.name].num_bodies + (env.scene[asset_cfg.name].num_bodies - 1),
                (mass_range[0] + (mass_range[1] - mass_range[0]) * env.torch_rand(len(env_ids), 1, device=env.device)).expand(-1, 1),
            ),
        ),
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mass_range": (0.0, 4.0),
        },
    )


@configclass
class LowLevelPayloadEnvCfg(LowLevelWrenchTrapezoidEnvCfg):
    """Go2 low-level locomotion with PAINT wrench + randomized 0-4kg payload.

    Changes from LowLevelWrenchTrapezoidEnvCfg:
    - Robot: UNITREE_GO2_PAYLOAD_CFG (Go2 + 3kg baseline payload link)
    - Event: randomize_payload_mass (0-4kg uniform per reset)
    """

    events: PayloadEventCfg = PayloadEventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = UNITREE_GO2_PAYLOAD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
