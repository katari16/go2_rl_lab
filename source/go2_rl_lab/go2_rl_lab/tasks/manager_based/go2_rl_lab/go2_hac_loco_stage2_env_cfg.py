"""HAC-LOCO stage 2 env config — rough terrain (no stairs).

Inherits from the compliant no-foot XYZ env and enables terrain generation
with the same mild roughness/slopes used in V3 low-level finetuning.
This ensures the high-level compliance policy trains on the same terrain
distribution as the frozen low-level it rides on.

Robot uses the standard Go2 actuator (Kp=25) since we freeze the V3
low-level checkpoint at runtime — the low-level policy itself handles
the gains.
"""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

from . import mdp
from .go2_compliant_no_foot_xyz_env_cfg import (
    UnitreeGo2CompliantNoFootXyzEnvCfg,
)


@configclass
class TerrainCurriculumCfg:
    """Terrain curriculum — progress to harder terrain when walking well."""
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


@configclass
class HacLocoStage2EnvCfg(UnitreeGo2CompliantNoFootXyzEnvCfg):
    """Rough terrain variant for HAC-LOCO stage 2 sweep.

    Same obs/rewards/events as the flat compliant-no-foot-xyz env,
    but with generated terrain (boxes, random rough, slopes — no stairs).
    """

    curriculum: TerrainCurriculumCfg = TerrainCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        # Enable generated terrain
        self.scene.terrain.terrain_type = "generator"
        # Remove stairs, keep only mild roughness and gentle slopes
        tg = self.scene.terrain.terrain_generator
        if tg is not None:
            for stair_key in ["pyramid_stairs", "pyramid_stairs_inv"]:
                if stair_key in tg.sub_terrains:
                    del tg.sub_terrains[stair_key]
            if "boxes" in tg.sub_terrains:
                tg.sub_terrains["boxes"].proportion = 0.3
                tg.sub_terrains["boxes"].grid_height_range = (0.02, 0.08)
            if "random_rough" in tg.sub_terrains:
                tg.sub_terrains["random_rough"].proportion = 0.4
                tg.sub_terrains["random_rough"].noise_range = (0.01, 0.05)
                tg.sub_terrains["random_rough"].noise_step = 0.01
            if "hf_pyramid_slope" in tg.sub_terrains:
                tg.sub_terrains["hf_pyramid_slope"].proportion = 0.15
                tg.sub_terrains["hf_pyramid_slope"].slope_range = (0.0, 0.3)
            if "hf_pyramid_slope_inv" in tg.sub_terrains:
                tg.sub_terrains["hf_pyramid_slope_inv"].proportion = 0.15
                tg.sub_terrains["hf_pyramid_slope_inv"].slope_range = (0.0, 0.3)
