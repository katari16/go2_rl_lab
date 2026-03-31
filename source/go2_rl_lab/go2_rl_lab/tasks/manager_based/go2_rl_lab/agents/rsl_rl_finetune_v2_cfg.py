"""Agent configs for V2 finetuning experiments (4 jobs).

All use CompliantOnPolicyRunner with same architecture/hyperparameters.
20k iterations from scratch.
"""

from isaaclab.utils import configclass

from .rsl_rl_compliant_no_foot_xyz_cfg import CompliantNoFootXyzRunnerCfg


def _make_v2_cfg(variant: int, description: str):
    """Create a V2 finetune runner config."""

    @configclass
    class Cfg(CompliantNoFootXyzRunnerCfg):
        __doc__ = f"V{variant}: {description}"
        experiment_name: str = f"go2_finetune_v{variant}"
        max_iterations: int = 20000
        save_interval: int = 500

    Cfg.__name__ = f"FinetuneV{variant}RunnerCfg"
    Cfg.__qualname__ = f"FinetuneV{variant}RunnerCfg"
    return Cfg


FinetuneV1RunnerCfg = _make_v2_cfg(1, "Low gains + R2 + standing pose (flat)")
FinetuneV2RunnerCfg = _make_v2_cfg(2, "Low gains + R2 + standing pose + hip penalty (flat)")
FinetuneV3RunnerCfg = _make_v2_cfg(3, "Low gains + R2 + standing pose (rough)")
FinetuneV4RunnerCfg = _make_v2_cfg(4, "Low gains + R2 + standing pose + hip penalty (rough)")


@configclass
class FinetunePaceRunnerCfg(CompliantNoFootXyzRunnerCfg):
    """PACE: Low gains (Kp=8, Kd=0.4) + PACE actuator + R2 + standing pose (rough)."""
    experiment_name: str = "go2_finetune_pace"
    max_iterations: int = 20000
    save_interval: int = 500
