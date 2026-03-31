"""Agent configs for low-level locomotion finetuning experiments (8 jobs).

All use CompliantOnPolicyRunner with same architecture/hyperparameters.
Only experiment_name and max_iterations differ.
Train for 25k iterations (longer than baseline 10k to let regularizers converge).
"""

from isaaclab.utils import configclass

from .rsl_rl_compliant_no_foot_xyz_cfg import CompliantNoFootXyzRunnerCfg


def _make_finetune_cfg(job_id: int, description: str):
    """Create a finetune runner config for a given job."""

    @configclass
    class Cfg(CompliantNoFootXyzRunnerCfg):
        __doc__ = f"Job {job_id}: {description}"
        experiment_name: str = f"go2_finetune_j{job_id}"
        max_iterations: int = 25000
        save_interval: int = 500

    Cfg.__name__ = f"FinetuneJ{job_id}RunnerCfg"
    Cfg.__qualname__ = f"FinetuneJ{job_id}RunnerCfg"
    return Cfg


FinetuneJ1RunnerCfg = _make_finetune_cfg(1, "Low gains (Kp=8) + R1 Baseline rewards")
FinetuneJ2RunnerCfg = _make_finetune_cfg(2, "Low gains (Kp=8) + R2 Enhanced Smoothness")
FinetuneJ3RunnerCfg = _make_finetune_cfg(3, "Low gains (Kp=8) + R3 Active Gait")
FinetuneJ4RunnerCfg = _make_finetune_cfg(4, "Low gains (Kp=8) + R4 Deep Compliant Style")
FinetuneJ5RunnerCfg = _make_finetune_cfg(5, "Baseline gains (Kp=25) + R1 Baseline rewards")
FinetuneJ6RunnerCfg = _make_finetune_cfg(6, "Baseline gains (Kp=25) + R2 Enhanced Smoothness")
FinetuneJ7RunnerCfg = _make_finetune_cfg(7, "Baseline gains (Kp=25) + R3 Active Gait")
FinetuneJ8RunnerCfg = _make_finetune_cfg(8, "Baseline gains (Kp=25) + R4 Deep Compliant Style")
