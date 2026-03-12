"""Agent configs for HAC-LOCO stage 2 sweep — 8 variations (R1-R8).

2x2x2 factorial:
  - Reward type: penalty vs positive
  - Gravity correction: off vs on
  - Tracking reward: on vs off

All share: w_comp=2.0, action_clip=3.0, rough terrain, 5k iterations.

Usage:
    python train.py --task Go2-HAC-LOCO-Stage2-R1-v0 \
        --stage1_checkpoint /path/to/stage1/model_XXXX.pt --headless
"""

from isaaclab.utils import configclass

from .rsl_rl_hac_loco_stage2_cfg import HacLocoStage2RunnerCfg

# Base compliance dict — can't unpack from @configclass at class-definition time
_BASE_COMPLIANCE = {
    "stage1_checkpoint": "",
    "alpha": 10.0,
    "beta": 10.0,
    "compliance_reward_weight": 2.0,
    "max_force": 20.0,
    "robot_mass": 15.0,
}


# ── R1: penalty, no gravity correction, with tracking reward ──────────────
@configclass
class HacLocoStage2R1Cfg(HacLocoStage2RunnerCfg):
    max_iterations: int = 5000
    experiment_name: str = "go2_hac_loco_stage2_R1"
    compliance: dict = {
        **_BASE_COMPLIANCE,
        "reward_type": "penalty",
        "gravity_correction": False,
        "use_tracking_reward": True,
    }


# ── R2: positive, no gravity correction, with tracking reward ─────────────
@configclass
class HacLocoStage2R2Cfg(HacLocoStage2RunnerCfg):
    max_iterations: int = 5000
    experiment_name: str = "go2_hac_loco_stage2_R2"
    compliance: dict = {
        **_BASE_COMPLIANCE,
        "reward_type": "positive",
        "gravity_correction": False,
        "use_tracking_reward": True,
    }


# ── R3: penalty, gravity correction, with tracking reward ─────────────────
@configclass
class HacLocoStage2R3Cfg(HacLocoStage2RunnerCfg):
    max_iterations: int = 5000
    experiment_name: str = "go2_hac_loco_stage2_R3"
    compliance: dict = {
        **_BASE_COMPLIANCE,
        "reward_type": "penalty",
        "gravity_correction": True,
        "use_tracking_reward": True,
    }


# ── R4: positive, gravity correction, with tracking reward ────────────────
@configclass
class HacLocoStage2R4Cfg(HacLocoStage2RunnerCfg):
    max_iterations: int = 5000
    experiment_name: str = "go2_hac_loco_stage2_R4"
    compliance: dict = {
        **_BASE_COMPLIANCE,
        "reward_type": "positive",
        "gravity_correction": True,
        "use_tracking_reward": True,
    }


# ── R5: penalty, no gravity correction, NO tracking reward ────────────────
@configclass
class HacLocoStage2R5Cfg(HacLocoStage2RunnerCfg):
    max_iterations: int = 5000
    experiment_name: str = "go2_hac_loco_stage2_R5"
    compliance: dict = {
        **_BASE_COMPLIANCE,
        "reward_type": "penalty",
        "gravity_correction": False,
        "use_tracking_reward": False,
    }


# ── R6: positive, no gravity correction, NO tracking reward ───────────────
@configclass
class HacLocoStage2R6Cfg(HacLocoStage2RunnerCfg):
    max_iterations: int = 5000
    experiment_name: str = "go2_hac_loco_stage2_R6"
    compliance: dict = {
        **_BASE_COMPLIANCE,
        "reward_type": "positive",
        "gravity_correction": False,
        "use_tracking_reward": False,
    }


# ── R7: penalty, gravity correction, NO tracking reward ───────────────────
@configclass
class HacLocoStage2R7Cfg(HacLocoStage2RunnerCfg):
    max_iterations: int = 5000
    experiment_name: str = "go2_hac_loco_stage2_R7"
    compliance: dict = {
        **_BASE_COMPLIANCE,
        "reward_type": "penalty",
        "gravity_correction": True,
        "use_tracking_reward": False,
    }


# ── R8: positive, gravity correction, NO tracking reward ──────────────────
@configclass
class HacLocoStage2R8Cfg(HacLocoStage2RunnerCfg):
    max_iterations: int = 5000
    experiment_name: str = "go2_hac_loco_stage2_R8"
    compliance: dict = {
        **_BASE_COMPLIANCE,
        "reward_type": "positive",
        "gravity_correction": True,
        "use_tracking_reward": False,
    }
