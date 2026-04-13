from __future__ import annotations

import torch
from dataclasses import MISSING

from isaaclab.actuators import DelayedPDActuator, DelayedPDActuatorCfg
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions
from pace_sim2real.utils import PaceDCMotorCfg


@configclass
class UnitreeActuatorCfg_Go2PACE(PaceDCMotorCfg):
    """Go2 Actuator Profile with PACE identified physical parameters."""
    
    # Physical Limits (Go2 hardware constants)
    saturation_effort = 23.7
    effort_limit = 23.7
    velocity_limit = 30.0
    max_delay = 5

    # --- IDENTIFIED PHYSICAL IDENTITY ---
    armature = {
        "FR_hip_joint": 0.0160, "FR_thigh_joint": 0.0108, "FR_calf_joint": 0.0286,
        "FL_hip_joint": 0.0160, "FL_thigh_joint": 0.0150, "FL_calf_joint": 0.0282,
        "RR_hip_joint": 0.0064, "RR_thigh_joint": 0.0135, "RR_calf_joint": 0.0315,
        "RL_hip_joint": 0.0100, "RL_thigh_joint": 0.0140, "RL_calf_joint": 0.0295,
    }
    viscous_friction = {
        "FR_hip_joint": 0.5615, "FR_thigh_joint": 0.4219, "FR_calf_joint": 0.2667,
        "FL_hip_joint": 0.3317, "FL_thigh_joint": 0.2501, "FL_calf_joint": 0.3808,
        "RR_hip_joint": 0.4284, "RR_thigh_joint": 0.3290, "RR_calf_joint": 0.2762,
        "RL_hip_joint": 0.3973, "RL_thigh_joint": 0.3252, "RL_calf_joint": 0.2421,
    }
    friction = {
        "FR_hip_joint": 0.0645, "FR_thigh_joint": 0.0709, "FR_calf_joint": 0.1322,
        "FL_hip_joint": 0.0779, "FL_thigh_joint": 0.0141, "FL_calf_joint": 0.1585,
        "RR_hip_joint": 0.1138, "RR_thigh_joint": 0.2519, "RR_calf_joint": 0.4289,
        "RL_hip_joint": 0.0719, "RL_thigh_joint": 0.1269, "RL_calf_joint": 0.0284,
    }
    # encoder bias as list, ordered by URDF joint order:
    # FL_hip, FL_thigh, FL_calf, FR_hip, FR_thigh, FR_calf,
    # RL_hip, RL_thigh, RL_calf, RR_hip, RR_thigh, RR_calf
    encoder_bias = [
        0.0939, 0.2101, 0.1569,   # FL
        -0.0718, 0.2051, 0.1565,   # FR
        -0.0485, 0.3641, -0.1367,  # RL
        -0.2326, 0.0542, 0.3806,   # RR
    ]


@configclass
class UnitreeActuatorCfg_Go2PACE_LowGain(PaceDCMotorCfg):
    """Go2 PACE actuator with low PD gains (Kp=8, Kd=0.4).

    PACE parameters from run 26_03_10_10-00-32 (params 199).
    """

    # Physical limits (Go2 hardware constants)
    saturation_effort = 23.7
    effort_limit = 23.7
    velocity_limit = 30.0
    max_delay = 4  # identified delay ~3.52 steps

    # --- IDENTIFIED PHYSICAL PARAMETERS (run 26_03_10_10-00-32) ---
    # PACE output order: FR, FL, RR, RL (hip, thigh, calf each)
    armature = {
        "FR_hip_joint": 0.0164, "FR_thigh_joint": 0.0102, "FR_calf_joint": 0.0224,
        "FL_hip_joint": 0.0102, "FL_thigh_joint": 0.0105, "FL_calf_joint": 0.0208,
        "RR_hip_joint": 0.0063, "RR_thigh_joint": 0.0063, "RR_calf_joint": 0.0307,
        "RL_hip_joint": 0.0042, "RL_thigh_joint": 0.0080, "RL_calf_joint": 0.0278,
    }
    viscous_friction = {
        "FR_hip_joint": 0.3275, "FR_thigh_joint": 0.2374, "FR_calf_joint": 0.5056,
        "FL_hip_joint": 0.2717, "FL_thigh_joint": 0.1572, "FL_calf_joint": 1.7110,
        "RR_hip_joint": 0.2537, "RR_thigh_joint": 0.1789, "RR_calf_joint": 0.6863,
        "RL_hip_joint": 0.3052, "RL_thigh_joint": 0.1903, "RL_calf_joint": 0.3440,
    }
    friction = {
        "FR_hip_joint": 0.0420, "FR_thigh_joint": 0.0324, "FR_calf_joint": 0.0144,
        "FL_hip_joint": 0.0375, "FL_thigh_joint": 0.0219, "FL_calf_joint": 0.0102,
        "RR_hip_joint": 0.0103, "RR_thigh_joint": 0.1298, "RR_calf_joint": 0.0206,
        "RL_hip_joint": 0.0802, "RL_thigh_joint": 0.0331, "RL_calf_joint": 0.0136,
    }
    # encoder bias in URDF joint order: FL, FR, RL, RR
    encoder_bias = [
        0.1736, 0.1487, 0.1855,   # FL (PACE indices 3,4,5)
        0.0211, 0.2577, 0.0313,   # FR (PACE indices 0,1,2)
        0.0351, 0.2485, 0.0049,   # RL (PACE indices 9,10,11)
        0.0022, 0.0619, 0.1643,   # RR (PACE indices 6,7,8)
    ]


@configclass
class UnitreeActuatorCfg_Go2PACE_April(PaceDCMotorCfg):
    """Go2 PACE actuator identified in April run (26_04_12_23-04-12, params 199).

    Identified with low PD gains (Kp=8, Kd=0.4). Delay param ~3.46 steps.
    Encoder bias was fixed at zero during this optimization run.
    """

    # Physical limits (Go2 hardware constants)
    saturation_effort = 23.7
    effort_limit = 23.7
    velocity_limit = 30.0
    max_delay = 4  # identified delay ~3.46 steps

    # --- IDENTIFIED PHYSICAL PARAMETERS (run 26_04_12_23-04-12) ---
    # PACE output order: FR, FL, RR, RL (hip, thigh, calf each)
    armature = {
        "FR_hip_joint": 0.0117, "FR_thigh_joint": 0.0098, "FR_calf_joint": 0.0217,
        "FL_hip_joint": 0.0086, "FL_thigh_joint": 0.0118, "FL_calf_joint": 0.0237,
        "RR_hip_joint": 0.0080, "RR_thigh_joint": 0.0083, "RR_calf_joint": 0.0266,
        "RL_hip_joint": 0.0039, "RL_thigh_joint": 0.0091, "RL_calf_joint": 0.0270,
    }
    viscous_friction = {
        "FR_hip_joint": 0.3321, "FR_thigh_joint": 0.2331, "FR_calf_joint": 0.5150,
        "FL_hip_joint": 0.2638, "FL_thigh_joint": 0.1626, "FL_calf_joint": 1.6673,
        "RR_hip_joint": 0.2465, "RR_thigh_joint": 0.1856, "RR_calf_joint": 0.6598,
        "RL_hip_joint": 0.2977, "RL_thigh_joint": 0.1955, "RL_calf_joint": 0.3352,
    }
    friction = {
        "FR_hip_joint": 0.0198, "FR_thigh_joint": 0.0352, "FR_calf_joint": 0.0186,
        "FL_hip_joint": 0.0306, "FL_thigh_joint": 0.0270, "FL_calf_joint": 0.0192,
        "RR_hip_joint": 0.0250, "RR_thigh_joint": 0.1312, "RR_calf_joint": 0.0138,
        "RL_hip_joint": 0.1634, "RL_thigh_joint": 0.0530, "RL_calf_joint": 0.0253,
    }
    # encoder bias fixed at zero during this run; URDF joint order: FL, FR, RL, RR
    encoder_bias = [
        0.0, 0.0, 0.0,   # FL
        0.0, 0.0, 0.0,   # FR
        0.0, 0.0, 0.0,   # RL
        0.0, 0.0, 0.0,   # RR
    ]