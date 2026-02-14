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