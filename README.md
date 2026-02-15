  **Force-Compliant Quadruped Locomotion with Sim2Real Transfer**                                                     
                                         
  A reinforcement learning framework for training and deploying locomotion policies on the Unitree Go2, built on      
  NVIDIA Isaac Lab. This project addresses two critical challenges in legged robotics: bridging the sim-to-real gap   
  through system identification, and achieving compliant robot behavior for safe human-robot interaction.

  Submitted to the **NVIDIA GTC Hackathon 2026**.

  ---

  ## Overview

  Training RL policies for quadruped locomotion in simulation has become remarkably accessible with frameworks like
  Isaac Lab. But deploying these policies on real hardware reveals two fundamental problems:

  1. **The Sim-to-Real Gap** — Policies trained in simulation often fail to transfer reliably. Most existing
  approaches neglect actuator-specific energy losses or depend on complex, hand-tuned reward formulations.

  2. **Stiff, Unsafe Behavior** — Control policies trained using deep RL often generate stiff, high-frequency motions
  in response to unexpected disturbances. Policies learn to *reject* external forces to maintain stability, creating
  potentially dangerous behavior for human-robot interaction.

  This repository provides an end-to-end solution: from system identification and policy training to sim2sim
  validation and real-world deployment.

  ## Key Features

  - **PACE-based System Identification** — Adapted the [PACE framework](https://arxiv.org/abs/2410.09714) (Precise
  Adaptation through Continuous Evolution) for the Unitree Go2. Augments the IsaacLab `DCMotorCfg` with parameters
  optimized from real-world data: armature, viscous friction, friction, and encoder bias.
  - **Force-Compliant Policies** — End-to-end policy architecture where compliance is learned implicitly through a
  multi-stage reward structure. During normal walking, the robot receives full tracking rewards. After a disturbance,
  a recovery stage allows the robot to smoothly yield before resuming precise tracking.
  - **Sim2Sim Validation** — MuJoCo-based deployment pipeline using `unitree_mujoco` for validating policies before
  real hardware deployment, with full FSM and keyboard/joystick control.
  - **Zero-Shot Sim2Real Deployment** — Direct deployment on the Unitree Go2 via DDS communication with the
  unitree_sdk2, no fine-tuning required on hardware.
  - **Self-Contained Environment Configs** — Single-file environment configurations with all rewards, observations,
  events, and scene definitions in one place.

  ## Repository Structure

  go2_rl_lab/
  ├── assets/
  │   └── unitree/                    # Robot URDF/USD and actuator configs (PACE-optimized)
  ├── tasks/
  │   └── manager_based/
  │       └── go2_rl_lab/
  │           ├── go2_velocity_env_cfg.py   # Self-contained env config
  │           ├── mdp/
  │           │   ├── init.py           # Re-exports IsaacLab + custom mdp functions
  │           │   └── rewards.py            # Custom reward functions (compliance, gait, etc.)
  │           └── agents/
  │               └── rsl_rl_ppo_cfg.py     # PPO training configuration
  ├── sim2sim/
  │   ├── sim2sim_deploy.py           # MuJoCo sim2sim with FSM + keyboard/joystick
  │   └── logs/                       # Deployment logs (.npz)
  ├── deploy_real/
  │   ├── deploy_isaac_config_propioceptive.py  # Real robot deployment script
  │   ├── configs/
  │   │   └── go2_isaaclab.yaml       # Deployment config (obs order, gains, joint mapping)
  │   └── common/
  │       ├── remote_controller.py    # Joystick button/stick parsing
  │       ├── rotation_helper.py      # Gravity projection, IMU transforms
  │       └── command_helper.py       # Motor command utilities
  ├── pre_train/
  │   └── policy_isaaclab_propioceptive.pt  # Trained policy (TorchScript)
  └── pace/
      └── ...                         # System identification data and scripts

  ## Getting Started

  ### Prerequisites

  - **NVIDIA Isaac Sim 4.5+** and **Isaac Lab** (latest version required for actuator model support)
  - **Python 3.10+**
  - **PyTorch 2.x**
  - Unitree Go2 with SDK2 (for real deployment)
  - `unitree_mujoco` (for sim2sim validation)

  ### Installation

  ```bash
  # Clone the repository
  git clone https://github.com/<your-username>/go2_rl_lab.git
  cd go2_rl_lab

  # Install as Isaac Lab extension
  pip install -e .

  Training

  # Train on rough terrain
  python train.py --task Go2-Velocity-Rough-v0 --num_envs 4096

  # Train on flat terrain (change terrain_type to "plane" in env config)
  python train.py --task Go2-Velocity-Flat-v0 --num_envs 4096

  Sim2Sim Validation (MuJoCo)

  # Terminal 1: Start MuJoCo simulator
  cd ~/unitree_mujoco/simulate_python && python3 unitree_mujoco.py

  # Terminal 2: Run sim2sim deployment
  cd sim2sim && python3 sim2sim_deploy.py

  Keyboard controls:

  ┌───────┬─────────────────────┐
  │  Key  │       Action        │
  ├───────┼─────────────────────┤
  │ Enter │ Stand up            │
  ├───────┼─────────────────────┤
  │ Space │ Start policy        │
  ├───────┼─────────────────────┤
  │ W / S │ Forward / Backward  │
  ├───────┼─────────────────────┤
  │ A / D │ Strafe left / right │
  ├───────┼─────────────────────┤
  │ Q / E │ Turn left / right   │
  ├───────┼─────────────────────┤
  │ Esc   │ Stop and lie down   │
  └───────┴─────────────────────┘

  Real Robot Deployment

  cd deploy_real
  python3 deploy_isaac_config_propioceptive.py <network_interface> go2_isaaclab.yaml

  Use the Unitree joystick: START to stand, A to start the policy, SELECT to stop. Left stick controls velocity, right
   stick controls yaw rate.

  Approach

  1. System Identification with PACE

  Actuator drive dynamics are largely linear, enabling fast optimization. The process:

  1. Data Collection — Excite the robot's joints using chirp signals while recording joint positions, velocities, and
  torques.
  2. Large-Scale Optimization — Run parallelized simulations in Isaac Lab to find actuator parameters that best match
  real-world behavior.
  3. Retrain — Use the identified parameters to retrain policies with a more accurate simulator.

  The result: policies that capture the true actuator dynamics, enabling reliable zero-shot deployment.

  2. Force-Compliant Policy Training

  Standard RL policies are trained for robustness through domain randomization and push recovery. This teaches the
  robot to fight against external forces — which makes it stiff and unsafe.

  Our approach uses a multi-stage reward structure:

  - Normal walking — Full velocity tracking rewards drive precise locomotion.
  - Post-disturbance recovery — After an external force is detected, tracking rewards are given regardless of
  deviation from the command. This allows the robot to yield to forces and recover smoothly, rather than aggressively
  rejecting them.

  The result: policies that are both robust (they don't fall over) and compliant (they yield to significant forces),
  enabling safe human-robot interaction without any force sensors.

  3. Observation Space (45-dim Proprioceptive)

  obs[0:3]   — Base angular velocity (IMU gyroscope)
  obs[3:6]   — Projected gravity (from IMU quaternion)
  obs[6:9]   — Velocity commands (vx, vy, wz)
  obs[9:21]  — Joint positions relative to default (Isaac convention)
  obs[21:33] — Joint velocities (Isaac convention)
  obs[33:45] — Last action

  No exteroceptive sensors required. Force estimation is learned implicitly from proprioceptive history.

  Results

  ┌──────────────────────┬───────────────────────────────────────┬───────────────────────────────────────┐
  │                      │             Without PACE              │               With PACE               │
  ├──────────────────────┼───────────────────────────────────────┼───────────────────────────────────────┤
  │ Sim-to-real transfer │ Sluggish, inconsistent motor response │ Smooth, reliable zero-shot deployment │
  ├──────────────────────┼───────────────────────────────────────┼───────────────────────────────────────┤
  │ Tracking accuracy    │ Poor, especially rear legs            │ Consistent across all joints          │
  ├──────────────────────┼───────────────────────────────────────┼───────────────────────────────────────┤
  │ Gait quality         │ Unstable, jerky                       │ Natural, controlled                   │
  └──────────────────────┴───────────────────────────────────────┴───────────────────────────────────────┘

  ┌───────────────────┬────────────────────────────────┬────────────────────────────────────┐
  │                   │          Stiff Policy          │          Compliant Policy          │
  ├───────────────────┼────────────────────────────────┼────────────────────────────────────┤
  │ Push response     │ Aggressive force rejection     │ Smooth yielding in force direction │
  ├───────────────────┼────────────────────────────────┼────────────────────────────────────┤
  │ Human interaction │ Unsafe, high-frequency torques │ Safe, controlled response          │
  ├───────────────────┼────────────────────────────────┼────────────────────────────────────┤
  │ Energy efficiency │ High (fighting disturbances)   │ Lower (working with disturbances)  │
  ├───────────────────┼────────────────────────────────┼────────────────────────────────────┤
  │ Recovery          │ Abrupt                         │ Gradual, smooth                    │
  └───────────────────┴────────────────────────────────┴────────────────────────────────────┘

  Applications

  - Guided locomotion — Pull the robot like guiding a dog on a leash
  - Payload carrying — Use the robot as a barrow that follows your lead
  - Collaborative robotics — Move the robot out of the way safely, without a joystick
  - Construction & warehousing — Robots that navigate safely around people

  Future Work

  - Separate force estimator network — A dedicated network that maps proprioceptive measurements to external force
  estimates, enabling explicit force-based velocity command generation for improved interpretability and energy
  efficiency.
  - Payload-aware locomotion — Real-time analysis of payload dynamics to adjust gait and velocity commands.
  - Multi-robot collaboration — Compliant policies for robot-robot physical interaction.

  Acknowledgments

  - NVIDIA — Isaac Sim, Isaac Lab, and the GTC Hackathon
  - Unitree Robotics — Go2 platform and SDK
  - PACE Authors — Original system identification framework

  License
