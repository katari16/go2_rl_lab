"""Export policy and force estimator as separate JIT models.

Loads a CompliantOnPolicyRunner checkpoint and exports:
  - policy.pt          — JIT-traced actor (input: full obs, output: actions)
  - estimator.pt       — JIT-traced force estimator (input: flattened history, output: force_hat)

Both are saved to `exported_policy/` inside the checkpoint's log directory.

Usage:
    python scripts/rsl_rl/play_export.py --task Go2-LowLevel-v0 --num_envs 1 \
        --checkpoint logs/rsl_rl/go2_compliant_no_foot_xyz/2026-03-05_07-58-04/model_16200.pt
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Export policy and estimator as JIT models.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--task", type=str, default=None, help="Task name.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="RL agent config entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os

import gymnasium as gym
import torch

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import go2_rl_lab.tasks  # noqa: F401


class EstimatorForwardWrapper(torch.nn.Module):
    """Thin wrapper that exposes the full forward path for JIT tracing.

    Delegates to estimator._forward() which handles all variants:
    plain encoder, temporal decay, TCN preprocessor, TCN replacement.

    Input:  obs_history [batch, temporal_steps * obs_dim]
    Output: force_hat   [batch, force_dim]
    """

    def __init__(self, estimator):
        super().__init__()
        self._estimator = estimator

    def forward(self, obs_history: torch.Tensor) -> torch.Tensor:
        _z_t, force_hat = self._estimator._forward(obs_history)
        return force_hat


@hydra_task_config(args_cli.task, args_cli.agent)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 1
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Resolve checkpoint
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[play_export] Checkpoint: {resume_path}")

    # Create env + runner
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner_class_name = agent_cfg.class_name
    print(f"[play_export] Runner class: {runner_class_name}")

    if runner_class_name == "CompliantOnPolicyRunner":
        from go2_rl_lab.estimator.compliant_on_policy_runner import CompliantOnPolicyRunner
        train_cfg = agent_cfg.to_dict()
        if args_cli.checkpoint:
            train_cfg["estimator_checkpoint"] = args_cli.checkpoint
        runner = CompliantOnPolicyRunner(env, train_cfg, log_dir=None, device=agent_cfg.device)
    elif runner_class_name == "ForceOnPolicyRunner":
        from go2_rl_lab.estimator.force_runner import ForceOnPolicyRunner
        runner = ForceOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class for export: {runner_class_name}")

    runner.load(resume_path)
    runner.eval_mode()

    # ── Export directory ──────────────────────────────────────────────────
    export_dir = os.path.join(os.path.dirname(resume_path), "exported_policy")
    os.makedirs(export_dir, exist_ok=True)

    # ── Export policy ─────────────────────────────────────────────────────
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_dir, filename="policy.onnx")
    print(f"[play_export] Policy exported to {export_dir}/policy.pt")

    # ── Export force estimator ────────────────────────────────────────────
    if hasattr(runner, "estimator"):
        estimator = runner.estimator
        estimator.eval()

        # Wrap for clean JIT trace (only encoder + f_head)
        wrapper = EstimatorForwardWrapper(estimator)
        wrapper.eval()

        # Create dummy input for tracing
        temporal_steps = runner._temporal_steps
        obs_dim = runner._num_one_step_obs
        force_dim = runner._force_dim
        history_dim = temporal_steps * obs_dim
        dummy_input = torch.zeros(1, history_dim, device=next(estimator.parameters()).device)

        traced = torch.jit.trace(wrapper, dummy_input)
        estimator_path = os.path.join(export_dir, "estimator.pt")
        traced.save(estimator_path)

        print(f"[play_export] Estimator exported to {estimator_path}")
        print(f"  Input:  [{history_dim}] = {temporal_steps} steps x {obs_dim} obs_dim")
        print(f"  Output: [{force_dim}] force estimate")

        # Save metadata for deployment
        import json
        meta = {
            "temporal_steps": temporal_steps,
            "obs_dim": obs_dim,
            "force_dim": force_dim,
            "history_input_dim": history_dim,
            "policy_obs_dim": obs_dim + force_dim,
        }
        meta_path = os.path.join(export_dir, "estimator_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[play_export] Metadata saved to {meta_path}")
    else:
        print("[play_export] No force estimator found on runner, skipping estimator export.")

    print(f"\n[play_export] All exports saved to: {export_dir}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
