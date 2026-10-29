import argparse
import sys
from datetime import datetime
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import franka_task.config as cfg
from franka_task.gym_env import FrankaPickPlaceEnv, level_tag
from franka_rl.run_config import write_run_config
from franka_rl.train_sac import (WAYPOINT_SUFFIX, check_resume_reward, make_env,
                                 monitor_path, remaining_timesteps, saved_timesteps)

RESUME_MODEL = "latest.zip"
RESUME_VECNORM = "latest_vecnormalize.pkl"

# 2048 steps = ~10 episodes of 200 steps per update.
PPO_N_STEPS = 2048
PPO_BATCH_SIZE = 64
PPO_N_EPOCHS = 10
PPO_GAE_LAMBDA = 0.95
PPO_CLIP_RANGE = 0.2


class PPOResumePointCallback(BaseCallback):

    def __init__(self, save_freq, run_dir, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.run_dir = Path(run_dir)

    def _save(self):
        self.model.save(str(self.run_dir / RESUME_MODEL))
        self.model.get_vec_normalize_env().save(str(self.run_dir / RESUME_VECNORM))
        if self.verbose:
            print(f"[resume point] {self.num_timesteps} steps -> {self.run_dir / RESUME_MODEL}")

    def _on_step(self):
        if self.n_calls % self.save_freq == 0:
            self._save()

        return True

    def _on_training_end(self):
        self._save()


def resolve_resume(resume):
    """ Accepts a run directory or an explicit .zip, returns (model, vecnormalize, run_dir). """
    path = Path(resume)

    if path.is_dir():
        run_dir, model_path = path, path / RESUME_MODEL
    else:
        run_dir, model_path = path.parent, path

    if not model_path.exists():
        raise FileNotFoundError(f"No resume point at {model_path}")

    vecnorm_path = run_dir / RESUME_VECNORM

    return model_path, vecnorm_path if vecnorm_path.exists() else None, run_dir


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--level', type=str, default='L1', choices=['L1', 'L2', 'L3'],
                        help="Difficulty level (default: 'L1')")
    parser.add_argument('--perturbation', type=str, default=None, choices=cfg.PERTURBATIONS,
                        help="L3 perturbation type (default: None - required for L3)")
    parser.add_argument('--seed', type=int, default=0,
                        help="Random seed (default: 0)")
    parser.add_argument('--timesteps', type=int, default=5000,
                        help="Total timesteps to reach, resume included (default: 5000)")
    parser.add_argument('--curriculum-rate', type=float, default=cfg.CURRICULUM_RATE,
                        help=f"Curriculum rate (default: {cfg.CURRICULUM_RATE})")
    parser.add_argument('--tensorboard', action='store_true',
                        help="Log to TensorBoard in <output>/<run_name>/tb (default: disabled)")
    parser.add_argument('--output', type=str, default='data/runs',
                        help="Root dir for run outputs (default: data/runs)")
    parser.add_argument('--resume', type=str, default=None,
                        help="Continue a run: path to its dir or to a saved .zip")
    parser.add_argument('--save-freq', type=int, default=10000,
                        help="Steps between resume-point saves (default: 10000)")
    parser.add_argument('--checkpoint-freq', type=int, default=20000,
                        help="Steps between policy checkpoints (default: 20000)")
    parser.add_argument('--transport-waypoint', action='store_true',
                        help="Transport reward through a point above the goal table (default: straight line)")

    return parse_checked(parser.parse_args())


def parse_checked(args):
    if args.save_freq < 1 or args.checkpoint_freq < 1:
        raise ValueError("Both --save-freq and --checkpoint-freq must be positive")
    FrankaPickPlaceEnv._check_perturbation(args.level, args.perturbation)

    return args


def train_ppo():
    args = parse_args()

    if args.resume:
        model_path, vecnorm_path, run_dir = resolve_resume(args.resume)
        check_resume_reward(run_dir, args.transport_waypoint)
        budget = remaining_timesteps(args.timesteps, saved_timesteps(model_path))
    else:
        run_name = f"ppo_{level_tag(args.level, args.perturbation)}_seed{args.seed}_{datetime.now():%Y%m%d_%H%M}"
        if args.transport_waypoint:
            run_name += WAYPOINT_SUFFIX
        run_dir = Path(args.output) / run_name
        model_path = vecnorm_path = None
        budget = args.timesteps

    run_name = run_dir.name
    checkpoint_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(args.level, args.seed, monitor_path(run_dir, bool(args.resume)),
                   args.curriculum_rate, args.transport_waypoint, args.perturbation)
    venv = DummyVecEnv([lambda: env])
    if vecnorm_path:
        venv = VecNormalize.load(str(vecnorm_path), venv)
    else:
        venv = VecNormalize(venv, norm_obs=False, norm_reward=True, gamma=cfg.GAMMA)
    tb_log = str(run_dir / "tb") if args.tensorboard else None

    if model_path:
        model = PPO.load(str(model_path), env=venv, tensorboard_log=tb_log, device="cpu")
        print(f"Resumed {model_path} at {model.num_timesteps} steps, "
              f"reward statistics: {'restored' if vecnorm_path else 'RESET (no save point yet)'}")
    else:
        model = PPO(
            "MlpPolicy",
            env=venv,
            learning_rate=3e-4,
            n_steps=PPO_N_STEPS,
            batch_size=PPO_BATCH_SIZE,
            n_epochs=PPO_N_EPOCHS,
            gamma=cfg.GAMMA,
            gae_lambda=PPO_GAE_LAMBDA,
            clip_range=PPO_CLIP_RANGE,
            ent_coef=0.0,
            verbose=1,
            tensorboard_log=tb_log,
            seed=args.seed,
            device="cpu"    # SB3: PPO with an MLP policy is faster on CPU than on GPU
        )

    write_run_config(run_dir, "ppo", args, model.num_timesteps)

    callbacks = [
        CheckpointCallback(
            save_freq=args.checkpoint_freq,
            save_path=str(checkpoint_dir)
        ),
        PPOResumePointCallback(args.save_freq, run_dir, verbose=1),
    ]

    model.learn(
        total_timesteps=budget,
        callback=callbacks,
        tb_log_name=run_name,
        progress_bar=sys.stdout.isatty(),
        reset_num_timesteps=not args.resume,
    )
    model.save(str(run_dir / "ppo_final"))


if __name__ == "__main__":
    train_ppo()
