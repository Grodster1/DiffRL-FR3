import argparse
from datetime import datetime
from pathlib import Path
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from franka_rl.gym_env import FrankaPickPlaceEnv

EPISODE_INFO_KEYS = ("is_success", "is_grasped", "dropped", "min_ee_to_cube")

def make_env(level, seed, log_dir=None):
    env = FrankaPickPlaceEnv(level=level)
    env = Monitor(env, log_dir, info_keywords=EPISODE_INFO_KEYS)
    env.reset(seed=seed)

    return env

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--level', type=str, default='L1', choices=['L1', 'L2', 'L3'],
                        help="Difficulty level (default: 'L1')")
    parser.add_argument('--seed', type=int, default=0,
                        help="Random seed (default: 0)")
    parser.add_argument('--timesteps', type=int, default=5000,
                        help="Timesteps (default: 5000)")
    parser.add_argument('--tensorboard', action='store_true',
                        help="Log to TensorBoard in <output>/<run_name>/tb (default: disabled)")
    parser.add_argument('--output', type=str, default='data/runs',
                        help="Root dir for run outputs (default: data/runs)")

    return parser.parse_args()

def train_sac():
    args = parse_args()

    run_name = f"sac_{args.level}_seed{args.seed}_{datetime.now():%Y%m%d_%H%M}"
    run_dir = Path(args.output) / run_name
    checkpoint_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(args.level, args.seed, log_dir=str(run_dir))

    model = SAC(
        "MlpPolicy",
        env=env,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        ent_coef="auto",
        learning_starts=1000,
        verbose=1,
        tensorboard_log=str(run_dir / "tb") if args.tensorboard else None,
        seed=args.seed,
        device="auto"
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(args.timesteps // 5, 1),
        save_path=str(checkpoint_dir)
    )

    model.learn(
        total_timesteps=args.timesteps,
        callback=checkpoint_cb,
        tb_log_name=run_name,
        progress_bar=True,
    )
    model.save(str(run_dir / "sac_final"))

if __name__ == "__main__":
    train_sac()
