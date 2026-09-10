import argparse
import sys
import franka_rl.config as cfg
from datetime import datetime
from pathlib import Path
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.save_util import load_from_zip_file
from franka_rl.gym_env import FrankaPickPlaceEnv

EPISODE_INFO_KEYS = ("is_success", "is_grasped", "dropped", "min_ee_to_cube",
                     "sim_dt_mean", "sim_dt_max")

RESUME_MODEL = "latest.zip"
RESUME_BUFFER = "latest_replay_buffer.pkl"

def make_env(level, seed, monitor_path=None):
    env = FrankaPickPlaceEnv(level=level)
    env = Monitor(env, monitor_path, info_keywords=EPISODE_INFO_KEYS)
    env.reset(seed=seed)

    return env

class ResumePointCallback(BaseCallback):
    """ Writes a single, fixed-name resume point (model + replay buffer), overwritten
        in place. SAC cannot be resumed from the policy alone - without the buffer the
        learner restarts from an empty off-policy dataset """

    def __init__(self, save_freq, run_dir, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.run_dir = Path(run_dir)

    def _save(self):
        self.model.save(str(self.run_dir / RESUME_MODEL))
        self.model.save_replay_buffer(str(self.run_dir / RESUME_BUFFER))
        if self.verbose:
            print(f"[resume point] {self.num_timesteps} steps -> {self.run_dir / RESUME_MODEL}")

    def _on_step(self):
        if self.n_calls % self.save_freq == 0:
            self._save()

        return True

    def _on_training_end(self):
        self._save()

def resolve_resume(resume):
    """ Accepts a run directory or an explicit .zip and returns (model, buffer, run_dir).
        The buffer is optional - a run interrupted before the first save point still
        resumes, just with an empty buffer. """

    path = Path(resume)

    if path.is_dir():
        run_dir, model_path = path, path / RESUME_MODEL
    else:
        run_dir, model_path = path.parent, path

    if not model_path.exists():
        raise FileNotFoundError(f"No resume point at {model_path}")

    buffer_path = run_dir / RESUME_BUFFER

    return model_path, buffer_path if buffer_path.exists() else None, run_dir

def saved_timesteps(model_path):
    """ Step counter out of a saved zip without building the policy - lets the
        budget be checked before anything touches ROS or spawns a Monitor file. """
    data, _, _ = load_from_zip_file(model_path, load_data=True, device="cpu")

    return data["num_timesteps"]

def remaining_timesteps(target, done):
    """ SB3 adds `total_timesteps` to the step counter when `reset_num_timesteps=False`,
        so passing the target verbatim would overshoot by everything already trained.
        `--timesteps` is a target here, not a per-invocation budget: resuming a run to
        the same number is a no-op, not another full run. """

    left = target - done

    if left <= 0:
        raise SystemExit(f"Run already reached {done} steps, --timesteps {target} adds nothing")

    return left

def monitor_path(run_dir, resumed):
    """ Monitor opens its file in write mode, so resuming into the same run dir would
        truncate the previous monitor.csv. Each resume gets its own numbered file;
        SB3's load_results() globs *monitor.csv and picks up all of them. """

    if not resumed:
        return str(run_dir)

    n = 1 + len(list(run_dir.glob("resume*.monitor.csv")))

    return str(run_dir / f"resume{n}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--level', type=str, default='L1', choices=['L1', 'L2', 'L3'],
                        help="Difficulty level (default: 'L1')")
    parser.add_argument('--seed', type=int, default=0,
                        help="Random seed (default: 0)")
    parser.add_argument('--timesteps', type=int, default=5000,
                        help="Total timesteps to reach, resume included (default: 5000)")
    parser.add_argument('--tensorboard', action='store_true',
                        help="Log to TensorBoard in <output>/<run_name>/tb (default: disabled)")
    parser.add_argument('--output', type=str, default='data/runs',
                        help="Root dir for run outputs (default: data/runs)")
    parser.add_argument('--resume', type=str, default=None,
                        help="Continue a run: path to its dir or to a saved .zip")
    parser.add_argument('--save-freq', type=int, default=10000,
                        help="Steps between resume-point saves (default: 10000)")
    parser.add_argument('--buffer-size', type=int, default=1_000_000,
                        help="Replay buffer capacity (default: 1000000)")

    return parse_checked(parser.parse_args())

def parse_checked(args):
    if args.save_freq < 1:
        raise ValueError("--save-freq must be positive")

    return args

def train_sac():
    args = parse_args()

    if args.resume:
        model_path, buffer_path, run_dir = resolve_resume(args.resume)
        budget = remaining_timesteps(args.timesteps, saved_timesteps(model_path))
    else:
        run_name = f"sac_{args.level}_seed{args.seed}_{datetime.now():%Y%m%d_%H%M}"
        run_dir = Path(args.output) / run_name
        model_path = buffer_path = None
        budget = args.timesteps

    run_name = run_dir.name
    checkpoint_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(args.level, args.seed, monitor_path(run_dir, bool(args.resume)))
    tb_log = str(run_dir / "tb") if args.tensorboard else None

    if model_path:
        model = SAC.load(str(model_path), env=env, tensorboard_log=tb_log, device="auto")
        if buffer_path:
            model.load_replay_buffer(str(buffer_path))
        print(f"Resumed {model_path} at {model.num_timesteps} steps, "
              f"buffer: {model.replay_buffer.size() if buffer_path else 'EMPTY (no save point yet)'}")
    else:
        model = SAC(
            "MlpPolicy",
            env=env,
            learning_rate=3e-4,
            buffer_size=args.buffer_size,
            batch_size=256,
            gamma=cfg.GAMMA,
            tau=0.005,
            ent_coef="auto",
            learning_starts=1000,
            verbose=1,
            tensorboard_log=tb_log,
            seed=args.seed,
            device="auto"
        )

    callbacks = [
        CheckpointCallback(
            save_freq=max(args.timesteps // 5, 1),
            save_path=str(checkpoint_dir)
        ),
        ResumePointCallback(args.save_freq, run_dir, verbose=1),
    ]

    model.learn(
        total_timesteps=budget,
        callback=callbacks,
        tb_log_name=run_name,
        # A rich progress bar redrawing into a redirected log file buries the metric
        # tables under thousands of repainted frames - long runs are started detached.
        progress_bar=sys.stdout.isatty(),
        reset_num_timesteps=not args.resume,
    )
    model.save(str(run_dir / "sac_final"))

if __name__ == "__main__":
    train_sac()
