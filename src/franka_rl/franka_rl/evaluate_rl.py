"""Evaluation of a trained SAC or PPO policy on FrankaPickPlaceEnv."""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.utils import set_random_seed

import franka_task.config as cfg
from franka_task.evaluation import print_summary, run_episode, summarize
from franka_task.gym_env import FrankaPickPlaceEnv

MODEL_IN_RUN_DIR = "latest.zip"
ALGORITHMS = {"sac": SAC, "ppo": PPO}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str,
                        help="Run directory (uses latest.zip) or an explicit .zip")
    parser.add_argument('--algo', type=str, default=None, choices=sorted(ALGORITHMS),
                        help="Algorithm of the model (default: from the run name, sac_* / ppo_*)")
    parser.add_argument('--level', type=str, default='L1', choices=['L1', 'L2', 'L3'],
                        help="Difficulty level (default: 'L1')")
    parser.add_argument('--perturbation', type=str, default=None, choices=cfg.PERTURBATIONS,
                        help="L3 perturbation type (default: None - required for L3)")
    parser.add_argument('--seed', type=int, default=1000,
                        help="Seed of the evaluation episodes - keep it away from training seeds (default: 1000)")
    parser.add_argument('--episodes', type=int, default=50,
                        help="Number of episodes to roll out (default: 50)")
    parser.add_argument('--curriculum-rate', type=float, default=0.0,
                        help="Share of episodes started holding the cube (default: 0.0 - the real task)")
    parser.add_argument('--stochastic', action='store_true',
                        help="Sample actions like during training instead of the deterministic mean")
    parser.add_argument('--output', type=str, default='evaluation',
                        help="Root dir for evaluation outputs (default: evaluation)")

    return parse_checked(parser.parse_args())


def parse_checked(args):
    if args.episodes < 1:
        raise ValueError("--episodes must be positive")
    if not 0.0 <= args.curriculum_rate <= 1.0:
        raise ValueError("--curriculum-rate must be in [0, 1]")
    FrankaPickPlaceEnv._check_perturbation(args.level, args.perturbation)

    return args


def resolve_model(model):
    """ Accepts a run directory or an explicit .zip, returns the .zip path. """
    path = Path(model)
    model_path = path / MODEL_IN_RUN_DIR if path.is_dir() else path

    if not model_path.exists():
        raise FileNotFoundError(f"No model at {model_path}")

    return model_path


def infer_algo(model_path):
    """ Algorithm from the run name: <run>/latest.zip or <run>/checkpoints/rl_model_*.zip. """
    for parent in (model_path.parent, model_path.parent.parent):
        for algo in ALGORITHMS:
            if parent.name.startswith(f"{algo}_"):
                return algo

    raise ValueError(f"Cannot tell the algorithm of {model_path} from its run name - pass --algo")


def evaluate_rl():
    args = parse_args()
    model_path = resolve_model(args.model)

    set_random_seed(args.seed)
    algo = args.algo or infer_algo(model_path)
    model = ALGORITHMS[algo].load(str(model_path), device="cpu")
    deterministic = not args.stochastic

    def act(obs):
        action, _ = model.predict(obs, deterministic=deterministic)
        return action

    mode = "stochastic" if args.stochastic else "deterministic"
    pert = f"_{args.perturbation}" if args.perturbation else ""
    run_name = (f"{model_path.parent.name}_{model.num_timesteps}_{mode}{pert}"
                f"_cr{args.curriculum_rate:g}_seed{args.seed}_{datetime.now():%Y%m%d_%H%M}")
    run_dir = Path(args.output) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    env = FrankaPickPlaceEnv(level=args.level, perturbation=args.perturbation,
                             curriculum_rate=args.curriculum_rate)
    if env.observation_space.shape != model.observation_space.shape:
        env.close()
        raise ValueError(f"Model observation shape {model.observation_space.shape} "
                         f"does not match the env {env.observation_space.shape}")
    env.reset(seed=args.seed)

    records = []
    start = time.monotonic()
    try:
        with open(run_dir / "episodes.jsonl", "w") as f:
            for i in range(args.episodes):
                record = run_episode(env, act)
                records.append(record)
                f.write(json.dumps(record) + "\n")
                f.flush()
                print(f"[{i + 1}/{args.episodes}] return={record['return']:8.2f} "
                      f"len={record['length']:3d} cur={int(record['curriculum_start'])} "
                      f"goal={record['goal_side']:5s} pert={int(record['perturbed'])} "
                      f"grasp={int(record['is_grasped'])} release_att={record['release_attempts']:3d} "
                      f"success={int(record['is_success'])} drop={int(record['dropped'])} "
                      f"xy={record['final_cube_to_goal_xy']:.3f}")
    finally:
        env.close()

    if not records:
        return

    groups = {"all": records}
    if 0.0 < args.curriculum_rate < 1.0:
        groups["standard_start"] = [r for r in records if not r["curriculum_start"]]
        groups["curriculum_start"] = [r for r in records if r["curriculum_start"]]
    if args.perturbation:
        groups["perturbed"] = [r for r in records if r["perturbed"]]
        groups["unperturbed"] = [r for r in records if not r["perturbed"]]
    for side in ("left", "right"):
        if args.level != "L1":
            groups[f"goal_{side}"] = [r for r in records if r["goal_side"] == side]

    summary = {
        "model": str(model_path),
        "algo": algo,
        "num_timesteps": int(model.num_timesteps),
        "mode": mode,
        "level": args.level,
        "perturbation": args.perturbation,
        "seed": args.seed,
        "curriculum_rate": args.curriculum_rate,
        "wall_time_s": time.monotonic() - start,
    }
    for name, group in groups.items():
        if group:
            summary[name] = summarize(group)
            print_summary(f"{mode} | {name}", summary[name])

    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nwritten to {run_dir}")


if __name__ == "__main__":
    evaluate_rl()
