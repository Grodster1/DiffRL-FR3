"""Evaluation of a trained SAC policy on FrankaPickPlaceEnv.

Training-time monitor.csv records episodes rolled out with *sampled* SAC actions,
so its success rate mixes the policy with exploration noise. This script rolls the
policy out deterministically (the action the thesis reports and compares with DP),
from the standard start (curriculum_rate = 0) unless told otherwise.

`release_attempts` counts steps where the policy commands the gripper open while
holding the cube - it separates "never tries to release" from "releases off target".
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.utils import set_random_seed

from franka_rl.gym_env import FrankaPickPlaceEnv

MODEL_IN_RUN_DIR = "latest.zip"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str,
                        help="Run directory (uses latest.zip) or an explicit .zip")
    parser.add_argument('--level', type=str, default='L1', choices=['L1', 'L2', 'L3'],
                        help="Difficulty level (default: 'L1')")
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

    return args


def resolve_model(model):
    """ Accepts a run directory or an explicit .zip, returns the .zip path. """
    path = Path(model)
    model_path = path / MODEL_IN_RUN_DIR if path.is_dir() else path

    if not model_path.exists():
        raise FileNotFoundError(f"No model at {model_path}")

    return model_path


def wilson_interval(successes, n, z=1.96):
    """ 95% Wilson score interval for a binomial rate. Unlike the normal
        approximation it stays inside [0, 1] and is usable at 0/n and n/n,
        which is where small evaluation runs tend to land. """
    if n == 0:
        return float("nan"), float("nan")

    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom

    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def run_episode(env: FrankaPickPlaceEnv, act):
    """ Rolls out one episode with `act(obs) -> action`.
        Returns the episode record built from the terminal info dict. """
    obs, info = env.reset()
    total_reward = 0.0
    steps = 0
    release_attempts = 0
    terminated = truncated = False

    while not (terminated or truncated):
        action = act(obs)
        if action[3] > 0 and env.obs_dict["is_grasped"]:
            release_attempts += 1
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        steps += 1

    return {
        "return": total_reward,
        "length": steps,
        "is_success": bool(info["is_success"]),
        "is_grasped": bool(info["is_grasped"]),
        "dropped": bool(info["dropped"]),
        "curriculum_start": bool(info["curriculum_start"]),
        "release_attempts": release_attempts,
        "min_ee_to_cube": float(info["min_ee_to_cube"]),
        "min_cube_to_goal": float(info["min_cube_to_goal"]),
        "final_cube_to_goal_xy": float(np.linalg.norm(info["cube_to_goal"][:2])),
        "ik_failures": int(info["ik_failures"]),
        "terminated": bool(terminated),
    }


def summarize(records):
    """ Aggregates episode records into the numbers worth reading. """
    n = len(records)
    successes = sum(r["is_success"] for r in records)
    returns = np.array([r["return"] for r in records])
    lengths = np.array([r["length"] for r in records])
    total_steps = int(lengths.sum())
    ci_low, ci_high = wilson_interval(successes, n)
    # Where the cube ended when the episode neither succeeded nor dropped it -
    # tells "placed just off target" apart from "never put down".
    missed = [r["final_cube_to_goal_xy"] for r in records if not (r["is_success"] or r["dropped"])]

    return {
        "episodes": n,
        "total_steps": total_steps,
        "success_rate": successes / n,
        "success_ci95": [ci_low, ci_high],
        "grasp_rate": float(np.mean([r["is_grasped"] for r in records])),
        "drop_rate": float(np.mean([r["dropped"] for r in records])),
        "release_attempt_rate": float(np.mean([r["release_attempts"] > 0 for r in records])),
        "return_mean": float(returns.mean()),
        "return_std": float(returns.std()),
        "length_mean": float(lengths.mean()),
        "missed_final_xy_median": float(np.median(missed)) if missed else float("nan"),
        "ik_failure_rate": float(sum(r["ik_failures"] for r in records) / max(total_steps, 1)),
    }


def print_summary(title, stats):
    print(f"\n=== {title} ===")
    print(f"episodes             : {stats['episodes']}  ({stats['total_steps']} steps)")
    print(f"success_rate         : {stats['success_rate']:.3f}  "
          f"(95% CI {stats['success_ci95'][0]:.3f}-{stats['success_ci95'][1]:.3f})")
    print(f"grasp_rate           : {stats['grasp_rate']:.3f}")
    print(f"drop_rate            : {stats['drop_rate']:.3f}")
    print(f"release_attempt_rate : {stats['release_attempt_rate']:.3f}")
    print(f"return               : {stats['return_mean']:.2f} +/- {stats['return_std']:.2f}")
    print(f"episode length       : {stats['length_mean']:.1f}")
    print(f"missed: final |xy|   : {stats['missed_final_xy_median']:.3f} m (median, no success, no drop)")
    print(f"ik_failure_rate      : {stats['ik_failure_rate']:.3f}")


def evaluate_sac():
    args = parse_args()
    model_path = resolve_model(args.model)

    set_random_seed(args.seed)
    model = SAC.load(str(model_path), device="cpu")
    deterministic = not args.stochastic

    def act(obs):
        action, _ = model.predict(obs, deterministic=deterministic)
        return action

    mode = "stochastic" if args.stochastic else "deterministic"
    run_name = (f"{model_path.parent.name}_{model.num_timesteps}_{mode}"
                f"_cr{args.curriculum_rate:g}_seed{args.seed}_{datetime.now():%Y%m%d_%H%M}")
    run_dir = Path(args.output) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    env = FrankaPickPlaceEnv(level=args.level, curriculum_rate=args.curriculum_rate)
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

    summary = {
        "model": str(model_path),
        "num_timesteps": int(model.num_timesteps),
        "mode": mode,
        "level": args.level,
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
    evaluate_sac()
