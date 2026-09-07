"""Random-policy baseline for FrankaPickPlaceEnv.

Answers one question: can the grasp condition fire at all under random actions? 
If `grasp_rate` is 0 over a few hundred episodes, the replay buffer will never 
contain a grasp transition and no amount of timesteps will fix that. 
In that case the task needs a curriculum or denser shaping instead.
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import franka_rl.config as cfg
from franka_rl.gym_env import FrankaPickPlaceEnv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--level', type=str, default='L1', choices=['L1', 'L2', 'L3'],
                        help="Difficulty level (default: 'L1')")
    parser.add_argument('--seed', type=int, default=0,
                        help="Random seed (default: 0)")
    parser.add_argument('--episodes', type=int, default=100,
                        help="Number of episodes to roll out (default: 100)")
    parser.add_argument('--output', type=str, default='data/runs',
                        help="Root dir for run outputs (default: data/runs)")

    return parser.parse_args()


def run_episode(env:FrankaPickPlaceEnv, action_rng):
    """ Rolls out one episode of uniformly random actions.
        Returns the episode record built from the terminal info dict. """
    _, info = env.reset()
    total_reward = 0.0
    steps = 0
    terminated = truncated = False

    while not (terminated or truncated):
        action = action_rng.uniform(-1.0, 1.0, size=env.action_space.shape)
        _, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        steps += 1

    return {
        "return": total_reward,
        "length": steps,
        "is_success": bool(info["is_success"]),
        "is_grasped": bool(info["is_grasped"]),
        "dropped": bool(info["dropped"]),
        "min_ee_to_cube": float(info["min_ee_to_cube"]),
        "ik_failures": int(info["ik_failures"]),
        "terminated": bool(terminated),
    }


def summarize(records, wall_time):
    """ Aggregates episode records into the numbers worth reading. """
    returns = np.array([r["return"] for r in records])
    lengths = np.array([r["length"] for r in records])
    min_dists = np.array([r["min_ee_to_cube"] for r in records])
    ik_failures = np.array([r["ik_failures"] for r in records])
    total_steps = int(lengths.sum())

    return {
        "episodes": len(records),
        "total_steps": total_steps,
        "fps": total_steps / wall_time if wall_time > 0 else float("nan"),
        "success_rate": float(np.mean([r["is_success"] for r in records])),
        "grasp_rate": float(np.mean([r["is_grasped"] for r in records])),
        "drop_rate": float(np.mean([r["dropped"] for r in records])),
        "return_mean": float(returns.mean()),
        "return_std": float(returns.std()),
        "return_max": float(returns.max()),
        "length_mean": float(lengths.mean()),
        "min_ee_to_cube_best": float(min_dists.min()),
        "min_ee_to_cube_p10": float(np.percentile(min_dists, 10)),
        "min_ee_to_cube_median": float(np.median(min_dists)),
        "ik_failure_rate": float(ik_failures.sum() / max(total_steps, 1)),
    }


def print_summary(stats, grasp_dist_thresh):
    print("\n=== random-policy baseline ===")
    print(f"episodes            : {stats['episodes']}  ({stats['total_steps']} steps, "
          f"{stats['fps']:.1f} FPS)")
    print(f"success_rate        : {stats['success_rate']:.3f}")
    print(f"grasp_rate          : {stats['grasp_rate']:.3f}")
    print(f"drop_rate           : {stats['drop_rate']:.3f}")
    print(f"return              : {stats['return_mean']:.2f} +/- {stats['return_std']:.2f} "
          f"(best {stats['return_max']:.2f})")
    print(f"episode length      : {stats['length_mean']:.1f}")
    print(f"min |ee-cube| best  : {stats['min_ee_to_cube_best']:.3f} m "
          f"(grasp needs < {grasp_dist_thresh:.3f})")
    print(f"min |ee-cube| p10   : {stats['min_ee_to_cube_p10']:.3f} m")
    print(f"min |ee-cube| median: {stats['min_ee_to_cube_median']:.3f} m")
    print(f"ik_failure_rate     : {stats['ik_failure_rate']:.3f}")


def random_baseline():
    args = parse_args()

    run_name = f"random_{args.level}_seed{args.seed}_{datetime.now():%Y%m%d_%H%M}"
    run_dir = Path(args.output) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    env = FrankaPickPlaceEnv(level=args.level)
    env.reset(seed=args.seed)
    action_rng = np.random.default_rng(args.seed)

    records = []
    start = time.monotonic()
    try:
        with open(run_dir / "episodes.jsonl", "w") as f:
            for i in range(args.episodes):
                record = run_episode(env, action_rng)
                records.append(record)
                f.write(json.dumps(record) + "\n")
                f.flush()
                print(f"[{i + 1}/{args.episodes}] return={record['return']:8.2f} "
                      f"len={record['length']:3d} grasp={int(record['is_grasped'])} "
                      f"success={int(record['is_success'])} "
                      f"min_d={record['min_ee_to_cube']:.3f}")
    finally:
        env.close()

    if not records:
        return

    stats = summarize(records, time.monotonic() - start)
    stats["level"] = args.level
    stats["seed"] = args.seed
    with open(run_dir / "summary.json", "w") as f:
        json.dump(stats, f, indent=2)

    print_summary(stats, cfg.GRASP_DIST_THRESH)
    print(f"\nwritten to {run_dir}")


if __name__ == "__main__":
    random_baseline()
