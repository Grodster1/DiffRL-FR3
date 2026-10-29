"""Evaluation protocol shared by every method (SAC, PPO, Diffusion Policy): one episode
rollout with any `act(obs) -> action`, the per-episode record and its aggregation.
"""

import numpy as np

from franka_task.gym_env import FrankaPickPlaceEnv


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
    """ Rolls out one episode with `act(obs) -> action`."""
    obs, info = env.reset()
    if hasattr(act, "reset"):
        act.reset()
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
        "goal_side": info["goal_side"],
        "perturbed": bool(info["perturbed"]),
        "perturb_step": info["perturb_step"],
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
