import argparse
import json
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

import franka_rl.config as cfg
from franka_rl.expert import PHASE_BUDGET, ScriptedExpert
from franka_rl.gym_env import FrankaPickPlaceEnv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--level', type=str, default='L1', choices=['L1', 'L2', 'L3'],
                        help="Difficulty level (default: 'L1')")
    parser.add_argument('--seed', type=int, default=0,
                        help="Random seed (default: 0)")
    parser.add_argument('--episodes', type=int, default=100,
                        help="Number of successfully collected episodes (default: 100)")
    parser.add_argument('--max-attempts', type=int, default=200,
                        help="Number of maximum attemtps (default: 200)")
    parser.add_argument('--output', type=str, default='data/demos',
                        help="Root dir for run outputs (default: data/demos)")
    parser.add_argument('--save', action='store_true',
                        help="Write successful episodes to disk (default: off)")
    parser.add_argument('--verbose', action='store_true',
                        help="Per-step phase trace (default: off)")

    return parser.parse_args()

def run_episode(env:FrankaPickPlaceEnv, expert:ScriptedExpert, seed, verbose=False):
    """ Rolls out one episode of actions performed by expert.
        Returns the episode record built from the terminal info dict. """
    obs, info = env.reset(seed=seed)
    expert.reset()
    obs_buf = [obs] # T+1 - due to final observation which is not doable
    act_buf, rew_buf = [], [] # T
    total_reward = 0.0
    terminated = truncated = False

    while not (terminated or truncated):
        action = expert.act(env.obs_dict)
        obs, reward, terminated, truncated, info = env.step(action)
        
        act_buf.append(action)
        rew_buf.append(reward)
        obs_buf.append(obs)
        total_reward += float(reward)
        if verbose:
            d = np.linalg.norm(env.obs_dict["ee_to_cube"])
            print(f"  {len(act_buf):3d} {expert.phase:<9} "
                  f"|ee-cube|={d:.3f} grip={env.obs_dict['gripper']:.3f} "
                  f"a=[{action[0]:+.2f} {action[1]:+.2f} {action[2]:+.2f} {action[3]:+.1f}]")
        
        if expert.failed:
            break
        
    traj = {
        "obs": np.asarray(obs_buf, dtype=np.float32),
        "action": np.asarray(act_buf, dtype=np.float32),
        "reward": np.asarray(rew_buf, dtype=np.float32),
        "goal_pos": np.asarray(env.goal_pos, dtype=np.float32)
    }
        
    record = {
        "return": total_reward,
        "length": len(act_buf),
        "is_success": bool(info["is_success"]),
        "is_grasped": bool(info["is_grasped"]),
        "dropped": bool(info["dropped"]),
        "min_ee_to_cube": float(info["min_ee_to_cube"]),
        "ik_failures": int(info["ik_failures"]),
        "terminated": bool(terminated),
        "expert_failed": bool(expert.failed),
        "failure_phase": expert.failure_phase,
    }
        
    return record, traj
    
def summarize(records, wall_time):
    """ Aggregates episode records into the numbers worth reading. """
    returns = np.array([r["return"] for r in records])
    lengths = np.array([r["length"] for r in records])
    min_dists = np.array([r["min_ee_to_cube"] for r in records])
    ik_failures = np.array([r["ik_failures"] for r in records])
    expert_failures = sum(r["expert_failed"] for r in records)
    failure_phases = Counter(r["failure_phase"] for r in records if r["expert_failed"])
    total_steps = int(lengths.sum())

    return {
        "episodes": len(records),
        "total_steps": total_steps,
        "fps": total_steps / wall_time if wall_time > 0 else float("nan"),
        "success_rate": float(np.mean([r["is_success"] for r in records])),
        "grasp_rate": float(np.mean([r["is_grasped"] for r in records])),
        "drop_rate": float(np.mean([r["dropped"] for r in records])),
        "expert_failure_rate": float(expert_failures / len(records)),
        "failure_phases": dict(failure_phases.most_common()),
        "return_mean": float(returns.mean()),
        "return_std": float(returns.std()),
        "return_max": float(returns.max()),
        "length_mean": float(lengths.mean()),
        "min_ee_to_cube_best": float(min_dists.min()),
        "min_ee_to_cube_p10": float(np.percentile(min_dists, 10)),
        "min_ee_to_cube_median": float(np.median(min_dists)),
        "ik_failure_rate": float(ik_failures.sum() / max(total_steps, 1)),
    }
    
def save_episodes(run_dir, index, traj):
    """ Saves episodes in .npz format with traj unpacking """
    np.savez_compressed(run_dir / f"ep_{index:04d}.npz", **traj)
    
def print_summary(stats, grasp_dist_thresh):
    print("\n=== expert-driven actions ===")
    print(f"attempts            : {stats['episodes']}  ({stats['total_steps']} steps, "
          f"{stats['fps']:.1f} FPS)")
    print(f"demos collected     : {stats['demos_collected']}")
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
    print(f"expert_failure_rate : {stats['expert_failure_rate']:.3f}")
    if stats["failure_phases"]:
        breakdown = "  ".join(f"{phase}={n}" for phase, n in stats["failure_phases"].items())
        print(f"  gave up in        : {breakdown}")
    
def collect_demos():
    args = parse_args()
    
    run_name = f"demo_{args.level}_seed{args.seed}_{datetime.now():%Y%m%d_%H%M}"
    run_dir = Path(args.output) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    env = FrankaPickPlaceEnv(level=args.level)
    expert = ScriptedExpert()

    # The expert's phase budgets must run out before the env truncates, otherwise
    # a stuck episode reports "out of time" instead of naming the phase it hung in.

    budget = sum(v for v in PHASE_BUDGET.values() if v < env.max_episode_steps)
    assert budget < env.max_episode_steps, (
        f"phase budgets ({budget}) leave no room for SETTLE within "
        f"max_episode_steps={env.max_episode_steps}"
    )


    records = []
    n_success = attempts = 0
    start = time.monotonic()
    try:
        with open(run_dir / "episodes.jsonl", "w") as f:
            while n_success < args.episodes and attempts < args.max_attempts:
                record, traj = run_episode(env, expert,
                                            args.seed + attempts,
                                            args.verbose)
                attempts += 1
                records.append(record)
                f.write(json.dumps(record) + "\n")
                f.flush()
                
                if record["is_success"]:
                    if args.save:
                        save_episodes(run_dir, n_success, traj)
                    n_success += 1
                    
                print(f"[{attempts}] success={int(record['is_success'])} "
                      f"len={record['length']:3d} "
                      f"phase={record['failure_phase'] or '-'} "
                      f"({n_success}/{args.episodes} collected)")
    finally:
        env.close()
    
    if not records:
        return

    stats = summarize(records, time.monotonic() - start)
    stats["level"] = args.level
    stats["seed"] = args.seed
    stats["attempts"] = attempts
    stats["demos_collected"] = n_success
    stats["saved_to_disk"] = bool(args.save)
    
    meta = {
        "level": args.level,
        "seed": args.seed,
        "dt": env.dt,
        "action_scale": cfg.ACTION_SCALE,
        "obs_dim": cfg.OBS_DIM,
        "obs_layout": ["ee_pos(3)", "ee_rot6d(6)", "gripper(1)",
                       "ee_to_cube(3)", "cube_to_goal(3)", "is_grasped(1)"],
        "rel_scale": cfg.REL_SCALE.tolist(),
        "workspace_box": cfg.WORKSPACE_BOX.tolist(),
        "gripper_range": list(cfg.GRIPPER_RANGE),
        "n_episodes": n_success,
        "attempts": attempts,
    }
    
    with open(run_dir / "summary.json", "w") as f:
        json.dump(stats, f, indent=2)
        
    with open(run_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
        
    print_summary(stats, cfg.GRASP_DIST_THRESH)
    print(f"\nwritten to {run_dir}")
                
if __name__ == "__main__":
    collect_demos()
    