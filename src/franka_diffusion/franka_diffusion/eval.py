import argparse, json, time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from diffusers.training_utils import EMAModel

from franka_rl import config as cfg
from franka_rl.gym_env import FrankaPickPlaceEnv
from franka_rl.evaluate_sac import run_episode, summarize, print_summary

from .policy import DiffusionPolicy

CHECKPOINT_IN_RUN_DIR = "latest.pt"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=str,
                        help="Run directory (uses latest.pt) or an explicit .pt")
    parser.add_argument('--level', type=str, default='L1', choices=['L1', 'L2', 'L3'],
                        help="Difficulty level (default: 'L1')")
    parser.add_argument('--perturbation', type=str, default=None, choices=cfg.PERTURBATIONS,
                        help="L3 perturbation type (default: None - required for L3)")
    parser.add_argument('--seed', type=int, default=1000,
                        help="Seed of the evaluation episodes - keep it away from training seeds (default: 1000)")
    parser.add_argument('--episodes', type=int, default=50,
                        help="Number of episodes to roll out (default: 50)")
    parser.add_argument('--output', type=str, default='evaluation',
                        help="Root dir for evaluation outputs (default: evaluation)")
    parser.add_argument('--device', type=str, default= 'cuda' if torch.cuda.is_available() else 'cpu',
                        help="Torch computing device: 'cpu', 'cuda', 'cuda:0', 'mps' (Mac) etc. (default: auto-detection cuda/cpu)")

    return parse_checked(parser.parse_args())

def parse_checked(args):
    if args.episodes < 1:
        raise ValueError("--episodes must be positive")
    FrankaPickPlaceEnv._check_perturbation(args.level, args.perturbation)

    return args

def resolve_checkpoint(checkpoint):
    """ Accepts a run directory or an explicit .pt, returns the .pt path. """
    path = Path(checkpoint)
    checkpoint_path = path / CHECKPOINT_IN_RUN_DIR if path.is_dir() else path
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"No checkpoint at {checkpoint_path}")
    return checkpoint_path

def load_policy(checkpoint_path, device):
    """ Rebuilds DiffusionPolicy from the checkpoint config with the EMA weights."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    policy = DiffusionPolicy(**checkpoint["config"]["policy"])
    ema = EMAModel(policy.parameters(), power=0.75)
    ema.load_state_dict(checkpoint["ema"])
    ema.copy_to(policy.parameters())

    return policy.to(device).eval(), checkpoint


class DiffusionPolicyRunner:
    """ act(obs) for run_episode: keeps the observation history and the queue of
        planned actions, queries the policy only when the queue runs out. """

    def __init__(self, policy, device):
        self.policy = policy
        self.device = torch.device(device)
        self.latencies = []  # ms per predict_action call, over the whole evaluation
        self.obs_history = deque(maxlen=policy.n_obs_steps)
        self.action_queue = deque()

    def reset(self):
        """ Called by run_episode after every env.reset(). """
        self.obs_history.clear()
        self.action_queue.clear()

    def __call__(self, obs):
        obs = np.asarray(obs, dtype=np.float32)
        
        if len(self.obs_history) == 0:
            self.obs_history.extend([obs] * self.obs_history.maxlen)
        else:
            self.obs_history.append(obs)

        if not self.action_queue:
            obs_batch = torch.from_numpy(np.stack(self.obs_history)).unsqueeze(0).to(self.device)
            actions = self._timed_predict(obs_batch)[0].cpu().numpy()
            self.action_queue.extend(actions)

        return self.action_queue.popleft().astype(np.float32)

    def _timed_predict(self, obs_batch):
        """ predict_action with its wall time appended to self.latencies. """
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        action = self.policy.predict_action(obs_batch)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.latencies.append((time.perf_counter() - start) * 1e3)
        return action


def summarize_latency(latencies_ms, n_action_steps):
    lat = np.asarray(latencies_ms)
    return {
        "queries": int(lat.size),
        "mean_ms": float(lat.mean()),
        "p95_ms": float(np.percentile(lat, 95)),
        "max_ms": float(lat.max()),
        "per_env_step_ms": float(lat.mean() / n_action_steps),
    }

def evaluate_dp():
    args = parse_args()
    checkpoint_path = resolve_checkpoint(args.checkpoint)

    # DDIM starts from random noise - without a seed the rollouts are not repeatable
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    policy, checkpoint = load_policy(checkpoint_path, args.device)
    config = checkpoint["config"]
    epoch = checkpoint["epoch"] + 1
    trained_on = config["args"]["level"]
    if trained_on != args.level:
        print(f"warning: policy trained on {trained_on}, evaluating on {args.level}")

    run_dir_name = checkpoint_path.parent.name
    if run_dir_name == "checkpoints":
        run_dir_name = checkpoint_path.parent.parent.name
    pert = f"_{args.perturbation}" if args.perturbation else ""
    run_name = f"{run_dir_name}_epoch{epoch}{pert}_seed{args.seed}_{datetime.now():%Y%m%d_%H%M}"
    out_dir = Path(args.output) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    env = FrankaPickPlaceEnv(level=args.level, perturbation=args.perturbation)
    if env.observation_space.shape != (config["policy"]["obs_dim"],):
        env.close()
        raise ValueError(f"Policy obs_dim {config['policy']['obs_dim']} "
                         f"does not match the env {env.observation_space.shape}")
    env.reset(seed=args.seed)

    runner = DiffusionPolicyRunner(policy, args.device)
    records = []
    start = time.monotonic()
    try:
        with open(out_dir / "episodes.jsonl", "w") as f:
            for i in range(args.episodes):
                record = run_episode(env, runner)
                records.append(record)
                f.write(json.dumps(record) + "\n")
                f.flush()
                print(f"[{i + 1}/{args.episodes}] return={record['return']:8.2f} "
                      f"len={record['length']:3d} goal={record['goal_side']:5s} "
                      f"pert={int(record['perturbed'])} grasp={int(record['is_grasped'])} "
                      f"release_att={record['release_attempts']:3d} "
                      f"success={int(record['is_success'])} drop={int(record['dropped'])} "
                      f"xy={record['final_cube_to_goal_xy']:.3f}")
    finally:
        env.close()

    if not records:
        return

    groups = {"all": records}
    if args.perturbation:
        groups["perturbed"] = [r for r in records if r["perturbed"]]
        groups["unperturbed"] = [r for r in records if not r["perturbed"]]
    if args.level != "L1":
        for side in ("left", "right"):
            groups[f"goal_{side}"] = [r for r in records if r["goal_side"] == side]

    summary = {
        "checkpoint": str(checkpoint_path),
        "algo": "dp",
        "epoch": epoch,
        "step": checkpoint["step"],
        "trained_on": trained_on,
        "level": args.level,
        "perturbation": args.perturbation,
        "seed": args.seed,
        "device": str(args.device),
        "num_inference_steps": config["policy"]["num_inference_steps"],
        "wall_time_s": time.monotonic() - start,
        "inference_latency": summarize_latency(runner.latencies, policy.n_action_steps),
    }
    for name, group in groups.items():
        if group:
            summary[name] = summarize(group)
            print_summary(f"dp epoch {epoch} | {name}", summary[name])

    lat = summary["inference_latency"]
    print(f"\ninference: {lat['mean_ms']:.1f} ms mean, {lat['p95_ms']:.1f} ms p95 "
          f"over {lat['queries']} queries ({lat['per_env_step_ms']:.1f} ms per env step)")

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nwritten to {out_dir}")


if __name__ == "__main__":
    evaluate_dp()
