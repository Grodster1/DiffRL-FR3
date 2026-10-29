# /// script
# requires-python = ">=3.10"
# dependencies = ["huggingface_hub>=0.25"]
# ///
""" Pushes one training run (without its replay buffer) to a private Hugging Face Hub repo"""

import argparse
import fnmatch
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EVALUATION_DIR = REPO_ROOT / "evaluation"

# Only needed to --resume on the machine that trained; ~1-2 GB per run.
ALWAYS_SKIPPED = ("*replay_buffer.pkl",)
CHECKPOINTS = ("checkpoints/*",)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.strip())
    parser.add_argument("run_dir", type=Path, help="Run directory, e.g. data/runs/sac_L2_seed0_..._wp")
    parser.add_argument("--repo", required=True, help="Hub repo id, e.g. Grodster/diffrl-fr3-runs")
    parser.add_argument("--with-checkpoints", action="store_true",
                        help="Also push checkpoints/ (default: only latest/final models)")
    parser.add_argument("--log", type=Path, action="append", default=[],
                        help="Training log to push next to the run (repeatable)")
    parser.add_argument("--no-evaluation", action="store_true",
                        help="Skip evaluation/<run_name>_* results (default: pushed)")
    parser.add_argument("--dry-run", action="store_true", help="List files, push nothing")

    return parser.parse_args()


def is_skipped(rel_path, with_checkpoints):
    patterns = ALWAYS_SKIPPED + (() if with_checkpoints else CHECKPOINTS)
    return any(fnmatch.fnmatch(rel_path, p) for p in patterns)


def select_files(run_dir, with_checkpoints, logs, with_evaluation):
    """ (local path, path in repo) pairs. Paths in the repo follow this repository. """
    run_dir = run_dir.resolve()
    in_repo_run = f"data/runs/{run_dir.name}"
    files = []

    for path in sorted(run_dir.rglob("*")):
        rel = path.relative_to(run_dir).as_posix()
        if path.is_file() and not is_skipped(rel, with_checkpoints):
            files.append((path, f"{in_repo_run}/{rel}"))

    for log in logs:
        files.append((log.resolve(), f"{in_repo_run}/logs/{log.name}"))

    if with_evaluation and EVALUATION_DIR.is_dir():
        # evaluate_rl names them <run>_<timesteps>_<mode>_...; the digits keep <run>_wp out
        own = re.compile(rf"{re.escape(run_dir.name)}_\d+_")
        for eval_dir in sorted(d for d in EVALUATION_DIR.iterdir() if own.match(d.name)):
            for path in sorted(eval_dir.rglob("*")):
                if path.is_file():
                    files.append((path, f"evaluation/{path.relative_to(EVALUATION_DIR).as_posix()}"))

    return files


def human(n_bytes):
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:.0f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TB"


def main():
    args = parse_args()
    if not args.run_dir.is_dir():
        sys.exit(f"No run directory at {args.run_dir}")
    if not (args.run_dir / "run_config.json").exists():
        print("warning: no run_config.json - run predates it, its code version is not recorded")

    files = select_files(args.run_dir, args.with_checkpoints, args.log, not args.no_evaluation)
    total = sum(p.stat().st_size for p, _ in files)
    for path, in_repo in files:
        print(f"{human(path.stat().st_size):>8}  {in_repo}")
    print(f"{len(files)} files, {human(total)} -> {args.repo}")

    if args.dry_run:
        return

    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi()
    api.create_repo(args.repo, private=True, exist_ok=True)
    commit = api.create_commit(
        args.repo,
        operations=[CommitOperationAdd(path_in_repo=r, path_or_fileobj=str(p)) for p, r in files],
        commit_message=f"Push {args.run_dir.resolve().name}",
    )
    print(f"pushed: {commit.commit_url}")


if __name__ == "__main__":
    main()
