import argparse, csv, json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.training_utils import EMAModel

from .dataset import DemonstrationDataset
from .policy import DiffusionPolicy

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--demos', type=str, nargs='+', required=True,
                        help="One or more demo dirs with ep_*.npz (e.g. data/demos/demo_L1_seed0_20261008_1035)")
    parser.add_argument('--level', type=str, default='L1', choices=['L1', 'L2', 'L3'],
                        help="Level the demos come from, used in the run name (default: 'L1')")
    parser.add_argument('--seed', type=int, default=0,
                        help="Random seed for the split, weights and noise (default: 0)")
    parser.add_argument('--epochs', type=int, default=2000,
                        help="Training epochs (default: 2000)")
    parser.add_argument('--batch-size', type=int, default=256,
                        help="Batch size (default: 256)")
    parser.add_argument('--lr', type=float, default=1e-4,
                        help="Peak learning rate after warmup (default: 1e-4)")
    parser.add_argument('--val-ratio', type=float, default=0.1,
                        help="Fraction of episodes held out for validation (default: 0.1)")
    parser.add_argument('--checkpoint-every', type=int, default=100,
                        help="Epochs between checkpoints (default: 100)")
    parser.add_argument('--output', type=str, default='data/runs',
                        help="Root dir for run outputs (default: data/runs)")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help="Torch device (default: cuda if available)")

    return parse_checked(parser.parse_args())

def parse_checked(args):
    if not 0.0 <= args.val_ratio < 1.0:
        raise ValueError("--val-ratio must be in [0, 1)")
    if args.epochs < 1 or args.batch_size < 1 or args.checkpoint_every < 1:
        raise ValueError("--epochs, --batch-size and --checkpoint-every must be positive")

    return args

def split_episodes(n_episodes, val_ratio, seed):
    """ Random train/val split of episode indices, reproducible for a given seed."""
    n_val = int(round(n_episodes * val_ratio))
    if val_ratio > 0:
        n_val = max(n_val, 1)
    if n_val >= n_episodes:
        raise ValueError(f"val_ratio={val_ratio} leaves no training episodes out of {n_episodes}")

    perm = np.random.default_rng(seed).permutation(n_episodes)
    train_ids = sorted(int(i) for i in perm[n_val:])
    val_ids = sorted(int(i) for i in perm[:n_val])

    return train_ids, val_ids

def count_episodes(demo_dirs):
    """ Number of ep_*.npz files over all demo dirs - the range episode_ids index into. """
    return sum(len(list(Path(d).glob("ep_*.npz"))) for d in demo_dirs)

@torch.no_grad()
def validation_loss(policy, loader, device, seed=0):
    """ Mean loss over the val set. Noise and diffusion steps are drawn from a fixed
        seed inside a forked RNG, so epochs are comparable and training noise is untouched. """
    device = torch.device(device)
    cuda_devices = []
    if device.type == "cuda":
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]

    policy.eval()
    total, n = 0.0, 0
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            B = batch["action"].shape[0]
            total += policy.compute_loss(batch).item() * B
            n += B

    return total / n if n > 0 else float("nan")

def save_checkpoint(path, policy, ema, optimizer, lr_scheduler, epoch, step, config, full=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "ema": ema.state_dict(),
        "epoch": epoch,
        "step": step,
        "config": config,
    }
    if full:
        checkpoint["policy"] = policy.state_dict()
        checkpoint["optimizer"] = optimizer.state_dict()
        checkpoint["lr_scheduler"] = lr_scheduler.state_dict()

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, tmp_path)
    tmp_path.replace(path)
def read_demo_meta(demo_dirs):
    """ meta.json of every demo dir; mixing dirs recorded with a different observation
        layout or action scale would silently train on inconsistent data. """
    metas = []
    for d in demo_dirs:
        with open(Path(d) / "meta.json") as f:
            metas.append(json.load(f))
    for key in ("obs_dim", "action_scale", "dt"):
        values = {m[key] for m in metas}
        if len(values) > 1:
            raise ValueError(f"demo dirs disagree on {key}: {sorted(values)}")
    return metas

def train_dp():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    run_dir = Path(args.output) / f"dp_{args.level}_seed{args.seed}_{datetime.now():%Y%m%d_%H%M}"
    run_dir.mkdir(parents=True, exist_ok=False)

    metas = read_demo_meta(args.demos)
    train_ids, val_ids = split_episodes(count_episodes(args.demos), args.val_ratio, args.seed)

    window = dict(horizon=16, n_obs_steps=2, n_action_steps=8)
    train_set = DemonstrationDataset(args.demos, episode_ids=train_ids, **window)
    val_set = DemonstrationDataset(args.demos, episode_ids=val_ids, **window)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    obs, action = train_set.episodes[0]
    policy_kwargs = dict(obs_dim=obs.shape[1], action_dims=action.shape[1],
                         num_train_steps=100, num_inference_steps=10, **window)
    config = {
        "policy": policy_kwargs,
        "args": vars(args),
        "demo_meta": metas,
        "train_ids": train_ids,
        "val_ids": val_ids,
        "train_windows": len(train_set),
        "val_windows": len(val_set),
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    policy = DiffusionPolicy(**policy_kwargs).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr,
                                  betas=(0.95, 0.999), weight_decay=1e-6)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=500,
                                                   num_training_steps=args.epochs * len(train_loader))
    ema = EMAModel(policy.parameters(), power=0.75)

    print(f"run dir     : {run_dir}")
    print(f"episodes    : {len(train_ids)} train / {len(val_ids)} val")
    print(f"windows     : {len(train_set)} train / {len(val_set)} val")
    print(f"steps       : {len(train_loader)}/epoch, {args.epochs * len(train_loader)} total on {device}")

    step = 0
    epoch = -1
    try:
        with open(run_dir / "metrics.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "step", "train_loss", "val_loss", "lr"])

            for epoch in range(args.epochs):
                policy.train()
                total, n = 0.0, 0
                for batch in train_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    loss = policy.compute_loss(batch)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    lr_scheduler.step()
                    ema.step(policy.parameters())
                    step += 1

                    B = batch["action"].shape[0]
                    total += loss.item() * B
                    n += B

                # Validate on the EMA weights - those are the ones evaluation uses
                ema.store(policy.parameters())
                ema.copy_to(policy.parameters())
                val_loss = validation_loss(policy, val_loader, device, seed=args.seed)
                ema.restore(policy.parameters())

                train_loss = total / n
                lr = lr_scheduler.get_last_lr()[0]
                writer.writerow([epoch + 1, step, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{lr:.3e}"])
                f.flush()

                if (epoch + 1) % 10 == 0 or epoch == 0:
                    print(f"[{epoch + 1}/{args.epochs}] step={step} "
                          f"train={train_loss:.4f} val={val_loss:.4f} lr={lr:.2e}")

                if (epoch + 1) % args.checkpoint_every == 0:
                    save_checkpoint(run_dir / "checkpoints" / f"epoch_{epoch + 1:04d}.pt",
                                    policy, ema, optimizer, lr_scheduler, epoch, step, config, full=False)
                    save_checkpoint(run_dir / "latest.pt",
                                    policy, ema, optimizer, lr_scheduler, epoch, step, config)
    except KeyboardInterrupt:
        print("interrupted - saving latest.pt")
    finally:
        if step > 0:
            save_checkpoint(run_dir / "latest.pt",
                            policy, ema, optimizer, lr_scheduler, epoch, step, config)
