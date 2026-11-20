"""Train the Deform360 dynamics (deform360/model.py) on closed-gripper latent segments (deform360/prepare.py).

    python -m deform360.train --latent-dir segments/train --val-latent-dir segments/val --out-dir runs/deform360/dynamics

Each step predicts the next latent from five history latents and one reference frame; the prediction is fed back as history
for `--ar-schedule` further steps. The loss weighs the object (with a one-voxel band) and the rest of the grid equally, with
5x weight in a 9^3-cell box around each gripper.
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deform360.data import SegmentDataset
from deform360.model import N_HISTORY, SpatialCoarse

MAX_HIST_NOISE, INFER_HIST_NOISE, DISCOUNT, GRIPPER_WEIGHT = 0.04, 0.02, 0.95, 5.0


def balanced_mse(pred, target, region, gripper):
    """Mean error inside `region` and outside it, averaged 50/50 per example; cells in `gripper` count 5x in both."""
    err = (pred.float() - target.float()).square().mean(1)
    mask = region.bool()
    w = 1 + (GRIPPER_WEIGHT - 1) * gripper.to(err)
    inside, outside = (w * mask).flatten(1).sum(1), (w * ~mask).flatten(1).sum(1)
    fg = (err * w * mask).flatten(1).sum(1) / inside.clamp_min(1)
    bg = (err * w * ~mask).flatten(1).sum(1) / outside.clamp_min(1)
    whole = (err * w).flatten(1).sum(1) / w.flatten(1).sum(1)
    return torch.where((inside > 0) & (outside > 0), 0.5 * (fg + bg), whole).mean()


def step(model, z_hist, actions, anchor, anchor_action, noise):
    """z_hist (B, 5, ...), actions (B, 6, 23) of the history and the target frame, noise (B, 5) -> next latent (B, 8, 32, 32, 32).
    The history is expected to already carry that noise."""
    B = len(z_hist)
    z = torch.cat([z_hist, torch.randn_like(z_hist[:, :1])], 1)
    return model(z, actions, torch.cat([noise, torch.ones(B, 1, device=z.device)], 1), anchor, anchor_action)


def rollout_loss(model, batch, max_ar, device, train):
    z, act = batch["z_seq"].to(device, non_blocking=True), batch["actions"].to(device, non_blocking=True)
    anchor, anchor_act = batch["anchor"].to(device), batch["anchor_action"].to(device)
    obj = batch["obj_occ"].to(device)
    obj = (F.max_pool3d(obj.flatten(0, 1)[:, None], 3, 1, 1) > 0.5).reshape(obj.shape)  # one-voxel band
    z_hist, total = z[:, :N_HISTORY].clone(), 0.0
    for ar in range(1 + max_ar):
        t = N_HISTORY + ar
        if train:
            noise = torch.rand(len(z), N_HISTORY, device=device) * MAX_HIST_NOISE
            hist = z_hist + torch.randn_like(z_hist) * noise[:, :, None, None, None, None]
        else:
            noise, hist = torch.full((len(z), N_HISTORY), INFER_HIST_NOISE, device=device), z_hist
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = step(model, hist, act[:, t - N_HISTORY : t + 1], anchor, anchor_act, noise)
        pred = pred.float()
        if train:
            loss = DISCOUNT**ar * balanced_mse(pred, z[:, t], obj[:, t], model.gripper_region(act[:, t]))
            loss.backward()
            total += loss.item()
        else:
            total += F.mse_loss(pred, z[:, t]).item() / (1 + max_ar)
        z_hist = torch.cat([z_hist, pred.detach()[:, None]], 1)[:, -N_HISTORY:]
    return total


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--latent-dir", required=True)
    p.add_argument("--val-latent-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--ar-schedule", type=int, nargs="+", default=[1, 2, 3, 4, 5], help="fed-back steps per stage (rollout length up to 5)")
    p.add_argument("--ar-schedule-epochs", type=int, nargs="+", default=[20, 40, 60, 80, 240], help="last epoch of each stage")
    p.add_argument("--width", type=int, default=176, help="176 is the 51 M model")
    p.add_argument("--lr", type=float, default=6e-5)
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--iters", type=int, default=500)
    p.add_argument("--val-iters", type=int, default=20)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--save-interval", type=int, default=10)
    p.add_argument("--resume", default=None)
    p.add_argument("--seed", type=int, default=None)
    a = p.parse_args()
    if a.seed is not None:
        random.seed(a.seed)
        np.random.seed(a.seed)
        torch.manual_seed(a.seed)
    device, out = "cuda", Path(a.out_dir)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    seq_len = N_HISTORY + 1 + max(a.ar_schedule)
    norm = out / "action_norm_params.npz"
    if not norm.exists():
        mean, std = SegmentDataset(a.latent_dir, seq_len).compute_action_norm_params()
        np.savez(norm, action_mean=mean, action_std=std)
    d = np.load(norm)
    amean, astd = d["action_mean"], d["action_std"]
    train_ds = SegmentDataset(a.latent_dir, seq_len, amean, astd, train=True)
    val_ds = SegmentDataset(a.val_latent_dir, seq_len, amean, astd, train=False)
    train_loader = DataLoader(train_ds, batch_size=a.batch_size, shuffle=True, num_workers=a.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=a.batch_size, shuffle=False, num_workers=a.num_workers, drop_last=True)
    model = SpatialCoarse(a.width).to(device)
    model.set_action_norm(amean, astd)
    print(f"[model] {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters, {len(train_ds.files)} train and {len(val_ds.files)} val segments", flush=True)
    n_epochs = a.ar_schedule_epochs[-1]
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0001)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs * a.iters, eta_min=a.min_lr)
    start, best = 0, float("inf")
    if a.resume:
        ck = torch.load(a.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        opt.load_state_dict(ck["optimizer_state_dict"])
        sched.load_state_dict(ck["scheduler_state_dict"])
        start, best = ck["epoch"] + 1, ck["best_val"]
    for epoch in range(start, n_epochs):
        model.train()
        max_ar = next((ar for end, ar in zip(a.ar_schedule_epochs, a.ar_schedule) if epoch < end), a.ar_schedule[-1])
        total, it = 0.0, iter(train_loader)
        for _ in range(a.iters):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(train_loader)
                batch = next(it)
            opt.zero_grad()
            total += rollout_loss(model, batch, max_ar, device, train=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            sched.step()
        model.eval()
        val, n = 0.0, 0
        with torch.no_grad():
            for bi, batch in enumerate(val_loader):
                if bi >= a.val_iters:
                    break
                val, n = val + rollout_loss(model, batch, max_ar, device, train=False), n + 1
        val /= max(n, 1)
        print(f"[ep {epoch}] ar {max_ar} train {total / a.iters:.5f} val {val:.5f} lr {sched.get_last_lr()[0]:.2e}", flush=True)
        ck = dict(model_state_dict=model.state_dict(), optimizer_state_dict=opt.state_dict(), scheduler_state_dict=sched.state_dict(), epoch=epoch, best_val=best, action_mean=amean, action_std=astd, args=vars(a))
        torch.save(ck, out / "checkpoints" / "latest.pth")
        if val < best:
            best = ck["best_val"] = val
            torch.save(ck, out / "checkpoints" / "best.pth")
        if (epoch + 1) % a.save_interval == 0:
            torch.save(ck, out / "checkpoints" / f"model_{epoch}.pth")


if __name__ == "__main__":
    main()
