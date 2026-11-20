"""Deform360 latent segments for the dynamics: closed-gripper segments, random frame strides, one reference frame.

One file per segment (written by deform360/prepare.py):

    z        (T, M, 8, 32, 32, 32) float16   latents; M encodings of the same frame (colour variants), one is drawn per sample
    action   (T, 22) float32                 see deform360/prepare.py
    obj_occ  (T, 32, 32, 32) uint8           cells that hold the object
"""
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

MAX_STRIDE = 5


def with_stride(action, stride):
    return np.concatenate([np.asarray(action, np.float32), np.full((len(action), 1), stride, np.float32)], 1)


class SegmentDataset(Dataset):

    def __init__(self, folder, seq_len, action_mean=None, action_std=None, train=True):
        self.files = sorted(Path(folder).glob("*.h5"))
        assert self.files, f"no latent segments in {folder}"
        self.seq_len, self.train, self.handles, self.index = seq_len, train, {}, []
        for ei, fp in enumerate(self.files):
            with h5py.File(fp, "r") as f:
                self.index += [(ei, s) for s in range(max(0, len(f["z"]) - seq_len + 1))]
        self.action_mean = None if action_mean is None else np.asarray(action_mean, np.float32)
        self.action_std = None if action_std is None else np.asarray(action_std, np.float32)

    def __len__(self):
        return len(self.index)

    def compute_action_norm_params(self):
        """Mean and std of the 22 action numbers over all segments (near-constant ones keep std 1); the stride is left as it is."""
        a = np.concatenate([h5py.File(fp, "r")["action"][:].astype(np.float32) for fp in self.files], 0)
        std = a.std(0)
        std[std < 0.02] = 1.0
        return np.r_[a.mean(0), 0].astype(np.float32), np.r_[std, 1].astype(np.float32)

    def __getitem__(self, i):
        ei, base = self.index[i]
        if ei not in self.handles:
            self.handles[ei] = h5py.File(self.files[ei], "r")
        f = self.handles[ei]
        T = len(f["z"])
        rng = np.random.default_rng(None if self.train else 1200 + i)
        # training draws a stride that fits the segment, then a start; validation is always stride 1 from a fixed start
        d = int(rng.integers(1, min(MAX_STRIDE, (T - 1) // (self.seq_len - 1)) + 1)) if self.train else 1
        s = int(rng.integers(0, T - (self.seq_len - 1) * d)) if self.train else base
        ix = s + np.arange(self.seq_len) * d
        ref = int(rng.integers(0, s + 4 * d + 1))  # the reference: any frame up to the last history frame
        enc = int(rng.integers(f["z"].shape[1])) if self.train else 0
        raw = f["action"][:]
        act, anchor_act = with_stride(raw[ix], d), with_stride(raw[[ref]], 1)
        if self.action_mean is not None:
            act, anchor_act = (act - self.action_mean) / self.action_std, (anchor_act - self.action_mean) / self.action_std
        return dict(
            z_seq=torch.from_numpy(np.stack([f["z"][t, enc] for t in ix]).astype(np.float32)),
            actions=torch.from_numpy(act),
            anchor=torch.from_numpy(f["z"][ref, enc].astype(np.float32)[None]),
            anchor_action=torch.from_numpy(anchor_act),
            obj_occ=torch.from_numpy(f["obj_occ"][ix].astype(np.float32)),
        )
