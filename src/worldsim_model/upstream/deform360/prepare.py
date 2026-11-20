"""Deform360: gripper actions and closed-gripper segments for the dynamics.

Input: one latent file per episode, with `z` (T, 8, 32, 32, 32) or (T, M, 8, 32, 32, 32), `obj_occ` (T, 32, 32, 32), `raw_frame`
(T,) (the frame of the recording each latent comes from) and the attribute `uuid` (`<object>/<episode>`), and the Deform360
recordings, where `<root>/<uuid>/robot/robot.npy` holds `actions` (frames, arms, 5, 3): per arm the position, the three rows of
the rotation and the opening.

Output: one file per segment in which every gripper is closed, with the 22-number action per frame:

    per arm [contact point xyz, first two rows of the rotation, opening]; then one flag per arm, 1 if the arm exists

The contact point is the recorded position moved 17 cm along the gripper's local y axis, and everything is rotated from the
recording's y-up frame into the z-up frame of the packed points and latents. Only closed-gripper frames are kept because the
object does not move while the gripper is open: on all frames, copying the last latent is close to optimal and the model
learns to ignore the action.

    python -m deform360.prepare --latents latents/train --raw /data/deform360/processed --out segments/train
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import binary_erosion

Y_UP_TO_Z_UP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], np.float32)
CONTACT_OFFSET = np.array([0, 0.17, 0], np.float32)  # metres along the gripper's local y: lands near the fingertips
MIN_FRAMES, EDGE_FRAMES, MIN_GAP = 30, 3, 0.015


def robot_actions(raw_root, uuid):
    a = np.load(Path(raw_root) / uuid / "robot" / "robot.npy", allow_pickle=True).item()["actions"]
    return np.asarray(a[:, None] if a.ndim == 3 else a, np.float32)  # (frames, arms, 5, 3)


def gripper_actions(a):
    """(T, arms, 5, 3) recorded poses -> (T, 22) actions in the z-up frame."""
    act = np.zeros((len(a), 22), np.float32)
    for arm in range(min(a.shape[1], 2)):
        rot = Y_UP_TO_Z_UP[None] @ a[:, arm, 1:4]
        act[:, arm * 10 : arm * 10 + 3] = a[:, arm, 0] @ Y_UP_TO_Z_UP.T + rot @ CONTACT_OFFSET
        act[:, arm * 10 + 3 : arm * 10 + 9] = rot[:, :2].reshape(-1, 6)
        act[:, arm * 10 + 9] = a[:, arm, 4, 0]
        act[:, 20 + arm] = 1
    return act


def opening_levels(x):
    """The closed and the open opening of one arm over a whole episode (two-cluster medians), or None when it has only one."""
    x = x[np.isfinite(x)]
    lo, hi = np.percentile(x, [10, 90])
    for _ in range(30):
        low, high = x[x <= (lo + hi) / 2], x[x > (lo + hi) / 2]
        if min(len(low), len(high)) < 10:
            return None
        new = np.array([np.median(low), np.median(high)])
        if np.abs(new - [lo, hi]).max() < 1e-5:
            break
        lo, hi = new
    return None if hi - lo < MIN_GAP else (float(lo), float(hi))


def closed_frames(a):
    """(frames, arms, 5, 3) -> bool (frames,): every arm closed, i.e. below closed + 15 % of the way to open. None if an arm
    never opens or never closes."""
    ok = np.isfinite(a).all((1, 2, 3))
    for arm in range(a.shape[1]):
        levels = opening_levels(a[:, arm, 4, 0])
        if levels is None:
            return None
        ok &= (a[:, arm, 4, 0] >= 0) & (a[:, arm, 4, 0] <= levels[0] + 0.15 * (levels[1] - levels[0]))
    return ok


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--latents", required=True, help="folder of per-episode latent files")
    p.add_argument("--raw", required=True, help="Deform360 recordings: <raw>/<uuid>/robot/robot.npy")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    n = frames = 0
    for fp in sorted(Path(a.latents).glob("*.h5")):
        with h5py.File(fp, "r") as f:
            uuid, raw = str(f.attrs["uuid"]), f["raw_frame"][:]
            poses = robot_actions(a.raw, uuid)
            closed = closed_frames(poses)
            if closed is None:
                print(f"skip {uuid}: no distinct open and closed gripper")
                continue
            valid = binary_erosion(closed[raw], structure=np.ones(2 * EDGE_FRAMES + 1), border_value=0)
            ids = np.flatnonzero(valid)
            action = gripper_actions(poses[raw])
            for seg in np.split(ids, np.flatnonzero((np.diff(ids) != 1) | (np.diff(raw[ids]) != 1)) + 1):
                if len(seg) < MIN_FRAMES:
                    continue
                b, e = int(seg[0]), int(seg[-1]) + 1
                with h5py.File(out / f"segment_{n:05d}.h5", "w") as g:
                    z = f["z"][b:e]
                    g.create_dataset("z", data=z[:, None] if z.ndim == 5 else z, chunks=(1,) + (z.shape[1:] if z.ndim == 6 else (1,) + z.shape[1:]))
                    g.create_dataset("action", data=action[b:e])
                    g.create_dataset("obj_occ", data=f["obj_occ"][b:e], compression="lzf")
                    g.create_dataset("raw_frame", data=raw[b:e])
                    g.attrs.update(uuid=uuid, source=str(fp))
                n, frames = n + 1, frames + len(seg)
    (out / "summary.json").write_text(json.dumps(dict(segments=n, frames=frames)))
    print(f"{n} segments, {frames} frames -> {out}")


if __name__ == "__main__":
    main()
