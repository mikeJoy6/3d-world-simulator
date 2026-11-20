"""Make each packed episode self-contained: one file per episode that training reads alone.

Into a copy of `packed/episode_<E>.hdf5` this writes what used to sit in side folders, then stores the image stacks as lossless
video (h5video.py):

  render/camera_<c>/obj_mask       the final object mask: the SAM3 mask of `sam3_masks/` where one exists
  render/camera_<c>/gripper_mask   from `gripper_supervision/masks_halo6_surface/`   (full recipe)
  labels/robot/<level>, labels/gripper/<level>, attrs bounds   from `robot_label_cache/`   (full recipe)
  tactile_targets/xyz, sig, ...    from `tactile_targets/`   (tactile tasks)

Side folders that are missing are skipped, and the file then supports the minimal recipe only. Nothing is changed in place:

    python scripts/consolidate_packed.py --root data/bimanual_cloth_v3 --out data_v3/bimanual_cloth_v3/packed --workers 8

The depth targets of training are the packed depth, so `fs_disp/` is not needed after packing.
"""
import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from h5_to_video import convert, verify  # noqa: E402

GZ = dict(compression="gzip", compression_opts=4, shuffle=True)


def one(job):
    root, src, dst, check = (Path(job[0]), Path(job[1]), Path(job[2]), job[3])
    if dst.exists():
        return dst.name, "exists", 0, 0
    t0, tmp = time.time(), Path(str(dst) + ".tmp")
    convert(str(src), str(tmp))
    if check:  # before anything is added: every array of the source reads back identical
        verify(str(src), str(tmp))
    added = []
    with h5py.File(tmp, "r+") as f:
        ep, T = int(f.attrs["episode"]), int(f.attrs["num_frames"])
        cams = [int(c) for c in f.attrs.get("render_cam_ids", f.attrs["cam_ids"])]
        changed = 0
        for c in cams:
            g = f.get(f"render/camera_{c}")
            box = root / "sam3_masks" / f"episode_{ep}_camera_{c}.h5"
            if g is None or not box.exists():
                continue
            h, w = int(g.attrs["H"]), int(g.attrs["W"])
            x0, y0, x1, y1 = (int(v) for v in g.attrs["crop"])
            old = g["obj_mask"][...] if "obj_mask" in g else np.zeros((T, (h * w + 7) // 8), np.uint8)
            new = old.copy()
            with h5py.File(box, "r") as s:
                for row, frame in enumerate(s["frame_indices"][:]):
                    if 0 <= frame < T:
                        new[frame] = np.packbits(s["mask_union"][row, y0:y1, x0:x1].astype(bool).reshape(-1))
            changed += int((new != old).any(1).sum())
            if "obj_mask" in g:
                del g["obj_mask"]
            g.create_dataset("obj_mask", data=new, chunks=(1, new.shape[1]), **GZ)
        if changed:
            added.append(f"object masks ({changed} frames differed)")
        side = root / "gripper_supervision" / "masks_halo6_surface" / f"episode_{ep}.h5"
        if side.exists():
            with h5py.File(side, "r") as s:
                for c in cams:
                    g = f.get(f"render/camera_{c}")
                    if g is None or f"camera_{c}" not in s:
                        continue
                    assert [int(v) for v in s[f"camera_{c}"].attrs["crop"]] == [int(v) for v in g.attrs["crop"]], f"{dst.name}: gripper mask crop of camera {c} does not match the render crop"
                    m = s[f"camera_{c}/gripper_mask"][...]
                    g.create_dataset("gripper_mask", data=m, chunks=(1, m.shape[1]), **GZ)
            added.append("gripper masks")
        side = root / "robot_label_cache" / f"episode_{ep}.h5"
        if side.exists():
            with h5py.File(side, "r") as s:
                lab = f.create_group("labels")
                lab.attrs["bounds"] = s.attrs["bounds"]
                for kind in ("robot", "gripper"):
                    s.copy(kind, lab)
            added.append("robot labels")
        side = root / "tactile_targets" / f"episode_{ep}.npz"
        if side.exists():
            z = np.load(side)
            g = f.create_group("tactile_targets")
            for k in z.files:
                if z[k].ndim == 0:
                    g.attrs[k] = z[k][()]
                else:
                    g.create_dataset(k, data=z[k], chunks=(1,) + z[k].shape[1:], **GZ)
            added.append("tactile targets")
        f.attrs["format_version"] = "packed_v3"
    os.replace(tmp, dst)
    return dst.name, ", ".join(added) or "nothing to add", os.path.getsize(src) / 1e6, os.path.getsize(dst) / 1e6, time.time() - t0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="dataset folder holding packed/ and the side folders")
    p.add_argument("--out", required=True, help="folder for the self-contained episodes")
    p.add_argument("--episodes", nargs="+", type=int, default=None)
    p.add_argument("--verify", action="store_true", help="read the converted episode back and compare every source array, bit for bit")
    p.add_argument("--workers", type=int, default=1)
    a = p.parse_args()
    root = Path(a.root)
    files = sorted((root / "packed").glob("episode_*.hdf5"), key=lambda q: int(q.stem.split("_")[1]))
    if a.episodes is not None:
        files = [f for f in files if int(f.stem.split("_")[1]) in set(a.episodes)]
    os.makedirs(a.out, exist_ok=True)
    assert os.path.realpath(a.out) != os.path.realpath(root / "packed"), "--out is the source folder"
    jobs = [(str(root), str(f), str(Path(a.out) / f.name), a.verify) for f in files]
    with ProcessPoolExecutor(a.workers) as ex:
        for r in ex.map(one, jobs) if a.workers > 1 else map(one, jobs):
            print(f"{r[0]}: {r[1]}" + (f"; {r[2]:.0f} -> {r[3]:.0f} MB, {r[4]:.0f} s" if len(r) > 4 else ""), flush=True)


if __name__ == "__main__":
    main()
