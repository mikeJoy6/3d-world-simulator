"""Gripper loss-ROI masks: splat silhouette + depth surfaces within a 6-fine-voxel halo.

Expand the cached gripper voxel volume (robot_label_cache/episode_E.h5,
gripper/2 = finest grid) by 6 fine voxels, keep the observed depth surfaces
(packed depth, valid_mask) inside it, project them into each cropped RGB
camera (packed render/ K + viewmat), 2px raster fill, and union with the
original splat silhouette (gripper_supervision/masks/episode_E.h5).

Writes gripper_supervision/masks_halo6_surface/episode_E.h5 with
camera_<c>/gripper_mask (T, ceil(Hc*Wc/8)) bit-packed, crop attrs copied from
render/.  These are LOSS WEIGHTS only, never segmentation/opacity/geometry targets.

Both inputs (robot_label_cache/, gripper_supervision/masks/) come from the
external arm-splat fitting step (see processing/README.md).

Usage:
    python scripts/bake_gripper_halo.py --root <PW_DATA_ROOT>/bimanual_box --workers 8
"""
import argparse
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import h5py

import sys
from pathlib import Path as _P

sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import h5video  # noqa: E402
import numpy as np
from scipy.ndimage import maximum_filter

HALO = 6  # fine voxels


def bake(args):
    root, ep = args
    R = Path(root)
    cv2.setNumThreads(1)
    out = R / "gripper_supervision/masks_halo6_surface"
    dst = out / f"episode_{ep}.h5"
    if dst.exists():
        return f"{ep}: exists"
    with h5video.File(R / "robot_label_cache" / f"episode_{ep}.h5") as cache, \
            h5video.File(R / "packed" / f"episode_{ep}.hdf5") as pack, \
            h5video.File(R / "gripper_supervision/masks" / f"episode_{ep}.h5") as old, \
            h5py.File(dst.with_suffix(".tmp"), "w") as f:
        ds = cache["gripper/2"]
        shape = tuple(ds.attrs["shape"])                 # (nz, ny, nx)
        lo, hi = cache.attrs["bounds"]
        voxel = (hi - lo) / np.array(shape[::-1])
        n = len(ds)
        cams = [int(c) for c in pack.attrs["cam_ids"]]
        dest, gcam = {}, {}
        for c in cams:
            g = pack[f"render/camera_{c}"]
            gcam[c] = g
            h, w = int(g.attrs["H"]), int(g.attrs["W"])
            sg = f.create_group(f"camera_{c}")
            for k, v in g.attrs.items():
                sg.attrs[k] = v
            dest[c] = sg.create_dataset("gripper_mask", (n, (h * w + 7) // 8), dtype="u1",
                                        chunks=(1, (h * w + 7) // 8),
                                        compression="gzip", compression_opts=1)
        for t in range(n):
            grid = np.unpackbits(ds[t], count=int(np.prod(shape))).reshape(shape)
            grid = maximum_filter(grid, size=2 * HALO + 1, mode="constant")
            for c, g in gcam.items():
                h, w = int(g.attrs["H"]), int(g.attrs["W"])
                mask = np.unpackbits(old[f"camera_{c}/gripper_mask"][t], count=h * w).reshape(h, w).copy()
                vm = g["viewmat"][t]
                K = g["K"][t]
                depth = pack[f"img/camera_{c}/depth"][t].astype(np.float32)
                valid = np.unpackbits(pack[f"img/camera_{c}/valid_mask"][t],
                                      count=depth.size).reshape(depth.shape) > 0
                yy, xx = np.nonzero(valid & (depth > 0) & np.isfinite(depth))
                z = depth[yy, xx]
                Ki = pack[f"cam/camera_{c}/K_ir"][t]
                E = pack[f"cam/camera_{c}/E_cam2world"][t]
                ir = np.column_stack(((xx - Ki[0, 2]) * z / Ki[0, 0], (yy - Ki[1, 2]) * z / Ki[1, 1], z))
                world = ir @ E[:3, :3].T + E[:3, 3]
                ij = np.floor((world - lo) / voxel).astype(np.int32)
                inside = ((ij >= 0) & (ij < np.array(shape[::-1]))).all(1)
                world, ij = world[inside], ij[inside]
                near = grid[ij[:, 2], ij[:, 1], ij[:, 0]] > 0
                camera = world[near] @ vm[:3, :3].T + vm[:3, 3]
                camera = camera[camera[:, 2] > .001]
                uv = camera @ K.T
                uv = uv[:, :2] / uv[:, 2:]
                uv = np.rint(uv).astype(np.int32)
                uv = uv[(uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)]
                dots = np.zeros((h, w), np.uint8)
                dots[uv[:, 1], uv[:, 0]] = 1
                mask |= cv2.dilate(dots, np.ones((5, 5), np.uint8))
                dest[c][t] = np.packbits(mask.ravel())
        f.attrs["frames"] = n
        f.attrs["episode"] = ep
        f.attrs["halo_fine_voxels"] = HALO
        f.attrs["halo_meters"] = HALO * voxel
        f.attrs["source"] = ("Observed depth surfaces within the calibrated gripper voxel volume "
                             f"expanded by {HALO} fine voxels, projected into native cropped RGB; "
                             "2px raster fill; union original splat silhouette. LOSS ROI ONLY.")
    os.replace(dst.with_suffix(".tmp"), dst)
    return f"{ep}: complete"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("PACK_ROOT"),
                    help="dataset root (<PW_DATA_ROOT>/<ds>); env PACK_ROOT")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--episodes", default="all", help="'all' (every packed episode) or 0,1,2")
    a = ap.parse_args()
    if not a.root:
        ap.error("--root (or PACK_ROOT) required")
    R = Path(a.root)
    (R / "gripper_supervision/masks_halo6_surface").mkdir(parents=True, exist_ok=True)
    if a.episodes == "all":
        eps = sorted(int(re.search(r"(\d+)", p.name).group(1))
                     for p in (R / "packed").glob("episode_*.hdf5"))
    else:
        eps = list(map(int, a.episodes.split(",")))
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for msg in pool.map(bake, [(str(R), e) for e in eps]):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
