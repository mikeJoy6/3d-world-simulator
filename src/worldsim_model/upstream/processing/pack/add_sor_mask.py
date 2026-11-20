"""Bake a per-pixel SOR filter mask into packed episodes.

For every frame: project all cameras' valid pixels into the fused world cloud
(exactly as the loader does), run statistical-outlier removal (SOR) on the fused
cloud, then write back a per-camera bit-packed `filter_mask` = (valid AND kept).
The loader can then project `filter_mask` pixels instead of `valid_mask` to get
the SOR-cleaned cloud, with no load-time SOR cost.

SOR is run on the SCENE cloud only (tactile is added at load, never filtered).

Usage (what `./pw pack` runs):
    python -m processing.pack.add_sor_mask --dataset bimanual_cloth_v3 \
        --sor-k 10 --sor-std 2.0 --workers 24 --sor-threads 8
"""
import argparse, glob, os, sys, time
import multiprocessing as mp
from pathlib import Path
import numpy as np

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parents[2]))
import h5video  # noqa: E402
from processing import paths                          # noqa: E402
from processing.pc.geometry import sor_mask           # noqa: E402


def _unpack(packed_row, H, W):
    return np.unpackbits(packed_row)[: H * W].reshape(H, W).astype(bool)


def _lift(depth, vy, vx, K, E):
    z = depth[vy, vx].astype(np.float32)
    fx, fy = float(K[0, 0]), float(K[1, 1]); cx, cy = float(K[0, 2]), float(K[1, 2])
    x = (vx.astype(np.float32) - cx) * z / fx
    y = (vy.astype(np.float32) - cy) * z / fy
    pir = np.stack([x, y, z], -1)
    R = E[:3, :3].astype(np.float32); t = E[:3, 3].astype(np.float32)
    return (pir @ R.T + t).astype(np.float32)


def _process_episode(t):
    fp, sor_k, sor_std, n_threads = t
    t0 = time.time()
    try:
        with h5video.File(fp, "r+") as f:
            T = int(f.attrs["num_frames"])
            cam_ids = [int(c) for c in f.attrs["cam_ids"]]
            fds = {}
            HWc = {}
            for c in cam_ids:
                key = f"img/camera_{c}/filter_mask"
                if key in f:
                    del f[key]
                HWc[c] = int(np.prod(f[f"img/camera_{c}/depth"].shape[1:3]))   # per-camera size
                fds[c] = f.create_dataset(key, (T, (HWc[c] + 7) // 8), dtype="uint8",
                                          chunks=(1, (HWc[c] + 7) // 8),
                                          compression="gzip", compression_opts=4)
            kept_tot = tot = 0
            for ti in range(T):
                per_cam, pts_list = [], []
                for c in cam_ids:
                    depth = f[f"img/camera_{c}/depth"][ti].astype(np.float32)
                    valid = _unpack(f[f"img/camera_{c}/valid_mask"][ti], *depth.shape)
                    vy, vx = np.nonzero(valid)
                    K = f[f"cam/camera_{c}/K_ir"][ti]; E = f[f"cam/camera_{c}/E_cam2world"][ti]
                    per_cam.append((c, vy, vx, depth.shape))
                    pts_list.append(_lift(depth, vy, vx, K, E) if len(vy) else np.zeros((0, 3), np.float32))
                pts = np.concatenate(pts_list) if pts_list else np.zeros((0, 3), np.float32)
                keep = sor_mask(pts, sor_k, sor_std, n_threads=n_threads) if len(pts) > sor_k else np.ones(len(pts), bool)
                off = 0
                for (c, vy, vx, hw), p in zip(per_cam, pts_list):
                    n = len(p); kc = keep[off:off + n]; off += n
                    fm = np.zeros(hw, bool)
                    if n:
                        fm[vy[kc], vx[kc]] = True
                    fds[c][ti] = np.packbits(fm.reshape(-1))
                kept_tot += int(keep.sum()); tot += len(pts)
            f.attrs["filter_kind"] = "sor"
            f.attrs["sor_k"] = int(sor_k); f.attrs["sor_std"] = float(sor_std)
        return (os.path.basename(fp), True, kept_tot, tot, time.time() - t0, "")
    except Exception as e:
        return (os.path.basename(fp), False, 0, 0, time.time() - t0, f"{type(e).__name__}: {str(e)[:160]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--packed-dir", default=None,
                    help="default: <PW_DATA_ROOT>/<dataset>/packed")
    ap.add_argument("--sor-k", type=int, default=10)
    ap.add_argument("--sor-std", type=float, default=2.0)
    ap.add_argument("--episodes", default="all")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--sor-threads", type=int, default=4,
                    help="cKDTree.query threads PER worker process. With W workers "
                         "on a C-core box use ~max(1, C//W) to avoid oversubscription "
                         "(sor_mask's default workers=-1 spawns ALL cores per process).")
    args = ap.parse_args()

    pdir = args.packed_dir or str(paths.packed_dir(args.dataset))
    files = sorted(glob.glob(os.path.join(pdir, "episode_*.hdf5")),
                   key=lambda p: int(os.path.basename(p).split("_")[1].split(".")[0]))
    if args.episodes != "all":
        want = set()
        for part in args.episodes.split(","):
            part = part.strip()
            if "-" in part:
                a, b = part.split("-", 1); want.update(range(int(a), int(b) + 1))
            elif part:
                want.add(int(part))
        files = [p for p in files if int(os.path.basename(p).split("_")[1].split(".")[0]) in want]

    tasks = [(p, args.sor_k, args.sor_std, args.sor_threads) for p in files]
    workers = max(1, min(args.workers, len(tasks)))
    print(f"SOR filter: {len(tasks)} eps  k={args.sor_k} std={args.sor_std}  workers={workers}", flush=True)
    print(f"  packed dir: {pdir}", flush=True)

    done = 0
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as pool:
        for name, ok, kept, tot, dt, err in pool.imap_unordered(_process_episode, tasks):
            done += 1
            if ok:
                pct = 100 * (tot - kept) / max(tot, 1)
                print(f"[{done}/{len(tasks)}] {name}: OK removed {pct:.1f}% ({tot-kept}/{tot}) ({dt:.0f}s)", flush=True)
            else:
                print(f"[{done}/{len(tasks)}] {name}: FAILED {err} ({dt:.0f}s)", flush=True)
    print("[sor] DONE", flush=True)


if __name__ == "__main__":
    main()
