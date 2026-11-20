"""Write IR-only copies of raw ALOHA episodes for off-site stereo processing.

Keeps, per camera: left_ir, right_ir, intrinsics (K_ir), rgb_intrinsics, extrinsics, rgb_to_ir_extrinsic, baseline,
plus the small non-image datasets (joint_action, cartesian_action, stage, timestamp). Drops every *_color stream
(~60% of the bytes). run_fs.py reads the result unchanged (pass it as --src-dir).

usage: python -m processing.fs_stereo.export_ir_only --src <dir with episode_N.hdf5> --dst <dir> [--workers 8] [--episodes 0,1]
"""
import argparse, glob, os, time
from concurrent.futures import ProcessPoolExecutor
import h5py

import sys
from pathlib import Path as _P

sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
import h5video  # noqa: E402

KEEP_SUFFIX = ("_left_ir", "_right_ir", "_intrinsics", "_rgb_intrinsics", "_extrinsics", "_rgb_to_ir_extrinsic", "_baseline")


def export_one(args):
    src, dst = args
    if os.path.exists(dst):
        return dst, "exists"
    tmp = dst + ".tmp"; t0 = time.time()
    with h5video.File(src, "r") as fi, h5py.File(tmp, "w") as fo:
        for k, v in fi.attrs.items(): fo.attrs[k] = v
        for k in fi:
            if k == "observations": continue
            fi.copy(k, fo)
        obs = fi["observations"]; go = fo.create_group("observations")
        for k in obs:
            if k == "images": continue
            obs.copy(k, go)
        gi = obs["images"]; goi = go.create_group("images")
        for k in gi:
            if k.endswith("_color"): continue
            if not k.endswith(KEEP_SUFFIX): continue
            d = gi[k]
            if isinstance(d, h5video.VideoDataset):    # already stored as video: copy the packets as they are
                gi.copy(k, goi)
            elif k.endswith("_ir"):    # IR frames: keep the source chunking/compression (lzf, one frame per chunk)
                o = goi.create_dataset(k, shape=d.shape, dtype=d.dtype, chunks=d.chunks, compression=d.compression)
                for i in range(0, d.shape[0], 32): o[i:i + 32] = d[i:i + 32]
            else:
                goi.create_dataset(k, data=d[()])
            for ak, av in d.attrs.items(): goi[k].attrs[ak] = av
    os.replace(tmp, dst)
    return dst, f"{os.path.getsize(dst) / 2**30:.2f} GB in {time.time() - t0:.0f}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True); ap.add_argument("--dst", required=True)
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--episodes", default="all")
    a = ap.parse_args()
    os.makedirs(a.dst, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.src, "episode_*.hdf5")), key=lambda p: int(os.path.basename(p).split("_")[1].split(".")[0]))
    if a.episodes != "all":
        want = {int(x) for x in a.episodes.split(",")}
        files = [f for f in files if int(os.path.basename(f).split("_")[1].split(".")[0]) in want]
    jobs = [(f, os.path.join(a.dst, os.path.basename(f))) for f in files]
    print(f"{len(jobs)} episodes {a.src} -> {a.dst} ({a.workers} workers)", flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        for dst, msg in ex.map(export_one, jobs):
            print(f"  {os.path.basename(dst)}: {msg}", flush=True)


if __name__ == "__main__":
    main()
