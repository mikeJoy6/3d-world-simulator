"""Re-store the image stacks of HDF5 episodes as lossless video (see h5video.py). Nothing is changed in place: every file is
written new under --out, and --verify decodes it again and compares every array with the source, bit for bit.

    python scripts/h5_to_video.py data/bimanual_cloth_v3/raw/episode_0.hdf5 --out data_v/bimanual_cloth_v3/raw --verify
    python scripts/h5_to_video.py data/bimanual_cloth_v3/fs_disp --out data_v/bimanual_cloth_v3/fs_disp --verify --workers 8

Colour, infrared, depth and disparity stacks become video. Every other dataset is copied as it is, except large arrays stored
without any compression (the packed render masks), which get gzip. Files read exactly as before through h5video.File.
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
import h5video  # noqa: E402


def as_video(name, ds):
    """Colour, depth and disparity stacks always; 8-bit single-channel stacks only when they are infrared images (masks stay gzip)."""
    return h5video.suitable(ds) and (ds.ndim == 4 or ds.dtype != np.uint8 or name.endswith("_ir"))


def convert(src, dst):
    """-> dict of megabytes: source file, new file, and the source bytes that went to video."""
    video = 0
    tmp = str(dst) + ".part"
    with h5py.File(src, "r") as s, h5py.File(tmp, "w") as d:
        for k, v in s.attrs.items():
            d.attrs[k] = v

        def visit(name, o):
            nonlocal video
            parent = d.require_group(os.path.dirname(name)) if os.path.dirname(name) else d
            base = os.path.basename(name)
            if isinstance(o, h5py.Group):
                g = d.require_group(name)
                for k, v in o.attrs.items():
                    g.attrs[k] = v
                return
            if as_video(name, o):
                out = h5video.write(parent, base, o)
                for k, v in o.attrs.items():
                    out.attrs[k] = v
                video += o.id.get_storage_size()
            elif o.compression is None and o.nbytes > (1 << 20) and o.dtype.kind in "biuf" and o.ndim >= 1:
                out = parent.create_dataset(base, data=o[...], compression="gzip", compression_opts=4, shuffle=True, chunks=(1,) + o.shape[1:] if o.ndim > 1 else True)
                for k, v in o.attrs.items():
                    out.attrs[k] = v
            else:
                s.copy(o, parent, name=base)

        s.visititems(visit)
    os.replace(tmp, dst)
    return dict(src=os.path.getsize(src) / 1e6, dst=os.path.getsize(dst) / 1e6, video=video / 1e6)


def verify(src, dst):
    """Every dataset of ``src`` equals the one in ``dst``, video or not. -> number of datasets compared."""
    n = 0
    with h5py.File(src, "r") as s, h5video.File(dst, "r") as d:
        names = []
        s.visititems(lambda name, o: names.append(name) if isinstance(o, h5py.Dataset) else None)
        for name in names:
            a, b = s[name], d[name]
            assert tuple(a.shape) == tuple(b.shape) and a.dtype == b.dtype, f"{name}: {a.shape} {a.dtype} vs {b.shape} {b.dtype}"
            if isinstance(b, h5video.VideoDataset):
                for t in range(a.shape[0]):
                    x, y = a[t], b[t]
                    assert x.tobytes() == y.tobytes(), f"{name} frame {t} differs"
            elif a.dtype.kind == "O":
                x, y = a[...], b[...]
                assert all(np.array_equal(p, q) for p, q in zip(x.reshape(-1), y.reshape(-1))), f"{name} differs"
            else:
                assert a.shape == () and a[()] == b[()] or np.array_equal(a[...], b[...], equal_nan=a.dtype.kind == "f"), f"{name} differs"
            n += 1
    return n


def one(job):
    src, dst, check = job
    t = time.time()
    if not os.path.exists(dst):
        r = convert(src, dst)
    else:
        r = dict(src=os.path.getsize(src) / 1e6, dst=os.path.getsize(dst) / 1e6, video=0)
    r["n"] = verify(src, dst) if check else 0
    r["name"], r["s"] = os.path.basename(src), time.time() - t
    return r


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src", nargs="+", help="HDF5 files, or folders of them")
    p.add_argument("--out", required=True, help="folder for the new files (same file names)")
    p.add_argument("--verify", action="store_true", help="decode the new file and compare it with the source, bit for bit")
    p.add_argument("--workers", type=int, default=1)
    a = p.parse_args()
    files = []
    for s in a.src:
        s = Path(s)
        files += sorted(f for f in s.iterdir() if f.suffix in (".hdf5", ".h5")) if s.is_dir() else [s]
    os.makedirs(a.out, exist_ok=True)
    jobs = [(str(f), str(Path(a.out) / f.name), a.verify) for f in files]
    assert all(os.path.realpath(s) != os.path.realpath(d) for s, d, _ in jobs), "--out is the source folder"
    tot = [0.0, 0.0]
    with ProcessPoolExecutor(a.workers) as ex:
        for r in ex.map(one, jobs) if a.workers > 1 else map(one, jobs):
            tot[0] += r["src"]
            tot[1] += r["dst"]
            print(f"{r['name']}: {r['src']:.0f} -> {r['dst']:.0f} MB ({100 * r['dst'] / r['src']:.0f}%), {r['s']:.0f} s" + (f", {r['n']} arrays identical" if a.verify else ""), flush=True)
    print(f"total: {tot[0] / 1e3:.2f} -> {tot[1] / 1e3:.2f} GB ({100 * tot[1] / max(tot[0], 1e-9):.0f}%)")


if __name__ == "__main__":
    main()
