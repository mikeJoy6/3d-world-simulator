"""Seed / rescale cam_crops.json for a dataset whose cameras run at a different
resolution than the template's (e.g. cloth_v3 fixed cams 848x480 and wrists
848x480 vs the 2026-10 collections at 1280x720 fixed / 640x360 wrist).

Each camera's crop rectangle is scaled by (W_ds / W_tpl, H_ds / H_tpl) measured
from the raw episode files, so the crop covers the same field of view, then
clamped to the image.  Image sizes come from the first episode of each dataset.

    python -m processing.pack.seed_crops --dataset bimanual_box_v2 --template bimanual_cloth_v3
    python -m processing.pack.seed_crops --dataset bimanual_rope_free --clamp     # only clamp to the image size
"""
import argparse, glob, json, os, re


import h5video

from processing import paths


def image_sizes(root):
    """{cam_index: (W, H)} from the first raw episode under <root>/raw (or <root>)."""
    raw = os.path.join(root, "raw") if os.path.isdir(os.path.join(root, "raw")) else root
    files = sorted(glob.glob(os.path.join(raw, "episode_*.hdf5")),
                   key=lambda p: int(re.search(r"(\d+)", os.path.basename(p)).group(1)))
    if not files:
        raise SystemExit(f"no episode_*.hdf5 under {raw}")
    out = {}
    with h5video.File(files[0], "r") as h:
        g = h["observations/images"]
        for k in g:
            m = re.fullmatch(r"camera_(\d+)_color", k)
            if m:
                H, W = g[k].shape[1:3]
                out[int(m.group(1))] = (int(W), int(H))
    return out


def scale_crops(crops, src_sizes, dst_sizes):
    out = {}
    for key, rect in crops.items():
        m = re.fullmatch(r"camera_(\d+)", key)
        if not m:
            out[key] = rect
            continue
        c = int(m.group(1))
        if c not in dst_sizes:
            continue                                  # camera absent in this dataset
        Wd, Hd = dst_sizes[c]
        if c in src_sizes:
            Ws, Hs = src_sizes[c]
            sx, sy = Wd / Ws, Hd / Hs
        else:
            sx = sy = 1.0
        x0, y0, x1, y1 = rect
        x0, x1 = int(round(x0 * sx)), int(round(x1 * sx))
        y0, y1 = int(round(y0 * sy)), int(round(y1 * sy))
        x0 = max(0, min(x0, Wd - 8)); y0 = max(0, min(y0, Hd - 8))
        x1 = max(x0 + 8, min(x1, Wd)); y1 = max(y0 + 8, min(y1, Hd))
        out[key] = [x0, y0, x1, y1]
    for c, (W, H) in dst_sizes.items():               # cameras the template did not have
        out.setdefault(f"camera_{c}", [0, 0, W, H])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--template", help="dataset whose cam_crops.json is scaled (default: the dataset's own file)")
    ap.add_argument("--clamp", action="store_true", help="no scaling, only clamp the existing rects to the images")
    ap.add_argument("--out", help="output path (default <dataset>/cam_crops.json)")
    a = ap.parse_args()
    root = str(paths.root(a.dataset))
    dst = image_sizes(root)
    if a.template and not a.clamp:
        troot = str(paths.root(a.template))
        src = image_sizes(troot)
        crops = json.load(open(os.path.join(troot, "cam_crops.json")))
    else:
        src = dst
        crops = json.load(open(os.path.join(root, "cam_crops.json")))
    new = scale_crops(crops, src, dst)
    out = a.out or os.path.join(root, "cam_crops.json")
    for k in sorted(new):
        c = int(k.split("_")[1])
        print(f"{k}: {crops.get(k)} @{src.get(c)} -> {new[k]} @{dst.get(c)}")
    json.dump(new, open(out, "w"), indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
