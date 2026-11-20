"""Bake render supervision INTO the packed episode files (centralized format).

Writes a `render/camera_{c}/` group into each packed episode so training reads
ONLY packed/ — no separate render/ dir, no raw/fs_disp/sam3 at train time:
  render/camera_{c}/jpg      (T,) vlen uint8   JPEG of cropped (+cam2 blackout +LRC) COLOR RGB
  render/camera_{c}/K        (T,3,3) float32   crop-shifted RGB intrinsics
  render/camera_{c}/viewmat  (T,4,4) float32   inv(E_packed @ rgb_to_ir); NaN = skip cam
  render/camera_{c}/obj_mask (T, ceil(Hc*Wc/8)) uint8  bit-packed SAM3 object mask,
                             cropped to the SAME window as the RGB (only if masks exist)
  render/camera_{c}/valid_mask (T, ceil(Hc*Wc/8)) uint8  supervisable pixels; only for
                             blackout / height-mask cams (no-source holes = 0)

obj mask is the SAM3 mask_union (SAM3 segments camera_X_color, i.e. it is
already in the colour frame), cropped identically to the RGB. Additive write
(mode 'a'): the point-cloud datasets are untouched. Resumable via --skip-existing.

Usage (what `./pw pack` runs):
    python -m processing.pack.add_render_to_packed --dataset bimanual_cloth_v3 \
        --blackout-cams 2 --workers 32 --skip-existing
"""
import argparse
import glob
import json
import multiprocessing as mp
import os
import re
import sys
import time
from pathlib import Path

import cv2
import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import h5video  # noqa: E402
from processing import paths  # noqa: E402


def _sam3_map(sam3_dir, ep_num, c):
    """Return (h5, {src_frame_idx: mask_row}) for episode/camera, or (None, None)."""
    p = os.path.join(sam3_dir, f"episode_{ep_num}_camera_{c}.h5")
    if not os.path.exists(p):
        return None, None
    h = h5video.File(p, "r")
    fi = h["frame_indices"][:]
    return h, {int(v): int(i) for i, v in enumerate(fi)}


def frame_keep_and_valid(disp, Kir, Krgb, Trgb2ir, E, baseline, box, H, W):
    """(keep, valid) bool masks for a blackout cam.
    keep  — pixel keeps its real color (has an IR source AND its world point is
            inside `box`); everything else is blacked in the stored jpg.
    valid — pixel is SUPERVISABLE: it has an IR source (real color or true
            outside-box background). No-source (LRC/reprojection-gap) pixels are
            invalid — their appearance is unknown, so the render loss must skip
            them rather than treat the black fill as ground truth."""
    x0, x1, y0, y1, z0, z1 = box
    M = E @ Trgb2ir
    if not np.isfinite(M).all() or abs(np.linalg.det(M[:3, :3])) < 1e-9:
        # zeroed/degenerate extrinsics frame (recording artifact): no usable
        # geometry -> nothing kept, nothing supervisable.
        none = np.zeros((H, W), bool)
        return none, none
    z = Kir[0, 0] * baseline / np.clip(disp.astype(np.float32), 1e-3, None)
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    x = (u - Kir[0, 2]) / Kir[0, 0] * z
    y = (v - Kir[1, 2]) / Kir[1, 1] * z
    world = (np.stack([x, y, z], -1).reshape(-1, 3) @ E[:3, :3].T + E[:3, 3])
    inside = ((world[:, 0] > x0) & (world[:, 0] < x1) & (world[:, 1] > y0)
              & (world[:, 1] < y1) & (world[:, 2] > z0) & (world[:, 2] < z1))
    Vr = np.linalg.inv(E @ Trgb2ir)
    pc = world @ Vr[:3, :3].T + Vr[:3, 3]
    zz = pc[:, 2]
    ok = zz > 0.05
    uu = np.round(Krgb[0, 0] * pc[:, 0] / np.clip(zz, 1e-6, None) + Krgb[0, 2]).astype(np.int32)
    vv = np.round(Krgb[1, 1] * pc[:, 1] / np.clip(zz, 1e-6, None) + Krgb[1, 2]).astype(np.int32)
    ok &= (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
    idx = np.nonzero(ok)[0]
    order = idx[np.argsort(-zz[idx])]
    src = np.full((H, W), -1, np.int64)
    src[vv[order], uu[order]] = order
    keep = np.where(src >= 0, inside[np.clip(src, 0, None)], False)
    keep = cv2.medianBlur(keep.astype(np.uint8) * 255, 5) > 127
    valid = cv2.medianBlur((src >= 0).astype(np.uint8) * 255, 5) > 127
    return keep, valid


def bake_episode(packed_path, raw_dir, disp_dir, sam3_dir, crops, box, blackout, quality,
                 height_cams=(), height_max=None):
    with h5video.File(packed_path, "r") as p:
        src = str(p.attrs.get("src_episode", os.path.basename(packed_path)))
        cams = [int(x) for x in p.attrs["cam_ids"]]
        E_all = {c: p[f"cam/camera_{c}/E_cam2world"][:].astype(np.float64) for c in cams}
        # episode number for the SAM3 filename: prefer the packed attr, else the src name
        if "episode" in p.attrs:
            ep_num = int(p.attrs["episode"])
        else:
            ep_num = int(re.search(r"(\d+)", src).group(1))
    r = h5video.File(os.path.join(raw_dir, src), "r")
    d = h5video.File(os.path.join(disp_dir, f"disp_{src}"), "r")
    T = r[f"observations/images/camera_{cams[0]}_color"].shape[0]
    vt = h5py.vlen_dtype(np.uint8)
    # per-cam color calibration, matching what pack_dataset applies to the
    # cloud RGB (render targets must live in the same color space as the
    # cloud colors).
    ccal = {}
    cfg_p = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(packed_path))), "config.json")
    if os.path.exists(cfg_p):
        cfgj = json.load(open(cfg_p))
        for k, v in (cfgj.get("color_calib") or {}).items():
            if k.startswith("camera_"):
                ccal[int(k.split("_")[1])] = (
                    np.asarray(v.get("gain", [1, 1, 1]), np.float32),
                    np.asarray(v.get("offset", [0, 0, 0]), np.float32))

    with h5video.File(packed_path, "a") as o:
        g0 = o.require_group("render")
        g0.attrs["table_bbox"] = json.dumps(box)
        g0.attrs["blackout_cams"] = blackout
        g0.attrs["height_mask_cams"] = list(height_cams)
        if height_max is not None:
            g0.attrs["height_mask_zmax"] = float(height_max)
        # "unrepresentable" box for the height-mask cams: ABOVE height_max AND
        # inside the table footprint (x/y) — that's where the moving arms live.
        # High static background beyond the table (wall/rig/floor) stays
        # supervised: the bank explains it. Footprint = table_bbox x/y when a
        # table box is configured, else unbounded.
        fx0, fx1, fy0, fy1 = (box[0], box[1], box[2], box[3]) if any(box) \
            else (-1e9, 1e9, -1e9, 1e9)
        hbox = [fx0, fx1, fy0, fy1, float(height_max or 0.0), 1e9]
        for c in cams:
            Hi, Wi = r[f"observations/images/camera_{c}_color"].shape[1:3]
            x0, y0, x1, y1 = crops.get(f"camera_{c}", [0, 0, Wi, Hi])
            x0, y0 = max(0, min(int(x0), Wi - 1)), max(0, min(int(y0), Hi - 1))
            x1, y1 = max(x0 + 1, min(int(x1), Wi)), max(y0 + 1, min(int(y1), Hi))   # crops seeded at another resolution
            Hc, Wc = y1 - y0, x1 - x0
            gc = o.require_group(f"render/camera_{c}")
            gc.attrs["crop"] = [x0, y0, x1, y1]; gc.attrs["H"] = Hc; gc.attrs["W"] = Wc
            for name in ("jpg", "K", "viewmat", "obj_mask", "valid_mask"):
                if name in gc:
                    del gc[name]                      # clean re-bake
            jd = gc.create_dataset("jpg", (T,), dtype=vt)
            Ko = gc.create_dataset("K", (T, 3, 3), dtype=np.float32)
            Vo = gc.create_dataset("viewmat", (T, 4, 4), dtype=np.float32)
            hm = c in height_cams and c not in blackout
            vo = (gc.create_dataset("valid_mask", (T, (Hc * Wc + 7) // 8), dtype=np.uint8)
                  if (c in blackout or hm) else None)  # supervisable px (no-source holes = 0)
            sam_h, sam_idx = _sam3_map(sam3_dir, ep_num, c) if sam3_dir else (None, None)
            packed_len = (Hc * Wc + 7) // 8
            mo = (gc.create_dataset("obj_mask", (T, packed_len), dtype=np.uint8)
                  if sam_h is not None else None)

            Krgb_all = r[f"observations/images/camera_{c}_rgb_intrinsics"][:].astype(np.float64)
            Trel_all = r[f"observations/images/camera_{c}_rgb_to_ir_extrinsic"][:].astype(np.float64)
            Kir_all = d[f"meta/camera_{c}/K_ir"][:].astype(np.float64)
            base_all = d[f"meta/camera_{c}/baseline"][:].astype(np.float64)
            for fr in range(T):
                rgb = r[f"observations/images/camera_{c}_color"][fr]
                if c in ccal:
                    cg, co = ccal[c]
                    rgb = np.clip(rgb.astype(np.float32) * cg + co,
                                  0, 255).astype(np.uint8)
                H, W = rgb.shape[:2]
                if c in blackout:
                    keep, valid = frame_keep_and_valid(
                        d[f"disp/camera_{c}"][fr], Kir_all[fr], Krgb_all[fr],
                        Trel_all[fr], E_all[c][fr], float(base_all[fr]), box, H, W)
                    rgb = rgb.copy(); rgb[~keep] = 0
                    if vo is not None:
                        vo[fr] = np.packbits(valid[y0:y1, x0:x1].ravel())
                elif hm:
                    # HEIGHT MASK: pixels whose FS-depth point is ABOVE height_max
                    # AND inside the table footprint (= out-of-bbox arm parts) are
                    # per-frame UNSUPERVISED — RGB is kept real (not blacked), the
                    # losses just skip them. Below-height background and high
                    # static scenery beyond the table stay supervised (the bank's
                    # job). No-source holes are invalid too (appearance unknown).
                    in_high, valid = frame_keep_and_valid(
                        d[f"disp/camera_{c}"][fr], Kir_all[fr], Krgb_all[fr],
                        Trel_all[fr], E_all[c][fr], float(base_all[fr]), hbox, H, W)
                    vo[fr] = np.packbits((valid & ~in_high)[y0:y1, x0:x1].ravel())
                rgb = rgb[y0:y1, x0:x1]
                K = Krgb_all[fr].astype(np.float32).copy(); K[0, 2] -= x0; K[1, 2] -= y0
                try:
                    vm = np.linalg.inv(E_all[c][fr] @ Trel_all[fr]).astype(np.float32)
                except np.linalg.LinAlgError:
                    vm = np.full((4, 4), np.nan, np.float32)
                ok, buf = cv2.imencode(".jpg", np.ascontiguousarray(rgb[..., ::-1]),
                                       [cv2.IMWRITE_JPEG_QUALITY, quality])
                jd[fr] = np.frombuffer(buf.tobytes(), np.uint8)
                Ko[fr] = K; Vo[fr] = vm
                if mo is not None:
                    row = sam_idx.get(fr)
                    m = ((sam_h["mask_union"][row] > 0)[y0:y1, x0:x1]
                         if row is not None else np.zeros((Hc, Wc), bool))
                    mo[fr] = np.packbits(m.ravel())
            if sam_h is not None:
                sam_h.close()
    r.close(); d.close()


def _worker(a):
    """Interruption-safe: bake into a COPY, then atomically replace the original.
    A kill mid-write leaves a stale .bake.tmp; the original is never touched."""
    f, raw_dir, disp_dir, sam3_dir, crops, box, blackout, quality, hcams, hmax = a
    cv2.setNumThreads(1)
    t0 = time.time()
    tmp = f + ".bake.tmp"
    import shutil
    shutil.copy2(f, tmp)
    try:
        bake_episode(tmp, raw_dir, disp_dir, sam3_dir, crops, box, blackout, quality,
                     height_cams=hcams, height_max=hmax)
        os.replace(tmp, f)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return os.path.basename(f), time.time() - t0


def _has_render(packed_path):
    try:
        with h5video.File(packed_path, "r") as p:
            c0 = int(p.attrs["cam_ids"][0])
            return f"render/camera_{c0}/jpg" in p
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None,
                    help="resolve --root/--raw-dir/--disp-dir/--sam3-dir via processing.paths")
    ap.add_argument("--root", default=None, help="dataset root (has packed/ + JSONs)")
    ap.add_argument("--raw-dir", default=None, help="default <root>/raw")
    ap.add_argument("--disp-dir", default=None, help="default <root>/fs_disp")
    ap.add_argument("--sam3-dir", default=None, help="default <root>/sam3_masks (None if absent)")
    ap.add_argument("--blackout-cams", type=int, nargs="*", default=[2])
    ap.add_argument("--height-mask-cams", type=int, nargs="*", default=[],
                    help="cams whose ABOVE-height pixels (FS-depth world z > "
                         "--height-max) are marked INVALID per frame (RGB kept, "
                         "losses skip them). For views that see arm parts above "
                         "a low particle bbox.")
    ap.add_argument("--height-max", type=float, default=None,
                    help="z cut for --height-mask-cams; default = config.json bbox z_max")
    ap.add_argument("--jpeg-quality", type=int, default=92)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--skip-existing", action="store_true")
    a = ap.parse_args()
    if a.root is None:
        if a.dataset is None:
            ap.error("--dataset or --root required")
        a.root = str(paths.root(a.dataset))
    a.raw_dir = a.raw_dir or os.path.join(a.root, "raw")
    a.disp_dir = a.disp_dir or os.path.join(a.root, "fs_disp")
    if a.sam3_dir is None:
        cand = os.path.join(a.root, "sam3_masks")
        a.sam3_dir = cand if os.path.isdir(cand) else None

    crops = json.load(open(os.path.join(a.root, "cam_crops.json")))
    if a.blackout_cams:
        bx = json.load(open(os.path.join(a.root, "table_bbox.json")))
        box = [bx["x"][0], bx["x"][1], bx["y"][0], bx["y"][1], bx["z"][0], bx["z"][1]]
    else:
        box = [0, 0, 0, 0, 0, 0]
    if a.height_mask_cams and a.height_max is None:
        a.height_max = float(json.load(open(os.path.join(a.root, "config.json")))["bbox"][5])
    print(f"crops={crops}\nbox={box} blackout={a.blackout_cams} "
          f"height_mask={a.height_mask_cams}@z<{a.height_max} sam3={a.sam3_dir} "
          f"workers={a.workers}", flush=True)

    files = sorted(glob.glob(os.path.join(a.root, "packed", "episode_*.hdf5")),
                   key=lambda p: int(re.search(r"(\d+)", os.path.basename(p)).group(1)))
    jobs = []
    for f in files:
        if a.skip_existing and _has_render(f):
            continue
        jobs.append((f, a.raw_dir, a.disp_dir, a.sam3_dir, crops, box, a.blackout_cams,
                     a.jpeg_quality, a.height_mask_cams, a.height_max))
    print(f"{len(jobs)}/{len(files)} packed episodes to bake ({a.workers} workers)", flush=True)
    if a.workers > 1 and len(jobs) > 1:
        with mp.Pool(a.workers) as pool:
            for i, (name, dt) in enumerate(pool.imap_unordered(_worker, jobs)):
                print(f"  [{i + 1}/{len(jobs)}] {name} ({dt:.1f}s)", flush=True)
    else:
        for i, job in enumerate(jobs):
            name, dt = _worker(job)
            print(f"  [{i + 1}/{len(jobs)}] {name} ({dt:.1f}s)", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
