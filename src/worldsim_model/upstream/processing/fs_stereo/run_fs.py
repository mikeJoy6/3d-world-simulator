"""Run FoundationStereo on every frame of every episode of a dataset, saving
the raw disparity (no masking at write time; the packer applies LRC).

Reads:  <PW_DATA_ROOT>/<dataset>/raw/episode_*.hdf5      (or --src-dir)
Writes: <PW_DATA_ROOT>/<dataset>/fs_disp/disp_episode_*.hdf5  (or --out-dir)

Output (one HDF5 per source episode):
  /disp/camera_{c}              (T, H, W) float16  gzip; left-IR-referenced disparity
  /disp_right/camera_{c}        (T, H, W) float16  right-IR-referenced (--pair both|right)
  /meta/camera_{c}/K_ir         (T, 3, 3) float32
  /meta/camera_{c}/K_rgb        (T, 3, 3) float32
  /meta/camera_{c}/E_cam2world  (T, 4, 4) float32  inv(raw camera_c_extrinsics)
  /meta/camera_{c}/T_rgb_to_ir  (T, 4, 4) float32
  /meta/camera_{c}/baseline     (T,)      float32
  attrs: src_episode, num_frames, valid_iters, output_kind='raw_disp',
         pair, has_right_disp

depth = K_ir[0,0] * baseline / disp.  The pipeline needs BOTH sides
(`--pair both`): pack_dataset's LRC mask reads disp_right.  Per-camera IR
size is honoured (wrist cams can be 360x640 while fixed cams are 720x1280).
Colour streams are not needed, so IR-only exports (export_ir_only.py) work.
Throughput (A6000-class GPU, valid_iters 32): ~0.92 s per 720x1280 frame-side,
~0.39 s at 480x848, ~0.25 s per 360x640 wrist frame-side.

Usage:
    python -m processing.fs_stereo.run_fs --dataset bimanual_cloth_v3 --pair both --skip-existing
    python -m processing.fs_stereo.run_fs --dataset bimanual_box_v2 --pair both --episodes 0,1,2 \
        --src-dir /path/to/ir_only --out-dir /path/to/out/fs_disp
"""
import argparse, glob, os, sys, time
from pathlib import Path
import numpy as np
import h5py
import torch

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parents[2]))
from processing import paths as data_paths  # noqa: E402  (avoid local var clash)

FS_DIR = str(data_paths.FS_REPO)          # env FS_REPO
sys.path.insert(0, FS_DIR)
sys.path.insert(0, os.path.join(FS_DIR, "core"))
import h5video  # noqa: E402

from omegaconf import OmegaConf
from core.foundation_stereo import FoundationStereo
from core.utils.utils import InputPadder

CKPT_DIR = f"{FS_DIR}/pretrained_models/23-51-11"


def invert_rigid(E):
    out = np.eye(4, dtype=np.float32)
    R = E[:3, :3].astype(np.float32); t = E[:3, 3].astype(np.float32)
    out[:3, :3] = R.T; out[:3, 3] = -R.T @ t
    return out


def load_model(valid_iters):
    cfg = OmegaConf.load(f"{CKPT_DIR}/cfg.yaml")
    if "vit_size" not in cfg:
        cfg["vit_size"] = "vitl"
    cfg.valid_iters = valid_iters
    torch.autograd.set_grad_enabled(False)
    model = FoundationStereo(OmegaConf.create(cfg))
    ckpt = torch.load(f"{CKPT_DIR}/model_best_bp2.pth", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.cuda().eval()
    return model, int(ckpt["global_step"])


def fs_disp(model, L, R, iters):
    L3 = np.stack([L, L, L], axis=-1)
    R3 = np.stack([R, R, R], axis=-1)
    t0 = torch.from_numpy(L3).cuda().float().permute(2, 0, 1).unsqueeze(0)
    t1 = torch.from_numpy(R3).cuda().float().permute(2, 0, 1).unsqueeze(0)
    padder = InputPadder(t0.shape, divis_by=32, force_square=False)
    t0p, t1p = padder.pad(t0, t1)
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True):
        disp = model.forward(t0p, t1p, iters=iters, test_mode=True)
    return padder.unpad(disp.float()).squeeze().cpu().numpy().astype(np.float32)


def fs_disp_right(model, L, R, iters):
    """Right-IR-referenced disparity, via the horizontal-flip trick.

    FoundationStereo always returns disparity for its FIRST (left/reference)
    argument, positive toward decreasing column. To get disparity defined on
    the RIGHT image instead, flip both images horizontally, run with the
    flipped right image as the reference, then flip the result back. The
    returned map is in the right-IR pixel frame, where the matching left-IR
    column is `u + disp_right` (sign mirrors the left-ref convention).

    Used only for the packer's left-right consistency (LRC) check; not for
    back-projection itself.
    """
    Lf = L[:, ::-1]
    Rf = R[:, ::-1]
    d = fs_disp(model, Rf, Lf, iters)        # reference = flipped right
    return np.ascontiguousarray(d[:, ::-1])  # flip back into right frame


REQUIRED_PER_CAM = ["left_ir", "right_ir", "intrinsics", "baseline",
                    "extrinsics", "rgb_intrinsics", "rgb_to_ir_extrinsic"]   # colour is NOT needed (IR-only exports, export_ir_only.py)


def check_episode(src_path):
    """Return list of missing keys (empty if episode is well-formed).
    Uses the cameras actually present in the file, not a fixed count."""
    with h5video.File(src_path, "r") as f:
        keys = set(f["observations/images"].keys())
    cam_ids = sorted({int(k.split("_")[1]) for k in keys
                      if k.startswith("camera_") and k.endswith("_left_ir")})
    if not cam_ids:
        return ["no cameras found"]
    missing = []
    for c in cam_ids:
        for k in REQUIRED_PER_CAM:
            if f"camera_{c}_{k}" not in keys:
                missing.append(f"camera_{c}_{k}")
    return missing


def process_episode(model, src_path, out_path, args):
    with h5video.File(src_path, "r") as src:
        cam_ids = sorted({int(k.split("_")[1])
                          for k in src["observations/images"].keys()
                          if k.startswith("camera_") and k.endswith("_left_ir")})
        T = src[f"/observations/images/camera_{cam_ids[0]}_left_ir"].shape[0]
        H = src[f"/observations/images/camera_{cam_ids[0]}_left_ir"].shape[1]
        W = src[f"/observations/images/camera_{cam_ids[0]}_left_ir"].shape[2]
        if getattr(args, "max_frames", 0) and args.max_frames > 0:
            T = min(T, int(args.max_frames))
        print(f"  cameras={cam_ids}  T={T}  HxW={H}x{W}", flush=True)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        tmp_path = out_path + ".tmp"
        pair = args.pair
        want_left = pair != "right"
        want_right = pair in ("right", "both")
        with h5py.File(tmp_path, "w") as dst:
            disp_dsets, disp_r_dsets, meta = {}, {}, {}
            for c in cam_ids:
                if want_left:
                    # per-camera IR size (wrist cams can be 360x640 while fixed cams are 720x1280)
                    H, W = src[f"/observations/images/camera_{c}_left_ir"].shape[1:3]
                    disp_dsets[c] = dst.create_dataset(
                        f"disp/camera_{c}", shape=(T, H, W), dtype="float16",
                        chunks=(1, H, W), compression="gzip", compression_opts=4)
                if want_right:
                    H, W = src[f"/observations/images/camera_{c}_left_ir"].shape[1:3]
                    disp_r_dsets[c] = dst.create_dataset(
                        f"disp_right/camera_{c}", shape=(T, H, W), dtype="float16",
                        chunks=(1, H, W), compression="gzip", compression_opts=4)
                meta[c] = {
                    "K_ir":        dst.create_dataset(f"meta/camera_{c}/K_ir",        shape=(T, 3, 3), dtype="float32"),
                    "K_rgb":       dst.create_dataset(f"meta/camera_{c}/K_rgb",       shape=(T, 3, 3), dtype="float32"),
                    "E_cam2world": dst.create_dataset(f"meta/camera_{c}/E_cam2world", shape=(T, 4, 4), dtype="float32"),
                    "T_rgb_to_ir": dst.create_dataset(f"meta/camera_{c}/T_rgb_to_ir", shape=(T, 4, 4), dtype="float32"),
                    "baseline":    dst.create_dataset(f"meta/camera_{c}/baseline",    shape=(T,),      dtype="float32"),
                }

            ep_t0 = time.time()
            disp_range = {c: [np.inf, -np.inf] for c in cam_ids}
            for t in range(T):
                for c in cam_ids:
                    L  = src[f"/observations/images/camera_{c}_left_ir"][t]
                    R_ = src[f"/observations/images/camera_{c}_right_ir"][t]
                    K_ir   = src[f"/observations/images/camera_{c}_intrinsics"][t]
                    K_rgb  = src[f"/observations/images/camera_{c}_rgb_intrinsics"][t]
                    T_r2i  = src[f"/observations/images/camera_{c}_rgb_to_ir_extrinsic"][t]
                    E_raw  = src[f"/observations/images/camera_{c}_extrinsics"][t]
                    B  = float(src[f"/observations/images/camera_{c}_baseline"][t])

                    if want_left:
                        disp = fs_disp(model, L, R_, args.valid_iters)
                        disp_dsets[c][t] = disp.astype(np.float16)
                    if want_right:
                        disp_r = fs_disp_right(model, L, R_, args.valid_iters)
                        disp_r_dsets[c][t] = disp_r.astype(np.float16)
                    if not want_left:
                        disp = disp_r                    # range stats on right
                    meta[c]["K_ir"][t]        = K_ir.astype(np.float32)
                    meta[c]["K_rgb"][t]       = K_rgb.astype(np.float32)
                    meta[c]["E_cam2world"][t] = invert_rigid(E_raw)
                    meta[c]["T_rgb_to_ir"][t] = T_r2i.astype(np.float32)
                    meta[c]["baseline"][t]    = B
                    disp_range[c][0] = min(disp_range[c][0], float(disp.min()))
                    disp_range[c][1] = max(disp_range[c][1], float(disp.max()))

                if (t + 1) % 25 == 0 or t == T - 1:
                    el = time.time() - ep_t0
                    fps = (t + 1) / el
                    eta = (T - t - 1) / max(fps, 1e-6)
                    print(f"    [{os.path.basename(src_path)}] frame {t+1}/{T} "
                          f"({fps:.2f} fps, ETA {eta:6.1f}s)", flush=True)

            dst.attrs["src_episode"]   = os.path.basename(src_path)
            dst.attrs["num_frames"]    = T
            dst.attrs["valid_iters"]   = int(args.valid_iters)
            dst.attrs["output_kind"]   = "raw_disp"
            dst.attrs["has_right_disp"] = want_right
            dst.attrs["pair"]          = pair
        os.rename(tmp_path, out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    help="logical dataset name; resolves all paths via processing.paths")
    ap.add_argument("--episodes", default="all",
                    help="'all' or comma-separated indices, e.g. '0,1,2'.")
    ap.add_argument("--valid-iters", type=int, default=32)
    ap.add_argument("--max-frames", type=int, default=0,
                    help="If >0, cap at this many frames per episode.")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--pair", choices=["left", "right", "both"], default="both",
                    help="which IR reference to compute: both (default; disp/ + "
                         "disp_right/, two passes per frame), left (disp/ only), "
                         "or right (disp_right/ only, via the flip trick). The "
                         "packer's LRC needs both.")
    ap.add_argument("--src-dir", default=None,
                    help="override the raw episode dir (default: paths.raw_dir)")
    ap.add_argument("--out-dir", default=None,
                    help="override the output dir (default: paths.disp_dir)")
    args = ap.parse_args()

    src_dir = Path(args.src_dir) if args.src_dir else data_paths.raw_dir(args.dataset)
    out_dir = Path(args.out_dir) if args.out_dir else data_paths.disp_dir(args.dataset)
    out_dir.mkdir(parents=True, exist_ok=True)

    src_files = sorted(glob.glob(str(src_dir / "episode_*.hdf5")),
                       key=lambda p: int(os.path.basename(p).split("_")[1].split(".")[0]))
    if args.episodes != "all":
        wanted = {int(x) for x in args.episodes.split(",")}
        src_files = [p for p in src_files
                     if int(os.path.basename(p).split("_")[1].split(".")[0]) in wanted]
    print(f"dataset:    {args.dataset}")
    print(f"  src_dir:  {src_dir}")
    print(f"  out_dir:  {out_dir}")
    print(f"Will process {len(src_files)} episode(s)", flush=True)

    model, step = load_model(args.valid_iters)
    print(f"FS loaded (global_step={step})", flush=True)

    skipped = []
    for i, src_path in enumerate(src_files):
        idx = int(os.path.basename(src_path).split("_")[1].split(".")[0])
        out_path = str(out_dir / f"disp_episode_{idx}.hdf5")
        if args.skip_existing and os.path.exists(out_path):
            print(f"[{i+1}/{len(src_files)}] skip (exists): {out_path}", flush=True)
            continue
        try:
            missing = check_episode(src_path)
        except Exception as e:
            print(f"[{i+1}/{len(src_files)}] SKIP unreadable: {src_path}  ({type(e).__name__}: {str(e)[:150]})", flush=True)
            skipped.append((src_path, f"unreadable: {type(e).__name__}"))
            continue
        if missing:
            print(f"[{i+1}/{len(src_files)}] SKIP malformed: {src_path}  missing {len(missing)} keys "
                  f"(first: {missing[:3]})", flush=True)
            skipped.append((src_path, f"missing keys: {missing[:3]}"))
            continue
        print(f"[{i+1}/{len(src_files)}] {src_path} -> {out_path}", flush=True)
        t0 = time.time()
        try:
            process_episode(model, src_path, out_path, args)
            print(f"    done in {time.time()-t0:.1f}s", flush=True)
        except Exception as e:
            # Mid-processing failure (e.g., truncated chunk only triggers on read).
            # Clean up the .tmp so a future --skip-existing rerun can try again.
            tmp = out_path + ".tmp"
            if os.path.exists(tmp):
                try: os.remove(tmp)
                except Exception: pass
            print(f"    FAILED after {time.time()-t0:.1f}s: {type(e).__name__}: {str(e)[:200]}", flush=True)
            skipped.append((src_path, f"runtime: {type(e).__name__}"))
            continue

    if skipped:
        print(f"\n=== Skipped {len(skipped)} episode(s) ===", flush=True)
        for p, why in skipped:
            print(f"  {p}  →  {why}", flush=True)


if __name__ == "__main__":
    main()
