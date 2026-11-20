"""Pack a dataset into the train-ready per-episode format (packed_v2).

For every episode this writes ONE self-contained HDF5
    <PW_DATA_ROOT>/<dataset>/packed/episode_<E>.hdf5
holding the per-pixel ingredients to rebuild the point cloud ON THE FLY at
train time, with all geometric filtering baked into one per-pixel
`valid_mask` so the loader does zero filtering:

  point cloud(frame) = { lift(depth[p], K_ir, E)  for every pixel p where valid_mask[p] }
  color(p)           = rgb_in_ir[p]          (RGB pre-projected into the IR frame)
  object label(p)    = obj_mask[p]           (SAM3, pre-projected into the IR frame)

`valid_mask` = LRC consistency & depth in [min_z,max_z] & RGB-reprojection
in-frame & world point in bbox.  `depth` itself is FULL-SCENE (LRC & z in
[min_z, max_z_full], walls included, 0 = hole) so the packed file alone feeds
the rendered-depth loss / depth carve, no fs_disp needed at train time.

Inputs read (per episode):
    fs_disp/disp_episode_<E>.hdf5   disp, disp_right, meta (K_ir,K_rgb,T_rgb_to_ir,E_cam2world,baseline)
    raw/episode_<E>.hdf5            camera_<c>_color, joints, base poses, tactile pads, actions
    sam3_masks/episode_<E>_camera_<c>.h5   mask_per_obj + obj_ids + frame_indices   [optional, --no-obj]
    config.json                     cameras.calibration, bbox, robot_base, tactile_stream_swap,
                                    moving_cams, camera_depth_limits, color_calib, episode_overrides
    cam_perframe/ep_<E>_cam<C>.npz  E_cam2world (T,4,4) per moving camera (wrist cams)

Usage (what `./pw pack` runs):
    python -m processing.pack.pack_dataset --dataset bimanual_cloth_v3 --episodes all \
        --lrc-thresh 2 --min-depth 0.10 --max-depth 2.0 --workers 16
"""
import argparse, glob, json, os, sys, time
import multiprocessing as mp
from pathlib import Path
import cv2, h5py, numpy as np

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parents[2]))
import h5video  # noqa: E402
from processing import paths                                   # noqa: E402
from processing.config import DatasetConfig                    # noqa: E402
from processing.pc.geometry import (                           # noqa: E402
    lrc_mask, _bilinear_sample_rgb, apply_tweak,
    _tweaks_from_config, _base_offset_from_config)
DEFAULT_FINGER_FOR_KEY = {"left_robot_left_finger": "vx300s/left_finger_link", "left_robot_right_finger": "vx300s/right_finger_link",
                          "right_robot_left_finger": "vx300s/left_finger_link", "right_robot_right_finger": "vx300s/right_finger_link"}

_VLEN = h5py.vlen_dtype(np.uint8)


def jpeg_encode(rgb):
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                           [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return np.frombuffer(buf.tobytes(), dtype=np.uint8)


def load_sam3(dataset, ep, cam):
    """Return (mask_per_obj (N,K,Hr,Wr) bool, obj_ids (K,), {src_frame:row}) or (None,None,None).

    Per-object masks (not the union) so we can write a LABEL map:
    label = obj_id+1 (0 = none)."""
    p = paths.sam3_masks_dir(dataset) / f"episode_{ep}_camera_{cam}.h5"
    if not p.exists():
        return None, None, None
    with h5video.File(p, "r") as f:
        mpo = f["mask_per_obj"][:].astype(bool)          # (N,K,Hr,Wr)
        oid = f["obj_ids"][:].astype(int)                # (K,)
        fi = f["frame_indices"][:].astype(int)
    return mpo, oid, {int(s): r for r, s in enumerate(fi)}


def label_image(mpo_row, obj_ids):
    """(K,Hr,Wr) bool + obj_ids → (Hr,Wr) uint8 label map (obj_id+1; 0=none).
    Higher obj_id wins on overlap (arbitrary but deterministic)."""
    Hr, Wr = mpo_row.shape[1:]
    lab = np.zeros((Hr, Wr), np.uint8)
    for k in np.argsort(obj_ids):            # ascending → higher id overwrites
        lab[mpo_row[k]] = int(obj_ids[k]) + 1
    return lab


def pack_cam_frame(disp, disp_right, color, K_ir, K_rgb, T_r2i, E_c2w,
                   baseline, bbox, lrc_thresh, min_z, max_z, sam3_label,
                   max_z_full=3.5):
    """One camera, one frame → (depth f16, valid bool, rgb_in_ir u8, obj_label u8).

    `sam3_label` is an (Hr,Wr) uint8 label map in the RGB frame (0=none,
    obj_id+1 otherwise), or None. Output `obj_label` is that label sampled
    into the IR frame at valid pixels (0 elsewhere).

    The returned depth is FULL-SCENE (LRC ∧ z∈[min_z,max_z_full], incl. the
    wall/background beyond the bbox and max_z) so the packed file alone feeds
    the rendered-depth loss + depth carve + FS bank init — no fs_disp needed
    at train time. The bbox/in-frame cut lives only in `valid` (the cloud)."""
    H, W = disp.shape
    fx, fy = float(K_ir[0, 0]), float(K_ir[1, 1])
    cx, cy = float(K_ir[0, 2]), float(K_ir[1, 2])
    depth = (fx * baseline) / np.maximum(disp, 1e-3)
    lrc_ok = lrc_mask(disp, disp_right, lrc_thresh) & np.isfinite(depth) & (disp > 0.5)
    valid_full = lrc_ok & (depth > min_z) & (depth < max_z_full)
    valid = lrc_ok & (depth > min_z) & (depth < max_z)

    uu = np.arange(W, dtype=np.float32)[None, :].repeat(H, 0)
    vv = np.arange(H, dtype=np.float32)[:, None].repeat(W, 1)
    x_ir = (uu - cx) * depth / fx
    y_ir = (vv - cy) * depth / fy
    pts_ir = np.stack([x_ir, y_ir, depth], axis=-1)            # (H,W,3)

    # reproject IR→RGB to (a) sample color and (b) drop out-of-frame pixels
    R_r2i = T_r2i[:3, :3].astype(np.float32); t_r2i = T_r2i[:3, 3].astype(np.float32)
    R_i2r = R_r2i.T; t_i2r = -R_i2r @ t_r2i
    pts_rgb = pts_ir @ R_i2r.T + t_i2r
    Z = pts_rgb[..., 2]; in_front = Z > 1e-3
    fxr, fyr = float(K_rgb[0, 0]), float(K_rgb[1, 1])
    cxr, cyr = float(K_rgb[0, 2]), float(K_rgb[1, 2])
    Hr, Wr = color.shape[:2]
    u_rgb = fxr * pts_rgb[..., 0] / np.where(in_front, Z, 1.0) + cxr
    v_rgb = fyr * pts_rgb[..., 1] / np.where(in_front, Z, 1.0) + cyr
    valid &= in_front & (u_rgb >= 0) & (u_rgb < Wr - 1) & (v_rgb >= 0) & (v_rgb < Hr - 1)

    # world transform + bbox crop (in world frame)
    R = E_c2w[:3, :3].astype(np.float32); t = E_c2w[:3, 3].astype(np.float32)
    pw = pts_ir @ R.T + t
    valid &= ((pw[..., 0] >= bbox[0]) & (pw[..., 0] <= bbox[1]) &
              (pw[..., 1] >= bbox[2]) & (pw[..., 1] <= bbox[3]) &
              (pw[..., 2] >= bbox[4]) & (pw[..., 2] <= bbox[5]))

    rgb_in_ir = np.zeros((H, W, 3), np.uint8)
    obj = np.zeros((H, W), np.uint8)
    vy, vx = np.nonzero(valid)
    if len(vy):
        rgb_in_ir[vy, vx] = _bilinear_sample_rgb(color, u_rgb[vy, vx], v_rgb[vy, vx])
        if sam3_label is not None:
            ui = np.clip(np.round(u_rgb[vy, vx]).astype(np.int32), 0, Wr - 1)
            vi = np.clip(np.round(v_rgb[vy, vx]).astype(np.int32), 0, Hr - 1)
            obj[vy, vx] = sam3_label[vi, ui]

    depth_store = np.where(valid_full, depth, 0.0).astype(np.float16)  # zeros compress well
    return depth_store, valid, rgb_in_ir, obj


def pack_episode(dataset, ep, cfg, tweaks, bbox, base_off_per_arm,
                 lrc_thresh, min_z, max_z, out_path, no_obj=False,
                 max_z_full=3.5):
    disp_path = paths.disp_dir(dataset) / f"disp_episode_{ep}.hdf5"
    raw_path = paths.raw_dir(dataset) / f"episode_{ep}.hdf5"
    df = h5video.File(disp_path, "r"); sf = h5video.File(raw_path, "r")
    try:
        # fixed (config) cameras; moving cameras are added below.
        cfg_cams = {int(k.split("_")[1]) for k in cfg.cameras.keys()}
        cam_ids = sorted({int(k.split("_")[1]) for k in df["disp"].keys()
                          if k.startswith("camera_")} & cfg_cams)
        T = int(df.attrs["num_frames"])
        # MOVING cameras (config "moving_cams": ["camera_3", ...]): wrist cams.
        # Per-frame E_cam2world comes from the pose sidecar
        # cam_perframe/ep_<E>_cam<C>.npz (produced off-pipeline from the arm
        # pose fits, see processing/README.md), falling back to the per-frame
        # estimate recorded in fs_disp meta (the robot's own FK) unless
        # config "require_moving_camera_sidecars" is set.
        mcamE = {}
        camera_depth_limits = (getattr(cfg, "extra", None) or {}).get("camera_depth_limits", {})
        for name in ((getattr(cfg, "extra", None) or {}).get("moving_cams") or []):
            mc = int(str(name).split("_")[-1])
            if f"camera_{mc}" not in df["disp"]:
                continue
            sc = paths.root(dataset) / "cam_perframe" / f"ep_{ep}_cam{mc}.npz"
            if sc.exists():
                mcamE[mc] = np.load(sc)["E_cam2world"].astype(np.float64)
            else:
                if (getattr(cfg, "extra", None) or {}).get("require_moving_camera_sidecars", False):
                    raise FileNotFoundError(f"Required moving-camera pose sidecar: {sc}")
                mcamE[mc] = df[f"meta/camera_{mc}/E_cam2world"][:].astype(np.float64)
            cam_ids = sorted(set(cam_ids) | {mc})
        # per-cam color calibration (config color_calib): applied to EVERY
        # camera so the whole pack lives in ONE neutral color space
        ccal = {}
        for k, v in ((getattr(cfg, "extra", None) or {})
                     .get("color_calib") or {}).items():
            if k.startswith("camera_"):
                ccal[int(k.split("_")[1])] = (
                    np.asarray(v.get("gain", [1, 1, 1]), np.float32),
                    np.asarray(v.get("offset", [0, 0, 0]), np.float32))
        # per-episode-range overrides (config "episode_overrides"): a camera
        # that got bumped for episodes [lo,hi] gets that range's extrinsic
        # tweak / color gains swapped in for just those episodes.
        from processing.config import (episode_override_for,
                                       cam_calibration_to_legacy_tweak)
        ov_cams, ov_ccal = episode_override_for(cfg, ep)
        if ov_cams:
            tweaks = dict(tweaks)          # don't mutate the shared base dict
            for k, cal in ov_cams.items():
                tweaks[int(k.split("_")[1])] = cam_calibration_to_legacy_tweak(cal)
            print(f"    ep{ep}: extrinsics override for {sorted(ov_cams)}")
        for k, v in ov_ccal.items():
            if k.startswith("camera_"):
                ccal[int(k.split("_")[1])] = (
                    np.asarray(v.get("gain", [1, 1, 1]), np.float32),
                    np.asarray(v.get("offset", [0, 0, 0]), np.float32))
        H = df[f"disp/camera_{cam_ids[0]}"].shape[1]
        W = df[f"disp/camera_{cam_ids[0]}"].shape[2]
        if "disp_right" not in df:
            raise RuntimeError(f"ep{ep}: no disp_right (run_fs --pair both) — LRC needs it")

        # SAM3 masks per cam (optional, per-object so the label map keeps ids).
        # --no-obj packs geometry only (masks not ready yet).
        sam3 = ({c: (None, None, None) for c in cam_ids} if no_obj
                else {c: load_sam3(dataset, ep, c) for c in cam_ids})
        has_obj = any(mpo is not None for mpo, _, _ in sam3.values())
        obj_ids_all = sorted({int(o) for mpo, oid, _ in sam3.values()
                              if oid is not None for o in oid})

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        tmp = out_path + ".tmp"
        g = h5py.File(tmp, "w")

        depth_ds, rgb_ds, valid_ds, obj_ds = {}, {}, {}, {}
        Kir_ds, E_ds = {}, {}
        for c in cam_ids:
            # per-camera image size: the wrist cams of the 2026-10 collections run at 640x360 while the fixed
            # cams are 1280x720 (attrs H/W = first camera, kept for old readers; readers use the dataset shapes)
            Hc, Wc = (int(v) for v in df[f"disp/camera_{c}"].shape[1:3]); HWc = Hc * Wc
            depth_ds[c] = g.create_dataset(f"img/camera_{c}/depth", (T, Hc, Wc),
                                           dtype="float16", chunks=(1, Hc, Wc),
                                           compression="gzip", compression_opts=4)
            rgb_ds[c] = g.create_dataset(f"img/camera_{c}/rgb_in_ir", (T,), dtype=_VLEN)
            valid_ds[c] = g.create_dataset(f"img/camera_{c}/valid_mask",
                                           (T, (HWc + 7) // 8), dtype="uint8",
                                           chunks=(1, (HWc + 7) // 8),
                                           compression="gzip", compression_opts=4)
            if has_obj:
                # LABEL map (0=none, obj_id+1) — uint8 (H,W), gzip (mostly zeros)
                obj_ds[c] = g.create_dataset(f"img/camera_{c}/obj_mask",
                                             (T, Hc, Wc), dtype="uint8",
                                             chunks=(1, Hc, Wc),
                                             compression="gzip", compression_opts=4)
            Kir_ds[c] = g.create_dataset(f"cam/camera_{c}/K_ir", (T, 3, 3), dtype="float32")
            E_ds[c] = g.create_dataset(f"cam/camera_{c}/E_cam2world", (T, 4, 4), dtype="float32")

        n_valid_total = 0
        for t in range(T):
            for c in cam_ids:
                disp = df[f"disp/camera_{c}"][t].astype(np.float32)
                disp_r = df[f"disp_right/camera_{c}"][t].astype(np.float32)
                K_ir = df[f"meta/camera_{c}/K_ir"][t]
                K_rgb = df[f"meta/camera_{c}/K_rgb"][t]
                T_r2i = df[f"meta/camera_{c}/T_rgb_to_ir"][t]
                E_raw = df[f"meta/camera_{c}/E_cam2world"][t]
                B = float(df[f"meta/camera_{c}/baseline"][t])
                color = sf[f"/observations/images/camera_{c}_color"][t]
                if c in mcamE:
                    Em = mcamE[c]
                    E_eff = Em[min(t, len(Em) - 1)]
                else:
                    E_eff = apply_tweak(E_raw, tweaks.get(c, {}))
                if c in ccal:
                    cg, co = ccal[c]
                    color = np.clip(color.astype(np.float32) * cg + co,
                                    0, 255).astype(np.uint8)

                sam_label = None
                mpo, oid, idxmap = sam3[c]
                if mpo is not None and t in idxmap:
                    sam_label = label_image(mpo[idxmap[t]], oid)

                limits = camera_depth_limits.get(f"camera_{c}", {})
                depth, valid, rgb_ir, obj = pack_cam_frame(
                    disp, disp_r, color, K_ir, K_rgb, T_r2i, E_eff, B,
                    bbox, lrc_thresh, float(limits.get("min_depth", min_z)),
                    float(limits.get("max_depth", max_z)), sam_label,
                    max_z_full=float(limits.get("max_depth_full", max_z_full)))

                depth_ds[c][t] = depth
                rgb_ds[c][t] = jpeg_encode(rgb_ir)
                valid_ds[c][t] = np.packbits(valid.reshape(-1))
                if has_obj:
                    obj_ds[c][t] = obj
                Kir_ds[c][t] = K_ir.astype(np.float32)
                E_ds[c][t] = E_eff.astype(np.float32)
                n_valid_total += int(valid.sum())
            if (t + 1) % 50 == 0 or t == T - 1:
                print(f"    ep{ep} frame {t+1}/{T}", flush=True)

        # ---- robot state + tactile pads (loader does the FK) ----
        def copy(src_key, dst_key):
            if src_key in sf:
                g.create_dataset(dst_key, data=sf[src_key][:])
        copy("observations/full_joint_pos", "robot/full_joint_pos")
        copy("observations/joint_pos", "robot/joint_pos")
        copy("observations/robot_base_pose_in_world", "robot/base_pose_in_world")
        copy("joint_action", "action/joint_action")
        copy("cartesian_action", "action/cartesian_action")
        copy("stage", "meta/stage")
        copy("timestamp", "meta/timestamp")
        tac_keys = sorted([k for k in sf["observations"].keys()
                           if "finger" in k and "timestamp" not in k
                           and k in DEFAULT_FINGER_FOR_KEY])
        for k in tac_keys:
            g.create_dataset(f"tactile/{k}", data=sf[f"observations/{k}"][:].astype(np.float32))

        if "observations/robot_base_pose_in_world" in sf:
            n_arms = (sf["observations/robot_base_pose_in_world"].shape[1]
                      if sf["observations/robot_base_pose_in_world"].ndim == 4 else 1)
        else:
            n_arms = 0          # non-ALOHA robot: no base pose stream
        g.attrs.update(dict(
            dataset=dataset, episode=ep, num_frames=T, H=H, W=W,
            cam_ids=np.array(cam_ids, np.int32), n_arms=int(n_arms),
            bbox=np.array(bbox, np.float32),
            robot_base=json.dumps(base_off_per_arm),
            occlusion_mask="lrc", lrc_thresh=float(lrc_thresh),
            min_z=float(min_z), max_z=float(max_z),
            max_z_full=float(max_z_full),
            camera_depth_limits=json.dumps(camera_depth_limits),
            depth_semantics="full_scene_lrc",   # depth = LRC'd FULL scene (0=hole);
            #                                     bbox/in-frame cut is valid_mask only
            tactile_stream_swap=bool(cfg.tactile_stream_swap),
            tactile_keys=json.dumps(tac_keys),
            has_obj_mask=bool(has_obj),
            obj_mask_kind="label",   # uint8 (H,W): 0=none, obj_id+1
            obj_labels=json.dumps({str(o + 1): f"obj{o}" for o in obj_ids_all}),
            src_episode=os.path.basename(raw_path),
            format_version="packed_v2",
            mask_semantics="valid=LRC&depth&inframe&bbox; project valid pixels, no load-time filtering",
        ))
        g.close()
        os.replace(tmp, out_path)
        return n_valid_total / max(T, 1)
    finally:
        df.close(); sf.close()


def _pack_one(t):
    """Pool worker: unpack args → pack_episode, never raise (return status tuple)."""
    (dataset, ep, cfg, tweaks, bbox, base_off, lrc, minz, maxz, out_path, no_obj,
     maxz_full) = t
    t0 = time.time()
    try:
        avg = pack_episode(dataset, ep, cfg, tweaks, bbox, base_off, lrc, minz, maxz,
                           out_path, no_obj=no_obj, max_z_full=maxz_full)
        return (ep, True, float(avg), time.time() - t0, "")
    except Exception as e:
        return (ep, False, 0.0, time.time() - t0, f"{type(e).__name__}: {str(e)[:160]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--episodes", default="all")
    ap.add_argument("--lrc-thresh", type=float, default=5.0)
    ap.add_argument("--min-depth", type=float, default=0.10)
    ap.add_argument("--max-depth", type=float, default=2.0)
    ap.add_argument("--max-depth-full", type=float, default=3.5,
                    help="z cut for the FULL-SCENE depth dataset (walls incl.); "
                         "--max-depth only bounds the bbox'd cloud valid_mask")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--workers", type=int, default=8,
                    help="parallel episode workers (episodes are independent); 1 = sequential")
    ap.add_argument("--no-obj", action="store_true",
                    help="skip SAM3 obj_mask entirely (pack geometry only)")
    args = ap.parse_args()

    cfg = DatasetConfig.load(paths.config_path(args.dataset))
    tweaks = _tweaks_from_config(cfg)
    bbox = list(cfg.bbox)
    base_off = _base_offset_from_config(cfg)
    out_dir = paths.root(args.dataset) / "packed"
    out_dir.mkdir(parents=True, exist_ok=True)

    disp_files = sorted(glob.glob(str(paths.disp_dir(args.dataset) / "disp_episode_*.hdf5")),
                        key=lambda p: int(os.path.basename(p).split("_")[2].split(".")[0]))
    eps = [int(os.path.basename(p).split("_")[2].split(".")[0]) for p in disp_files]
    if args.episodes != "all":
        want = set()
        for part in args.episodes.split(","):
            part = part.strip()
            if "-" in part:
                a, b = part.split("-", 1); want.update(range(int(a), int(b) + 1))
            elif part:
                want.add(int(part))
        eps = [e for e in eps if e in want]

    # build task list (respecting skip-existing)
    tasks = []
    for ep in eps:
        out_path = str(out_dir / f"episode_{ep}.hdf5")
        if os.path.exists(out_path) and not args.overwrite:
            print(f"ep{ep}: skip (exists)", flush=True); continue
        tasks.append((args.dataset, ep, cfg, tweaks, bbox, base_off,
                      args.lrc_thresh, args.min_depth, args.max_depth, out_path,
                      args.no_obj, args.max_depth_full))

    workers = max(1, min(args.workers, len(tasks)))
    print(f"pack {args.dataset}: {len(tasks)} eps to pack  bbox={bbox}  "
          f"lrc_thresh={args.lrc_thresh}  workers={workers}", flush=True)
    print(f"  out: {out_dir}", flush=True)

    done = 0
    if workers <= 1:
        for t in tasks:
            ep, ok, avg, dt, err = _pack_one(t); done += 1
            msg = f"OK avg {avg:.0f} valid px/frame" if ok else f"FAILED {err}"
            print(f"[{done}/{len(tasks)}] ep{ep}: {msg} ({dt:.1f}s)", flush=True)
    else:
        # episodes are independent; each worker opens its own files. spawn keeps
        # HDF5 state clean across processes.
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers) as pool:
            for ep, ok, avg, dt, err in pool.imap_unordered(_pack_one, tasks):
                done += 1
                msg = f"OK avg {avg:.0f} valid px/frame" if ok else f"FAILED {err}"
                print(f"[{done}/{len(tasks)}] ep{ep}: {msg} ({dt:.1f}s)", flush=True)
    print("[pack] DONE", flush=True)


if __name__ == "__main__":
    main()
