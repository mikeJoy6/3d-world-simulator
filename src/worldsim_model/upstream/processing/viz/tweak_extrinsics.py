"""Viser extrinsic + bbox tweaker for a dataset's fixed cameras (`./pw camera`).

Reads FoundationStereo disparity + per-frame K/E/T_rgb_to_ir/baseline from
the fs_disp HDF5, pulls color from the matching raw episode, applies an
LRC (or uniqueness) occlusion mask at a live-adjustable threshold, and lets you:

  - drag a 3D gizmo (arrows + rotation rings) per camera, with
    bidirectional sync to per-camera translate/rot_local/rot_world sliders
  - adjust a world-frame bbox via 6 sliders + see its wireframe
  - scrub a frame slider
  - save tweaks + bbox into config.json (mutates `cameras` and/or `bbox`
    sections only; everything else preserved; previous file archived to
    .config_history/)

Moving (wrist) cameras are not in config.cameras and are skipped here: their
per-frame poses come from cam_perframe/ sidecars (see processing/README.md).

Usage:
    python -m processing.viz.tweak_extrinsics --dataset bimanual_cloth_v3 --episode 0
"""
import argparse, sys, time
from pathlib import Path
import numpy as np
import viser

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parents[2]))
import h5video  # noqa: E402
from processing import paths              # noqa: E402
from processing.config import (           # noqa: E402
    DatasetConfig, CamCalibration, legacy_tweaks_dict,
    update_bbox, update_all_camera_calibrations,
    update_episode_override, episode_override_for,
    cam_calibration_to_legacy_tweak,
)

NUM_CAMS = 3


def invert_rigid(E):
    out = np.eye(4, dtype=np.float32)
    R = E[:3, :3].astype(np.float32); t = E[:3, 3].astype(np.float32)
    out[:3, :3] = R.T; out[:3, 3] = -R.T @ t
    return out


def _euler_to_R(roll_deg, pitch_deg, yaw_deg):
    import transforms3d
    return transforms3d.euler.euler2mat(
        np.deg2rad(roll_deg), np.deg2rad(pitch_deg), np.deg2rad(yaw_deg),
        axes="sxyz").astype(np.float32)


def _R_to_euler_sxyz(R):
    import transforms3d
    R64 = np.ascontiguousarray(R, dtype=np.float64)
    return transforms3d.euler.mat2euler(R64, axes="sxyz")


def _R_to_wxyz(R):
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        S = 2.0 * np.sqrt(tr + 1.0)
        return np.array([0.25*S, (R[2,1]-R[1,2])/S, (R[0,2]-R[2,0])/S, (R[1,0]-R[0,1])/S], np.float32)
    if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        return np.array([(R[2,1]-R[1,2])/S, 0.25*S, (R[0,1]+R[1,0])/S, (R[0,2]+R[2,0])/S], np.float32)
    if R[1, 1] > R[2, 2]:
        S = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        return np.array([(R[0,2]-R[2,0])/S, (R[0,1]+R[1,0])/S, 0.25*S, (R[1,2]+R[2,1])/S], np.float32)
    S = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
    return np.array([(R[1,0]-R[0,1])/S, (R[0,2]+R[2,0])/S, (R[1,2]+R[2,1])/S, 0.25*S], np.float32)


def _R_from_wxyz(wxyz):
    w, x, y, z = (float(v) for v in wxyz)
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)],
    ], dtype=np.float32)


def uniqueness_mask(disp, atol=1.0):
    H, W = disp.shape
    u_R = np.round(np.arange(W, dtype=np.float32)[None, :] - disp).astype(np.int32)
    in_R = (u_R >= 0) & (u_R < W)
    u_R_c = np.clip(u_R, 0, W - 1)
    max_d = np.full((H, W), -np.inf, dtype=np.float32)
    vs = np.arange(H, dtype=np.int32)[:, None].repeat(W, axis=1)
    np.maximum.at(max_d, (vs, u_R_c), disp)
    return (disp >= max_d[vs, u_R_c] - atol) & in_R


def lrc_mask(disp_l, disp_r, thresh=1.0):
    """Left-right consistency mask on the LEFT-IR disparity. Survives where
    |disp_L - disp_R[matched right column]| <= thresh. `disp_r` is the
    right-IR-referenced map from run_fs --pair both."""
    H, W = disp_l.shape
    xs = np.arange(W, dtype=np.float32)[None, :]
    x_R = np.round(xs - disp_l).astype(np.int32)
    in_R = (x_R >= 0) & (x_R < W)
    x_R_c = np.clip(x_R, 0, W - 1)
    rows = np.arange(H, dtype=np.int32)[:, None]
    return in_R & (np.abs(disp_l - disp_r[rows, x_R_c]) <= thresh)


def _bilinear_sample_rgb(img, u, v):
    H, W = img.shape[:2]
    u0 = np.floor(u).astype(np.int32); v0 = np.floor(v).astype(np.int32)
    a = (u - u0).astype(np.float32); b = (v - v0).astype(np.float32)
    u0c = np.clip(u0, 0, W - 2); v0c = np.clip(v0, 0, H - 2)
    out = np.zeros((u.shape[0], 3), dtype=np.float32)
    for ch in range(3):
        top = (1 - a) * img[v0c, u0c, ch] + a * img[v0c, u0c + 1, ch]
        bot = (1 - a) * img[v0c + 1, u0c, ch] + a * img[v0c + 1, u0c + 1, ch]
        out[:, ch] = (1 - b) * top + b * bot
    return out.astype(np.uint8)


def back_project_to_cam_frame(depth_m, color, K_ir, K_rgb, T_r2i,
                              *, stride, min_z, max_z):
    if stride > 1:
        depth_m = depth_m[::stride, ::stride]
    H, W = depth_m.shape
    valid = (depth_m > min_z) & (depth_m < max_z) & np.isfinite(depth_m)
    if not valid.any():
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8)
    vs, us = np.nonzero(valid)
    z = depth_m[vs, us].astype(np.float32)
    fx_ir, fy_ir = float(K_ir[0, 0]), float(K_ir[1, 1])
    cx_ir, cy_ir = float(K_ir[0, 2]), float(K_ir[1, 2])
    u_orig = us.astype(np.float32) * stride
    v_orig = vs.astype(np.float32) * stride
    x_ir = (u_orig - cx_ir) * z / fx_ir
    y_ir = (v_orig - cy_ir) * z / fy_ir
    pts_ir = np.stack([x_ir, y_ir, z], axis=-1)
    R_r2i = T_r2i[:3, :3].astype(np.float32); t_r2i = T_r2i[:3, 3].astype(np.float32)
    R_i2r = R_r2i.T; t_i2r = -R_i2r @ t_r2i
    pts_rgb = pts_ir @ R_i2r.T + t_i2r
    Z_rgb = pts_rgb[:, 2]
    in_front = Z_rgb > 1e-3
    fx_rgb, fy_rgb = float(K_rgb[0, 0]), float(K_rgb[1, 1])
    cx_rgb, cy_rgb = float(K_rgb[0, 2]), float(K_rgb[1, 2])
    u_rgb = fx_rgb * pts_rgb[:, 0] / np.where(in_front, Z_rgb, 1.0) + cx_rgb
    v_rgb = fy_rgb * pts_rgb[:, 1] / np.where(in_front, Z_rgb, 1.0) + cy_rgb
    H_rgb, W_rgb = color.shape[:2]
    in_frame = in_front & (u_rgb >= 0) & (u_rgb < W_rgb - 1) \
               & (v_rgb >= 0) & (v_rgb < H_rgb - 1)
    pts_ir_k = pts_ir[in_frame]
    rgb = _bilinear_sample_rgb(color, u_rgb[in_frame], v_rgb[in_frame])
    return pts_ir_k.astype(np.float32), rgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    help="dataset name; resolves all paths via processing.paths")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--min-depth", type=float, default=0.10)
    ap.add_argument("--max-depth", type=float, default=2.0)
    ap.add_argument("--save-override", default=None, metavar="LO-HI",
                    help="Per-episode-range mode: Save writes the tweaks into "
                         "config episode_overrides for episodes LO..HI "
                         "(inclusive) instead of the base cameras section. "
                         "Sliders init from that override (falling back to "
                         "base). Use for a camera bumped mid-collection.")
    ap.add_argument("--override-cams", default=None, metavar="0,1,...",
                    help="With --save-override: only these cams are written "
                         "into the override entry (default: all).")
    args = ap.parse_args()

    ov_range = None
    ov_cams_save = None
    if args.save_override:
        lo, _, hi = args.save_override.partition("-")
        ov_range = (int(lo), int(hi))
        if args.override_cams:
            ov_cams_save = {int(c) for c in args.override_cams.split(",")}

    cfg_path  = paths.config_path(args.dataset)
    if not cfg_path.exists():
        raise SystemExit(f"no config.json at {cfg_path} — `./pw setup <ds>` seeds one from $TEMPLATE")
    cfg = DatasetConfig.load(cfg_path)

    # All episodes that have BOTH a disp file and a raw source (for the
    # in-GUI episode dropdown). T_max sizes the shared frame slider.
    import re as _re
    episodes, T_by_ep = [], {}
    for p in sorted(paths.disp_dir(args.dataset).glob("disp_episode_*.hdf5"),
                    key=lambda p: int(_re.search(r"(\d+)", p.name).group(1))):
        e = int(_re.search(r"(\d+)", p.name).group(1))
        if (paths.raw_dir(args.dataset) / f"episode_{e}.hdf5").exists():
            with h5video.File(p, "r") as f:
                T_by_ep[e] = int(f.attrs["num_frames"])
            episodes.append(e)
    if not episodes:
        raise SystemExit(f"no disp+raw episode pairs under {paths.disp_dir(args.dataset)}")
    if args.episode not in episodes:
        raise SystemExit(f"episode {args.episode} not available (have {episodes[0]}..{episodes[-1]})")
    T_max = max(T_by_ep.values())

    state = {"df": None, "sf": None, "T": 0, "has_right": False, "ep": None}

    def open_episode(ep):
        for h in ("df", "sf"):
            if state[h] is not None:
                try: state[h].close()
                except Exception: pass
        disp_path = paths.disp_dir(args.dataset) / f"disp_episode_{ep}.hdf5"
        src_path  = paths.raw_dir(args.dataset)  / f"episode_{ep}.hdf5"
        state["df"] = h5video.File(disp_path, "r")
        state["sf"] = h5video.File(src_path, "r")
        state["T"]  = int(state["df"].attrs["num_frames"])
        state["has_right"] = "disp_right" in state["df"]
        state["ep"] = ep
        print(f"episode {ep}  T={state['T']}  disp_right={state['has_right']}")

    open_episode(args.episode)
    df = state["df"]   # only for the cam auto-detect below
    has_right = state["has_right"]
    print(f"  disp_right present: {has_right}"
          + ("" if has_right else "  (LRC toggle disabled — rerun run_fs --pair both)"))

    # Auto-detect cameras present in this disp file.
    global NUM_CAMS
    detected = sorted(int(k.split("_")[1]) for k in df["disp"].keys()
                       if k.startswith("camera_"))
    # FIXED cams only: fs_disp also carries the wrist cams (camera_3/4),
    # whose meta E is per-frame FK — no static calibration to tweak here.
    if cfg.cameras:
        _cfg_cams = {int(k.split("_")[1]) for k in cfg.cameras.keys()}
        _skipped = [c for c in detected if c not in _cfg_cams]
        detected = [c for c in detected if c in _cfg_cams]
        if _skipped:
            print(f"  skipping non-fixed cam(s) {_skipped} (not in config.cameras; "
                  f"moving cams use cam_perframe/ sidecars)")
    NUM_CAMS = (max(detected) + 1) if detected else 3
    print(f"dataset: {args.dataset}  episodes: {len(episodes)} ({episodes[0]}..{episodes[-1]})"
          f"  detected cams: {detected}")
    print(f"  config: {cfg_path}")

    # Adapt config → per-cam tweak dict matching the old slider initialization.
    tweak_legacy = legacy_tweaks_dict(cfg)
    tweak = {c: {"dx":0.0,"dy":0.0,"dz":0.0,
                 "droll":0.0,"dpitch":0.0,"dyaw":0.0,
                 "wdroll":0.0,"wdpitch":0.0,"wdyaw":0.0} for c in range(NUM_CAMS)}
    for c in range(NUM_CAMS):
        if c in tweak_legacy:
            tweak[c].update({k: tweak_legacy[c][k] for k in tweak[c].keys()})
    if ov_range is not None:
        # Init from the existing override for this range (else keep base).
        ov_cams0, _ = episode_override_for(cfg, ov_range[0])
        for name, cal in ov_cams0.items():
            c = int(name.split("_")[1])
            if c < NUM_CAMS:
                lg = cam_calibration_to_legacy_tweak(cal)
                tweak[c].update({k: lg[k] for k in tweak[c].keys()})
        print(f"  OVERRIDE MODE: Save writes episodes {ov_range[0]}..{ov_range[1]} "
              f"(cams {sorted(ov_cams_save) if ov_cams_save else 'all'}); "
              f"init from override: {sorted(ov_cams0) or 'none (base)'}")
    bbox = list(cfg.bbox) if cfg.bbox else [-0.40, 0.40, -0.40, 0.40, -0.05, 0.50]
    print(f"  cameras: {sorted(tweak.keys())} (from config)")
    print(f"  bbox: {bbox}")

    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    print(f"Server up at http://localhost:{args.port}")
    server.scene.add_frame("/world", axes_length=0.05, axes_radius=0.002)

    _BOX_EDGES = np.array([
        (0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
        (0,4),(1,5),(2,6),(3,7)], dtype=np.int32)
    def _bbox_segments(b):
        x0,x1,y0,y1,z0,z1 = b
        corners = np.array([
            [x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],
            [x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]], np.float32)
        return np.stack([corners[_BOX_EDGES[:,0]], corners[_BOX_EDGES[:,1]]], axis=1)
    bbox_handle = server.scene.add_line_segments(
        "/bbox", points=_bbox_segments(bbox),
        colors=np.array((50, 200, 230), np.uint8), line_width=2.0)

    pc_handles = {c: server.scene.add_point_cloud(
        f"/cam{c}/pc", points=np.zeros((1, 3), np.float32),
        colors=np.zeros((1, 3), np.uint8), point_size=0.003) for c in range(NUM_CAMS)}
    ctrls = {c: server.scene.add_transform_controls(
        f"/cam{c}/ctrl", scale=0.10, line_width=2.5, depth_test=False)
        for c in range(NUM_CAMS)}

    # ---- GUI ----
    with server.gui.add_folder("Playback"):
        ep_dropdown = server.gui.add_dropdown(
            "episode", options=[str(e) for e in episodes],
            initial_value=str(args.episode))
        # Shared slider sized to the LONGEST episode (viser can't resize a
        # slider); render() clamps to the current episode's T.
        frame_slider = server.gui.add_slider("frame", min=0, max=max(0, T_max - 1),
                                              step=1, initial_value=min(100, state["T"] - 1))
        play_btn  = server.gui.add_button("▶ Play")
        pause_btn = server.gui.add_button("⏸ Pause")
        fps_slider = server.gui.add_slider("playback fps", min=1, max=60, step=1,
                                            initial_value=10)
        loop_cb = server.gui.add_checkbox("loop", initial_value=True)

    with server.gui.add_folder("Display"):
        cam_toggles = {c: server.gui.add_checkbox(f"cam{c}", initial_value=True) for c in range(NUM_CAMS)}
        pt_size = server.gui.add_slider("point size (m)", min=0.0005, max=0.02,
                                         step=0.0005, initial_value=0.003)
        stride_w = server.gui.add_slider("stride", min=1, max=8, step=1,
                                          initial_value=args.stride)

    with server.gui.add_folder("Occlusion mask"):
        # Mask mode: none / uniqueness / LRC. LRC needs disp_right in the file.
        mask_mode = server.gui.add_dropdown(
            "mode",
            options=(["none", "uniqueness", "lrc"] if has_right
                     else ["none", "uniqueness"]),
            initial_value=("lrc" if has_right else "uniqueness"))
        atol_w  = server.gui.add_slider("uniq atol (px)", min=0.0, max=3.0,
                                         step=0.1, initial_value=1.0)
        lrc_w   = server.gui.add_slider("LRC thresh (px)", min=0.0, max=10.0,
                                         step=0.1, initial_value=1.0)
        if not has_right:
            lrc_w.disabled = True

    with server.gui.add_folder("Bbox crop"):
        bbox_on   = server.gui.add_checkbox("bbox crop on", initial_value=True)
        bbox_show = server.gui.add_checkbox("show wireframe", initial_value=True)
        b_sliders = {
            "xmin": server.gui.add_slider("xmin (m)", min=-1.5, max=1.5, step=0.005, initial_value=bbox[0]),
            "xmax": server.gui.add_slider("xmax (m)", min=-1.5, max=1.5, step=0.005, initial_value=bbox[1]),
            "ymin": server.gui.add_slider("ymin (m)", min=-1.5, max=1.5, step=0.005, initial_value=bbox[2]),
            "ymax": server.gui.add_slider("ymax (m)", min=-1.5, max=1.5, step=0.005, initial_value=bbox[3]),
            "zmin": server.gui.add_slider("zmin (m)", min=-0.5, max=1.5, step=0.005, initial_value=bbox[4]),
            "zmax": server.gui.add_slider("zmax (m)", min=-0.5, max=1.5, step=0.005, initial_value=bbox[5]),
        }
        bbox_save_btn = server.gui.add_button(f"Save bbox → {cfg_path.name}")
        bbox_reset_btn = server.gui.add_button("Reset bbox to default")

    sliders = {}
    for c in range(NUM_CAMS):
        with server.gui.add_folder(f"cam{c} extrinsic tweak"):
            sliders[c] = {
                "dx":     server.gui.add_slider(f"cam{c} dx (mm)",     min=-1500.0, max=1500.0, step=1.0, initial_value=tweak[c]["dx"]*1000),
                "dy":     server.gui.add_slider(f"cam{c} dy (mm)",     min=-1500.0, max=1500.0, step=1.0, initial_value=tweak[c]["dy"]*1000),
                "dz":     server.gui.add_slider(f"cam{c} dz (mm)",     min=-1500.0, max=1500.0, step=1.0, initial_value=tweak[c]["dz"]*1000),
                "droll":  server.gui.add_slider(f"cam{c} droll  (cam-local, deg)", min=-180.0, max=180.0, step=0.5, initial_value=tweak[c]["droll"]),
                "dpitch": server.gui.add_slider(f"cam{c} dpitch (cam-local, deg)", min=-180.0, max=180.0, step=0.5, initial_value=tweak[c]["dpitch"]),
                "dyaw":   server.gui.add_slider(f"cam{c} dyaw   (cam-local, deg)", min=-180.0, max=180.0, step=0.5, initial_value=tweak[c]["dyaw"]),
                "wdroll":  server.gui.add_slider(f"cam{c} wdroll  (world-X, deg)", min=-180.0, max=180.0, step=0.5, initial_value=tweak[c]["wdroll"]),
                "wdpitch": server.gui.add_slider(f"cam{c} wdpitch (world-Y, deg)", min=-180.0, max=180.0, step=0.5, initial_value=tweak[c]["wdpitch"]),
                "wdyaw":   server.gui.add_slider(f"cam{c} wdyaw   (world-Z, deg)", min=-180.0, max=180.0, step=0.5, initial_value=tweak[c]["wdyaw"]),
                "reset":  server.gui.add_button(f"cam{c} reset"),
            }

    if ov_range is not None:
        save_btn = server.gui.add_button(
            f"Save OVERRIDE eps {ov_range[0]}-{ov_range[1]} → {cfg_path.name}")
    else:
        save_btn = server.gui.add_button(f"Save tweaks → {cfg_path.name} (cameras)")
    status_text = server.gui.add_text("status", initial_value="-", disabled=True)

    cache = {"ep": None, "frame_idx": None, "stride": None, "mode": None,
             "atol": None, "lrc": None, "data": None}

    def load_frame(t, stride, mode, atol, lrc_thresh):
        df, sf = state["df"], state["sf"]
        out = {}
        for c in range(NUM_CAMS):
            disp   = df[f"disp/camera_{c}"][t].astype(np.float32)
            K_ir   = df[f"meta/camera_{c}/K_ir"][t]
            K_rgb  = df[f"meta/camera_{c}/K_rgb"][t]
            T_r2i  = df[f"meta/camera_{c}/T_rgb_to_ir"][t]
            E_orig = df[f"meta/camera_{c}/E_cam2world"][t]
            B      = float(df[f"meta/camera_{c}/baseline"][t])
            color  = sf[f"/observations/images/camera_{c}_color"][t]
            depth  = (float(K_ir[0, 0]) * B) / np.maximum(disp, 1e-3)
            if mode == "lrc" and state["has_right"]:
                disp_r = df[f"disp_right/camera_{c}"][t].astype(np.float32)
                m = lrc_mask(disp, disp_r, thresh=lrc_thresh)
                depth = np.where(m, depth, 0.0)
            elif mode == "uniqueness":
                m = uniqueness_mask(disp, atol=atol)
                depth = np.where(m, depth, 0.0)
            pts_cam, rgb = back_project_to_cam_frame(
                depth, color, K_ir, K_rgb, T_r2i,
                stride=stride, min_z=args.min_depth, max_z=args.max_depth)
            out[c] = {"pts_cam": pts_cam, "rgb": rgb, "E_orig": E_orig}
        return out

    def current_tweaked_E(c, E_orig):
        droll  = float(sliders[c]["droll"].value)
        dpitch = float(sliders[c]["dpitch"].value)
        dyaw   = float(sliders[c]["dyaw"].value)
        wdroll  = float(sliders[c]["wdroll"].value)
        wdpitch = float(sliders[c]["wdpitch"].value)
        wdyaw   = float(sliders[c]["wdyaw"].value)
        dx_m   = float(sliders[c]["dx"].value) / 1000.0
        dy_m   = float(sliders[c]["dy"].value) / 1000.0
        dz_m   = float(sliders[c]["dz"].value) / 1000.0
        if all(abs(v) < 1e-9 for v in (droll, dpitch, dyaw,
                                         wdroll, wdpitch, wdyaw,
                                         dx_m, dy_m, dz_m)):
            return E_orig.astype(np.float32)
        R_local = _euler_to_R(droll, dpitch, dyaw)
        R_world = _euler_to_R(wdroll, wdpitch, wdyaw)
        E_new = E_orig.astype(np.float32).copy()
        # Pre-multiply by R_world (world-axis rotation, pivot=camera position).
        # Post-multiply by R_local (camera-axis rotation).
        E_new[:3, :3] = R_world @ E_orig[:3, :3] @ R_local
        E_new[:3, 3]  = E_orig[:3, 3]  + np.array([dx_m, dy_m, dz_m], np.float32)
        return E_new

    def current_bbox():
        return [float(b_sliders[k].value)
                for k in ("xmin","xmax","ymin","ymax","zmin","zmax")]

    def render():
        T = state["T"]
        t = min(int(frame_slider.value), T - 1)
        stride = int(stride_w.value)
        atol = float(atol_w.value)
        mode = str(mask_mode.value)
        lrc_thresh = float(lrc_w.value)
        key = (state["ep"], t, stride, mode, atol, lrc_thresh)
        cached_key = (cache["ep"], cache["frame_idx"], cache["stride"], cache["mode"],
                      cache["atol"], cache["lrc"])
        if key != cached_key:
            cache["data"] = load_frame(t, stride, mode, atol, lrc_thresh)
            (cache["ep"], cache["frame_idx"], cache["stride"], cache["mode"],
             cache["atol"], cache["lrc"]) = key
        b = current_bbox()
        # Track per-cam point counts so the status line can show what bbox
        # filtering keeps. Useful for picking a workspace bbox: drag zmin up
        # to crop the table, watch the total count drop.
        per_cam_n = {c: 0 for c in range(NUM_CAMS)}
        n_before_bbox_total = 0
        with server.atomic():
            for c in range(NUM_CAMS):
                d = cache["data"][c]
                if not cam_toggles[c].value or len(d["pts_cam"]) == 0:
                    pc_handles[c].visible = False
                    ctrls[c].visible = False
                    continue
                E_new = current_tweaked_E(c, d["E_orig"])
                R_n, t_n = E_new[:3, :3], E_new[:3, 3]
                world_pts = (d["pts_cam"] @ R_n.T + t_n).astype(np.float32)
                rgb = d["rgb"]
                n_before_bbox_total += len(world_pts)
                if bbox_on.value:
                    m = ((world_pts[:,0]>=b[0]) & (world_pts[:,0]<=b[1]) &
                         (world_pts[:,1]>=b[2]) & (world_pts[:,1]<=b[3]) &
                         (world_pts[:,2]>=b[4]) & (world_pts[:,2]<=b[5]))
                    world_pts = world_pts[m]; rgb = rgb[m]
                per_cam_n[c] = len(world_pts)
                pc_handles[c].points = world_pts
                pc_handles[c].colors = rgb
                pc_handles[c].point_size = float(pt_size.value)
                pc_handles[c].visible = True
                ctrls[c].visible = True
        n_kept = sum(per_cam_n.values())
        per_cam_str = "  ".join(f"c{c}={per_cam_n[c]:,}" for c in range(NUM_CAMS)
                                 if cam_toggles[c].value)
        if bbox_on.value:
            pct = (100.0 * n_kept / max(1, n_before_bbox_total))
            status_text.value = (
                f"ep {state['ep']}  frame {t}/{T-1}  bbox-on  "
                f"kept {n_kept:,} / {n_before_bbox_total:,}  ({pct:.1f}%)   "
                f"per-cam: {per_cam_str}   "
                f"z=[{b[4]:+.3f},{b[5]:+.3f}]m")
        else:
            status_text.value = (
                f"ep {state['ep']}  frame {t}/{T-1}  bbox-off   "
                f"total {n_kept:,}   per-cam: {per_cam_str}")

    # ---- bidirectional gizmo <-> sliders sync ----
    suppress = {c: False for c in range(NUM_CAMS)}

    def push_sliders_to_ctrl(c):
        if suppress[c]: return
        # Need E_orig for this camera at current frame.
        if cache["data"] is None:
            return
        d = cache["data"][c]
        E_new = current_tweaked_E(c, d["E_orig"])
        suppress[c] = True
        ctrls[c].position = tuple(E_new[:3, 3].tolist())
        ctrls[c].wxyz = tuple(_R_to_wxyz(E_new[:3, :3]).tolist())
        suppress[c] = False
        render()

    def pull_ctrl_to_sliders(c):
        if suppress[c]: return
        if cache["data"] is None:
            return
        d = cache["data"][c]
        E_base = d["E_orig"]
        new_pos = np.array(ctrls[c].position, np.float32)
        new_R = _R_from_wxyz(ctrls[c].wxyz)
        # Decompose with the EXISTING world rotation held fixed. Dragging the
        # gizmo only updates the local R; wd* sliders are user-only.
        wdroll  = float(sliders[c]["wdroll"].value)
        wdpitch = float(sliders[c]["wdpitch"].value)
        wdyaw   = float(sliders[c]["wdyaw"].value)
        R_world = _euler_to_R(wdroll, wdpitch, wdyaw)
        dt = new_pos - E_base[:3, 3]
        # R_world @ E_base[:3,:3] @ R_local = new_R  →  R_local = (R_world @ E_base[:3,:3]).T @ new_R
        R_local = (R_world @ E_base[:3, :3]).T @ new_R
        roll, pitch, yaw = _R_to_euler_sxyz(R_local)
        suppress[c] = True
        sliders[c]["dx"].value = float(np.clip(dt[0] * 1000, -1500, 1500))
        sliders[c]["dy"].value = float(np.clip(dt[1] * 1000, -1500, 1500))
        sliders[c]["dz"].value = float(np.clip(dt[2] * 1000, -1500, 1500))
        sliders[c]["droll"].value  = float(np.clip(np.degrees(roll),  -180, 180))
        sliders[c]["dpitch"].value = float(np.clip(np.degrees(pitch), -180, 180))
        sliders[c]["dyaw"].value   = float(np.clip(np.degrees(yaw),   -180, 180))
        # wd* unchanged on gizmo drag.
        suppress[c] = False
        render()

    # Slider/control wiring.
    for c in range(NUM_CAMS):
        for key, w in sliders[c].items():
            if key == "reset":
                @w.on_click
                def _reset(_e, c_loc=c):
                    for k in ("dx","dy","dz","droll","dpitch","dyaw",
                              "wdroll","wdpitch","wdyaw"):
                        sliders[c_loc][k].value = 0.0
                    push_sliders_to_ctrl(c_loc)
            else:
                w.on_update(lambda _, c_loc=c: push_sliders_to_ctrl(c_loc))
        ctrls[c].on_update(lambda _, c_loc=c: pull_ctrl_to_sliders(c_loc))

    def _apply_ep_override(ep):
        """Outside --save-override mode: re-init sliders from base config +
        this episode's episode_overrides entry (viewing per-episode fitted
        extrinsics). Saving base tweaks while viewing an overridden episode
        would bake that episode's values into the base — the status line
        warns when an override is active."""
        if ov_range is not None:
            return
        ov_cams, _ = episode_override_for(cfg, ep)
        for c in range(NUM_CAMS):
            merged = dict(tweak[c])
            name = f"camera_{c}"
            if name in ov_cams:
                lg = cam_calibration_to_legacy_tweak(ov_cams[name])
                merged.update({k: lg[k] for k in merged.keys()})
            for k, w in sliders[c].items():
                if k == "reset":
                    continue
                v = merged[k] * 1000 if k in ("dx", "dy", "dz") else merged[k]
                if abs(w.value - v) > 1e-9:
                    w.value = v
        if ov_cams:
            status_text.value = (f"ep {ep}: episode_override active for "
                                 f"{sorted(ov_cams)} (don't Save base tweaks)")

    _apply_ep_override(state["ep"])

    @ep_dropdown.on_update
    def _switch_episode(_):
        ep = int(ep_dropdown.value)
        if ep == state["ep"]:
            return
        playing["v"] = False
        open_episode(ep)
        _apply_ep_override(ep)
        # Clamp the shared slider into the new episode's range and re-render.
        if int(frame_slider.value) > state["T"] - 1:
            frame_slider.value = state["T"] - 1   # triggers render via its own cb
        else:
            render()
        for c_loc in range(NUM_CAMS):
            push_sliders_to_ctrl(c_loc)   # E_orig may differ per episode
        status_text.value = f"switched to episode {ep} (T={state['T']})"

    frame_slider.on_update(lambda _: render())
    pt_size.on_update(lambda _: render())
    stride_w.on_update(lambda _: render())
    mask_mode.on_update(lambda _: render())
    atol_w.on_update(lambda _: render())
    lrc_w.on_update(lambda _: render())
    for cb in cam_toggles.values():
        cb.on_update(lambda _: render())

    def _on_bbox_change(_=None):
        bbox_handle.points = _bbox_segments(current_bbox())
        bbox_handle.visible = bool(bbox_show.value)
        render()
    for w in b_sliders.values():
        w.on_update(_on_bbox_change)
    bbox_on.on_update(_on_bbox_change)
    bbox_show.on_update(_on_bbox_change)

    @bbox_save_btn.on_click
    def _save_bbox(_):
        b = current_bbox()
        update_bbox(cfg_path, b)
        status_text.value = f"Saved bbox → {cfg_path.name}"
        print(f"Saved bbox into {cfg_path}: {b}")

    @bbox_reset_btn.on_click
    def _reset_bbox(_):
        defaults = (-0.40, 0.40, -0.40, 0.40, -0.05, 0.50)
        for k, v in zip(("xmin","xmax","ymin","ymax","zmin","zmax"), defaults):
            b_sliders[k].value = v
        _on_bbox_change()

    @save_btn.on_click
    def _save_tweaks(_):
        cams_out = {}
        for c in range(NUM_CAMS):
            if ov_range is not None and ov_cams_save is not None and c not in ov_cams_save:
                continue
            cams_out[f"camera_{c}"] = CamCalibration(
                translate=[float(sliders[c]["dx"].value)/1000.0,
                           float(sliders[c]["dy"].value)/1000.0,
                           float(sliders[c]["dz"].value)/1000.0],
                rot_local_deg=[float(sliders[c]["droll"].value),
                               float(sliders[c]["dpitch"].value),
                               float(sliders[c]["dyaw"].value)],
                rot_world_deg=[float(sliders[c]["wdroll"].value),
                               float(sliders[c]["wdpitch"].value),
                               float(sliders[c]["wdyaw"].value)],
            )
        if ov_range is not None:
            update_episode_override(cfg_path, ov_range[0], ov_range[1],
                                    cameras=cams_out)
            status_text.value = (f"Saved override eps {ov_range[0]}-{ov_range[1]} "
                                 f"({sorted(cams_out)}) → {cfg_path.name}")
            print(f"Saved episode_overrides [{ov_range[0]},{ov_range[1]}] "
                  f"into {cfg_path}:")
        else:
            update_all_camera_calibrations(cfg_path, cams_out)
            status_text.value = f"Saved cameras → {cfg_path.name}"
            print(f"Saved camera calibrations into {cfg_path}:")
        for c, cal in cams_out.items():
            print(f"  {c}: t={cal.translate} loc={cal.rot_local_deg} wld={cal.rot_world_deg}")

    # ---- Playback ----
    playing = {"v": False}
    @play_btn.on_click
    def _(_): playing["v"] = True
    @pause_btn.on_click
    def _(_): playing["v"] = False

    render()
    # Initial gizmo placement (now that cache is populated).
    for c in range(NUM_CAMS):
        push_sliders_to_ctrl(c)

    print("Drag the gizmo rings/arrows or use sliders. 'Save tweaks/bbox' → JSON.")
    while True:
        if playing["v"]:
            cur = int(frame_slider.value)
            nxt = cur + 1
            if nxt >= state["T"]:
                if loop_cb.value:
                    nxt = 0
                else:
                    nxt = state["T"] - 1
                    playing["v"] = False
            frame_slider.value = nxt
            time.sleep(1.0 / max(float(fps_slider.value), 1.0))
        else:
            time.sleep(0.05)


if __name__ == "__main__":
    main()
